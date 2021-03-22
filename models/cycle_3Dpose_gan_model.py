import torch
import itertools
from .base_model import BaseModel
from .cycle_gan_model import CycleGANModel
import os
from collections import OrderedDict
from util.util import mkdirs
from pose_3d_model.mvn.models.triangulation import AlgebraicTriangulationNet
from pose_3d_model.mvn.utils.cfg import load_config
from pose_3d_model.mvn.utils.multiview import Camera
from pose_3d_model.mvn.models.loss import KeypointsMSESmoothLoss
from util.pose_utils import PoseVisualizer, SkeletonFormat
import numpy as np

class Cycle3DPoseGANModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        return CycleGANModel.modify_commandline_options(parser, is_train)

    def __init__(self, opt, rank):
        """
        Initialize the Cycle3DPoseGANModel class.
        Parameters:
            opt (Option class)-- experiment flags
        """

        BaseModel.__init__(self, opt, rank)
        self.rank = rank
        self.shared_loss_weight = opt.shared_loss_weight

        self.image_width = opt.crop_size_width
        self.image_height = opt.crop_size_height
        self.org_image_height = opt.org_image_height
        self.org_image_width = opt.org_image_width
        self.image_load_width = opt.load_size
        self.image_load_height = opt.org_image_height/(opt.org_image_width/opt.load_size)

        self.view_net0 = CycleGANModel(opt, rank)
        self.view_net0.save_dir = os.path.join(self.view_net0.save_dir, 'view0')
        if rank == 0:
            mkdirs(self.view_net0.save_dir)

        self.view_net1 = CycleGANModel(opt,rank)
        self.view_net1.save_dir = os.path.join(self.view_net1.save_dir, 'view1')
        if rank == 0:
            mkdirs(self.view_net1.save_dir)

        with torch.no_grad():
            self.pose_3d_cfg = load_config(opt.pose_3d_config)
            self.pose_3d_net = AlgebraicTriangulationNet(self.pose_3d_cfg, self.device, rank)
            self.pose_3d_net = self.pose_3d_net.to(rank)
            self.pose_3d_net = torch.nn.parallel.DistributedDataParallel(module=self.pose_3d_net, device_ids=[rank])  # multi-GPUs
            self.pose_3d_net.eval()

        self.scale_keypoints_3d = self.pose_3d_cfg.opt.scale_keypoints_3d

        # load camera proj matrices
        self.cam_view0_A = self.load_cam(os.path.join(opt.dataroot, opt.calibration_file_A), opt.view_name0)
        self.cam_view1_A = self.load_cam(os.path.join(opt.dataroot, opt.calibration_file_A), opt.view_name1)
        self.cam_view0_B = self.load_cam(os.path.join(opt.dataroot, opt.calibration_file_B), opt.view_name0)
        self.cam_view1_B = self.load_cam(os.path.join(opt.dataroot, opt.calibration_file_B), opt.view_name1)

        if self.isTrain:
            # initialize optimizers
            # schedulers will be automatically created by function <BaseModel.setup>
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.view_net0.netG_A.parameters(), self.view_net0.netG_B.parameters(), self.view_net1.netG_A.parameters(), self.view_net1.netG_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.view_net0.netD_A.parameters(), self.view_net0.netD_B.parameters(),self.view_net1.netD_A.parameters(), self.view_net1.netD_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            # Define loss function
            self.criterion3DPose = KeypointsMSESmoothLoss(opt.MSE_3DPose_threshold)

    def set_input(self, input):
        self.view_net0.set_input(input['view0'])
        self.view_net1.set_input(input['view1'])

        self.prepare_3d_pose_input(input)


    def forward(self):
        self.view_net0.forward()
        self.view_net1.forward()

        with torch.no_grad():
            # set images_fake_B
            images_fake_B = self.combine_to_batch(self.view_net0.fake_B, self.view_net1.fake_B)
            keypoints_3d_pred_B, keypoints_2d_pred_B, heatmaps_pred_B, confidences_3d_pred_B = self.pose_3d_net.forward(images_fake_B, self.personA_proj_matrices_for_pred_batch)

            self.keypoints_2d_pred_B = keypoints_2d_pred_B
            self.keypoints_3d_pred_B = keypoints_3d_pred_B
            self.confidences_3d_pred_B = confidences_3d_pred_B

            # set images_fake_A
            images_fake_A = self.combine_to_batch(self.view_net0.fake_A, self.view_net1.fake_A)
            keypoints_3d_pred_A, keypoints_2d_pred_A, heatmaps_pred_A, confidences_3d_pred_A = self.pose_3d_net.forward(images_fake_A, self.personB_proj_matrices_for_pred_batch)
            self.keypoints_2d_pred_A = keypoints_2d_pred_A
            self.keypoints_3d_pred_A = keypoints_3d_pred_A
            self.confidences_3d_pred_A = confidences_3d_pred_A

    def optimize_parameters(self):

        self.forward()      # compute fake images and reconstruction images.
        # G_A and G_B
        self.set_requires_grad([self.view_net0.netD_A, self.view_net0.netD_B, self.view_net1.netD_A, self.view_net1.netD_B], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.view_net0.set_loss_G()
        self.view_net1.set_loss_G()
        # Joint 3D loss
        self.shared_loss_A = self.criterion3DPose(self.keypoints_3d_pred_A * self.scale_keypoints_3d, self.personB_3d_pose_gt_batch * self.scale_keypoints_3d, self.personB_3d_pose_gt_binary_validity_batch)
        self.shared_loss_B = self.criterion3DPose(self.keypoints_3d_pred_B * self.scale_keypoints_3d, self.personA_3d_pose_gt_batch * self.scale_keypoints_3d, self.personA_3d_pose_gt_binary_validity_batch)
        self.loss_G = self.view_net0.loss_G + self.view_net1.loss_G + self.shared_loss_weight*(self.shared_loss_A + self.shared_loss_B)
        self.loss_G.backward()
        self.optimizer_G.step()       # update G_A and G_B's weights

        # D_A and D_B
        self.set_requires_grad([self.view_net0.netD_A, self.view_net0.netD_B, self.view_net1.netD_A, self.view_net1.netD_B], True)
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        self.view_net0.backward_D_A()      # calculate gradients for view0 D_A
        self.view_net0.backward_D_B()      # calculate graidents for view0 D_B
        self.view_net1.backward_D_A()      # calculate gradients for view1 D_A
        self.view_net1.backward_D_B()      # calculate graidents for view1 D_B
        self.optimizer_D.step()         # update D_A and D_B's weights


    def save_networks(self, epoch):
        self.view_net0.save_networks(epoch)
        self.view_net1.save_networks(epoch)

    def get_current_losses(self):

        errors_ret = OrderedDict()
        errors_ret["loss_G"] = self.loss_G
        errors_ret["loss_shared_A"] = self.shared_loss_A
        errors_ret["loss_shared_B"] = self.shared_loss_B

        losses = self.view_net0.get_current_losses()
        for label, image in losses.items():
            errors_ret['view0_' + label] = losses[label]

        losses = self.view_net1.get_current_losses()
        for label, image in losses.items():
            errors_ret['view1_' + label] = losses[label]

        return errors_ret

    def compute_visuals(self):
        """Calculate additional output images for visdom and HTML visualization"""
        from pose_3d_model.mvn.utils.img import resize_image

        self.view0_fake_B_2d = self.view_net0.fake_B[0, :, :, :].cpu().detach().numpy()
        self.view0_fake_B_2d = (np.transpose(self.view0_fake_B_2d, (1, 2, 0)) + 1) / 2.0 * 255.0
        self.view0_fake_B_2d = resize_image(self.view0_fake_B_2d, (self.pose_3d_cfg.image_shape[0], self.pose_3d_cfg.image_shape[1]))

        self.view0_fake_B_2d = PoseVisualizer.draw_2d_skeleton(self.view0_fake_B_2d,
                                                          self.keypoints_2d_pred_B[0, 0, :, :].cpu().detach().numpy(),
                                                               (self.confidences_3d_pred_B[0, 0,:].cpu().detach().numpy() > 0.2).astype(np.int))

        self.view0_fake_A_2d = self.view_net0.fake_A[0, :, :, :].cpu().detach().numpy()
        self.view0_fake_A_2d = (np.transpose(self.view0_fake_A_2d, (1, 2, 0)) + 1) / 2.0 * 255.0
        self.view0_fake_A_2d = resize_image(self.view0_fake_A_2d, (self.pose_3d_cfg.image_shape[0], self.pose_3d_cfg.image_shape[1]))

        self.view0_fake_A_2d = PoseVisualizer.draw_2d_skeleton(self.view0_fake_A_2d,
                                                          self.keypoints_2d_pred_A[0, 0, :, :].cpu().detach().numpy(),
                                                               (self.confidences_3d_pred_A[0, 0,:].cpu().detach().numpy() > 0.2).astype(np.int))

        self.view1_fake_B_2d = self.view_net1.fake_B[0, :, :, :].cpu().detach().numpy()
        self.view1_fake_B_2d = (np.transpose(self.view1_fake_B_2d, (1, 2, 0)) + 1) / 2.0 * 255.0
        self.view1_fake_B_2d = resize_image(self.view1_fake_B_2d, (self.pose_3d_cfg.image_shape[0], self.pose_3d_cfg.image_shape[1]))

        self.view1_fake_B_2d = PoseVisualizer.draw_2d_skeleton(self.view1_fake_B_2d,
                                                                self.keypoints_2d_pred_B[0, 1, :,:].cpu().detach().numpy(),
                                                               (self.confidences_3d_pred_B[0, 1,:].cpu().detach().numpy() > 0.2).astype(np.int))

        self.view1_fake_A_2d = self.view_net1.fake_A[0, :, :, :].cpu().detach().numpy()
        self.view1_fake_A_2d = (np.transpose(self.view1_fake_A_2d, (1, 2, 0)) + 1) / 2.0 * 255.0
        self.view1_fake_A_2d = resize_image(self.view1_fake_A_2d, (self.pose_3d_cfg.image_shape[0], self.pose_3d_cfg.image_shape[1]))

        self.view1_fake_A_2d = PoseVisualizer.draw_2d_skeleton(self.view1_fake_A_2d,
                                                                self.keypoints_2d_pred_A[0, 1, :,:].cpu().detach().numpy(),
                                                               (self.confidences_3d_pred_A[0, 1,:].cpu().detach().numpy() > 0.2).astype(np.int))

        self.real_3D_A = PoseVisualizer.calc_3d_pose_img(self.org_personA_3d_pose_gt_batch[0,:,:].cpu().detach().numpy(),
                                                         self.personA_3d_pose_gt_binary_validity_batch[0,:,:].cpu().detach().numpy().astype(np.int),
                                                         SkeletonFormat.COCO)

        self.fake_3D_A = PoseVisualizer.calc_3d_pose_img(self.keypoints_3d_pred_A[0, :, :].cpu().detach().numpy(),
                                                         (self.confidences_3d_pred_A[0,0,:].cpu().detach().numpy()>0.2).astype(np.int),
                                                         SkeletonFormat.COCO, "dist=" + str(self.shared_loss_A.cpu().detach().numpy()))

        self.real_3D_B = PoseVisualizer.calc_3d_pose_img(self.org_personB_3d_pose_gt_batch[0,:,:].cpu().detach().numpy(),
                                                         self.personB_3d_pose_gt_binary_validity_batch[0,:,:].cpu().detach().numpy().astype(np.int),
                                                         SkeletonFormat.COCO)

        self.fake_3D_B = PoseVisualizer.calc_3d_pose_img(self.keypoints_3d_pred_B[0, :, :].cpu().detach().numpy(),
                                                         (self.confidences_3d_pred_B[0,0,:].cpu().detach().numpy()>0.2).astype(np.int),
                                                         SkeletonFormat.COCO,
                                                         "dist=" + str(self.shared_loss_B.cpu().detach().numpy()))

        self.view1_real_A_2d = self.view_net1.real_A[0, :, :, :].cpu().detach().numpy()
        self.view1_real_A_2d = (np.transpose(self.view1_real_A_2d, (1, 2, 0)) + 1) / 2.0 * 255.0
        self.view1_real_A_2d = resize_image(self.view1_real_A_2d, (self.view_net1.real_A.shape[2], self.view_net1.real_A.shape[3]))

        self.view1_real_A_2d = PoseVisualizer.draw_2d_skeleton(self.view1_real_A_2d,
                                                               self.personA_2d_kpt_view1[0, :,:].cpu().detach().numpy(),
                                                               (self.personA_3d_pose_gt_binary_validity_batch[0,:,:].cpu().detach().numpy()).astype(np.int))


        self.view0_real_A_2d = self.view_net0.real_A[0, :, :, :].cpu().detach().numpy()
        self.view0_real_A_2d = (np.transpose(self.view0_real_A_2d, (1, 2, 0)) + 1) / 2.0 * 255.0
        self.view0_real_A_2d = resize_image(self.view0_real_A_2d, (self.view_net0.real_A.shape[2], self.view_net0.real_A.shape[3]))

        self.view0_real_A_2d = PoseVisualizer.draw_2d_skeleton(self.view0_real_A_2d,
                                                               self.personA_2d_kpt_view0[0, :,:].cpu().detach().numpy(),
                                                               (self.personA_3d_pose_gt_binary_validity_batch[0,:,:].cpu().detach().numpy()).astype(np.int))

        self.view1_real_B_2d = self.view_net1.real_B[0, :, :, :].cpu().detach().numpy()
        self.view1_real_B_2d = (np.transpose(self.view1_real_B_2d, (1, 2, 0)) + 1) / 2.0 * 255.0
        self.view1_real_B_2d = resize_image(self.view1_real_B_2d, (self.view_net1.real_B.shape[2], self.view_net1.real_B.shape[3]))

        self.view1_real_B_2d = PoseVisualizer.draw_2d_skeleton(self.view1_real_B_2d,
                                                               self.personB_2d_kpt_view1[0, :, :].cpu().detach().numpy(),
                                                               (self.personB_3d_pose_gt_binary_validity_batch[0, :, :].cpu().detach().numpy()).astype(np.int))

        self.view0_real_B_2d = self.view_net0.real_B[0, :, :, :].cpu().detach().numpy()
        self.view0_real_B_2d = (np.transpose(self.view0_real_B_2d, (1, 2, 0)) + 1) / 2.0 * 255.0
        self.view0_real_B_2d = resize_image(self.view0_real_B_2d, (self.view_net0.real_B.shape[2], self.view_net0.real_B.shape[3]))

        self.view0_real_B_2d = PoseVisualizer.draw_2d_skeleton(self.view0_real_B_2d,
                                                               self.personB_2d_kpt_view0[0, :, :].cpu().detach().numpy(),
                                                               (self.personB_3d_pose_gt_binary_validity_batch[0, :, :].cpu().detach().numpy()).astype(np.int))

    def get_current_visuals(self):
        """Return visualization images. train.py will display these images with visdom, and save the images to a HTML"""
        visual_ret = OrderedDict()

        view0_visuals = self.view_net0.get_current_visuals()
        for label,image in view0_visuals.items():
            visual_ret['view0_'+label] = view0_visuals[label]

        view1_visuals = self.view_net1.get_current_visuals()
        for label,image in view1_visuals.items():
            visual_ret['view1_'+label] = view1_visuals[label]

        # pose visuals
        visual_ret['real_3D_A'] = self.real_3D_A
        visual_ret['fake_3D_B'] = self.fake_3D_B
        visual_ret['real_3D_B'] = self.real_3D_B
        visual_ret['fake_3D_A'] = self.fake_3D_A
        visual_ret['view0_fake_B_2d'] = self.view0_fake_B_2d
        visual_ret['view0_fake_A_2d'] = self.view0_fake_A_2d
        visual_ret['view1_fake_B_2d'] = self.view1_fake_B_2d
        visual_ret['view1_fake_A_2d'] = self.view1_fake_A_2d

        visual_ret['view0_real_B_2d'] = self.view0_real_B_2d
        visual_ret['view0_real_A_2d'] = self.view0_real_A_2d
        visual_ret['view1_real_B_2d'] = self.view1_real_B_2d
        visual_ret['view1_real_A_2d'] = self.view1_real_A_2d

        return visual_ret

    def setup(self, opt):
        """
        Load and print networks including the 3D Pose model; create schedulers

        Parameters:
            opt (Option class) -- experiment flags
        """
        super().setup(opt)
        state_dict = torch.load(self.pose_3d_cfg.model.checkpoint)
        print(self.pose_3d_cfg.model.checkpoint)
        self.pose_3d_net.module.load_state_dict(state_dict, strict=True)
        print("Successfully loaded pretrained weights for pose_3d_net model")

    def load_networks(self, epoch):
        self.view_net0.load_networks(epoch)
        self.view_net1.load_networks(epoch)


    def load_cam(self, calibration_filename, view_name):
        import numpy as np
        import json
        calibration_file = open(calibration_filename)
        calibration = json.load(calibration_file)

        for cam in calibration['cameras']:
           if cam['name'] == view_name:
               camera = Camera(np.matrix(cam['R']), np.array(cam['t']).reshape((3, 1)), np.matrix(cam['K']), np.array(cam['distCoef']), view_name)
               break

        image_shape_before_resize = (self.org_image_height, self.org_image_width)
        image_shape = (self.image_load_height, self.image_load_width)

        camera.update_after_resize(image_shape_before_resize, image_shape)

        return camera


    def prepare_3d_pose_input(self, input):

        self.personA_3d_pose_gt_batch = input['personA_3d_pose_gt'][:, :, :3]
        self.org_personA_3d_pose_gt_batch = input['org_personA_3d_pose_gt'][:, :, :3]
        self.personA_3d_pose_gt_validity_batch = input['personA_3d_pose_gt'][:, :, 3:]
        self.personA_3d_pose_gt_binary_validity_batch = (self.personA_3d_pose_gt_validity_batch > 0.0).type(torch.float32)

        # We take only the nose keypoint and ignore the other 4 face keypoints
        # self.personA_3d_pose_gt_binary_validity_batch[:,1:5,:] = 0

        self.personA_3d_pose_gt_batch = self.personA_3d_pose_gt_batch.cuda(self.rank)
        self.org_personA_3d_pose_gt_batch = self.org_personA_3d_pose_gt_batch.cuda(self.rank)
        self.personA_3d_pose_gt_binary_validity_batch = self.personA_3d_pose_gt_binary_validity_batch.cuda(self.rank)

        self.personB_3d_pose_gt_batch = input['personB_3d_pose_gt'][:, :, :3]
        self.org_personB_3d_pose_gt_batch = input['org_personB_3d_pose_gt'][:, :, :3]
        self.personB_3d_pose_gt_validity_batch = input['personB_3d_pose_gt'][:, :, 3:]
        self.personB_3d_pose_gt_binary_validity_batch = (self.personB_3d_pose_gt_validity_batch > 0.0).type(torch.float32)

        # We take only the nose keypoint and ignore the other 4 face keypoints
        # self.personB_3d_pose_gt_binary_validity_batch[:,1:5,:] = 0

        self.personB_3d_pose_gt_batch = self.personB_3d_pose_gt_batch.cuda(self.rank)
        self.org_personB_3d_pose_gt_batch = self.org_personB_3d_pose_gt_batch.cuda(self.rank)
        self.personB_3d_pose_gt_binary_validity_batch = self.personB_3d_pose_gt_binary_validity_batch.cuda(self.rank)

        actual_3d_batch_size = input['personB_3d_pose_gt'][:, :, :1].shape[0]
        self.personA_proj_matrices_batch = self.prepare_proj_matricies_batch(actual_3d_batch_size, self.cam_view0_A, input['view0']['crop_pos_A'], self.cam_view1_A, input['view1']['crop_pos_A'])
        self.personA_proj_matrices_batch = self.personA_proj_matrices_batch.to(self.rank)
        self.crop_pos_A = input['view0']['crop_pos_A']
        self.personB_proj_matrices_batch = self.prepare_proj_matricies_batch(actual_3d_batch_size, self.cam_view0_B, input['view0']['crop_pos_B'], self.cam_view1_B, input['view1']['crop_pos_B'])
        self.personB_proj_matrices_batch = self.personB_proj_matrices_batch.to(self.rank)
        self.crop_pos_B = input['view0']['crop_pos_B']

        # update proj mat for 3d prediction
        self.personA_proj_matrices_for_pred_batch = self.prepare_proj_matricies_batch(actual_3d_batch_size, self.cam_view0_A, input['view0']['crop_pos_A'], self.cam_view1_A, input['view1']['crop_pos_A'], applyResizeUpdate=True)
        self.personA_proj_matrices_for_pred_batch = self.personA_proj_matrices_for_pred_batch.to(self.rank)
        self.crop_pos_A = input['view0']['crop_pos_A']
        self.personB_proj_matrices_for_pred_batch = self.prepare_proj_matricies_batch(actual_3d_batch_size, self.cam_view0_B, input['view0']['crop_pos_B'], self.cam_view1_B, input['view1']['crop_pos_B'], applyResizeUpdate=True)
        self.personB_proj_matrices_for_pred_batch = self.personB_proj_matrices_for_pred_batch.to(self.rank)
        self.crop_pos_B = input['view0']['crop_pos_B']


        # update_valid_keypoints: self.personA_3d_pose_gt_binary_validity_batch
        # calculate 2D projection of 3d pose to each view
        from pose_3d_model.mvn.utils.multiview import project_3d_points_to_image_plane_without_distortion as project

        personA_2d_kpt_view0=[]
        personA_2d_kpt_view1=[]
        for i in range(0, actual_3d_batch_size):
            curr_personA_2d_kpt_view0 = project(self.personA_proj_matrices_batch[i,0], self.org_personA_3d_pose_gt_batch[i,:,:])
            curr_personA_2d_kpt_view1 = project(self.personA_proj_matrices_batch[i,1], self.org_personA_3d_pose_gt_batch[i,:,:])
            personA_2d_kpt_view0.append(curr_personA_2d_kpt_view0)
            personA_2d_kpt_view1.append(curr_personA_2d_kpt_view1)

        self.personA_2d_kpt_view0 = torch.stack(personA_2d_kpt_view0, dim=0)
        self.personA_2d_kpt_view1 = torch.stack(personA_2d_kpt_view1, dim=0)

        # union of valid 2d kpts
        self.personA_3d_pose_gt_binary_validity_batch = torch.unsqueeze(torch.logical_and(
                                                                          torch.logical_or(
                                                                              torch.logical_and(
                                                                                  torch.logical_and(self.personA_2d_kpt_view0[:,:,0]>0, self.personA_2d_kpt_view0[:,:,1]>0),
                                                                                  torch.logical_and(self.personA_2d_kpt_view0[:,:,0]<self.image_width, self.personA_2d_kpt_view0[:,:,1]<self.image_height)),
                                                                              torch.logical_and(
                                                                                           torch.logical_and(self.personA_2d_kpt_view1[:,:,0]>0, self.personA_2d_kpt_view1[:,:,1]>0),
                                                                                           torch.logical_and(self.personA_2d_kpt_view1[:,:,0]<self.image_width, self.personA_2d_kpt_view1[:,:,1]<self.image_height))),
                                                                          self.personA_3d_pose_gt_binary_validity_batch[:,:,0]>0), dim=2)

        self.personA_3d_pose_gt_binary_validity_batch = self.personA_3d_pose_gt_binary_validity_batch.type(torch.float32)

        personB_2d_kpt_view0=[]
        personB_2d_kpt_view1=[]

        for i in range(0, actual_3d_batch_size):
            curr_personB_2d_kpt_view0 = project(self.personB_proj_matrices_batch[i,0], self.org_personB_3d_pose_gt_batch[i,:,:])
            curr_personB_2d_kpt_view1 = project(self.personB_proj_matrices_batch[i,1], self.org_personB_3d_pose_gt_batch[i,:,:])
            personB_2d_kpt_view0.append(curr_personB_2d_kpt_view0)
            personB_2d_kpt_view1.append(curr_personB_2d_kpt_view1)

        self.personB_2d_kpt_view0 = torch.stack(personB_2d_kpt_view0, dim=0)
        self.personB_2d_kpt_view1 = torch.stack(personB_2d_kpt_view1, dim=0)

        self.personB_3d_pose_gt_binary_validity_batch = torch.unsqueeze(torch.logical_and(
                                                            torch.logical_or(
                                                                torch.logical_and(
                                                                    torch.logical_and(self.personB_2d_kpt_view0[:, :, 0] > 0, self.personB_2d_kpt_view0[:, :, 1] > 0),
                                                                    torch.logical_and(self.personB_2d_kpt_view0[:, :, 0] < self.image_width,
                                                                                      self.personB_2d_kpt_view0[:, :, 1] < self.image_height)),
                                                                torch.logical_and(
                                                                    torch.logical_and(self.personB_2d_kpt_view1[:, :, 0] > 0, self.personB_2d_kpt_view1[:, :, 1] > 0),
                                                                    torch.logical_and(self.personB_2d_kpt_view1[:, :, 0] < self.image_width,
                                                                                      self.personB_2d_kpt_view1[:, :, 1] < self.image_height))),
                                                            self.personB_3d_pose_gt_binary_validity_batch[:, :, 0] > 0), dim=2)

        self.personB_3d_pose_gt_binary_validity_batch = self.personB_3d_pose_gt_binary_validity_batch.type(torch.float32)


    def prepare_proj_matricies_batch(self, batch_size, cam_view0, crop_pos_view0, cam_view1, crop_pos_view1, applyResizeUpdate=False):
        """
         Calculate projection matricies after camera update according to crop position
        """
        import copy

        image_shape_before = (self.image_load_height,  self.image_load_width) # image shape after scaling from original size e.g scale from 1080x1920 to 270x480
        image_shape_after = (self.pose_3d_cfg.image_shape[0], self.pose_3d_cfg.image_shape[1]) # image shape input to 3d pose estimator

        proj_matricies_batch = []
        for i in range(0, batch_size):
            curr_cam_view0 = copy.deepcopy(cam_view0)
            curr_cam_view1 = copy.deepcopy(cam_view1)

            proj_matricies_list = []
            bbox = [crop_pos_view0[0][i], crop_pos_view0[1][i], crop_pos_view0[0][i]+self.image_width-1, crop_pos_view0[1][i]+self.image_height-1]
            curr_cam_view0.update_after_crop(bbox)
            if applyResizeUpdate:
                curr_cam_view0.update_after_resize(image_shape_before, image_shape_after)
            proj_matrix_view0 = curr_cam_view0.projection
            bbox = [crop_pos_view1[0][i], crop_pos_view1[1][i], crop_pos_view1[0][i]+self.image_width-1, crop_pos_view1[1][i]+self.image_height-1]
            curr_cam_view1.update_after_crop(bbox)
            if applyResizeUpdate:
                curr_cam_view1.update_after_resize(image_shape_before, image_shape_after)
            proj_matrix_view1 = curr_cam_view1.projection

            proj_matricies_list.append(torch.from_numpy(proj_matrix_view0))
            proj_matricies_list.append(torch.from_numpy(proj_matrix_view1))
            proj_matricies_both = torch.stack(proj_matricies_list, dim=0)

            proj_matricies_batch.append(proj_matricies_both)

        proj_matricies = torch.stack(proj_matricies_batch, dim=0)
        return proj_matricies


    def combine_to_batch(self, input_view0, input_view1):
        """
         Prepare input for 3D pose estimator
        """

        combined_batch=[]

        assert(len(input_view0) == len(input_view1))
        actual_batch_size = self.opt.batch_size
        views0_list =[]
        views1_list=[]
        for i in range(0,actual_batch_size):
            img_view0 = (input_view0[i].clone())
            img_view0 = ((img_view0 +1)/2)

            img_view1 = input_view1[i].clone()
            img_view1 = ((img_view1 +1)/2)

            views0_list.append(img_view0)
            views1_list.append(img_view1)

        views0_list = torch.stack(views0_list, dim=0)
        views1_list = torch.stack(views1_list, dim=0)

        import torchvision.transforms as transforms
        import torch.nn.functional as F
        views0_list = F.interpolate(views0_list, size=[self.pose_3d_cfg.image_shape[0], self.pose_3d_cfg.image_shape[1]], mode='bilinear', align_corners=False)
        views1_list = F.interpolate(views1_list, size=[self.pose_3d_cfg.image_shape[0], self.pose_3d_cfg.image_shape[1]], mode='bilinear', align_corners=False)

        for i in range(0,actual_batch_size):
            both_views_list = []
            img_view0 = views0_list[i]
            transform = transforms.Compose([transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
            img_view0 = transform(img_view0)

            img_view1 = views1_list[i]
            img_view1 = transform(img_view1)

            both_views_list.append(img_view0)
            both_views_list.append(img_view1)
            combined_batch.append(torch.stack(both_views_list, dim=0))

        return torch.stack(combined_batch, dim=0)

