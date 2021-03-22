import torch
import itertools
from .base_model import BaseModel
from .cycle_gan_model import CycleGANModel
import os
from collections import OrderedDict
from util.util import mkdirs


class CycleDualViewGANModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        return CycleGANModel.modify_commandline_options(parser, is_train)

    def __init__(self, opt, rank):
        """
        Initialize the CycleDualViewGANModel class.

        Parameters:
            opt (Option class)--  experiment flags
        """

        BaseModel.__init__(self, opt, rank)
        self.rank = rank

        self.view_net0 = CycleGANModel(opt, rank)
        self.view_net0.save_dir = os.path.join(self.view_net0.save_dir, 'view0')
        if rank == 0:
            mkdirs(self.view_net0.save_dir)

        self.view_net1 = CycleGANModel(opt,rank)
        self.view_net1.save_dir = os.path.join(self.view_net1.save_dir, 'view1')
        if rank == 0:
            mkdirs(self.view_net1.save_dir)

        if self.isTrain:
            # Specify the training losses that will be ploted and printed out in the Console
            # when the training/test scripts will call <BaseModel.get_current_losses>

            self.optimizer_G = torch.optim.Adam(itertools.chain(self.view_net0.netG_A.parameters(), self.view_net0.netG_B.parameters(), self.view_net1.netG_A.parameters(), self.view_net1.netG_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.view_net0.netD_A.parameters(), self.view_net0.netD_B.parameters(),self.view_net1.netD_A.parameters(), self.view_net1.netD_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        self.view_net0.set_input(input['view0'])
        self.view_net1.set_input(input['view1'])


    def forward(self):
        self.view_net0.forward()
        self.view_net1.forward()

    def optimize_parameters(self):

        self.forward()      # compute fake images and reconstruction images.
        # G_A and G_B
        self.set_requires_grad([self.view_net0.netD_A, self.view_net0.netD_B, self.view_net1.netD_A, self.view_net1.netD_B], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.view_net0.set_loss_G()
        self.view_net1.set_loss_G()
        self.loss_G = self.view_net0.loss_G + self.view_net1.loss_G

        self.loss_G.backward()
        self.optimizer_G.step()       # update G_A and G_B's weights
        # D_A and D_B
        self.set_requires_grad([self.view_net0.netD_A, self.view_net0.netD_B, self.view_net1.netD_A, self.view_net1.netD_B], True)
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        self.view_net0.backward_D_A()      # calculate gradients for D_A
        self.view_net0.backward_D_B()      # calculate graidents for D_B
        self.view_net1.backward_D_A()      # calculate gradients for D_A
        self.view_net1.backward_D_B()      # calculate graidents for D_B
        self.optimizer_D.step()  # update D_A and D_B's weights


    def save_networks(self, epoch):
        self.view_net0.save_networks(epoch)
        self.view_net1.save_networks(epoch)

    def get_current_losses(self):

        errors_ret = OrderedDict()
        errors_ret["loss_G"] = self.loss_G

        losses = self.view_net0.get_current_losses()
        for label, image in losses.items():
            errors_ret['view0_' + label] = losses[label]

        losses = self.view_net1.get_current_losses()
        for label, image in losses.items():
            errors_ret['view1_' + label] = losses[label]

        return errors_ret

    def get_current_visuals(self):
        """Return visualization images. train.py will display these images with visdom, and save the images to a HTML"""
        visual_ret = OrderedDict()
        visuals = self.view_net0.get_current_visuals()
        for label,image in visuals.items():
            visual_ret['view0_'+label] = visuals[label]

        visuals = self.view_net1.get_current_visuals()
        for label,image in visuals.items():
            visual_ret['view1_'+label] = visuals[label]

        return visual_ret


    def load_networks(self, epoch):
        self.view_net0.load_networks(epoch)
        self.view_net1.load_networks(epoch)


