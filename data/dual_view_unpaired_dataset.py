import os.path
from data.base_dataset import BaseDataset, get_transform
from data.view_unpaired_dataset import ViewUnpairedDataset
from data.image_folder import make_dataset, is_json_file
from datasets.cmu_panoptic import CMUPanoptic
import random


class DualViewUnpairedDataset(BaseDataset):
    """
    This dataset class holds two unpaired datasets one per view (ViewUnpairedDataset).
    """

    def __init__(self, opt):
        """
        Parameters:
            opt (Option class) -- experiment flags
        """
        BaseDataset.__init__(self, opt)
        self.view0_dataset = ViewUnpairedDataset(opt, opt.view_name0)
        self.view1_dataset = ViewUnpairedDataset(opt, opt.view_name1)

        self.personA_3d_pose_annotation_dir = os.path.join(opt.dataroot, 'annotations_3d', opt.personA)
        self.personB_3d_pose_annotation_dir = os.path.join(opt.dataroot, 'annotations_3d', opt.personB)

        self.personA_3d_pose_annotation_files = sorted(make_dataset(self.personA_3d_pose_annotation_dir, opt.max_dataset_size, is_json_file))
        self.personB_3d_pose_annotation_files = sorted(make_dataset(self.personB_3d_pose_annotation_dir, opt.max_dataset_size, is_json_file))

        self.A_3d_pose_size = len(self.personA_3d_pose_annotation_files)
        self.B_3d_pose_size = len(self.personB_3d_pose_annotation_files)

        assert(self.view0_dataset.A_size == self.view1_dataset.A_size)
        assert(self.view0_dataset.B_size == self.view1_dataset.B_size)
        assert(self.view0_dataset.A_size == self.A_3d_pose_size)
        assert(self.view0_dataset.B_size == self.B_3d_pose_size)

        #calculate A height and B height self.A_height, self.B_height
        self.A_height = self.calc_person_height(os.path.join(opt.dataroot, 'annotations_3d', opt.personA, opt.height_ref_frame_A))
        self.B_height = self.calc_person_height(os.path.join(opt.dataroot, 'annotations_3d', opt.personB, opt.height_ref_frame_B))

    def calc_person_height(self, ref_frame_name):
        person_3d_pose_gt = CMUPanoptic.read_3d_pose_annotation_file(ref_frame_name)
        #calc height - distance between nose location (pos 0) and average location of two ankles (pos 15, 16)
        import numpy as np
        return np.sqrt(np.sum((person_3d_pose_gt[0,0:3] - (np.mean(person_3d_pose_gt[(15,16),0:3],0)))**2))


    def normalize_pose(self, input_pose, scale_ratio):

        # shift coord according to ankle, scale, shift back
        if input_pose[15, 2] > input_pose[16, 2]:
            ref_idx = 16
        else:
            ref_idx = 15

        ankle_coord = input_pose[ref_idx, :]  # check which leg is lower
        out_pose = (input_pose - ankle_coord) * scale_ratio
        out_pose = out_pose + ankle_coord
        return out_pose

    def __getitem__(self, index):

        index_B = random.randint(0, self.view0_dataset.B_size - 1)  # randomize the index for domain B to avoid fixed pairs
        view0_item = self.view0_dataset.__getitem__(index, index_B)
        view1_item = self.view1_dataset.__getitem__(index, index_B)

        if (self.A_3d_pose_size > 0) and (self.A_3d_pose_size > 0):
            personA_3d_pose_gt = CMUPanoptic.read_3d_pose_annotation_file(self.personA_3d_pose_annotation_files[index % self.A_3d_pose_size])
            personB_3d_pose_gt = CMUPanoptic.read_3d_pose_annotation_file(self.personB_3d_pose_annotation_files[index_B % self.B_3d_pose_size])
            org_personA_3d_pose_gt = personA_3d_pose_gt # NOT normalized 3d pose
            org_personB_3d_pose_gt = personB_3d_pose_gt # NOT normalized 3d pose
            personA_3d_pose_gt = self.normalize_pose(personA_3d_pose_gt, self.B_height / self.A_height) # normalized 3d pose of person A according to person B height
            personB_3d_pose_gt = self.normalize_pose(personB_3d_pose_gt, self.A_height / self.B_height) # normalized 3d pose of person B according to person A height

        else:
            # Useful for debugging the multi-view framework without 3D constraints
            personA_3d_pose_gt = []
            personB_3d_pose_gt = []

        return {'view0': view0_item, 'view1': view1_item, 'personA_3d_pose_gt': personA_3d_pose_gt, 'personB_3d_pose_gt': personB_3d_pose_gt, 'org_personA_3d_pose_gt': org_personA_3d_pose_gt, 'org_personB_3d_pose_gt': org_personB_3d_pose_gt}


    def __len__(self):
         return max(self.view0_dataset.__len__(), self.view1_dataset.__len__())
