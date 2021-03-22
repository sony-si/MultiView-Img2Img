import os.path
from data.base_dataset import BaseDataset, scale_width_and_crop_height_func
from data.image_folder import make_dataset
from PIL import Image

class ViewUnpairedDataset(BaseDataset):
    """
    This dataset class loads unpaired datasets for a single a view.
    It requires a pair of image folders, one for each person
    """

    def __init__(self, opt, subdir = ''):
        """
        Parameters:
            opt (an Option class) -- contains experiment flags
        """
        BaseDataset.__init__(self, opt)

        self.dir_A_images = os.path.join(opt.dataroot, subdir, opt.personA)
        self.dir_B_images = os.path.join(opt.dataroot, subdir, opt.personB)

        self.A_paths = sorted(make_dataset(self.dir_A_images, opt.max_dataset_size))
        self.B_paths = sorted(make_dataset(self.dir_B_images, opt.max_dataset_size))

        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B

    def __getitem__(self, index_A, index_B):

        A_path = self.A_paths[index_A % self.A_size]

        B_path = self.B_paths[index_B]
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')
        [A, crop_pos_A] = scale_width_and_crop_height_func(A_img, self.opt)
        [B, crop_pos_B] = scale_width_and_crop_height_func(B_img, self.opt)

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path, 'crop_pos_A': crop_pos_A, 'crop_pos_B': crop_pos_B}

    def __len__(self):
        return max(self.A_size, self.B_size)
