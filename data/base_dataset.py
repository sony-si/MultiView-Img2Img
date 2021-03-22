"""This module implements an abstract base class (ABC) 'BaseDataset' for datasets.

It also includes common transformation functions (e.g., get_transform, __scale_width), which can be later used in subclasses.
"""
import random
import numpy as np
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
from abc import ABC, abstractmethod
import PIL

class BaseDataset(data.Dataset, ABC):
    """This class is an abstract base class (ABC) for datasets.

    To create a subclass, you need to implement the following four functions:
    -- <__init__>:                      initialize the class, first call BaseDataset.__init__(self, opt).
    -- <__len__>:                       return the size of dataset.
    -- <__getitem__>:                   get a data point.
    -- <modify_commandline_options>:    (optionally) add dataset-specific options and set default options.
    """

    def __init__(self, opt):
        """Initialize the class; save the options in the class

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        self.opt = opt
        self.root = opt.dataroot

    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        return parser

    @abstractmethod
    def __len__(self):
        """Return the total number of images in the dataset."""
        return 0

    @abstractmethod
    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns:
            a dictionary of data with their names. It ususally contains the data itself and its metadata information.
        """
        pass


def scale_width_and_crop_height_func(img, params):
    import PIL
    load_size = params.load_size
    crop_size = [params.crop_size_height, params.crop_size_width]
    crop_size_h = params.crop_size_height
    method = PIL.Image.BICUBIC

    use_aug = params.use_aug
    if use_aug:
        img = __scale_width(img, load_size, crop_size_h, method)
        x = random.randint(0, np.maximum(0, load_size - crop_size[1]))
        y = random.randint(0, int(img.height - crop_size[0]))
        crop_pos = [x,np.int64(y)]
        img = __crop(img, crop_pos, crop_size)
        transform = transforms.Compose([
                                      transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    else:
        img = __scale_width(img, load_size, crop_size_h, method)
        x = int(img.width - crop_size[1]) / 2
        y = int(img.height -  crop_size[0]) / 2
        crop_pos = [np.int64(x), np.int64(y)]
        img = __crop(img, crop_pos, crop_size)
        transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    return [np.array(transform(img)), crop_pos]


def get_transform(opt, grayscale=False, method=Image.BICUBIC):

    transform_list = []
    if grayscale:
        transform_list.append(transforms.Grayscale(1))
    if 'resize' in opt.preprocess:
        osize = [opt.load_size, opt.load_size]
        transform_list.append(transforms.Resize(osize, method))
    elif 'scale_width' in opt.preprocess and 'crop' in opt.preprocess:
         transform_list.append(transforms.Lambda(lambda img: scale_width_and_crop_height_func(img, opt) ))

    return transforms.Compose(transform_list)


def __scale_width(img, target_size, crop_size, method=PIL.Image.BICUBIC):
    ow, oh = img.size
    if ow == target_size and oh >= crop_size:
        return img
    w = target_size
    h = int(max(target_size * oh / ow, crop_size))
    resize_img = transforms.Resize((h, w), interpolation=method)
    return resize_img(img)


def __scale(img, target_size, method=PIL.Image.BICUBIC):
    w = target_size
    h = target_size
    resize_img = transforms.Resize((h, w), interpolation=method)
    return resize_img(img)


def __crop(img, pos, size):
    ow, oh = img.size
    x1, y1 = pos
    [th, tw] = size
    if (ow > tw or oh > th):
        return img.crop((x1, y1, x1 + tw, y1 + th))
    return img

