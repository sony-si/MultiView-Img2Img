"""
 This file was copied from github.com/karfly/learnable-triangulation-pytorch and modified for this project needs.
 The license of the file is in: github.com/karfly/learnable-triangulation-pytorch/blob/master/LICENSE
"""
import yaml
from easydict import EasyDict as edict


def load_config(path):
    with open(path) as fin:
        config = edict(yaml.safe_load(fin))

    return config
