"""
 This file was copied from github.com/karfly/learnable-triangulation-pytorch and modified for this project needs.
 The license of the file is in: github.com/karfly/learnable-triangulation-pytorch/blob/master/LICENSE
"""
import os
import yaml
import json
import re

import torch


def config_to_str(config):
    return yaml.dump(yaml.safe_load(json.dumps(config)))  # fuck yeah


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def calc_gradient_norm(named_parameters):
    total_norm = 0.0
    for name, p in named_parameters:
        # print(name)
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2

    total_norm = total_norm ** (1. / 2)

    return total_norm
