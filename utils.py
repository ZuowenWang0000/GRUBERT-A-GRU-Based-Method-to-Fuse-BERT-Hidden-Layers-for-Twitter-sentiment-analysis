import torch
import torch.optim as optim
import os
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from nltk import word_tokenize
from collections import namedtuple
import json
import os

class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

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


def adjust_learning_rate(optimizer, scale_factor):
    """
    Shrinks learning rate by a specified factor.

    :param optimizer: optimizer whose learning rates must be decayed
    :param scale_factor: factor to scale by
    """

    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * scale_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))

def save_checkpoint(epoch, model, optimizer, save_checkpoint_path):
    """
    Save model checkpoint.

    :param epoch: epoch number
    :param model: model
    :param optimizer: optimizer
    :param save_checkpoint_path: path where to save the checkpoint
    """
    state = {'epoch': epoch,
             'model': model,
             'optimizer': optimizer,
             # 'word_map': word_map
             }
    filename = 'checkpoint_han_'+'epoch_'+str(epoch)+'.pth.tar'
    torch.save(state, os.path.join(save_checkpoint_path, filename))

def clip_gradient(optimizer, grad_clip):
    """
    Clip gradients computed during backpropagation to prevent gradient explosion.

    :param optimizer: optimized with the gradients to be clipped
    :param grad_clip: gradient clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def get_config(config_path):
    """
    Gets the config from a given path.
    :param config_path: the path to load the config from
    """
    with open(config_path) as config_file:
        config = json.load(config_file)

    return config


class AttrDict(dict):
    """
    Small class to allow accessing a dict in dot notation (e.g. my_dict.property instead of my_dict["property"]).
    Dicts can be cast to this class, e.g. attr_dict = AttrDict(my_dict)
    """
    def __getattr__(self, name):
        return self[name]

    def __setattr__(self, name, value):
        self[name] = value


def config_to_namedtuple(obj):
    """
    Converts a JSON file in Python dict format to an AttrDict that can be accessed using dot notation.
    :param obj: a JSON dict to be converted to an AttrDict.
    """
    if isinstance(obj, dict):
        for key, value in obj.items():
            obj[key] = config_to_namedtuple(value)
        # return namedtuple('GenericDict', obj.keys())(**obj)
        return AttrDict(obj)
    elif isinstance(obj, list):
        return [config_to_namedtuple(item) for item in obj]
    else:
        return obj
