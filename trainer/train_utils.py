"""
utils for training, model loading and saving
"""
import torch
import sys, os
import torch
from glob import glob
import numpy as np
from os.path import isfile


def convertMillis(millis):
    seconds = int((millis / 1000) % 60)
    minutes = int((millis / (1000 * 60)) % 60)
    hours = int((millis / (1000 * 60 * 60)))
    return hours, minutes, seconds


def convertSecs(sec):
    seconds = int(sec % 60)
    minutes = int((sec / 60) % 60)
    hours = int((sec / (60 * 60)))
    return hours, minutes, seconds


def get_val_min_ck(exp_path):
    file = glob(exp_path + 'val_min=*')
    if len(file) == 0:
        return None
    log = np.load(file[0])
    path = exp_path + "/checkpoints/" + str(log[2])
    if not isfile(path):
        return None
    # print('Found best checkpoint:', path)
    return log[2]


def find_best_checkpoint(exp_path, checkpoints):
    """if val min is presented, use that to find, otherwise use the latest one"""
    val_min_ck = get_val_min_ck(exp_path)
    if val_min_ck is None:
        checkpoints = [os.path.splitext(os.path.basename(path))[0].split('_')[-1] for path in checkpoints]
        checkpoints = np.array(checkpoints, dtype=float)
        checkpoints = np.sort(checkpoints)
        path = exp_path + "/checkpoints/" + 'checkpoint_{}h:{}m:{}s_{}.tar'.format(
            *[*convertSecs(checkpoints[-1]), checkpoints[-1]])
    else:
        path = exp_path + "/checkpoints/" + val_min_ck
    return path


def load_checkpoint(model, exp_path, checkpoint, multi_gpus=True):
    checkpoint_path = exp_path + "/checkpoints/"
    if checkpoint is None:
        checkpoints = glob(checkpoint_path + '/*')
        if len(checkpoints) == 0:
            print('No checkpoints found at {}'.format(checkpoint_path))
            return 0, 0
        path = find_best_checkpoint(exp_path, checkpoints)
    else:
        path = checkpoint_path + '{}'.format(checkpoint)
    checkpoint = torch.load(path)
    print('Loaded checkpoint from: {}'.format(path))

    state_dict = {}
    if multi_gpus:
        # load checkpoint saved by distributed data parallel
        for k, v in checkpoint['model_state_dict'].items():
            newk = k.replace('module.', '')
            state_dict[newk] = v
    else:
        state_dict = checkpoint['model_state_dict']

    model.load_state_dict(state_dict)
    epoch = checkpoint['epoch']
    training_time = checkpoint['training_time']
    return model, epoch, training_time