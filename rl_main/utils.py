import math

import numpy as np

import torch

torch.manual_seed(0) # set random seed

def exp_moving_average(values, window):
    """ Numpy implementation of EMA
    """
    if window >= len(values):
        if len(values) == 0:
            sma = 0.0
        else:
            sma = np.mean(np.asarray(values))
        a = [sma] * len(values)
    else:
        weights = np.exp(np.linspace(-1., 0., window))
        weights /= weights.sum()
        a = np.convolve(values, weights, mode='full')[:len(values)]
        a[:window] = a[window]
    return a


def get_conv2d_size(w, h, kernel_size, padding_size, stride):
    return math.floor((w - kernel_size + 2 * padding_size) / stride) + 1, math.floor((h - kernel_size + 2 * padding_size) / stride) + 1


def get_pool2d_size(w, h, kernel_size, stride):
    return math.floor((w - kernel_size) / stride) + 1, math.floor((h - kernel_size) / stride) + 1



