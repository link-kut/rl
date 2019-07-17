import numpy as np

import torch
torch.manual_seed(0) # set random seed


def exp_moving_average(values, window):
    """ Numpy implementation of EMA
    """
    if window >= len(values):
        sma = np.mean(np.asarray(values))
        a = [sma] * len(values)
    else:
        weights = np.exp(np.linspace(-1., 0., window))
        weights /= weights.sum()
        a = np.convolve(values, weights, mode='full')[:len(values)]
        a[:window] = a[window]
    return a
