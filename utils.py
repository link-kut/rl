import math

import numpy as np

from Distributed_Transfer_PPO.constants import ENVIRONMENT_ID
import torch
torch.manual_seed(0) # set random seed
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


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


def state_preprocess_cartpole(state):
    return state[2:]
