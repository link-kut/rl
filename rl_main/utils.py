import glob
import math
import os
import sys
import numpy as np
import torch
import torch.nn as nn

from rl_main import rl_utils
from rl_main.conf.names import RLAlgorithmName, ModelName

idx = os.getcwd().index("{0}rl".format(os.sep))
PROJECT_HOME = os.getcwd()[:idx+1] + "rl{0}".format(os.sep)
sys.path.append(PROJECT_HOME)

from rl_main.main_constants import MODE_SYNCHRONIZATION, MODE_GRADIENTS_UPDATE, MODE_PARAMETERS_TRANSFER, \
    ENVIRONMENT_ID, RL_ALGORITHM, DEEP_LEARNING_MODEL, PROJECT_HOME, PYTHON_PATH, MY_PLATFORM, OPTIMIZER, PPO_K_EPOCH, \
    HIDDEN_1_SIZE, HIDDEN_2_SIZE, HIDDEN_3_SIZE

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


def get_conv2d_size(w, h, kernel_size, padding, stride):
    return math.floor((w - kernel_size + 2 * padding) / stride) + 1, math.floor((h - kernel_size + 2 * padding) / stride) + 1


def get_pool2d_size(w, h, kernel_size, stride):
    return math.floor((w - kernel_size) / stride) + 1, math.floor((h - kernel_size) / stride) + 1


def print_configuration(env, rl_model):
    print("*** MODE ***")
    if MODE_SYNCHRONIZATION:
        print(" MODE1: [SYNCHRONOUS_COMMUNICATION] vs. ASYNCHRONOUS_COMMUNICATION")
    else:
        print(" MODE1: SYNCHRONOUS_COMMUNICATION vs. [ASYNCHRONOUS_COMMUNICATION]")

    if MODE_GRADIENTS_UPDATE:
        print(" MODE2: [GRADIENTS_UPDATE] vs. NO GRADIENTS_UPDATE")
    else:
        print(" MODE2: GRADIENTS_UPDATE vs. [NO GRADIENTS_UPDATE]")

    if MODE_PARAMETERS_TRANSFER:
        print(" MODE3: [PARAMETERS_TRANSFER] vs. NO PARAMETERS_TRANSFER")
    else:
        print(" MODE3: PARAMETERS_TRANSFER vs. [NO PARAMETERS_TRANSFER]")

    print("\n*** MY_PLATFORM & ENVIRONMENT ***")
    print(" Platform:" + MY_PLATFORM.value)
    print(" Environment Name:" + ENVIRONMENT_ID.value)
    print(" Action Space: {0} - {1}".format(env.get_n_actions(), env.action_meaning))

    print("\n*** RL ALGORITHM ***")
    print(" RL Algorithm:" + RL_ALGORITHM.value)
    if RL_ALGORITHM == RLAlgorithmName.PPO_CONTINUOUS_TORCH_V0 or RL_ALGORITHM == RLAlgorithmName.PPO_DISCRETE_TORCH_V0:
        print(" PPO_K_EPOCH: {0}".format(PPO_K_EPOCH))

    print("\n*** MODEL ***")
    print(" Deep Learning Model:" + DEEP_LEARNING_MODEL.value)
    if DEEP_LEARNING_MODEL == ModelName.ActorCriticCNN:
        print(" input_width: {0}, input_height: {1}, input_channels: {2}, a_size: {3}, continuous: {4}".format(
            rl_model.input_width,
            rl_model.input_height,
            rl_model.input_channels,
            rl_model.a_size,
            rl_model.continuous
        ))
    elif DEEP_LEARNING_MODEL == ModelName.ActorCriticMLP:
        print(" s_size: {0}, hidden_1: {1}, hidden_2: {2}, hidden_3: {3}, a_size: {4}, continuous: {5}".format(
            rl_model.s_size,
            rl_model.hidden_1_size,
            rl_model.hidden_2_size,
            rl_model.hidden_3_size,
            rl_model.a_size,
            rl_model.continuous
        ))

    print("\n*** Optimizer ***")
    print(" Optimizer:" + OPTIMIZER.value)

    print()


def ask_file_removal():
    response = input("DELETE All Graphs, Logs, and Model Files? [y/n]: ")

    if response == "Y" or response == "y":
        files = glob.glob(os.path.join(PROJECT_HOME, "graphs", "*"))
        for f in files:
            os.remove(f)

        files = glob.glob(os.path.join(PROJECT_HOME, "logs", "*"))
        for f in files:
            os.remove(f)

        files = glob.glob(os.path.join(PROJECT_HOME, "out_err", "*"))
        for f in files:
            os.remove(f)

        files = glob.glob(os.path.join(PROJECT_HOME, "model_save_files", "*"))
        for f in files:
            os.remove(f)


def make_output_folders():
    if not os.path.exists(os.path.join(PROJECT_HOME, "graphs")):
        os.makedirs(os.path.join(PROJECT_HOME, "graphs"))

    if not os.path.exists(os.path.join(PROJECT_HOME, "logs")):
        os.makedirs(os.path.join(PROJECT_HOME, "logs"))

    if not os.path.exists(os.path.join(PROJECT_HOME, "out_err")):
        os.makedirs(os.path.join(PROJECT_HOME, "out_err"))

    if not os.path.exists(os.path.join(PROJECT_HOME, "model_save_files")):
        os.makedirs(os.path.join(PROJECT_HOME, "model_save_files"))


def run_chief():
    try:
        os.system(PYTHON_PATH + " " + os.path.join(PROJECT_HOME, "rl_main", "chief_workers", "chief_mqtt_main.py"))
        sys.stdout = open(os.path.join(PROJECT_HOME, "out_err", "chief_stdout.out"), "wb")
        sys.stderr = open(os.path.join(PROJECT_HOME, "out_err", "chief_stderr.out"), "wb")
    except KeyboardInterrupt:
        sys.stdout.flush()
        sys.stdout.flush()


def run_worker(worker_id):
    try:
        os.system(PYTHON_PATH + " " + os.path.join(PROJECT_HOME, "rl_main", "chief_workers", "worker_mqtt_main.py") + " {0}".format(worker_id))
        sys.stdout = open(os.path.join(PROJECT_HOME, "out_err", "worker_{0}_stdout.out").format(worker_id), "wb")
        sys.stderr = open(os.path.join(PROJECT_HOME, "out_err", "worker_{0}_stderr.out").format(worker_id), "wb")
    except KeyboardInterrupt:
        sys.stdout.flush()
        sys.stderr.flush()


class AddBias(nn.Module):
    def __init__(self, bias):
        super(AddBias, self).__init__()
        self._bias = nn.Parameter(bias.unsqueeze(1))

    def forward(self, x):
        if x.dim() == 2:
            bias = self._bias.t().view(1, -1)
        else:
            bias = self._bias.t().view(1, -1, 1, 1)

        return x + bias


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module