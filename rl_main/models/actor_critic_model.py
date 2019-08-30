# https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail
import numpy as np
import torch
import torch.nn as nn

from rl_main.main_constants import *
from rl_main.models.distributions import DistCategorical, DistDiagGaussian

import torch.nn.functional as F
# from torch.distributions import Categorical
from random import random, randint
import math
from rl_main.utils import AddBiases, util_init


EPS_START = 0.9     # e-greedy threshold start value
EPS_END = 0.05      # e-greedy threshold end value
EPS_DECAY = 200     # e-greedy threshold decay


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Policy(nn.Module):
    def __init__(self, s_size, a_size, continuous, device):
        super(Policy, self).__init__()

        if MODE_DEEP_LEARNING_MODEL == "CNN":
            self.base = CNNBase(
                input_channels=s_size[0],
                input_width=s_size[1],
                input_height=s_size[2],
                continuous=continuous
            )
        elif MODE_DEEP_LEARNING_MODEL == "MLP":
            self.base = MLPBase(
                num_inputs=s_size,
                continuous=continuous
            )
        else:
            raise NotImplementedError

        self.continuous = continuous

        if self.continuous:
            num_outputs = a_size
            self.dist = DistDiagGaussian(self.base.output_size, num_outputs)
        else:
            num_outputs = a_size
            self.dist = DistCategorical(self.base.output_size, num_outputs)

        self.avg_gradients = {}
        self.continuous = continuous
        self.device = device

        self.reset_average_gradients()

        self.steps_done = 0

    def forward(self, inputs):
        raise NotImplementedError

    def act(self, inputs, deterministic=False):
        inputs = torch.tensor(inputs, dtype=torch.float)

        _, actor_features = self.base(inputs)
        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()
        action = torch.tensor([action.item()], dtype=torch.float)

        action_log_probs = dist.log_probs(action)

        return action, action_log_probs

    def get_value(self, inputs):
        value, _ = self.base(inputs)
        return value

    def evaluate_actions(self, inputs, actions):
        value, actor_features = self.base(inputs)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(actions)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy

    def reset_average_gradients(self):
        for layer_name, layer in self.base.layers_info.items():
            named_parameters = layer.named_parameters()
            self.avg_gradients[layer_name] = {}
            for name, param in named_parameters:
                self.avg_gradients[layer_name][name] = torch.zeros(size=param.size())

        named_parameters = self.dist.named_parameters()
        self.avg_gradients["actor_linear"] = {}
        for name, param in named_parameters:
            self.avg_gradients["actor_linear"][name] = torch.zeros(size=param.size())

    def get_gradients_for_current_parameters(self):
        gradients = {}

        for layer_name, layer in self.base.layers_info.items():
            named_parameters = layer.named_parameters()
            gradients[layer_name] = {}
            for name, param in named_parameters:
                gradients[layer_name][name] = param.grad

        named_parameters = self.dist.named_parameters()
        gradients["actor_linear"] = {}
        for name, param in named_parameters:
            gradients["actor_linear"][name] = param.grad

        return gradients

    def set_gradients_to_current_parameters(self, gradients):
        for layer_name, layer in self.base.layers_info.items():
            named_parameters = layer.named_parameters()
            for name, param in named_parameters:
                param.grad = gradients[layer_name][name]

        named_parameters = self.dist.named_parameters()
        for name, param in named_parameters:
            param.grad = gradients["actor_linear"][name]

    def accumulate_gradients(self, gradients):
        for layer_name, layer in self.base.layers_info.items():
            named_parameters = layer.named_parameters()
            for name, param in named_parameters:
                self.avg_gradients[layer_name][name] += gradients[layer_name][name]

        named_parameters = self.dist.named_parameters()
        for name, param in named_parameters:
            self.avg_gradients["actor_linear"][name] += gradients["actor_linear"][name]

    def get_average_gradients(self, num_workers):
        for layer_name, layer in self.base.layers_info.items():
            named_parameters = layer.named_parameters()
            for name, param in named_parameters:
                self.avg_gradients[layer_name][name] /= num_workers
        named_parameters = self.dist.named_parameters()
        for name, param in named_parameters:
            self.avg_gradients["actor_linear"][name] /= num_workers

    def get_parameters(self):
        parameters = {}

        for layer_name, layer in self.base.layers_info.items():
            named_parameters = layer.named_parameters()
            parameters[layer_name] = {}
            for name, param in named_parameters:
                parameters[layer_name][name] = param.data

        named_parameters = self.dist.named_parameters()
        parameters["actor_linear"] = {}
        for name, param in named_parameters:
            parameters["actor_linear"][name] = param.data

        return parameters

    def transfer_process(self, parameters, soft_transfer, soft_transfer_tau):
        for layer_name, layer in self.base.layers_info.items():
            named_parameters = layer.named_parameters()
            for name, param in named_parameters:
                if soft_transfer:
                    param.data = param.data * soft_transfer_tau + parameters[layer_name][name] * (1 - soft_transfer_tau)
                else:
                    param.data = parameters[layer_name][name]

        named_parameters = self.dist.named_parameters()
        for name, param in named_parameters:
            if soft_transfer:
                param.data = param.data * soft_transfer_tau + parameters["actor_linear"][name] * (1 - soft_transfer_tau)
            else:
                param.data = parameters["actor_linear"][name]


class MLPBase(nn.Module):
    def __init__(self, num_inputs, continuous):
        super(MLPBase, self).__init__()

        self.hidden_1_size = HIDDEN_1_SIZE
        self.hidden_2_size = HIDDEN_2_SIZE
        self.hidden_3_size = HIDDEN_3_SIZE
        self.continuous = continuous

        init_ = lambda m: util_init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2))

        if self.continuous:
            activation = nn.Tanh()
        else:
            activation = nn.LeakyReLU()

        self.actor = nn.Sequential(
            init_(nn.Linear(num_inputs, self.hidden_1_size)), activation,
            init_(nn.Linear(self.hidden_1_size, self.hidden_2_size)), activation,
            init_(nn.Linear(self.hidden_2_size, self.hidden_3_size)), activation,
        )

        self.critic = nn.Sequential(
            init_(nn.Linear(num_inputs, self.hidden_1_size)), activation,
            init_(nn.Linear(self.hidden_1_size, self.hidden_2_size)), activation,
            init_(nn.Linear(self.hidden_2_size, self.hidden_3_size)), activation,
        )

        self.critic_linear = init_(nn.Linear(self.hidden_3_size, 1))

        self.layers_info = {'actor':self.actor, 'critic':self.critic, 'critic_linear':self.critic_linear}

        self.train()

    def forward(self, inputs):
        hidden_critic = self.critic(inputs)
        hidden_actor = self.actor(inputs)

        return self.critic_linear(hidden_critic), hidden_actor

    @property
    def output_size(self):
        return self.hidden_3_size


class CNNBase(nn.Module):
    def __init__(self, input_channels, input_width, input_height, continuous):
        super(CNNBase, self).__init__()
        self.cnn_hidden_size = CNN_HIDDEN_SIZE

        self.hidden_1_size = HIDDEN_1_SIZE
        self.hidden_2_size = HIDDEN_2_SIZE
        self.hidden_3_size = HIDDEN_3_SIZE
        self.continuous = continuous

        init_ = lambda m: util_init(m, nn.init.orthogonal_, lambda x: nn.init. constant_(x, 0), nn.init.calculate_gain('relu'))

        from rl_main.utils import get_conv2d_size, get_pool2d_size
        w, h = get_conv2d_size(w=input_width, h=input_height, kernel_size=2, padding=0, stride=1)
        w, h = get_conv2d_size(w=w, h=h, kernel_size=2, padding=0, stride=1)
        w, h = get_pool2d_size(w=w, h=h, kernel_size=2, stride=1)
        w, h = get_conv2d_size(w=w, h=h, kernel_size=2, padding=0, stride=1)
        w, h = get_pool2d_size(w=w, h=h, kernel_size=2, stride=1)

        self.actor = nn.Sequential(
            init_(nn.Conv2d(in_channels=input_channels, out_channels=8, kernel_size=2, padding=0, stride=1)),
            nn.BatchNorm2d(num_features=8), nn.LeakyReLU(),
            init_(nn.Conv2d(in_channels=8, out_channels=4, kernel_size=2, padding=0, stride=1)),
            nn.BatchNorm2d(num_features=4), nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),
            init_(nn.Conv2d(in_channels=4, out_channels=3, kernel_size=2, padding=0, stride=1)),
            nn.BatchNorm2d(num_features=3), nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),
            Flatten(),
            init_(nn.Linear(3 * w * h, self.cnn_hidden_size)),
            nn.LeakyReLU()
        )

        init_ = lambda m: util_init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0))

        self.critic = nn.Sequential(
            init_(nn.Linear(input_width * input_height * input_channels, self.hidden_1_size)), nn.Tanh(),
            init_(nn.Linear(self.hidden_1_size, self.hidden_2_size)), nn.Tanh(),
            init_(nn.Linear(self.hidden_2_size, self.hidden_3_size)), nn.Tanh(),
        )

        self.critic_linear = init_(nn.Linear(self.hidden_3_size, 1))

        self.layers_info = {'actor': self.actor, 'critic': self.critic, 'critic_linear': self.critic_linear}

        self.train()

    def forward(self, inputs, masks):
        inputs = inputs / 255.0
        hidden_actor = self.actor(inputs)

        inputs_flatten = torch.flatten(inputs)
        hidden_critic = self.critic(inputs_flatten)

        return self.critic_linear(hidden_critic), hidden_actor

    @property
    def output_size(self):
        return self.cnn_hidden_size

