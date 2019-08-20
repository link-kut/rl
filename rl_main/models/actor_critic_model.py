# https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail
import numpy as np
import torch
import torch.nn as nn

from rl_main.main_constants import HIDDEN_1_SIZE, HIDDEN_2_SIZE, HIDDEN_3_SIZE, CNN_HIDDEN_SIZE
from rl_main.models.distributions import Categorical, DiagGaussian
from rl_main.utils import init


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Policy(nn.Module):
    def __init__(self, observation_shape, action_space, continuous):
        super(Policy, self).__init__()

        if len(observation_shape) == 3:
            self.base = CNNBase(
                input_channels=observation_shape[0],
                input_width=observation_shape[1],
                input_height=observation_shape[2],
                continuous=continuous
            )
        elif len(observation_shape) == 1:
            self.base = MLPBase(
                num_inputs=observation_shape[0],
                continuous=continuous
            )
        else:
            raise NotImplementedError

        if continuous:
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(self.base.output_size, num_outputs)
        else:
            num_outputs = action_space.n
            self.dist = Categorical(self.base.output_size, num_outputs)

    def forward(self, inputs, masks):
        raise NotImplementedError

    def act(self, inputs, masks, deterministic=False):
        value, actor_features = self.base(inputs, masks)
        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        # dist_entropy = dist.entropy().mean()

        return value, action, action_log_probs

    def get_value(self, inputs, masks):
        value, _ = self.base(inputs, masks)
        return value

    def evaluate_actions(self, inputs, action):
        value, actor_features = self.base(inputs)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy


class CNNBase(nn.Module):
    def __init__(self, input_channels, input_width, input_height, num_actions, continuous):
        super(CNNBase, self).__init__()
        self.continuous = continuous
        self.cnn_hidden_size = CNN_HIDDEN_SIZE

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init. constant_(x, 0), nn.init.calculate_gain('relu'))

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

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0))

        self.critic = nn.Sequential(
            init_(nn.Linear(input_width * input_height * input_channels, self.hidden_1_size)), nn.Tanh(),
            init_(nn.Linear(self.hidden_1_size, self.hidden_2_size)), nn.Tanh(),
            init_(nn.Linear(self.hidden_2_size, self.hidden_3_size)), nn.Tanh(),
        )

        self.critic_linear = init_(nn.Linear(self.hidden_3_size, 1))

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


class MLPBase(nn.Module):
    def __init__(self, num_inputs, continuous):
        super(MLPBase, self).__init__()

        self.continuous = continuous
        self.hidden_1_size = HIDDEN_1_SIZE
        self.hidden_2_size = HIDDEN_2_SIZE
        self.hidden_3_size = HIDDEN_3_SIZE

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), np.sqrt(2))

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
            init_(nn.Linear(num_inputs, self.hidden_1_size)), nn.Tanh(),
            init_(nn.Linear(self.hidden_1_size, self.hidden_2_size)), nn.Tanh(),
            init_(nn.Linear(self.hidden_2_size, self.hidden_3_size)), nn.Tanh(),
        )

        self.critic_linear = init_(nn.Linear(self.hidden_3_size, 1))

        self.train()

    def forward(self, inputs):
        hidden_critic = self.critic(inputs)
        hidden_actor = self.actor(inputs)

        return self.critic_linear(hidden_critic), hidden_actor

    @property
    def output_size(self):
        return self.hidden_3_size
