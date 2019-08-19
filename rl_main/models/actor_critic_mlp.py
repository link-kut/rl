import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from random import random, randint
import math

from rl_main.main_constants import HIDDEN_1_SIZE, HIDDEN_2_SIZE, HIDDEN_3_SIZE, EPSILON_GREEDY_ACT

EPS_START = 0.9     # e-greedy threshold start value
EPS_END = 0.05      # e-greedy threshold end value
EPS_DECAY = 200     # e-greedy threshold decay


class ActorCriticMLP(nn.Module):
    def __init__(self, s_size, a_size, continuous, device):
        super(ActorCriticMLP, self).__init__()
        self.continuous = continuous

        if continuous:
            self.actor_fc_layer = nn.Sequential(
                nn.Linear(s_size, HIDDEN_1_SIZE),
                nn.Tanh(),
                nn.Linear(HIDDEN_1_SIZE, HIDDEN_2_SIZE),
                nn.Tanh(),
                nn.Linear(HIDDEN_2_SIZE, a_size),
                nn.Tanh(),
            )

            self.critic_fc_layer = nn.Sequential(
                nn.Linear(s_size, HIDDEN_1_SIZE),
                nn.Tanh(),
                nn.Linear(HIDDEN_1_SIZE, HIDDEN_2_SIZE),
                nn.Tanh(),
                nn.Linear(HIDDEN_2_SIZE, 1),
                nn.Tanh(),
            )
        else:
            self.actor_fc_layer = nn.Sequential(
                nn.Linear(s_size, HIDDEN_1_SIZE),
                nn.LeakyReLU(),
                nn.Linear(HIDDEN_1_SIZE, HIDDEN_2_SIZE),
                nn.LeakyReLU(),
                nn.Linear(HIDDEN_2_SIZE, a_size),
                nn.LeakyReLU(),
            )

            self.critic_fc_layer = nn.Sequential(
                nn.Linear(s_size, HIDDEN_1_SIZE),
                nn.LeakyReLU(),
                nn.Linear(HIDDEN_1_SIZE, HIDDEN_2_SIZE),
                nn.LeakyReLU(),
                nn.Linear(HIDDEN_2_SIZE, 1),
                nn.LeakyReLU(),
            )

        self.action_log_std = nn.Parameter(torch.zeros(a_size))

        self.avg_gradients = {}
        self.continuous = continuous
        self.device = device

        self.reset_average_gradients()

        self.steps_done = 0

    def reset_average_gradients(self):
        named_parameters = self.actor_fc_layer.named_parameters()
        self.avg_gradients["actor_fc_layer"] = {}
        for name, param in named_parameters:
            self.avg_gradients["actor_fc_layer"][name] = torch.zeros(size=param.size())

        named_parameters = self.critic_fc_layer.named_parameters()
        self.avg_gradients["critic_fc_layer"] = {}
        for name, param in named_parameters:
            self.avg_gradients["critic_fc_layer"][name] = torch.zeros(size=param.size())

    def pi(self, state, softmax_dim=0):
        action_mean = self.actor_fc_layer(state)
        out = F.softmax(action_mean, dim=softmax_dim)
        return out

    def act(self, state):
        state = torch.from_numpy(state).float().to(self.device)
        prob = self.pi(state).to(self.device)

        if EPSILON_GREEDY_ACT:
            eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * self.steps_done / EPS_DECAY)
            self.steps_done += 1

            if random() > eps_threshold:
                m = Categorical(prob)
                action = m.sample().item()
            else:
                action = randint(0, 7)
                print(action)
        else:
            m = Categorical(prob)

            action = m.sample().item()

        return action, prob.squeeze(0)[action].item()

    def v(self, state):
        v = self.critic_fc_layer(state)
        return v

    def __get_dist(self, state):
        state = torch.tensor(state, dtype=torch.float)
        action_mean = self.actor_fc_layer(state)
        action_log_std = self.action_log_std.expand_as(action_mean)

        return torch.distributions.Normal(action_mean, action_log_std.exp())

    def continuous_act(self, state):
        dist = self.__get_dist(state)

        if EPSILON_GREEDY_ACT:
            eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * self.steps_done / EPS_DECAY)
            self.steps_done += 1

            if random() > eps_threshold:
                action = dist.sample()
            else:
                action = torch.randn(1)
        else:
            action = dist.sample()

        action_log_probs = dist.log_prob(action).sum(-1, keepdim=True)

        return action, action_log_probs

    def evaluate(self, state):
        dist = self.__get_dist(state)
        action = dist.sample()

        action_log_probs = dist.log_prob(action).sum(-1, keepdim=True)
        dist_entropy = dist.entropy().sum(-1).mean()

        return action_log_probs, dist_entropy

    def get_gradients_for_current_parameters(self):
        gradients = {}

        named_parameters = self.actor_fc_layer.named_parameters()
        gradients["actor_fc_layer"] = {}
        for name, param in named_parameters:
            gradients["actor_fc_layer"][name] = param.grad

        named_parameters = self.critic_fc_layer.named_parameters()
        gradients["critic_fc_layer"] = {}
        for name, param in named_parameters:
            gradients["critic_fc_layer"][name] = param.grad

        return gradients

    def set_gradients_to_current_parameters(self, gradients):
        named_parameters = self.actor_fc_layer.named_parameters()
        for name, param in named_parameters:
            param.grad = gradients["actor_fc_layer"][name]

        named_parameters = self.critic_fc_layer.named_parameters()
        for name, param in named_parameters:
            param.grad = gradients["critic_fc_layer"][name]

    def accumulate_gradients(self, gradients):
        named_parameters = self.actor_fc_layer.named_parameters()
        for name, param in named_parameters:
            self.avg_gradients["actor_fc_layer"][name] += gradients["actor_fc_layer"][name]

        named_parameters = self.critic_fc_layer.named_parameters()
        for name, param in named_parameters:
            self.avg_gradients["critic_fc_layer"][name] += gradients["critic_fc_layer"][name]

    def get_average_gradients(self, num_workers):
        named_parameters = self.actor_fc_layer.named_parameters()
        for name, param in named_parameters:
            self.avg_gradients["actor_fc_layer"][name] /= num_workers

        named_parameters = self.critic_fc_layer.named_parameters()
        for name, param in named_parameters:
            self.avg_gradients["critic_fc_layer"][name] /= num_workers

    def get_parameters(self):
        parameters = {}

        named_parameters = self.actor_fc_layer.named_parameters()
        parameters['actor_fc_layer'] = {}
        for name, param in named_parameters:
            parameters["actor_fc_layer"][name] = param.data

        named_parameters = self.critic_fc_layer.named_parameters()
        parameters['critic_fc_layer'] = {}
        for name, param in named_parameters:
            parameters["critic_fc_layer"][name] = param.data

        return parameters

    def transfer_process(self, parameters, soft_transfer, soft_transfer_tau):
        named_parameters = self.actor_fc_layer.named_parameters()
        for name, param in named_parameters:
            if soft_transfer:
                param.data = param.data * soft_transfer_tau + parameters["actor_fc_layer"][name] * (1 - soft_transfer_tau)
            else:
                param.data = parameters["actor_fc_layer"][name]

        named_parameters = self.critic_fc_layer.named_parameters()
        for name, param in named_parameters:
            if soft_transfer:
                param.data = param.data * soft_transfer_tau + parameters["critic_fc_layer"][name] * (1 - soft_transfer_tau)
            else:
                param.data = parameters["critic_fc_layer"][name]
