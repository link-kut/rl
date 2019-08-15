import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from random import sample, random, randint
import math

from rl_main.conf.constants_mine import HIDDEN_1_SIZE, HIDDEN_2_SIZE, HIDDEN_3_SIZE, EPSILON_GREEDY_ACT

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
                nn.Linear(HIDDEN_2_SIZE, HIDDEN_3_SIZE),
                nn.Tanh(),
                nn.Linear(HIDDEN_3_SIZE, a_size),
                nn.Tanh(),
            )

            self.critic_fc_layer = nn.Sequential(
                nn.Linear(s_size, HIDDEN_1_SIZE),
                nn.Tanh(),
                nn.Linear(HIDDEN_1_SIZE, HIDDEN_2_SIZE),
                nn.Tanh(),
                nn.Linear(HIDDEN_2_SIZE, HIDDEN_3_SIZE),
                nn.Tanh(),
                nn.Linear(HIDDEN_3_SIZE, 1),
                nn.Tanh(),
            )
        else:
            self.actor_fc_layer = nn.Sequential(
                nn.Linear(s_size, HIDDEN_1_SIZE),
                nn.LeakyReLU(),
                nn.Linear(HIDDEN_1_SIZE, HIDDEN_2_SIZE),
                nn.LeakyReLU(),
                nn.Linear(HIDDEN_2_SIZE, HIDDEN_3_SIZE),
                nn.LeakyReLU(),
                nn.Linear(HIDDEN_3_SIZE, a_size),
                nn.LeakyReLU(),
            )

            self.critic_fc_layer = nn.Sequential(
                nn.Linear(s_size, HIDDEN_1_SIZE),
                nn.LeakyReLU(),
                nn.Linear(HIDDEN_1_SIZE, HIDDEN_2_SIZE),
                nn.LeakyReLU(),
                nn.Linear(HIDDEN_2_SIZE, HIDDEN_3_SIZE),
                nn.LeakyReLU(),
                nn.Linear(HIDDEN_3_SIZE, 1),
                nn.LeakyReLU(),
            )

        self.action_std = 0.5
        self.action_var = torch.full((a_size,), self.action_std * self.action_std).to(device)

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

        # named_parameters = self.action_mean.named_parameters()
        # self.avg_gradients["action_mean"] = {}
        # for name, param in named_parameters:
        #     self.avg_gradients["action_mean"][name] = torch.zeros(size=param.size())

    def pi(self, state, softmax_dim=0):
        action_mean = self.actor_fc_layer(state)
        out = F.softmax(action_mean, dim=softmax_dim)
        return out

    # def __get_dist(self, state):
    #     state = state.clone().detach().requires_grad_(True)
    #     # state = torch.from_numpy(state).float().to(self.device)
    #     out, _ = self.forward(state)
    #     out = torch.tanh(self.action_mean(out))
    #     out_log_std = self.action_log_std.expand_as(out)
    #
    #     return torch.distributions.Normal(out, out_log_std.exp())

    def v(self, state):
        v = self.critic_fc_layer(state)
        return v

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

    def continuous_act(self, state):
        state = torch.tensor(state, dtype=torch.float)
        action_mean = self.actor_fc_layer(state)
        cov_mat = torch.diag(self.action_var)

        dist = torch.distributions.MultivariateNormal(action_mean, cov_mat)
        action = dist.sample()
        action_logprob = dist.log_prob(action)

        return action, action_logprob

    def evaluate(self, state, action):
        state = torch.tensor(state, dtype=torch.float)
        action_mean = torch.squeeze(self.actor_fc_layer(state))

        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var)

        dist = torch.distributions.MultivariateNormal(action_mean, cov_mat)

        action_logprobs = dist.log_prob(torch.squeeze(action))
        dist_entropy = dist.entropy()
        state_value = self.critic_fc_layer(state)

        return action_logprobs, torch.squeeze(state_value), dist_entropy

    # def continuous_act(self, state):
    #     state = torch.tensor(state, dtype=torch.float)
    #     # state = torch.from_numpy(state).float().to(self.device)
    #     dist = self.__get_dist(state)
    #     action = sample(dist.sample().tolist(), k=1)[0]
    #     _action = torch.tensor(action, dtype=torch.float)
    #     action_log_probs = sample(dist.log_prob(_action).sum(-1, keepdim=True).tolist(), k=1)[0]
    #
    #     return action, action_log_probs

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

        # named_parameters = self.action_mean.named_parameters()
        # gradients["action_mean"] = {}
        # for name, param in named_parameters:
        #     gradients["action_mean"][name] = param.grad

        return gradients

    def set_gradients_to_current_parameters(self, gradients):
        named_parameters = self.actor_fc_layer.named_parameters()
        for name, param in named_parameters:
            param.grad = gradients["actor_fc_layer"][name]

        named_parameters = self.critic_fc_layer.named_parameters()
        for name, param in named_parameters:
            param.grad = gradients["critic_fc_layer"][name]

        # named_parameters = self.action_mean.named_parameters()
        # for name, param in named_parameters:
        #     param.grad = gradients["action_mean"][name]

    def accumulate_gradients(self, gradients):
        named_parameters = self.actor_fc_layer.named_parameters()
        for name, param in named_parameters:
            self.avg_gradients["actor_fc_layer"][name] += gradients["actor_fc_layer"][name]

        named_parameters = self.critic_fc_layer.named_parameters()
        for name, param in named_parameters:
            self.avg_gradients["critic_fc_layer"][name] += gradients["critic_fc_layer"][name]

        # named_parameters = self.action_mean.named_parameters()
        # for name, param in named_parameters:
        #     self.avg_gradients["action_mean"][name] += gradients["action_mean"][name]

    def get_average_gradients(self, num_workers):
        named_parameters = self.actor_fc_layer.named_parameters()
        for name, param in named_parameters:
            self.avg_gradients["actor_fc_layer"][name] /= num_workers

        named_parameters = self.critic_fc_layer.named_parameters()
        for name, param in named_parameters:
            self.avg_gradients["critic_fc_layer"][name] /= num_workers

        # named_parameters = self.action_mean.named_parameters()
        # for name, param in named_parameters:
        #     self.avg_gradients["action_mean"][name] /= num_workers

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

        # named_parameters = self.action_mean.named_parameters()
        # parameters['action_mean'] = {}
        # for name, param in named_parameters:
        #     parameters["action_mean"][name] = param.data

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

        # named_parameters = self.action_mean.named_parameters()
        # for name, param in named_parameters:
        #     if soft_transfer:
        #         param.data = param.data * soft_transfer_tau + parameters["action_mean"][name] * (1 - soft_transfer_tau)
        #     else:
        #         param.data = parameters["action_mean"][name]
