import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from conf.constants_mine import HIDDEN_1_SIZE, HIDDEN_2_SIZE, HIDDEN_3_SIZE


class ActorCriticMLP(nn.Module):
    def __init__(self, s_size, a_size, continuous, device):
        super(ActorCriticMLP, self).__init__()
        self.fc0 = nn.Linear(s_size, HIDDEN_1_SIZE)
        self.fc1 = nn.Linear(HIDDEN_1_SIZE, HIDDEN_2_SIZE)
        self.fc2 = nn.Linear(HIDDEN_2_SIZE, HIDDEN_3_SIZE)
        self.fc3 = nn.Linear(HIDDEN_3_SIZE, a_size)
        self.fc3_v = nn.Linear(HIDDEN_3_SIZE, 1)

        self.fc = []
        self.fc.append(self.fc0)
        self.fc.append(self.fc1)
        self.fc.append(self.fc2)
        self.fc.append(self.fc3)

        self.avg_gradients = {}
        self.continuous = continuous
        self.device = device

        self.reset_average_gradients()

        # for m in self.modules():
        #     if isinstance(m, nn.Linear):
        #         init.kaiming_normal_(m.weight.data)
        #         init.zeros_(m.bias.data)

    def reset_average_gradients(self):
        named_parameters = self.fc0.named_parameters()
        self.avg_gradients["fc0"] = {}
        for name, param in named_parameters:
            self.avg_gradients["fc0"][name] = torch.zeros_like(param.data).to(self.device)

        named_parameters = self.fc1.named_parameters()
        self.avg_gradients["fc1"] = {}
        for name, param in named_parameters:
            self.avg_gradients["fc1"][name] = torch.zeros_like(param.data).to(self.device)

        named_parameters = self.fc2.named_parameters()
        self.avg_gradients["fc2"] = {}
        for name, param in named_parameters:
            self.avg_gradients["fc2"][name] = torch.zeros_like(param.data).to(self.device)

        named_parameters = self.fc3.named_parameters()
        self.avg_gradients["fc3"] = {}
        for name, param in named_parameters:
            self.avg_gradients["fc3"][name] = torch.zeros_like(param.data).to(self.device)

        named_parameters = self.fc3_v.named_parameters()
        self.avg_gradients["fc3_v"] = {}
        for name, param in named_parameters:
            self.avg_gradients["fc3_v"][name] = torch.zeros_like(param.data).to(self.device)

    def pi(self, state, softmax_dim=0):
        state = F.leaky_relu(self.fc0(state))
        state = F.leaky_relu(self.fc1(state))
        state = F.leaky_relu(self.fc2(state))
        state = self.fc3(state)
        out = F.softmax(state, dim=softmax_dim)
        return out

    def v(self, state):
        state = F.leaky_relu(self.fc0(state))
        state = F.leaky_relu(self.fc1(state))
        state = F.leaky_relu(self.fc2(state))
        v = self.fc3_v(state)
        return v

    def act(self, state):
        state = torch.from_numpy(state).float().to(self.device)
        prob = self.pi(state).to(self.device)
        m = Categorical(prob)

        action = m.sample().item()

        return action, prob.squeeze(0)[action].item()

    def get_gradients_for_current_parameters(self):
        gradients = {}

        named_parameters = self.fc0.named_parameters()
        gradients["fc0"] = {}
        for name, param in named_parameters:
            gradients["fc0"][name] = param.grad

        named_parameters = self.fc1.named_parameters()
        gradients["fc1"] = {}
        for name, param in named_parameters:
            gradients["fc1"][name] = param.grad

        named_parameters = self.fc2.named_parameters()
        gradients["fc2"] = {}
        for name, param in named_parameters:
            gradients["fc2"][name] = param.grad

        named_parameters = self.fc3.named_parameters()
        gradients["fc3"] = {}
        for name, param in named_parameters:
            gradients["fc3"][name] = param.grad

        named_parameters = self.fc3_v.named_parameters()
        gradients["fc3_v"] = {}
        for name, param in named_parameters:
            gradients["fc3_v"][name] = param.grad

        return gradients

    def set_gradients_to_current_parameters(self, gradients):
        named_parameters = self.fc0.named_parameters()
        for name, param in named_parameters:
            param.grad = gradients["fc0"][name]

        named_parameters = self.fc1.named_parameters()
        for name, param in named_parameters:
            param.grad = gradients["fc1"][name]

        named_parameters = self.fc2.named_parameters()
        for name, param in named_parameters:
            param.grad = gradients["fc2"][name]

        named_parameters = self.fc3.named_parameters()
        for name, param in named_parameters:
            param.grad = gradients["fc3"][name]

        named_parameters = self.fc3_v.named_parameters()
        for name, param in named_parameters:
            param.grad = gradients["fc3_v"][name]

    def accumulate_gradients(self, gradients):
        named_parameters = self.fc0.named_parameters()
        for name, param in named_parameters:
            self.avg_gradients["fc0"][name] += gradients["fc0"][name]

        named_parameters = self.fc1.named_parameters()
        for name, param in named_parameters:
            self.avg_gradients["fc1"][name] += gradients["fc1"][name]

        named_parameters = self.fc2.named_parameters()
        for name, param in named_parameters:
            self.avg_gradients["fc2"][name] += gradients["fc2"][name]

        named_parameters = self.fc3.named_parameters()
        for name, param in named_parameters:
            self.avg_gradients["fc3"][name] += gradients["fc3"][name]

        named_parameters = self.fc3_v.named_parameters()
        for name, param in named_parameters:
            self.avg_gradients["fc3_v"][name] += gradients["fc3_v"][name]

    def get_average_gradients(self, num_workers):
        named_parameters = self.fc0.named_parameters()
        for name, param in named_parameters:
            self.avg_gradients["fc0"][name] /= num_workers

        named_parameters = self.fc1.named_parameters()
        for name, param in named_parameters:
            self.avg_gradients["fc1"][name] /= num_workers

        named_parameters = self.fc2.named_parameters()
        for name, param in named_parameters:
            self.avg_gradients["fc2"][name] /= num_workers

        named_parameters = self.fc3.named_parameters()
        for name, param in named_parameters:
            self.avg_gradients["fc3"][name] /= num_workers

        named_parameters = self.fc3_v.named_parameters()
        for name, param in named_parameters:
            self.avg_gradients["fc3_v"][name] /= num_workers

    def get_parameters(self):
        parameters = {}

        named_parameters = self.fc0.named_parameters()
        parameters['fc0'] = {}
        for name, param in named_parameters:
            parameters["fc0"][name] = param.data

        named_parameters = self.fc1.named_parameters()
        parameters['fc1'] = {}
        for name, param in named_parameters:
            parameters["fc1"][name] = param.data

        named_parameters = self.fc2.named_parameters()
        parameters['fc2'] = {}
        for name, param in named_parameters:
            parameters["fc2"][name] = param.data

        named_parameters = self.fc3.named_parameters()
        parameters['fc3'] = {}
        for name, param in named_parameters:
            parameters["fc3"][name] = param.data

        named_parameters = self.fc3_v.named_parameters()
        parameters['fc3_v'] = {}
        for name, param in named_parameters:
            parameters["fc3_v"][name] = param.data

        return parameters

    def transfer_process(self, parameters, soft_transfer, soft_transfer_tau):
        named_parameters = self.fc0.named_parameters()
        for name, param in named_parameters:
            if soft_transfer:
                param.data = param.data * soft_transfer_tau + parameters["fc0"][name] * (1 - soft_transfer_tau)
            else:
                param.data = parameters["fc0"][name]

        named_parameters = self.fc1.named_parameters()
        for name, param in named_parameters:
            if soft_transfer:
                param.data = param.data * soft_transfer_tau + parameters["fc1"][name] * (1 - soft_transfer_tau)
            else:
                param.data = parameters["fc1"][name]

        named_parameters = self.fc2.named_parameters()
        for name, param in named_parameters:
            if soft_transfer:
                param.data = param.data * soft_transfer_tau + parameters["fc2"][name] * (1 - soft_transfer_tau)
            else:
                param.data = parameters["fc2"][name]

        named_parameters = self.fc3.named_parameters()
        for name, param in named_parameters:
            if soft_transfer:
                param.data = param.data * soft_transfer_tau + parameters["fc3"][name] * (1 - soft_transfer_tau)
            else:
                param.data = parameters["fc3"][name]

        named_parameters = self.fc3_v.named_parameters()
        for name, param in named_parameters:
            if soft_transfer:
                param.data = param.data * soft_transfer_tau + parameters["fc3_v"][name] * (1 - soft_transfer_tau)
            else:
                param.data = parameters["fc3_v"][name]
