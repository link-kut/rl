import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.nn import init
from torchsummary import summary
from rl_main.main_constants import *

from rl_main.utils import get_conv2d_size, get_pool2d_size
import numpy as np

class ActorCriticCNN(nn.Module):
    def __init__(self, input_width, input_height, input_channels, a_size, continuous, device):
        super(ActorCriticCNN, self).__init__()

        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=8, kernel_size=2),
            nn.BatchNorm2d(8),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=2),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=1)
        )

        w, h = get_conv2d_size(w=input_width, h=input_height, kernel_size=2, padding_size=0, stride=1)
        w, h = get_conv2d_size(w=w, h=h, kernel_size=2, padding_size=0, stride=1)
        w, h = get_pool2d_size(w=w, h=h, kernel_size=2, stride=1)
        w, h = get_conv2d_size(w=w, h=h, kernel_size=2, padding_size=0, stride=1)
        w, h = get_pool2d_size(w=w, h=h, kernel_size=2, stride=1)

        self.fc_layer = nn.Sequential(
            nn.Linear(w * h * 32, 128),
            nn.LeakyReLU(),
            nn.Dropout2d(0.25),
            nn.Linear(128, 64),
            nn.LeakyReLU(),
        )

        self.fc_layer_for_continuous = nn.Sequential(
            nn.Linear(w * h * 32, 128),
            nn.Tanh(),
            nn.Dropout2d(0.25),
            nn.Linear(128, 64),
            nn.Tanh(),
        )

        self.fc_pi = nn.Linear(64, a_size)     # for pi
        self.fc_log_std = nn.Parameter(torch.zeros(1, a_size))  # for
        self.fc_v = nn.Linear(64, 1)        # for v

        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight.data)
                m.bias.data.fill_(0)

        self.continuous = continuous
        self.device = device

        self.avg_gradients = {}
        self.reset_average_gradients()

    def reset_average_gradients(self):
        named_parameters = self.conv_layer.named_parameters()
        self.avg_gradients["conv_layer"] = {}
        for name, param in named_parameters:
            self.avg_gradients["conv_layer"][name] = torch.zeros(size=param.size())

        named_parameters = self.fc_layer.named_parameters()
        self.avg_gradients["fc_layer"] = {}
        for name, param in named_parameters:
            self.avg_gradients["fc_layer"][name] = torch.zeros(size=param.size())

        named_parameters = self.fc_pi.named_parameters()
        self.avg_gradients["fc_pi"] = {}
        for name, param in named_parameters:
            self.avg_gradients["fc_pi"][name] = torch.zeros(size=param.size())

        named_parameters = self.fc_v.named_parameters()
        self.avg_gradients["fc_v"] = {}
        for name, param in named_parameters:
            self.avg_gradients["fc_v"][name] = torch.zeros(size=param.size())

    def forward(self, state):
        return self.pi(state)

    def pi(self, state, softmax_dim=-1):
        if type(state) is np.ndarray:
            state = torch.from_numpy(state).float().to(device)

        if len(state.size()) == 3:
            state = state.unsqueeze(dim=0)

        batch_size = state.size(0)
        state = self.conv_layer(state)
        state = state.view(batch_size, -1)
        state = self.fc_layer(state)
        state = self.fc_pi(state)
        out = F.softmax(state, dim=softmax_dim)
        return out

    def __get_dist(self, state):
        batch_size = state.size(0)
        state = self.conv_layer(state)
        state = state.view(batch_size, -1)
        state = self.fc_layer_for_continuous(state)
        out = self.fc_pi(state)
        out_log_std = self.fc_log_std.expand_as(out)
        return torch.distributions.Normal(out, out_log_std.exp())

    def v(self, state):
        batch_size = state.size(0)
        state = self.conv_layer(state)
        state = state.view(batch_size, -1)
        state = self.fc_layer(state)
        # state = self.fc_layer_for_continuous(state)
        v = self.fc_v(state)
        return v

    def act(self, state):
        if type(state) is np.ndarray:
            state = torch.from_numpy(state).float().to(self.device)

        prob = self.pi(state).cpu()
        m = Categorical(prob)

        action = m.sample().item()

        return action, prob.squeeze(0)[action].item()

    def continuous_act(self, state):
        dist = self.__get_dist(state)
        action = dist.sample()

        action_log_probs = dist.log_prob(action).sum(-1, keepdim=True)

        return action, action_log_probs

    def get_gradients_for_current_parameters(self):
        gradients = {}

        named_parameters = self.conv_layer.named_parameters()
        gradients["conv_layer"] = {}
        for name, param in named_parameters:
            gradients["conv_layer"][name] = param.grad

        named_parameters = self.fc_layer.named_parameters()
        gradients["fc_layer"] = {}
        for name, param in named_parameters:
            gradients["fc_layer"][name] = param.grad

        named_parameters = self.fc_pi.named_parameters()
        gradients["fc_pi"] = {}
        for name, param in named_parameters:
            gradients["fc_pi"][name] = param.grad

        named_parameters = self.fc_v.named_parameters()
        gradients["fc_v"] = {}
        for name, param in named_parameters:
            gradients["fc_v"][name] = param.grad

        return gradients

    def set_gradients_to_current_parameters(self, gradients):
        named_parameters = self.conv_layer.named_parameters()
        for name, param in named_parameters:
            param.grad = gradients["conv_layer"][name]

        named_parameters = self.fc_layer.named_parameters()
        for name, param in named_parameters:
            param.grad = gradients["fc_layer"][name]

        named_parameters = self.fc_pi.named_parameters()
        for name, param in named_parameters:
            param.grad = gradients["fc_pi"][name]

        named_parameters = self.fc_v.named_parameters()
        for name, param in named_parameters:
            param.grad = gradients["fc_v"][name]

    def accumulate_gradients(self, gradients):
        named_parameters = self.conv_layer.named_parameters()
        for name, param in named_parameters:
            self.avg_gradients["conv_layer"][name] += gradients["conv_layer"][name]

        named_parameters = self.fc_layer.named_parameters()
        for name, param in named_parameters:
            self.avg_gradients["fc_layer"][name] += gradients["fc_layer"][name]

        named_parameters = self.fc_pi.named_parameters()
        for name, param in named_parameters:
            self.avg_gradients["fc_pi"][name] += gradients["fc_pi"][name]

        named_parameters = self.fc_v.named_parameters()
        for name, param in named_parameters:
            self.avg_gradients["fc_v"][name] += gradients["fc_v"][name]

    def get_average_gradients(self, num_workers):
        named_parameters = self.conv_layer.named_parameters()
        for name, param in named_parameters:
            self.avg_gradients["conv_layer"][name] /= num_workers

        named_parameters = self.fc_layer.named_parameters()
        for name, param in named_parameters:
            self.avg_gradients["fc_layer"][name] /= num_workers

        named_parameters = self.fc_pi.named_parameters()
        for name, param in named_parameters:
            self.avg_gradients["fc_pi"][name] /= num_workers

        named_parameters = self.fc_v.named_parameters()
        for name, param in named_parameters:
            self.avg_gradients["fc_v"][name] /= num_workers

    def get_parameters(self):
        parameters = {}

        named_parameters = self.conv_layer.named_parameters()
        parameters['conv_layer'] = {}
        for name, param in named_parameters:
            parameters["conv_layer"][name] = param.data

        named_parameters = self.fc_layer.named_parameters()
        parameters['fc_layer'] = {}
        for name, param in named_parameters:
            parameters["fc_layer"][name] = param.data

        named_parameters = self.fc_pi.named_parameters()
        parameters['fc_pi'] = {}
        for name, param in named_parameters:
            parameters["fc_pi"][name] = param.data

        named_parameters = self.fc_v.named_parameters()
        parameters['fc_v'] = {}
        for name, param in named_parameters:
            parameters["fc_v"][name] = param.data

        return parameters

    def transfer_process(self, parameters, soft_transfer, soft_transfer_tau):
        named_parameters = self.conv_layer.named_parameters()
        for name, param in named_parameters:
            if soft_transfer:
                param.data = param.data * soft_transfer_tau + parameters["conv_layer"][name] * (1 - soft_transfer_tau)
            else:
                param.data = parameters["conv_layer"][name]

        named_parameters = self.fc_layer.named_parameters()
        for name, param in named_parameters:
            if soft_transfer:
                param.data = param.data * soft_transfer_tau + parameters["fc_layer"][name] * (1 - soft_transfer_tau)
            else:
                param.data = parameters["fc_layer"][name]

        named_parameters = self.fc_pi.named_parameters()
        for name, param in named_parameters:
            if soft_transfer:
                param.data = param.data * soft_transfer_tau + parameters["fc_pi"][name] * (1 - soft_transfer_tau)
            else:
                param.data = parameters["fc_pi"][name]

        named_parameters = self.fc_v.named_parameters()
        for name, param in named_parameters:
            if soft_transfer:
                param.data = param.data * soft_transfer_tau + parameters["fc_v"][name] * (1 - soft_transfer_tau)
            else:
                param.data = parameters["fc_v"][name]


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cnn = ActorCriticCNN(input_width=10, input_height=10, a_size=2, continuous=False, device=device)

    summary(cnn, input_size=(1, 10, 10))

    gradients = cnn.get_gradients_for_current_parameters()
    print(gradients)
