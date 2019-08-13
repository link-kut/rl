import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import math
from torch.nn import init
from torchsummary import summary

from utils import get_conv2d_size, get_pool2d_size


class CNN(nn.Module):
    def __init__(self, input_height, input_width, input_channels, a_size, continuous, device):
        super(CNN, self).__init__()

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
            nn.Linear(64, a_size)
        )

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
            self.avg_gradients["conv_layer"][name] = torch.zeros_like(param.data)

        named_parameters = self.fc_layer.named_parameters()
        self.avg_gradients["fc_layer"] = {}
        for name, param in named_parameters:
            self.avg_gradients["fc_layer"][name] = torch.zeros_like(param.data)

    def forward(self, state):
        batch_size = state.size(0)
        state = self.conv_layer(state)
        state = state.view(batch_size, -1)
        state = self.fc_layer(state)
        out = F.softmax(state, dim=0)
        return out

    def act(self, state):
        state = torch.from_numpy(state).float().to(self.device)
        prob = self.forward(state).cpu()
        m = Categorical(prob)

        action = m.sample().item()

        return action, prob.squeeze(0)[action].item()

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

        return gradients

    def set_gradients_to_current_parameters(self, gradients):
        named_parameters = self.conv_layer.named_parameters()
        for name, param in named_parameters:
            param.grad = gradients["conv_layer"][name]

        named_parameters = self.fc_layer.named_parameters()
        for name, param in named_parameters:
            param.grad = gradients["fc_layer"][name]

    def accumulate_gradients(self, gradients):
        named_parameters = self.conv_layer.named_parameters()
        for name, param in named_parameters:
            self.avg_gradients["conv_layer"][name] += gradients["conv_layer"][name]

        named_parameters = self.fc_layer.named_parameters()
        for name, param in named_parameters:
            self.avg_gradients["fc_layer"][name] += gradients["fc_layer"][name]

    def get_average_gradients(self, num_workers):
        named_parameters = self.conv_layer.named_parameters()
        for name, param in named_parameters:
            self.avg_gradients["conv_layer"][name] /= num_workers

        named_parameters = self.fc_layer.named_parameters()
        for name, param in named_parameters:
            self.avg_gradients["fc_layer"][name] /= num_workers

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


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cnn = CNN(
        input_height=CNN_INPUT_HEIGHT,
        input_width=CNN_INPUT_WIDTH,
        input_channels=CNN_INPUT_CHANNELS,
        a_size=2,
        device=device
    )

    summary(cnn, input_size=(CNN_INPUT_CHANNELS, CNN_INPUT_WIDTH, CNN_INPUT_HEIGHT))

    gradients = cnn.get_gradients_for_current_parameters()
    print(gradients)


if __name__ == "__main__":
    main()
