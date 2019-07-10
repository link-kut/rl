import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    def __init__(self, s_size, hidden_size, a_size, device):
        super(ActorCritic, self).__init__()
        self.layer_depth = len(hidden_size) + 1
        self.fc0 = nn.Linear(s_size, hidden_size[0])
        self.fc1 = nn.Linear(hidden_size[0], hidden_size[1])
        self.fc2 = nn.Linear(hidden_size[1], hidden_size[2])
        self.fc3 = nn.Linear(hidden_size[2], a_size)
        self.fc3_v = nn.Linear(hidden_size[2], 1)

        self.fc = []
        self.fc.append(self.fc0)
        self.fc.append(self.fc1)
        self.fc.append(self.fc2)
        self.fc.append(self.fc3)

        self.avg_gradients = {}
        self.avg_gradients["fc0"] = {}
        self.avg_gradients["fc1"] = {}
        self.avg_gradients["fc2"] = {}
        self.avg_gradients["fc3"] = {}
        self.avg_gradients["fc3_v"] = {}
        self.reset_average_gradients()

        self.device = device

    def pi(self, x, softmax_dim=0):
        x = F.leaky_relu(self.fc0(x))
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x, dim=softmax_dim)

    def v(self, x):
        x = F.leaky_relu(self.fc0(x))
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        v = self.fc3_v(x)
        return v

    def act(self, state):
        state = torch.from_numpy(state).float().to(self.device)
        prob = self.pi(state).cpu()
        m = Categorical(prob)
        action = m.sample().item()
        return action, prob.squeeze(0)[action].item()

    def get_gradients_for_current_parameters(self):
        gradients = {}
        for layer_id in range(self.layer_depth):
            gradients["fc{}".format(layer_id)] = {}
            gradients["fc{}".format(layer_id)]["weights"] = self.fc[layer_id].weight.grad
            gradients["fc{}".format(layer_id)]["bias"] = self.fc[layer_id].bias.grad

        gradients["fc3_v"] = {}
        gradients["fc3_v"]["weights"] = self.fc3_v.weight.grad
        gradients["fc3_v"]["bias"] = self.fc3_v.bias.grad

        return gradients

    def set_gradients_to_current_parameters(self, gradients):
        for layer_id in range(self.layer_depth):
            self.fc[layer_id].weight.grad = gradients["fc{}".format(layer_id)]["weights"]
            self.fc[layer_id].bias.grad = gradients["fc{}".format(layer_id)]["bias"]

        self.fc3_v.weight.grad = gradients["fc3_v"]["weights"]
        self.fc3_v.bias.grad = gradients["fc3_v"]["bias"]

    def make_average_gradients(self, gradients):
        for layer_id in range(self.layer_depth):
            self.avg_gradients["fc{}".format(layer_id)]["weights"] += gradients["fc{}".format(layer_id)]["weights"]
            self.avg_gradients["fc{}".format(layer_id)]["bias"] += gradients["fc{}".format(layer_id)]["bias"]

        self.avg_gradients["fc3_v"]["weights"] += self.avg_gradients["fc3_v"]["weights"]
        self.avg_gradients["fc3_v"]["bias"] += self.avg_gradients["fc3_v"]["bias"]

    def reset_average_gradients(self):
        for layer_id in range(self.layer_depth):
            self.avg_gradients["fc{}".format(layer_id)]["weights"] = torch.zeros_like(self.fc[layer_id].weight)
            self.avg_gradients["fc{}".format(layer_id)]["bias"] = torch.zeros_like(self.fc[layer_id].bias)

        self.avg_gradients["fc3_v"]["weights"] = torch.zeros_like(self.fc3_v.weight)
        self.avg_gradients["fc3_v"]["bias"] = torch.zeros_like(self.fc3_v.bias)
