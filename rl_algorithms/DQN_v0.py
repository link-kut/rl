import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from models.cnn import CNN

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQNAgent_v0:
    def __init__(self, env, worker_id, n_states, hidden_size, n_actions, gamma, env_render, logger, verbose):
        self.env = env

        self.worker_id = worker_id

        # discount rate
        self.gamma = gamma

        self.n_states = n_states
        self.hidden_size = hidden_size
        self.n_actions = n_actions

        self.trajectory = []

        # learning rate
        self.learning_rate = 0.001

        self.env_render = env_render
        self.logger = logger
        self.verbose = verbose

        self.model = self.build_model(self.n_states, self.hidden_size, self.n_actions)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        print("----------Worker {0}: {1}:--------".format(
            self.worker_id, "PPO",
        ))

    # Policy Network is 256-256-256-2 MLP
    def build_model(self, n_states, hidden_size, n_actions):
        model = CNN(n_states, hidden_size, n_actions, device).to(device)
        return model