import gym
import gym_gridworlds
import numpy as np
from rl_main.conf.names import EnvironmentName
from rl_main.environments.environment import Environment

# states
# 0   1   2   3
# 4   5   6   7
# 8   9   10  11
# 12  13  14  15

# actions
# 0: up 1: right 2: down 3: left

# rewards
# 0   -1  -1  -1
# -1  -1  -1  -1
# -1  -1  -1  -1
# -1  -1  -1  0

class GRIDWORLD_v0(Environment):
    def __init__(self):
        self.env = gym.make(EnvironmentName.GRIDWORLD_V0.value)
        super(GRIDWORLD_v0, self).__init__()
        self.continuous = True

    def get_n_states(self):
        n_states = self.env.observation_space.n
        return n_states

    def get_n_actions(self):
        n_actions = self.env.action_space.n
        return n_actions

    def get_state_shape(self):
        state_shape = self.env.observation_space.shape
        return state_shape

    def get_action_shape(self):
        action_shape = self.env.action_space.shape
        return action_shape

    def get_action_space(self):
        return None

    def get_state(self, post_state, action):
        next_state = -1.0
        for i, p in enumerate(self.env.P[action, post_state, :]):
            if p > 0.0:
                next_state = i
        return next_state

    def get_reward(self, action, state):
        reward = self.env.R[action, state]
        return reward

    @property
    def action_meanings(self):
        action_meanings = ["UP", "RIGHT", "DOWN", "LEFT"]
        return action_meanings

