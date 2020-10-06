import gym
import torch
import numpy as np

from rl_main.conf.names import EnvironmentName
from rl_main.environments.environment import Environment


class MountainCarContinuous_v0(Environment):
    def __init__(self):
        self.env = gym.make(EnvironmentName.MOUNTAINCARCONTINUOUS_V0.value)
        super(MountainCarContinuous_v0, self).__init__()
        self.action_shape = self.get_action_shape()
        self.state_shape = self.get_state_shape()

        self.continuous = True
        self.WIN_AND_LEARN_FINISH_SCORE = 200
        self.WIN_AND_LEARN_FINISH_CONTINUOUS_EPISODES = 10

    def get_n_states(self):
        n_states = self.env.observation_space.shape[0]
        return n_states

    def get_n_actions(self):
        n_actions = self.env.action_space.shape[0]
        return n_actions

    def get_state_shape(self):
        state_shape = self.env.observation_space.shape[0]
        return state_shape

    def get_action_shape(self):
        action_shape = (self.env.action_space.shape[0],)
        return action_shape

    def get_action_space(self):
        return self.env.action_space

    @property
    def action_meanings(self):
        action_meanings = ["LEFT", "RIGHT"]
        return action_meanings

    def reset(self):
        state = self.env.reset()
        return state

    def step(self, action):
        action = np.clip(action, self.env.action_space.low[0], self.env.action_space.high[0])
        next_state, reward, done, info = self.env.step(action)
        if next_state[0] > -0.2:
            reward = 1

        if next_state[0] >= 0.5:
            reward = 100

        adjusted_reward = reward

        return next_state, reward, adjusted_reward, done, info

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()




