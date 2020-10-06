import gym
import torch
from rl_main.conf.names import EnvironmentName
from rl_main.environments.environment import Environment


class Acrobot_v1(Environment):
    def __init__(self):
        self.env = gym.make(EnvironmentName.ACROBOT_V1.value)
        super(Acrobot_v1, self).__init__()
        self.action_shape = self.get_action_shape()
        self.state_shape = self.get_state_shape()

        self.continuous = False
        self.WIN_AND_LEARN_FINISH_SCORE = -100
        self.WIN_AND_LEARN_FINISH_CONTINUOUS_EPISODES = 10

    def get_n_states(self):
        n_states = int(self.env.observation_space.shape[0])
        return n_states

    def get_n_actions(self):
        n_actions = self.env.action_space.n
        return n_actions

    def get_state_shape(self):
        state_shape = list(self.env.observation_space.shape)
        state_shape[0] = int(state_shape[0])
        return tuple(state_shape)

    def get_action_shape(self):
        action_shape = (self.env.action_space.n,)
        return action_shape

    def get_action_space(self):
        return self.env.action_space

    @property
    def action_meanings(self):
        action_meanings = ["left", "stop", "right"]
        return action_meanings

    def reset(self):
        state = self.env.reset()
        return state

    def step(self, action):
        action = int(action.item())
        next_state, reward, done, info = self.env.step(action)

        adjusted_reward = reward

        return next_state, reward, adjusted_reward, done, info

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()
