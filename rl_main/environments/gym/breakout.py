import gym
import numpy as np

from rl_main.conf.constants_mine import ENVIRONMENT_ID
from rl_main.environments.environment import Environment


class BreakoutDeterministic_v4(Environment):
    def __init__(self):
        self.env = gym.make(ENVIRONMENT_ID.BREAKOUT_DETERMINISTIC_V4.value)
        super(BreakoutDeterministic_v4, self).__init__()
        self.action_shape = self.get_action_shape()
        self.state_shape = self.get_state_shape()
        self.cnn_input_height = self.state_shape[0]
        self.cnn_input_width = self.state_shape[1]
        self.cnn_input_channels = self.state_shape[2]
        self.continuous = False

    @staticmethod
    def to_grayscale(img):
        return np.mean(img, axis=2)

    @staticmethod
    def downsample(img):
        return img[::2, ::2]

    @staticmethod
    def transform_reward(reward):
        return np.sign(reward)

    def preprocess(self, img):
        gray_frame = self.to_grayscale(self.downsample(img))
        gray_frame = np.expand_dims(gray_frame, axis=0)
        return gray_frame

    def get_n_states(self):
        return None

    def get_n_actions(self):
        return self.env.action_space.n

    def get_state_shape(self):
        state_shape = (int(self.env.observation_space.shape[0]/2), int(self.env.observation_space.shape[1]/2), 1)
        return state_shape

    def get_action_shape(self):
        action_shape = self.env.action_space.n
        return action_shape,

    def reset(self):
        state = self.env.reset()
        return self.preprocess(state)

    def step(self, action):
        next_state, reward, done, info = self.env.step(action)

        adjusted_reward = self.transform_reward(reward)

        return self.preprocess(next_state), reward, adjusted_reward, done, info

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()
