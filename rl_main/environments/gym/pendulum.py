import gym

from rl_main.conf.constants_mine import ENVIRONMENT_ID
from rl_main.environments.environment import Environment


class Pendulum_v0(Environment):
    def __init__(self):
        self.env = gym.make(ENVIRONMENT_ID.PENDULUM_V0.value)
        super(Pendulum_v0, self).__init__()

    def get_n_states(self):
        n_states = self.env.observation_space.shape[0]
        return n_states

    def get_n_actions(self):
        n_actions = 1
        return n_actions

    def get_state_shape(self):
        state_shape = self.env.observation_space.shape
        return state_shape

    def get_action_shape(self):
        action_shape = (1,)
        return action_shape

    def reset(self):
        state = self.env.reset()
        return state

    def step(self, action):
        next_state, reward, done, info = self.env.step([action])

        adjusted_reward = reward / 100

        return next_state, reward, adjusted_reward, done, info

    def close(self):
        self.env.close()
