import gym
import gym_gridworlds
from rl_main.conf.names import EnvironmentName
from rl_main.environments.environment import Environment


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

    def get_state_transition_probability(self):
        P = self.env.P
        return P

    def get_reward(self):
        R = self.env.R
        return R
