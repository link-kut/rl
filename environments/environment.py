import roboschool, gym
from conf.constants_environments import ENVIRONMENT_ID
from environments.environment_names import Environment_Name

GYM_ENV_ID_LIST = [
    Environment_Name.CARTPOLE_V0.value,
    Environment_Name.ROBOSCHOOLANT_V1.value
]


class Environment:
    def __init__(self):
        if ENVIRONMENT_ID in GYM_ENV_ID_LIST:
            self.env = gym.make(ENVIRONMENT_ID)
        elif ENVIRONMENT_ID == Environment_Name.QUANSER_SERVO_2.value:
            pass
            #self.env =
        else:
            self.env = None

        self.n_states = self.get_n_states()
        self.n_actions = self.get_n_actions()

    def get_n_states(self):
        if ENVIRONMENT_ID == Environment_Name.CARTPOLE_V0.value:
            n_states = int(self.env.observation_space.shape[0] / 2)
        else:
            n_states = self.env.observation_space.shape[0]
        return n_states

    def get_n_actions(self):
        if ENVIRONMENT_ID == Environment_Name.ROBOSCHOOLANT_V1.value:
            n_actions = self.env.action_space.shape[0]
        else:
            n_actions = self.env.action_space.n
        return n_actions

    def reset(self):
        state = self.env.reset()
        if ENVIRONMENT_ID == Environment_Name.CARTPOLE_V0.value:
            state = state[2:]
        else:
            pass
        return state

    def step(self, action):
        next_state, reward, done, info = self.env.step(action)

        if ENVIRONMENT_ID == Environment_Name.CARTPOLE_V0.value:
            next_state = next_state[2:]
            adjusted_reward = reward / 100
        else:
            adjusted_reward = reward

        return next_state, reward, adjusted_reward, done, info

    def close(self):
        self.env.close()
