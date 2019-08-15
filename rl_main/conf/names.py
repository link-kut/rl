import enum
import os
idx = os.getcwd().index("{0}rl".format(os.sep))
PROJECT_HOME = os.getcwd()[:idx+1] + "rl{0}".format(os.sep)


class EnvironmentName(enum.Enum):
    CARTPOLE_V0 = "CartPole-v0"
    QUANSER_SERVO_2 = "Quanser_Servo_2"
    CHASER_V1 = os.path.join(PROJECT_HOME, "rl_main", "environments", "unity", "unity_envs", "Chaser_v1")
    BREAKOUT_DETERMINISTIC_V4 = "BreakoutDeterministic-v4"
    PENDULUM_V0 = 'Pendulum-v0'
    DRONE_RACING = "Drone_Racing"


class ModelName(enum.Enum):
    ActorCriticMLP = "Actor_Critic_MLP"
    CNN = "CNN"
    ActorCriticCNN = "Actor_Critic_CNN"


class RLAlgorithmName(enum.Enum):
    DQN_V0 = "DQN_v0"
    PPO_DISCRETE_TORCH_V0 = "PPO_Discrete_Torch_v0"
    PPO_CONTINUOUS_TORCH_V0 = "PPO_Continuous_Torch_v0"
