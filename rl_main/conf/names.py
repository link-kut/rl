import enum
import os
idx = os.getcwd().index("{0}rl".format(os.sep))
PROJECT_HOME = os.getcwd()[:idx+1] + "rl{0}".format(os.sep)


class OSName(enum.Enum):
    MAC = "MAC"
    WINDOWS = "WINDOWS"
    LINUX = "LINUX"


class EnvironmentName(enum.Enum):
    CARTPOLE_V0 = "CartPole-v0"
    BLACKJACK_V0 = "Blackjack-v0"
    QUANSER_SERVO_2 = "Quanser_Servo_2"
    CHASER_V1_MAC = os.path.join(PROJECT_HOME, "rl_main", "environments", "unity", "unity_envs", "Chaser_v1")
    CHASER_V1_WINDOWS = os.path.join(PROJECT_HOME, "rl_main", "environments", "unity", "unity_envs", "Chaser_v1.exe")
    BREAKOUT_DETERMINISTIC_V4 = "BreakoutDeterministic-v4"
    PENDULUM_V0 = 'Pendulum-v0'
    DRONE_RACING_MAC = os.path.join(PROJECT_HOME, "rl_main", "environments", "unity", "unity_envs", "DroneEnv_forMac")
    DRONE_RACING_WINDOWS = os.path.join(PROJECT_HOME, "rl_main", "environments", "unity", "unity_envs", "Dron_Racing.exe")
    GRIDWORLD_V0 = 'Gridworld-v0'
    FROZENLAKE_V0 = 'FrozenLake-v0'


class DeepLearningModelName(enum.Enum):
    ActorCriticMLP = "Actor_Critic_MLP"
    ActorCriticCNN = "Actor_Critic_CNN"
    NoModel = "NoModel"


class RLAlgorithmName(enum.Enum):
    DQN_V0 = "DQN_v0"
    PPO_V0 = "PPO_v0"
    Policy_Iteration = "DP_Policy_Iteration"
    Value_Iteration = "DP_Value_Iteration"
    Monte_Carlo_Control_V0 = "Monte_Carlo_Control_v0"


class OptimizerName(enum.Enum):
    NESTEROV = "nesterov"
    ADAM = "Adam"
