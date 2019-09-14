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


class ModelName(enum.Enum):
    ActorCriticModel = "Actor_Critic_Model"
    NoModel = "NoModel"


class RLAlgorithmName(enum.Enum):
    MC_Control_V0 = "MN_Control_v0"
    DQN_V0 = "DQN_v0"
    PPO_V0 = "PPO_v0"


class OptimizerName(enum.Enum):
    NESTEROV = "nesterov"
    ADAM = "Adam"
