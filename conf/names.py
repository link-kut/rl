import enum


class EnvironmentName(enum.Enum):
    CARTPOLE_V0 = "CartPole-v0"
    QUANSER_SERVO_2 = "Quanser_Servo_2"
    CHASER_V1 = "unity_envs/Chaser_v1"
    BREAKOUT_DETERMINISTIC_V4 = "BreakoutDeterministic-v4"


class ModelName(enum.Enum):
    ActorCriticMLP = "Actor_Critic_MLP"
    CNN = "CNN"


class RLAlgorithmName(enum.Enum):
    DQN_V0 = "DQN-v0"
    PPO_DISCRETE_V0 = "PPO-Discrete-v0"
