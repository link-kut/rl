import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# [GENERAL]
MY_PLATFORM = None
PYTHON_PATH = None
EMA_WINDOW = 10
VERBOSE = False
MODEL_SAVE = True

# [MQTT]
MQTT_SERVER = None
MQTT_PORT = 1883
MQTT_TOPIC_EPISODE_DETAIL = "Episode_Detail"
MQTT_TOPIC_SUCCESS_DONE = "Success_Done"
MQTT_TOPIC_FAIL_DONE = "Fail_Done"
MQTT_TOPIC_TRANSFER_ACK = "Transfer_Ack"
MQTT_TOPIC_UPDATE_ACK = "Update_Ack"
MQTT_TOPIC_ACK = "Ack"
MQTT_LOG = False

# MQTT for RIP
MQTT_SERVER_FOR_RIP = "192.168.0.10"
MQTT_PUB_TO_SERVO_POWER = 'motor_power'
MQTT_PUB_RESET = 'reset'
MQTT_SUB_FROM_SERVO = 'servo_info'
MQTT_SUB_MOTOR_LIMIT = 'motor_limit_info'
MQTT_SUB_RESET_COMPLETE = 'reset_complete'

# [WORKER]
NUM_WORKERS = 1

# [TRANSFER]
SOFT_TRANSFER = False
SOFT_TRANSFER_TAU = 0.3

# [TARGET_UPDATE]
SOFT_TARGET_UPDATE = False
SOFT_TARGET_UPDATE_TAU = 0.3

# [MLP_DEEP_LEARNING_MODEL]
HIDDEN_1_SIZE = 128
HIDDEN_2_SIZE = 128
HIDDEN_3_SIZE = 128

# [OPTIMIZATION]
MAX_EPISODES = 1000
GAMMA = 0.98 # discount factor

# [MODE]
MODE_SYNCHRONIZATION = True
MODE_GRADIENTS_UPDATE = True         # Distributed
MODE_PARAMETERS_TRANSFER = True     # Transfer
MODE_DEEP_LEARNING_MODEL = "MLP"    # "CNN" or "MLP"

# [TRAINING]
EPSILON_GREEDY_ACT = False

########################################################################################
# COPY THE FOLLOWINGS INTO "constants_mine.py" and ALTER ACCORDING TO YOUR APPLICATION #
########################################################################################
# ENV_RENDER = None
#
# PYTHON_PATH_MINE = "~/anaconda3/envs/rl/bin/python"
# MQTT_SERVER_MINE = "localhost"
#
# ENV_RENDER_MINE = False
# WIN_AND_LEARN_FINISH_SCORE_MINE = 195
# WIN_AND_LEARN_FINISH_CONTINUOUS_EPISODES_MINE = 100
#
# # [1. ENVIRONMENTS]
# ENVIRONMENT_ID = EnvironmentName.BREAKOUT_DETERMINISTIC_V4
#
# # [2. DEEP_LEARNING_MODELS]
# DEEP_LEARNING_MODEL = ModelName.CNN
#
# # [3. ALGORITHMS]
# RL_ALGORITHM = RLAlgorithmName.DQN_V0
