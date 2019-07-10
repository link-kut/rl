MQTT_SERVER = "localhost"
MQTT_PORT = 1883
MQTT_TOPIC_EPISODE_DETAIL = "Episode_Detail"
MQTT_TOPIC_SUCCESS_DONE = "Success_Done"
MQTT_TOPIC_FAIL_DONE = "Fail_Done"

MQTT_TOPIC_TRANSFER_ACK = "Transfer_Ack"
MQTT_TOPIC_UPDATE_ACK = "Update_Ack"
MQTT_TOPIC_ACK = "Ack"

NUM_WORKERS = 8
PYTHON_PATH="~/anaconda/envs/tf20/bin/python3"

NUM_WEIGHT_TRANSFER_HIDDEN_LAYERS = 4
SCORE_BASED_TRANSFER = False
LOSS_BASED_TRANSFER = False

SOFT_TRANSFER = False
SOFT_TRANSFER_TAU = 0.3

SOFT_TARGET_UPDATE = False
SOFT_TARGET_UPDATE_TAU = 0.3

WIN_REWARD = 195
MAX_EPISODES = 1000
ENVIRONMENT_ID = "CartPole-v0"
TRANSFER_POSSIBLE_EMA_SCORE = 150
TRANSFER_POSSIBLE_EMA_LOSS = 1.0
EMA_WINDOW = 10

HIDDEN_1_SIZE = 128
HIDDEN_2_SIZE = 128
HIDDEN_3_SIZE = 128
GAMMA = 0.98 # discount factor
VERBOSE = False

MODE_SYNCHRONIZATION = True
MODE_GRADIENTS_UPDATE = True
MODE_PARAMETERS_TRANSFER = True
