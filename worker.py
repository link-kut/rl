# -*- coding:utf-8 -*-
import pickle
import zlib
from collections import deque

from conf.constants_general import MQTT_PORT
from conf.constants_general import MQTT_TOPIC_EPISODE_DETAIL, MQTT_TOPIC_SUCCESS_DONE, MQTT_TOPIC_FAIL_DONE
from conf.constants_general import MQTT_TOPIC_TRANSFER_ACK, MQTT_TOPIC_UPDATE_ACK, MAX_EPISODES
from conf.constants_general import VERBOSE
from conf.constants_general import EMA_WINDOW, SOFT_TRANSFER, SOFT_TRANSFER_TAU
from conf.constants_general import HIDDEN_1_SIZE, HIDDEN_2_SIZE, HIDDEN_3_SIZE, GAMMA
from conf.constants_general import MODE_GRADIENTS_UPDATE, MODE_PARAMETERS_TRANSFER
from environments.environment import *

import sys

from logger import get_logger

from rl_algorithms.PPO_v0 import PPOAgent_v0

from utils import exp_moving_average

if len(sys.argv) < 2:
  print("[ERROR]: no worker id")
  sys.exit(-1)

worker_id = int(sys.argv[1])
logger = get_logger("worker_{0}".format(worker_id))

env = get_environment(owner="worker")

num_actions = 0
score = 0

global_max_ema_score = 0
global_min_ema_loss = 1000000000

local_scores = []
local_losses = []

score_dequeue = deque(maxlen=WIN_AND_LEARN_FINISH_CONTINUOUS_EPISODES)
loss_dequeue = deque(maxlen=WIN_AND_LEARN_FINISH_CONTINUOUS_EPISODES)

episode_broker = -1

agent = PPOAgent_v0(
    env=env,
    worker_id=worker_id,
    n_states=env.n_states,
    hidden_size=[HIDDEN_1_SIZE, HIDDEN_2_SIZE, HIDDEN_3_SIZE],
    n_actions=env.n_actions,
    gamma=GAMMA,
    env_render=ENV_RENDER,
    logger=logger,
    verbose=VERBOSE
)

is_success_or_fail_done = False


def on_connect(client, userdata, flags, rc):
    logger.info("Connected with result code {}".format(rc))
    client.subscribe(MQTT_TOPIC_TRANSFER_ACK)
    client.subscribe(MQTT_TOPIC_UPDATE_ACK)


def on_message(client, userdata, msg):
    global episode_broker

    msg_payload = zlib.decompress(msg.payload)
    msg_payload = pickle.loads(msg_payload)

    if msg.topic == MQTT_TOPIC_UPDATE_ACK:
        log_msg = "[RECV] TOPIC: {0}, PAYLOAD: 'episode_broker': {1}, avg_grad_length: {2} \n".format(
            msg.topic,
            msg_payload['episode_broker'],
            len(msg_payload['avg_gradients'])
        )

        logger.info(log_msg)

        if not is_success_or_fail_done and MODE_GRADIENTS_UPDATE:
            update_process(msg_payload['avg_gradients'])

        episode_broker = msg_payload["episode_broker"]
        print("Topic_Update: " + episode_broker)
        
    elif msg.topic == MQTT_TOPIC_TRANSFER_ACK:
        log_msg = "[RECV] TOPIC: {0}, PAYLOAD: 'episode_broker': {1}, parameters_length: {2} \n".format(
            msg.topic,
            msg_payload['episode_broker'],
            len(msg_payload['parameters'])
        )

        logger.info(log_msg)

        if not is_success_or_fail_done and MODE_PARAMETERS_TRANSFER:
            transfer_process(msg_payload['parameters'])

        episode_broker = msg_payload["episode_broker"]
        print("Transfer ack: " + episode_broker)

    else:
        print("pass")
        pass


def update_process(avg_gradients):
    agent.model.set_gradients_to_current_parameters(avg_gradients)
    agent.optimize_step()


def transfer_process(parameters):
    agent.transfer_process(parameters, SOFT_TRANSFER, SOFT_TRANSFER_TAU)


def send_msg(topic, msg):
    log_msg = "[SEND] TOPIC: {0}, PAYLOAD: 'episode': {1}, 'worker_id': {2} 'loss': {3}, 'score': {4} ".format(
        topic,
        msg['episode'],
        msg['worker_id'],
        msg['loss'],
        msg['score']
    )
    if topic == MQTT_TOPIC_SUCCESS_DONE:
        log_msg += "'parameters_length': {0}".format(len(msg['parameters']))
    elif topic == MQTT_TOPIC_EPISODE_DETAIL:
        log_msg += "'avg_grad_length': {0}".format(len(msg['avg_gradients']))
    elif topic == MQTT_TOPIC_FAIL_DONE:
        pass
    else:
        log_msg = None

    logger.info(log_msg)

    msg = pickle.dumps(msg, protocol=-1)
    msg = zlib.compress(msg)

    worker.publish(topic=topic, payload=msg, qos=0, retain=False)


worker = mqtt.Client("rl_worker_{0}".format(worker_id))
worker.on_connect = on_connect
worker.on_message = on_message
worker.connect(MQTT_SERVER, MQTT_PORT)

worker.loop_start()

cnt = 0
for episode in range(MAX_EPISODES):
    cnt+=1
    avg_gradients, loss, score = agent.on_episode(episode)

    local_losses.append(loss)
    local_scores.append(score)

    loss_dequeue.append(loss)
    score_dequeue.append(score)

    mean_score_over_recent_100_episodes = np.mean(score_dequeue)
    mean_loss_over_recent_100_episodes = np.mean(loss_dequeue)

    episode_msg = {
        "worker_id": worker_id,
        "episode": episode,
        "loss": loss,
        "score": score
    }

    if mean_score_over_recent_100_episodes >= WIN_AND_LEARN_FINISH_SCORE:
        log_msg = "******* Worker {0} - Solved in episode {1}: Mean score = {2}".format(
            worker_id,
            episode,
            mean_score_over_recent_100_episodes
        )
        logger.info(log_msg)
        print(log_msg)

        if MODE_PARAMETERS_TRANSFER:
            parameters = agent.get_parameters()
            episode_msg["parameters"] = parameters

        send_msg(MQTT_TOPIC_SUCCESS_DONE, episode_msg)
        is_success_or_fail_done = True
        break

    elif episode == MAX_EPISODES - 1:
        log_msg = "******* Worker {0} - Failed in episode {1}: Mean score = {2}".format(
            worker_id,
            episode,
            mean_score_over_recent_100_episodes
        )
        logger.info(log_msg)
        print(log_msg)

        episode_msg["avg_gradients"] = avg_gradients

        send_msg(MQTT_TOPIC_FAIL_DONE, episode_msg)
        is_success_or_fail_done = True
        break

    else:
        ema_loss = exp_moving_average(local_losses, EMA_WINDOW)[-1]
        ema_score = exp_moving_average(local_scores, EMA_WINDOW)[-1]

        log_msg = "Worker {0}-Ep.{1:>2d}: Loss={2:6.4f} (EMA: {3:6.4f}, Mean: {4:6.4f}), Score={5:5.1f} (EMA: {6:>4.2f}, Mean: {7:>4.2f})".format(
            worker_id,
            episode,
            loss,
            ema_loss,
            mean_loss_over_recent_100_episodes,
            score,
            ema_score,
            mean_score_over_recent_100_episodes
        )
        logger.info(log_msg)
        if VERBOSE: print(log_msg)

        if MODE_GRADIENTS_UPDATE:
            episode_msg["avg_gradients"] = avg_gradients

        send_msg(MQTT_TOPIC_EPISODE_DETAIL, episode_msg)

    while True:
        if episode == episode_broker:
            break
        time.sleep(0.01)

env.close()
time.sleep(1)
worker.loop_stop()
