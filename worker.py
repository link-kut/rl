# -*- coding:utf-8 -*-
import pickle
import time
import zlib
import numpy as np
from collections import deque

import paho.mqtt.client as mqtt
from Distributed_Transfer_PPO.constants import MQTT_SERVER, MQTT_PORT, ENVIRONMENT_ID
from Distributed_Transfer_PPO.constants import MQTT_TOPIC_EPISODE_DETAIL, MQTT_TOPIC_SUCCESS_DONE, MQTT_TOPIC_FAIL_DONE
from Distributed_Transfer_PPO.constants import MQTT_TOPIC_TRANSFER_ACK, MQTT_TOPIC_UPDATE_ACK, MAX_EPISODES
from Distributed_Transfer_PPO.constants import NUM_WEIGHT_TRANSFER_HIDDEN_LAYERS, WIN_REWARD, VERBOSE
from Distributed_Transfer_PPO.constants import EMA_WINDOW, SOFT_TRANSFER, SOFT_TRANSFER_TAU
from Distributed_Transfer_PPO.constants import HIDDEN_1_SIZE, HIDDEN_2_SIZE, HIDDEN_3_SIZE, GAMMA
from Distributed_Transfer_PPO.constants import MODE_GRADIENTS_UPDATE, MODE_PARAMETERS_TRANSFER

import sys
from Distributed_Transfer_PPO.logger import get_logger
import gym

from Distributed_Transfer_PPO.worker_rl_PPO import PPOAgent

from Distributed_Transfer_PPO.utils import exp_moving_average

if len(sys.argv) < 2:
  print("[ERROR]: no worker id")
  sys.exit(-1)

worker_id = int(sys.argv[1])
logger = get_logger("worker_{0}".format(worker_id))

env = gym.make(ENVIRONMENT_ID)

num_actions = 0
score = 0

n_inputs = int(env.observation_space.shape[0] / 2)
n_outputs = env.action_space.n

global_max_ema_score = 0
global_min_ema_loss = 1000000000

local_scores = []
local_losses = []

score_dequeue = deque(maxlen=100)
loss_dequeue = deque(maxlen=100)

episode_broker = -1

agent = PPOAgent(
    env,
    worker_id,
    logger,
    n_inputs=n_inputs,
    hidden_size=[HIDDEN_1_SIZE, HIDDEN_2_SIZE, HIDDEN_3_SIZE],
    n_outputs=n_outputs,
    gamma=GAMMA,
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

        if not is_success_or_fail_done:
            update_process(msg_payload['avg_gradients'])
        episode_broker = msg_payload["episode_broker"]
        
    elif msg.topic == MQTT_TOPIC_TRANSFER_ACK:
        log_msg = "[RECV] TOPIC: {0}, PAYLOAD: 'episode_broker': {1}, parameters_length: {2} \n".format(
            msg.topic,
            msg_payload['episode_broker'],
            len(msg_payload['parameters'])
        )

        logger.info(log_msg)

        if not is_success_or_fail_done:
            transfer_process(msg_payload['parameters'])
        episode_broker = msg_payload["episode_broker"]
    else:
        pass


def update_process(avg_gradients):
    agent.model.set_gradients_to_current_parameters(avg_gradients)
    agent.optimize_step()


def transfer_process(parameters):
    agent.transfer_process(parameters, NUM_WEIGHT_TRANSFER_HIDDEN_LAYERS, SOFT_TRANSFER, SOFT_TRANSFER_TAU)


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

for episode in range(MAX_EPISODES):
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

    if mean_score_over_recent_100_episodes >= WIN_REWARD:
        log_msg = "******* Worker {0} - Solved in episode {1}: Mean score = {2}".format(
            worker_id,
            episode,
            mean_score_over_recent_100_episodes
        )
        logger.info(log_msg)
        print(log_msg)

        if MODE_PARAMETERS_TRANSFER:
            parameters = agent.get_parameters(NUM_WEIGHT_TRANSFER_HIDDEN_LAYERS)
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