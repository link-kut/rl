# -*- coding:utf-8 -*-
import pickle
import time
import zlib

import torch

from environments.environment_rip import Environment
from utils import exp_moving_average

from conf.constants_general import MQTT_SERVER, MQTT_PORT
from conf.constants_general import MQTT_TOPIC_EPISODE_DETAIL, MQTT_TOPIC_SUCCESS_DONE, MQTT_TOPIC_FAIL_DONE
from conf.constants_general import MQTT_TOPIC_TRANSFER_ACK, MQTT_TOPIC_UPDATE_ACK
from conf.constants_general import NUM_WORKERS, EMA_WINDOW
from conf.constants_general import HIDDEN_1_SIZE, HIDDEN_2_SIZE, HIDDEN_3_SIZE
from conf.constants_general import MODE_SYNCHRONIZATION, MODE_GRADIENTS_UPDATE, MODE_PARAMETERS_TRANSFER

from conf.constants_environments import WIN_AND_LEARN_FINISH_CONTINUOUS_EPISODES

from models.actor_critic_mlp import ActorCriticMLP

import paho.mqtt.client as mqtt
from logger import get_logger

import matplotlib.pyplot as plt
from matplotlib import gridspec

from collections import deque
import numpy as np

# import warnings
# warnings.filterwarnings("ignore")

if MODE_SYNCHRONIZATION:
    print("MODE1: [SYNCHRONOUS_COMMUNICATION] vs. ASYNCHRONOUS_COMMUNICATION")
else:
    print("MODE1: SYNCHRONOUS_COMMUNICATION vs. [ASYNCHRONOUS_COMMUNICATION]")

if MODE_GRADIENTS_UPDATE:
    print("MODE2: [GRADIENTS_UPDATE] vs. NO GRADIENTS_UPDATE")
else:
    print("MODE2: GRADIENTS_UPDATE vs. [NO GRADIENTS_UPDATE]")

if MODE_PARAMETERS_TRANSFER:
    print("MODE3: [PARAMETERS_TRANSFER] vs. NO PARAMETERS_TRANSFER")
else:
    print("MODE3: PARAMETERS_TRANSFER vs. [NO PARAMETERS_TRANSFER]")

logger = get_logger("broker")

messages_received_from_workers = {}

NUM_DONE_WORKERS = 0
scores = {}
losses = {}

score_over_recent_100_episodes = {}

success_done_episode = {}
success_done_score = {}

global_max_ema_score = 0
global_min_ema_loss = 1000000000

episode_broker = 0
num_messages = 0

env = Environment()

print("env.n_states: {0}".format(env.n_states))
print("env.state_shape: {0}".format(env.state_shape))

print("env.n_actions: {0}".format(env.n_actions))
print("env.action_shape: {0}".format(env.action_shape))

num_actions = 0
score = 0

hidden_size = [HIDDEN_1_SIZE, HIDDEN_2_SIZE, HIDDEN_3_SIZE]
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = ActorCriticMLP(
    s_size=env.n_states,
    hidden_size=hidden_size,
    a_size=env.n_actions,
    device=device
).to(device)

for worker_id in range(NUM_WORKERS):
    scores[worker_id] = []
    losses[worker_id] = []

    success_done_episode[worker_id] = []
    success_done_score[worker_id] = []

    score_over_recent_100_episodes[worker_id] = deque(maxlen=WIN_AND_LEARN_FINISH_CONTINUOUS_EPISODES)


def update_loss_score(msg_payload):
    worker_id = msg_payload['worker_id']
    loss = msg_payload['loss']
    score = msg_payload['score']

    losses[worker_id].append(loss)
    scores[worker_id].append(score)
    score_over_recent_100_episodes[worker_id].append(score)

def save_graph():
    plt.clf()

    fig = plt.figure(figsize=(30, 2 * NUM_WORKERS))
    gs = gridspec.GridSpec(
        nrows=NUM_WORKERS,  # row 몇 개
        ncols=2,  # col 몇 개
        width_ratios=[5, 5],
        hspace=0.2
    )

    max_episodes = 1
    for worker_id in range(NUM_WORKERS):
        if len(scores[worker_id]) > max_episodes:
            max_episodes = len(scores[worker_id])

    ax = {}
    for row in range(NUM_WORKERS):
        ax[row] = {}
        for col in range(2):
            ax[row][col] = plt.subplot(gs[row * 2 + col])
            ax[row][col].set_xlim([0, max_episodes])
            ax[row][col].tick_params(axis='both', which='major', labelsize=10)

    for worker_id in range(NUM_WORKERS):
        ax[worker_id][0].plot(
            range(len(losses[worker_id])),
            losses[worker_id],
            c='blue'
        )
        ax[worker_id][0].plot(
            range(len(losses[worker_id])),
            exp_moving_average(losses[worker_id], EMA_WINDOW),
            c='green'
        )

        ax[worker_id][1].plot(
            range(len(scores[worker_id])),
            scores[worker_id],
            c='blue'
        )
        ax[worker_id][1].plot(
            range(len(scores[worker_id])),
            exp_moving_average(scores[worker_id], EMA_WINDOW),
            c='green'
        )

        ax[worker_id][1].scatter(
            success_done_episode[worker_id],
            success_done_score[worker_id],
            marker="*",
            s=70,
            c='red'
        )

    plt.savefig("./graphs/loss_score.png")
    plt.close('all')


def on_connect(client, userdata, flags, rc):
    logger.info("Connected with result code {}".format(rc))
    client.subscribe(MQTT_TOPIC_EPISODE_DETAIL)
    client.subscribe(MQTT_TOPIC_SUCCESS_DONE)
    client.subscribe(MQTT_TOPIC_FAIL_DONE)
    print("on_connect completed!")


def process_message(topic, msg_payload):
    global NUM_DONE_WORKERS

    update_loss_score(msg_payload)
    save_graph()

    if topic == MQTT_TOPIC_EPISODE_DETAIL and MODE_GRADIENTS_UPDATE:
        model.accumulate_gradients(msg_payload['avg_gradients'])

    elif topic == MQTT_TOPIC_SUCCESS_DONE:
        success_done_episode[msg_payload['worker_id']].append(msg_payload['episode'])
        success_done_score[msg_payload['worker_id']].append(msg_payload['score'])

        NUM_DONE_WORKERS += 1
        print("BROKER CHECK! - num_of_done_workers:", NUM_DONE_WORKERS)

    elif topic == MQTT_TOPIC_FAIL_DONE:
        NUM_DONE_WORKERS += 1
        print("BROKER CHECK! - num_of_done_workers:", NUM_DONE_WORKERS)

    else:
        pass


def on_message(client, userdata, msg):
    global episode_broker
    global messages_received_from_workers
    global num_messages

    msg_payload = zlib.decompress(msg.payload)
    msg_payload = pickle.loads(msg_payload)

    log_msg = "[RECV] TOPIC: {0}, PAYLOAD: 'episode': {1}, 'worker_id': {2}, 'loss': {3}, 'score': {4}".format(
        msg.topic,
        msg_payload['episode'],
        msg_payload['worker_id'],
        msg_payload['loss'],
        msg_payload['score']
    )
    logger.info(log_msg)

    if MODE_SYNCHRONIZATION:
        if msg_payload['episode'] not in messages_received_from_workers:
            messages_received_from_workers[msg_payload['episode']] = {}

        messages_received_from_workers[msg_payload['episode']][msg_payload["worker_id"]] = (msg.topic, msg_payload)

        if len(messages_received_from_workers[episode_broker]) == NUM_WORKERS - NUM_DONE_WORKERS:
            is_include_topic_success_done = False
            parameters_transferred = None
            worker_score_str = ""
            for worker_id in range(NUM_WORKERS):
                if worker_id in messages_received_from_workers[episode_broker]:
                    topic, msg_payload = messages_received_from_workers[episode_broker][worker_id]

                    process_message(topic=topic, msg_payload=msg_payload)

                    worker_score_str += "W{0}[{1:5.1f}/{2:5.1f}] ".format(
                        worker_id,
                        messages_received_from_workers[episode_broker][worker_id][1]['score'],
                        np.mean(score_over_recent_100_episodes[worker_id])
                    )
                    if topic == MQTT_TOPIC_SUCCESS_DONE:
                        parameters_transferred = msg_payload["parameters"]
                        is_include_topic_success_done = True

            if is_include_topic_success_done:
                send_transfer_ack(parameters_transferred)
            else:
                send_update_ack()

            messages_received_from_workers[episode_broker].clear()

            save_graph()

            print("episode_broker: {0:3d} - {1}".format(episode_broker, worker_score_str))
            episode_broker += 1
    else:
        process_message(msg.topic, msg_payload)

        if num_messages == 0 or num_messages % 200 == 0:
            save_graph()

        num_messages += 1


def send_transfer_ack(parameters_transferred):
    if MODE_PARAMETERS_TRANSFER:
        log_msg = "[SEND] TOPIC: {0}, PAYLOAD: 'episode': {1}, 'parameters_length: {2}\n".format(
            MQTT_TOPIC_TRANSFER_ACK,
            episode_broker,
            len(parameters_transferred)
        )

        transfer_msg = {
            "episode_broker": episode_broker,
            "parameters": parameters_transferred
        }
    else:
        log_msg = "[SEND] TOPIC: {0}, PAYLOAD: 'episode': {1}\n".format(
            MQTT_TOPIC_TRANSFER_ACK,
            episode_broker
        )

        transfer_msg = {
            "episode_broker": episode_broker
        }
    logger.info(log_msg)

    transfer_msg = pickle.dumps(transfer_msg, protocol=-1)
    transfer_msg = zlib.compress(transfer_msg)

    broker.publish(topic=MQTT_TOPIC_TRANSFER_ACK, payload=transfer_msg, qos=0, retain=False)

    model.reset_average_gradients()
    

def send_update_ack():
    if MODE_GRADIENTS_UPDATE:
        log_msg = "[SEND] TOPIC: {0}, PAYLOAD: 'episode': {1}, 'global_avg_grad_length: {2}\n".format(
            MQTT_TOPIC_UPDATE_ACK,
            episode_broker,
            len(model.avg_gradients)
        )

        model.get_average_gradients(NUM_WORKERS - NUM_DONE_WORKERS)

        grad_update_msg = {
            "episode_broker": episode_broker,
            "avg_gradients": model.avg_gradients
        }
    else:
        log_msg = "[SEND] TOPIC: {0}, PAYLOAD: 'episode': {1}\n".format(
            MQTT_TOPIC_UPDATE_ACK,
            episode_broker
        )

        grad_update_msg = {
            "episode_broker": episode_broker
        }
    logger.info(log_msg)

    grad_update_msg = pickle.dumps(grad_update_msg, protocol=-1)
    grad_update_msg = zlib.compress(grad_update_msg)

    broker.publish(topic=MQTT_TOPIC_UPDATE_ACK, payload=grad_update_msg, qos=0, retain=False)

    model.reset_average_gradients()


broker = mqtt.Client("dist_trans_ppo_broker")
broker.on_connect = on_connect
broker.on_message = on_message
broker.connect(MQTT_SERVER, MQTT_PORT)
broker.loop_start()

while True:
    time.sleep(1)
    if NUM_DONE_WORKERS == NUM_WORKERS:
        broker.loop_stop()
        break
