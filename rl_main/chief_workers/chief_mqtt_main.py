# -*- coding:utf-8 -*-
import pickle
import sys
import time
import zlib
import sys, os

from rl_main import utils, rl_utils

idx = os.getcwd().index("{0}rl".format(os.sep))
PROJECT_HOME = os.getcwd()[:idx+1] + "rl{0}".format(os.sep)
sys.path.append(PROJECT_HOME)

from rl_main.chief_workers.chief import Chief
from rl_main.main_constants import *

import paho.mqtt.client as mqtt
from rl_main.logger import get_logger
import numpy as np

logger = get_logger("chief")

env = rl_utils.get_environment()
rl_model = rl_utils.get_rl_model(env)

utils.print_configuration(env, rl_model)

chief = Chief(logger=logger, env=env, rl_model=rl_model)


def on_chief_connect(client, userdata, flags, rc):
    msg = "Chief is successfully connected with broker@{0}".format(MQTT_SERVER)
    logger.info(msg)
    client.subscribe(MQTT_TOPIC_EPISODE_DETAIL)
    client.subscribe(MQTT_TOPIC_SUCCESS_DONE)
    client.subscribe(MQTT_TOPIC_FAIL_DONE)
    print(msg)


def on_chief_log(mqttc, obj, level, string):
    print(string)


def on_chief_message(client, userdata, msg):
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
        if msg_payload['episode'] not in chief.messages_received_from_workers:
            chief.messages_received_from_workers[msg_payload['episode']] = {}

        chief.messages_received_from_workers[msg_payload['episode']][msg_payload["worker_id"]] = (msg.topic, msg_payload)

        if len(chief.messages_received_from_workers[chief.episode_chief]) == NUM_WORKERS - chief.NUM_DONE_WORKERS:
            is_include_topic_success_done = False
            parameters_transferred = None
            worker_score_str = ""
            for worker_id in range(NUM_WORKERS):
                if worker_id in chief.messages_received_from_workers[chief.episode_chief]:
                    topic, msg_payload = chief.messages_received_from_workers[chief.episode_chief][worker_id]

                    chief.process_message(topic=topic, msg_payload=msg_payload)

                    worker_score_str += "W{0}[{1:5.1f}/{2:5.1f}] ".format(
                        worker_id,
                        chief.messages_received_from_workers[chief.episode_chief][worker_id][1]['score'],
                        np.mean(chief.score_over_recent_100_episodes[worker_id])
                    )

                    if topic == MQTT_TOPIC_SUCCESS_DONE:
                        parameters_transferred = msg_payload["parameters"]
                        is_include_topic_success_done = True

            if is_include_topic_success_done:
                transfer_msg = chief.send_transfer_ack(parameters_transferred)
                chief_mqtt_client.publish(topic=MQTT_TOPIC_TRANSFER_ACK, payload=transfer_msg, qos=0, retain=False)
            else:
                grad_update_msg = chief.send_update_ack()
                chief_mqtt_client.publish(topic=MQTT_TOPIC_UPDATE_ACK, payload=grad_update_msg, qos=0, retain=False)

            chief.messages_received_from_workers[chief.episode_chief].clear()

            chief.save_graph()

            print("episode_chief:{0:3d} - {1}".format(chief.episode_chief, worker_score_str))
            chief.episode_chief += 1
    else:
        chief. process_message(msg.topic, msg_payload)

        if chief.num_messages == 0 or chief.num_messages % 200 == 0:
            chief.save_graph()

        chief.num_messages += 1


if __name__ == "__main__":
    chief_mqtt_client = mqtt.Client("dist_trans_chief")

    chief_mqtt_client.on_connect = on_chief_connect
    chief_mqtt_client.on_message = on_chief_message
    if MQTT_LOG:
        chief.on_log = on_chief_log

    chief_mqtt_client.connect(MQTT_SERVER, MQTT_PORT)
    chief_mqtt_client.loop_start()

    while True:
        stderr = sys.stderr
        sys.stderr = sys.stdout

        try:
            time.sleep(1)
            if chief.NUM_DONE_WORKERS == NUM_WORKERS:
                chief_mqtt_client.loop_stop()
                break
        except KeyboardInterrupt as error:
            print("=== {0:>8} is aborted by keyboard interrupt".format('Chief'))
        finally:
            sys.stderr = stderr
