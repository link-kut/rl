# -*- coding:utf-8 -*-
import glob
import os
import pickle
import time
import zlib
from collections import deque

import numpy as np
import torch

from rl_main.conf.constants_mine import *

from rl_main.main import PROJECT_HOME
from rl_main.rl_algorithms.DQN_v0 import DQNAgent_v0
from rl_main.rl_algorithms.PPO_Continuous_Torch_v0 import PPOContinuousActionAgent_v0
from rl_main.rl_algorithms.PPO_Discrete_Torch_v0 import PPODiscreteActionAgent_v0

from rl_main.utils import exp_moving_average
import rl_main.rl_utils as rl_utils

env = rl_utils.get_environment(owner="worker")


class Worker:
    def __init__(self, logger, worker_id, worker_mqtt_client):
        self.worker_id = worker_id
        self.worker_mqtt_client = worker_mqtt_client

        if RL_ALGORITHM == RLAlgorithmName.PPO_DISCRETE_TORCH_V0:
            self.agent = PPODiscreteActionAgent_v0(
                env=env,
                worker_id=worker_id,
                gamma=GAMMA,
                env_render=ENV_RENDER,
                logger=logger,
                verbose=VERBOSE
            )
        elif RL_ALGORITHM == RLAlgorithmName.DQN_V0:
            self.agent = DQNAgent_v0(
                env=env,
                worker_id=worker_id,
                gamma=GAMMA,
                env_render=ENV_RENDER,
                logger=logger,
                verbose=VERBOSE
            )
        elif RL_ALGORITHM == RLAlgorithmName.PPO_CONTINUOUS_TORCH_V0:
            self.agent = PPOContinuousActionAgent_v0(
                env=env,
                worker_id=worker_id,
                gamma=GAMMA,
                env_render=ENV_RENDER,
                logger=logger,
                verbose=VERBOSE
            )
        else:
            self.agent = None

        self.score = 0

        self.global_max_ema_score = 0
        self.global_min_ema_loss = 1000000000

        self.local_scores = []
        self.local_losses = []

        self.score_dequeue = deque(maxlen=WIN_AND_LEARN_FINISH_CONTINUOUS_EPISODES)
        self.loss_dequeue = deque(maxlen=WIN_AND_LEARN_FINISH_CONTINUOUS_EPISODES)

        self.episode_chief = -1

        self.is_success_or_fail_done = False
        self.logger = logger

    def update_process(self, avg_gradients):
        self.agent.model.set_gradients_to_current_parameters(avg_gradients)
        self.agent.optimize_step()

    def transfer_process(self, parameters):
        self.agent.transfer_process(parameters, SOFT_TRANSFER, SOFT_TRANSFER_TAU)

    def send_msg(self, topic, msg):
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

        self.logger.info(log_msg)

        msg = pickle.dumps(msg, protocol=-1)
        msg = zlib.compress(msg)

        self.worker_mqtt_client.publish(topic=topic, payload=msg, qos=0, retain=False)

    def start_train(self):
        cnt = 0
        for episode in range(MAX_EPISODES):
            cnt += 1
            avg_gradients, loss, score = self.agent.on_episode(episode)
            self.local_losses.append(loss)
            self.local_scores.append(score)

            self.loss_dequeue.append(loss)
            self.score_dequeue.append(score)

            mean_score_over_recent_100_episodes = np.mean(self.score_dequeue)
            mean_loss_over_recent_100_episodes = np.mean(self.loss_dequeue)

            episode_msg = {
                "worker_id": self.worker_id,
                "episode": episode,
                "loss": loss,
                "score": score
            }

            if MODEL_SAVE:
                files = glob.glob(os.path.join(PROJECT_HOME, "model_save_files", "*"))
                for f in files:
                    os.remove(f)

                torch.save(
                    self.agent.model.state_dict(),
                    os.path.join(
                        PROJECT_HOME, "model_save_files",
                        "{0}_{1}_{2}.{3}.pt".format(
                            ENVIRONMENT_ID.value,
                            DEEP_LEARNING_MODEL.value,
                            RL_ALGORITHM.value,
                            episode
                        )
                    )
                )

            if mean_score_over_recent_100_episodes >= WIN_AND_LEARN_FINISH_SCORE:
                log_msg = "******* Worker {0} - Solved in episode {1}: Mean score = {2}".format(
                    self.worker_id,
                    episode,
                    mean_score_over_recent_100_episodes
                )
                self.logger.info(log_msg)
                print(log_msg)

                if MODE_PARAMETERS_TRANSFER:
                    parameters = self.agent.get_parameters()
                    episode_msg["parameters"] = parameters

                self.send_msg(MQTT_TOPIC_SUCCESS_DONE, episode_msg)
                self.is_success_or_fail_done = True
                break

            elif episode == MAX_EPISODES - 1:
                log_msg = "******* Worker {0} - Failed in episode {1}: Mean score = {2}".format(
                    self.worker_id,
                    episode,
                    mean_score_over_recent_100_episodes
                )
                self.logger.info(log_msg)
                print(log_msg)

                episode_msg["avg_gradients"] = avg_gradients

                self.send_msg(MQTT_TOPIC_FAIL_DONE, episode_msg)
                self.is_success_or_fail_done = True
                break

            else:
                ema_loss = exp_moving_average(self.local_losses, EMA_WINDOW)[-1]
                ema_score = exp_moving_average(self.local_scores, EMA_WINDOW)[-1]

                log_msg = "Worker {0}-Ep.{1:>2d}: Loss={2:6.4f} (EMA: {3:6.4f}, Mean: {4:6.4f}), Score={5:5.1f} (EMA: {6:>4.2f}, Mean: {7:>4.2f})".format(
                    self.worker_id,
                    episode,
                    loss,
                    ema_loss,
                    mean_loss_over_recent_100_episodes,
                    score,
                    ema_score,
                    mean_score_over_recent_100_episodes
                )
                self.logger.info(log_msg)
                if VERBOSE: print(log_msg)

                if MODE_GRADIENTS_UPDATE:
                    episode_msg["avg_gradients"] = avg_gradients

                self.send_msg(MQTT_TOPIC_EPISODE_DETAIL, episode_msg)

            while True:
                if episode == self.episode_chief:
                    env.close()
                    break
                time.sleep(0.01)
