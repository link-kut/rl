# -*- coding: utf-8 -*-
import sys

import numpy as np
import torch
import torch.nn.functional as F

from rl_main import rl_utils
from rl_main.main_constants import device, PPO_K_EPOCH

lmbda = 0.95
eps_clip = 0.3
c1 = 0.5
c2 = 0.01


class PPODiscreteAction_v0:
    def __init__(self, env, worker_id, gamma, env_render, logger, verbose):
        self.env = env

        self.worker_id = worker_id

        # discount rate
        self.gamma = gamma

        self.trajectory = []

        # learning rate
        self.learning_rate = 0.0001

        self.env_render = env_render
        self.logger = logger
        self.verbose = verbose

        self.model = rl_utils.get_rl_model(self.env).to(device)

        self.optimizer = rl_utils.get_optimizer(
            parameters=self.model.parameters(),
            learning_rate=self.learning_rate
        )

    def put_data(self, transition):
        self.trajectory.append(transition)

    def get_trajectory_data(self):
        state_lst, action_lst, reward_lst, next_state_lst, prob_action_lst, done_mask_lst = [], [], [], [], [], []

        for transition in self.trajectory:
            s, a, r, s_prime, prob_a, done = transition

            if type(s) is np.ndarray:
                state_lst.append(s)
            else:
                state_lst.append(s.numpy())

            action_lst.append([a])
            reward_lst.append([r])

            if type(s) is np.ndarray:
                next_state_lst.append(s_prime)
            else:
                next_state_lst.append(s_prime.numpy())

            prob_action_lst.append([prob_a])

            done_mask = 0 if done else 1
            done_mask_lst.append([done_mask])

        state_lst = torch.tensor(state_lst, dtype=torch.float).to(device)
        action_lst = torch.tensor(action_lst).to(device)
        reward_lst = torch.tensor(reward_lst).to(device)
        next_state_lst = torch.tensor(next_state_lst, dtype=torch.float).to(device)
        done_mask_lst = torch.tensor(done_mask_lst, dtype=torch.float).to(device)
        prob_action_lst = torch.tensor(prob_action_lst).to(device)

        self.trajectory.clear()
        # print("state_lst.size()", state_lst.size())
        # print("action_lst.size()", action_lst.size())
        # print("reward_lst.size()", reward_lst.size())
        # print("next_state_lst.size()", next_state_lst.size())
        # print("done_mask_lst.size()", done_mask_lst.size())
        # print("prob_action_lst.size()", prob_action_lst.size())
        return state_lst, action_lst, reward_lst, next_state_lst, done_mask_lst, prob_action_lst

    def train_net(self):
        state_lst, action_lst, reward_lst, next_state_lst, done_mask_lst, prob_action_lst = self.get_trajectory_data()
        loss_sum = 0.0
        for i in range(PPO_K_EPOCH):
            v_target = reward_lst + self.gamma * self.model.v(next_state_lst) * done_mask_lst

            delta = v_target - self.model.v(state_lst)
            delta = delta.cpu().detach().numpy()

            advantage_lst = []
            advantage = 0.0
            for delta_t in delta[::-1]:
                advantage = self.gamma * lmbda * advantage + delta_t[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float).to(device)

            pi = self.model.pi(state_lst, softmax_dim=-1)
            new_prob_action_lst = pi.gather(dim=-1, index=action_lst)

            ratio = torch.exp(torch.log(new_prob_action_lst) - torch.log(prob_action_lst))  # a/b == exp(log(a)-log(b))

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * advantage
            entropy = new_prob_action_lst * torch.log(prob_action_lst + 1.e-10) + \
                      (1.0 - new_prob_action_lst) * torch.log(-prob_action_lst + 1.0 + 1.e-10)

            loss = -torch.min(surr1, surr2) + c1 * F.smooth_l1_loss(self.model.v(state_lst), v_target.detach()) - c2 * entropy

            # print("advantage: {0}".format(advantage[:3]))
            # print("pi: {0}".format(pi[:3]))
            # print("prob: {0}".format(new_prob_action_lst[:3]))
            # print("prob_action_lst: {0}".format(prob_action_lst[:3]))
            # print("new_prob_action_lst: {0}".format(new_prob_action_lst[:3]))
            # print("ratio: {0}".format(ratio[:3]))
            # print("surr1: {0}".format(surr1[:3]))
            # print("surr2: {0}".format(surr2[:3]))
            # print("entropy: {0}".format(entropy[:3]))
            # print("self.model.v(state_lst): {0}".format(self.model.v(state_lst)[:3]))
            # print("v_target: {0}".format(v_target[:3]))
            # print("F.smooth_l1_loss(self.model.v(state_lst), v_target.detach()): {0}".format(F.smooth_l1_loss(self.model.v(state_lst), v_target.detach())))
            # print("loss: {0}".format(loss[:3]))

            # params = self.model.get_parameters()
            # for layer in params:
            #     for name in params[layer]:
            #         print(layer, name, "params[layer][name]", params[layer][name])
            #         break
            #     break
            #
            # print("GRADIENT!!!")

            self.optimizer.zero_grad()
            loss.mean().backward()

            # grads = self.model.get_gradients_for_current_parameters()
            # for layer in params:
            #     for name in params[layer]:
            #         print(layer, name, "grads[layer][name]", grads[layer][name])
            #         break
            #     break

            self.optimize_step()

            # params = self.model.get_parameters()
            # for layer in params:
            #     for name in params[layer]:
            #         print(layer, name, "params[layer][name]", params[layer][name])
            #         break
            #     break

            loss_sum += loss.mean().item()


        gradients = self.model.get_gradients_for_current_parameters()
        return gradients, loss_sum / PPO_K_EPOCH

    def optimize_step(self):
        self.optimizer.step()

    def on_episode(self, episode):
        # in CartPole-v0:
        # state = [theta, angular speed]
        state = self.env.reset()

        done = False
        score = 0.0

        while not done:
            if self.env_render:
                self.env.render()

            action, prob = self.model.act(state)

            next_state, reward, adjusted_reward, done, info = self.env.step(action)

            self.put_data((state, action, adjusted_reward, next_state, prob, done))

            state = next_state
            score += reward
        gradients, loss = self.train_net()

        return gradients, loss, score

    def get_parameters(self):
        return self.model.get_parameters()

    def transfer_process(self, parameters, soft_transfer, soft_transfer_tau):
        self.model.transfer_process(parameters, soft_transfer, soft_transfer_tau)
