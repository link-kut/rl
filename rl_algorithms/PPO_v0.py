# -*- coding: utf-8 -*-

import numpy as np
import time
import torch
import torch.nn.functional as F
import torch.optim as optim
from models.actor_critic_mlp import ActorCriticMLP

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

lmbda = 0.95
eps_clip = 0.1
K_epoch = 10
c1 = 0.5
c2 = 0.01


class PPOAgent_v0:
    def __init__(self, env, worker_id, n_states, hidden_size, n_actions, gamma, env_render, logger, verbose):
        self.env = env

        self.worker_id = worker_id

        # discount rate
        self.gamma = gamma

        self.n_states = n_states
        self.hidden_size = hidden_size
        self.n_actions = n_actions

        self.trajectory = []

        # learning rate
        self.learning_rate = 0.001

        self.env_render = env_render
        self.logger = logger
        self.verbose = verbose

        self.model = self.build_model(self.n_states, self.hidden_size, self.n_actions)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

        print("----------Worker {0}: {1}:--------".format(
            self.worker_id, "PPO",
        ))

    # Policy Network is 256-256-256-2 MLP
    def build_model(self, n_states, hidden_size, n_actions):
        model = ActorCriticMLP(n_states, hidden_size, n_actions, device).to(device)
        return model

    def put_data(self, transition):
        self.trajectory.append(transition)

    def get_trajectory_data(self):
        state_lst, action_lst, reward_lst, next_state_lst, prob_action_lst, done_mask_lst = [], [], [], [], [], []

        for transition in self.trajectory:
            s, a, r, s_prime, prob_a, done = transition

            state_lst.append(s)
            action_lst.append([a])
            reward_lst.append([r])
            next_state_lst.append(s_prime)
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
        return state_lst, action_lst, reward_lst, next_state_lst, done_mask_lst, prob_action_lst

    def train_net(self):
        state_lst, action_lst, reward_lst, next_state_lst, done_mask_lst, prob_action_lst = self.get_trajectory_data()
        loss_sum = 0.0
        for i in range(K_epoch):
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

            pi = self.model.pi(state_lst, softmax_dim=1)
            new_prob_action_lst = pi.gather(dim=1, index=action_lst)
            ratio = torch.exp(torch.log(new_prob_action_lst) - torch.log(prob_action_lst))  # a/b == exp(log(a)-log(b))

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - eps_clip, 1 + eps_clip) * advantage
            entropy = new_prob_action_lst * torch.log(prob_action_lst + 1.e-10) + \
                      (1.0 - new_prob_action_lst) * torch.log(-prob_action_lst + 1.0 + 1.e-10)

            loss = -torch.min(surr1, surr2) + c1 * F.smooth_l1_loss(self.model.v(state_lst), v_target.detach()) - c2 * entropy

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimize_step()

            loss_sum += loss.mean().item()

        gradients = self.model.get_gradients_for_current_parameters()
        return gradients, loss_sum / K_epoch

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

            if done:
                break

        avg_gradients, loss = self.train_net()

        return avg_gradients, loss, score

    def get_parameters(self):
        return self.model.get_parameters()

    def transfer_process(self, parameters, soft_transfer, soft_transfer_tau):
        self.model.transfer_process(parameters, soft_transfer, soft_transfer_tau)
