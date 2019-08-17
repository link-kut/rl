import math
import random
from collections import namedtuple, deque

import torch.optim as optim
import torch.nn.functional as F

from rl_main.main_constants import *

from rl_main import rl_utils

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'adjusted_reward'))

BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE_PERIOD = 10


class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN_v0:
    def __init__(self, env, worker_id, gamma, env_render, logger, verbose):
        self.env = env

        self.worker_id = worker_id

        # discount rate
        self.gamma = gamma

        self.trajectory = []

        # learning rate
        self.learning_rate = 0.001

        self.env_render = env_render
        self.logger = logger
        self.verbose = verbose

        self.policy_model = rl_utils.get_rl_model(self.env)
        self.target_model = rl_utils.get_rl_model(self.env)
        self.target_model.load_state_dict(self.policy_model.state_dict())
        self.target_model.eval()

        self.optimizer = rl_utils.get_optimizer(
            parameters=self.policy_model.parameters(),
            learning_rate=self.learning_rate
        )

        self.memory = ReplayMemory(10000)
        self.steps_done = 0

        self.model = self.policy_model

    def on_episode(self, episode):
        state = self.env.reset()

        done = False
        score = 0.0

        while not done:
            if self.env_render:
                self.env.render()

            action = self.select_action(state)

            next_state, reward, adjusted_reward, done, _ = self.env.step(action.item())

            # Store the transition in memory
            self.memory.push(state, action, next_state, adjusted_reward)

            # Move to the next state
            state = next_state
            score += reward

            if done:
                break

        gradients, loss = self.train_net()

        # Update the target network, copying all weights and biases in DQN
        if episode % TARGET_UPDATE_PERIOD == 0:
            self.target_model.load_state_dict(self.policy_model.state_dict())

        return gradients, loss.item(), score

    # epsilon greedy policy
    def select_action(self, state):
        sample = random.random()
        epsilon_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * self.steps_done / EPS_DECAY)
        self.steps_done += 1
        if sample > epsilon_threshold:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self.policy_model(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.env.n_actions)]], device=device, dtype=torch.long)

    def train_net(self):
        if len(self.memory) < BATCH_SIZE:
            return

        transitions = self.memory.sample(BATCH_SIZE)

        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for detailed explanation).
        # This converts batch-array of Transitions to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=device,
            dtype=torch.bool
        )

        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.adjusted_reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_model(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        next_state_values[non_final_mask] = self.target_model(non_final_next_states).max(1)[0].detach()

        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_model.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        gradients = self.policy_model.get_gradients_for_current_parameters()

        return gradients, loss

    def get_parameters(self):
        return self.policy_model.get_parameters()

    def transfer_process(self, parameters, soft_transfer, soft_transfer_tau):
        self.policy_model.transfer_process(parameters, soft_transfer, soft_transfer_tau)

    def optimize_step(self):
        for param in self.policy_model.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()