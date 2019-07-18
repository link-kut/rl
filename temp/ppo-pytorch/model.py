import torch.nn as nn
import torch.nn.functional as F
import torch
import roboschool, gym

class Policy(nn.Module):
    def __init__(self, n_states, n_actions, device):
        super().__init__()

        self.actor = nn.Sequential(
            self.init_layer(nn.Linear(n_states, 64)),
            nn.Tanh(),
            self.init_layer(nn.Linear(64, 64)),
            nn.Tanh(),
        )

        self.critic = nn.Sequential(
            self.init_layer(nn.Linear(n_states, 64)),
            nn.Tanh(),
            self.init_layer(nn.Linear(64, 64)),
            nn.Tanh(),
            self.init_layer(nn.Linear(64, 1)),
        )

        # How we will define our normal distribution to sample action from
        self.action_mean = self.init_layer(nn.Linear(64, n_actions))

        self.action_log_std = nn.Parameter(torch.zeros(1, n_actions))

        self.train()

        self.device = device

    def pi(self, x):
        return self.actor(x)

    def v(self, x):
        return self.critic(x)

    # Init layer to have the proper weight initializations.
    def init_layer(self, layer):
        weight = layer.weight.data
        weight.normal_(0, 1)
        weight *= 1.0 / torch.sqrt(weight.pow(2).sum(1, keepdim=True))
        nn.init.constant_(layer.bias.data, 0)
        return layer

    def __get_dist(self, actor_features):
        action_mean = self.action_mean(actor_features)
        dist = torch.distributions.Normal(action_mean, self.action_log_std.exp())
        return dist


    def act(self, state, deterministic=False):
        state = torch.from_numpy(state).float().to(self.device)
        actor_features = self.pi(state)
        value = self.v(state)

        dist = self.__get_dist(actor_features)

        if deterministic:
            action = dist.mean
        else:
            action = dist.sample()

        print("###", action)
        print("###", dist.log_prob(action))

        action_log_probs = dist.log_prob(action).sum(-1, keepdim=True)
        dist_entropy = dist.entropy().sum(-1).mean()

        return value, action, action_log_probs

    def evaluate_actions(self, state, action):
        actor_features = self.pi(state)
        value = self.v(state)

        dist = self.__get_dist(actor_features)

        action_log_probs = dist.log_prob(action).sum(-1, keepdim=True)
        dist_entropy = dist.entropy().sum(-1).mean()

        return value, action_log_probs, dist_entropy


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    env = gym.make("RoboschoolAnt-v1")

    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.shape[0]

    p = Policy(n_states, n_actions, device)

    state = env.reset()
    value, action, action_log_probs = p.act(state)
    print(value)
    print(action)
    print(action_log_probs)

