import torch
import torch.nn as nn
from rl_main.utils import AddBiases, util_init


"""
Modify standard PyTorch distributions so they are compatible with this code.
"""

#
# Standardize distribution interfaces
#

# Categorical
FixedCategorical = torch.distributions.Categorical

old_sample = FixedCategorical.sample
FixedCategorical.sample = lambda self: old_sample(self).unsqueeze(-1)

log_prob_cat = FixedCategorical.log_prob
FixedCategorical.log_probs = lambda self, actions: log_prob_cat(self, actions.squeeze(-1)).view(actions.size(0), -1).sum(-1).unsqueeze(-1)

FixedCategorical.mode = lambda self: self.probs.argmax(dim=-1, keepdim=True)

# Normal
FixedNormal = torch.distributions.Normal

log_prob_normal = FixedNormal.log_prob
FixedNormal.log_probs = lambda self, actions: log_prob_normal(self, actions).sum(-1, keepdim=True)

normal_entropy = FixedNormal.entropy
FixedNormal.entropy = lambda self: normal_entropy(self).sum(-1)

FixedNormal.mode = lambda self: self.mean


class DistCategorical(nn.Module):
    def __init__(self, actor_linear):
        super(DistCategorical, self).__init__()

        self.linear = actor_linear

    def forward(self, x):
        x = self.linear(x)
        return FixedCategorical(logits=x)


class DistDiagGaussian(nn.Module):
    def __init__(self, actor_linear, num_outputs):
        super(DistDiagGaussian, self).__init__()

        self.linear = actor_linear
        self.logstd = AddBiases(torch.zeros(num_outputs))

    def forward(self, x):
        action_mean = self.linear(x)

        #  An ugly hack for my KFAC implementation.
        zeros = torch.zeros(action_mean.size())
        if x.is_cuda:
            zeros = zeros.cuda()

        action_logstd = self.logstd(zeros)
        return FixedNormal(action_mean, action_logstd.exp())
