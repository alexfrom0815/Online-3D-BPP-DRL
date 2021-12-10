import torch
import torch.nn as nn
import torch.nn.functional as F

from acktr.utils import AddBias, init

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
FixedCategorical.log_probs = lambda self, actions: log_prob_cat(
    self, actions.squeeze(-1)).view(actions.size(0), -1).sum(-1).unsqueeze(-1)

FixedCategorical.mode = lambda self: self.probs.argmax(dim=-1, keepdim=True)

# Normal
FixedNormal = torch.distributions.Normal

log_prob_normal = FixedNormal.log_prob
FixedNormal.log_probs = lambda self, actions: log_prob_normal(self, actions).sum(-1, keepdim=True)

normal_entropy = FixedNormal.entropy
FixedNormal.entropy = lambda self: normal_entropy(self).sum(-1)

FixedNormal.mode = lambda self: self.mean

# Bernoulli
FixedBernoulli = torch.distributions.Bernoulli

log_prob_bernoulli = FixedBernoulli.log_prob
FixedBernoulli.log_probs = lambda self, actions: log_prob_bernoulli(
    self, actions).view(actions.size(0), -1).sum(-1).unsqueeze(-1)

bernoulli_entropy = FixedBernoulli.entropy
FixedBernoulli.entropy = lambda self: bernoulli_entropy(self).sum(-1)
FixedBernoulli.mode = lambda self: torch.gt(self.probs, 0.5).float()

# remove the mask
def mask_softmax(mat, mask, dim=-1):
    mask = mask.float()
    mat = mat + mask * 1e4
    mat_max = torch.max(mat, dim=dim, keepdim=True)[0].detach()
    mat_exp = torch.exp(mat - mat_max)
    mat_exp = mat_exp * mask
    mat_sum = torch.sum(mat_exp, dim=dim, keepdim=True)
    mat_softmax = mat_exp / mat_sum
    return mat_softmax

class Categorical(nn.Module):

    def __init__(self, num_inputs, num_outputs):
        super(Categorical, self).__init__()

        init_ = lambda m: init(m, nn.init.orthogonal_,
                               lambda x: nn.init.constant_(x, 0),
                               gain=0.01)

        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x, mask):
        x = self.linear(x)

        p_ones = torch.ones_like(x)
        ones = torch.ones_like(mask)
        inver_mask = ones - mask

        lx = F.softmax(x - inver_mask * 14, dim=-1)
        lx = lx + 1e-5

        # branch 2
        # choose vaild actions
        fat_cat = FixedCategorical(probs=lx)

        # branch 1
        # minimaze invaild actions
        ax = F.softmax(x, dim=-1)
        bx = ax
        bx = bx * inver_mask

        # branch 3
        # magnify vaild actions dist_entropy
        dx = mask_softmax(x, mask) + 1e-12
        dx = dx * torch.log(dx)
        dx = dx * mask
        dx = -dx
        # dx = F.softmax(x - inver_mask * 14, dim=-1) + 1e-12
        # dx = dx * torch.log(dx)
        # dx = -dx

        return fat_cat, bx, dx

    def get_policy_distribution(self, x):
        x = self.linear(x)
        return x


class DiagGaussian(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(DiagGaussian, self).__init__()

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0))
        self.fc_mean = init_(nn.Linear(num_inputs, num_outputs))
        self.logstd = AddBias(torch.zeros(num_outputs))

    def forward(self, x):
        action_mean = self.fc_mean(x)
        #  An ugly hack for my KFAC implementation.
        zeros = torch.zeros(action_mean.size())
        if x.is_cuda:
            zeros = zeros.cuda()

        action_logstd = self.logstd(zeros)
        return FixedNormal(action_mean, action_logstd.exp())


class Bernoulli(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(Bernoulli, self).__init__()

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))

        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x):
        x = self.linear(x)
        return FixedBernoulli(logits=x)
