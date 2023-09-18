import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from acktr.algo.kfac import KFACOptimizer
import sys

class ACKTR():
    def __init__(self,
                 actor_critic,
                 value_loss_coef,
                 entropy_coef,
                 invaild_coef,
                 lr = None,
                 eps = None,
                 alpha = None,
                 max_grad_norm = None,
                 acktr = False,
                 args = None):

        self.actor_critic = actor_critic
        self.acktr = acktr

        self.value_loss_coef = value_loss_coef
        self.invaild_coef = invaild_coef
        self.max_grad_norm = max_grad_norm

        self.loss_func = nn.MSELoss(reduce=False, size_average=True)
        self.entropy_coef = entropy_coef
        self.args = args

        if acktr:
            self.optimizer = KFACOptimizer(actor_critic)
        else:
            self.optimizer = optim.RMSprop(actor_critic.parameters(), lr, eps=eps, alpha=alpha)


    def update(self, rollouts):
        # check_nan(self.actor_critic, 1)
        obs_shape = rollouts.obs.size()[2:]
        action_shape = rollouts.actions.size()[-1]
        num_steps, num_processes, _ = rollouts.rewards.size()
        mask_size = rollouts.location_masks.size()[-1]

        values, action_log_probs, dist_entropy, _, bad_prob, pred_mask = self.actor_critic.evaluate_actions(
            rollouts.obs[:-1].view(-1, *obs_shape),
            rollouts.recurrent_hidden_states[0].view(-1, self.actor_critic.recurrent_hidden_state_size),
            rollouts.masks[:-1].view(-1, 1),
            rollouts.actions.view(-1, action_shape),
            rollouts.location_masks[:-1].view(-1, mask_size))

        values = values.view(num_steps, num_processes, 1)
        action_log_probs = action_log_probs.view(num_steps, num_processes, 1)

        advantages = rollouts.returns[:-1] - values
        value_loss = advantages.pow(2).mean()
        action_loss = -(advantages.detach() * action_log_probs).mean()

        mask_len = self.args.container_size[0]*self.args.container_size[1]
        mask_len = mask_len * (1+ self.args.enable_rotation)
        pred_mask = pred_mask.reshape((num_steps,num_processes,mask_len))

        mask_truth = rollouts.location_masks[0:num_steps] 
        graph_loss = self.loss_func(pred_mask, mask_truth).mean()
        dist_entropy = dist_entropy.mean()
        prob_loss = bad_prob.mean()

        if self.acktr and self.optimizer.steps % self.optimizer.Ts == 0:
            # Sampled fisher, see Martens 2014
            self.actor_critic.zero_grad()
            pg_fisher_loss = -action_log_probs.mean()

            value_noise = torch.randn(values.size())
            if values.is_cuda:
                value_noise = value_noise.cuda()

            sample_values = values + value_noise
            vf_fisher_loss = -(values - sample_values.detach()).pow(2).mean() # detach

            fisher_loss = pg_fisher_loss + vf_fisher_loss + graph_loss * 1e-8
            # fisher_loss = pg_fisher_loss + vf_fisher_loss
            self.optimizer.acc_stats = True
            fisher_loss.backward(retain_graph=True)
            self.optimizer.acc_stats = False

        force = 0.5 * 10
        self.optimizer.zero_grad()
        loss = value_loss * self.value_loss_coef
        loss += action_loss
        loss += prob_loss * self.invaild_coef
        loss -= dist_entropy * self.entropy_coef
        loss += force * graph_loss
        loss.backward()

        if self.acktr == False:
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)

        self.optimizer.step()

        # return value_loss.item(), action_loss.item(), dist_entropy.item(), prob_loss.item()
        return value_loss.item(), action_loss.item(), dist_entropy.item(), prob_loss.item(), graph_loss.item()

def check_nan(model,index):
    for p in model.parameters():
        if np.isnan(p.grad.data.mean().item()):
            print('index '+ str(index) +' happened an error!')