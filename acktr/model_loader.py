import numpy as np
import torch
import gym
import copy
from acktr.model import Policy
from acktr.utils import get_rotation_mask, get_possible_position


class nnModel(object):
    def __init__(self, url, args):
        area = args.container_size[0]*args.container_size[1]
        self.alen = area * (1+args.enable_rotation)
        self.olen = args.channel * area
        self.height = args.container_size[2]
        self.device = torch.device(args.device)
        self._model = self._load_model(url, args)


    def _load_model(self, url, args):
        model_pretrained, ob_rms = torch.load(url)
        observation_space = gym.spaces.Box(low=0.0, high=self.height, shape=(self.olen, ))
        action_space = gym.spaces.Discrete(self.alen)
        actor_critic = Policy(obs_shape=observation_space.shape, action_space=action_space, base_kwargs={'recurrent': False, 'hidden_size': args.hidden_size, 'args': args})
        # print(actor_critic)

        load_dict = {k.replace('module.', ''): v for k, v in model_pretrained.items()}
        load_dict = {k.replace('add_bias.', ''): v for k, v in load_dict.items()}
        load_dict = {k.replace('_bias', 'bias'): v for k, v in load_dict.items()}

        for k, v in load_dict.items():
            if len(v.size()) <= 3:
                load_dict[k] = v.squeeze(dim=-1)

        actor_critic.load_state_dict(load_dict)
        actor_critic = actor_critic.to(self.device)
        return actor_critic

    def evaluate(self, obs, use_mask=True):
        x = copy.deepcopy(obs)
        x = torch.FloatTensor(x).to(self.device)

        value, logits, _, pred= self._model.base(x, 0, 0)
        poss = self._model.dist.get_policy_distribution(logits)
        pred = self._model.binary(pred)
        # pred = get_rotation_mask(torch.tensor(obs), [10,10,10])
        # pred = np.array(get_possible_position(torch.tensor(obs), [10,10,10]))

        value = float(value)
        poss = poss.cpu().detach().numpy()
        pred = pred.cpu().detach().numpy()

        # np.set_printoptions(precision=3, suppress=True)
        # print('---------------------------')
        # print(pred1.reshape(10,10))
        # print(pred2.reshape(10,10))

        def softmax(x):
            probs = np.exp(x - np.max(x))
            probs /= np.sum(probs)
            return probs

        poss_in_actions = softmax(poss)
        if use_mask:
            poss_in_actions = poss_in_actions * pred
        poss_in_actions = np.reshape(poss_in_actions, newshape=(-1,))
        return value, poss_in_actions

    def sample_action(self, obs):
        x = copy.deepcopy(obs)
        x = torch.FloatTensor(x).to(self.device)

        value, logits, _, pred= self._model.base(x, 0, 0)
        poss = self._model.dist.get_policy_distribution(logits)
        pred = self._model.binary(pred)

        value = float(value)
        cat = torch.distributions.Categorical(logits=poss+pred*7)
        action = int(cat.sample())

        return value, action


