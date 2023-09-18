import sys 
sys.path.append("..")
import numpy as np
import gym
from acktr.model_loader import nnModel
from acktr.utils import check_box, get_possible_position
from acktr.arguments import get_args
from gym.envs.registration import register

def slipingWindow(plain, new_plain_size, stride=10):
    x_np = new_plain_size[0]
    y_np = new_plain_size[1]
    x_p = plain.shape[0]
    y_p = plain.shape[1]
    x_length = x_p - x_np + 1
    y_length = y_p - y_np + 1
    for i in range(0, x_length, stride):
        for j in range(0, y_length, stride):
            new_plain = plain[i : i + x_np, j:j + y_np]
            yield new_plain, i, j

def decode(env, obs):
    action_space = env.space.get_action_space()
    plain = np.reshape(obs[:action_space],newshape=env.space.plain_size[:2])
    box_size = (obs[1 * action_space], obs[2 * action_space], obs[3 * action_space])
    return plain, box_size

def get_action(env, obs, nmodel, past_rewards, evaluations):
    plain, box_size = decode(env, obs)
    new_plain_size = args.container_size
    psw = slipingWindow(plain, new_plain_size)
    bin_num = (plain.shape[0]*plain.shape[1])/(new_plain_size[0]*new_plain_size[1])

    max_adv = -1e8
    max_action = None
    max_label = None
    new_value = None
    # print('------------------')

    for new_plain, dx, dy in psw:
        aspace = new_plain_size[0] * new_plain_size[1]
        ppp = np.reshape(new_plain, newshape=(-1,))
        new_obs = np.zeros(shape=(400,))
        new_obs[0*aspace:1*aspace] = ppp
        new_obs[1*aspace:2*aspace] = box_size[0]
        new_obs[2*aspace:3*aspace] = box_size[1]
        new_obs[3*aspace:4*aspace] = box_size[2]

        value, poss  = nmodel.evaluate(new_obs, False)
        mask = get_possible_position(new_obs, args.container_size)
        poss = poss * mask
        if np.sum(mask) == len(mask):
            continue
        cur_max_action = np.argmax(poss)

        lx = dx + (cur_max_action // new_plain_size[0])
        ly = dy + (cur_max_action % new_plain_size[0])
        cur_max_action = env.space.position_to_index((lx, ly))

        label = (dx,dy)
        # print(value)
        if past_rewards.get(label):
            cur_adv = bin_num * past_rewards[label][-1] + (value - evaluations[label][-1])
            # cur_adv = 10 * np.sum(past_rewards[label]) + value
        else:
            cur_adv = -0.2
        # print(cur_adv)
        if cur_adv > max_adv:
            new_value = value
            max_adv = cur_adv
            max_label = label
            max_action = cur_max_action
    if max_action is None:
        return 0, max_adv, (0, 0)
    else:
        if not evaluations.get(max_label):
            past_rewards[max_label] = []
            evaluations[max_label] = []
        evaluations[max_label].append(new_value)
        return max_action, max_adv, max_label


def test(env, args):
    model_url = '../pretrained_models/default_cut_2.pt'
    nmodel = nnModel(model_url, args)
    obs = env.reset()
    past_rewards = dict()
    evaluations = dict()

    while True:
        action, mp, label = get_action(env, obs, nmodel, past_rewards, evaluations)
        obs, reward, done, info = env.step([action])
        # print(reward)
        past_rewards[label].append(reward)
        if done:
            break
    return env.space.get_ratio(),len(env.space.boxes)

def bin_size(size):
    container_size = (size, size, 10)
    data_url = None
    if size == 20:
        data_url = '../dataset/4bins_cut_2.pt'
    elif size == 30:
        data_url = '../dataset/9bins_cut_2.pt'
    elif size == 40:
        data_url = '../dataset/16bins_cut_2.pt'
    elif size == 50:
        data_url = '../dataset/15bins_cut_2.pt'
    return container_size, data_url

def registration_envs():
    register(
        id='Bpp-v0',                                  # Format should be xxx-v0, xxx-v1
        entry_point='envs.bpp0:PackingGame',   # Expalined in envs/__init__.py
    )


if __name__ == '__main__':
    registration_envs()
    times = 100
    args = get_args()
    container_size, data_url = bin_size(20)
    env = gym.make(args.env_name,
                   box_set=args.box_size_set,
                   container_size=container_size, test=True,
                   data_name=data_url, data_type=args.data_type)
    
    ratio = 0.0
    num = 0.0
    for i in range(times):
        if (i+1) % 10==0:
            print("Case", i+1)
        r,n = test(env, args)
        ratio += r
        num += n
    ratio /= times
    num /= times

    print('average ratio: ', ratio)
    print('average item number: ', num)



