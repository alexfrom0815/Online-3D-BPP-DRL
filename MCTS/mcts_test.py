import gym
import copy
from monteCarlo import MCTree
import numpy as np
import time

import sys 
sys.path.append("..")
import config
from acktr.arguments import get_args
from acktr.model_loader import nnModel


def test(box_size_list, env, obser, simulation_times, search_depth, rollout_length, nmodel, args):
    sim_env = copy.deepcopy(env)
    size_idx = len(box_size_list)
    action_list = []
    box_size_distribution = dict()
    box_num = 0
    sum_reward = 0
    print("length: ", size_idx)


    mctree = MCTree(sim_env, obser, box_size_list, nmodel = nmodel, search_depth=search_depth, rollout_length=rollout_length)
    while True:
        # show some information
        print(box_size_list[:10])
        # print(sim_env.space.plain)
        # MCTS simulation
        pl = mctree.get_policy(simulation_times, zeta=1e-5)
        action = mctree.sample_action(pl)
        
        assert sim_env.next_box == box_size_list[0]
        obser, r, done, dt = sim_env.step([action])
        sum_reward += r
        if done:
            dt['reward'] = sum_reward
            # print('---------------------')
            # print(dt)
            # print('---------------------')
            # print(action_list)
            # print('---------------------')
            # print(sim_env.space.plain)
            # print('---------------------')
            for (key, value) in box_size_distribution.items():
                box_size_distribution[key] = value / box_num
            # print(box_size_distribution)
            # print('---------------------')
            return [dt['ratio'], dt['counter'], dt['reward']]

        # fetch new box    
        assert size_idx <= len(env.box_creator.box_list)
        next_box = copy.deepcopy(env.box_creator.box_list[size_idx])
        size_idx += 1
        # update dis
        # tribution
        box_num += 1
        new_put_box = tuple(box_size_list[0])
        if new_put_box not in box_size_distribution:
            box_size_distribution[new_put_box] = 0
        box_size_distribution[new_put_box] += 1
        # update action
        action_list.append(action)
        # to next node
        mctree.succeed(action, next_box, obser)

def compare_test(env, args_list, times=5 ,args=None):
    result = dict()
    case_num = len(args_list)
    print("Case number: %d"%times)
    nmodel = nnModel('../pretrained_models/default_cut_2.pt',  args)
    for i in range(times):
        print('Case %d' % (i+1))
        obser = env.reset()
        next_box_size_list = env.box_creator.preview(1000)
        # print(len(next_box_size_list))
        for j in range(case_num):
            if j not in result:
                result[j] = []
            arg = args_list[j]
            print(arg)
            start = time.time()
            ratio, counter, reward = test(next_box_size_list[:4], env, obser, *arg, nmodel, args)
            end = time.time()
            result[j].append([ratio, counter, reward, end-start])
        print('//////////////////////////////////////////////////')
    for (key, value) in result.items():
        result[key] = np.array(value)
    return result


if __name__ == '__main__':
    args = get_args()
    env = gym.make(args.env_name, box_set=args.box_size_set,
                   container_size=args.container_size, test=True,
                   data_name="../dataset/cut_2.pt", data_type=args.data_type)

    args_list = list()
    args_list.append([100, None, -1])
    result = compare_test(env, args_list, 100, args)
    for (key, value) in result.items():
        print(value[:, 0])
        print(value[:, 1])
        meanv = value.mean(axis=-2)
        print(meanv)
        print("avg_time_per_item", meanv[-1]/meanv[1])
        # print(value.var(axis=-2))



