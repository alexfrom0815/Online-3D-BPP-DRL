from time import clock
from acktr.model_loader import nnModel
from acktr.reorder import ReorderTree
import gym
import copy
import config

def run_sequence(nmodel, raw_env, preview_num, c_bound):
    env = copy.deepcopy(raw_env)
    obs = env.cur_observation
    default_counter = 0
    box_counter = 0
    start = clock()
    while True:
        box_list = env.box_creator.preview(preview_num)
        # print(box_list)
        tree = ReorderTree(nmodel, box_list, env, times=100)
        act, val, default = tree.reorder_search()
        obs, _, done, info = env.step([act])
        if done:
            end = clock()
            print('Time cost:', end-start)
            print('Ratio:', info['ratio'])
            return info['ratio'], info['counter'], end-start,default_counter/box_counter
        box_counter += 1
        default_counter += int(default)

def unified_test(url, config):
    nmodel = nnModel(url, config)
    data_url = config.data_dir+config.data_name
    env = gym.make(config.env_name, _adjust_ratio=0, adjust=False,
                    box_set=config.box_size_set,
                    container_size=config.container_size,
                    test=True, data_name=data_url,
                    enable_rotation=config.enable_rotation,
                    data_type=config.data_type)
    print('Env name: ', config.env_name)
    print('Data url: ', data_url)
    print('Model url: ', url)
    print('Case number: ', config.cases)
    print('pruning threshold: ', config.pruning_threshold)
    print('Known item number: ', config.preview)
    times = config.cases
    ratios = []
    avg_ratio, avg_counter, avg_time, avg_drate = 0.0, 0.0, 0.0, 0.0
    c_bound = config.pruning_threshold
    for i in range(times):
        if i % 10 == 0:
            print('case', i+1)
        env.reset()
        env.box_creator.preview(500)
        ratio, counter, time, depen_rate = run_sequence(nmodel, env, config.preview, c_bound)
        avg_ratio += ratio
        ratios.append(ratio)
        avg_counter += counter
        avg_time += time
        avg_drate += depen_rate

    print()
    print('All cases have been done!')
    print('----------------------------------------------')
    print('average space utilization: %.4f'%(avg_ratio/times))
    print('average put item number: %.4f'%(avg_counter/times))
    print('average sequence time: %.4f'%(avg_time/times))
    print('average time per item: %.4f'%(avg_time/avg_counter))
    print('----------------------------------------------')

if __name__ == '__main__':
    config.cases = 100
    config.preview = 1
    unified_test('pretrained_models/default_cut_2.pt', config)
    # config.enable_rotation = True
    # unified_test('pretrained_models/rotation_cut_2.pt', config)

    