from time import clock
from acktr.model_loader import nnModel
from acktr.reorder import ReorderTree
import gym
import copy
from gym.envs.registration import register
from acktr.arguments import get_args

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

def unified_test(url,  args, pruning_threshold = 0.5):
    nmodel = nnModel(url, args)
    data_url = './dataset/' +args.data_name
    env = gym.make(args.env_name,
                    box_set=args.box_size_set,
                    container_size=args.container_size,
                    test=True, data_name=data_url,
                    enable_rotation=args.enable_rotation,
                    data_type=args.data_type)
    print('Env name: ', args.env_name)
    print('Data url: ', data_url)
    print('Model url: ', url)
    print('Case number: ', args.cases)
    print('pruning threshold: ', pruning_threshold)
    print('Known item number: ', args.preview)
    times = args.cases
    ratios = []
    avg_ratio, avg_counter, avg_time, avg_drate = 0.0, 0.0, 0.0, 0.0
    c_bound = pruning_threshold
    for i in range(times):
        if i % 10 == 0:
            print('case', i+1)
        env.reset()
        env.box_creator.preview(500)
        ratio, counter, time, depen_rate = run_sequence(nmodel, env, args.preview, c_bound)
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

def registration_envs():
    register(
        id='Bpp-v0',                                  # Format should be xxx-v0, xxx-v1
        entry_point='envs.bpp0:PackingGame',   # Expalined in envs/__init__.py
    )

if __name__ == '__main__':
    registration_envs()
    args = get_args()
    pruning_threshold = 0.5  # pruning_threshold (default: 0.5)
    unified_test('pretrained_models/default_cut_2.pt', args, pruning_threshold)
    # args.enable_rotation = True
    # unified_test('pretrained_models/rotation_cut_2.pt', args, pruning_threshold)

    