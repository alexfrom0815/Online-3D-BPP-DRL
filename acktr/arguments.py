import argparse
import time

def get_args():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument(
       '--mode', default='train', help='Test trained model or train new model, test | train'
    )
    parser.add_argument(
        '--enable-rotation', action='store_true', default=False, help='Whether agent can rotate boxes'
    )
    parser.add_argument(
        '--load-model', action='store_true', default=False,  help='Whether to use trained model'
    )
    parser.add_argument(
        '--load-name', default='default_cut_2.pt', 
        help='The name of trained model, default directory can be change in \'config.py\', you can put new trained model in it'
    )
    parser.add_argument(
        '--data-name', default='cut_2.pt',
        help='The name of testing dataset, default directory can be change in \'config.py\''
    )
    parser.add_argument(
        '--item-size-range', default=(2,2,2,5,5,5), type=tuple, help='the item size range, (min_width, min_length, min_height, max_width, max_length, max_height)'
    )
    parser.add_argument(
        '--bin-size', default=(10, 10, 10), type=tuple, help='the size of bin, (width, length, height)'
    )
    parser.add_argument(
        '--use-cuda', action='store_true', default=False, help='whether to use cuda'
    )
    parser.add_argument(
        '--tensorboard', action='store_true', default=False, help='whether use tensorboard to tracing trainning process'
    )
    parser.add_argument(
        '--preview', default=1, type=int, help='the item number agent knows (ignored when training)'
    )
    parser.add_argument(
        '--item-seq', default='depen', help='item sequence generators (ignored when testing), depen|sample|md'
    )
    parser.add_argument(
        '--algorithm', default='acktr', type=str,  help='algorithm used, acktr|ppo|a2c'
    )
    parser.add_argument(
        '--gamma', default=1.0, type=float,  help='discount factor for rewards (default: 1.0)'
    )
    parser.add_argument(
        '--entropy_coef', default=0.01, type=float,  help='entropy term coefficient (default: 0.01)'
    )
    parser.add_argument(
        '--value_loss_coef', default=0.5, type=float,  help='value loss coefficient (default: 0.5)'
    )
    parser.add_argument(
        '--invalid_coef', default=2, type=float,  help='invalid action possibility term coefficient'
    )
    parser.add_argument(
        '--hidden_size', default=256, type=int,  help='hidden layer cell number (default: 256)'
    )
    parser.add_argument(
        '--learning_rate', default=1e-6, type=float,  help='learning rate for a2c (default: 1e-6)'
    )
    parser.add_argument(
        '--eps', default=1e-5, type=float,  help='RMSprop optimizer epsilon (default: 1e-5)'
    )
    parser.add_argument(
        '--alpha', default=0.99, type=float,  help='RMSprop optimizer apha (default: 0.99)'
    )
    parser.add_argument(
        '--num_processes', default=16, type=int,  help='how many training CPU processes to use (default: 16)'
    )
    args = parser.parse_args()

    assert args.mode in ['train', 'test']
    if args.mode == 'train' and args.load_model:
        print('continue training model \"%s\"'%args.load_name)
    if args.mode == 'test' and args.load_model:
        print('test trained model \"%s\"'%args.load_name)
    if args.mode == 'train' and not args.load_model:
        print('train new model')
    if args.mode == 'test' and not args.load_model:
        raise Exception('no trained model chosed')
    if args.mode not in ['test', 'train']:
        raise Exception('Unknown option \'%s\''%(args.mode))
    if args.item_seq not in ['depen', 'md', 'sample']:
        raise Exception('Unsupported generator \'%s\''%(args.item_seq))
    print('the dataset used: ', args.data_name)
    time.sleep(0.5)
    print('the range of item size:  ', args.item_size_range)
    time.sleep(0.5)
    print('the size of bin:  ', args.bin_size)
    time.sleep(0.5)
    print('the number of known items:  ', args.preview)
    time.sleep(0.5)
    print('item sequence generator:  ', args.item_seq)
    time.sleep(0.5)
    print('enable_rotation: ', args.enable_rotation)
    print('use cuda:  ', args.use_cuda)
    time.sleep(0.5)
    # generate item size set
    item_set = []
    for i in range(args.item_size_range[0],args.item_size_range[3]+1):
        for j in range(args.item_size_range[1],args.item_size_range[4]+1):
            for k in range(args.item_size_range[2],args.item_size_range[5]+1):
                item_set.append((i,j,k))
    args.item_set = item_set
    print('item set: ', item_set)
    return args


 
