import argparse
import time

def get_args():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument(
       '--mode', default='train', help='Test trained model or train new model, test | train'
    )
    parser.add_argument(
       '--env_name', default='Bpp-v0', type=str, help='bin packing environment name'
    )
    parser.add_argument(
       '--container_size', default=(10, 10, 10), type=int, help='container size along x, y and z axis'
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
        '--use-cuda', action='store_true', default=False, help='whether to use cuda'
    )
    parser.add_argument(
        '--tensorboard', action='store_true', default=False, help='whether use tensorboard to tracing trainning process'
    )
    parser.add_argument(
        '--preview', default=1, type=int, help='the item number agent knows (ignored when training)'
    )
    parser.add_argument(
        '--item-seq', default='cut1', help='item sequence generators (ignored when testing), cut1|cut2|rs'
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
    parser.add_argument(
        '--device', default=0, type=int,  help='device id (default: 0)'
    )
    parser.add_argument(
        '--save_interval', default=10, type=int,  help='save interval, one save per n updates (default: 100)'
    )
    parser.add_argument(
        '--log_interval', default=10, type=int,  help='log interval, one log per n updates (default: 10)'
    )
    parser.add_argument(
        '--save_model', action='store_true', default=False,  help='whether to save training model'
    )
    parser.add_argument(
        '--cases', default=100, type=int,  help='the number of sequences used for test (default 100)'
    )
    parser.add_argument(
        '--pretrain', action='store_true', default=False,  help='load whole model'
    )
    parser.add_argument(
        '--num_steps', default=5, type=int,  help='number of forward steps in A2C (default: 5)'
    )
    parser.add_argument(
        '--enable_rotation', action='store_true', default=False,  help='whether agent can rotate box'
    )
    parser.add_argument(
        '--data_name', default='cut_2.pt', help=' the name of dataset, check data_dir for details'
    )

    args = parser.parse_args()

    args.device = "cuda:" + str(args.device) if args.use_cuda else "cpu"
    args.bin_size = args.container_size
    args.pallet_size = args.container_size[0]
    args.channel = 4 # channels of CNN: 4 for hmap+next box, 5 for hmap nextbox+truemask
    args.data_type = args.item_seq

    box_range = args.item_size_range
    box_size_set = []
    for i in range(box_range[0], box_range[3] + 1):
        for j in range(box_range[1], box_range[4] + 1):
            for k in range(box_range[2], box_range[5] + 1):
                box_size_set.append((i, j, k))
    args.box_size_set = box_size_set

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
    if args.item_seq not in ['cut1', 'rs', 'cut2']:
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


 
