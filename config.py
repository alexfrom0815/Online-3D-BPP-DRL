'''
    this file contains many default parameters, 
    the parameters provided from command line will overwrite some of them
'''

# important model parameters
algo = 'acktr'  # algorithm to use: a2c | ppo | acktr
gamma = 1 # discount factor for rewards (default: 1)
num_processes = int(16) # how many training CPU processes to use (default: 16)
entropy_coef = 0.01 # entropy term coefficient (default: 0.01)
invalid_coef = 2 # invalid action possibility term coefficient (default: 0.1)
value_loss_coef = 0.5 # value loss coefficient (default: 0.5)
hidden_size = 256 # hidden layer cell number (default: 256)
pruning_threshold = 0.5 # pruning_threshold (default: 0.5)
preview = 1 # known item number (default: 1), 1 means 'not lookahead'

# default environment padrameters
env_name = 'Bpp-v0' # environment to train on
container_size = (10, 10, 10) # the size of bin(container)
box_range = (2, 2, 2, 5, 5, 5) # the item size range (x_min, y_min, z_min, x_max, y_max, z_max)
pallet_size = container_size[0]

box_size_set = [] 
for i in range(box_range[0],box_range[3]+1):
    for j in range(box_range[1],box_range[4]+1):
        for k in range(box_range[2],box_range[5]+1):
            box_size_set.append((i, j, k))

# other parameters of our environment
adjust = False # adjust agents' actions to touch corners or sides
adjust_ratio = 0  # adjust distance
input_format = 'cnn'  # the cnn vec
channel = 4 # channels of CNN: 4 for hmap+next box, 5 for hmap nextbox+truemask
data_type = 'depen'  # item sequence generators, depen|sample|md
give_up = False # whether agent can give up, now can only be False
enable_rotation = False # whether agent can rotate box

# saving and loading setting
cases = 100 # the number of sequences used for test (default 100)
save_model = True # whether to save training model
log_interval = int(10)  # log interval, one log per n updates (default: 10)'
save_interval = 10 # save interval, one save per n updates (default: 100)
eval_interval = None # eval interval, one eval per n updates (default: None)
save_dir = './saved_models/' # directory to save agent logs (default: ./saved_models/)
load_dir = './pretrained_models/' # directory to load agent logs (default: ./pretrained_models/)
load_name = 'default_cut_2.pt' # default trained model for testing or continuing training
data_dir = './dataset/' # the directory storing datasets
data_name = 'cut_2.pt' # the name of dataset, check 'data_dir' for details
tensorboard = True # whether use tensorboard to tracing trainning process
tbx_dir = './runs' # directory to save tensorboard logs (default: ./runs)
log_dir = './log' # directory to save agent logs (default: ./log)
image_folder = None # directory to save pictures (default: None)

pretrain = False # load whole model
load_model = False # load model parameters only


# CUDA setting
cuda_deterministic = False # sets flags for determinism when using CUDA (potentially slow!)
no_cuda = False # disable CUDA training
import torch
cuda = not no_cuda and torch.cuda.is_available() # whether to use cuda
device = 0 # which device to use (default: 0)

# other parameters
recurrent_policy = False # use a recurrent policy
use_linear_lr_decay = False # use a linear schedule on the learning rate

use_proper_time_limits = False # compute returns taking into account time limits
lr = 1e-6    # learning rate (default: 7e-4)
eps = 1e-5 # RMSprop optimizer epsilon (default: 1e-5)
alpha = 0.99 # RMSprop optimizer apha (default: 0.99)
max_grad_norm = 0.5 # max norm of gradients (default: 0.5)
seed = int(1) # random seed (default: 1)
num_steps = int(5) # number of forward steps in A2C (default: 5)
num_env_steps = 10e6 # number of environment steps to train (default: 10e6)

num_mini_batch = int(32) # number of batches for ppo (default: 32)
clip_param = float(0.2) # ppo clip parameter (default: 0.2)

