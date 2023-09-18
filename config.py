'''
    this file contains many default parameters, 
    the parameters provided from command line will overwrite some of them
'''

# default environment padrameters
container_size = (10, 10, 10) # the size of bin(container)
box_range = (2, 2, 2, 5, 5, 5) # the item size range (x_min, y_min, z_min, x_max, y_max, z_max)
pallet_size = container_size[0]

box_size_set = [] 
for i in range(box_range[0],box_range[3]+1):
    for j in range(box_range[1],box_range[4]+1):
        for k in range(box_range[2],box_range[5]+1):
            box_size_set.append((i, j, k))

# other parameters of our environment
channel = 4 # channels of CNN: 4 for hmap+next box, 5 for hmap nextbox+truemask
data_type = 'cut1'  # item sequence generators, cut1|rs|cut2
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
tbx_dir = './runs' # directory to save tensorboard logs (default: ./runs)
log_dir = './log' # directory to save agent logs (default: ./log)

pretrain = False # load whole model
load_model = False # load model parameters only


# CUDA setting
cuda_deterministic = False # sets flags for determinism when using CUDA (potentially slow!)
no_cuda = True # disable CUDA training
import torch
cuda = not no_cuda and torch.cuda.is_available() # whether to use cuda
device = 0 # which device to use (default: 0)


seed = int(1) # random seed (default: 1)
num_steps = int(5) # number of forward steps in A2C (default: 5)
num_env_steps = 10e6 # number of environment steps to train (default: 10e6)


