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

save_dir = './saved_models/' # directory to save agent logs (default: ./saved_models/)
load_dir = './pretrained_models/' # directory to load agent logs (default: ./pretrained_models/)
load_name = 'default_cut_2.pt' # default trained model for testing or continuing training
# data_dir = './dataset/' # the directory storing datasets
data_name = 'cut_2.pt' # the name of dataset, check 'data_dir' for details
seed = int(1) # random seed (default: 1)
