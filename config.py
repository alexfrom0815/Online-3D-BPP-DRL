'''
    this file contains many default parameters, 
    the parameters provided from command line will overwrite some of them
'''

save_dir = './saved_models/' # directory to save agent logs (default: ./saved_models/)
load_dir = './pretrained_models/' # directory to load agent logs (default: ./pretrained_models/)
load_name = 'default_cut_2.pt' # default trained model for testing or continuing training
# data_dir = './dataset/' # the directory storing datasets
data_name = 'cut_2.pt' # the name of dataset, check 'data_dir' for details
seed = int(1) # random seed (default: 1)
