import sys 
sys.path.append("..")
from acktr.model_loader import nnModel
import config
nmodel = nnModel('../pretrained_models/default_cut_2.pt', config)