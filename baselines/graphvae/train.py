
import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from random import shuffle

import data

def arg_parse():
    parser = argparse.ArgumentParser(description='GraphVAE arguments.')
    feature_parser.add_argument('--dataset', dest='dataset', 
            help='Input dataset.')

    parser.set_defaults(dataset='enzymes')
    return parser.parse_args()

def main():
    prog_args = arg_parse()

    print('CUDA', CUDA)
    ### running log

    if prog_args.dataset == 'enzymes':
        graphs= Graph_load_batch(min_num_nodes=10, name='ENZYMES')
    print(graphs)

if __name__ == '__main__':
    main()
