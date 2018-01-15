
import argparse
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
from random import shuffle
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR

import data

CUDA = 2

def train(args):
    epoch = 1
    optimizer = optim.Adam(list(output.parameters()), lr=args.lr)

def arg_parse():
    parser = argparse.ArgumentParser(description='GraphVAE arguments.')
    io_parser = parser.add_mutually_exclusive_group(required=False)
    io_parser.add_argument('--dataset', dest='dataset', 
            help='Input dataset.')

    parser.add_argument('--lr', dest='lr',
            help='Learning rate.')

    parser.set_defaults(dataset='enzymes',
                        lr=0.001)
    return parser.parse_args()

def main():
    prog_args = arg_parse()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(CUDA)
    print('CUDA', CUDA)
    ### running log

    if prog_args.dataset == 'enzymes':
        graphs= data.Graph_load_batch(min_num_nodes=10, name='ENZYMES')

    graphs_len = len(graphs)
    graphs_test = graphs[int(0.8 * graphs_len):]
    graphs_train = graphs[0:int(0.8*graphs_len)]
    max_num_node = max([graphs[i].number_of_nodes() for i in range(len(graphs))])

    print('total graph num: {}, training set: {}'.format(len(graphs),len(graphs_train)))
    print('max number node: {}'.format(args.max_num_node))

    dataset = Graph_sequence_sampler_pytorch(graphs_train,de=args.max_prev_node,max_num_node=args.max_num_node)

    print(graphs)

if __name__ == '__main__':
    main()
