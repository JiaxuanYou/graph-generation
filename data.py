import torch
import torchvision as tv
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
from random import shuffle

import networkx as nx
import pickle as pkl
import scipy.sparse as sp
import logging

import random
import shutil
import os
import time
from model import *
from utils import *




# load ENZYMES and PROTEIN and DD dataset
def Graph_load_batch(min_num_nodes = 20, max_num_nodes = 1000, name = 'ENZYMES',node_attributes = True,graph_labels=True):
    '''
    load many graphs, e.g. enzymes
    :return: a list of graphs
    '''
    print('Loading graph dataset: '+str(name))
    G = nx.Graph()
    # load data
    path = 'dataset/'+name+'/'
    data_adj = np.loadtxt(path+name+'_A.txt', delimiter=',').astype(int)
    if node_attributes:
        data_node_att = np.loadtxt(path+name+'_node_attributes.txt', delimiter=',')
    data_node_label = np.loadtxt(path+name+'_node_labels.txt', delimiter=',').astype(int)
    data_graph_indicator = np.loadtxt(path+name+'_graph_indicator.txt', delimiter=',').astype(int)
    if graph_labels:
        data_graph_labels = np.loadtxt(path+name+'_graph_labels.txt', delimiter=',').astype(int)


    data_tuple = list(map(tuple, data_adj))
    # print(len(data_tuple))
    # print(data_tuple[0])

    # add edges
    G.add_edges_from(data_tuple)
    # add node attributes
    for i in range(data_node_label.shape[0]):
        if node_attributes:
            G.add_node(i+1, feature = data_node_att[i])
        G.add_node(i+1, label = data_node_label[i])
    G.remove_nodes_from(list(nx.isolates(G)))

    # print(G.number_of_nodes())
    # print(G.number_of_edges())

    # split into graphs
    graph_num = data_graph_indicator.max()
    node_list = np.arange(data_graph_indicator.shape[0])+1
    graphs = []
    max_nodes = 0
    for i in range(graph_num):
        # find the nodes for each graph
        nodes = node_list[data_graph_indicator==i+1]
        G_sub = G.subgraph(nodes)
        if graph_labels:
            G_sub.graph['label'] = data_graph_labels[i]
        # print('nodes', G_sub.number_of_nodes())
        # print('edges', G_sub.number_of_edges())
        # print('label', G_sub.graph)
        if G_sub.number_of_nodes()>=min_num_nodes and G_sub.number_of_nodes()<=max_num_nodes:
            graphs.append(G_sub)
            if G_sub.number_of_nodes() > max_nodes:
                max_nodes = G_sub.number_of_nodes()
            # print(G_sub.number_of_nodes(), 'i', i)
    # print('Graph dataset name: {}, total graph num: {}'.format(name, len(graphs)))
    # logging.warning('Graphs loaded, total num: {}'.format(len(graphs)))
    print('Loaded')
    return graphs

def test_graph_load_DD():
    graphs, max_num_nodes = Graph_load_batch(min_num_nodes=10,name='DD',node_attributes=False,graph_labels=True)
    shuffle(graphs)
    plt.switch_backend('agg')
    plt.hist([len(graphs[i]) for i in range(len(graphs))], bins=100)
    plt.savefig('figures/test.png')
    plt.close()
    row = 4
    col = 4
    draw_graph_list(graphs[0:row*col], row=row,col=col, fname='figures/test')
    print('max num nodes',max_num_nodes)


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

# load cora, citeseer and pubmed dataset
def Graph_load(dataset = 'cora'):
    '''
    Load a single graph dataset
    :param dataset: dataset name
    :return:
    '''
    names = ['x', 'tx', 'allx', 'graph']
    objects = []
    for i in range(len(names)):
        load = pkl.load(open("dataset/ind.{}.{}".format(dataset, names[i]), 'rb'), encoding='latin1')
        # print('loaded')
        objects.append(load)
        # print(load)
    x, tx, allx, graph = tuple(objects)
    test_idx_reorder = parse_index_file("dataset/ind.{}.test.index".format(dataset))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    G = nx.from_dict_of_lists(graph)
    adj = nx.adjacency_matrix(G)
    return adj, features, G


######### code test ########
# adj, features,G = Graph_load()
# print(adj)
# print(G.number_of_nodes(), G.number_of_edges())

# _,_,G = Graph_load(dataset='citeseer')
# G = max(nx.connected_component_subgraphs(G), key=len)
# G = nx.convert_node_labels_to_integers(G)
#
# count = 0
# max_node = 0
# for i in range(G.number_of_nodes()):
#     G_ego = nx.ego_graph(G, i, radius=3)
#     # draw_graph(G_ego,prefix='test'+str(i))
#     m = G_ego.number_of_nodes()
#     if m>max_node:
#         max_node = m
#     if m>=50:
#         print(i, G_ego.number_of_nodes(), G_ego.number_of_edges())
#         count += 1
# print('count', count)
# print('max_node', max_node)




def bfs_seq(G, start_id):
    '''
    get a bfs node sequence
    :param G:
    :param start_id:
    :return:
    '''
    dictionary = dict(nx.bfs_successors(G, start_id))
    start = [start_id]
    output = [start_id]
    while len(start) > 0:
        next = []
        while len(start) > 0:
            current = start.pop(0)
            neighbor = dictionary.get(current)
            if neighbor is not None:
                #### a wrong example, should not permute here!
                # shuffle(neighbor)
                next = next + neighbor
        output = output + next
        start = next
    return output



def encode_adj(adj, max_prev_node=10, is_full = False):
    '''

    :param adj: n*n, rows means time step, while columns are input dimension
    :param max_degree: we want to keep row number, but truncate column numbers
    :return:
    '''
    if is_full:
        max_prev_node = adj.shape[0]-1

    # pick up lower tri
    adj = np.tril(adj, k=-1)
    n = adj.shape[0]
    adj = adj[1:n, 0:n-1]

    # use max_prev_node to truncate
    # note: now adj is a (n-1)*(n-1) matrix
    adj_output = np.zeros((adj.shape[0], max_prev_node))
    for i in range(adj.shape[0]):
        input_start = max(0, i - max_prev_node + 1)
        input_end = i + 1
        output_start = max_prev_node + input_start - input_end
        output_end = max_prev_node
        adj_output[i, output_start:output_end] = adj[i, input_start:input_end]
        adj_output[i,:] = adj_output[i,:][::-1] # reverse order

    return adj_output

def decode_adj(adj_output):
    '''
        recover to adj from adj_output
        note: here adj_output have shape (n-1)*m
    '''
    max_prev_node = adj_output.shape[1]
    adj = np.zeros((adj_output.shape[0], adj_output.shape[0]))
    for i in range(adj_output.shape[0]):
        input_start = max(0, i - max_prev_node + 1)
        input_end = i + 1
        output_start = max_prev_node + max(0, i - max_prev_node + 1) - (i + 1)
        output_end = max_prev_node
        adj[i, input_start:input_end] = adj_output[i,::-1][output_start:output_end] # reverse order
    adj_full = np.zeros((adj_output.shape[0]+1, adj_output.shape[0]+1))
    n = adj_full.shape[0]
    adj_full[1:n, 0:n-1] = np.tril(adj, 0)
    adj_full = adj_full + adj_full.T

    return adj_full


def encode_adj_flexible(adj):
    '''
    return a flexible length of output
    note that here there is no loss when encoding/decoding an adj matrix
    :param adj: adj matrix
    :return:
    '''
    # pick up lower tri
    adj = np.tril(adj, k=-1)
    n = adj.shape[0]
    adj = adj[1:n, 0:n-1]

    adj_output = []
    input_start = 0
    for i in range(adj.shape[0]):
        input_end = i + 1
        adj_slice = adj[i, input_start:input_end]
        adj_output.append(adj_slice)
        non_zero = np.nonzero(adj_slice)[0]
        input_start = input_end-len(adj_slice)+np.amin(non_zero)

    return adj_output



def decode_adj_flexible(adj_output):
    '''
    return a flexible length of output
    note that here there is no loss when encoding/decoding an adj matrix
    :param adj: adj matrix
    :return:
    '''
    adj = np.zeros((len(adj_output), len(adj_output)))
    for i in range(len(adj_output)):
        output_start = i+1-len(adj_output[i])
        output_end = i+1
        adj[i, output_start:output_end] = adj_output[i]
    adj_full = np.zeros((len(adj_output)+1, len(adj_output)+1))
    n = adj_full.shape[0]
    adj_full[1:n, 0:n-1] = np.tril(adj, 0)
    adj_full = adj_full + adj_full.T

    return adj_full

def test_encode_decode_adj():
######## code test ###########
    G = nx.ladder_graph(5)
    G = nx.grid_2d_graph(20,20)
    G = nx.ladder_graph(200)
    G = nx.karate_club_graph()
    G = nx.connected_caveman_graph(2,3)
    print(G.number_of_nodes())
    
    adj = np.asarray(nx.to_numpy_matrix(G))
    G = nx.from_numpy_matrix(adj)
    #
    start_idx = np.random.randint(adj.shape[0])
    x_idx = np.array(bfs_seq(G, start_idx))
    adj = adj[np.ix_(x_idx, x_idx)]
    
    print('adj\n',adj)
    adj_output = encode_adj(adj,max_prev_node=5)
    print('adj_output\n',adj_output)
    adj_recover = decode_adj(adj_output,max_prev_node=5)
    print('adj_recover\n',adj_recover)
    print('error\n',np.amin(adj_recover-adj),np.amax(adj_recover-adj))
    
    
    adj_output = encode_adj_flexible(adj)
    for i in range(len(adj_output)):
        print(len(adj_output[i]))
    adj_recover = decode_adj_flexible(adj_output)
    print(adj_recover)
    print(np.amin(adj_recover-adj),np.amax(adj_recover-adj))



def encode_adj_full(adj):
    '''
    return a n-1*n-1*2 tensor, the first dimension is an adj matrix, the second show if each entry is valid
    :param adj: adj matrix
    :return:
    '''
    # pick up lower tri
    adj = np.tril(adj, k=-1)
    n = adj.shape[0]
    adj = adj[1:n, 0:n-1]
    adj_output = np.zeros((adj.shape[0],adj.shape[1],2))
    adj_len = np.zeros(adj.shape[0])

    for i in range(adj.shape[0]):
        non_zero = np.nonzero(adj[i,:])[0]
        input_start = np.amin(non_zero)
        input_end = i + 1
        adj_slice = adj[i, input_start:input_end]
        # write adj
        adj_output[i,0:adj_slice.shape[0],0] = adj_slice[::-1] # put in reverse order
        # write stop token (if token is 0, stop)
        adj_output[i,0:adj_slice.shape[0],1] = 1 # put in reverse order
        # write sequence length
        adj_len[i] = adj_slice.shape[0]

    return adj_output,adj_len

def decode_adj_full(adj_output):
    '''
    return an adj according to adj_output
    :param
    :return:
    '''
    # pick up lower tri
    adj = np.zeros((adj_output.shape[0]+1,adj_output.shape[1]+1))

    for i in range(adj_output.shape[0]):
        non_zero = np.nonzero(adj_output[i,:,1])[0] # get valid sequence
        input_end = np.amax(non_zero)
        adj_slice = adj_output[i, 0:input_end+1, 0] # get adj slice
        # write adj
        output_end = i+1
        output_start = i+1-input_end-1
        adj[i+1,output_start:output_end] = adj_slice[::-1] # put in reverse order
    adj = adj + adj.T
    return adj

def test_encode_decode_adj_full():
########### code test #############
    # G = nx.ladder_graph(10)
    G = nx.karate_club_graph()
    # get bfs adj
    adj = np.asarray(nx.to_numpy_matrix(G))
    G = nx.from_numpy_matrix(adj)
    start_idx = np.random.randint(adj.shape[0])
    x_idx = np.array(bfs_seq(G, start_idx))
    adj = adj[np.ix_(x_idx, x_idx)]
    
    adj_output, adj_len = encode_adj_full(adj)
    print('adj\n',adj)
    print('adj_output[0]\n',adj_output[:,:,0])
    print('adj_output[1]\n',adj_output[:,:,1])
    # print('adj_len\n',adj_len)
    
    adj_recover = decode_adj_full(adj_output)
    print('adj_recover\n', adj_recover)
    print('error\n',adj_recover-adj)
    print('error_sum\n',np.amax(adj_recover-adj), np.amin(adj_recover-adj))






########## use pytorch dataloader
class Graph_sequence_sampler_pytorch(torch.utils.data.Dataset):
    def __init__(self, G_list, max_num_node=None, max_prev_node=None, iteration=20000):
        self.adj_all = []
        self.len_all = []
        for G in G_list:
            self.adj_all.append(np.asarray(nx.to_numpy_matrix(G)))
            self.len_all.append(G.number_of_nodes())
        if max_num_node is None:
            self.n = max(self.len_all)
        else:
            self.n = max_num_node
        if max_prev_node is None:
            print('calculating max previous node, total iteration: {}'.format(iteration))
            self.max_prev_node = max(self.calc_max_prev_node(iter=iteration))
            print('max previous node: {}'.format(self.max_prev_node))
        else:
            self.max_prev_node = max_prev_node

        # self.max_prev_node = max_prev_node

        # # sort Graph in descending order
        # len_batch_order = np.argsort(np.array(self.len_all))[::-1]
        # self.len_all = [self.len_all[i] for i in len_batch_order]
        # self.adj_all = [self.adj_all[i] for i in len_batch_order]
    def __len__(self):
        return len(self.adj_all)
    def __getitem__(self, idx):
        adj_copy = self.adj_all[idx].copy()
        x_batch = np.zeros((self.n, self.max_prev_node))  # here zeros are padded for small graph
        x_batch[0,:] = 1 # the first input token is all ones
        y_batch = np.zeros((self.n, self.max_prev_node))  # here zeros are padded for small graph
        # generate input x, y pairs
        len_batch = adj_copy.shape[0]
        x_idx = np.random.permutation(adj_copy.shape[0])
        adj_copy = adj_copy[np.ix_(x_idx, x_idx)]
        adj_copy_matrix = np.asmatrix(adj_copy)
        G = nx.from_numpy_matrix(adj_copy_matrix)
        # then do bfs in the permuted G
        start_idx = np.random.randint(adj_copy.shape[0])
        x_idx = np.array(bfs_seq(G, start_idx))
        adj_copy = adj_copy[np.ix_(x_idx, x_idx)]
        adj_encoded = encode_adj(adj_copy.copy(), max_prev_node=self.max_prev_node)
        # get x and y and adj
        # for small graph the rest are zero padded
        y_batch[0:adj_encoded.shape[0], :] = adj_encoded
        x_batch[1:adj_encoded.shape[0] + 1, :] = adj_encoded
        return {'x':x_batch,'y':y_batch, 'len':len_batch}

    def calc_max_prev_node(self, iter=20000,topk=10):
        max_prev_node = []
        for i in range(iter):
            if i % (iter / 5) == 0:
                print('iter {} times'.format(i))
            adj_idx = np.random.randint(len(self.adj_all))
            adj_copy = self.adj_all[adj_idx].copy()
            # print('Graph size', adj_copy.shape[0])
            x_idx = np.random.permutation(adj_copy.shape[0])
            adj_copy = adj_copy[np.ix_(x_idx, x_idx)]
            adj_copy_matrix = np.asmatrix(adj_copy)
            G = nx.from_numpy_matrix(adj_copy_matrix)
            # then do bfs in the permuted G
            start_idx = np.random.randint(adj_copy.shape[0])
            x_idx = np.array(bfs_seq(G, start_idx))
            adj_copy = adj_copy[np.ix_(x_idx, x_idx)]
            # encode adj
            adj_encoded = encode_adj_flexible(adj_copy.copy())
            max_encoded_len = max([len(adj_encoded[i]) for i in range(len(adj_encoded))])
            max_prev_node.append(max_encoded_len)
        max_prev_node = sorted(max_prev_node)[-1*topk:]
        return max_prev_node



########## use pytorch dataloader
class Graph_sequence_sampler_pytorch_nobfs(torch.utils.data.Dataset):
    def __init__(self, G_list, max_num_node=None):
        self.adj_all = []
        self.len_all = []
        for G in G_list:
            self.adj_all.append(np.asarray(nx.to_numpy_matrix(G)))
            self.len_all.append(G.number_of_nodes())
        if max_num_node is None:
            self.n = max(self.len_all)
        else:
            self.n = max_num_node
    def __len__(self):
        return len(self.adj_all)
    def __getitem__(self, idx):
        adj_copy = self.adj_all[idx].copy()
        x_batch = np.zeros((self.n, self.n-1))  # here zeros are padded for small graph
        x_batch[0,:] = 1 # the first input token is all ones
        y_batch = np.zeros((self.n, self.n-1))  # here zeros are padded for small graph
        # generate input x, y pairs
        len_batch = adj_copy.shape[0]
        x_idx = np.random.permutation(adj_copy.shape[0])
        adj_copy = adj_copy[np.ix_(x_idx, x_idx)]
        adj_encoded = encode_adj(adj_copy.copy(), max_prev_node=self.n-1)
        # get x and y and adj
        # for small graph the rest are zero padded
        y_batch[0:adj_encoded.shape[0], :] = adj_encoded
        x_batch[1:adj_encoded.shape[0] + 1, :] = adj_encoded
        return {'x':x_batch,'y':y_batch, 'len':len_batch}

# dataset = Graph_sequence_sampler_pytorch_nobfs(graphs)
# print(dataset[1]['x'])
# print(dataset[1]['y'])
# print(dataset[1]['len'])







########## use pytorch dataloader
class Graph_sequence_sampler_pytorch_canonical(torch.utils.data.Dataset):
    def __init__(self, G_list, max_num_node=None, max_prev_node=None, iteration=20000):
        self.adj_all = []
        self.len_all = []
        for G in G_list:
            self.adj_all.append(np.asarray(nx.to_numpy_matrix(G)))
            self.len_all.append(G.number_of_nodes())
        if max_num_node is None:
            self.n = max(self.len_all)
        else:
            self.n = max_num_node
        if max_prev_node is None:
            # print('calculating max previous node, total iteration: {}'.format(iteration))
            # self.max_prev_node = max(self.calc_max_prev_node(iter=iteration))
            # print('max previous node: {}'.format(self.max_prev_node))
            self.max_prev_node = self.n-1
        else:
            self.max_prev_node = max_prev_node

        # self.max_prev_node = max_prev_node

        # # sort Graph in descending order
        # len_batch_order = np.argsort(np.array(self.len_all))[::-1]
        # self.len_all = [self.len_all[i] for i in len_batch_order]
        # self.adj_all = [self.adj_all[i] for i in len_batch_order]
    def __len__(self):
        return len(self.adj_all)
    def __getitem__(self, idx):
        adj_copy = self.adj_all[idx].copy()
        x_batch = np.zeros((self.n, self.max_prev_node))  # here zeros are padded for small graph
        x_batch[0,:] = 1 # the first input token is all ones
        y_batch = np.zeros((self.n, self.max_prev_node))  # here zeros are padded for small graph
        # generate input x, y pairs
        len_batch = adj_copy.shape[0]
        # adj_copy_matrix = np.asmatrix(adj_copy)
        # G = nx.from_numpy_matrix(adj_copy_matrix)
        # then do bfs in the permuted G
        # start_idx = G.number_of_nodes()-1
        # x_idx = np.array(bfs_seq(G, start_idx))
        # adj_copy = adj_copy[np.ix_(x_idx, x_idx)]
        adj_encoded = encode_adj(adj_copy, max_prev_node=self.max_prev_node)
        # get x and y and adj
        # for small graph the rest are zero padded
        y_batch[0:adj_encoded.shape[0], :] = adj_encoded
        x_batch[1:adj_encoded.shape[0] + 1, :] = adj_encoded
        return {'x':x_batch,'y':y_batch, 'len':len_batch}

    def calc_max_prev_node(self, iter=20000,topk=10):
        max_prev_node = []
        for i in range(iter):
            if i % (iter / 5) == 0:
                print('iter {} times'.format(i))
            adj_idx = np.random.randint(len(self.adj_all))
            adj_copy = self.adj_all[adj_idx].copy()
            # print('Graph size', adj_copy.shape[0])
            x_idx = np.random.permutation(adj_copy.shape[0])
            adj_copy = adj_copy[np.ix_(x_idx, x_idx)]
            adj_copy_matrix = np.asmatrix(adj_copy)
            G = nx.from_numpy_matrix(adj_copy_matrix)
            # then do bfs in the permuted G
            start_idx = np.random.randint(adj_copy.shape[0])
            x_idx = np.array(bfs_seq(G, start_idx))
            adj_copy = adj_copy[np.ix_(x_idx, x_idx)]
            # encode adj
            adj_encoded = encode_adj_flexible(adj_copy.copy())
            max_encoded_len = max([len(adj_encoded[i]) for i in range(len(adj_encoded))])
            max_prev_node.append(max_encoded_len)
        max_prev_node = sorted(max_prev_node)[-1*topk:]
        return max_prev_node



########## use pytorch dataloader
class Graph_sequence_sampler_pytorch_nll(torch.utils.data.Dataset):
    def __init__(self, G_list, max_num_node=None, max_prev_node=None, iteration=20000):
        self.adj_all = []
        self.len_all = []
        for G in G_list:
            adj = np.asarray(nx.to_numpy_matrix(G))
            adj_temp = self.calc_adj(adj)
            self.adj_all.extend(adj_temp)
            self.len_all.append(G.number_of_nodes())
        if max_num_node is None:
            self.n = max(self.len_all)
        else:
            self.n = max_num_node
        if max_prev_node is None:
            # print('calculating max previous node, total iteration: {}'.format(iteration))
            # self.max_prev_node = max(self.calc_max_prev_node(iter=iteration))
            # print('max previous node: {}'.format(self.max_prev_node))
            self.max_prev_node = self.n-1
        else:
            self.max_prev_node = max_prev_node

        # self.max_prev_node = max_prev_node

        # # sort Graph in descending order
        # len_batch_order = np.argsort(np.array(self.len_all))[::-1]
        # self.len_all = [self.len_all[i] for i in len_batch_order]
        # self.adj_all = [self.adj_all[i] for i in len_batch_order]
    def __len__(self):
        return len(self.adj_all)
    def __getitem__(self, idx):
        adj_copy = self.adj_all[idx].copy()
        x_batch = np.zeros((self.n, self.max_prev_node))  # here zeros are padded for small graph
        x_batch[0,:] = 1 # the first input token is all ones
        y_batch = np.zeros((self.n, self.max_prev_node))  # here zeros are padded for small graph
        # generate input x, y pairs
        len_batch = adj_copy.shape[0]
        # adj_copy_matrix = np.asmatrix(adj_copy)
        # G = nx.from_numpy_matrix(adj_copy_matrix)
        # then do bfs in the permuted G
        # start_idx = G.number_of_nodes()-1
        # x_idx = np.array(bfs_seq(G, start_idx))
        # adj_copy = adj_copy[np.ix_(x_idx, x_idx)]
        adj_encoded = encode_adj(adj_copy, max_prev_node=self.max_prev_node)
        # get x and y and adj
        # for small graph the rest are zero padded
        y_batch[0:adj_encoded.shape[0], :] = adj_encoded
        x_batch[1:adj_encoded.shape[0] + 1, :] = adj_encoded
        return {'x':x_batch,'y':y_batch, 'len':len_batch}

    def calc_adj(self,adj):
        max_iter = 10000
        adj_all = [adj]
        adj_all_len = 1
        i_old = 0
        for i in range(max_iter):
            adj_copy = adj.copy()
            x_idx = np.random.permutation(adj_copy.shape[0])
            adj_copy = adj_copy[np.ix_(x_idx, x_idx)]
            adj_copy_matrix = np.asmatrix(adj_copy)
            G = nx.from_numpy_matrix(adj_copy_matrix)
            # then do bfs in the permuted G
            start_idx = np.random.randint(adj_copy.shape[0])
            x_idx = np.array(bfs_seq(G, start_idx))
            adj_copy = adj_copy[np.ix_(x_idx, x_idx)]
            add_flag = True
            for adj_exist in adj_all:
                if np.array_equal(adj_exist, adj_copy):
                    add_flag = False
                    break
            if add_flag:
                adj_all.append(adj_copy)
                adj_all_len += 1
            if adj_all_len % 10 ==0:
                print('adj found:',adj_all_len,'iter used',i)
        return adj_all



# graphs = [nx.barabasi_albert_graph(20,3)]
# graphs = [nx.grid_2d_graph(4,4)]
# dataset = Graph_sequence_sampler_pytorch_nll(graphs)











############## below are codes not used in current version
############## they are based on pytorch default data loader, we should consider reimplement them in current datasets, since they are more efficient


# normal version
class Graph_sequence_sampler_truncate():
    '''
    the output will truncate according to the max_prev_node
    '''
    def __init__(self, G_list, max_node_num=25, batch_size=4, max_prev_node = 25):
        self.batch_size = batch_size
        self.n = max_node_num
        self.max_prev_node = max_prev_node

        self.adj_all = []
        for G in G_list:
            self.adj_all.append(np.asarray(nx.to_numpy_matrix(G)))

    def sample(self):
        # batch, length, feature
        x_batch = np.zeros((self.batch_size, self.n, self.max_prev_node)) # here zeros are padded for small graph
        y_batch = np.zeros((self.batch_size, self.n, self.max_prev_node)) # here zeros are padded for small graph
        len_batch = np.zeros(self.batch_size)
        # generate input x, y pairs
        for i in range(self.batch_size):
            # first sample and get a permuted adj
            adj_idx = np.random.randint(len(self.adj_all))
            adj_copy = self.adj_all[adj_idx].copy()
            len_batch[i] = adj_copy.shape[0]
            x_idx = np.random.permutation(adj_copy.shape[0])
            adj_copy = adj_copy[np.ix_(x_idx, x_idx)]
            adj_copy_matrix = np.asmatrix(adj_copy)
            G = nx.from_numpy_matrix(adj_copy_matrix)
            # then do bfs in the permuted G
            start_idx = np.random.randint(adj_copy.shape[0])
            x_idx = np.array(bfs_seq(G, start_idx))
            adj_copy = adj_copy[np.ix_(x_idx, x_idx)]
            adj_encoded = encode_adj(adj_copy.copy(), max_prev_node=self.max_prev_node)
            # get x and y and adj
            # for small graph the rest are zero padded
            y_batch[i, 0:adj_encoded.shape[0], :] = adj_encoded
            x_batch[i, 1:adj_encoded.shape[0]+1, :] = adj_encoded
        # sort in descending order
        len_batch_order = np.argsort(len_batch)[::-1]
        len_batch = len_batch[len_batch_order]
        x_batch = x_batch[len_batch_order,:,:]
        y_batch = y_batch[len_batch_order,:,:]

        return torch.from_numpy(x_batch).float(), torch.from_numpy(y_batch).float(), len_batch.astype('int').tolist()
    def calc_max_prev_node(self,iter):
        max_prev_node = []
        for i in range(iter):
            if i%(iter/10)==0:
                print(i)
            adj_idx = np.random.randint(len(self.adj_all))
            adj_copy = self.adj_all[adj_idx].copy()
            # print('Graph size', adj_copy.shape[0])
            x_idx = np.random.permutation(adj_copy.shape[0])
            adj_copy = adj_copy[np.ix_(x_idx, x_idx)]
            adj_copy_matrix = np.asmatrix(adj_copy)
            G = nx.from_numpy_matrix(adj_copy_matrix)
            time1 = time.time()
            # then do bfs in the permuted G
            start_idx = np.random.randint(adj_copy.shape[0])
            x_idx = np.array(bfs_seq(G, start_idx))
            adj_copy = adj_copy[np.ix_(x_idx, x_idx)]
            # encode adj
            adj_encoded = encode_adj_flexible(adj_copy.copy())
            max_encoded_len = max([len(adj_encoded[i]) for i in range(len(adj_encoded))])
            max_prev_node.append(max_encoded_len)
        max_prev_node = sorted(max_prev_node)[-100:]
        return max_prev_node


# graphs, max_num_nodes = Graph_load_batch(min_num_nodes=6, name='DD',node_attributes=False)
# dataset = Graph_sequence_sampler_truncate([nx.karate_club_graph()])
# max_prev_nodes = dataset.calc_max_prev_node(iter=10000)
# print(max_prev_nodes)
# x,y,len = dataset.sample()
# print('x',x)
# print('y',y)
# print(len)




# only output y_batch (which is needed in batch version of new model)
class Graph_sequence_sampler_fast():
    def __init__(self, G_list, max_node_num=25, batch_size=4, max_prev_node = 25):
        self.batch_size = batch_size
        self.G_list = G_list
        self.n = max_node_num
        self.max_prev_node = max_prev_node

        self.adj_all = []
        for G in G_list:
            self.adj_all.append(np.asarray(nx.to_numpy_matrix(G)))


    def sample(self):
        # batch, length, feature
        y_batch = np.zeros((self.batch_size, self.n, self.max_prev_node))  # here zeros are padded for small graph
        # generate input x, y pairs
        for i in range(self.batch_size):
            # first sample and get a permuted adj
            adj_idx = np.random.randint(len(self.adj_all))
            adj_copy = self.adj_all[adj_idx].copy()
            # print('graph size',adj_copy.shape[0])
            x_idx = np.random.permutation(adj_copy.shape[0])
            adj_copy = adj_copy[np.ix_(x_idx, x_idx)]
            adj_copy_matrix = np.asmatrix(adj_copy)
            G = nx.from_numpy_matrix(adj_copy_matrix)
            # then do bfs in the permuted G
            start_idx = np.random.randint(adj_copy.shape[0])
            x_idx = np.array(bfs_seq(G, start_idx))
            adj_copy = adj_copy[np.ix_(x_idx, x_idx)]
            # get the feature for the permuted G
            # dict = nx.bfs_successors(G, start_idx)
            # print('dict', dict, 'node num', self.G.number_of_nodes())
            # print('x idx', x_idx, 'len', len(x_idx))

            # print('adj')
            # np.set_printoptions(linewidth=200)
            # for print_i in range(adj_copy.shape[0]):
            #     print(adj_copy[print_i].astype(int))
            # adj_before = adj_copy.copy()

            # encode adj
            adj_encoded = encode_adj(adj_copy.copy(), max_prev_node=self.max_prev_node)
            # print('adj encoded')
            # np.set_printoptions(linewidth=200)
            # for print_i in range(adj_copy.shape[0]):
            #     print(adj_copy[print_i].astype(int))


            # decode adj
            # print('adj recover error')
            # adj_decode = decode_adj(adj_encoded.copy(), max_prev_node=self.max_prev_node)
            # adj_err = adj_decode-adj_copy
            # print(np.sum(adj_err))
            # if np.sum(adj_err)!=0:
            #     print(adj_err)
            # np.set_printoptions(linewidth=200)
            # for print_i in range(adj_err.shape[0]):
            #     print(adj_err[print_i].astype(int))

            # get x and y and adj
            # for small graph the rest are zero padded
            y_batch[i, 0:adj_encoded.shape[0], :] = adj_encoded


            # np.set_printoptions(linewidth=200,precision=3)
            # print('y\n')
            # for print_i in range(self.y_batch[i,:,:].shape[0]):
            #     print(self.y_batch[i,:,:][print_i].astype(int))
            # print('x\n')
            # for print_i in range(self.x_batch[i, :, :].shape[0]):
            #     print(self.x_batch[i, :, :][print_i].astype(int))
            # print('adj\n')
            # for print_i in range(self.adj_batch[i, :, :].shape[0]):
            #     print(self.adj_batch[i, :, :][print_i].astype(int))
            # print('adj_norm\n')
            # for print_i in range(self.adj_norm_batch[i, :, :].shape[0]):
            #     print(self.adj_norm_batch[i, :, :][print_i].astype(float))
            # print('feature\n')
            # for print_i in range(self.feature_batch[i, :, :].shape[0]):
            #     print(self.feature_batch[i, :, :][print_i].astype(float))


        # print('x_batch\n',self.x_batch)
        # print('y_batch\n',self.y_batch)

        return torch.from_numpy(y_batch).float()

# graphs, max_num_nodes = Graph_load_batch(min_num_nodes=6, name='PROTEINS_full')
# print(max_num_nodes)
# G = nx.ladder_graph(100)
# # G1 = nx.karate_club_graph()
# # G2 = nx.connected_caveman_graph(4,5)
# G_list = [G]
# dataset = Graph_sequence_sampler_fast(graphs, batch_size=128, max_node_num=max_num_nodes, max_prev_node=30)
# for i in range(5):
#     time0 = time.time()
#     y = dataset.sample()
#     time1 = time.time()
#     print(i,'time', time1 - time0)


# output size is flexible (using list to represent), batch size is 1
class Graph_sequence_sampler_flexible():
    def __init__(self, G_list):
        self.G_list = G_list
        self.adj_all = []
        for G in G_list:
            self.adj_all.append(np.asarray(nx.to_numpy_matrix(G)))

        self.y_batch = []
    def sample(self):
        # generate input x, y pairs
        # first sample and get a permuted adj
        adj_idx = np.random.randint(len(self.adj_all))
        adj_copy = self.adj_all[adj_idx].copy()
        # print('graph size',adj_copy.shape[0])
        x_idx = np.random.permutation(adj_copy.shape[0])
        adj_copy = adj_copy[np.ix_(x_idx, x_idx)]
        adj_copy_matrix = np.asmatrix(adj_copy)
        G = nx.from_numpy_matrix(adj_copy_matrix)
        # then do bfs in the permuted G
        start_idx = np.random.randint(adj_copy.shape[0])
        x_idx = np.array(bfs_seq(G, start_idx))
        adj_copy = adj_copy[np.ix_(x_idx, x_idx)]
        # get the feature for the permuted G
        # dict = nx.bfs_successors(G, start_idx)
        # print('dict', dict, 'node num', self.G.number_of_nodes())
        # print('x idx', x_idx, 'len', len(x_idx))

        # print('adj')
        # np.set_printoptions(linewidth=200)
        # for print_i in range(adj_copy.shape[0]):
        #     print(adj_copy[print_i].astype(int))
        # adj_before = adj_copy.copy()

        # encode adj
        adj_encoded = encode_adj_flexible(adj_copy.copy())
        # print('adj encoded')
        # np.set_printoptions(linewidth=200)
        # for print_i in range(adj_copy.shape[0]):
        #     print(adj_copy[print_i].astype(int))


        # decode adj
        # print('adj recover error')
        # adj_decode = decode_adj(adj_encoded.copy(), max_prev_node=self.max_prev_node)
        # adj_err = adj_decode-adj_copy
        # print(np.sum(adj_err))
        # if np.sum(adj_err)!=0:
        #     print(adj_err)
        # np.set_printoptions(linewidth=200)
        # for print_i in range(adj_err.shape[0]):
        #     print(adj_err[print_i].astype(int))

        # get x and y and adj
        # for small graph the rest are zero padded
        self.y_batch=adj_encoded


        # np.set_printoptions(linewidth=200,precision=3)
        # print('y\n')
        # for print_i in range(self.y_batch[i,:,:].shape[0]):
        #     print(self.y_batch[i,:,:][print_i].astype(int))
        # print('x\n')
        # for print_i in range(self.x_batch[i, :, :].shape[0]):
        #     print(self.x_batch[i, :, :][print_i].astype(int))
        # print('adj\n')
        # for print_i in range(self.adj_batch[i, :, :].shape[0]):
        #     print(self.adj_batch[i, :, :][print_i].astype(int))
        # print('adj_norm\n')
        # for print_i in range(self.adj_norm_batch[i, :, :].shape[0]):
        #     print(self.adj_norm_batch[i, :, :][print_i].astype(float))
        # print('feature\n')
        # for print_i in range(self.feature_batch[i, :, :].shape[0]):
        #     print(self.feature_batch[i, :, :][print_i].astype(float))

        return self.y_batch,adj_copy


# G = nx.ladder_graph(5)
# # G = nx.grid_2d_graph(20,20)
# # G = nx.ladder_graph(200)
# graphs = [G]
#
# graphs, max_num_nodes = Graph_load_batch(min_num_nodes=6, name='ENZYMES')
# sampler = Graph_sequence_sampler_flexible(graphs)
#
# y_max_all = []
# for i in range(10000):
#     y_raw,adj_copy = sampler.sample()
#     y_max = max(len(y_raw[i]) for i in range(len(y_raw)))
#     y_max_all.append(y_max)
#     # print('max bfs node',y_max)
# print('max', max(y_max_all))
# print(y[1])
# print(Variable(torch.FloatTensor(y[1])).cuda(CUDA))











########### potential use: an encoder along with the GraphRNN decoder
# preprocess the adjacency matrix
def preprocess(A):
    # Get size of the adjacency matrix
    size = len(A)
    # Get the degrees for each node
    degrees = np.sum(A, axis=1)+1

    # Create diagonal matrix D from the degrees of the nodes
    D = np.diag(np.power(degrees, -0.5).flatten())
    # Cholesky decomposition of D
    # D = np.linalg.cholesky(D)
    # Inverse of the Cholesky decomposition of D
    # D = np.linalg.inv(D)
    # Create an identity matrix of size x size
    I = np.eye(size)
    # Create A hat
    A_hat = A + I
    # Return A_hat
    A_normal = np.dot(np.dot(D,A_hat),D)
    return A_normal


# truncate the output seqence to save representation, and allowing for infinite generation
# now having a list of graphs
class Graph_sequence_sampler_bfs_permute_truncate_multigraph():
    def __init__(self, G_list, max_node_num=25, batch_size=4, max_prev_node = 25, feature = None):
        self.batch_size = batch_size
        self.G_list = G_list
        self.n = max_node_num
        self.max_prev_node = max_prev_node

        self.adj_all = []
        for G in G_list:
            self.adj_all.append(np.asarray(nx.to_numpy_matrix(G)))
        self.has_feature = feature

    def sample(self):

        # batch, length, feature
        # self.x_batch = np.ones((self.batch_size, self.n - 1, self.max_prev_node))
        x_batch = np.zeros((self.batch_size, self.n, self.max_prev_node)) # here zeros are padded for small graph
        # self.x_batch[:,0,:] = np.ones((self.batch_size, self.max_prev_node))  # first input is all ones
        # batch, length, feature
        y_batch = np.zeros((self.batch_size, self.n, self.max_prev_node)) # here zeros are padded for small graph
        # batch, length, length
        adj_batch = np.zeros((self.batch_size, self.n, self.n)) # here zeros are padded for small graph
        # batch, size, size
        adj_norm_batch = np.zeros((self.batch_size, self.n, self.n))  # here zeros are padded for small graph
        # batch, size, feature_len: degree and clustering coefficient
        if self.has_feature is None:
            feature_batch = np.zeros((self.batch_size, self.n, self.n)) # use one hot feature
        else:
            feature_batch = np.zeros((self.batch_size, self.n, 2))

        # generate input x, y pairs
        for i in range(self.batch_size):
            time0 = time.time()
            # first sample and get a permuted adj
            adj_idx = np.random.randint(len(self.adj_all))
            adj_copy = self.adj_all[adj_idx].copy()
            # print('Graph size', adj_copy.shape[0])
            x_idx = np.random.permutation(adj_copy.shape[0])
            adj_copy = adj_copy[np.ix_(x_idx, x_idx)]
            adj_copy_matrix = np.asmatrix(adj_copy)
            G = nx.from_numpy_matrix(adj_copy_matrix)
            time1 = time.time()
            # then do bfs in the permuted G
            start_idx = np.random.randint(adj_copy.shape[0])
            x_idx = np.array(bfs_seq(G, start_idx))
            adj_copy = adj_copy[np.ix_(x_idx, x_idx)]
            # get the feature for the permuted G
            node_list = [G.nodes()[i] for i in x_idx]
            feature_degree = np.array(list(G.degree(node_list).values()))[:,np.newaxis]
            feature_clustering = np.array(list(nx.clustering(G,nodes=node_list).values()))[:,np.newaxis]
            time2 = time.time()

            # dict = nx.bfs_successors(G, start_idx)
            # print('dict', dict, 'node num', self.G.number_of_nodes())
            # print('x idx', x_idx, 'len', len(x_idx))

            # print('adj')
            # np.set_printoptions(linewidth=200)
            # for print_i in range(adj_copy.shape[0]):
            #     print(adj_copy[print_i].astype(int))
            # adj_before = adj_copy.copy()

            # encode adj
            adj_encoded = encode_adj(adj_copy.copy(), max_prev_node=self.max_prev_node)
            # print('adj encoded')
            # np.set_printoptions(linewidth=200)
            # for print_i in range(adj_copy.shape[0]):
            #     print(adj_copy[print_i].astype(int))


            # decode adj
            # print('adj recover error')
            # adj_decode = decode_adj(adj_encoded.copy(), max_prev_node=self.max_prev_node)
            # adj_err = adj_decode-adj_copy
            # print(np.sum(adj_err))
            # if np.sum(adj_err)!=0:
            #     print(adj_err)
            # np.set_printoptions(linewidth=200)
            # for print_i in range(adj_err.shape[0]):
            #     print(adj_err[print_i].astype(int))

            # get x and y and adj
            # for small graph the rest are zero padded
            y_batch[i, 0:adj_encoded.shape[0], :] = adj_encoded
            x_batch[i, 1:adj_encoded.shape[0]+1, :] = adj_encoded
            adj_batch[i, 0:adj_copy.shape[0], 0:adj_copy.shape[0]] = adj_copy
            adj_copy_norm = preprocess(adj_copy)
            time3 = time.time()
            adj_norm_batch[i, 0:adj_copy.shape[0], 0:adj_copy.shape[0]] = adj_copy_norm

            if self.has_feature is None:
                feature_batch[i, 0:adj_copy.shape[0], 0:adj_copy.shape[0]] = np.eye(adj_copy.shape[0])
            else:
                feature_batch[i,0:adj_copy.shape[0],:] = np.concatenate((feature_degree,feature_clustering),axis=1)


            # np.set_printoptions(linewidth=200,precision=3)
            # print('y\n')
            # for print_i in range(self.y_batch[i,:,:].shape[0]):
            #     print(self.y_batch[i,:,:][print_i].astype(int))
            # print('x\n')
            # for print_i in range(self.x_batch[i, :, :].shape[0]):
            #     print(self.x_batch[i, :, :][print_i].astype(int))
            # print('adj\n')
            # for print_i in range(self.adj_batch[i, :, :].shape[0]):
            #     print(self.adj_batch[i, :, :][print_i].astype(int))
            # print('adj_norm\n')
            # for print_i in range(self.adj_norm_batch[i, :, :].shape[0]):
            #     print(self.adj_norm_batch[i, :, :][print_i].astype(float))
            # print('feature\n')
            # for print_i in range(self.feature_batch[i, :, :].shape[0]):
            #     print(self.feature_batch[i, :, :][print_i].astype(float))
            time4 = time.time()
            # print('1 ',time1-time0)
            # print('2 ',time2-time1)
            # print('3 ',time3-time2)
            # print('4 ',time4-time3)

        # print('x_batch\n',self.x_batch)
        # print('y_batch\n',self.y_batch)

        return torch.from_numpy(x_batch).float(), torch.from_numpy(y_batch).float(),\
               torch.from_numpy(adj_batch).float(), torch.from_numpy(adj_norm_batch).float(), torch.from_numpy(feature_batch).float()




















# generate own synthetic dataset
def Graph_synthetic(seed):
    G = nx.Graph()
    np.random.seed(seed)
    base = np.repeat(np.eye(5), 20, axis=0)
    rand = np.random.randn(100, 5) * 0.05
    node_features = base + rand

    # # print('node features')
    # for i in range(node_features.shape[0]):
    #     print(np.around(node_features[i], decimals=4))

    node_distance_l1 = np.ones((node_features.shape[0], node_features.shape[0]))
    node_distance_np = np.zeros((node_features.shape[0], node_features.shape[0]))
    for i in range(node_features.shape[0]):
        for j in range(node_features.shape[0]):
            if i != j:
                node_distance_l1[i,j] = np.sum(np.abs(node_features[i] - node_features[j]))
                # print('node distance', node_distance_l1[i,j])
                node_distance_np[i, j] = 1 / np.sum(np.abs(node_features[i] - node_features[j]) ** 2)

    print('node distance max', np.max(node_distance_l1))
    print('node distance min', np.min(node_distance_l1))
    node_distance_np_sum = np.sum(node_distance_np, axis=1, keepdims=True)
    embedding_dist = node_distance_np / node_distance_np_sum

    # generate the graph
    average_degree = 9
    for i in range(node_features.shape[0]):
        for j in range(i + 1, embedding_dist.shape[0]):
            p = np.random.rand()
            if p < embedding_dist[i, j] * average_degree:
                G.add_edge(i, j)

    G.remove_nodes_from(nx.isolates(G))
    print('num of nodes', G.number_of_nodes())
    print('num of edges', G.number_of_edges())

    G_deg = nx.degree_histogram(G)
    G_deg_sum = [a * b for a, b in zip(G_deg, range(0, len(G_deg)))]
    print('average degree', sum(G_deg_sum) / G.number_of_nodes())
    print('average path length', nx.average_shortest_path_length(G))
    print('diameter', nx.diameter(G))
    G_cluster = sorted(list(nx.clustering(G).values()))
    print('average clustering coefficient', sum(G_cluster) / len(G_cluster))
    print('Graph generation complete!')
    # node_features = np.concatenate((node_features, np.zeros((1,node_features.shape[1]))),axis=0)

    return G, node_features

# G = Graph_synthetic(10)



# return adj and features from a single graph
class GraphDataset_adj(torch.utils.data.Dataset):
    """Graph Dataset"""
    def __init__(self, G, features=None):
        self.G = G
        self.n = G.number_of_nodes()
        adj = np.asarray(nx.to_numpy_matrix(self.G))

        # permute adj
        subgraph_idx = np.random.permutation(self.n)
        # subgraph_idx = np.arange(self.n)
        adj = adj[np.ix_(subgraph_idx, subgraph_idx)]

        self.adj = torch.from_numpy(adj+np.eye(len(adj))).float()
        self.adj_norm = torch.from_numpy(preprocess(adj)).float()
        if features is None:
            self.features = torch.Tensor(self.n, self.n)
            self.features = nn.init.eye(self.features)
        else:
            features = features[subgraph_idx,:]
            self.features = torch.from_numpy(features).float()
        print('embedding size', self.features.size())
    def __len__(self):
        return 1
    def __getitem__(self, idx):
        sample = {'adj':self.adj,'adj_norm':self.adj_norm, 'features':self.features}
        return sample

# G = nx.karate_club_graph()
# dataset = GraphDataset_adj(G)
# train_loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True, num_workers=1)
# for data in train_loader:
#     print(data)


# return adj and features from a list of graphs
class GraphDataset_adj_batch(torch.utils.data.Dataset):
    """Graph Dataset"""
    def __init__(self, graphs, has_feature = True, num_nodes = 20):
        self.graphs = graphs
        self.has_feature = has_feature
        self.num_nodes = num_nodes
    def __len__(self):
        return len(self.graphs)
    def __getitem__(self, idx):
        adj_raw = np.asarray(nx.to_numpy_matrix(self.graphs[idx]))
        np.fill_diagonal(adj_raw,0) # in case the self connection already exists

        # sample num_nodes size subgraph
        subgraph_idx = np.random.permutation(adj_raw.shape[0])[0:self.num_nodes]
        adj_raw = adj_raw[np.ix_(subgraph_idx,subgraph_idx)]

        adj = torch.from_numpy(adj_raw+np.eye(len(adj_raw))).float()
        adj_norm = torch.from_numpy(preprocess(adj_raw)).float()
        adj_raw = torch.from_numpy(adj_raw).float()
        if self.has_feature:
            dictionary = nx.get_node_attributes(self.graphs[idx], 'feature')
            features = np.zeros((self.num_nodes, list(dictionary.values())[0].shape[0]))
            for i in range(self.num_nodes):
                features[i, :] = list(dictionary.values())[subgraph_idx[i]]
            # normalize
            features -= np.mean(features, axis=0)
            epsilon = 1e-6
            features /= (np.std(features, axis=0)+epsilon)
            features = torch.from_numpy(features).float()
        else:
            n = self.num_nodes
            features = torch.Tensor(n, n)
            features = nn.init.eye(features)

        sample = {'adj':adj,'adj_norm':adj_norm, 'features':features, 'adj_raw':adj_raw}
        return sample

# return adj and features from a list of graphs, batch size = 1, so that graphs can have various size each time
class GraphDataset_adj_batch_1(torch.utils.data.Dataset):
    """Graph Dataset"""

    def __init__(self, graphs, has_feature=True):
        self.graphs = graphs
        self.has_feature = has_feature

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        adj_raw = np.asarray(nx.to_numpy_matrix(self.graphs[idx]))
        np.fill_diagonal(adj_raw, 0)  # in case the self connection already exists
        n = adj_raw.shape[0]
        # give a permutation
        subgraph_idx = np.random.permutation(n)
        # subgraph_idx = np.arange(n)

        adj_raw = adj_raw[np.ix_(subgraph_idx, subgraph_idx)]

        adj = torch.from_numpy(adj_raw + np.eye(len(adj_raw))).float()
        adj_norm = torch.from_numpy(preprocess(adj_raw)).float()

        if self.has_feature:
            dictionary = nx.get_node_attributes(self.graphs[idx], 'feature')
            features = np.zeros((n, list(dictionary.values())[0].shape[0]))
            for i in range(n):
                features[i, :] = list(dictionary.values())[i]
            features = features[subgraph_idx, :]
            # normalize
            features -= np.mean(features, axis=0)
            epsilon = 1e-6
            features /= (np.std(features, axis=0) + epsilon)
            features = torch.from_numpy(features).float()
        else:
            features = torch.Tensor(n, n)
            features = nn.init.eye(features)

        sample = {'adj': adj, 'adj_norm': adj_norm, 'features': features}
        return sample




# get one node at a time, for a single graph
class GraphDataset(torch.utils.data.Dataset):
    """Graph Dataset"""
    def __init__(self, G, hops = 1, max_degree = 5, vocab_size = 35, embedding_dim = 35, embedding = None,  shuffle_neighbour = True):
        self.G = G
        self.shuffle_neighbour = shuffle_neighbour
        self.hops = hops
        self.max_degree = max_degree
        if embedding is None:
            self.embedding = torch.Tensor(vocab_size, embedding_dim)
            self.embedding = nn.init.eye(self.embedding)
        else:
            self.embedding = torch.from_numpy(embedding).float()
        print('embedding size', self.embedding.size())
    def __len__(self):
        return len(self.G.nodes())
    def __getitem__(self, idx):
        idx = idx+1
        idx_list = [idx]
        node_list = [self.embedding[idx].view(-1, self.embedding.size(1))]
        node_count_list = []
        for i in range(self.hops):
            # sample this hop
            adj_list = np.array([])
            adj_count_list = np.array([])
            for idx in idx_list:
                if self.shuffle_neighbour:
                    adj_list_new = list(self.G.adj[idx - 1])
                    random.shuffle(adj_list_new)
                    adj_list_new = np.array(adj_list_new) + 1
                else:
                    adj_list_new = np.array(list(self.G.adj[idx-1]))+1
                adj_count_list_new = np.array([len(adj_list_new)])
                adj_list = np.concatenate((adj_list, adj_list_new), axis=0)
                adj_count_list = np.concatenate((adj_count_list, adj_count_list_new), axis=0)
            # print(i, adj_list)
            # print(i, embedding(Variable(torch.from_numpy(adj_list)).long()))
            index = torch.from_numpy(adj_list).long()
            adj_list_emb = self.embedding[index]
            node_list.append(adj_list_emb)
            node_count_list.append(adj_count_list)
            idx_list = adj_list


        # padding, used as target
        idx_list = [idx]
        node_list_pad = [self.embedding[idx].view(-1, self.embedding.size(1))]
        node_count_list_pad = []
        node_adj_list = []
        for i in range(self.hops):
            adj_list = np.zeros(self.max_degree ** (i + 1))
            adj_count_list = np.ones(self.max_degree ** (i)) * self.max_degree
            for j, idx in enumerate(idx_list):
                if idx == 0:
                    adj_list_new = np.zeros(self.max_degree)
                else:
                    if self.shuffle_neighbour:
                        adj_list_new = list(self.G.adj[idx - 1])
                        # random.shuffle(adj_list_new)
                        adj_list_new = np.array(adj_list_new) + 1
                    else:
                        adj_list_new = np.array(list(self.G.adj[idx-1]))+1
                start_idx = j * self.max_degree
                incre_idx = min(self.max_degree, adj_list_new.shape[0])
                adj_list[start_idx:start_idx + incre_idx] = adj_list_new[:incre_idx]
            index = torch.from_numpy(adj_list).long()
            adj_list_emb = self.embedding[index]
            node_list_pad.append(adj_list_emb)
            node_count_list_pad.append(adj_count_list)
            idx_list = adj_list
            # calc adj matrix
            node_adj = torch.zeros(index.size(0),index.size(0))
            for first in range(index.size(0)):
                for second in range(first, index.size(0)):
                    if index[first]==index[second]:
                        node_adj[first,second] = 1
                        node_adj[second,first] = 1
                    elif self.G.has_edge(index[first],index[second]):
                        node_adj[first, second] = 0.5
                        node_adj[second, first] = 0.5
            node_adj_list.append(node_adj)


        node_list = list(reversed(node_list))
        node_count_list = list(reversed(node_count_list))
        node_list_pad = list(reversed(node_list_pad))
        node_count_list_pad = list(reversed(node_count_list_pad))
        node_adj_list = list(reversed(node_adj_list))
        sample = {'node_list':node_list, 'node_count_list':node_count_list,
                  'node_list_pad':node_list_pad, 'node_count_list_pad':node_count_list_pad, 'node_adj_list':node_adj_list}
        return sample


