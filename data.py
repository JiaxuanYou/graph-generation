import torch
import torchvision as tv
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt

import networkx as nx
import pickle as pkl
import scipy.sparse as sp
import logging

import random
from tensorboard_logger import configure, log_value
import shutil
import os
from model import *

# token config
PAD_token = 0
SOS_token = 1
EOS_token = 2
NODE_token = 3
# so the first meaningful entry will become 3


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

def Graph_load_batch(min_num_nodes = 20, name = 'ENZYMES'):
    '''
    load many graphs, e.g. enzymes
    :return: a list of graphs
    '''

    G = nx.Graph()
    # load data
    path = 'dataset/'+name+'/'
    data_adj = np.loadtxt(path+name+'_A.txt', delimiter=',').astype(int)
    data_node_att = np.loadtxt(path+name+'_node_attributes.txt', delimiter=',')
    data_node_label = np.loadtxt(path+name+'_node_labels.txt', delimiter=',').astype(int)
    data_graph_indicator = np.loadtxt(path+name+'_graph_indicator.txt', delimiter=',').astype(int)
    data_graph_labels = np.loadtxt(path+name+'_graph_labels.txt', delimiter=',').astype(int)


    data_tuple = list(map(tuple, data_adj))
    # print(len(data_tuple))
    # print(data_tuple[0])

    # add edges
    G.add_edges_from(data_tuple)
    # add node attributes
    for i in range(data_node_att.shape[0]):
        G.add_node(i+1, feature = data_node_att[i])
        G.add_node(i+1, label = data_node_label[i])
    G.remove_nodes_from(nx.isolates(G))

    # print(G.number_of_nodes())
    # print(G.number_of_edges())

    # split into graphs
    graph_num = data_graph_indicator.max()
    node_list = np.arange(data_graph_indicator.shape[0])+1
    graphs = []
    for i in range(graph_num):
        # find the nodes for each graph
        nodes = node_list[data_graph_indicator==i+1]
        G_sub = G.subgraph(nodes)
        G_sub.graph['label'] = data_graph_labels[i]
        # print('nodes', G_sub.number_of_nodes())
        # print('edges', G_sub.number_of_edges())
        # print('label', G_sub.graph)
        if G_sub.number_of_nodes()>=min_num_nodes:
            graphs.append(G_sub)
            # print(G_sub.number_of_nodes(), 'i', i)
    print('Graphs loaded, total num: ', len(graphs))
    logging.warning('Graphs loaded, total num: {}'.format(len(graphs)))

    return graphs

# Graph_load_batch()



def Graph_load(dataset = 'cora'):
    '''
    Load a single graph dataset
    :param dataset:
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


# adj, features,G = Graph_load()
# print(adj)
# print(G.number_of_nodes(), G.number_of_edges())



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




# return adj and features from a single graph
class GraphDataset_adj(torch.utils.data.Dataset):
    """Graph Dataset"""
    def __init__(self, G, features=None):
        self.G = G
        self.n = G.number_of_nodes()
        adj = np.asarray(nx.to_numpy_matrix(self.G))

        # permute adj
        # subgraph_idx = np.random.permutation(self.n)
        subgraph_idx = np.arange(self.n)
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

# return adj and features from a list of graphs
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


# graphs = Graph_load_batch(num_nodes=16)
# dataset = GraphDataset_adj_batch(graphs, num_nodes=16)
# # print(dataset[0])
# train_loader = torch.utils.data.DataLoader(dataset,
#         batch_size=16, shuffle=True, num_workers=0)
# for id, data in enumerate(train_loader):
#     # print('id', id)
#     if id ==10:
#         print(data)


# only get one-hop neighbour
# class GraphDataset(torch.utils.dataset.Dataset):
#     """Graph Dataset"""
#     def __init__(self, G, shuffle_neighbour = True, hops = 1):
#         self.G = G
#         self.shuffle_neighbour = shuffle_neighbour
#     def __len__(self):
#         return len(self.G.nodes())
#     def __getitem__(self, idx):
#         li = list(self.G.adj[idx])
#         if self.shuffle_neighbour:
#             random.shuffle(li)
#         li_new = [x+NODE_token for x in li] # allow for tokens
#         li_new.insert(0, idx+NODE_token) # node itself
#         li_new.insert(0, SOS_token) # SOS
#         li_new.append(EOS_token) # EOS
#         sample = {'nodes':torch.LongTensor(li_new)}
#         return sample



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



#### test code #####




# G = nx.karate_club_graph()
#
# dataset = GraphDataset(G, shuffle_neighbour = False, hops=3, max_degree=7, vocab_size=36, embedding_dim=36)
#
# a = dataset[0]
# print(a)

# # G = nx.LCF_graph(14,[5,-5],7)
# # G = nx.LCF_graph(20,[-9,-9],10)
#
# dataset = GraphDataset(G, shuffle_neighbour = False, hops=3, max_degree=7, vocab_size=36, embedding_dim=36)
#
# train_loader = torch.utils.dataset.DataLoader(dataset,
#         batch_size=1, shuffle=False, num_workers=1)
#
# embedding_size = dataset.embedding.size(1)
# encoder = Encoder(embedding_size, 3).cuda(CUDA)
# decoder = CNN_decoder(encoder.input_size*16, embedding_size, stride=2).cuda(CUDA)
#
#
#
# for batch_idx, dataset in enumerate(train_loader):
#     # for node in dataset['node_list']:
#     #     print('node_list', node)
#     # for node in dataset['node_count_list']:
#     #     print('node_count_list', node)
#     # for node in dataset['node_list_pad']:
#     #     print('node_list_pad', node)
#     # for node in dataset['node_count_list_pad']:
#     #     print('node_count_list_pad', node)
#
#     # print(dataset['node_list'][0])
#     # print(dataset['node_count_list'][0])
#
#     # test
#     # x0 = Variable(torch.randn(1, 1, embedding_size)).cuda(CUDA)
#     # x1 = Variable(torch.randn(1, 4, embedding_size)).cuda(CUDA)
#     # x2 = Variable(torch.randn(1, 12, embedding_size)).cuda(CUDA)
#     # x3 = Variable(torch.randn(1, 36, embedding_size)).cuda(CUDA)
#     # print(x3, x2, x1, x0)
#     # node_list = [x3, x2, x1, x0]
#     # count1 = torch.Tensor([3]).repeat(4)
#     # count2 = torch.Tensor([3]).repeat(12)
#     # node_count_list = [count2, count1]
#
#     print('encoder start')
#     y = encoder(dataset['node_list'], dataset['node_count_list'])
#     print('get embedding')
#     print(y.size())
#
#     print('decoder start')
#     x_pred = decoder(y)
#     x_pred = x_pred.view(x_pred.size(2),x_pred.size(1))
#     print('get reconstructed graph')
#     print(x_pred.size())
#     x_real = Variable(dataset['node_list_pad'][2]).cuda(CUDA)
#     x_real = x_real.view(x_real.size(1), x_real.size(2))
#     print(x_real.size())
#
#     x_pred = F.softmax(x_pred)
#     loss = F.binary_cross_entropy(x_pred, x_real)
#     print('loss', loss)
#
#
#
#
#
#     #
#     # if batch_idx == 0:
#     #     break