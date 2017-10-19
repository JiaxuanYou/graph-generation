import torch
import torchvision as tv
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt

import networkx as nx
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



def Graph_synthetic(seed):
    G = nx.Graph()
    np.random.seed(seed)
    base = np.repeat(np.eye(5), 20, axis=0)
    rand = np.random.randn(100, 5) * 0.05
    node_features = base + rand

    print('node features')
    for i in range(node_features.shape[0]):
        print(np.around(node_features[i], decimals=4))

    node_distance_l1 = np.ones((node_features.shape[0], node_features.shape[0]))
    node_distance_np = np.zeros((node_features.shape[0], node_features.shape[0]))
    for i in range(node_features.shape[0]):
        for j in range(node_features.shape[0]):
            if i != j:
                node_distance_l1[i,j] = np.sum(np.abs(node_features[i] - node_features[j]))
                print('node distance', node_distance_l1[i,j])
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
    node_features = np.concatenate((node_features, np.zeros((1,node_features.shape[1]))),axis=0)

    return G, node_features


# G = Graph_synthetic(10)



# only get one-hop neighbour
# class GraphDataset(torch.utils.data.Dataset):
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


        node_list = list(reversed(node_list))
        node_count_list = list(reversed(node_count_list))
        node_list_pad = list(reversed(node_list_pad))
        node_count_list_pad = list(reversed(node_count_list_pad))
        sample = {'node_list':node_list, 'node_count_list':node_count_list,
                  'node_list_pad':node_list_pad, 'node_count_list_pad':node_count_list_pad}
        return sample



#### test code #####
#
# G = nx.karate_club_graph()
# # G = nx.LCF_graph(14,[5,-5],7)
# # G = nx.LCF_graph(20,[-9,-9],10)
#
# dataset = GraphDataset(G, shuffle_neighbour = False, hops=3, max_degree=7, vocab_size=36, embedding_dim=36)
#
# train_loader = torch.utils.data.DataLoader(dataset,
#         batch_size=1, shuffle=False, num_workers=1)
#
# embedding_size = dataset.embedding.size(1)
# encoder = Encoder(embedding_size, 3).cuda(CUDA)
# decoder = CNN_decoder(encoder.input_size*16, embedding_size, stride=2).cuda(CUDA)
#
#
#
# for batch_idx, data in enumerate(train_loader):
#     # for node in data['node_list']:
#     #     print('node_list', node)
#     # for node in data['node_count_list']:
#     #     print('node_count_list', node)
#     # for node in data['node_list_pad']:
#     #     print('node_list_pad', node)
#     # for node in data['node_count_list_pad']:
#     #     print('node_count_list_pad', node)
#
#     # print(data['node_list'][0])
#     # print(data['node_count_list'][0])
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
#     y = encoder(data['node_list'], data['node_count_list'])
#     print('get embedding')
#     print(y.size())
#
#     print('decoder start')
#     x_pred = decoder(y)
#     x_pred = x_pred.view(x_pred.size(2),x_pred.size(1))
#     print('get reconstructed graph')
#     print(x_pred.size())
#     x_real = Variable(data['node_list_pad'][2]).cuda(CUDA)
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