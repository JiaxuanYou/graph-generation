import networkx as nx
import numpy as np
import torch

class GraphAdjSampler(torch.utils.data.Dataset):
    def __init__(self, G_list, max_num_nodes, features='id'):
        self.max_num_nodes = max_num_nodes
        self.adj_all = []
        self.len_all = []
        self.feature_all = []

        for G in G_list:
            # the diagonal entries are 1 since they denote node probability
            self.adj_all.append(
                    np.asarray(nx.to_numpy_matrix(G)) + np.identity(G.number_of_nodes()))
            self.len_all.append(G.number_of_nodes())
            if features == 'id':
                self.feature_all.append(np.identity(max_num_nodes))

    def __len__(self):
        return len(self.adj_all)

    def __getitem__(self, idx):
        adj = self.adj_all[idx]
        num_nodes = adj.shape[0]
        adj_padded = np.zeros((self.max_num_nodes, self.max_num_nodes))
        adj_padded[:num_nodes, :num_nodes] = adj

        adj_decoded = np.zeros(self.max_num_nodes * (self.max_num_nodes + 1) // 2)
        node_idx = 0
        print(adj_padded)
        for i in range(self.max_num_nodes):
            for j in range(i+1):
                adj_decoded[node_idx] = adj_padded[i, j]
                node_idx += 1
        print(adj_decoded)
        print(np.tril(adj_padded))
        
        return {'adj':adj_padded,
                'adj_decoded':adj_decoded, 
                'features':self.feature_all[idx].copy()}

