import numpy as np
import torch

class GraphAdjSampler(torch.utils.data.Dataset):
    def __init__(self, G_list, max_num_node=None, iteration=20000):
        self.adj_all = []
        self.len_all = []
        for G in G_list:
            self.adj_all.append(np.asarray(nx.to_numpy_matrix(G)))
            self.len_all.append(G.number_of_nodes())
        if max_num_node is None:
            self.max_num_node = max(self.len_all)
        else:
            self.max_num_node = max_num_node

    def __len__(self):
        return len(self.adj_all)

    def __getitem__(self, idx):
        adj_copy = self.adj_all[idx].copy()
        num_nodes = len(adj_copy)
        adj_padded = np.zeros((self.max_num_node, self.max_num_node))
        adj_padded[:num_nodes, :num_nodes] = adj_copy

        adj_decoded = np.zeros(self.max_num_node * (self.max_num_node + 1) / 2)
        idx = 0
        for i in range(self.max_num_node):
            for j in range(i+1):
                adj_decoded[idx] = adj_copy[i, j]
                idx += 1
        
        return {'adj':adj_copy,'adj_decoded':adj_decoded}

