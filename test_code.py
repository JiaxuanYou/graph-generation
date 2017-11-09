# import cPickle
import pickle
import sys
import networkx as nx
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import numpy as np
import community
from random import shuffle

# G = nx.ladder_graph(4)
# print(nx.info(G))
# print(nx.cycle_basis(G,0))
# print(nx.cycle_basis(G))
# dict = nx.bfs_successors(G,3)
# print(dict)

plt.switch_backend('agg')

G = nx.karate_club_graph()

length = nx.all_pairs_shortest_path_length(G)



# G_cluster = sorted(list(nx.clustering(G).values()))
# print(G_cluster)
# plt.hist(np.array(G_cluster),bins='auto')
# plt.savefig('figures/test.png')
# plt.close()

# PIK = "graphs/pickle.dat"
#
# G = nx.karate_club_graph()
# data = [G,G,G,G,G]
# with open(PIK, "wb") as f:
#     pickle.dump(data, f)
# with open(PIK, "rb") as f:
#     list = pickle.load(f)
# print(list[0].nodes(True))

# adj = np.zeros((10,10))
# adj[0:5,0:5] = np.eye(5)
# adj = adj[~np.all(adj == 0, axis=1)]
# adj = adj[:,~np.all(adj == 0, axis=0)]
# print(adj)
# adj = np.asmatrix(adj)
#
# G = nx.from_numpy_matrix(adj)
# print(G.nodes())


# x_idx = np.array(list(nx.dfs_preorder_nodes(G, 0)))
# G = nx.grid_2d_graph(5,5)
# idx = np.random.permutation(25)
# print(G.nodes())
# node_list = [G.nodes()[i] for i in idx]
# print(node_list)
# print(G.degree(G.nodes()))
# print(list(G.degree(node_list).values()))
# print(nx.clustering(G,nodes=node_list))




# start_id = 27
# dict = nx.bfs_successors(G,start_id)
# print(dict)





# def bfs_seq(G,start_id):
#     dict = nx.bfs_successors(G, start_id)
#     start = [start_id]
#     output = [start_id]
#     while len(start)>0:
#         next = []
#         while len(start)>0:
#             current = start.pop(0)
#             neighbor = dict.get(current)
#             if neighbor is not None:
#                 next = next + neighbor
#         output = output + next
#         start = next
#     return output
#
# print(bfs_seq(G,start_id))

# print(x_idx)

# parts = community.best_partition(G)
# values = [parts.get(node) for node in G.nodes()]
# print(values)
# colors = []
# for i in range(len(values)):
# 	if values[i]==0:
# 		colors.append('red')
# 	if values[i]==1:
# 		colors.append('green')
# 	if values[i]==2:
# 		colors.append('blue')
# 	if values[i]==3:
# 		colors.append('black')
#
#
# spring_pos = nx.spring_layout(G)
# plt.switch_backend('agg')
# plt.axis("off")
# nx.draw_networkx(G,pos = spring_pos, with_labels=False, node_size = 35, node_color=colors)
# plt.savefig('figures/test.png', dpi=200)
# plt.close()

# def draw_graph(G, prefix = 'test'):
#     plt.switch_backend('agg')
#     options = {
#         'node_color': 'black',
#         'node_size': 10,
#         'width': 1
#     }
#     plt.figure()
#     plt.subplot()
#     nx.draw_networkx(G, **options)
#     plt.savefig('figures/test.png', dpi=200)
#     plt.close()
#
# G = nx.balanced_tree(r = 3, h = 3)
# print(G.nodes(True))
# draw_graph(G)

# a = torch.LongTensor(range(18)).view(2,3,3)
# idx = torch.LongTensor([0,1,1])
# print(a)
# print(a[idx,range(len(idx)),:])
#
#

# a = torch.randn(3, 4, 2)
# a_norm = torch.norm(a, p=2, dim=2,keepdim = True)
# a = a/a_norm
# b = a[:,0:1,:]
# print(a @ a.permute(0,2,1))
# print(a @ b.permute(0,2,1))
# # print(a @ b.t())

# a = torch.ones(1,8)*-1
# print(a)
# print(F.sigmoid(a))

# a = np.arange(-3,7,1)
# print(a)
# print(np.nonzero(a>0)[0])
# b = np.random.choice(np.nonzero(a>0)[0])
# print(b)
# print(a[b])


# a = np.ones((5,5))
# print(a)
# print(np.tril(a,-1))


# G = nx.gnp_random_graph(100,0.02)
#
# degree_sequence=sorted(nx.degree(G).values(),reverse=True) # degree sequence
# print(degree_sequence)

#
# batch_num = 4
# node_num = 3
# adj_init = torch.eye(node_num).view(1,node_num,node_num).repeat(batch_num,1,1)
# print(adj_init)




# reload(sys)  # Reload does the trick!
# sys.setdefaultencoding('unicode')
# print(sys.getdefaultencoding())
# a = torch.ones(2,3)
# b = torch.ones(3,2)
#
# a_batch = torch.ones(2,3,2,3)
# b_batch = torch.ones(3,2)
#
#
# # print(torch.matmul(a,b))
# print(torch.matmul(a_batch,b_batch))
# # print(a)



# dataset = 'cora'
#
#
# names = ['x', 'tx', 'allx', 'graph']
# objects = []
# for i in range(len(names)):
#     load = pkl.load(open("dataset/ind.{}.{}".format(dataset, names[i]), 'rb'),encoding='latin1')
#     print('loaded')
#     objects.append(load)
#     # print(load)
#
# print(type(objects[0]))



#
# with open('dataset/ind.citeseer.graph', 'r') as f:
#     a = pickle.load(f)
#     print(a)
#
# NAMES = ['x', 'y', 'tx', 'ty', 'allx', 'graph']
# OBJECTS = []
# for i in range(len(NAMES)):
#     with open('dataset/ind.{}.{}'.format(DATASET,NAMES[i]), 'rb') as f:
#         OBJECTS.append(cPickle.load(f))
# x, y, tx, ty, graph = tuple(OBJECTS)