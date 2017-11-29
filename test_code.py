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
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import torch.nn as nn


CUDA = 2
# G = nx.ladder_graph(4)
# print(nx.info(G))
# print(nx.cycle_basis(G,0))
# print(nx.cycle_basis(G))
# dict = nx.bfs_successors(G,3)
# print(dict)

# plt.switch_backend('agg')
#
# G = nx.karate_club_graph()
#
# length = nx.all_pairs_shortest_path_length(G)


# def gumbel_softmax(logits, temperature, eps=1e-9):
#     # get gumbel noise
#     noise = torch.rand(logits.size())
#     noise.add_(eps).log_().neg_()
#     noise.add_(eps).log_().neg_()
#     noise = Variable(noise)
#
#     x = (logits + noise) / temperature
#     x = F.softmax(x)
#     return x
#
# def gumbel_sigmoid(logits, temperature, eps=1e-9):
#     # get gumbel noise
#     noise = torch.rand(logits.size())
#     noise.add_(eps).log_().neg_()
#     noise.add_(eps).log_().neg_()
#     noise = Variable(noise)
#
#     x = (logits + noise) / temperature
#     x = F.sigmoid(x)
#     return x
#
# logits = Variable(torch.randn(8,1)*5)
# x = gumbel_sigmoid(logits=logits, temperature=1)
# print(logits)
# print(x)


# x = Variable(torch.zeros(2,1)).cuda(CUDA)
# y = x.data
# print(x)
# print(y)

# a = np.arange(0,11,3)
# b = range(0,11,3)
# print(a)
# print(b[1])



################# learn autograd
# a = Variable(torch.ones(2), requires_grad = True)
# b = Variable(torch.ones(2)*2)
# c = torch.sum(a*b)
# d = c*2
#
# d.backward(retain_graph=True)
# # d.backward()
# print(a.grad)
#
# d = c*5
# d.backward()
# print(a.grad)

# y = torch.Tensor([-1,0,1,2,3,4,0])
# print(y)
# print(torch.squeeze(torch.nonzero(y<=0)))


############# test the behaviour of using .long()
y = torch.rand(5,1)





# x = torch.LongTensor([1])
# print(x)
# print(x.size())
# print(len(x.size()))

# a = Variable(torch.rand(5, 3), requires_grad=True)
# a = a.clone() # Otherwise we change inplace a leaf Variable
# print(a)
#
# ind = Variable(torch.LongTensor([3]))
# a.index_fill_(0, ind, 0)
#
# print(a)
#
# a[1, :] = 0
#
# print(a)

# x = Variable(torch.randn(10, 20, 3)).cuda()
# lens = np.arange(1,11)[::-1]
#
# # print('x_before', x)
# x = pack_padded_sequence(x, lens, batch_first=True)
# # print('x_pack', x)
# print('x_unpack', pad_packed_sequence(x,batch_first=True)[0])
#
# lstm = nn.LSTM(3, 2, batch_first=True).cuda()
# h0 = Variable(torch.zeros(1, 10, 2)).cuda()
# c0 = Variable(torch.zeros(1, 10, 2)).cuda()
#
# packed_h, (packed_h_t, packed_c_t) = lstm(x, (h0, c0))
# h, _ = pad_packed_sequence(packed_h,batch_first=True)
# print(h.size()) # Size 20 x 10 x 50 instead of 10 x 20 x 50
# print(h)



# p_list = np.linspace(0,1,100)
# pred = p_list+(1-p_list)*p_list+(1-p_list)*(1-p_list)*p_list
# print(p_list)
# print(pred)




# a = torch.Tensor([1,2,3,4])
# print(a[-3:])

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


# torch.manual_seed(123)
# for i in range(10):
#     a = torch.randn([2])
#     print(a)