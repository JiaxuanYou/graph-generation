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
from main import *
from main_DGMG import *
from utils import *

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
# y = torch.rand(5,1)





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

# x = Variable(torch.randn(5, 20, 3)).cuda()
# # x = torch.randn(5, 20, 3).cuda()
# lens = np.arange(1,6)[::-1]
#
# print('x_before', x)
# x = pack_padded_sequence(x, lens, batch_first=True)
# print('x_pack', x.data)
# print('x_unpack', pad_packed_sequence(x,batch_first=True)[0])

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



#
# x = np.array([[1,0,0], [0,2,0], [1,1,0]])
# print(x)
# print(np.nonzero(x))
# print(x[np.nonzero(x)])

# G = nx.karate_club_graph()
# length_all = nx.all_pairs_shortest_path_length(G)
# width_max = 0
# for i in range(G.number_of_nodes()):
#     length = np.array(list(length_all[i].values()))
#     width = np.amax(np.bincount(length))
#     if width>width_max:
#         width_max=width
# print(width_max)

# a = np.ones((5,5))
# print(a)
# a[0,:] = 0
# print(a)

# n=23
# m=1
# G = nx.barabasi_albert_graph(n,m)
# degree = np.array(nx.degree_histogram(G))
# degree_norm = degree/np.sum(degree)
# print(degree)
# print(degree_norm)
#
# for i in range(10):
#     print('nodes',G.number_of_nodes())
#     print('edges',G.number_of_edges())
#     print('prediction', (n-m)*m)

# for i in range(3):
#     G = nx.barabasi_albert_graph(5,1)
#     adj = np.asarray(nx.to_numpy_matrix(G))
#     print('adj before\n', adj)
#
#     x_idx = np.random.permutation(adj.shape[0])
#     adj = adj[np.ix_(x_idx, x_idx)]
#     print('adj after\n', adj)




# args = Args()
#
# if args.graph_type == 'ladder':
#     graphs = []
#     for i in range(100, 201):
#         graphs.append(nx.ladder_graph(i))
#     args.max_prev_node = 10
# if args.graph_type == 'tree':
#     graphs = []
#     for i in range(2, 5):
#         for j in range(3, 5):
#             graphs.append(nx.balanced_tree(i, j))
#     args.max_prev_node = 256
# if args.graph_type == 'caveman':
#     graphs = []
#     for i in range(5, 10):
#         for j in range(5, 25):
#             graphs.append(nx.connected_caveman_graph(i, j))
#     args.max_prev_node = 50
# if args.graph_type == 'grid':
#     graphs = []
#     for i in range(10, 20):
#         for j in range(10, 20):
#             graphs.append(nx.grid_2d_graph(i, j))
#     args.max_prev_node = 40
# if args.graph_type == 'barabasi':
#     graphs = []
#     for i in range(100, 200):
#         graphs.append(nx.barabasi_albert_graph(i, 2))
#     args.max_prev_node = 130
#     # real graphs
# if args.graph_type == 'enzymes':
#     graphs = Graph_load_batch(min_num_nodes=10, name='ENZYMES')
#     args.max_prev_node = 25
# if args.graph_type == 'protein':
#     graphs = Graph_load_batch(min_num_nodes=20, name='PROTEINS_full')
#     args.max_prev_node = 80
# if args.graph_type == 'DD':
#     graphs = Graph_load_batch(min_num_nodes=100, max_num_nodes=500, name='DD', node_attributes=False, graph_labels=True)
#     args.max_prev_node = 230
#
# print(args.graph_type)
# print('number of graph', len(graphs))
# print('aver number of node', sum([graphs[i].number_of_nodes() for i in range(len(graphs))])/len(graphs))
# print('aver number of edge', sum([graphs[i].number_of_edges() for i in range(len(graphs))])/len(graphs))
# print('max number of node', max([graphs[i].number_of_nodes() for i in range(len(graphs))]))
#



# m = nn.Sigmoid()
# weight = torch.arange(1,11).view(1,10,1).repeat(5,1,2)
#
# # weight[:,-5:,:] *= 5
# print(weight)
# loss = nn.BCELoss(size_average=True,weight=weight)
# input = Variable(torch.ones(5,10,2)*0.8, requires_grad=True)
# target = Variable(torch.ones(5,10,2)*0.2)
# output = loss(m(input), target)
# output.backward()
# print(output)


# a = torch.arange(1,11)
# print(a)
# print(a/10*20)

# a = torch.LongTensor([[1,2,3],[4,5,6],[7,8,9]]).view(3,3,1)
# b = torch.LongTensor([[1,2,3],[4,5,6],[7,8,9]]).view(3,3,1)*2
# c = torch.LongTensor([[1,2,3],[4,5,6],[7,8,9]]).view(3,3,1)*3
# all = torch.cat((a,b,c),dim=2)
# all_vec = all.view(-1,3)
# all_restore = all.view(3,3,3)
# print(all)
# print(all_vec)
# print(all_restore)

# a = torch.Tensor([1,2,3,4,5])
# print(a)
# print(a[0:])

# a = range(10)
# print(type(list(a)))
# print(a)
# print(list(a))


# a = list(range(10))
# print(a)
# print(a[9:])

# print(np.arange(0,10))
# print(np.arange(9,-1,-1))
#
# y_len = list(range(10))
# output_y_len = []
# output_y_len_bin = np.bincount(np.array(y_len))
# for i in range(len(output_y_len_bin)-1,0,-1):
#     count_temp = np.sum(output_y_len_bin[i:])
#     output_y_len.extend([i]*count_temp)
# print('y_len',y_len)
# print('output_y_len',output_y_len)
# print('batch_size_real',len(output_y_len))
# print('batch_size_pred',sum(y_len))


# check networkit package usage
# import networkit as nt
# g = nt.generators.DynamicForestFireGenerator(p=0.3,False,r=0.3).generate()

# for i in range(10):
#     pass
# print(i)

# a = np.zeros(10)
# print(a)

######## view graph completion
# args = Args()
# epoch = 3000
# sample_time = 2
# graph_pred = load_graph_list(args.graph_save_path + args.fname_pred + str(epoch) +'_'+str(sample_time) + 'graph_completion.dat')
# shuffle(graph_pred)
# draw_graph_list(graph_pred[0:16], row=4, col=4, fname=args.figure_save_path + 'test'+ args.fname_pred + str(epoch) +'_'+str(sample_time))
# graph_real = load_graph_list(args.graph_save_path + args.fname_real + str(0) + '.dat')
# shuffle(graph_real)
# draw_graph_list(graph_real[0:16], row=4, col=4, fname=args.figure_save_path + 'test'+ args.fname_real + str(epoch) +'_'+str(sample_time))
#
# a = np.load('timing/GraphRNN_VAE_conditional_barabasi_4_128_.npy')
# for i in a:
#     print(i)


# G = nx.ladder_graph(4)
# adj_list = G.subgraph(0).adjacency_list()
# # adj_list[0].append(101010)
# print(adj_list)

#
# a = []
# for i in a:
#     print(i)

# a = [1,2,3,4,5]
#
# b = Variable(torch.Tensor([5,2,2,3]))
# print(b)
# print(b.size(0))
# print(b.expand(10,4))

#
# m = nn.Softmax()
# input = Variable(torch.randn(2, 3))
# print(input)
# print(m(input))

# print(list(range(1)))

# print(nx.complete_graph(1).number_of_nodes())

# a = list(range(10))
# b = list(range(10))
# shuffle(b)
# mapping=dict(zip(a, b))
# print(mapping)
#
# G = nx.karate_club_graph()
# print(G.nodes())
#
# adj_list = [[1],[0]]
# adj_dict = dict(zip(list(range(len(adj_list))), adj_list))
# dol= {0:[1], 1:[0]} # single edge (0,1)
#
# print(adj_dict)
# print(dol)
#
# G=nx.from_dict_of_lists(dol)
# adj_list = G.adjacency_list()
# print(adj_list)
#
# a = torch.LongTensor([10,1])
# b = torch.zeros(a.size())
# print(b)

# a = Variable(torch.Tensor([10])).cuda()
# print(a)
# print(type(a.data[0]))

# args = Args_DGMG()
# args = Args()
#
# epoch = 3000
# sample_time = 1
# # graph_pred = load_graph_list(args.graph_save_path + args.fname_pred + str(epoch) + '.dat')
# # shuffle(graph_pred)
# # draw_graph_list(graph_pred[0:16], row=4, col=4, fname=args.figure_save_path + 'test'+ args.fname_pred + str(epoch))
# # graph_real = load_graph_list(args.graph_save_path + args.fname_train + str(0) + '.dat')
# # shuffle(graph_real)
# # draw_graph_list(graph_real[0:16], row=4, col=4, fname=args.figure_save_path + 'test'+ args.fname_train + str(0))
#
#
# # dir = '/dfs/scratch0/jiaxuany0/'
# dir = ''
# graph_pred = load_graph_list(dir+args.graph_save_path + args.fname_pred + str(epoch) +'_'+str(sample_time) + '.dat')
# graph_pred_len_list = np.array([len(graph_pred[i]) for i in range(len(graph_pred))])
# pred_order = np.argsort(graph_pred_len_list)
# graph_pred = [graph_pred[i] for i in pred_order if graph_pred[i].number_of_nodes()>=2]
# for i in range(len(graph_pred)):
#     print(graph_pred[i].number_of_nodes())
# draw_graph_list(graph_pred[0:16], row=4, col=4, fname=args.figure_save_path + 'test'+ args.fname_pred + str(epoch) +'_'+str(sample_time))
# graph_real = load_graph_list(dir+args.graph_save_path + args.fname_train + str(0) + '.dat')
# shuffle(graph_real)
# draw_graph_list(graph_real[0:16], row=4, col=4, fname=args.figure_save_path + 'test'+ args.fname_train + str(epoch) +'_'+str(sample_time))

#
# a = list(range(1000,3001,100))
# print(a)

# a,b,c = [],[],[]
# print(a,b,c)


################# print average statistics #############
# # dir = '/dfs/scratch0/jiaxuany0/graphs/'
# dir = 'graphs/'
#
# # model = '_MLP_'
# # model = '_VAE_conditional_'
# model = '_RNN_new_'
#
# # dataset
# # dataset = 'barabasi'
# # dataset = 'barabasi_small'
# dataset = 'caveman'
# # dataset = 'grid'
# # dataset = 'citeseer'
# # dataset = 'citeseer_small'
# # dataset = 'DD'
# # dataset = 'grid_small'
# # dataset = 'enzymes'
# # dataset = 'enzymes_small'
#
# sample_time = 1
#
# if 'small' in dataset:
#     hidden = '64'
# else:
#     hidden = '128'
#
# fname_real = 'GraphRNN'+model+dataset+'_4_'+hidden+'_test_0' #real
# fname_pred = 'GraphRNN'+model+dataset+'_4_'+hidden+'_pred_3000_'+str(sample_time) # pred
#
# # fname_real = 'GraphRNN'+model+dataset+'_4_'+hidden+'_0'+'_test_0' #real
# # fname_pred = 'GraphRNN'+model+dataset+'_4_'+hidden+'_0'+'_pred_3000_'+str(sample_time) # pred
#
# # fname_real = 'Baseline_DGMG_grid_small_64_test_0'
# # fname_pred = 'Baseline_DGMG_grid_small_64_pred_2000'
#
# graphs_real = load_graph_list(dir+fname_real+'.dat',is_real=True)
# graphs_pred = load_graph_list(dir+fname_pred+'.dat',is_real=False)
#
# graphs_real = graphs_real[int(len(graphs_real)*0.8):]
# graphs_real = [graphs_real[i] for i in range(len(graphs_real)) if (graphs_real[i].number_of_nodes()>=50 and graphs_real[i].number_of_nodes()<=500)]
# graphs_pred = [graphs_pred[i] for i in range(len(graphs_pred)) if (graphs_pred[i].number_of_nodes()>=50 and graphs_pred[i].number_of_nodes()<=500)]
#
# # shuffle(graphs_real)
# # shuffle(graphs_pred)
#
# graphs_real_len = [graph.number_of_nodes() for graph in graphs_real]
# graphs_pred_len = [graph.number_of_nodes() for graph in graphs_pred]
# print(graphs_real_len)
# print(graphs_pred_len)
#
# ######### for grid only##############
# # graphs_pred_new = []
# # for real_len in graphs_real_len:
# #     if real_len in graphs_pred_len:
# #         id = graphs_pred_len.index(real_len)
# #         print('Found!!',real_len,id)
# #         graphs_pred_new.append(graphs_pred[id])
#
# # for pred_len in graphs_pred_len:
# #     if pred_len<100:
# #         id = graphs_pred_len.index(pred_len)
# #         graphs_pred_new.append(graphs_pred[id])
#
# # graphs_pred = graphs_pred_new
# ######################################
#
#
# print('real graph count',len(graphs_real))
# print('pred graph count',len(graphs_pred))
#
#
#
# ### plot sample graphs
# draw_graph_list(graphs_real[0:16],4,4,'figures/test_graph'+fname_real+'.png',layout='spring')
# draw_graph_list(graphs_pred[0:16],4,4,'figures/test_graph'+fname_pred+'.png',layout='spring')
# #
# #
# #
# # ### print distribution of graph size
# # plt.switch_backend('agg')
# # plt.hist(graphs_real_len)
# # plt.savefig('figures/test_len'+fname_real+'.png', dpi=200)
# # plt.close()
# #
# # plt.switch_backend('agg')
# # plt.hist(graphs_pred_len)
# # plt.savefig('figures/test_len'+fname_pred+'.png', dpi=200)
# # plt.close()
# #
# # ### print average clustering
# # plt.switch_backend('agg')
# # graphs_clustering_real = []
# # for graph in graphs_real:
# #     graphs_clustering_real.extend(list(nx.clustering(graph).values()))
# # bins = np.linspace(0,1,50)
# # plt.hist(np.array(graphs_clustering_real), bins=bins, align='left')
# # plt.savefig('figures/test_clustering'+fname_real+'.png', dpi=200)
# #
# # plt.switch_backend('agg')
# # graphs_clustering_pred = []
# # for graph in graphs_pred:
# #     graphs_clustering_pred.extend(list(nx.clustering(graph).values()))
# # bins = np.linspace(0,1,50)
# # plt.hist(np.array(graphs_clustering_pred), bins=bins, align='left')
# # plt.savefig('figures/test_clustering'+fname_pred+'.png', dpi=200)
# #
# #
# # ### print average degree
# # plt.switch_backend('agg')
# # graphs_degree_real = []
# # for graph in graphs_real:
# #     graphs_degree_real.extend(list(graph.degree(graph.nodes()).values()))
# # bins = np.linspace(0,40,40)
# # plt.hist(np.array(graphs_degree_real), bins=bins, align='left')
# # plt.savefig('figures/test_degree'+fname_real+'.png', dpi=200)
# #
# # plt.switch_backend('agg')
# # graphs_degree_pred= []
# # for graph in graphs_pred:
# #     graphs_degree_pred.extend(list(graph.degree(graph.nodes()).values()))
# # bins = np.linspace(0,40,40)
# # plt.hist(np.array(graphs_degree_pred), bins=bins, align='left')
# # plt.savefig('figures/test_degree'+fname_pred+'.png', dpi=200)
# #
# #
#








# c = 2
# k = 10
# p = 0.1
# path_count = int(np.ceil(p*k))
# G = nx.caveman_graph(c,k)
#
# # remove 50% edges
# p = 0.5
# for (u, v) in list(G.edges()):
#     if np.random.rand()<p and ((u<k and v<k)or(u>=k and v>=k)):
#         G.remove_edge(u, v)
# # add path_count links
# for i in range(path_count):
#     u = np.random.randint(0,k)
#     v = np.random.randint(k,k*2)
#     G.add_edge(u,v)
#
# draw_graph(G, 'caveman_sparse'+str(k))



# dir = '/dfs/scratch0/jiaxuany0/graphs/'
# name = 'GraphRNN_VAE_conditional_citeseer_small_4_64_pred_3000_3.dat'
# graphs = load_graph_list(dir+name)
# print(len(graphs))














