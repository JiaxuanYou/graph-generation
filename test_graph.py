import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import os
import torch
import math

G = nx.karate_club_graph()
# G = nx.LCF_graph(6,[3,-3],3)
# G = nx.LCF_graph(14,[5,-5],7)
G = nx.Graph()

# for edge in G.edges():
#     G[edge[0]][edge[1]]['weight'] = 1

# print(G.edges())
# print(G[0][13])

np.random.seed(10)
base = np.repeat(np.eye(5),20,axis=0)
rand = np.random.randn(100,5)*0.05
node_features = base+rand

# print(node_features, node_features.shape)

node_distance_np = np.zeros((node_features.shape[0], node_features.shape[0]))
for i in range(node_features.shape[0]):
    for j in range(node_features.shape[0]):
        if i!= j:
            node_distance_np[i,j] = 1/np.sum(np.abs(node_features[i]-node_features[j])**2)

node_distance_np_sum = np.sum(node_distance_np, axis = 1, keepdims=True)
embedding_dist = node_distance_np/node_distance_np_sum
# print(embedding_dist[0])
# print(np.sum(embedding_dist[0]))



# 0 generate the graph
average_degree = 9
for i in range(node_features.shape[0]):
    for j in range(i+1, embedding_dist.shape[0]):
        p = np.random.rand()
        if p<embedding_dist[i,j]*average_degree:
            G.add_edge(i,j)

print('num of nodes',G.number_of_nodes())
print('num of edges',G.number_of_edges())
G.remove_nodes_from(nx.isolates(G))


plt.switch_backend('agg')
options = {
    'node_color': 'black',
    'node_size': 10,
    'width': 1,
}
plt.figure()
plt.subplot()
nx.draw_networkx(G, **options)
plt.savefig('figures/graph_synthetic_view.png', dpi = 300)
plt.close()


# 1 degree distribution
plt.switch_backend('agg')
G_deg = nx.degree_histogram(G)
# print('original', G_deg, sum(G_deg)/len(G_deg))
plt.plot(range(len(G_deg)), G_deg, 'r', linewidth = 2)

plt.savefig('figures/graph_synthetic_degree.png')
plt.close()
print(G_deg)

G_deg_sum = [a*b for a,b in zip(G_deg,range(0,len(G_deg)))]
print('average degree', sum(G_deg_sum)/G.number_of_nodes())

# 2 path length
print('average path length', nx.average_shortest_path_length(G))
print('diameter', nx.diameter(G))

# 3 clustering coefficient
plt.switch_backend('agg')
G_cluster = sorted(list(nx.clustering(G).values()))

print('average clustering coefficient',  sum(G_cluster)/len(G_cluster))
# plt.plot(range(len(G_cluster)), G_cluster, 'r', linewidth = 2)
plt.hist(G_cluster, bins = 10)
plt.savefig('figures/graph_synthetic_clustering_coefficient.png')
plt.close()



# idx_list = [3]
# node_list = []
# node_count_list = []
#
# # no padding
# for i in range(3):
#     # sample this hop
#     adj_list = np.array([])
#     adj_count_list = np.array([])
#     for idx in idx_list:
#         adj_list_new = np.array(list(G.adj[idx]))
#         adj_count_list_new = np.array([len(adj_list_new)])
#         adj_list = np.concatenate((adj_list,adj_list_new),axis=0)
#         adj_count_list = np.concatenate((adj_count_list, adj_count_list_new),axis=0)
#     node_list.append(adj_list)
#     node_count_list.append(adj_count_list)
#     idx_list = adj_list
#
# for adj_list in node_list:
#     print(adj_list.shape)
#     print(adj_list)
# for adj_count_list in node_count_list:
#     print(adj_count_list.shape)
#     print(adj_count_list)




# # padding
# max_degree = 5
# idx_list = [3]
# node_list = []
# node_count_list = []
# for i in range(3):
#     adj_list = np.ones(max_degree**(i+1))*-1
#     adj_count_list = np.ones(max_degree**(i))*max_degree
#     for j, idx in enumerate(idx_list):
#         if idx==-1:
#             adj_list_new = np.ones(max_degree)*-1
#         else:
#             adj_list_new = np.array(list(G.adj[idx]))
#         start_idx = j*max_degree
#         incre_idx = min(max_degree, adj_list_new.shape[0])
#         adj_list[start_idx:start_idx+incre_idx] = adj_list_new[:incre_idx]
#     node_list.append(adj_list)
#     node_count_list.append(adj_count_list)
#     idx_list = adj_list
#
# for adj_list in node_list:
#     print(adj_list.shape)
#     print(adj_list)
# for adj_count_list in node_count_list:
#     print(adj_count_list.shape)
#     print(adj_count_list)








# for i in range(len(G.nodes())):
#     print(list(G.adj[i]))
#
# # clean saving directory
# if not os.path.exists("figures"):
#     os.makedirs("figures")
#
# # plt.subplot(1)
# # plt.switch_backend('agg')
# # nx.draw(G, with_labels=True, font_weight='bold')
# # plt.savefig('figures/network_view.png')
#
#
#
# for lr in [0.001]:
#     for hidden_size in [2]:
#         for run in range(1):
#             embedding = np.load('saves/embedding_lr_'+str(lr)+'_hidden_'+str(hidden_size)+'_run_'+str(run)+'.npy')[3:3+14]
#             print(embedding)
#             embedded = TSNE(n_components=2).fit_transform(embedding)
#             print(embedded)
#             plt.switch_backend('agg')
#             plt.scatter(embedded[:,0], embedded[:,1])
#             for i in range(embedded.shape[0]):
#                 plt.text(embedded[i, 0], embedded[i, 1], str(i))
#                 for j in list(G.adj[i]):
#                     plt.plot([embedded[i,0],embedded[j,0]],[embedded[i,1],embedded[j,1]],
#                              color = 'r', linewidth = 0.5)
#             plt.savefig('figures/embedding_lr_'+str(lr)+'_hidden_'+str(hidden_size)+'_run_'+str(run)+'.png')
