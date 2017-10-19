from train_rnn import *
import numpy as np
import copy
import torch.multiprocessing as mp
# from node2vec.src.main import *
import node2vec.src.main as nv
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA






# model configuration
# hidden_size = 16 # hidden vector size for a single GRU layer
input_size = 128 # embedding vector size for each node
# embedding_size = 3+14 # the number of embedding vocabulary
n_layers = 1
# train configuration
# lr = 0.01
############# node2vec config###############
args = nv.config(dimension=input_size, walk_length = 80, num_walks = 10, window_size = 2)
############################################

# clean logging directory
if os.path.isdir("logs"):
    shutil.rmtree("logs")
configure("logs/logs_toy", flush_secs=1)

# clean saving directory
if not os.path.exists("saves"):
    os.makedirs("saves")

# Generate Graph
G = nx.karate_club_graph()
# G = nx.LCF_graph(14,[5,-5],7)
# G = nx.LCF_graph(20,[-9,-9],10)

graphdataset = GraphDataset(G, shuffle_neighbour = True)
# run node2vec
for edge in G.edges():
    G[edge[0]][edge[1]]['weight'] = 1
embedding = nv.node2vec_main(G, args)
# print(embedding)
print('embedding.shape', embedding.shape)

embedding_dist = np.zeros((embedding.shape[0],embedding.shape[0]))

G_adj = np.asarray(nx.to_numpy_matrix(G))
# print(np.std(G_adj,axis=0)/np.mean(G_adj,axis=0))
G_adj_sum = np.repeat(np.sum(G_adj, axis = 1, keepdims=True), G_adj.shape[0] ,axis=1)

alpha = 10
print('alpha', alpha)
for i in range(embedding_dist.shape[0]):
    for j in range(embedding_dist.shape[1]):
        if i!=j:
            embedding_dist[i][j] = np.exp(embedding[i] @ embedding[j].transpose() * alpha)

embedding_dist_sum = np.repeat(np.sum(embedding_dist, axis = 1, keepdims=True), embedding_dist.shape[0] ,axis=1)
embedding_dist = embedding_dist/embedding_dist_sum
embedding_dist = embedding_dist*G_adj_sum
# print(np.std(embedding_dist, axis=0)/np.mean(embedding_dist, axis=0))
print(embedding_dist)
print(embedding_dist[0])
print(np.sum(embedding_dist[0]))



# # 1 degree distribution
# plt.switch_backend('agg')
# G_deg = nx.degree_histogram(G)
# print('original', G_deg)
# plt.plot(range(len(G_deg)), G_deg, 'r', linewidth = 2)
#
# for i in range(100):
#     G_new = nx.Graph()
#     for i in range(embedding_dist.shape[0]):
#         for j in range(i+1, embedding_dist.shape[1]):
#             rand = np.random.rand()
#             if rand<embedding_dist[i][j]:
#                 G_new.add_edge(i, j)
#     G_new.remove_nodes_from(nx.isolates(G_new))
#     G_new_deg = nx.degree_histogram(G_new)
#     print('generated', G_new_deg)
#     plt.plot(range(len(G_new_deg)), G_new_deg, 'b', linewidth = 0.5)
# plt.savefig('figures/degree_distribution'+str(alpha)+'.png')
# plt.close()





c = list(nx.k_clique_communities(G, 4))
print(c)

# embedded = TSNE(n_components=2).fit_transform(embedding)
embedded = embedding
pca = PCA(n_components=2)
embedded = pca.fit(embedding).transform(embedding)
plt.switch_backend('agg')
plt.scatter(embedded[:,0], embedded[:,1])
for i in range(embedded.shape[0]):
    plt.text(embedded[i, 0], embedded[i, 1], str(i))
    for j in list(G.adj[i]):
        plt.plot([embedded[i,0],embedded[j,0]],[embedded[i,1],embedded[j,1]],
                 color = 'r', linewidth = 0.5)
plt.savefig('figures/embedding_node2vec.png')




#
# # 2 clustering coefficient
# plt.switch_backend('agg')
# G_cluster = sorted(list(nx.clustering(G).values()))
#
# print('original',  sum(G_cluster)/len(G_cluster), G.number_of_nodes(), G.number_of_edges())
# # plt.plot(range(len(G_cluster)), G_cluster, 'r', linewidth = 2)
# plt.hist(G_cluster, bins = 10)
#
# cluster_mean = 0
# for i in range(100):
#     G_new = nx.Graph()
#     for i in range(embedding_dist.shape[0]):
#         for j in range(i+1, embedding_dist.shape[1]):
#             rand = np.random.rand()
#             if rand<embedding_dist[i][j]:
#                 G_new.add_edge(i, j)
#     G_new.remove_nodes_from(nx.isolates(G_new))
#     G_cluster_new = sorted(list(nx.clustering(G_new).values()))
#     # print('generated', sum(G_cluster_new)/len(G_cluster_new))
#     cluster_mean += sum(G_cluster_new)/len(G_cluster_new)/100
#     # plt.plot(range(len(G_cluster_new)), G_cluster_new, 'b', linewidth = 0.5)
#     # plt.hist(G_cluster_new, bins=10)
# # plt.savefig('figures/clustering_distribution'+str(alpha)+'.png')
# print('generated', cluster_mean, G_new.number_of_nodes(), G_new.number_of_edges())
# print('*******************************')
#
# cluster_mean = 0
# for i in range(100):
#     n = G.number_of_nodes()
#     e = G.number_of_edges()
#     p = 2*e/(n*(n-1))
#     G_new = nx.erdos_renyi_graph(n,p)
#     G_cluster_new = sorted(list(nx.clustering(G_new).values()))
#     # print('generated', sum(G_cluster_new)/len(G_cluster_new))
#     cluster_mean += sum(G_cluster_new) / len(G_cluster_new) / 100
#     # plt.plot(range(len(G_cluster_new)), G_cluster_new, 'b', linewidth = 0.5)
#     # plt.hist(G_cluster_new, bins=10)
# # plt.savefig('figures/clustering_distribution'+str(alpha)+'.png')
# print('generated', cluster_mean, G_new.number_of_nodes(), G_new.number_of_edges())
#
# # plt.close()
#
#
