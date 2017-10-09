import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import os

# G = nx.karate_club_graph()
G = nx.LCF_graph(6,[3,-3],3)

print(G.nodes())

for i in range(len(G.nodes())):
    print(list(G.adj[i]))

# clean saving directory
if not os.path.exists("figures"):
    os.makedirs("figures")

plt.subplot(1)
plt.switch_backend('agg')
nx.draw(G, with_labels=True, font_weight='bold')
plt.savefig('figures/network_view.png')

# for lr in [0.01]:
#     for hidden_size in [2]:
#         for run in range(5):
#             embedding = np.load('saves/embedding_lr_'+str(lr)+'_hidden_'+str(hidden_size)+'_run_'+str(run)+'.npy')[3:3+6]
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
