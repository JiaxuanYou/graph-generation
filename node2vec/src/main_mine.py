from main import *

G = nx.karate_club_graph()
for edge in G.edges():
    G[edge[0]][edge[1]]['weight'] = 1

embedding = node2vec_main(G, args)

print(embedding)