import networkx as nx
import matplotlib.pyplot as plt


G = nx.karate_club_graph()

print(len(G.nodes()))
print(G.edges())

print(list(G.adj[0]))
