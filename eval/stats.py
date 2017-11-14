import numpy as np
import networkx as nx
import pickle as pkl

def degree_stats(graph_ref_list, graph_target_list):
    ref_deg = np.array(nx.degree_histogram(G))
    target_deg = np.array(nx.degree_histogram(G))
    deg_dist_mmd = mmd(ref_deg, target_deg, kernel=gaussian_emd, sigma=1)


