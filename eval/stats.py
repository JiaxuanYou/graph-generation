import numpy as np
import networkx as nx
import pickle as pkl

import eval.mmd as mmd

def degree_stats(graph_ref_list, graph_pred_list):
    ''' Compute the distance between the degree distributions of two unordered sets of graphs.
    Args:
      graph_ref_list, graph_target_list: two lists of networkx graphs to be evaluated
    '''
    sample_ref = []
    sample_pred = []
    for i in range(len(graph_ref_list)):
        sample_ref.append(np.array(nx.degree_histogram(graph_ref_list[i])))

        # in case an empty graph is generated
        if not graph_pred_list[i].number_of_nodes() == 0:
            sample_pred.append(np.array(nx.degree_histogram(graph_pred_list[i])))
    mmd_dist = mmd.compute_mmd(sample_ref, sample_pred, kernel=mmd.gaussian_emd)
    return mmd_dist

def clustering_stats(graph_ref_list, graph_pred_list, bins=100):
    sample_ref = []
    sample_pred = []
    for i in range(len(graph_ref_list)):
        clustering_coeffs_list = list(nx.clustering(graph_ref_list[i]).values())
        hist, _ = np.histogram(
                clustering_coeffs_list, bins=bins, range=(0.0, 1.0), density=True)
        sample_ref.append(hist)

        clustering_coeffs_list = list(nx.clustering(graph_pred_list[i]).values())
        hist, _ = np.histogram(
                clustering_coeffs_list, bins=bins, range=(0.0, 1.0), density=True)
        sample_pred.append(hist)
    
    mmd_dist = mmd.compute_mmd(sample_ref, sample_pred, kernel=mmd.gaussian_emd,
                               sigma=1.0, distance_scaling=2.0/bins)
    return mmd_dist

