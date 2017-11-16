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
        sample_pred.append(np.array(nx.degree_histogram(graph_pred_list[i])))
    mmd_dist = mmd.compute_mmd(sample_ref, sample_pred, kernel=mmd.gaussian_emd)
    return mmd_dist


