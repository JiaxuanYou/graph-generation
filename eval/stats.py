import concurrent.futures
from datetime import datetime
from functools import partial
import numpy as np
import networkx as nx
import pickle as pkl

import eval.mmd as mmd

PRINT_TIME = False

def degree_worker(G):
    return np.array(nx.degree_histogram(G))

def degree_stats(graph_ref_list, graph_pred_list, is_parallel=False):
    ''' Compute the distance between the degree distributions of two unordered sets of graphs.
    Args:
      graph_ref_list, graph_target_list: two lists of networkx graphs to be evaluated
    '''
    sample_ref = []
    sample_pred = []
    graph_pred_list_remove_empty = [G for G in graph_pred_list if not G.number_of_nodes() == 0]

    prev = datetime.now()
    if is_parallel:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for deg_hist in executor.map(degree_worker, graph_ref_list):
                sample_ref.append(deg_hist)
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for deg_hist in executor.map(degree_worker, graph_pred_list_remove_empty):
                sample_pred.append(deg_hist)

    else:
        for i in range(len(graph_ref_list)):
            sample_ref.append(np.array(nx.degree_histogram(graph_ref_list[i])))
            # in case an empty graph is generated
            sample_pred.append(np.array(nx.degree_histogram(graph_pred_list[i])))
    mmd_dist = mmd.compute_mmd(sample_ref, sample_pred, kernel=mmd.gaussian_emd)
    elapsed = datetime.now() - prev
    if PRINT_TIME:
        print('Time computing degree mmd: ', elapsed)
    return mmd_dist

def clustering_worker(param):
    G, bins = param
    clustering_coeffs_list = list(nx.clustering(G).values())
    hist, _ = np.histogram(
            clustering_coeffs_list, bins=bins, range=(0.0, 1.0), density=True)
    return hist

def clustering_stats(graph_ref_list, graph_pred_list, bins=100, is_parallel=True):
    sample_ref = []
    sample_pred = []
    graph_pred_list_remove_empty = [G for G in graph_pred_list if not G.number_of_nodes() == 0]

    prev = datetime.now()
    if is_parallel:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for clustering_hist in executor.map(clustering_worker, 
                    [(G, bins) for G in graph_ref_list]):
                sample_ref.append(clustering_hist)
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for clustering_hist in executor.map(clustering_worker, 
                    [(G, bins) for G in graph_pred_list_remove_empty]):
                sample_pred.append(clustering_hist)
    else:
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
    elapsed = datetime.now() - prev
    if PRINT_TIME:
        print('Time computing clustering mmd: ', elapsed)
    return mmd_dist

