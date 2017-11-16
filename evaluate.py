import numpy as np
import os
import pickle
import re

import eval.stats

def load_graph_list(fname):
    with open(fname, "rb") as f:
        list = pickle.load(f)
    return list

def extract_result_id(name, after_word):
    pos = name.find(after_word) + len(after_word)
    end_pos = name.find('.dat')
    return name[pos:end_pos]

def perturb(graph_list, p_add, p_del):
    ''' Perturb the list of graphs by adding/removing edges.
    Args:
        p_add: probability of adding edges
        p_del: probability of removing edges
    Returns:
        A list of graphs that are perturbed from the original graphs
    '''
    perturbed_graph_list = []
    for G_original in graph_list:
        G = G_original.copy()
        trials = np.random.binomial(1, p_del, size=G.number_of_edges())
        i = 0
        edges = G.edges()
        for (u, v) in edges:
            if trials[i] == 1:
                G.remove_edge(u, v)
            i += 1

        nodes = G.nodes()
        for i in range(len(nodes)):
            u = nodes[i]
            trials = np.random.binomial(1, p_add, size=G.number_of_nodes())
            j = 0
            for v in nodes:
                if trials[j] == 1 and not i == j:
                    G.add_edge(u, v)
                j += 1

        perturbed_graph_list.append(G)
    return perturbed_graph_list

def eval_list(real_graphs_filename, pred_graphs_filename):
    real_graphs_dict = {}
    pred_graphs_dict = {}

    for fname in real_graphs_filename:
        real_graphs_dict[extract_result_id(fname, 'real_')] = fname
    for fname in pred_graphs_filename:
        pred_graphs_dict[extract_result_id(fname, 'pred_')] = fname
    
    for result_id in real_graphs_dict.keys():
        real_g_list = load_graph_list(real_graphs_dict[result_id])
        pred_g_list = load_graph_list(pred_graphs_dict[result_id])
        perturbed_g_list = perturb(real_g_list, 0.05, 0.05)

        dist = eval.stats.degree_stats(real_g_list, pred_g_list)
        print('dist between real and pred (', result_id, '): ', dist)
    
        dist = eval.stats.degree_stats(real_g_list, perturbed_g_list)
        print('dist between real and perturbed: ', dist)


if __name__ == '__main__':
    datadir = "/dfs/scratch0/rexy/graph_gen_data/"
    prefix = "GraphRNN_enzymes_50_26000_"
    real_graphs_filename = [datadir + f for f in os.listdir(datadir) 
            if re.match(prefix + 'real.*\.dat', f)]
    pred_graphs_filename = [datadir + f for f in os.listdir(datadir) 
            if re.match(prefix + 'pred.*\.dat', f)]
    eval_list(real_graphs_filename, pred_graphs_filename)
    

