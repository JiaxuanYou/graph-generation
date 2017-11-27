import numpy as np
import os
import pickle
import re
from random import shuffle


import eval.stats
# import main.Args
from main import *

def load_graph_list(fname):
    with open(fname, "rb") as f:
        list = pickle.load(f)
    return list

def extract_result_id_and_epoch(name, prefix, after_word):
    '''
    Args:
        eval_every: the number of epochs between consecutive evaluations
    Returns:
        A tuple of (id, epoch number)
    '''
    pos = name.find(after_word) + len(after_word)
    end_pos = name.find('.dat')
    result_id = name[pos:end_pos]

    pos = name.find(prefix) + len(prefix)
    end_pos = name.find('_', pos)
    epochs = int(name[pos:end_pos])
    return result_id, epochs

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

def eval_list(real_graphs_filename, pred_graphs_filename, prefix, eval_every):
    real_graphs_dict = {}
    pred_graphs_dict = {}

    for fname in real_graphs_filename:
        result_id, epochs = extract_result_id_and_epoch(fname, prefix, 'real_')
        if not epochs % eval_every == 0:
            continue
        if result_id not in real_graphs_dict:
            real_graphs_dict[result_id] = {}
        real_graphs_dict[result_id][epochs] = fname
    for fname in pred_graphs_filename:
        result_id, epochs = extract_result_id_and_epoch(fname, prefix, 'pred_')
        if not epochs % eval_every == 0:
            continue
        if result_id not in pred_graphs_dict:
            pred_graphs_dict[result_id] = {}
        pred_graphs_dict[result_id][epochs] = fname
    
    for result_id in real_graphs_dict.keys():
        for epochs in sorted(real_graphs_dict[result_id]):
            real_g_list = load_graph_list(real_graphs_dict[result_id][epochs])
            pred_g_list = load_graph_list(pred_graphs_dict[result_id][epochs])
            shuffle(real_g_list)
            shuffle(pred_g_list)
            perturbed_g_list = perturb(real_g_list, 0.001, 0.02)

            #dist = eval.stats.degree_stats(real_g_list, pred_g_list)
            dist = eval.stats.clustering_stats(real_g_list, pred_g_list)
            print('dist between real and pred (', result_id, ') at epoch ', epochs, ': ', dist)
    
            #dist = eval.stats.degree_stats(real_g_list, perturbed_g_list)
            dist = eval.stats.clustering_stats(real_g_list, perturbed_g_list)
            print('dist between real and perturbed: ', dist)

            mid = len(real_g_list) // 2
            #dist = eval.stats.degree_stats(real_g_list[:mid], real_g_list[mid:])
            dist = eval.stats.clustering_stats(real_g_list[:mid], real_g_list[mid:])
            print('dist among real: ', dist)


def eval_list_fname(real_graphs_filename, pred_graphs_filename, eval_every):
    for i in range(len(real_graphs_filename)):
        real_g_list = load_graph_list(real_graphs_filename[i])
        pred_g_list = load_graph_list(pred_graphs_filename[i])
        shuffle(real_g_list)
        shuffle(pred_g_list)
        perturbed_g_list = perturb(real_g_list, 0.001, 0.02)

        dist_degree = eval.stats.degree_stats(real_g_list, pred_g_list)
        dist_clustering = eval.stats.clustering_stats(real_g_list, pred_g_list)
        print('degree dist between real and pred at epoch ', i*eval_every, ': ', dist_degree)
        print('clustering dist between real and pred at epoch ', i*eval_every, ': ', dist_clustering)


        dist_degree = eval.stats.degree_stats(real_g_list, perturbed_g_list)
        dist_clustering = eval.stats.clustering_stats(real_g_list, perturbed_g_list)
        print('degree dist between real and perturbed: ', dist_degree)
        print('clustering dist between real and perturbed: ', dist_clustering)


        mid = len(real_g_list) // 2
        dist_degree = eval.stats.degree_stats(real_g_list[:mid], real_g_list[mid:])
        dist_clustering = eval.stats.clustering_stats(real_g_list[:mid], real_g_list[mid:])
        print('degree dist among real: ', dist_degree)
        print('clustering dist among real: ', dist_clustering)


def eval_performance(datadir, prefix, args=None,eval_every=500):
    if args is None:
        real_graphs_filename = [datadir + f for f in os.listdir(datadir)
                if re.match(prefix + '.*real.*\.dat', f)]
        pred_graphs_filename = [datadir + f for f in os.listdir(datadir)
                if re.match(prefix + '.*pred.*\.dat', f)]
        eval_list(real_graphs_filename, pred_graphs_filename, prefix, 200)

    else:
        real_graphs_filename = [datadir + args.graph_save_path + args.note + '_' + args.graph_type + '_' + \
                     str(epoch) + '_pred_' + str(args.num_layers) + '_' + str(args.bptt) + '_' + str(args.bptt_len) + '.dat' for epoch in range(0,50001,eval_every)]
        pred_graphs_filename = [datadir + args.graph_save_path + args.note + '_' + args.graph_type + '_' + \
                 str(epoch) + '_real_' + str(args.num_layers) + '_' + str(args.bptt) + '_' + str(args.bptt_len) + '.dat' for epoch in range(0,50001,eval_every)]
        eval_list_fname(real_graphs_filename, pred_graphs_filename,eval_every=eval_every)


if __name__ == '__main__':
    #datadir = "/dfs/scratch0/rexy/graph_gen_data/"
    #prefix = "GraphRNN_enzymes_50_"
    #datadir = "/lfs/local/0/jiaxuany/pycharm/graphs_share/"
    datadir = "/lfs/local/0/jiaxuany/pycharm/"
    #prefix = "GraphRNN_enzymes_50_"
    prefix = "GraphRNN_structure_enzymes_50_"
    # eval_performance(datadir, prefix)
    args = Args()
    print(args.graph_type)
    eval_performance(datadir, prefix,args)
    

