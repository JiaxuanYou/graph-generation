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
        print(real_graphs_dict[result_id])
        print(pred_graphs_dict[result_id])
        dist = eval.stats.degree_stats(real_g_list, pred_g_list)
        print(len(real_g_list))
        print(len(pred_g_list))
        print(dist)
    


if __name__ == '__main__':
    datadir = "/dfs/scratch0/rexy/graph_gen_data/"
    prefix = "GraphRNN_enzymes_50_26000_"
    real_graphs_filename = [datadir + f for f in os.listdir(datadir) 
            if re.match(prefix + 'real.*\.dat', f)]
    pred_graphs_filename = [datadir + f for f in os.listdir(datadir) 
            if re.match(prefix + 'pred.*\.dat', f)]
    eval_list(real_graphs_filename, pred_graphs_filename)
    

