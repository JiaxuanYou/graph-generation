import argparse
import numpy as np
import os
import pickle
import re
from random import shuffle

import eval.stats
# import main.Args
from main import *


# load a list of graphs
def load_graph_list(fname):
    with open(fname, "rb") as f:
        graph_list = pickle.load(f)
    for i in range(len(graph_list)):
        graph_list[i] = nx.convert_node_labels_to_integers(graph_list[i])
        # print(graph_list[i].nodes())
        # graph_list[i].remove_nodes_from(list(nx.isolates(graph_list[i])))
        graph_list[i] = pick_connected_component(graph_list[i])
    return graph_list

def find_nearest_idx(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx

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


def perturb(graph_list, p_del, p_add=None):
    ''' Perturb the list of graphs by adding/removing edges.
    Args:
        p_add: probability of adding edges. If None, estimate it according to graph density,
            such that the expected number of added edges is equal to that of deleted edges.
        p_del: probability of removing edges
    Returns:
        A list of graphs that are perturbed from the original graphs
    '''
    perturbed_graph_list = []
    for G_original in graph_list:
        G = G_original.copy()
        trials = np.random.binomial(1, p_del, size=G.number_of_edges())
        i = 0
        edges = list(G.edges())
        for (u, v) in edges:
            if trials[i] == 1:
                G.remove_edge(u, v)
            i += 1

        nodes = list(G.nodes())
        for i in range(len(nodes)):
            u = nodes[i]
            if p_add is None:
                num_nodes = G.number_of_nodes()
                p_add_est = p_del * 2 * G.number_of_edges() / (num_nodes * (num_nodes - 1))
            else:
                p_add_est = p_add
            trials = np.random.binomial(1, p_add_est, size=G.number_of_nodes())
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
            perturbed_g_list = perturb(real_g_list, 0.05)

            # dist = eval.stats.degree_stats(real_g_list, pred_g_list)
            dist = eval.stats.clustering_stats(real_g_list, pred_g_list)
            print('dist between real and pred (', result_id, ') at epoch ', epochs, ': ', dist)

            # dist = eval.stats.degree_stats(real_g_list, perturbed_g_list)
            dist = eval.stats.clustering_stats(real_g_list, perturbed_g_list)
            print('dist between real and perturbed: ', dist)

            mid = len(real_g_list) // 2
            # dist = eval.stats.degree_stats(real_g_list[:mid], real_g_list[mid:])
            dist = eval.stats.clustering_stats(real_g_list[:mid], real_g_list[mid:])
            print('dist among real: ', dist)


def eval_list_fname(real_graph_filename, pred_graphs_filename, eval_every, out_file_prefix=None):
    if out_file_prefix is not None:
        out_files = {
            'train': open(out_file_prefix + '_train.txt', 'w'),
            'compare': open(out_file_prefix + '_compare.txt', 'w')
        }

    out_files['train'].write('degree,clustering\n')
    out_files['compare'].write('metric,real,ours,perturbed\n')

    results = {
        'deg': {
            'real': 1,
            'ours': 1,
            'perturbed': 1},
        'clustering': {
            'real': 1,
            'ours': 1,
            'perturbed': 1}
    }
    num_graphs = len(pred_graphs_filename)
    for i in range(num_graphs):
        # load data
        real_g_list = load_graph_list(real_graph_filename)
        pred_g_list_raw = load_graph_list(pred_graphs_filename[i])
        shuffle(real_g_list)
        if len(real_g_list)>200:
            real_g_list = real_g_list[0:200]
        shuffle(pred_g_list_raw)
        # get length
        real_g_len_list = np.array([len(real_g_list[i]) for i in range(len(real_g_list))])
        pred_g_len_list_raw = np.array([len(pred_g_list_raw[i]) for i in range(len(pred_g_list_raw))])
        # get perturb real
        perturbed_g_list_001 = perturb(real_g_list, 0.01)
        perturbed_g_list_005 = perturb(real_g_list, 0.05)
        perturbed_g_list_010 = perturb(real_g_list, 0.10)


        # select pred samples
        pred_g_list = []
        pred_g_len_list = []
        for value in real_g_len_list:
            pred_idx = find_nearest_idx(pred_g_len_list_raw, value)
            pred_g_list.append(pred_g_list_raw[pred_idx])
            pred_g_len_list.append(pred_g_len_list_raw[pred_idx])
            # delete
            pred_g_len_list_raw = np.delete(pred_g_len_list_raw, pred_idx)
            del pred_g_list_raw[pred_idx]
            if len(pred_g_list) == len(real_g_list):
                break
        # pred_g_len_list = np.array(pred_g_len_list)
        print('################## epoch {} ##################'.format(i*eval_every))

        print('real average nodes',
              sum([real_g_list[i].number_of_nodes() for i in range(len(real_g_list))]) / len(real_g_list))
        print('pred average nodes',
              sum([pred_g_list[i].number_of_nodes() for i in range(len(pred_g_list))]) / len(pred_g_list))
        print('num of real graphs', len(real_g_list))
        print('num of pred graphs', len(pred_g_list))


        mid = len(real_g_list) // 2
        dist_degree = eval.stats.degree_stats(real_g_list[:mid], real_g_list[mid:])
        dist_clustering = eval.stats.clustering_stats(real_g_list[:mid], real_g_list[mid:])
        print('degree dist among real: ', dist_degree)
        print('clustering dist among real: ', dist_clustering)
        results['deg']['real'] += dist_degree
        results['clustering']['real'] += dist_clustering

        dist_degree = eval.stats.degree_stats(real_g_list, pred_g_list)
        dist_clustering = eval.stats.clustering_stats(real_g_list, pred_g_list)
        print('degree dist between real and pred at epoch ', i*eval_every, ': ', dist_degree)
        print('clustering dist between real and pred at epoch ', i * eval_every, ': ', dist_clustering)
        out_files['train'].write(str(dist_degree) + ',')
        out_files['train'].write(str(dist_clustering) + ',')
        results['deg']['ours'] = min(dist_degree, results['deg']['ours'])
        results['clustering']['ours'] = min(dist_clustering, results['clustering']['ours'])

        # dist_degree = eval.stats.degree_stats(real_g_list, perturbed_g_list_001)
        # dist_clustering = eval.stats.clustering_stats(real_g_list, perturbed_g_list_001)
        # print('degree dist between real and perturbed 0.01: ', dist_degree)
        # print('clustering dist between real and perturbed 0.01: ', dist_clustering)
        #
        # dist_degree = eval.stats.degree_stats(real_g_list, perturbed_g_list_010)
        # dist_clustering = eval.stats.clustering_stats(real_g_list, perturbed_g_list_010)
        # print('degree dist between real and perturbed 0.10: ', dist_degree)
        # print('clustering dist between real and perturbed 0.10: ', dist_clustering)
        #
        # dist_degree = eval.stats.degree_stats(real_g_list, perturbed_g_list_005)
        # dist_clustering = eval.stats.clustering_stats(real_g_list, perturbed_g_list_005)
        # print('degree dist between real and perturbed 0.05: ', dist_degree)
        # print('clustering dist between real and perturbed 0.05: ', dist_clustering)
        #
        # results['deg']['perturbed'] += dist_degree
        # results['clustering']['perturbed'] += dist_clustering

        out_files['train'].write('\n')

    for metric, methods in results.items():
        methods['real'] /= num_graphs
        methods['perturbed'] /= num_graphs

    for metric, methods in results.items():
        out_files['compare'].write(metric + ',' +
                                   str(methods['real']) + ',' +
                                   str(methods['ours']) + ',' +
                                   str(methods['perturbed']) + '\n')

    for _, file in out_files.items():
        file.close()


def eval_performance(datadir, prefix=None, args=None, eval_every=200, out_file_prefix=None):
    if args is None:
        real_graphs_filename = [datadir + f for f in os.listdir(datadir)
                                if re.match(prefix + '.*real.*\.dat', f)]
        pred_graphs_filename = [datadir + f for f in os.listdir(datadir)
                                if re.match(prefix + '.*pred.*\.dat', f)]
        eval_list(real_graphs_filename, pred_graphs_filename, prefix, 200)

    else:
        real_graph_filename = datadir+args.graph_save_path + args.fname_real + '0.dat'
        # for proposed model
        pred_graphs_filename = [datadir+args.graph_save_path + args.fname_pred+str(epoch)+'.dat' for epoch in range(eval_every,3001,eval_every)]
        # for baseline model
        #pred_graphs_filename = [datadir+args.fname_baseline+'.dat']
        print(pred_graphs_filename)

        eval_list_fname(real_graph_filename, pred_graphs_filename, eval_every=eval_every,
                        out_file_prefix=out_file_prefix)


def export_graphs_to_txt(filename, output_filename_prefix):
    g_list = load_graph_list(filename)
    i = 0
    for G in g_list:
        f = open(output_filename_prefix + '_' + str(i) + '.txt', 'w')
        for (u, v) in G.edges():
            f.write(str(u) + '\t' + str(v) + '\n')
        i += 1


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='Evaluation arguments.')
    # feature_parser = parser.add_mutually_exclusive_group(required=False)
    # feature_parser.add_argument('--export-real', dest='export', action='store_true')
    # feature_parser.add_argument('--no-export-real', dest='export', action='store_false')
    # parser.set_defaults(export=False)
    # prog_args = parser.parse_args()

    # datadir = "/dfs/scratch0/rexy/graph_gen_data/"
    # datadir = "/lfs/local/0/jiaxuany/pycharm/graphs_share/"
    datadir = "/lfs/local/0/jiaxuany/pycharm/"
    # prefix = "GraphRNN_enzymes_50_"
    # prefix = "GraphRNN_structure_enzymes_50_"
    args = Args()

    # if prog_args.export:
    #     filename = args.graph_save_path + args.note + '_' + args.graph_type + '_' + \
    #                str(0) + '_real_' + str(args.num_layers) + '_' + str(args.bptt) + '_' + str(
    #         args.bptt_len) + '_' + str(args.gumbel)
    #     input_path = datadir + filename + '.dat'
    #     if not os.path.isdir('eval_results'):
    #         os.makedirs('eval_results')
    #     if not os.path.isdir('eval_results/ground_truth'):
    #         os.makedirs('eval_results/ground_truth')
    #     output_prefix = 'eval_results/ground_truth/' + args.graph_type
    #     print('Export ground truth to ', output_prefix)
    #     export_graphs_to_txt(input_path, output_prefix)

    print(args.fname)
    out_file_prefix = 'eval_results/' + args.fname
    if not os.path.isdir('eval_results'):
        os.makedirs('eval_results')
    eval_performance(datadir, args=args, out_file_prefix=out_file_prefix)

