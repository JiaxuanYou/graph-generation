import argparse
import numpy as np
import os
import pickle
import re
from random import shuffle

import eval.stats
import utils
# import main.Args
from main import *

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
            real_g_list = utils.load_graph_list(real_graphs_dict[result_id][epochs])
            pred_g_list = utils.load_graph_list(pred_graphs_dict[result_id][epochs])
            shuffle(real_g_list)
            shuffle(pred_g_list)
            perturbed_g_list = perturb(real_g_list, 0.05)

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


def compute_basic_stats(real_g_list, target_g_list):
    dist_degree = eval.stats.degree_stats(real_g_list, target_g_list)
    dist_clustering = eval.stats.clustering_stats(real_g_list, target_g_list)
    return dist_degree, dist_clustering

def eval_list_fname(real_graph_filename, pred_graphs_filename, baselines,
        eval_every, epoch_range=None, out_file_prefix=None):
    ''' Evaluate list of predicted graphs compared to ground truth, stored in files.
    Args:
        baselines: dict mapping name of the baseline to list of generated graphs.
    '''

    if out_file_prefix is not None:
        out_files = {
                'train': open(out_file_prefix + '_train.txt', 'w+'),
                'compare': open(out_file_prefix + '_compare.txt', 'w+')
        }

    out_files['train'].write('degree,clustering,orbits4\n')
    
    line = 'metric,real,ours,perturbed'
    for bl in baselines:
        line += ',' + bl
    line += '\n'
    out_files['compare'].write(line)

    results = {
            'deg': {
                    'real': 0,
                    'ours': 100, # take min over all training epochs
                    'perturbed': 0,
                    'kron': 0},
            'clustering': {
                    'real': 0,
                    'ours': 100,
                    'perturbed': 0,
                    'kron': 0},
            'orbits4': {
                    'real': 0,
                    'ours': 100,
                    'perturbed': 0,
                    'kron': 0}
    }


    num_evals = len(pred_graphs_filename)
    if epoch_range is None:
        epoch_range = [i * eval_every for i in range(num_evals)] 
    for i in range(num_evals):
        real_g_list = utils.load_graph_list(real_graph_filename)
        #pred_g_list = utils.load_graph_list(pred_graphs_filename[i])

        # contains all predicted G
        pred_g_list_raw = utils.load_graph_list(pred_graphs_filename[i])
        if len(real_g_list)>200:
            real_g_list = real_g_list[0:200]

        shuffle(real_g_list)
        shuffle(pred_g_list_raw)

        # get length
        real_g_len_list = np.array([len(real_g_list[i]) for i in range(len(real_g_list))])
        pred_g_len_list_raw = np.array([len(pred_g_list_raw[i]) for i in range(len(pred_g_list_raw))])
        # get perturb real
        #perturbed_g_list_001 = perturb(real_g_list, 0.01)
        perturbed_g_list_005 = perturb(real_g_list, 0.05)
        #perturbed_g_list_010 = perturb(real_g_list, 0.10)


        # select pred samples
        # The number of nodes are sampled from the similar distribution as the training set
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
        print('################## epoch {} ##################'.format(epoch_range[i]))

        # info about graph size
        print('real average nodes',
              sum([real_g_list[i].number_of_nodes() for i in range(len(real_g_list))]) / len(real_g_list))
        print('pred average nodes',
              sum([pred_g_list[i].number_of_nodes() for i in range(len(pred_g_list))]) / len(pred_g_list))
        print('num of real graphs', len(real_g_list))
        print('num of pred graphs', len(pred_g_list))

        # ========================================
        # Evaluation
        # ========================================
        mid = len(real_g_list) // 2
        dist_degree, dist_clustering = compute_basic_stats(real_g_list[:mid], real_g_list[mid:])
        #dist_4cycle = eval.stats.motif_stats(real_g_list[:mid], real_g_list[mid:])
        dist_4orbits = eval.stats.orbit_stats_all(real_g_list[:mid], real_g_list[mid:])
        print('degree dist among real: ', dist_degree)
        print('clustering dist among real: ', dist_clustering)
        #print('4 cycle dist among real: ', dist_4cycle)
        print('orbits dist among real: ', dist_4orbits)
        results['deg']['real'] += dist_degree
        results['clustering']['real'] += dist_clustering
        results['orbits4']['real'] += dist_4orbits

        dist_degree, dist_clustering = compute_basic_stats(real_g_list, pred_g_list)
        #dist_4cycle = eval.stats.motif_stats(real_g_list, pred_g_list)
        dist_4orbits = eval.stats.orbit_stats_all(real_g_list, pred_g_list)
        print('degree dist between real and pred at epoch ', epoch_range[i], ': ', dist_degree)
        print('clustering dist between real and pred at epoch ', epoch_range[i], ': ', dist_clustering)
        #print('4 cycle dist between real and pred at epoch: ', epoch_range[i], dist_4cycle)
        print('orbits dist between real and pred at epoch ', epoch_range[i], ': ', dist_4orbits)
        results['deg']['ours'] = min(dist_degree, results['deg']['ours'])
        results['clustering']['ours'] = min(dist_clustering, results['clustering']['ours'])
        results['orbits4']['ours'] = min(dist_4orbits, results['orbits4']['ours'])

        # performance at training time
        out_files['train'].write(str(dist_degree) + ',')
        out_files['train'].write(str(dist_clustering) + ',')
        out_files['train'].write(str(dist_4orbits) + ',')

        dist_degree, dist_clustering = compute_basic_stats(real_g_list, perturbed_g_list_005)
        #dist_4cycle = eval.stats.motif_stats(real_g_list, perturbed_g_list_005)
        dist_4orbits = eval.stats.orbit_stats_all(real_g_list, perturbed_g_list_005)
        print('degree dist between real and perturbed at epoch ', epoch_range[i], ': ', dist_degree)
        print('clustering dist between real and perturbed at epoch ', epoch_range[i], ': ', dist_clustering)
        #print('4 cycle dist between real and perturbed at epoch: ', epoch_range[i], dist_4cycle)
        print('orbits dist between real and perturbed at epoch ', epoch_range[i], ': ', dist_4orbits)
        results['deg']['perturbed'] += dist_degree
        results['clustering']['perturbed'] += dist_clustering
        results['orbits4']['perturbed'] += dist_4orbits

        if i == 0:
            # Baselines
            for baseline in baselines:
                dist_degree, dist_clustering = compute_basic_stats(real_g_list, baselines[baseline])
                dist_4orbits = eval.stats.orbit_stats_all(real_g_list, baselines[baseline])
                results['deg'][baseline] = dist_degree
                results['clustering'][baseline] = dist_clustering
                results['orbits4'][baseline] = dist_4orbits
                print('Kron: deg=', dist_degree, ', clustering=', dist_clustering, 
                        ', orbits4=', dist_4orbits)

        out_files['train'].write('\n')

    for metric, methods in results.items():
        methods['real'] /= num_evals
        methods['perturbed'] /= num_evals

    # Write results
    for metric, methods in results.items():
        line = metric+','+ \
                str(methods['real'])+','+ \
                str(methods['ours'])+','+ \
                str(methods['perturbed'])
        for baseline in baselines:
            line += ',' + str(methods[baseline])
        line += '\n'

        out_files['compare'].write(line)

    for _, out_f in out_files.items():
        out_f.close()


def eval_performance(datadir, prefix=None, args=None, eval_every=200, out_file_prefix=None,
        sample_time = 2, baselines={}):
    if args is None:
        real_graphs_filename = [datadir + f for f in os.listdir(datadir)
                if re.match(prefix + '.*real.*\.dat', f)]
        pred_graphs_filename = [datadir + f for f in os.listdir(datadir)
                if re.match(prefix + '.*pred.*\.dat', f)]
        eval_list(real_graphs_filename, pred_graphs_filename, prefix, 200)

    else:
        # # for vanilla graphrnn
        # real_graphs_filename = [datadir + args.graph_save_path + args.note + '_' + args.graph_type + '_' + \
        #              str(epoch) + '_pred_' + str(args.num_layers) + '_' + str(args.bptt) + '_' + str(args.bptt_len) + '.dat' for epoch in range(0,50001,eval_every)]
        # pred_graphs_filename = [datadir + args.graph_save_path + args.note + '_' + args.graph_type + '_' + \
        #          str(epoch) + '_real_' + str(args.num_layers) + '_' + str(args.bptt) + '_' + str(args.bptt_len) + '.dat' for epoch in range(0,50001,eval_every)]
        
        real_graph_filename = datadir+args.graph_save_path + args.fname_test + '0.dat'
        # for proposed model
        end_epoch = 3001
        epoch_range = range(eval_every, end_epoch, eval_every)
        pred_graphs_filename = [datadir+args.graph_save_path + args.fname_pred+str(epoch)+'_'+str(sample_time)+'.dat'
                for epoch in epoch_range]
        # for baseline model
        #pred_graphs_filename = [datadir+args.fname_baseline+'.dat']

        #real_graphs_filename = [datadir + args.graph_save_path + args.note + '_' + args.graph_type + '_' + \
        #         str(epoch) + '_real_' + str(args.num_layers) + '_' + str(args.bptt) + '_' + str(
        #         args.bptt_len) + '_' + str(args.gumbel) + '.dat' for epoch in range(10000, 50001, eval_every)]
        #pred_graphs_filename = [datadir + args.graph_save_path + args.note + '_' + args.graph_type + '_' + \
        #         str(epoch) + '_pred_' + str(args.num_layers) + '_' + str(args.bptt) + '_' + str(
        #         args.bptt_len) + '_' + str(args.gumbel) + '.dat' for epoch in range(10000, 50001, eval_every)]

        eval_list_fname(real_graph_filename, pred_graphs_filename, baselines,
                        epoch_range=epoch_range, 
                        eval_every=eval_every,
                        out_file_prefix=out_file_prefix)

def process_kron(kron_dir):
    txt_files = []
    for f in os.listdir(kron_dir):
        filename = os.fsdecode(f)
        if filename.endswith('.txt'):
            txt_files.append(filename)
        elif filename.endswith('.dat'):
            return utils.load_graph_list(os.path.join(kron_dir, filename))
    G_list = []
    for filename in txt_files:
        G_list.append(utils.snap_txt_output_to_nx(os.path.join(kron_dir, filename)))

    out_fname = kron_dir + 'kron.dat'
    with open(out_fname, 'wb') as out_f:
        pickle.dump(G_list, out_f)
    return G_list
 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluation arguments.')
    feature_parser = parser.add_mutually_exclusive_group(required=False)
    feature_parser.add_argument('--export-real', dest='export', action='store_true')
    feature_parser.add_argument('--no-export-real', dest='export', action='store_false')
    feature_parser.add_argument('--kron-dir', dest='kron_dir', 
            help='Directory where graphs generated by kronecker method is stored.')
    
    parser.set_defaults(export=False, kron_dir='')
    prog_args = parser.parse_args()

    # datadir = "/dfs/scratch0/rexy/graph_gen_data/"
    ## the following dir has all experiment data, including 1-3 sample times that you can choose from
    datadir = ''
    # datadir = "/dfs/scratch0/jiaxuany0/"


    #datadir = "/lfs/local/0/jiaxuany/pycharm/graphs_share/"
    #datadir = "/lfs/local/0/jiaxuany/pycharm/"
    prefix = "GraphRNN_structure_enzymes_50_"
    args = Args()

    if prog_args.export:
        real_graph_filename = args.graph_save_path + args.fname_real + '0.dat'
        #filename = args.graph_save_path + args.note + '_' + args.graph_type + '_' + \
        #             str(0) + '_real_' + str(args.num_layers) + '_' + str(args.bptt) + '_' + \
        #             str(args.bptt_len) + '_' + str(args.gumbel) 
        input_path = datadir + real_graph_filename
        if not os.path.isdir('eval_results'):
            os.makedirs('eval_results')
        if not os.path.isdir('eval_results/ground_truth'):
            os.makedirs('eval_results/ground_truth')
        out_dir = os.path.join('eval_results/ground_truth', args.graph_type)
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)
        output_prefix = os.path.join(out_dir, args.graph_type)
        print('Export ground truth to ', output_prefix)
        utils.export_graphs_to_txt(input_path, output_prefix)
    else:
        baselines = {}
        if not prog_args.kron_dir == '':
            baselines['kron'] = process_kron(prog_args.kron_dir)

        print(args.graph_type)
        out_file_prefix = 'eval_results/' + args.graph_type + '_' + args.note
        if not os.path.isdir('eval_results'):
            os.makedirs('eval_results')
        eval_performance(datadir, args=args,
                out_file_prefix=out_file_prefix,sample_time=args.sample_time,
                baselines=baselines)

