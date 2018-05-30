import argparse
import numpy as np
import os
import re
from random import shuffle

import eval.stats
import utils
# import main.Args
from baselines.baseline_simple import *

class Args_evaluate():
    def __init__(self):
        # loop over the settings
        # self.model_name_all = ['GraphRNN_MLP','GraphRNN_RNN','Internal','Noise']
        # self.model_name_all = ['E-R', 'B-A']
        self.model_name_all = ['GraphRNN_RNN']
        # self.model_name_all = ['Baseline_DGMG']

        # list of dataset to evaluate
        # use a list of 1 element to evaluate a single dataset
        self.dataset_name_all = ['caveman', 'grid', 'barabasi', 'citeseer', 'DD']
        # self.dataset_name_all = ['citeseer_small','caveman_small']
        # self.dataset_name_all = ['barabasi_noise0','barabasi_noise2','barabasi_noise4','barabasi_noise6','barabasi_noise8','barabasi_noise10']
        # self.dataset_name_all = ['caveman_small', 'ladder_small', 'grid_small', 'ladder_small', 'enzymes_small', 'barabasi_small','citeseer_small']

        self.epoch_start=100
        self.epoch_end=3001
        self.epoch_step=100

def find_nearest_idx(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx

def extract_result_id_and_epoch(name, prefix, suffix):
    '''
    Args:
        eval_every: the number of epochs between consecutive evaluations
        suffix: real_ or pred_
    Returns:
        A tuple of (id, epoch number) extracted from the filename string
    '''
    pos = name.find(suffix) + len(suffix)
    end_pos = name.find('.dat')
    result_id = name[pos:end_pos]

    pos = name.find(prefix) + len(prefix)
    end_pos = name.find('_', pos)
    epochs = int(name[pos:end_pos])
    return result_id, epochs


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



def clean_graphs(graph_real, graph_pred):
    ''' Selecting graphs generated that have the similar sizes.
    It is usually necessary for GraphRNN-S version, but not the full GraphRNN model.
    '''
    shuffle(graph_real)
    shuffle(graph_pred)

    # get length
    real_graph_len = np.array([len(graph_real[i]) for i in range(len(graph_real))])
    pred_graph_len = np.array([len(graph_pred[i]) for i in range(len(graph_pred))])

    # select pred samples
    # The number of nodes are sampled from the similar distribution as the training set
    pred_graph_new = []
    pred_graph_len_new = []
    for value in real_graph_len:
        pred_idx = find_nearest_idx(pred_graph_len, value)
        pred_graph_new.append(graph_pred[pred_idx])
        pred_graph_len_new.append(pred_graph_len[pred_idx])

    return graph_real, pred_graph_new

def load_ground_truth(dir_input, dataset_name, model_name='GraphRNN_RNN'):
    ''' Read ground truth graphs.
    '''
    if not 'small' in dataset_name:
        hidden = 128
    else:
        hidden = 64
    if model_name=='Internal' or model_name=='Noise' or model_name=='B-A' or model_name=='E-R':
        fname_test = dir_input + 'GraphRNN_MLP' + '_' + dataset_name + '_' + str(args.num_layers) + '_' + str(
                hidden) + '_test_' + str(0) + '.dat'
    else:
        fname_test = dir_input + model_name + '_' + dataset_name + '_' + str(args.num_layers) + '_' + str(
                hidden) + '_test_' + str(0) + '.dat'
    try:
        graph_test = utils.load_graph_list(fname_test,is_real=True)
    except:
        print('Not found: ' + fname_test)
        logging.warning('Not found: ' + fname_test)
        return None
    return graph_test

def eval_single_list(graphs, dir_input, dataset_name):
    ''' Evaluate a list of graphs by comparing with graphs in directory dir_input.
    Args:
        dir_input: directory where ground truth graph list is stored
        dataset_name: name of the dataset (ground truth)
    '''
    graph_test = load_ground_truth(dir_input, dataset_name)
    graph_test_len = len(graph_test)
    graph_test = graph_test[int(0.8 * graph_test_len):] # test on a hold out test set
    mmd_degree = eval.stats.degree_stats(graph_test, graphs)
    mmd_clustering = eval.stats.clustering_stats(graph_test, graphs)
    try:
        mmd_4orbits = eval.stats.orbit_stats_all(graph_test, graphs)
    except:
        mmd_4orbits = -1
    print('deg: ', mmd_degree)
    print('clustering: ', mmd_clustering)
    print('orbits: ', mmd_4orbits)

def evaluation_epoch(dir_input, fname_output, model_name, dataset_name, args, is_clean=True, epoch_start=1000,epoch_end=3001,epoch_step=100):
    with open(fname_output, 'w+') as f:
        f.write('sample_time,epoch,degree_validate,clustering_validate,orbits4_validate,degree_test,clustering_test,orbits4_test\n')

        # TODO: Maybe refactor into a separate file/function that specifies THE naming convention
        # across main and evaluate
        if not 'small' in dataset_name:
            hidden = 128
        else:
            hidden = 64
        # read real graph
        if model_name=='Internal' or model_name=='Noise' or model_name=='B-A' or model_name=='E-R':
            fname_test = dir_input + 'GraphRNN_MLP' + '_' + dataset_name + '_' + str(args.num_layers) + '_' + str(
                hidden) + '_test_' + str(0) + '.dat'
        elif 'Baseline' in model_name:
            fname_test = dir_input + model_name + '_' + dataset_name + '_' + str(64) + '_test_' + str(0) + '.dat'
        else:
            fname_test = dir_input + model_name + '_' + dataset_name + '_' + str(args.num_layers) + '_' + str(
                hidden) + '_test_' + str(0) + '.dat'
        try:
            graph_test = utils.load_graph_list(fname_test,is_real=True)
        except:
            print('Not found: ' + fname_test)
            logging.warning('Not found: ' + fname_test)
            return None

        graph_test_len = len(graph_test)
        graph_train = graph_test[0:int(0.8 * graph_test_len)] # train
        graph_validate = graph_test[0:int(0.2 * graph_test_len)] # validate
        graph_test = graph_test[int(0.8 * graph_test_len):] # test on a hold out test set

        graph_test_aver = 0
        for graph in graph_test:
            graph_test_aver+=graph.number_of_nodes()
        graph_test_aver /= len(graph_test)
        print('test average len',graph_test_aver)


        # get performance for proposed approaches
        if 'GraphRNN' in model_name:
            # read test graph
            for epoch in range(epoch_start,epoch_end,epoch_step):
                for sample_time in range(1,4):
                    # get filename
                    fname_pred = dir_input + model_name + '_' + dataset_name + '_' + str(args.num_layers) + '_' + str(hidden) + '_pred_' + str(epoch) + '_' + str(sample_time) + '.dat'
                    # load graphs
                    try:
                        graph_pred = utils.load_graph_list(fname_pred,is_real=False) # default False
                    except:
                        print('Not found: '+ fname_pred)
                        logging.warning('Not found: '+ fname_pred)
                        continue
                    # clean graphs
                    if is_clean:
                        graph_test, graph_pred = clean_graphs(graph_test, graph_pred)
                    else:
                        shuffle(graph_pred)
                        graph_pred = graph_pred[0:len(graph_test)]
                    print('len graph_test', len(graph_test))
                    print('len graph_validate', len(graph_validate))
                    print('len graph_pred', len(graph_pred))

                    graph_pred_aver = 0
                    for graph in graph_pred:
                        graph_pred_aver += graph.number_of_nodes()
                    graph_pred_aver /= len(graph_pred)
                    print('pred average len', graph_pred_aver)

                    # evaluate MMD test
                    mmd_degree = eval.stats.degree_stats(graph_test, graph_pred)
                    mmd_clustering = eval.stats.clustering_stats(graph_test, graph_pred)
                    try:
                        mmd_4orbits = eval.stats.orbit_stats_all(graph_test, graph_pred)
                    except:
                        mmd_4orbits = -1
                    # evaluate MMD validate
                    mmd_degree_validate = eval.stats.degree_stats(graph_validate, graph_pred)
                    mmd_clustering_validate = eval.stats.clustering_stats(graph_validate, graph_pred)
                    try:
                        mmd_4orbits_validate = eval.stats.orbit_stats_all(graph_validate, graph_pred)
                    except:
                        mmd_4orbits_validate = -1
                    # write results
                    f.write(str(sample_time)+','+
                            str(epoch)+','+
                            str(mmd_degree_validate)+','+
                            str(mmd_clustering_validate)+','+
                            str(mmd_4orbits_validate)+','+ 
                            str(mmd_degree)+','+
                            str(mmd_clustering)+','+
                            str(mmd_4orbits)+'\n')
                    print('degree',mmd_degree,'clustering',mmd_clustering,'orbits',mmd_4orbits)

        # get internal MMD (MMD between ground truth validation and test sets)
        if model_name == 'Internal':
            mmd_degree_validate = eval.stats.degree_stats(graph_test, graph_validate)
            mmd_clustering_validate = eval.stats.clustering_stats(graph_test, graph_validate)
            try:
                mmd_4orbits_validate = eval.stats.orbit_stats_all(graph_test, graph_validate)
            except:
                mmd_4orbits_validate = -1
            f.write(str(-1) + ',' + str(-1) + ',' + str(mmd_degree_validate) + ',' + str(
                mmd_clustering_validate) + ',' + str(mmd_4orbits_validate)
                    + ',' + str(-1) + ',' + str(-1) + ',' + str(-1) + '\n')


        # get MMD between ground truth and its perturbed graphs
        if model_name == 'Noise':
            graph_validate_perturbed = perturb(graph_validate, 0.05)
            mmd_degree_validate = eval.stats.degree_stats(graph_test, graph_validate_perturbed)
            mmd_clustering_validate = eval.stats.clustering_stats(graph_test, graph_validate_perturbed)
            try:
                mmd_4orbits_validate = eval.stats.orbit_stats_all(graph_test, graph_validate_perturbed)
            except:
                mmd_4orbits_validate = -1
            f.write(str(-1) + ',' + str(-1) + ',' + str(mmd_degree_validate) + ',' + str(
                mmd_clustering_validate) + ',' + str(mmd_4orbits_validate)
                    + ',' + str(-1) + ',' + str(-1) + ',' + str(-1) + '\n')

        # get E-R MMD
        if model_name == 'E-R':
            graph_pred = Graph_generator_baseline(graph_train,generator='Gnp')
            # clean graphs
            if is_clean:
                graph_test, graph_pred = clean_graphs(graph_test, graph_pred)
            print('len graph_test', len(graph_test))
            print('len graph_pred', len(graph_pred))
            mmd_degree = eval.stats.degree_stats(graph_test, graph_pred)
            mmd_clustering = eval.stats.clustering_stats(graph_test, graph_pred)
            try:
                mmd_4orbits_validate = eval.stats.orbit_stats_all(graph_test, graph_pred)
            except:
                mmd_4orbits_validate = -1
            f.write(str(-1) + ',' + str(-1) + ',' + str(-1) + ',' + str(-1) + ',' + str(-1)
                    + ',' + str(mmd_degree) + ',' + str(mmd_clustering) + ',' + str(mmd_4orbits_validate) + '\n')


        # get B-A MMD
        if model_name == 'B-A':
            graph_pred = Graph_generator_baseline(graph_train, generator='BA')
            # clean graphs
            if is_clean:
                graph_test, graph_pred = clean_graphs(graph_test, graph_pred)
            print('len graph_test', len(graph_test))
            print('len graph_pred', len(graph_pred))
            mmd_degree = eval.stats.degree_stats(graph_test, graph_pred)
            mmd_clustering = eval.stats.clustering_stats(graph_test, graph_pred)
            try:
                mmd_4orbits_validate = eval.stats.orbit_stats_all(graph_test, graph_pred)
            except:
                mmd_4orbits_validate = -1
            f.write(str(-1) + ',' + str(-1) + ',' + str(-1) + ',' + str(-1) + ',' + str(-1)
                    + ',' + str(mmd_degree) + ',' + str(mmd_clustering) + ',' + str(mmd_4orbits_validate) + '\n')

        # get performance for baseline approaches
        if 'Baseline' in model_name:
            # read test graph
            for epoch in range(epoch_start, epoch_end, epoch_step):
                # get filename
                fname_pred = dir_input + model_name + '_' + dataset_name + '_' + str(
                    64) + '_pred_' + str(epoch) + '.dat'
                # load graphs
                try:
                    graph_pred = utils.load_graph_list(fname_pred, is_real=True)  # default False
                except:
                    print('Not found: ' + fname_pred)
                    logging.warning('Not found: ' + fname_pred)
                    continue
                # clean graphs
                if is_clean:
                    graph_test, graph_pred = clean_graphs(graph_test, graph_pred)
                else:
                    shuffle(graph_pred)
                    graph_pred = graph_pred[0:len(graph_test)]
                print('len graph_test', len(graph_test))
                print('len graph_validate', len(graph_validate))
                print('len graph_pred', len(graph_pred))

                graph_pred_aver = 0
                for graph in graph_pred:
                    graph_pred_aver += graph.number_of_nodes()
                graph_pred_aver /= len(graph_pred)
                print('pred average len', graph_pred_aver)

                # evaluate MMD test
                mmd_degree = eval.stats.degree_stats(graph_test, graph_pred)
                mmd_clustering = eval.stats.clustering_stats(graph_test, graph_pred)
                try:
                    mmd_4orbits = eval.stats.orbit_stats_all(graph_test, graph_pred)
                except:
                    mmd_4orbits = -1
                # evaluate MMD validate
                mmd_degree_validate = eval.stats.degree_stats(graph_validate, graph_pred)
                mmd_clustering_validate = eval.stats.clustering_stats(graph_validate, graph_pred)
                try:
                    mmd_4orbits_validate = eval.stats.orbit_stats_all(graph_validate, graph_pred)
                except:
                    mmd_4orbits_validate = -1
                # write results
                f.write(str(-1) + ',' + str(epoch) + ',' + str(mmd_degree_validate) + ',' + str(
                    mmd_clustering_validate) + ',' + str(mmd_4orbits_validate)
                        + ',' + str(mmd_degree) + ',' + str(mmd_clustering) + ',' + str(mmd_4orbits) + '\n')
                print('degree', mmd_degree, 'clustering', mmd_clustering, 'orbits', mmd_4orbits)



        return True

def evaluation(args_evaluate,dir_input, dir_output, model_name_all, dataset_name_all, args, overwrite = True):
    ''' Evaluate the performance of a set of models on a set of datasets.
    '''
    for model_name in model_name_all:
        for dataset_name in dataset_name_all:
            # check output exist
            fname_output = dir_output+model_name+'_'+dataset_name+'.csv'
            print('processing: '+dir_output + model_name + '_' + dataset_name + '.csv')
            logging.info('processing: '+dir_output + model_name + '_' + dataset_name + '.csv')
            if overwrite==False and os.path.isfile(fname_output):
                print(dir_output+model_name+'_'+dataset_name+'.csv exists!')
                logging.info(dir_output+model_name+'_'+dataset_name+'.csv exists!')
                continue
            evaluation_epoch(dir_input,fname_output,model_name,dataset_name,args,is_clean=True, epoch_start=args_evaluate.epoch_start,epoch_end=args_evaluate.epoch_end,epoch_step=args_evaluate.epoch_step)







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

    return G_list
 

if __name__ == '__main__':
    args = Args()
    args_evaluate = Args_evaluate()

    parser = argparse.ArgumentParser(description='Evaluation arguments.')
    feature_parser = parser.add_mutually_exclusive_group(required=False)
    feature_parser.add_argument('--export-real', dest='export', action='store_true')
    feature_parser.add_argument('--no-export-real', dest='export', action='store_false')
    feature_parser.add_argument('--kron-dir', dest='kron_dir', 
            help='Directory where graphs generated by kronecker method is stored.')

    parser.add_argument('--testfile', dest='test_file',
            help='The file that stores list of graphs to be evaluated. Only used when 1 list of '
                 'graphs is to be evaluated.')
    parser.add_argument('--dir-prefix', dest='dir_prefix',
            help='The file that stores list of graphs to be evaluated. Can be used when evaluating multiple'
                 'models on multiple datasets.')
    parser.add_argument('--graph-type', dest='graph_type',
            help='Type of graphs / dataset.')
    
    parser.set_defaults(export=False, kron_dir='', test_file='',
                        dir_prefix='',
                        graph_type=args.graph_type)
    prog_args = parser.parse_args()

    # dir_prefix = prog_args.dir_prefix
    # dir_prefix = "/dfs/scratch0/jiaxuany0/"
    dir_prefix = args.dir_input


    time_now = strftime("%Y-%m-%d %H:%M:%S", gmtime())
    if not os.path.isdir('logs/'):
        os.makedirs('logs/')
    logging.basicConfig(filename='logs/evaluate' + time_now + '.log', level=logging.INFO)

    if prog_args.export:
        if not os.path.isdir('eval_results'):
            os.makedirs('eval_results')
        if not os.path.isdir('eval_results/ground_truth'):
            os.makedirs('eval_results/ground_truth')
        out_dir = os.path.join('eval_results/ground_truth', prog_args.graph_type)
        if not os.path.isdir(out_dir):
            os.makedirs(out_dir)
        output_prefix = os.path.join(out_dir, prog_args.graph_type)
        print('Export ground truth to prefix: ', output_prefix)

        if prog_args.graph_type == 'grid':
            graphs = []
            for i in range(10,20):
                for j in range(10,20):
                    graphs.append(nx.grid_2d_graph(i,j))
            utils.export_graphs_to_txt(graphs, output_prefix)
        elif prog_args.graph_type == 'caveman':
            graphs = []
            for i in range(2, 3):
                for j in range(30, 81):
                    for k in range(10):
                        graphs.append(caveman_special(i,j, p_edge=0.3))
            utils.export_graphs_to_txt(graphs, output_prefix)
        elif prog_args.graph_type == 'citeseer':
            graphs = utils.citeseer_ego()
            utils.export_graphs_to_txt(graphs, output_prefix)
        else:
            # load from directory
            input_path = dir_prefix + real_graph_filename
            g_list = utils.load_graph_list(input_path)
            utils.export_graphs_to_txt(g_list, output_prefix)
    elif not prog_args.kron_dir == '':
        kron_g_list = process_kron(prog_args.kron_dir)
        fname = os.path.join(prog_args.kron_dir, prog_args.graph_type + '.dat')
        print([g.number_of_nodes() for g in kron_g_list])
        utils.save_graph_list(kron_g_list, fname)
    elif not prog_args.test_file == '':
        # evaluate single .dat file containing list of test graphs (networkx format)
        graphs = utils.load_graph_list(prog_args.test_file)
        eval_single_list(graphs, dir_input=dir_prefix+'graphs/', dataset_name='grid')
    ## if you don't try kronecker, only the following part is needed
    else:
        if not os.path.isdir(dir_prefix+'eval_results'):
            os.makedirs(dir_prefix+'eval_results')
        evaluation(args_evaluate,dir_input=dir_prefix+"graphs/", dir_output=dir_prefix+"eval_results/",
                   model_name_all=args_evaluate.model_name_all,dataset_name_all=args_evaluate.dataset_name_all,args=args,overwrite=True)




