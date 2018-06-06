"""Stochastic block model."""

import argparse
import os
from time import time

import edward as ed
import networkx as nx
import numpy as np
import tensorflow as tf

from edward.models import Bernoulli, Multinomial, Beta, Dirichlet, PointMass, Normal
from observations import karate
from sklearn.metrics.cluster import adjusted_rand_score

import utils

CUDA = 2
ed.set_seed(int(time()))
#ed.set_seed(42)

# DATA
#X_data, Z_true = karate("data")

def disjoint_cliques_test_graph(num_cliques, clique_size):
    G = nx.disjoint_union_all([nx.complete_graph(clique_size) for _ in range(num_cliques)])
    return nx.to_numpy_matrix(G)

def mmsb(N, K, data):
    # sparsity
    rho = 0.3
    # MODEL
    # probability of belonging to each of K blocks for each node
    gamma = Dirichlet(concentration=tf.ones([K]))
    # block connectivity
    Pi = Beta(concentration0=tf.ones([K, K]), concentration1=tf.ones([K, K]))
    # probability of belonging to each of K blocks for all nodes
    Z = Multinomial(total_count=1.0, probs=gamma, sample_shape=N)
    # adjacency
    X = Bernoulli(probs=(1 - rho) * tf.matmul(Z, tf.matmul(Pi, tf.transpose(Z))))
    
    # INFERENCE (EM algorithm)
    qgamma = PointMass(params=tf.nn.softmax(tf.Variable(tf.random_normal([K]))))
    qPi = PointMass(params=tf.nn.sigmoid(tf.Variable(tf.random_normal([K, K]))))
    qZ = PointMass(params=tf.nn.softmax(tf.Variable(tf.random_normal([N, K]))))
    
    #qgamma = Normal(loc=tf.get_variable("qgamma/loc", [K]),
    #                scale=tf.nn.softplus(
    #                        tf.get_variable("qgamma/scale", [K])))
    #qPi = Normal(loc=tf.get_variable("qPi/loc", [K, K]),
    #                scale=tf.nn.softplus(
    #                        tf.get_variable("qPi/scale", [K, K])))
    #qZ = Normal(loc=tf.get_variable("qZ/loc", [N, K]),
    #                scale=tf.nn.softplus(
    #                        tf.get_variable("qZ/scale", [N, K])))
    
    #inference = ed.KLqp({gamma: qgamma, Pi: qPi, Z: qZ}, data={X: data})
    inference = ed.MAP({gamma: qgamma, Pi: qPi, Z: qZ}, data={X: data})
    
    #inference.run()
    n_iter = 6000
    inference.initialize(optimizer=tf.train.AdamOptimizer(learning_rate=0.01), n_iter=n_iter)
    
    tf.global_variables_initializer().run()
    
    for _ in range(inference.n_iter):
        info_dict = inference.update()
        inference.print_progress(info_dict)
    
    inference.finalize()
    print('qgamma after: ', qgamma.mean().eval())
    return qZ.mean().eval(), qPi.eval()

def arg_parse():
    parser = argparse.ArgumentParser(description='MMSB arguments.')
    parser.add_argument('--dataset', dest='dataset', 
            help='Input dataset.')
    parser.add_argument('--K', dest='K', type=int,
            help='Number of blocks.')
    parser.add_argument('--samples-per-G', dest='samples', type=int,
            help='Number of samples for every graph.')

    parser.set_defaults(dataset='community',
                        K=4,
                        samples=1)
    return parser.parse_args()

def graph_gen_from_blockmodel(B, Z):
    n_blocks = len(B)
    B = np.array(B)
    Z = np.array(Z)
    adj_prob = np.dot(Z, np.dot(B, np.transpose(Z)))
    adj = np.random.binomial(1, adj_prob * 0.3)
    return nx.from_numpy_matrix(adj)

if __name__ == '__main__':
    prog_args = arg_parse()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(CUDA)
    print('CUDA', CUDA)

    X_dataset = []
    #X_data = nx.to_numpy_matrix(nx.connected_caveman_graph(4, 7))
    if prog_args.dataset == 'clique_test':
        X_data = disjoint_cliques_test_graph(4, 7)
        X_dataset.append(X_data)
    elif prog_args.dataset == 'citeseer':
        graphs = utils.citeseer_ego()
        X_dataset = [nx.to_numpy_matrix(g) for g in graphs]
    elif prog_args.dataset == 'community':
        graphs = []
        for i in range(2, 3):
            for j in range(30, 81):
                for k in range(10):
                    graphs.append(utils.caveman_special(i,j, p_edge=0.3))
        X_dataset = [nx.to_numpy_matrix(g) for g in graphs]
    elif prog_args.dataset == 'grid':
        graphs = []
        for i in range(10,20):
            for j in range(10,20):
                graphs.append(nx.grid_2d_graph(i,j))
        X_dataset = [nx.to_numpy_matrix(g) for g in graphs]
    elif prog_args.dataset.startswith('community'):
        graphs = []
        num_communities = int(prog_args.dataset[-1])
        print('Creating dataset with ', num_communities, ' communities')
        c_sizes = np.random.choice([12, 13, 14, 15, 16, 17], num_communities)
        for k in range(3000):
            graphs.append(utils.n_community(c_sizes, p_inter=0.01))
        X_dataset = [nx.to_numpy_matrix(g) for g in graphs]

    print('Number of graphs: ', len(X_dataset))
    K = prog_args.K  # number of clusters
    gen_graphs = []
    for i in range(len(X_dataset)):
        if i % 5 == 0:
            print(i)
            X_data = X_dataset[i]
            N = X_data.shape[0]  # number of vertices

            Zp, B = mmsb(N, K, X_data)
            #print("Block: ", B)
            Z_pred = Zp.argmax(axis=1)
            print("Result (label flip can happen):")
            #print("prob: ", Zp)
            print("Predicted")
            print(Z_pred)
            #print(Z_true)
            #print("Adjusted Rand Index =", adjusted_rand_score(Z_pred, Z_true))
            for j in range(prog_args.samples):
                gen_graphs.append(graph_gen_from_blockmodel(B, Zp))

    save_path = '/lfs/local/0/rexy/graph-generation/eval_results/mmsb/'
    utils.save_graph_list(gen_graphs, os.path.join(save_path, prog_args.dataset + '.dat'))

