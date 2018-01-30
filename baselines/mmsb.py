"""Stochastic block model."""

import argparse
import os
from time import time

import edward as ed
import networkx as nx
import numpy as np
import tensorflow as tf

from edward.models import Bernoulli, Multinomial, Beta, Dirichlet, PointMass
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
    rho = 0
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
    #qgamma = PointMass(params=tf.nn.softmax(tf.Variable(tf.constant(1.0, shape=[K]))))
    qPi = PointMass(params=tf.nn.sigmoid(tf.Variable(tf.random_normal([K, K]))))
    qZ = PointMass(params=tf.nn.softmax(tf.Variable(tf.random_normal([N, K]))))
    #qZ = PointMass(params=tf.nn.softmax(tf.Variable(tf.constant(1.0, shape=[N, K]))))
    
    inference = ed.MAP({gamma: qgamma, Pi: qPi, Z: qZ}, data={X: data})
    #inference = ed.MAP({Pi: qPi, Z: qZ}, data={X: data})
    
    n_iter = 20000
    inference.initialize(optimizer=tf.train.RMSPropOptimizer(learning_rate=0.01), n_iter=n_iter)
    
    tf.global_variables_initializer().run()
    print(qgamma.mean().eval())
    
    for _ in range(inference.n_iter):
        info_dict = inference.update()
        inference.print_progress(info_dict)
    
    inference.finalize()
    print('qgamma after: ', qgamma.mean().eval())
    return qZ.mean().eval(), qPi.eval()

def arg_parse():
    parser = argparse.ArgumentParser(description='MMSB arguments.')
    io_parser = parser.add_mutually_exclusive_group(required=False)
    io_parser.add_argument('--dataset', dest='dataset', 
            help='Input dataset.')

    parser.set_defaults(dataset='clique_test',
                        )
    return parser.parse_args()

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

    for i in range(len(X_dataset)):
        print('Number of graphs: ', len(graphs))
        N = X_data.shape[0]  # number of vertices
        K = 4  # number of clusters

        Zp, B = mmsb(N, K, X_data)
        print("Block: ", B)
        Z_pred = Zp.argmax(axis=1)
        print("Result (label flip can happen):")
        print("prob: ", Zp)
        print("Predicted")
        print(Z_pred)
        #print("True")
        #print(Z_true)
        #print("Adjusted Rand Index =", adjusted_rand_score(Z_pred, Z_true))

