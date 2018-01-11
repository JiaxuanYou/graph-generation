"""Stochastic block model."""

from time import time

import edward as ed
import networkx as nx
import numpy as np
import tensorflow as tf

from edward.models import Bernoulli, Multinomial, Beta, Dirichlet, PointMass
from observations import karate
from sklearn.metrics.cluster import adjusted_rand_score

ed.set_seed(int(time()))
#ed.set_seed(42)

# DATA
#X_data, Z_true = karate("data")

def mmsb(N, K, data):
    # MODEL
    # probability of belonging to each of K blocks for each node
    gamma = Dirichlet(concentration=tf.ones([K]))
    # block connectivity
    Pi = Beta(concentration0=tf.ones([K, K]), concentration1=tf.ones([K, K]))
    # probability of belonging to each of K blocks for all nodes
    Z = Multinomial(total_count=1.0, probs=gamma, sample_shape=N)
    # adjacency
    X = Bernoulli(probs=tf.matmul(Z, tf.matmul(Pi, tf.transpose(Z))))
    
    # INFERENCE (EM algorithm)
    qgamma = PointMass(params=tf.nn.softmax(tf.Variable(tf.random_normal([K]))))
    qPi = PointMass(params=tf.nn.sigmoid(tf.Variable(tf.random_normal([K, K]))))
    qZ = PointMass(params=tf.nn.softmax(tf.Variable(tf.random_normal([N, K]))))
    
    inference = ed.MAP({gamma: qgamma, Pi: qPi, Z: qZ}, data={X: data})
    
    n_iter = 1000000
    inference.initialize(optimizer=tf.train.AdamOptimizer(learning_rate=0.01), n_iter=n_iter)
    
    tf.global_variables_initializer().run()
    
    for _ in range(inference.n_iter):
        info_dict = inference.update()
        inference.print_progress(info_dict)
    
    inference.finalize()
    return qZ.mean().eval(), qPi.eval()

if __name__ == '__main__':
    X_data = nx.to_numpy_matrix(nx.connected_caveman_graph(4, 7))
    print(X_data)
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

