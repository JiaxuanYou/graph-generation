'''
Reference implementation of node2vec. 

Author: Aditya Grover

For more details, refer to the paper:
node2vec: Scalable Feature Learning for Networks
Aditya Grover and Jure Leskovec 
Knowledge Discovery and Data Mining (KDD), 2016
'''

import argparse
import numpy as np
import networkx as nx
import node2vec.src.node2vec as node2vec
from gensim.models import Word2Vec
from six import string_types, iteritems


import os
from pathlib import Path


# def parse_args():
# 	'''
# 	Parses the node2vec arguments.
# 	'''
# 	parser = argparse.ArgumentParser(description="Run node2vec.")
#
# 	parser.add_argument('--input', nargs='?', default='graph/karate.edgelist',
# 	                    help='Input graph path')
#
# 	parser.add_argument('--output', nargs='?', default='emb/karate.emb',
# 	                    help='Embeddings path')
#
# 	parser.add_argument('--dimensions', type=int, default=128,
# 	                    help='Number of dimensions. Default is 128.')
#
# 	parser.add_argument('--walk-length', type=int, default=80,
# 	                    help='Length of walk per source. Default is 80.')
#
# 	parser.add_argument('--num-walks', type=int, default=10,
# 	                    help='Number of walks per source. Default is 10.')
#
# 	parser.add_argument('--window-size', type=int, default=10,
#                     	help='Context size for optimization. Default is 10.')
#
# 	parser.add_argument('--iter', default=1, type=int,
#                       help='Number of epochs in SGD')
#
# 	parser.add_argument('--workers', type=int, default=8,
# 	                    help='Number of parallel workers. Default is 8.')
#
# 	parser.add_argument('--p', type=float, default=1,
# 	                    help='Return hyperparameter. Default is 1.')
#
# 	parser.add_argument('--q', type=float, default=1,
# 	                    help='Inout hyperparameter. Default is 1.')
#
# 	parser.add_argument('--weighted', dest='weighted', action='store_true',
# 	                    help='Boolean specifying (un)weighted. Default is unweighted.')
# 	parser.add_argument('--unweighted', dest='unweighted', action='store_false')
# 	parser.set_defaults(weighted=False)
#
# 	parser.add_argument('--directed', dest='directed', action='store_true',
# 	                    help='Graph is (un)directed. Default is undirected.')
# 	parser.add_argument('--undirected', dest='undirected', action='store_false')
# 	parser.set_defaults(directed=False)
#
# 	return parser.parse_args()

class config():
	def __init__(self, dimension = 128, walk_length = 80, num_walks = 10, window_size = 10, iter = 1, workers = 8, p = 1, q = 1, weighted= False, directed = False):
		dir_path = os.path.dirname(os.path.realpath(__file__))
		parent_path = Path(dir_path).parent
		# 'Input graph path'
		self.input = str(parent_path)+'/graph/karate.edgelist'
		# 'Embeddings path'
		self.output = str(parent_path)+'/emb/karate.emb'
		# 'Number of dimensions. Default is 128.'
		self.dimensions = dimension
		# 'Length of walk per source. Default is 80.'
		self.walk_length = walk_length
		# 'Number of walks per source. Default is 10.'
		self.num_walks = num_walks
		# 'Context size for optimization. Default is 10.'
		self.window_size = window_size
		# 'Number of epochs in SGD'
		self.iter = iter
		# 'Number of parallel workers. Default is 8.'
		self.workers = workers
		# 'Return hyperparameter. Default is 1.'
		self.p = p
		# 'Inout hyperparameter. Default is 1.'
		self.q = q
		# 'Boolean specifying (un)weighted. Default is unweighted.'
		self.weighted = weighted
		self.unweighted = not self.weighted
		# 'Graph is (un)directed. Default is undirected.'
		self.directed = directed
		self.undirected = not self.directed


# args = config()

def read_graph(args):
	'''
	Reads the input network in networkx.
	'''
	if args.weighted:
		G = nx.read_edgelist(args.input, nodetype=int, data=(('weight',float),), create_using=nx.DiGraph())
	else:
		G = nx.read_edgelist(args.input, nodetype=int, create_using=nx.DiGraph())
		for edge in G.edges():
			G[edge[0]][edge[1]]['weight'] = 1

	if not args.directed:
		G = G.to_undirected()

	return G

def learn_embeddings(walks,args):
	'''
	Learn embeddings by optimizing the Skipgram objective using SGD.
	'''
	walks = [map(str, walk) for walk in walks]
	model = Word2Vec(walks, size=args.dimensions, window=args.window_size, min_count=0, sg=1, workers=args.workers, iter=args.iter)

	embedding = np.zeros([len(model.wv.vocab), args.dimensions])
	for word, vocab in sorted(iteritems(model.wv.vocab), key=lambda item: -item[1].count):
		row = model.wv.syn0[vocab.index]
		word = int(word)
		# print('word', word, type(word), 'row', row, row.shape)
		embedding[word,:] = row

	# save .emb
	# model.wv.save_word2vec_format(args.output)
	
	return embedding

def node2vec_main(nx_G, args):
	'''
	Pipeline for representational learning for all nodes in a graph.
	'''
	G = node2vec.Graph(nx_G, args.directed, args.p, args.q)
	G.preprocess_transition_probs()
	walks = G.simulate_walks(args.num_walks, args.walk_length)
	embedding = learn_embeddings(walks, args)
	return embedding

# if __name__ == "__main__":
# 	args = parse_args()
# 	main(args)
