from train import *

args = Args()

graphs = create_graphs.create(args)

dataset = Graph_sequence_sampler_pytorch(graphs,max_prev_node=None,max_num_node=None)