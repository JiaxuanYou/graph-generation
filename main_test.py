import warnings
warnings.filterwarnings("ignore")

from train import *
from create_graphs import *
from evaluate import *
import argparse

parser = argparse.ArgumentParser()
# Dataset Info
parser.add_argument('--dataset_norm', dest='dataset_norm', type=str,
                    help='Normal class dataset used to train the model')
parser.add_argument('--dataset_anom', dest='dataset_anom',type=str,
                    help='The anomalous dataset')
parser.add_argument('--load_epoch', type=int, 
                    help='The epoch for which to load the pre-trained model')
parser.add_argument('--max_iter', type=int, default=5,
                    help='The number of permuations to consider for each graph')
parser.add_argument('--anom_test', action='store_true', 
                    help='Whether or not we calculate the train and test lls' )
parser.set_defaults(anom_test=False)


"""
    
    Example of run:

    
"""


# User inputed args
user_args = parser.parse_args()


# Generate the NLL predictions for different experiments
# so that we can avoid re-calculating these during visualization

# Save results based on the model we trained and
# the dataset that we trained on as the normal class
nll_dir = 'nll' + '/' + user_args.dataset_norm + '/'
if not os.path.isdir(nll_dir):
        os.mkdir(nll_dir)



# Generate the model datasets (i.e. the dataset used to train the model - normal data)
# Note: Even if we do not want to compute the nlls over the normal set
# the dataset parameters in args_normal are important (i.e. max_prev_node)
args_norm, train_norm, _, test_norm = get_graph_data(user_args.dataset_norm, isModelDataset=True)
# Save the max_previous node to allow for model 
# compatability on future datasets
print ()
print ("Normal Graph Distribution:", user_args.dataset_norm)
print ()
max_prev_node = args_norm.max_prev_node

# Model initialization
# Using GraphRNN

rnn = GRU_plain(input_size=args_norm.max_prev_node, embedding_size=args_norm.embedding_size_rnn,
                        hidden_size=args_norm.hidden_size_rnn, num_layers=args_norm.num_layers, has_input=True,
                        has_output=True, output_size=args_norm.hidden_size_rnn_output).to(device)
output = GRU_plain(input_size=1, embedding_size=args_norm.embedding_size_rnn_output,
                        hidden_size=args_norm.hidden_size_rnn_output, num_layers=args_norm.num_layers, has_input=True,
                        has_output=True, output_size=1).to(device)

# Load from the state dict
# Set the epoch we are loading from
fname_rnn = args_norm.model_save_path + args_norm.fname + 'lstm_' + str(user_args.load_epoch) + '.dat'
fname_out = args_norm.model_save_path + args_norm.fname + 'output_' + str(user_args.load_epoch) + '.dat'

rnn.load_state_dict(torch.load(fname_rnn))
output.load_state_dict(torch.load(fname_out))

epoch = user_args.load_epoch
print('model loaded!, epoch: {}'.format(user_args.load_epoch))

# Initialize the dataset and dataloader for analyzing the nlls. 
# Note that we use batch_size = 1 because we want the nll for each
# data point not an average across a batch.
# We should also consider using Graph_sequence_Sampler_pytorch_nll!!!!
# This sampler expands the size of the dataset by for each graph 
# creating many different bfs permutations. The idea behind this sampler
# is to test the models permutation invariance. 

# If --anom_test flag is true than we only want to compute
# the scores for the anomalous graph as we have already computed
# them for the train and test
if not user_args.anom_test:
    train_dataset = Graph_sequence_sampler_pytorch_rand(train_norm,max_prev_node=args_norm.max_prev_node,max_num_node=args_norm.max_num_node)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, num_workers=args_norm.num_workers)

    test_dataset = Graph_sequence_sampler_pytorch_rand(test_norm, max_prev_node=args_norm.max_prev_node,max_num_node=args_norm.max_num_node)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, num_workers=args_norm.num_workers)

    train_nlls, train_avg_nlls = calc_nll(args_norm, train_loader, rnn, output, max_iter=user_args.max_iter, log=1) #, max_iter=user_args.max_iter, load_epoch=user_args.load_epoch, log=1)

    # Compute the avg_nll for each graph averaged over the max_iter permuations
    train_avg_graph_nlls = np.array(train_avg_nlls)
    train_avg_graph_nlls = train_avg_graph_nlls.reshape((user_args.max_iter, len(train_loader)))
    train_avg_graph_nlls = np.mean(train_avg_graph_nlls, axis=0)

    # Analysis of the test data set nlls.
    # We really gotta train over more data!
    test_nlls, test_avg_nlls = calc_nll(args_norm, test_loader, rnn, output, max_iter=user_args.max_iter, log=1) #, load_epoch=user_args.load_epoch, max_iter=user_args.max_iter, log=1)

    test_avg_graph_nlls = np.array(test_avg_nlls)
    test_avg_graph_nlls = test_avg_graph_nlls.reshape((user_args.max_iter, len(test_loader)))
    test_avg_graph_nlls = np.mean(test_avg_graph_nlls, axis=0)

    np.save(nll_dir + 'train_avg_graph_nlls.npy', train_avg_graph_nlls)
    np.save(nll_dir + 'test_avg_graph_nlls.npy', test_avg_graph_nlls)

# Calculate NLLs for anomalous graphs
args_anom, graphs_anom = get_graph_data(user_args.dataset_anom, isModelDataset=False)

# Note that instead of passing in user_args.max_prev_node, we pass in the saved max_prev_node
# saved specifically for the normal dataset - hack to allow for GraphRNN to work on any dataset
anom_dataset = Graph_sequence_sampler_pytorch_rand(graphs_anom,max_prev_node=max_prev_node,max_num_node=args_anom.max_num_node)
anom_loader = torch.utils.data.DataLoader(anom_dataset, batch_size=1, num_workers=args_anom.num_workers)

# Let's see how the nlls of the ladder graphs compare to 
# the trained on enzymes.
# NOTE we pass in the args_norm because the model we are
# using are trained on the normal class dataset
anom_nlls, anom_avg_nlls = calc_nll(args_norm, anom_loader, rnn, output, max_iter=user_args.max_iter, log=1) #, max_iter=user_args.max_iter, load_epoch=user_args.load_epoch, train_dataset=user_args.dataset_norm, log=1)

anom_avg_graph_nlls = np.array(anom_avg_nlls)
anom_avg_graph_nlls = anom_avg_graph_nlls.reshape((user_args.max_iter, len(anom_loader)))
anom_avg_graph_nlls = np.mean(anom_avg_graph_nlls, axis=0)
np.save(nll_dir + user_args.dataset_anom + '_avg_graph_nlls.npy', train_avg_graph_nlls)

