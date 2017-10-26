from train_gcn import *
from random import shuffle
from time import gmtime, strftime

class Args():
    def __init__(self):
        self.nodes_num = 6
        self.batch_size_train = 700
        self.batch_size_test = 39
        self.epochs = 4000
        self.epochs_log = 4
        self.epochs_test = 4

        self.seed = 1
        self.shuffle = True
        self.num_workers = 0
        self.lr = 0.01
        self.lr_rate = 0.1
        self.momentum = 0.5

        # encoder
        self.input_dim = 18 # for enzyme dataset
        self.output_dim = 16
        self.hidden_dim = 32

        self.milestones = [100, 400]
if __name__ == '__main__':
    # clean logging directory
    # if os.path.isdir("logs"):
    #     shutil.rmtree("logs")
    # configure("logs/logs_toy", flush_secs=30)
    time = strftime("%Y-%m-%d %H:%M:%S", gmtime())
    logging.basicConfig(filename='logs/train'+time+'.log',level=logging.DEBUG)

    args = Args()
    torch.manual_seed(args.seed)

    ######### load graph #########
    # G = nx.karate_club_graph()
    # adj, features, G = Graph_load()
    # features = features.toarray()
    # dict = nx.get_node_attributes(G)



    # G = nx.LCF_graph(14,[5,-5],7)
    # G = nx.LCF_graph(20,[-9,-9],10)
    # G, embedding = Graph_synthetic(10)
    # args.input_dim = G.number_of_nodes()

    graphs = Graph_load_batch(min_num_nodes=args.nodes_num,name='PROTEINS_full')
    # graphs = Graph_load_batch(min_num_nodes=args.nodes_num,name='ENZYMES')

    shuffle(graphs)
    dictionary = nx.get_node_attributes(graphs[0], 'feature')
    args.input_dim = list(dictionary.values())[0].shape[0]

    ############ for single graph
    # dataset_train = GraphDataset_adj(G, features=features)
    # dataset_test = GraphDataset_adj(G, features=features)
    # args.batch_size_train = 1
    # args.batch_size_train = 1
    # args.input_dim = features.shape[1]

    # ############## for multiple graphs
    split = int(len(graphs)*0.9)
    print('split',split)
    dataset_train = GraphDataset_adj_batch_1(graphs[0:split])
    dataset_test = GraphDataset_adj_batch_1(graphs[split:])
    args.batch_size_train = 1
    args.batch_size_test = 1

    # split = args.batch_size_train
    # dataset_train = GraphDataset_adj_batch(graphs[0:split], num_nodes=args.nodes_num)
    # dataset_test = GraphDataset_adj_batch(graphs[split:], num_nodes=args.nodes_num)





    encoder = GCN_encoder(args)
    decoder = GCN_decoder()
    train(args, dataset_train, dataset_test, encoder, decoder)

    # for lr in [0.03, 0.003, 0.0003]:
    #     for output_dim in [4, 8, 16, 32]:
    #         args.lr = lr
    #         args.output_dim = output_dim
    #         args.hidden_dim = output_dim*2
    #         print(args.lr, args.output_dim, args.hidden_dim)
    #         # dataset = GraphDataset_adj(G)
    #         encoder = GCN_encoder(args)
    #         decoder = GCN_decoder()
    #         train(args,dataset, encoder, decoder)