from train import *

if __name__ == '__main__':
    # All necessary arguments are defined in args.py
    args = Args()
    

    print('CUDA Available:', torch.cuda.is_available())
    print('File name prefix',args.fname)
    # check if necessary directories exist
    if not os.path.isdir(args.model_save_path):
        os.makedirs(args.model_save_path)
    if not os.path.isdir(args.graph_save_path):
        os.makedirs(args.graph_save_path)
    if not os.path.isdir(args.figure_save_path):
        os.makedirs(args.figure_save_path)
    if not os.path.isdir(args.timing_save_path):
        os.makedirs(args.timing_save_path)
    if not os.path.isdir(args.figure_prediction_save_path):
        os.makedirs(args.figure_prediction_save_path)
    if not os.path.isdir(args.nll_save_path):
        os.makedirs(args.nll_save_path)
    
    time = strftime("%Y-%m-%d %H:%M:%S", gmtime())
    # logging.basicConfig(filename='logs/train' + time + '.log', level=logging.DEBUG)
    if args.clean_tensorboard:
        if os.path.isdir("tensorboard"):
            shutil.rmtree("tensorboard")
    configure("tensorboard/run"+time, flush_secs=5)

    graphs, labels, num_classes = create_graphs.create_graph_class(args) # Need to get the number of classes from create_graphs!!!
    
    # split datasets
    random.seed(8)
    grouped_list = list(zip(graphs, labels))
    random.shuffle(grouped_list)
    graphs, labels = zip(*grouped_list)

    graphs_len = len(graphs)
    graphs_train = graphs[0:int(0.8*graphs_len)]
    labels_train = labels[0:int(0.8*graphs_len)]
    graphs_test = graphs[int(0.8 * graphs_len):]
    labels_test = labels[int(0.8 * graphs_len):]

    graph_test_len = 0
    for graph in graphs_test:
        graph_test_len += graph.number_of_nodes()
    graph_test_len /= len(graphs_test)
    print('graph_test_len', graph_test_len)

    # This is used in the GraphRNN!
    # Should maybe be over just train graph
    args.max_num_node = max([graphs[i].number_of_nodes() for i in range(len(graphs))])
    print (args.max_num_node)
    max_num_edge = max([graphs[i].number_of_edges() for i in range(len(graphs))])
    min_num_edge = min([graphs[i].number_of_edges() for i in range(len(graphs))])

    # show graphs statistics
    print('total graph num: {}, training set: {}'.format(len(graphs),len(graphs_train)))
    print('max number node: {}'.format(args.max_num_node))
    print('max/min number edge: {}; {}'.format(max_num_edge,min_num_edge))
    # This is important because it defines how far back the GraphRNN
    # has to predict
    print('max previous node: {}'.format(args.max_prev_node))

    # Save ground truth graphs
    # To get train and test set, after loading you need to manually slice
    save_graph_list(graphs, args.graph_save_path + args.fname_train + '0.dat')
    save_graph_list(graphs, args.graph_save_path + args.fname_test + '0.dat')
    print('train and test graphs saved at: ', args.graph_save_path + args.fname_test + '0.dat')

    # We should do the max_prev_node for the random graphs!
    # Need to include the graph labels!!!
    # Note use the updated graph sampler with random number generation fixed!
    dataset_train = Graph_sequence_sampler_pytorch_rand_graph_class(graphs_train, labels_train,max_prev_node=args.max_prev_node,max_num_node=args.max_num_node)
    dataset_test = Graph_sequence_sampler_pytorch_rand_graph_class(graphs_test, labels_test,max_prev_node=args.max_prev_node,max_num_node=args.max_num_node)

    if args.max_prev_node is None:
        args.max_prev_node = dataset_train.max_prev_node

    # Do we want to randomly sample or just go over all of the graphs??/
    sample_strategy = torch.utils.data.sampler.WeightedRandomSampler([1.0 / len(dataset_train) for i in range(len(dataset_train))],
                                                                     num_samples=args.batch_size*args.batch_ratio, replacement=True)
    dataset_loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, num_workers=args.num_workers,
                                               sampler=sample_strategy)

    #sample_strategy = torch.utils.data.sampler.WeightedRandomSampler([1.0 / len(dataset_test) for i in range(len(dataset_test))],
    #                                                                 num_samples=args.batch_size*args.batch_ratio, replacement=True)
    # Should shuffle, but do not need to randomly sample! We actually do want to go over all of the graph
    dataset_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True)
                                               #sampler=sample_strategy)


    
    rnn = GRU_Graph_Class(input_size=args.max_prev_node, embedding_size=args.embedding_size_rnn,
                hidden_size=args.hidden_size_rnn, num_layers=args.num_layers, has_input=True,
                has_output=True, output_size=args.hidden_size_rnn_output, classes=num_classes).to(device)

    output = GRU_plain(input_size=1, embedding_size=args.embedding_size_rnn_output,
                       hidden_size=args.hidden_size_rnn_output, num_layers=args.num_layers, has_input=True,
                       has_output=True, output_size=1).to(device)


    ### start training
    # Make sure we create new directories to save this shit
    train_graph_class(args, dataset_loader_train, dataset_loader_test, rnn, output)
