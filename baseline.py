from main import *
from scipy.linalg import toeplitz
import pyemd
import scipy.optimize as opt

def Graph_generator_baseline_train_rulebased(graphs,generator='BA'):
    graph_nodes = [graphs[i].number_of_nodes() for i in range(len(graphs))]
    graph_edges = [graphs[i].number_of_edges() for i in range(len(graphs))]
    parameter = {}
    for i in range(len(graph_nodes)):
        nodes = graph_nodes[i]
        edges = graph_edges[i]
        # based on rule, calculate optimal parameter
        if generator=='BA':
            # BA optimal: nodes = n; edges = (n-m)*m
            n = nodes
            m = (n - np.sqrt(n**2-4*edges))/2
            parameter_temp = [n,m,1]
        if generator=='Gnp':
            # Gnp optimal: nodes = n; edges = ((n-1)*n/2)*p
            n = nodes
            p = float(edges)/((n-1)*n/2)
            parameter_temp = [n,p,1]
        # update parameter list
        if nodes not in parameter.keys():
            parameter[nodes] = parameter_temp
        else:
            count = parameter[nodes][-1]
            parameter[nodes] = [(parameter[nodes][i]*count+parameter_temp[i])/(count+1) for i in range(len(parameter[nodes]))]
            parameter[nodes][-1] = count+1
    # print(parameter)
    return parameter

def Graph_generator_baseline(graph_train, pred_num=1000, generator='BA'):
    graph_nodes = [graph_train[i].number_of_nodes() for i in range(len(graph_train))]
    graph_edges = [graph_train[i].number_of_edges() for i in range(len(graph_train))]
    repeat = pred_num//len(graph_train)
    graph_pred = []
    for i in range(len(graph_nodes)):
        nodes = graph_nodes[i]
        edges = graph_edges[i]
        # based on rule, calculate optimal parameter
        if generator=='BA':
            # BA optimal: nodes = n; edges = (n-m)*m
            n = nodes
            m = int((n - np.sqrt(n**2-4*edges))/2)
            for j in range(repeat):
                graph_pred.append(nx.barabasi_albert_graph(n,m))
        if generator=='Gnp':
            # Gnp optimal: nodes = n; edges = ((n-1)*n/2)*p
            n = nodes
            p = float(edges)/((n-1)*n/2)
            for j in range(repeat):
                graph_pred.append(nx.fast_gnp_random_graph(n, p))
    return graph_pred

def emd_distance(x, y, distance_scaling=1.0):
    support_size = max(len(x), len(y))
    d_mat = toeplitz(range(support_size)).astype(np.float)
    distance_mat = d_mat / distance_scaling

    # convert histogram values x and y to float, and make them equal len
    x = x.astype(np.float)
    y = y.astype(np.float)
    if len(x) < len(y):
        x = np.hstack((x, [0.0] * (support_size - len(x))))
    elif len(y) < len(x):
        y = np.hstack((y, [0.0] * (support_size - len(y))))

    emd = pyemd.emd(x, y, distance_mat)
    return emd

# def Loss(x,args):
#     '''
#
#     :param x: 1-D array, parameters to be optimized
#     :param args: tuple (n, G, generator, metric).
#     n: n for pred graph;
#     G: real graph in networkx format;
#     generator: 'BA', 'Gnp', 'Powerlaw';
#     metric: 'degree', 'clustering'
#     :return: Loss: emd distance
#     '''
#     # get argument
#     generator = args[2]
#     metric = args[3]
#
#     # get real and pred graphs
#     G_real = args[1]
#     if generator=='BA':
#         G_pred = nx.barabasi_albert_graph(args[0],int(np.rint(x)))
#     if generator=='Gnp':
#         G_pred = nx.fast_gnp_random_graph(args[0],x)
#
#     # define metric
#     if metric == 'degree':
#         G_real_hist = np.array(nx.degree_histogram(G_real))
#         G_real_hist = G_real_hist / np.sum(G_real_hist)
#         G_pred_hist = np.array(nx.degree_histogram(G_pred))
#         G_pred_hist = G_pred_hist/np.sum(G_pred_hist)
#     if metric == 'clustering':
#         G_real_hist, _ = np.histogram(
#             np.array(list(nx.clustering(G_real).values())), bins=50, range=(0.0, 1.0), density=False)
#         G_real_hist = G_real_hist / np.sum(G_real_hist)
#         G_pred_hist, _ = np.histogram(
#             np.array(list(nx.clustering(G_pred).values())), bins=50, range=(0.0, 1.0), density=False)
#         G_pred_hist = G_pred_hist / np.sum(G_pred_hist)
#
#     loss = emd_distance(G_real_hist,G_pred_hist)
#     return loss

def Loss(x,n,G_real,generator,metric):
    '''

    :param x: 1-D array, parameters to be optimized
    :param
    n: n for pred graph;
    G: real graph in networkx format;
    generator: 'BA', 'Gnp', 'Powerlaw';
    metric: 'degree', 'clustering'
    :return: Loss: emd distance
    '''
    # get argument

    # get real and pred graphs
    if generator=='BA':
        G_pred = nx.barabasi_albert_graph(n,int(np.rint(x)))
    if generator=='Gnp':
        G_pred = nx.fast_gnp_random_graph(n,x)

    # define metric
    if metric == 'degree':
        G_real_hist = np.array(nx.degree_histogram(G_real))
        G_real_hist = G_real_hist / np.sum(G_real_hist)
        G_pred_hist = np.array(nx.degree_histogram(G_pred))
        G_pred_hist = G_pred_hist/np.sum(G_pred_hist)
    if metric == 'clustering':
        G_real_hist, _ = np.histogram(
            np.array(list(nx.clustering(G_real).values())), bins=50, range=(0.0, 1.0), density=False)
        G_real_hist = G_real_hist / np.sum(G_real_hist)
        G_pred_hist, _ = np.histogram(
            np.array(list(nx.clustering(G_pred).values())), bins=50, range=(0.0, 1.0), density=False)
        G_pred_hist = G_pred_hist / np.sum(G_pred_hist)

    loss = emd_distance(G_real_hist,G_pred_hist)
    return loss

def optimizer_brute(x_min, x_max, x_step, n, G_real, generator, metric):
    loss_all = []
    x_list = np.arange(x_min,x_max,x_step)
    for x_test in x_list:
        loss_all.append(Loss(x_test,n,G_real,generator,metric))
    x_optim = x_list[np.argmin(np.array(loss_all))]
    return x_optim

def Graph_generator_baseline_train_optimizationbased(graphs,generator='BA',metric='degree'):
    graph_nodes = [graphs[i].number_of_nodes() for i in range(len(graphs))]
    parameter = {}
    for i in range(len(graph_nodes)):
        print('graph ',i)
        nodes = graph_nodes[i]
        if generator=='BA':
            n = nodes
            m = optimizer_brute(1,10,1, nodes, graphs[i], generator, metric)
            parameter_temp = [n,m,1]
        elif generator=='Gnp':
            n = nodes
            p = optimizer_brute(1e-6,1,0.01, nodes, graphs[i], generator, metric)
            ## if use evolution
            # result = opt.differential_evolution(Loss,bounds=[(0,1)],args=(nodes, graphs[i], generator, metric),maxiter=1000)
            # p = result.x
            parameter_temp = [n, p, 1]

        # update parameter list
        if nodes not in parameter.keys():
            parameter[nodes] = parameter_temp
        else:
            count = parameter[nodes][2]
            parameter[nodes] = [(parameter[nodes][i]*count+parameter_temp[i])/(count+1) for i in range(len(parameter[nodes]))]
            parameter[nodes][2] = count+1
    print(parameter)
    return parameter



def Graph_generator_baseline_test(graph_nodes, parameter, generator='BA'):
    graphs = []
    for i in range(len(graph_nodes)):
        nodes = graph_nodes[i]
        if not nodes in parameter.keys():
            nodes = min(parameter.keys(), key=lambda k: abs(k - nodes))
        if generator=='BA':
            n = int(parameter[nodes][0])
            m = int(np.rint(parameter[nodes][1]))
            print(n,m)
            graph = nx.barabasi_albert_graph(n,m)
        if generator=='Gnp':
            n = int(parameter[nodes][0])
            p = parameter[nodes][1]
            print(n,p)
            graph = nx.fast_gnp_random_graph(n,p)
        graphs.append(graph)
    return graphs


if __name__ == '__main__':
    args = Args()

    print('File name prefix', args.fname)
    ### load datasets
    graphs = []
    # synthetic graphs
    if args.graph_type=='ladder':
        graphs = []
        for i in range(100, 201):
            graphs.append(nx.ladder_graph(i))
        args.max_prev_node = 10
    if args.graph_type=='tree':
        graphs = []
        for i in range(2,5):
            for j in range(3,5):
                graphs.append(nx.balanced_tree(i,j))
        args.max_prev_node = 256
    if args.graph_type=='caveman':
        graphs = []
        for i in range(5,10):
            for j in range(5,25):
                    graphs.append(nx.connected_caveman_graph(i, j))
        args.max_prev_node = 50
    if args.graph_type=='grid':
        graphs = []
        for i in range(10,20):
            for j in range(10,20):
                graphs.append(nx.grid_2d_graph(i,j))
        args.max_prev_node = 40
    if args.graph_type=='barabasi':
        graphs = []
        for i in range(100,200):
            graphs.append(nx.barabasi_albert_graph(i,2))
        args.max_prev_node = 130
    # real graphs
    if args.graph_type == 'enzymes':
        graphs= Graph_load_batch(min_num_nodes=10, name='ENZYMES')
        args.max_prev_node = 25
    if args.graph_type == 'protein':
        graphs = Graph_load_batch(min_num_nodes=20, name='PROTEINS_full')
        args.max_prev_node = 80
    if args.graph_type == 'DD':
        graphs = Graph_load_batch(min_num_nodes=100, max_num_nodes=500, name='DD',node_attributes=False,graph_labels=True)
        args.max_prev_node = 230


    graph_nodes = [graphs[i].number_of_nodes() for i in range(len(graphs))]
    graph_edges = [graphs[i].number_of_edges() for i in range(len(graphs))]

    args.max_num_node = max(graph_nodes)

    # show graphs statistics
    print('total graph num: {}'.format(len(graphs)))
    print('max number node: {}'.format(args.max_num_node))
    print('max previous node: {}'.format(args.max_prev_node))

    # start baseline generation method

    generator = args.generator_baseline
    metric = args.metric_baseline
    print(args.fname_baseline + '.dat')

    if metric=='general':
        parameter = Graph_generator_baseline_train_rulebased(graphs,generator=generator)
    else:
        parameter = Graph_generator_baseline_train_optimizationbased(graphs,generator=generator,metric=metric)
    graphs_generated = Graph_generator_baseline_test(graph_nodes, parameter,generator)

    save_graph_list(graphs_generated,args.fname_baseline + '.dat')