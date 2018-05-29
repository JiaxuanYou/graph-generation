# an implementation for "Learning Deep Generative Models of Graphs"
from main import *

class Args_DGMG():
    def __init__(self):
        ### CUDA
        self.cuda = 2

        ### model type
        self.note = 'Baseline_DGMG' # do GCN after adding each edge
        # self.note = 'Baseline_DGMG_fast' # do GCN only after adding each node

        ### data config
        self.graph_type = 'caveman_small'
        # self.graph_type = 'grid_small'
        # self.graph_type = 'ladder_small'
        # self.graph_type = 'enzymes_small'
        # self.graph_type = 'barabasi_small'
        # self.graph_type = 'citeseer_small'

        self.max_num_node = 20

        ### network config
        self.node_embedding_size = 64
        self.test_graph_num = 200


        ### training config
        self.epochs = 2000  # now one epoch means self.batch_ratio x batch_size
        self.load_epoch = 2000
        self.epochs_test_start = 100
        self.epochs_test = 100
        self.epochs_log = 100
        self.epochs_save = 100
        if 'fast' in self.note:
            self.is_fast = True
        else:
            self.is_fast = False

        self.lr = 0.001
        self.milestones = [300, 600, 1000]
        self.lr_rate = 0.3

        ### output config
        self.model_save_path = 'model_save/'
        self.graph_save_path = 'graphs/'
        self.figure_save_path = 'figures/'
        self.timing_save_path = 'timing/'
        self.figure_prediction_save_path = 'figures_prediction/'
        self.nll_save_path = 'nll/'


        self.fname = self.note + '_' + self.graph_type + '_' + str(self.node_embedding_size)
        self.fname_pred = self.note + '_' + self.graph_type + '_' + str(self.node_embedding_size) + '_pred_'
        self.fname_train = self.note + '_' + self.graph_type + '_' + str(self.node_embedding_size) + '_train_'
        self.fname_test = self.note + '_' + self.graph_type + '_' + str(self.node_embedding_size) + '_test_'

        self.load = False
        self.save = True


def train_DGMG_epoch(epoch, args, model, dataset, optimizer, scheduler, is_fast = False):
    model.train()
    graph_num = len(dataset)
    order = list(range(graph_num))
    shuffle(order)


    loss_addnode = 0
    loss_addedge = 0
    loss_node = 0
    for i in order:
        model.zero_grad()

        graph = dataset[i]
        # do random ordering: relabel nodes
        node_order = list(range(graph.number_of_nodes()))
        shuffle(node_order)
        order_mapping = dict(zip(graph.nodes(), node_order))
        graph = nx.relabel_nodes(graph, order_mapping, copy=True)


        # NOTE: when starting loop, we assume a node has already been generated
        node_count = 1
        node_embedding = [Variable(torch.ones(1,args.node_embedding_size)).cuda()] # list of torch tensors, each size: 1*hidden


        loss = 0
        while node_count<=graph.number_of_nodes():
            node_neighbor = graph.subgraph(list(range(node_count))).adjacency_list()  # list of lists (first node is zero)
            node_neighbor_new = graph.subgraph(list(range(node_count+1))).adjacency_list()[-1] # list of new node's neighbors

            # 1 message passing
            # do 2 times message passing
            node_embedding = message_passing(node_neighbor, node_embedding, model)

            # 2 graph embedding and new node embedding
            node_embedding_cat = torch.cat(node_embedding, dim=0)
            graph_embedding = calc_graph_embedding(node_embedding_cat, model)
            init_embedding = calc_init_embedding(node_embedding_cat, model)

            # 3 f_addnode
            p_addnode = model.f_an(graph_embedding)
            if node_count < graph.number_of_nodes():
                # add node
                node_neighbor.append([])
                node_embedding.append(init_embedding)
                if is_fast:
                    node_embedding_cat = torch.cat(node_embedding, dim=0)
                # calc loss
                loss_addnode_step = F.binary_cross_entropy(p_addnode,Variable(torch.ones((1,1))).cuda())
                # loss_addnode_step.backward(retain_graph=True)
                loss += loss_addnode_step
                loss_addnode += loss_addnode_step.data
            else:
                # calc loss
                loss_addnode_step = F.binary_cross_entropy(p_addnode, Variable(torch.zeros((1, 1))).cuda())
                # loss_addnode_step.backward(retain_graph=True)
                loss += loss_addnode_step
                loss_addnode += loss_addnode_step.data
                break


            edge_count = 0
            while edge_count<=len(node_neighbor_new):
                if not is_fast:
                    node_embedding = message_passing(node_neighbor, node_embedding, model)
                    node_embedding_cat = torch.cat(node_embedding, dim=0)
                    graph_embedding = calc_graph_embedding(node_embedding_cat, model)

                # 4 f_addedge
                p_addedge = model.f_ae(graph_embedding)

                if edge_count < len(node_neighbor_new):
                    # calc loss
                    loss_addedge_step = F.binary_cross_entropy(p_addedge, Variable(torch.ones((1, 1))).cuda())
                    # loss_addedge_step.backward(retain_graph=True)
                    loss += loss_addedge_step
                    loss_addedge += loss_addedge_step.data

                    # 5 f_nodes
                    # excluding the last node (which is the new node)
                    node_new_embedding_cat = node_embedding_cat[-1,:].expand(node_embedding_cat.size(0)-1,node_embedding_cat.size(1))
                    s_node = model.f_s(torch.cat((node_embedding_cat[0:-1,:],node_new_embedding_cat),dim=1))
                    p_node = F.softmax(s_node.permute(1,0))
                    # get ground truth
                    a_node = torch.zeros((1,p_node.size(1)))
                    # print('node_neighbor_new',node_neighbor_new, edge_count)
                    a_node[0,node_neighbor_new[edge_count]] = 1
                    a_node = Variable(a_node).cuda()
                    # add edge
                    node_neighbor[-1].append(node_neighbor_new[edge_count])
                    node_neighbor[node_neighbor_new[edge_count]].append(len(node_neighbor)-1)
                    # calc loss
                    loss_node_step = F.binary_cross_entropy(p_node,a_node)
                    # loss_node_step.backward(retain_graph=True)
                    loss += loss_node_step
                    loss_node += loss_node_step.data

                else:
                    # calc loss
                    loss_addedge_step = F.binary_cross_entropy(p_addedge, Variable(torch.zeros((1, 1))).cuda())
                    # loss_addedge_step.backward(retain_graph=True)
                    loss += loss_addedge_step
                    loss_addedge += loss_addedge_step.data
                    break

                edge_count += 1
            node_count += 1

        # update deterministic and lstm
        loss.backward()
        optimizer.step()
        scheduler.step()

    loss_all = loss_addnode + loss_addedge + loss_node

    if epoch % args.epochs_log==0:
        print('Epoch: {}/{}, train loss: {:.6f}, graph type: {}, hidden: {}'.format(
            epoch, args.epochs,loss_all[0], args.graph_type, args.node_embedding_size))


    # loss_sum += loss.data[0]*x.size(0)
    # return loss_sum




def train_DGMG_forward_epoch(args, model, dataset, is_fast = False):
    model.train()
    graph_num = len(dataset)
    order = list(range(graph_num))
    shuffle(order)


    loss_addnode = 0
    loss_addedge = 0
    loss_node = 0
    for i in order:
        model.zero_grad()

        graph = dataset[i]
        # do random ordering: relabel nodes
        node_order = list(range(graph.number_of_nodes()))
        shuffle(node_order)
        order_mapping = dict(zip(graph.nodes(), node_order))
        graph = nx.relabel_nodes(graph, order_mapping, copy=True)


        # NOTE: when starting loop, we assume a node has already been generated
        node_count = 1
        node_embedding = [Variable(torch.ones(1,args.node_embedding_size)).cuda()] # list of torch tensors, each size: 1*hidden


        loss = 0
        while node_count<=graph.number_of_nodes():
            node_neighbor = graph.subgraph(list(range(node_count))).adjacency_list()  # list of lists (first node is zero)
            node_neighbor_new = graph.subgraph(list(range(node_count+1))).adjacency_list()[-1] # list of new node's neighbors

            # 1 message passing
            # do 2 times message passing
            node_embedding = message_passing(node_neighbor, node_embedding, model)

            # 2 graph embedding and new node embedding
            node_embedding_cat = torch.cat(node_embedding, dim=0)
            graph_embedding = calc_graph_embedding(node_embedding_cat, model)
            init_embedding = calc_init_embedding(node_embedding_cat, model)

            # 3 f_addnode
            p_addnode = model.f_an(graph_embedding)
            if node_count < graph.number_of_nodes():
                # add node
                node_neighbor.append([])
                node_embedding.append(init_embedding)
                if is_fast:
                    node_embedding_cat = torch.cat(node_embedding, dim=0)
                # calc loss
                loss_addnode_step = F.binary_cross_entropy(p_addnode,Variable(torch.ones((1,1))).cuda())
                # loss_addnode_step.backward(retain_graph=True)
                loss += loss_addnode_step
                loss_addnode += loss_addnode_step.data
            else:
                # calc loss
                loss_addnode_step = F.binary_cross_entropy(p_addnode, Variable(torch.zeros((1, 1))).cuda())
                # loss_addnode_step.backward(retain_graph=True)
                loss += loss_addnode_step
                loss_addnode += loss_addnode_step.data
                break


            edge_count = 0
            while edge_count<=len(node_neighbor_new):
                if not is_fast:
                    node_embedding = message_passing(node_neighbor, node_embedding, model)
                    node_embedding_cat = torch.cat(node_embedding, dim=0)
                    graph_embedding = calc_graph_embedding(node_embedding_cat, model)

                # 4 f_addedge
                p_addedge = model.f_ae(graph_embedding)

                if edge_count < len(node_neighbor_new):
                    # calc loss
                    loss_addedge_step = F.binary_cross_entropy(p_addedge, Variable(torch.ones((1, 1))).cuda())
                    # loss_addedge_step.backward(retain_graph=True)
                    loss += loss_addedge_step
                    loss_addedge += loss_addedge_step.data

                    # 5 f_nodes
                    # excluding the last node (which is the new node)
                    node_new_embedding_cat = node_embedding_cat[-1,:].expand(node_embedding_cat.size(0)-1,node_embedding_cat.size(1))
                    s_node = model.f_s(torch.cat((node_embedding_cat[0:-1,:],node_new_embedding_cat),dim=1))
                    p_node = F.softmax(s_node.permute(1,0))
                    # get ground truth
                    a_node = torch.zeros((1,p_node.size(1)))
                    # print('node_neighbor_new',node_neighbor_new, edge_count)
                    a_node[0,node_neighbor_new[edge_count]] = 1
                    a_node = Variable(a_node).cuda()
                    # add edge
                    node_neighbor[-1].append(node_neighbor_new[edge_count])
                    node_neighbor[node_neighbor_new[edge_count]].append(len(node_neighbor)-1)
                    # calc loss
                    loss_node_step = F.binary_cross_entropy(p_node,a_node)
                    # loss_node_step.backward(retain_graph=True)
                    loss += loss_node_step
                    loss_node += loss_node_step.data*p_node.size(1)

                else:
                    # calc loss
                    loss_addedge_step = F.binary_cross_entropy(p_addedge, Variable(torch.zeros((1, 1))).cuda())
                    # loss_addedge_step.backward(retain_graph=True)
                    loss += loss_addedge_step
                    loss_addedge += loss_addedge_step.data
                    break

                edge_count += 1
            node_count += 1


    loss_all = loss_addnode + loss_addedge + loss_node

    # if epoch % args.epochs_log==0:
    #     print('Epoch: {}/{}, train loss: {:.6f}, graph type: {}, hidden: {}'.format(
    #         epoch, args.epochs,loss_all[0], args.graph_type, args.node_embedding_size))


    return loss_all[0]/len(dataset)







def test_DGMG_epoch(args, model, is_fast=False):
    model.eval()
    graph_num = args.test_graph_num

    graphs_generated = []
    for i in range(graph_num):
        # NOTE: when starting loop, we assume a node has already been generated
        node_neighbor = [[]]  # list of lists (first node is zero)
        node_embedding = [Variable(torch.ones(1,args.node_embedding_size)).cuda()] # list of torch tensors, each size: 1*hidden

        node_count = 1
        while node_count<=args.max_num_node:
            # 1 message passing
            # do 2 times message passing
            node_embedding = message_passing(node_neighbor, node_embedding, model)

            # 2 graph embedding and new node embedding
            node_embedding_cat = torch.cat(node_embedding, dim=0)
            graph_embedding = calc_graph_embedding(node_embedding_cat, model)
            init_embedding = calc_init_embedding(node_embedding_cat, model)

            # 3 f_addnode
            p_addnode = model.f_an(graph_embedding)
            a_addnode = sample_tensor(p_addnode)
            # print(a_addnode.data[0][0])
            if a_addnode.data[0][0]==1:
                # print('add node')
                # add node
                node_neighbor.append([])
                node_embedding.append(init_embedding)
                if is_fast:
                    node_embedding_cat = torch.cat(node_embedding, dim=0)
            else:
                break

            edge_count = 0
            while edge_count<args.max_num_node:
                if not is_fast:
                    node_embedding = message_passing(node_neighbor, node_embedding, model)
                    node_embedding_cat = torch.cat(node_embedding, dim=0)
                    graph_embedding = calc_graph_embedding(node_embedding_cat, model)

                # 4 f_addedge
                p_addedge = model.f_ae(graph_embedding)
                a_addedge = sample_tensor(p_addedge)
                # print(a_addedge.data[0][0])

                if a_addedge.data[0][0]==1:
                    # print('add edge')
                    # 5 f_nodes
                    # excluding the last node (which is the new node)
                    node_new_embedding_cat = node_embedding_cat[-1,:].expand(node_embedding_cat.size(0)-1,node_embedding_cat.size(1))
                    s_node = model.f_s(torch.cat((node_embedding_cat[0:-1,:],node_new_embedding_cat),dim=1))
                    p_node = F.softmax(s_node.permute(1,0))
                    a_node = gumbel_softmax(p_node, temperature=0.01)
                    _, a_node_id = a_node.topk(1)
                    a_node_id = int(a_node_id.data[0][0])
                    # add edge
                    node_neighbor[-1].append(a_node_id)
                    node_neighbor[a_node_id].append(len(node_neighbor)-1)
                else:
                    break

                edge_count += 1
            node_count += 1
        # save graph
        node_neighbor_dict = dict(zip(list(range(len(node_neighbor))), node_neighbor))
        graph = nx.from_dict_of_lists(node_neighbor_dict)
        graphs_generated.append(graph)

    return graphs_generated












########### train function for LSTM + VAE
def train_DGMG(args, dataset_train, model):
    # check if load existing model
    if args.load:
        fname = args.model_save_path + args.fname + 'model_' + str(args.load_epoch) + '.dat'
        model.load_state_dict(torch.load(fname))

        args.lr = 0.00001
        epoch = args.load_epoch
        print('model loaded!, lr: {}'.format(args.lr))
    else:
        epoch = 1

    # initialize optimizer
    optimizer = optim.Adam(list(model.parameters()), lr=args.lr)

    scheduler = MultiStepLR(optimizer, milestones=args.milestones, gamma=args.lr_rate)

    # start main loop
    time_all = np.zeros(args.epochs)
    while epoch <= args.epochs:
        time_start = tm.time()
        # train
        train_DGMG_epoch(epoch, args, model, dataset_train, optimizer, scheduler, is_fast=args.is_fast)
        time_end = tm.time()
        time_all[epoch - 1] = time_end - time_start
        # print('time used',time_all[epoch - 1])
        # test
        if epoch % args.epochs_test == 0 and epoch >= args.epochs_test_start:
            graphs = test_DGMG_epoch(args,model, is_fast=args.is_fast)
            fname = args.graph_save_path + args.fname_pred + str(epoch) + '.dat'
            save_graph_list(graphs, fname)
            # print('test done, graphs saved')

        # save model checkpoint
        if args.save:
            if epoch % args.epochs_save == 0:
                fname = args.model_save_path + args.fname + 'model_' + str(epoch) + '.dat'
                torch.save(model.state_dict(), fname)
        epoch += 1
    np.save(args.timing_save_path + args.fname, time_all)







########### train function for LSTM + VAE
def train_DGMG_nll(args, dataset_train,dataset_test, model,max_iter=1000):
    # check if load existing model
    fname = args.model_save_path + args.fname + 'model_' + str(args.load_epoch) + '.dat'
    model.load_state_dict(torch.load(fname))

    fname_output = args.nll_save_path + args.note + '_' + args.graph_type + '.csv'
    with open(fname_output, 'w+') as f:
        f.write('train,test\n')
        # start main loop
        for iter in range(max_iter):
            nll_train = train_DGMG_forward_epoch(args, model, dataset_train, is_fast=args.is_fast)
            nll_test = train_DGMG_forward_epoch(args, model, dataset_test, is_fast=args.is_fast)
            print('train', nll_train, 'test', nll_test)
            f.write(str(nll_train) + ',' + str(nll_test) + '\n')





if __name__ == '__main__':
    args = Args_DGMG()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)
    print('CUDA', args.cuda)
    print('File name prefix',args.fname)


    graphs = []
    for i in range(4, 10):
        graphs.append(nx.ladder_graph(i))
    model = DGM_graphs(h_size = args.node_embedding_size).cuda()

    if args.graph_type == 'ladder_small':
        graphs = []
        for i in range(2, 11):
            graphs.append(nx.ladder_graph(i))
        args.max_prev_node = 10
    # if args.graph_type == 'caveman_small':
    #     graphs = []
    #     for i in range(2, 5):
    #         for j in range(2, 6):
    #             for k in range(10):
    #                 graphs.append(nx.relaxed_caveman_graph(i, j, p=0.1))
    #     args.max_prev_node = 20
    if args.graph_type=='caveman_small':
        graphs = []
        for i in range(2, 3):
            for j in range(6, 11):
                for k in range(20):
                    graphs.append(caveman_special(i, j, p_edge=0.8))
        args.max_prev_node = 20
    if args.graph_type == 'grid_small':
        graphs = []
        for i in range(2, 5):
            for j in range(2, 6):
                graphs.append(nx.grid_2d_graph(i, j))
        args.max_prev_node = 15
    if args.graph_type == 'barabasi_small':
        graphs = []
        for i in range(4, 21):
            for j in range(3, 4):
                for k in range(10):
                    graphs.append(nx.barabasi_albert_graph(i, j))
        args.max_prev_node = 20

    if args.graph_type == 'enzymes_small':
        graphs_raw = Graph_load_batch(min_num_nodes=10, name='ENZYMES')
        graphs = []
        for G in graphs_raw:
            if G.number_of_nodes()<=20:
                graphs.append(G)
        args.max_prev_node = 15

    if args.graph_type == 'citeseer_small':
        _, _, G = Graph_load(dataset='citeseer')
        G = max(nx.connected_component_subgraphs(G), key=len)
        G = nx.convert_node_labels_to_integers(G)
        graphs = []
        for i in range(G.number_of_nodes()):
            G_ego = nx.ego_graph(G, i, radius=1)
            if (G_ego.number_of_nodes() >= 4) and (G_ego.number_of_nodes() <= 20):
                graphs.append(G_ego)
        shuffle(graphs)
        graphs = graphs[0:200]
        args.max_prev_node = 15

    # remove self loops
    for graph in graphs:
        edges_with_selfloops = graph.selfloop_edges()
        if len(edges_with_selfloops) > 0:
            graph.remove_edges_from(edges_with_selfloops)

    # split datasets
    random.seed(123)
    shuffle(graphs)
    graphs_len = len(graphs)
    graphs_test = graphs[int(0.8 * graphs_len):]
    graphs_train = graphs[0:int(0.8 * graphs_len)]

    args.max_num_node = max([graphs[i].number_of_nodes() for i in range(len(graphs))])
    # args.max_num_node = 2000
    # show graphs statistics
    print('total graph num: {}, training set: {}'.format(len(graphs), len(graphs_train)))
    print('max number node: {}'.format(args.max_num_node))
    print('max previous node: {}'.format(args.max_prev_node))

    # save ground truth graphs
    # save_graph_list(graphs, args.graph_save_path + args.fname_train + '0.dat')
    # save_graph_list(graphs, args.graph_save_path + args.fname_test + '0.dat')
    # print('train and test graphs saved')

    ## if use pre-saved graphs
    # dir_input = "graphs/"
    # fname_test = args.graph_save_path + args.fname_test + '0.dat'
    # graphs = load_graph_list(fname_test, is_real=True)
    # graphs_test = graphs[int(0.8 * graphs_len):]
    # graphs_train = graphs[0:int(0.8 * graphs_len)]
    # graphs_validate = graphs[0:int(0.2 * graphs_len)]

    # print('train')
    # for graph in graphs_validate:
    #     print(graph.number_of_nodes())
    # print('test')
    # for graph in graphs_test:
    #     print(graph.number_of_nodes())



    ### train
    train_DGMG(args,graphs,model)

    ### calc nll
    # train_DGMG_nll(args, graphs_validate,graphs_test, model,max_iter=1000)









    # for j in range(1000):
    #     graph = graphs[0]
    #     # do random ordering: relabel nodes
    #     node_order = list(range(graph.number_of_nodes()))
    #     shuffle(node_order)
    #     order_mapping = dict(zip(graph.nodes(), node_order))
    #     graph = nx.relabel_nodes(graph, order_mapping, copy=True)
    #     print(graph.nodes())