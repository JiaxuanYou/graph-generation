from main import *

class Args_DGMG():
    def __init__(self):
        ### CUDA
        self.cuda = 2

        ### model type
        self.note = 'Baseline_DGMG'

        ### data config
        ## used for paper
        # self.graph_type = 'caveman_small'
        # self.graph_type = 'grid_small'
        self.graph_type = 'ladder_small'

        # self.graph_type = 'enzymes'
        # self.graph_type = 'barabasi_small'
        # self.graph_type = 'citeseer_small'
        # self.graph_type = 'barabasi_noise'


        ### network config
        self.node_embedding_size = 8
        self.test_graph_num = 100


        ### training config
        self.epochs = 2000  # now one epoch means self.batch_ratio x batch_size
        self.epochs_test_start = 0
        self.epochs_test = 0
        self.epochs_log = 1
        self.epochs_save = 10

        self.lr = 0.01
        self.milestones = [200, 500, 1000]
        self.lr_rate = 0.3

        ### output config
        self.load = False
        self.save = False


def train_DGMG_epoch(epoch, args, model, dataset, optimizer, scheduler):
    # todo: do random ordering
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

        # NOTE: when starting loop, we assume a node has already been generated
        node_count = 1
        node_embedding = [Variable(torch.ones(1,args.node_embedding_size)).cuda()] # list of torch tensors, each size: 1*hidden

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
                # calc loss
                loss_addnode_step = F.binary_cross_entropy(p_addnode,Variable(torch.ones((1,1))).cuda())
                loss_addnode_step.backward(retain_graph=True)
                loss_addnode += loss_addnode_step.data
            else:
                # calc loss
                loss_addnode_step = F.binary_cross_entropy(p_addnode, Variable(torch.zeros((1, 1))).cuda())
                loss_addnode_step.backward(retain_graph=True)
                loss_addnode += loss_addnode_step.data
                break

            edge_count = 0
            while edge_count<=len(node_neighbor_new):
                node_embedding = message_passing(node_neighbor, node_embedding, model)
                node_embedding_cat = torch.cat(node_embedding, dim=0)
                graph_embedding = calc_graph_embedding(node_embedding_cat, model)

                # 4 f_addedge
                p_addedge = model.f_ae(graph_embedding)

                if edge_count < len(node_neighbor_new):
                    # calc loss
                    loss_addedge_step = F.binary_cross_entropy(p_addedge, Variable(torch.ones((1, 1))).cuda())
                    loss_addedge_step.backward(retain_graph=True)
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
                    # calc loss
                    loss_node_step = F.binary_cross_entropy(p_node,a_node)
                    loss_node_step.backward(retain_graph=True)
                    loss_node += loss_node_step.data

                else:
                    # calc loss
                    loss_addedge_step = F.binary_cross_entropy(p_addedge, Variable(torch.zeros((1, 1))).cuda())
                    loss_addedge_step.backward(retain_graph=True)
                    loss_addedge += loss_addedge_step.data
                    break

                edge_count += 1
            node_count += 1

        # update deterministic and lstm
        optimizer.step()
        scheduler.step()

    loss = loss_addnode + loss_addedge + loss_node

    if epoch % args.epochs_log==0:
        print('Epoch: {}/{}, train loss: {:.6f}, graph type: {}, hidden: {}'.format(
            epoch, args.epochs,loss[0], args.graph_type, args.node_embedding_size))


    # loss_sum += loss.data[0]*x.size(0)
    # return loss_sum




def test_DGMG_epoch(epoch, args, model, optimizer, scheduler):
    # todo: do random ordering
    model.eval()
    graph_num = args.test_graph_num

    graphs_generated = []
    for i in range(graph_num):
        graph = nx.complete_graph(1)

        # NOTE: when starting loop, we assume a node has already been generated
        node_count = 1
        node_embedding = [Variable(torch.ones(1,args.node_embedding_size)).cuda()] # list of torch tensors, each size: 1*hidden

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
                # calc loss
                loss_addnode_step = F.binary_cross_entropy(p_addnode,Variable(torch.ones((1,1))).cuda())
                loss_addnode_step.backward(retain_graph=True)
                loss_addnode += loss_addnode_step.data
            else:
                # calc loss
                loss_addnode_step = F.binary_cross_entropy(p_addnode, Variable(torch.zeros((1, 1))).cuda())
                loss_addnode_step.backward(retain_graph=True)
                loss_addnode += loss_addnode_step.data
                break

            edge_count = 0
            while edge_count<=len(node_neighbor_new):
                node_embedding = message_passing(node_neighbor, node_embedding, model)
                node_embedding_cat = torch.cat(node_embedding, dim=0)
                graph_embedding = calc_graph_embedding(node_embedding_cat, model)

                # 4 f_addedge
                p_addedge = model.f_ae(graph_embedding)

                if edge_count < len(node_neighbor_new):
                    # calc loss
                    loss_addedge_step = F.binary_cross_entropy(p_addedge, Variable(torch.ones((1, 1))).cuda())
                    loss_addedge_step.backward(retain_graph=True)
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
                    # calc loss
                    loss_node_step = F.binary_cross_entropy(p_node,a_node)
                    loss_node_step.backward(retain_graph=True)
                    loss_node += loss_node_step.data

                else:
                    # calc loss
                    loss_addedge_step = F.binary_cross_entropy(p_addedge, Variable(torch.zeros((1, 1))).cuda())
                    loss_addedge_step.backward(retain_graph=True)
                    loss_addedge += loss_addedge_step.data
                    break

                edge_count += 1
            node_count += 1

        # update deterministic and lstm
        optimizer.step()
        scheduler.step()

    loss = loss_addnode + loss_addedge + loss_node

    if epoch % args.epochs_log==0:
        print('Epoch: {}/{}, train loss: {:.6f}, graph type: {}, hidden: {}'.format(
            epoch, args.epochs,loss[0], args.graph_type, args.node_embedding_size))


    # loss_sum += loss.data[0]*x.size(0)
    # return loss_sum










########### train function for LSTM + VAE
def train_DGMG(args, dataset_train, model):
    # check if load existing model
    if args.load:
        fname = args.model_save_path + args.fname + 'model_' + str(args.load_epoch) + '.dat'
        rnn.load_state_dict(torch.load(fname))

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
        train_DGMG_epoch(epoch, args, model, dataset_train, optimizer, scheduler)
        time_end = tm.time()
        time_all[epoch - 1] = time_end - time_start
        print('time used',time_all[epoch - 1])
        # # test
        # if epoch % args.epochs_test == 0 and epoch >= args.epochs_test_start:
        #     print('test done, graphs saved')

        # save model checkpoint
        if args.save:
            if epoch % args.epochs_save == 0:
                fname = args.model_save_path + args.fname + 'model_' + str(epoch) + '.dat'
                torch.save(rnn.state_dict(), fname)
        epoch += 1
    np.save(args.timing_save_path + args.fname, time_all)

if __name__ == '__main__':
    args = Args_DGMG()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda)
    print('CUDA', args.cuda)
    # print('File name prefix', args.fname)

    graphs = []
    for i in range(10, 26):
        graphs.append(nx.ladder_graph(i))
    model = DGM_graphs(h_size = args.node_embedding_size).cuda()

    train_DGMG(args,graphs,model)









# def train_DGMG_epoch(epoch, args, model, dataset, optimizer, scheduler):
#     rnn.train()
#     output.train()
#     graph_num = len(dataset)
#     order = shuffle(range(graph_num))
#
#     loss_sum = 0
#     for i in order:
#         rnn.zero_grad()
#         output.zero_grad()
#         x_unsorted = dataset[i]['x'].float()
#
#         # initialize lstm hidden state according to batch size
#         rnn.hidden = rnn.init_hidden(batch_size=x_unsorted.size(0))
#
#         x = Variable(x).cuda()
#         y = Variable(y).cuda()
#
#         # get node_neighbor_real through loading data
#         node_neighbor_real = []
#
#         node_neighbor = [] # list of lists
#         node_embedding = [] # list of torch tensors, each size: 1*hidden
#
#
#         # NOTE: when starting loop, we assume a node has already been generated
#         node_count = 1
#         while True:
#             # 1 message passing
#             # do 2 times message passing
#             node_embedding = message_passing(node_neighbor, node_embedding, model)
#
#             # 2 graph embedding and new node embedding
#             node_embedding_cat = torch.cat(node_embedding, dim=0)
#             graph_embedding = calc_graph_embedding(node_embedding_cat, model)
#             init_embedding = calc_init_embedding(node_embedding_cat, model)
#
#             # 3 f_addnode
#             p_addnode = model.f_an(graph_embedding)
#             if node_count>=len(node_neighbor_real): # node count exceeding ground truth, break
#                 break
#             # else:
#             #     a_addnode = sample_tensor(p_addnode)
#             #     if a_addnode==0:
#             #         break
#
#             edge_count = 0
#             while True:
#                 # 4 f_addedge
#                 p_addedge = model.f_ae(graph_embedding)
#                 if edge_count >= len(node_neighbor_real):  # edge count exceeding ground truth, break
#                     break
#                 # else:
#                 #     a_addedge = sample_tensor(p_addedge)
#                 #     if a_addedge == 0:
#                 #         break
#
#                 # 5 f_nodes
#                 init_embedding_cat = init_embedding.expand(node_embedding_cat.size(0),init_embedding.size(1))
#                 s_node = model.f_s(torch.cat((node_embedding_cat,init_embedding_cat),dim=1))
#                 p_node = F.softmax(s_node.permute(1,0))
#                 # todo: put groud truth here
#
#                 # else:
#                 #     a_node = gumbel_softmax(p_node, temperature=0.01)
#                 #     _, a_node_id = a_node.topk(1)
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#         h = rnn(x, pack=True, input_len=y_len)
#         y_pred = output(h)
#         y_pred = F.sigmoid(y_pred)
#         # clean
#         y_pred = pack_padded_sequence(y_pred, y_len, batch_first=True)
#         y_pred = pad_packed_sequence(y_pred, batch_first=True)[0]
#         # use cross entropy loss
#         loss = binary_cross_entropy_weight(y_pred, y)
#         loss.backward()
#         # update deterministic and lstm
#         optimizer_output.step()
#         optimizer_rnn.step()
#         scheduler_output.step()
#         scheduler_rnn.step()
#
#
#         if epoch % args.epochs_log==0 and batch_idx==0: # only output first batch's statistics
#             print('Epoch: {}/{}, train loss: {:.6f}, graph type: {}, num_layer: {}, hidden: {}'.format(
#                 epoch, args.epochs,loss.data[0], args.graph_type, args.num_layers, args.hidden_size_rnn))
#
#         # logging
#         log_value('loss_'+args.fname, loss.data[0], epoch*args.batch_ratio+batch_idx)
#
#         loss_sum += loss.data[0]*x.size(0)
#     return loss_sum