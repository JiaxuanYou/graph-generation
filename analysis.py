# this file is used to plot images
from main import *

args = Args()
print(args.graph_type)
# epoch = 16000
epoch = 50000

# for baseline model
if args.note == 'GraphRNN' or 'GraphRNN_VAE' or 'GraphRNN_VAE_nobn' or 'GraphRNN_VAE_simple':
    for num_layers in range(4,5):
        # give file name and figure name
        fname_pred = args.graph_save_path + args.note + '_' + args.graph_type + '_' + \
                     str(epoch) + '_pred_' + str(num_layers) + '_' + str(args.bptt) + '_' + str(args.bptt_len)+ '_' + str(args.gumbel)
        fname_real = args.graph_save_path + args.note + '_' + args.graph_type + '_' + \
                     str(epoch) + '_real_' + str(num_layers) + '_' + str(args.bptt) + '_' + str(args.bptt_len)+ '_' + str(args.gumbel)
        figname = args.figure_save_path + args.note + '_' + args.graph_type + '_' + \
                     str(epoch) + str(num_layers) + '_' + str(args.bptt) + '_' + str(args.bptt_len)+ '_' + str(args.gumbel)

        # fname_real = args.graph_save_path + args.note + '_' + args.graph_type + '_' + str(args.graph_node_num) + '_' + \
        #              str(epoch) + '_real_' + str(True) + '_' + str(num_layers)
        # fname_pred = args.graph_save_path + args.note + '_' + args.graph_type + '_' + str(args.graph_node_num) + '_' + \
        #              str(epoch) + '_pred_' + str(True) + '_' + str(num_layers)
        # figname = args.figure_save_path + args.note + '_' + args.graph_type + '_' + str(args.graph_node_num) + '_' + \
        #           str(epoch) + '_' + str(num_layers)
        print(fname_real)

        # load data
        graph_real_list = load_graph_list(fname_real + '.dat')
        graph_pred_list = load_graph_list(fname_pred + '.dat')
        graph_real_len_list = np.array([len(graph_real_list[i]) for i in range(len(graph_real_list))])
        graph_pred_len_list = np.array([len(graph_pred_list[i]) for i in range(len(graph_pred_list))])
        real_order = np.argsort(graph_real_len_list)[::-1]
        pred_order = np.argsort(graph_pred_len_list)[::-1]
        # print(real_order)
        # print(pred_order)
        graph_real_list = [graph_real_list[i] for i in real_order]
        graph_pred_list = [graph_pred_list[i] for i in pred_order]

        # shuffle(graph_real_list)
        # shuffle(graph_pred_list)
        print('real average nodes', sum([graph_real_list[i].number_of_nodes() for i in range(len(graph_real_list))])/len(graph_real_list))
        print('pred average nodes', sum([graph_pred_list[i].number_of_nodes() for i in range(len(graph_pred_list))])/len(graph_pred_list))
        print('num of graphs', len(graph_real_list))

        # draw all graphs
        for iter in range(8):
            print('iter', iter)
            graph_list = []
            for i in range(8):
                index = 8 * iter + i
                graph_real_list[index].remove_nodes_from(nx.isolates(graph_real_list[index]))
                graph_pred_list[index].remove_nodes_from(nx.isolates(graph_pred_list[index]))
                graph_list.append(graph_real_list[index])
                graph_list.append(graph_pred_list[index])
                print('real', graph_real_list[index].number_of_nodes())
                print('pred', graph_pred_list[index].number_of_nodes())

            draw_graph_list(graph_list, row=4, col=4, fname=figname + '_' + str(iter))

# for new model
elif args.note == 'GraphRNN_structure' and args.is_flexible==False:
    for num_layers in range(4,5):
        # give file name and figure name
        # fname_real = args.graph_save_path + args.note + '_' + args.graph_type + '_' + str(args.graph_node_num) + '_' + \
        #                      str(epoch) + '_real_bptt_' + str(args.bptt)+'_'+str(num_layers)+'_dilation_'+str(args.is_dilation)+'_flexible_'+str(args.is_flexible)+'_bn_'+str(args.is_bn)+'_lr_'+str(args.lr)
        # fname_pred = args.graph_save_path + args.note + '_' + args.graph_type + '_' + str(args.graph_node_num) + '_' + \
        #                      str(epoch) + '_pred_bptt_' + str(args.bptt)+'_'+str(num_layers)+'_dilation_'+str(args.is_dilation)+'_flexible_'+str(args.is_flexible)+'_bn_'+str(args.is_bn)+'_lr_'+str(args.lr)

        fname_pred = args.graph_save_path + args.note + '_' + args.graph_type + '_' + \
                     str(epoch) + '_pred_' + str(args.num_layers) + '_' + str(args.bptt)+ '_' + str(args.bptt_len) + '_' + str(args.hidden_size)
        fname_real = args.graph_save_path + args.note + '_' + args.graph_type + '_' + \
                     str(epoch) + '_real_' + str(args.num_layers) + '_' + str(args.bptt)+ '_' + str(args.bptt_len) + '_' + str(args.hidden_size)
        figname = args.figure_save_path + args.note + '_' + args.graph_type + '_' + \
                     str(epoch) + '_pred_' + str(args.num_layers) + '_' + str(args.bptt)+ '_' + str(args.bptt_len) + '_' + str(args.hidden_size)
        print(fname_real)
        # load data
        graph_real_list = load_graph_list(fname_real+'.dat')
        graph_pred_list = load_graph_list(fname_pred+'.dat')

        graph_real_len_list = np.array([len(graph_real_list[i]) for i in range(len(graph_real_list))])
        graph_pred_len_list = np.array([len(graph_pred_list[i]) for i in range(len(graph_pred_list))])
        real_order = np.argsort(graph_real_len_list)[::-1]
        pred_order = np.argsort(graph_pred_len_list)[::-1]
        # print(real_order)
        # print(pred_order)
        graph_real_list = [graph_real_list[i] for i in real_order]
        graph_pred_list = [graph_pred_list[i] for i in pred_order]

        shuffle(graph_pred_list)


        print('real average nodes',
              sum([graph_real_list[i].number_of_nodes() for i in range(len(graph_real_list))]) / len(graph_real_list))
        print('pred average nodes',
              sum([graph_pred_list[i].number_of_nodes() for i in range(len(graph_pred_list))]) / len(graph_pred_list))
        print('num of graphs', len(graph_real_list))

        # draw all graphs
        for iter in range(2):
            print('iter', iter)
            graph_list = []
            for i in range(8):
                index = 8*iter + i
                graph_real_list[index].remove_nodes_from(nx.isolates(graph_real_list[index]))
                graph_pred_list[index].remove_nodes_from(nx.isolates(graph_pred_list[index]))
                graph_list.append(graph_real_list[index])
                graph_list.append(graph_pred_list[index])
                print('real', graph_real_list[index].number_of_nodes())
                print('pred', graph_pred_list[index].number_of_nodes())
            draw_graph_list(graph_list, row=4, col=4, fname=figname+'_'+str(iter))


# for new model
elif args.note == 'GraphRNN_structure' and args.is_flexible==True:
    for num_layers in range(4,5):
        graph_real_list = []
        graph_pred_list = []
        epoch_end = 30000
        for epoch in [epoch_end-500*(8-i) for i in range(8)]:
            # give file name and figure name
            fname_real = args.graph_save_path + args.note + '_' + args.graph_type + '_' + str(args.graph_node_num) + '_' + \
                                 str(epoch) + '_real_bptt_' + str(args.bptt)+'_'+str(num_layers)+'_dilation_'+str(args.is_dilation)+'_flexible_'+str(args.is_flexible)+'_bn_'+str(args.is_bn)+'_lr_'+str(args.lr)
            fname_pred = args.graph_save_path + args.note + '_' + args.graph_type + '_' + str(args.graph_node_num) + '_' + \
                                 str(epoch) + '_pred_bptt_' + str(args.bptt)+'_'+str(num_layers)+'_dilation_'+str(args.is_dilation)+'_flexible_'+str(args.is_flexible)+'_bn_'+str(args.is_bn)+'_lr_'+str(args.lr)

            # load data
            graph_real_list += load_graph_list(fname_real+'.dat')
            graph_pred_list += load_graph_list(fname_pred+'.dat')
        print('num of graphs', len(graph_real_list))

        figname = args.figure_save_path + args.note + '_' + args.graph_type + '_' + str(args.graph_node_num) + '_' + \
                  str(epoch) + str(args.sample_when_validate) + '_' + str(num_layers) + '_dilation_' + str(args.is_dilation) + '_flexible_' + str(args.is_flexible) + '_bn_' + str(args.is_bn) + '_lr_' + str(args.lr)

        # draw all graphs
        for iter in range(1):
            print('iter', iter)
            graph_list = []
            for i in range(8):
                index = 8*iter + i
                graph_real_list[index].remove_nodes_from(nx.isolates(graph_real_list[index]))
                graph_pred_list[index].remove_nodes_from(nx.isolates(graph_pred_list[index]))
                graph_list.append(graph_real_list[index])
                graph_list.append(graph_pred_list[index])
            draw_graph_list(graph_list, row=4, col=4, fname=figname+'_'+str(iter))