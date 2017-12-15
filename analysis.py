# this file is used to plot images
from main import *

args = Args()
print(args.graph_type, args.note)
# epoch = 16000
epoch = 3000
sample_time = 3


def find_nearest_idx(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx

# for baseline model
for num_layers in range(4,5):
    # give file name and figure name
    fname_real = args.graph_save_path + args.fname_real + str(0)
    fname_pred = args.graph_save_path + args.fname_pred + str(epoch) +'_'+str(sample_time)
    figname = args.figure_save_path + args.fname + str(epoch) +'_'+str(sample_time)

    # fname_real = args.graph_save_path + args.note + '_' + args.graph_type + '_' + str(args.graph_node_num) + '_' + \
    #              str(epoch) + '_real_' + str(True) + '_' + str(num_layers)
    # fname_pred = args.graph_save_path + args.note + '_' + args.graph_type + '_' + str(args.graph_node_num) + '_' + \
    #              str(epoch) + '_pred_' + str(True) + '_' + str(num_layers)
    # figname = args.figure_save_path + args.note + '_' + args.graph_type + '_' + str(args.graph_node_num) + '_' + \
    #           str(epoch) + '_' + str(num_layers)
    print(fname_real)
    print(fname_pred)


    # load data
    graph_real_list = load_graph_list(fname_real + '.dat')
    shuffle(graph_real_list)
    graph_pred_list_raw = load_graph_list(fname_pred + '.dat')
    graph_real_len_list = np.array([len(graph_real_list[i]) for i in range(len(graph_real_list))])
    graph_pred_len_list_raw = np.array([len(graph_pred_list_raw[i]) for i in range(len(graph_pred_list_raw))])

    graph_pred_list = graph_pred_list_raw
    graph_pred_len_list = graph_pred_len_list_raw


    # # select samples
    # graph_pred_list = []
    # graph_pred_len_list = []
    # for value in graph_real_len_list:
    #     pred_idx = find_nearest_idx(graph_pred_len_list_raw, value)
    #     graph_pred_list.append(graph_pred_list_raw[pred_idx])
    #     graph_pred_len_list.append(graph_pred_len_list_raw[pred_idx])
    #     # delete
    #     graph_pred_len_list_raw=np.delete(graph_pred_len_list_raw, pred_idx)
    #     del graph_pred_list_raw[pred_idx]
    #     if len(graph_pred_list)==200:
    #         break
    # graph_pred_len_list = np.array(graph_pred_len_list)



    # # select pred data within certain range
    # len_min = np.amin(graph_real_len_list)
    # len_max = np.amax(graph_real_len_list)
    # pred_index = np.where((graph_pred_len_list>=len_min)&(graph_pred_len_list<=len_max))
    # # print(pred_index[0])
    # graph_pred_list = [graph_pred_list[i] for i in pred_index[0]]
    # graph_pred_len_list = graph_pred_len_list[pred_index[0]]



    # real_order = np.argsort(graph_real_len_list)
    # pred_order = np.argsort(graph_pred_len_list)
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
    print('num of real graphs', len(graph_real_list))
    print('num of pred graphs', len(graph_pred_list))


    # # draw all graphs
    # for iter in range(8):
    #     print('iter', iter)
    #     graph_list = []
    #     for i in range(8):
    #         index = 8 * iter + i
    #         # graph_real_list[index].remove_nodes_from(list(nx.isolates(graph_real_list[index])))
    #         # graph_pred_list[index].remove_nodes_from(list(nx.isolates(graph_pred_list[index])))
    #         graph_list.append(graph_real_list[index])
    #         graph_list.append(graph_pred_list[index])
    #         print('real', graph_real_list[index].number_of_nodes())
    #         print('pred', graph_pred_list[index].number_of_nodes())
    #
    #     draw_graph_list(graph_list, row=4, col=4, fname=figname + '_' + str(iter))

    # draw all graphs
    for iter in range(8):
        print('iter', iter)
        graph_list = []
        for i in range(8):
            index = 32 * iter + i
            # graph_real_list[index].remove_nodes_from(list(nx.isolates(graph_real_list[index])))
            # graph_pred_list[index].remove_nodes_from(list(nx.isolates(graph_pred_list[index])))
            # graph_list.append(graph_real_list[index])
            graph_list.append(graph_pred_list[index])
            # print('real', graph_real_list[index].number_of_nodes())
            print('pred', graph_pred_list[index].number_of_nodes())

        draw_graph_list(graph_list, row=4, col=4, fname=figname + '_' + str(iter)+'_pred')

    # draw all graphs
    for iter in range(8):
        print('iter', iter)
        graph_list = []
        for i in range(8):
            index = 16 * iter + i
            # graph_real_list[index].remove_nodes_from(list(nx.isolates(graph_real_list[index])))
            # graph_pred_list[index].remove_nodes_from(list(nx.isolates(graph_pred_list[index])))
            graph_list.append(graph_real_list[index])
            # graph_list.append(graph_pred_list[index])
            print('real', graph_real_list[index].number_of_nodes())
            # print('pred', graph_pred_list[index].number_of_nodes())

        draw_graph_list(graph_list, row=4, col=4, fname=figname + '_' + str(iter)+'_real')

#
# # for new model
# elif args.note == 'GraphRNN_structure' and args.is_flexible==False:
#     for num_layers in range(4,5):
#         # give file name and figure name
#         # fname_real = args.graph_save_path + args.note + '_' + args.graph_type + '_' + str(args.graph_node_num) + '_' + \
#         #                      str(epoch) + '_real_bptt_' + str(args.bptt)+'_'+str(num_layers)+'_dilation_'+str(args.is_dilation)+'_flexible_'+str(args.is_flexible)+'_bn_'+str(args.is_bn)+'_lr_'+str(args.lr)
#         # fname_pred = args.graph_save_path + args.note + '_' + args.graph_type + '_' + str(args.graph_node_num) + '_' + \
#         #                      str(epoch) + '_pred_bptt_' + str(args.bptt)+'_'+str(num_layers)+'_dilation_'+str(args.is_dilation)+'_flexible_'+str(args.is_flexible)+'_bn_'+str(args.is_bn)+'_lr_'+str(args.lr)
#
#         fname_pred = args.graph_save_path + args.note + '_' + args.graph_type + '_' + \
#                      str(epoch) + '_pred_' + str(args.num_layers) + '_' + str(args.bptt)+ '_' + str(args.bptt_len) + '_' + str(args.hidden_size)
#         fname_real = args.graph_save_path + args.note + '_' + args.graph_type + '_' + \
#                      str(epoch) + '_real_' + str(args.num_layers) + '_' + str(args.bptt)+ '_' + str(args.bptt_len) + '_' + str(args.hidden_size)
#         figname = args.figure_save_path + args.note + '_' + args.graph_type + '_' + \
#                      str(epoch) + '_pred_' + str(args.num_layers) + '_' + str(args.bptt)+ '_' + str(args.bptt_len) + '_' + str(args.hidden_size)
#         print(fname_real)
#         # load data
#         graph_real_list = load_graph_list(fname_real+'.dat')
#         graph_pred_list = load_graph_list(fname_pred+'.dat')
#
#         graph_real_len_list = np.array([len(graph_real_list[i]) for i in range(len(graph_real_list))])
#         graph_pred_len_list = np.array([len(graph_pred_list[i]) for i in range(len(graph_pred_list))])
#         real_order = np.argsort(graph_real_len_list)[::-1]
#         pred_order = np.argsort(graph_pred_len_list)[::-1]
#         # print(real_order)
#         # print(pred_order)
#         graph_real_list = [graph_real_list[i] for i in real_order]
#         graph_pred_list = [graph_pred_list[i] for i in pred_order]
#
#         shuffle(graph_pred_list)
#
#
#         print('real average nodes',
#               sum([graph_real_list[i].number_of_nodes() for i in range(len(graph_real_list))]) / len(graph_real_list))
#         print('pred average nodes',
#               sum([graph_pred_list[i].number_of_nodes() for i in range(len(graph_pred_list))]) / len(graph_pred_list))
#         print('num of graphs', len(graph_real_list))
#
#         # draw all graphs
#         for iter in range(2):
#             print('iter', iter)
#             graph_list = []
#             for i in range(8):
#                 index = 8*iter + i
#                 graph_real_list[index].remove_nodes_from(nx.isolates(graph_real_list[index]))
#                 graph_pred_list[index].remove_nodes_from(nx.isolates(graph_pred_list[index]))
#                 graph_list.append(graph_real_list[index])
#                 graph_list.append(graph_pred_list[index])
#                 print('real', graph_real_list[index].number_of_nodes())
#                 print('pred', graph_pred_list[index].number_of_nodes())
#             draw_graph_list(graph_list, row=4, col=4, fname=figname+'_'+str(iter))
#
#
# # for new model
# elif args.note == 'GraphRNN_structure' and args.is_flexible==True:
#     for num_layers in range(4,5):
#         graph_real_list = []
#         graph_pred_list = []
#         epoch_end = 30000
#         for epoch in [epoch_end-500*(8-i) for i in range(8)]:
#             # give file name and figure name
#             fname_real = args.graph_save_path + args.note + '_' + args.graph_type + '_' + str(args.graph_node_num) + '_' + \
#                                  str(epoch) + '_real_bptt_' + str(args.bptt)+'_'+str(num_layers)+'_dilation_'+str(args.is_dilation)+'_flexible_'+str(args.is_flexible)+'_bn_'+str(args.is_bn)+'_lr_'+str(args.lr)
#             fname_pred = args.graph_save_path + args.note + '_' + args.graph_type + '_' + str(args.graph_node_num) + '_' + \
#                                  str(epoch) + '_pred_bptt_' + str(args.bptt)+'_'+str(num_layers)+'_dilation_'+str(args.is_dilation)+'_flexible_'+str(args.is_flexible)+'_bn_'+str(args.is_bn)+'_lr_'+str(args.lr)
#
#             # load data
#             graph_real_list += load_graph_list(fname_real+'.dat')
#             graph_pred_list += load_graph_list(fname_pred+'.dat')
#         print('num of graphs', len(graph_real_list))
#
#         figname = args.figure_save_path + args.note + '_' + args.graph_type + '_' + str(args.graph_node_num) + '_' + \
#                   str(epoch) + str(args.sample_when_validate) + '_' + str(num_layers) + '_dilation_' + str(args.is_dilation) + '_flexible_' + str(args.is_flexible) + '_bn_' + str(args.is_bn) + '_lr_' + str(args.lr)
#
#         # draw all graphs
#         for iter in range(1):
#             print('iter', iter)
#             graph_list = []
#             for i in range(8):
#                 index = 8*iter + i
#                 graph_real_list[index].remove_nodes_from(nx.isolates(graph_real_list[index]))
#                 graph_pred_list[index].remove_nodes_from(nx.isolates(graph_pred_list[index]))
#                 graph_list.append(graph_real_list[index])
#                 graph_list.append(graph_pred_list[index])
#             draw_graph_list(graph_list, row=4, col=4, fname=figname+'_'+str(iter))