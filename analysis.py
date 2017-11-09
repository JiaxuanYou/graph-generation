# this file is used to plot images
from main import *

args = Args()
epoch = 5500
# give file name and figure name
fname_real = args.graph_save_path + args.note + '_' + args.graph_type + '_' + str(args.graph_node_num) + '_' + \
                     str(epoch) + '_real_' + str(args.sample_when_validate)+'_'+str(args.num_layers)
fname_pred = args.graph_save_path + args.note + '_' + args.graph_type + '_' + str(args.graph_node_num) + '_' + \
                     str(epoch) + '_pred_' + str(args.sample_when_validate)+'_'+str(args.num_layers)
figname = args.figure_save_path + args.note + '_' + args.graph_type + '_' + str(args.graph_node_num) + '_' + \
                     str(epoch) + str(args.sample_when_validate)+'_'+str(args.num_layers)

# load data
graph_real_list = load_graph_list(fname_real+'.dat')
graph_pred_list = load_graph_list(fname_pred+'.dat')
print('num of graphs', len(graph_real_list))
# and shuffle
# graph_list = []
# for i in range(8):
#     graph_real_list[i].remove_nodes_from(nx.isolates(graph_real_list[i]))
#     graph_pred_list[i].remove_nodes_from(nx.isolates(graph_pred_list[i]))
#     graph_list.append(graph_real_list[i])
#     graph_list.append(graph_pred_list[i])

# plot figures [including all network properties]
# draw_graph_list(graph_list,row=4,col=4,fname = figname)
# draw_graph_list(graph_pred_list[0:16],row=4,col=4,fname = figname_pred+'.png')

# draw all graphs
for iter in range(16):
    print('iter', iter)
    graph_list = []
    for i in range(8):
        index = 8*iter + i
        graph_real_list[index].remove_nodes_from(nx.isolates(graph_real_list[index]))
        graph_pred_list[index].remove_nodes_from(nx.isolates(graph_pred_list[index]))
        graph_list.append(graph_real_list[index])
        graph_list.append(graph_pred_list[index])
    draw_graph_list(graph_list, row=4, col=4, fname=figname+'_'+str(iter))