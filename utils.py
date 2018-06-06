import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
# import node2vec.src.main as nv
from sklearn.decomposition import PCA
import community
import pickle
import re

import data

def citeseer_ego():
    _, _, G = data.Graph_load(dataset='citeseer')
    G = max(nx.connected_component_subgraphs(G), key=len)
    G = nx.convert_node_labels_to_integers(G)
    graphs = []
    for i in range(G.number_of_nodes()):
        G_ego = nx.ego_graph(G, i, radius=3)
        if G_ego.number_of_nodes() >= 50 and (G_ego.number_of_nodes() <= 400):
            graphs.append(G_ego)
    return graphs

def caveman_special(c=2,k=20,p_path=0.1,p_edge=0.3):
    p = p_path
    path_count = max(int(np.ceil(p * k)),1)
    G = nx.caveman_graph(c, k)
    # remove 50% edges
    p = 1-p_edge
    for (u, v) in list(G.edges()):
        if np.random.rand() < p and ((u < k and v < k) or (u >= k and v >= k)):
            G.remove_edge(u, v)
    # add path_count links
    for i in range(path_count):
        u = np.random.randint(0, k)
        v = np.random.randint(k, k * 2)
        G.add_edge(u, v)
    G = max(nx.connected_component_subgraphs(G), key=len)
    return G

def n_community(c_sizes, p_inter=0.01):
    graphs = [nx.gnp_random_graph(c_sizes[i], 0.7, seed=i) for i in range(len(c_sizes))]
    G = nx.disjoint_union_all(graphs)
    communities = list(nx.connected_component_subgraphs(G))
    for i in range(len(communities)):
        subG1 = communities[i]
        nodes1 = list(subG1.nodes())
        for j in range(i+1, len(communities)):
            subG2 = communities[j]
            nodes2 = list(subG2.nodes())
            has_inter_edge = False
            for n1 in nodes1:
                for n2 in nodes2:
                    if np.random.rand() < p_inter:
                        G.add_edge(n1, n2)
                        has_inter_edge = True
            if not has_inter_edge:
                G.add_edge(nodes1[0], nodes2[0])
    #print('connected comp: ', len(list(nx.connected_component_subgraphs(G))))
    return G

def perturb(graph_list, p_del, p_add=None):
    ''' Perturb the list of graphs by adding/removing edges.
    Args:
        p_add: probability of adding edges. If None, estimate it according to graph density,
            such that the expected number of added edges is equal to that of deleted edges.
        p_del: probability of removing edges
    Returns:
        A list of graphs that are perturbed from the original graphs
    '''
    perturbed_graph_list = []
    for G_original in graph_list:
        G = G_original.copy()
        trials = np.random.binomial(1, p_del, size=G.number_of_edges())
        edges = list(G.edges())
        i = 0
        for (u, v) in edges:
            if trials[i] == 1:
                G.remove_edge(u, v)
            i += 1
        if p_add is None:
            num_nodes = G.number_of_nodes()
            p_add_est = np.sum(trials) / (num_nodes * (num_nodes - 1) / 2 -
                    G.number_of_edges())
        else:
            p_add_est = p_add

        nodes = list(G.nodes())
        tmp = 0
        for i in range(len(nodes)):
            u = nodes[i]
            trials = np.random.binomial(1, p_add_est, size=G.number_of_nodes())
            j = 0
            for j in range(i+1, len(nodes)):
                v = nodes[j]
                if trials[j] == 1:
                    tmp += 1
                    G.add_edge(u, v)
                j += 1

        perturbed_graph_list.append(G)
    return perturbed_graph_list



def perturb_new(graph_list, p):
    ''' Perturb the list of graphs by adding/removing edges.
    Args:
        p_add: probability of adding edges. If None, estimate it according to graph density,
            such that the expected number of added edges is equal to that of deleted edges.
        p_del: probability of removing edges
    Returns:
        A list of graphs that are perturbed from the original graphs
    '''
    perturbed_graph_list = []
    for G_original in graph_list:
        G = G_original.copy()
        edge_remove_count = 0
        for (u, v) in list(G.edges()):
            if np.random.rand()<p:
                G.remove_edge(u, v)
                edge_remove_count += 1
        # randomly add the edges back
        for i in range(edge_remove_count):
            while True:
                u = np.random.randint(0, G.number_of_nodes())
                v = np.random.randint(0, G.number_of_nodes())
                if (not G.has_edge(u,v)) and (u!=v):
                    break
            G.add_edge(u, v)
        perturbed_graph_list.append(G)
    return perturbed_graph_list





def imsave(fname, arr, vmin=None, vmax=None, cmap=None, format=None, origin=None):
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    from matplotlib.figure import Figure

    fig = Figure(figsize=arr.shape[::-1], dpi=1, frameon=False)
    canvas = FigureCanvas(fig)
    fig.figimage(arr, cmap=cmap, vmin=vmin, vmax=vmax, origin=origin)
    fig.savefig(fname, dpi=1, format=format)


def save_prediction_histogram(y_pred_data, fname_pred, max_num_node, bin_n=20):
    bin_edge = np.linspace(1e-6, 1, bin_n + 1)
    output_pred = np.zeros((bin_n, max_num_node))
    for i in range(max_num_node):
        output_pred[:, i], _ = np.histogram(y_pred_data[:, i, :], bins=bin_edge, density=False)
        # normalize
        output_pred[:, i] /= np.sum(output_pred[:, i])
    imsave(fname=fname_pred, arr=output_pred, origin='upper', cmap='Greys_r', vmin=0.0, vmax=3.0 / bin_n)


# draw a single graph G
def draw_graph(G, prefix = 'test'):
    parts = community.best_partition(G)
    values = [parts.get(node) for node in G.nodes()]
    colors = []
    for i in range(len(values)):
        if values[i] == 0:
            colors.append('red')
        if values[i] == 1:
            colors.append('green')
        if values[i] == 2:
            colors.append('blue')
        if values[i] == 3:
            colors.append('yellow')
        if values[i] == 4:
            colors.append('orange')
        if values[i] == 5:
            colors.append('pink')
        if values[i] == 6:
            colors.append('black')

    # spring_pos = nx.spring_layout(G)
    plt.switch_backend('agg')
    plt.axis("off")

    pos = nx.spring_layout(G)
    nx.draw_networkx(G, with_labels=True, node_size=35, node_color=colors,pos=pos)


    # plt.switch_backend('agg')
    # options = {
    #     'node_color': 'black',
    #     'node_size': 10,
    #     'width': 1
    # }
    # plt.figure()
    # plt.subplot()
    # nx.draw_networkx(G, **options)
    plt.savefig('figures/graph_view_'+prefix+'.png', dpi=200)
    plt.close()

    plt.switch_backend('agg')
    G_deg = nx.degree_histogram(G)
    G_deg = np.array(G_deg)
    # plt.plot(range(len(G_deg)), G_deg, 'r', linewidth = 2)
    plt.loglog(np.arange(len(G_deg))[G_deg>0], G_deg[G_deg>0], 'r', linewidth=2)
    plt.savefig('figures/degree_view_' + prefix + '.png', dpi=200)
    plt.close()

    # degree_sequence = sorted(nx.degree(G).values(), reverse=True)  # degree sequence
    # plt.loglog(degree_sequence, 'b-', marker='o')
    # plt.title("Degree rank plot")
    # plt.ylabel("degree")
    # plt.xlabel("rank")
    # plt.savefig('figures/degree_view_' + prefix + '.png', dpi=200)
    # plt.close()


# G = nx.grid_2d_graph(8,8)
# G = nx.karate_club_graph()
# draw_graph(G)


# draw a list of graphs [G]
def draw_graph_list(G_list, row, col, fname = 'figures/test', layout='spring', is_single=False,k=1,node_size=55,alpha=1,width=1.3):
    # # draw graph view
    # from pylab import rcParams
    # rcParams['figure.figsize'] = 12,3
    plt.switch_backend('agg')
    for i,G in enumerate(G_list):
        plt.subplot(row,col,i+1)
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1,
                        wspace=0, hspace=0)
        # if i%2==0:
        #     plt.title('real nodes: '+str(G.number_of_nodes()), fontsize = 4)
        # else:
        #     plt.title('pred nodes: '+str(G.number_of_nodes()), fontsize = 4)

        # plt.title('num of nodes: '+str(G.number_of_nodes()), fontsize = 4)

        # parts = community.best_partition(G)
        # values = [parts.get(node) for node in G.nodes()]
        # colors = []
        # for i in range(len(values)):
        #     if values[i] == 0:
        #         colors.append('red')
        #     if values[i] == 1:
        #         colors.append('green')
        #     if values[i] == 2:
        #         colors.append('blue')
        #     if values[i] == 3:
        #         colors.append('yellow')
        #     if values[i] == 4:
        #         colors.append('orange')
        #     if values[i] == 5:
        #         colors.append('pink')
        #     if values[i] == 6:
        #         colors.append('black')
        plt.axis("off")
        if layout=='spring':
            pos = nx.spring_layout(G,k=k/np.sqrt(G.number_of_nodes()),iterations=100)
            # pos = nx.spring_layout(G)

        elif layout=='spectral':
            pos = nx.spectral_layout(G)
        # # nx.draw_networkx(G, with_labels=True, node_size=2, width=0.15, font_size = 1.5, node_color=colors,pos=pos)
        # nx.draw_networkx(G, with_labels=False, node_size=1.5, width=0.2, font_size = 1.5, linewidths=0.2, node_color = 'k',pos=pos,alpha=0.2)

        if is_single:
            # node_size default 60, edge_width default 1.5
            nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color='#336699', alpha=1, linewidths=0, font_size=0)
            nx.draw_networkx_edges(G, pos, alpha=alpha, width=width)
        else:
            nx.draw_networkx_nodes(G, pos, node_size=1.5, node_color='#336699',alpha=1, linewidths=0.2, font_size = 1.5)
            nx.draw_networkx_edges(G, pos, alpha=0.3,width=0.2)

        # plt.axis('off')
        # plt.title('Complete Graph of Odd-degree Nodes')
        # plt.show()
    plt.tight_layout()
    plt.savefig(fname+'.png', dpi=600)
    plt.close()

    # # draw degree distribution
    # plt.switch_backend('agg')
    # for i, G in enumerate(G_list):
    #     plt.subplot(row, col, i + 1)
    #     G_deg = np.array(list(G.degree(G.nodes()).values()))
    #     bins = np.arange(20)
    #     plt.hist(np.array(G_deg), bins=bins, align='left')
    #     plt.xlabel('degree', fontsize = 3)
    #     plt.ylabel('count', fontsize = 3)
    #     G_deg_mean = 2*G.number_of_edges()/float(G.number_of_nodes())
    #     # if i % 2 == 0:
    #     #     plt.title('real average degree: {:.2f}'.format(G_deg_mean), fontsize=4)
    #     # else:
    #     #     plt.title('pred average degree: {:.2f}'.format(G_deg_mean), fontsize=4)
    #     plt.title('average degree: {:.2f}'.format(G_deg_mean), fontsize=4)
    #     plt.tick_params(axis='both', which='major', labelsize=3)
    #     plt.tick_params(axis='both', which='minor', labelsize=3)
    # plt.tight_layout()
    # plt.savefig(fname+'_degree.png', dpi=600)
    # plt.close()
    #
    # # draw clustering distribution
    # plt.switch_backend('agg')
    # for i, G in enumerate(G_list):
    #     plt.subplot(row, col, i + 1)
    #     G_cluster = list(nx.clustering(G).values())
    #     bins = np.linspace(0,1,20)
    #     plt.hist(np.array(G_cluster), bins=bins, align='left')
    #     plt.xlabel('clustering coefficient', fontsize=3)
    #     plt.ylabel('count', fontsize=3)
    #     G_cluster_mean = sum(G_cluster) / len(G_cluster)
    #     # if i % 2 == 0:
    #     #     plt.title('real average clustering: {:.4f}'.format(G_cluster_mean), fontsize=4)
    #     # else:
    #     #     plt.title('pred average clustering: {:.4f}'.format(G_cluster_mean), fontsize=4)
    #     plt.title('average clustering: {:.4f}'.format(G_cluster_mean), fontsize=4)
    #     plt.tick_params(axis='both', which='major', labelsize=3)
    #     plt.tick_params(axis='both', which='minor', labelsize=3)
    # plt.tight_layout()
    # plt.savefig(fname+'_clustering.png', dpi=600)
    # plt.close()
    #
    # # draw circle distribution
    # plt.switch_backend('agg')
    # for i, G in enumerate(G_list):
    #     plt.subplot(row, col, i + 1)
    #     cycle_len = []
    #     cycle_all = nx.cycle_basis(G)
    #     for item in cycle_all:
    #         cycle_len.append(len(item))
    #
    #     bins = np.arange(20)
    #     plt.hist(np.array(cycle_len), bins=bins, align='left')
    #     plt.xlabel('cycle length', fontsize=3)
    #     plt.ylabel('count', fontsize=3)
    #     G_cycle_mean = 0
    #     if len(cycle_len)>0:
    #         G_cycle_mean = sum(cycle_len) / len(cycle_len)
    #     # if i % 2 == 0:
    #     #     plt.title('real average cycle: {:.4f}'.format(G_cycle_mean), fontsize=4)
    #     # else:
    #     #     plt.title('pred average cycle: {:.4f}'.format(G_cycle_mean), fontsize=4)
    #     plt.title('average cycle: {:.4f}'.format(G_cycle_mean), fontsize=4)
    #     plt.tick_params(axis='both', which='major', labelsize=3)
    #     plt.tick_params(axis='both', which='minor', labelsize=3)
    # plt.tight_layout()
    # plt.savefig(fname+'_cycle.png', dpi=600)
    # plt.close()
    #
    # # draw community distribution
    # plt.switch_backend('agg')
    # for i, G in enumerate(G_list):
    #     plt.subplot(row, col, i + 1)
    #     parts = community.best_partition(G)
    #     values = np.array([parts.get(node) for node in G.nodes()])
    #     counts = np.sort(np.bincount(values)[::-1])
    #     pos = np.arange(len(counts))
    #     plt.bar(pos,counts,align = 'edge')
    #     plt.xlabel('community ID', fontsize=3)
    #     plt.ylabel('count', fontsize=3)
    #     G_community_count = len(counts)
    #     # if i % 2 == 0:
    #     #     plt.title('real average clustering: {}'.format(G_community_count), fontsize=4)
    #     # else:
    #     #     plt.title('pred average clustering: {}'.format(G_community_count), fontsize=4)
    #     plt.title('average clustering: {}'.format(G_community_count), fontsize=4)
    #     plt.tick_params(axis='both', which='major', labelsize=3)
    #     plt.tick_params(axis='both', which='minor', labelsize=3)
    # plt.tight_layout()
    # plt.savefig(fname+'_community.png', dpi=600)
    # plt.close()



    # plt.switch_backend('agg')
    # G_deg = nx.degree_histogram(G)
    # G_deg = np.array(G_deg)
    # # plt.plot(range(len(G_deg)), G_deg, 'r', linewidth = 2)
    # plt.loglog(np.arange(len(G_deg))[G_deg>0], G_deg[G_deg>0], 'r', linewidth=2)
    # plt.savefig('figures/degree_view_' + prefix + '.png', dpi=200)
    # plt.close()

    # degree_sequence = sorted(nx.degree(G).values(), reverse=True)  # degree sequence
    # plt.loglog(degree_sequence, 'b-', marker='o')
    # plt.title("Degree rank plot")
    # plt.ylabel("degree")
    # plt.xlabel("rank")
    # plt.savefig('figures/degree_view_' + prefix + '.png', dpi=200)
    # plt.close()



# directly get graph statistics from adj, obsoleted
def decode_graph(adj, prefix):
    adj = np.asmatrix(adj)
    G = nx.from_numpy_matrix(adj)
    # G.remove_nodes_from(nx.isolates(G))
    print('num of nodes: {}'.format(G.number_of_nodes()))
    print('num of edges: {}'.format(G.number_of_edges()))
    G_deg = nx.degree_histogram(G)
    G_deg_sum = [a * b for a, b in zip(G_deg, range(0, len(G_deg)))]
    print('average degree: {}'.format(sum(G_deg_sum) / G.number_of_nodes()))
    if nx.is_connected(G):
        print('average path length: {}'.format(nx.average_shortest_path_length(G)))
        print('average diameter: {}'.format(nx.diameter(G)))
    G_cluster = sorted(list(nx.clustering(G).values()))
    print('average clustering coefficient: {}'.format(sum(G_cluster) / len(G_cluster)))
    cycle_len = []
    cycle_all = nx.cycle_basis(G, 0)
    for item in cycle_all:
        cycle_len.append(len(item))
    print('cycles', cycle_len)
    print('cycle count', len(cycle_len))
    draw_graph(G, prefix=prefix)


def get_graph(adj):
    '''
    get a graph from zero-padded adj
    :param adj:
    :return:
    '''
    # remove all zeros rows and columns
    adj = adj[~np.all(adj == 0, axis=1)]
    adj = adj[:, ~np.all(adj == 0, axis=0)]
    adj = np.asmatrix(adj)
    G = nx.from_numpy_matrix(adj)
    return G

# save a list of graphs
def save_graph_list(G_list, fname):
    with open(fname, "wb") as f:
        pickle.dump(G_list, f)


# pick the first connected component
def pick_connected_component(G):
    node_list = nx.node_connected_component(G,0)
    return G.subgraph(node_list)

def pick_connected_component_new(G):
    adj_list = G.adjacency_list()
    for id,adj in enumerate(adj_list):
        id_min = min(adj)
        if id<id_min and id>=1:
        # if id<id_min and id>=4:
            break
    node_list = list(range(id)) # only include node prior than node "id"
    G = G.subgraph(node_list)
    G = max(nx.connected_component_subgraphs(G), key=len)
    return G

# load a list of graphs
def load_graph_list(fname,is_real=True):
    with open(fname, "rb") as f:
        graph_list = pickle.load(f)
    for i in range(len(graph_list)):
        edges_with_selfloops = graph_list[i].selfloop_edges()
        if len(edges_with_selfloops)>0:
            graph_list[i].remove_edges_from(edges_with_selfloops)
        if is_real:
            graph_list[i] = max(nx.connected_component_subgraphs(graph_list[i]), key=len)
            graph_list[i] = nx.convert_node_labels_to_integers(graph_list[i])
        else:
            graph_list[i] = pick_connected_component_new(graph_list[i])
    return graph_list


def export_graphs_to_txt(g_list, output_filename_prefix):
    i = 0
    for G in g_list:
        f = open(output_filename_prefix + '_' + str(i) + '.txt', 'w+')
        for (u, v) in G.edges():
            idx_u = G.nodes().index(u)
            idx_v = G.nodes().index(v)
            f.write(str(idx_u) + '\t' + str(idx_v) + '\n')
        i += 1

def snap_txt_output_to_nx(in_fname):
    G = nx.Graph()
    with open(in_fname, 'r') as f:
        for line in f:
            if not line[0] == '#':
                splitted = re.split('[ \t]', line)

                # self loop might be generated, but should be removed
                u = int(splitted[0])
                v = int(splitted[1])
                if not u == v:
                    G.add_edge(int(u), int(v))
    return G

def test_perturbed():
    
    graphs = []
    for i in range(100,101):
        for j in range(4,5):
            for k in range(500):
                graphs.append(nx.barabasi_albert_graph(i,j))
    g_perturbed = perturb(graphs, 0.9)
    print([g.number_of_edges() for g in graphs])
    print([g.number_of_edges() for g in g_perturbed])

if __name__ == '__main__':
    #test_perturbed()
    #graphs = load_graph_list('graphs/' + 'GraphRNN_RNN_community4_4_128_train_0.dat')
    #graphs = load_graph_list('graphs/' + 'GraphRNN_RNN_community4_4_128_pred_2500_1.dat')
    graphs = load_graph_list('eval_results/mmsb/' + 'community41.dat')
    
    for i in range(0, 160, 16):
        draw_graph_list(graphs[i:i+16], 4, 4, fname='figures/community4_' + str(i))

