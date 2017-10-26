from model import *
from data import *
import numpy as np
import copy
from torch.optim.lr_scheduler import MultiStepLR
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
import logging

def make_graph(adj_all, key = 'real'):
    num_node_list = []
    num_edge_list = []
    degree_list = []
    path_len_list = []
    diameter_list = []
    clustering_list = []
    for i in range(adj_all.shape[0]):
    # for i in range(1):
        adj = adj_all[i,:,:]
        # adj = np.triu(adj)
        # print(adj)
        adj = np.asmatrix(adj)
        G = nx.from_numpy_matrix(adj)
        # print('num of nodes', G.number_of_nodes())
        # print('num of edges', G.number_of_edges())
        num_node_list.append(G.number_of_nodes())
        num_edge_list.append(G.number_of_edges())

        G_deg = nx.degree_histogram(G)
        G_deg_sum = [a * b for a, b in zip(G_deg, range(0, len(G_deg)))]
        # print('average degree', sum(G_deg_sum) / G.number_of_nodes())
        degree_list.append(sum(G_deg_sum) / G.number_of_nodes())
        if nx.is_connected(G):
            # print('average path length', nx.average_shortest_path_length(G))
            # print('diameter', nx.diameter(G))
            path_len_list.append(nx.average_shortest_path_length(G))
            diameter_list.append(nx.diameter(G))
        G_cluster = sorted(list(nx.clustering(G).values()))
        # print('average clustering coefficient', sum(G_cluster) / len(G_cluster))
        clustering_list.append(sum(G_cluster) / len(G_cluster))


        # plt.switch_backend('agg')
        # options = {
        #     'node_color': 'black',
        #     'node_size': 10,
        #     'width': 1
        # }
        # plt.figure()
        # plt.subplot()
        # nx.draw_circular(G, **options)
        # plt.savefig('figures/graph_generated_' + key + str(i) +'.png', dpi=200)
        # plt.close()

    # print('num of nodes', sum(num_node_list)/len(num_node_list))
    # print('num of edges', sum(num_edge_list)/len(num_edge_list))
    # print('average degree', sum(degree_list)/len(degree_list))
    # if len(path_len_list)>0:
    #     print('average path length', sum(path_len_list)/len(path_len_list))
    #     print('diameter', sum(diameter_list)/len(diameter_list))
    # print('average clustering coefficient', sum(clustering_list)/len(clustering_list))

    logging.warning('num of nodes: {}'.format(sum(num_node_list) / len(num_node_list)))
    logging.warning('num of edges: {}'.format(sum(num_edge_list) / len(num_edge_list)))
    logging.warning('average degree: {}'.format(sum(degree_list) / len(degree_list)))
    if len(path_len_list) > 0:
        logging.warning('average path length: {}'.format(sum(path_len_list) / len(path_len_list)))
        logging.warning('average diameter: {}'.format(sum(diameter_list) / len(diameter_list)))
    logging.warning('average clustering coefficient: {}'.format(sum(clustering_list) / len(clustering_list)))

    return G

def make_graph_random(adj_all, key = 'real'):
    num_node_list = []
    num_edge_list = []
    degree_list = []
    path_len_list = []
    diameter_list = []
    clustering_list = []
    for i in range(adj_all.shape[0]):
    # for i in range(1):
        adj = adj_all[i,:,:]
        ratio = 0.4
        # print(adj*ratio)
        adj_int = np.random.binomial(1,adj*ratio,(adj.shape[0],adj.shape[1]))

        adj_int = np.asmatrix(adj_int)
        G = nx.from_numpy_matrix(adj_int)
        # print('num of nodes', G.number_of_nodes())
        # print('num of edges', G.number_of_edges())
        num_node_list.append(G.number_of_nodes())
        num_edge_list.append(G.number_of_edges())

        G_deg = nx.degree_histogram(G)
        G_deg_sum = [a * b for a, b in zip(G_deg, range(0, len(G_deg)))]
        # print('average degree', sum(G_deg_sum) / G.number_of_nodes())
        degree_list.append(sum(G_deg_sum) / G.number_of_nodes())
        if nx.is_connected(G):
            # print('average path length', nx.average_shortest_path_length(G))
            # print('diameter', nx.diameter(G))
            path_len_list.append(nx.average_shortest_path_length(G))
            diameter_list.append(nx.diameter(G))
        G_cluster = sorted(list(nx.clustering(G).values()))
        # print('average clustering coefficient', sum(G_cluster) / len(G_cluster))
        clustering_list.append(sum(G_cluster) / len(G_cluster))


        # plt.switch_backend('agg')
        # options = {
        #     'node_color': 'black',
        #     'node_size': 10,
        #     'width': 1
        # }
        # plt.figure()
        # plt.subplot()
        # nx.draw_circular(G, **options)
        # plt.savefig('figures/graph_generated_' + key + str(i) +'.png', dpi=200)
        # plt.close()

    # print('num of nodes', sum(num_node_list)/len(num_node_list))
    # print('num of edges', sum(num_edge_list)/len(num_edge_list))
    # print('average degree', sum(degree_list)/len(degree_list))
    # print('average path length', sum(path_len_list)/len(path_len_list))
    # print('diameter', sum(diameter_list)/len(diameter_list))
    # print('average clustering coefficient', sum(clustering_list)/len(clustering_list))
    logging.warning('num of nodes: {}'.format(sum(num_node_list)/len(num_node_list)))
    logging.warning('num of edges: {}'.format(sum(num_edge_list)/len(num_edge_list)))
    logging.warning('average degree: {}'.format(sum(degree_list)/len(degree_list)))
    if len(path_len_list) > 0:
        logging.warning('average path length: {}'.format(sum(path_len_list)/len(path_len_list)))
        logging.warning('average diameter: {}'.format(sum(diameter_list)/len(diameter_list)))
    logging.warning('average clustering coefficient: {}'.format(sum(clustering_list)/len(clustering_list)))

    return G


def train(args, dataset_train, dataset_test, encoder, decoder):
    torch.manual_seed(args.seed)
    train_loader = torch.utils.data.DataLoader(dataset_train,
                                               batch_size=args.batch_size_train, shuffle=True, num_workers=1)
    test_loader = torch.utils.data.DataLoader(dataset_test,
                                               batch_size=args.batch_size_test, shuffle=True, num_workers=1)
    # train_loader = dataset
    # optimizer = optim.SGD(list(encoder.parameters())+list(decoder.parameters()), lr=args.lr, momentum=args.momentum)
    optimizer = optim.Adam(list(encoder.parameters())+list(decoder.parameters()), lr=args.lr)
    scheduler = MultiStepLR(optimizer, milestones=args.milestones, gamma=args.lr_rate)
    for epoch in range(args.epochs):
        train_epoch(epoch, args, encoder, decoder, train_loader, optimizer, scheduler)
        if epoch%args.epochs_test == 0:
            test_epoch(epoch, args, encoder, decoder, test_loader)



def train_epoch(epoch, args, encoder, decoder, data_loader, optimizer, scheduler):
    encoder.train()
    decoder.train()
    loss_mean = 0
    correct_sum = 0
    all_sum = 0
    auc_mean = 0
    ap_mean = 0
    real_score_mean = 0
    pred_score_mean = 0

    count = 0
    for data in data_loader:
        A_real = Variable(data['adj']).cuda(CUDA)
        # A_real = A_real.view(1, A_real.size(0), A_real.size(1))
        A_norm = Variable(data['adj_norm']).cuda(CUDA)
        # A_norm = A_norm.view(1, A_norm.size(0), A_norm.size(1))
        x = Variable(data['features']).cuda(CUDA)
        # x = x.view(1, x.size(0), x.size(1))

        z = encoder(x,A_norm)
        # print('A_norm', A_norm.size(), 'z', z.size())
        # for batch_num = 1
        # z = z.view(-1, z.size(0),z.size(1))
        A_pred = decoder(z)

        # weight = (A_real.size(1) * A_real.size(1) - A_real.sum()) / A_real.sum().float()
        weight = 1
        # norm = A_real.size(1) * A_real.size(1) / ((A_real.size(1) * A_real.size(1) - A_real.sum()) * 2)
        # print('weight', weight,'norm',norm)
        A_real_smooth = A_real.clone()
        A_real_smooth[A_real==1] = 0.9
        # print(A_real)
        loss = F.binary_cross_entropy_with_logits(A_pred,A_real_smooth, weight=weight)/A_real.size(0)
        loss_mean += loss.data[0]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if epoch % args.epochs_log == 0 and count==0:
            # print(z)
            print('mean z',torch.mean(z).data[0])
            eps = 1e-6
            print('grad ratio w1',torch.mean(encoder.conv1.weight.grad/(encoder.conv1.weight+eps)).data[0])
            print('grad ratio w2', torch.mean(encoder.conv2.weight.grad / (encoder.conv2.weight + eps)).data[0])
            print('average predict', torch.mean(A_pred).data[0])
            # print('size', A_pred.size())
            # print('pred', A_pred.data[0])
            # for i in range(37):
            #     print(A_pred[0,i,:].data.view(-1,))

        A_pred = F.sigmoid(A_pred)
        A_pred_raw = A_pred.clone()
        # calc accuracy
        thresh = 0.65
        A_pred[A_pred>thresh] = 1
        A_pred[A_pred<=thresh] = 0
        A_real = A_real.long()
        A_pred = A_pred.long()
        correct = torch.eq(A_pred, A_real).long().sum()
        all = A_pred.size(0)*A_pred.size(1)*A_pred.size(2)

        true = A_real.view(-1).data.cpu().numpy()
        pred = A_pred_raw.view(-1).data.cpu().numpy()
        auc = roc_auc_score(true, pred)
        ap = average_precision_score(true,pred)

        correct_sum += correct.data[0]
        all_sum += all
        auc_mean += auc
        ap_mean += ap
        real_score_mean += A_real.float().mean().data[0]
        pred_score_mean += A_pred_raw.mean().data[0]

        count += 1

    loss_mean /= count
    auc_mean /= count
    ap_mean /= count
    real_score_mean /= count
    pred_score_mean /= count
    accuracy = correct_sum/float(all_sum)
    if epoch%args.epochs_log==0:
        # print('Epoch: {}/{}, train loss: {:.6f}, accuracy: {}/{} {:.4f}, auc: {:.4f}, ap: {:.4f}, real mean: {:.4f}, pred mean: {:.4f}'.format(
        #     epoch, args.epochs, loss_mean, correct_sum, all_sum, accuracy, auc_mean, ap_mean, real_score_mean, pred_score_mean))
        logging.warning('Epoch: {}/{}, train loss: {:.6f}, accuracy: {}/{} {:.4f}, auc: {:.4f}, ap: {:.4f}, real mean: {:.4f}, pred mean: {:.4f}'.format(
            epoch, args.epochs, loss_mean, correct_sum, all_sum, accuracy, auc_mean, ap_mean, real_score_mean, pred_score_mean))

    # log_value('train_loss', loss_mean, epoch)
    # log_value('accuracy', accuracy, epoch)
    # log_value('auc_mean', auc_mean, epoch)
    # log_value('ap_mean', ap_mean, epoch)



def test_epoch(epoch, args, encoder, decoder, data_loader):
    # the eval() mode seems to be not stable
    encoder.train()
    decoder.train()
    loss_mean = 0
    correct_sum = 0
    all_sum = 0
    auc_mean = 0
    ap_mean = 0
    real_score_mean = 0
    pred_score_mean = 0

    count = 0
    for data in data_loader:
        A_real = Variable(data['adj']).cuda(CUDA)
        # A_real = A_real.view(1, A_real.size(0), A_real.size(1))
        A_norm = Variable(data['adj_norm']).cuda(CUDA)
        # A_norm = A_norm.view(1, A_norm.size(0), A_norm.size(1))
        x = Variable(data['features']).cuda(CUDA)
        # x = x.view(1, x.size(0), x.size(1))

        z = encoder(x, A_norm)
        # z = z.view(-1, z.size(0),z.size(1))
        # if count==0:
        #     print(torch.mean(z).data[0])
        A_pred = decoder(z)

        # weight = (A_real.size(1) * A_real.size(1) - A_real.sum()) / A_real.sum().float()
        weight = 1
        # norm = A_real.size(1) * A_real.size(1) / ((A_real.size(1) * A_real.size(1) - A_real.sum()) * 2)
        # print('weight', weight,'norm',norm)
        loss = F.binary_cross_entropy_with_logits(A_pred, A_real, weight=weight) / A_real.size(0)
        loss_mean += loss.data[0]


        A_pred = F.sigmoid(A_pred)
        A_pred_raw = A_pred.clone()

        # calc accuracy
        thresh = 0.65
        A_pred[A_pred > thresh] = 1
        A_pred[A_pred <= thresh] = 0
        A_real = A_real.long()
        A_pred = A_pred.long()
        correct = torch.eq(A_pred, A_real).long().sum()
        all = A_pred.size(0) * A_pred.size(1) * A_pred.size(2)

        true = A_real.view(-1).data.cpu().numpy()
        pred = A_pred_raw.view(-1).data.cpu().numpy()


        auc = roc_auc_score(true, pred)
        ap = average_precision_score(true, pred)

        correct_sum += correct.data[0]
        all_sum += all
        auc_mean += auc
        ap_mean += ap
        real_score_mean += A_real.float().mean().data[0]
        pred_score_mean += A_pred_raw.mean().data[0]
        count += 1
        # print(A_pred_raw)
        if count <= 16 and epoch>=0:
            # print('make real graph')
            logging.warning('real graph {}'.format(count))
            make_graph(A_real.data.cpu().numpy(),key='real'+str(epoch))
            # print('make pred graph')
            logging.warning('pred graph {}'.format(count))
            make_graph_random(A_pred_raw.data.cpu().numpy(),key='pred'+str(epoch))
            make_graph(A_pred.data.cpu().numpy(),key='pred'+str(epoch))


    loss_mean /= count
    auc_mean /= count
    ap_mean /= count
    real_score_mean /= count
    pred_score_mean /= count
    accuracy = correct_sum / float(all_sum)
    # print('Epoch: {}/{}, test loss: {:.6f}, accuracy: {}/{} {:.4f}, auc: {:.4f}, ap: {:.4f}, real mean: {:.4f}, pred mean: {:.4f}'.format(
    #     epoch, args.epochs, loss_mean, correct_sum, all_sum, accuracy, auc_mean, ap_mean, real_score_mean, pred_score_mean))
    logging.warning('Epoch: {}/{}, test loss: {:.6f}, accuracy: {}/{} {:.4f}, auc: {:.4f}, ap: {:.4f}, real mean: {:.4f}, pred mean: {:.4f}'.format(
        epoch, args.epochs, loss_mean, correct_sum, all_sum, accuracy, auc_mean, ap_mean, real_score_mean, pred_score_mean))