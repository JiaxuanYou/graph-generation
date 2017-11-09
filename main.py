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
import node2vec.src.main as nv
from sklearn.decomposition import PCA
import logging
from time import gmtime, strftime
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from utils import *
from model import *
from data import *
from random import shuffle
import pickle



CUDA = 0



class Args():
    def __init__(self):
        self.seed = 123

        ### data config
        # self.graph_type = 'star'
        # self.graph_type = 'ladder'
        # self.graph_type = 'karate'
        # self.graph_type = 'tree'
        # self.graph_type = 'caveman'
        # self.graph_type = 'grid'
        # self.graph_type = 'barabasi'
        self.graph_type = 'enzymes'
        # self.graph_type = 'protein'


        self.graph_node_num = 50
        self.max_prev_node = 40


        ### network config
        ## GraphRNN
        self.input_size = 64
        self.hidden_size = 64
        self.batch_size = 128
        self.num_layers = 3
        ## GCN
        self.output_dim = 64
        self.hidden_dim = 64

        ### training config
        self.lr = 0.01
        self.epochs = 50000
        self.epochs_test = 500
        self.epochs_log = 500
        self.epochs_save = 500
        self.milestones = [8000, 16000]
        self.lr_rate = 0.1
        self.sample_when_validate = True
        # self.sample_when_validate = False

        ### output config
        self.model_save_path = 'model_save/'
        self.graph_save_path = 'graphs/'
        self.figure_save_path = 'figures/'
        self.load = False
        self.save = True
        # self.note = 'GraphRNN'
        self.note = 'GraphRNN_AE'
        # self.note = 'GCN'



# ############# node2vec config###############
# args = nv.config(dimension=128, walk_length = 80, num_walks = 10, window_size = 2)
# ############################################
# ######### try node2vec #############
# for edge in G.edges():
#     G[edge[0]][edge[1]]['weight'] = 1
# embedding = nv.node2vec_main(G, args)
# print(embedding)
# # print('embedding.shape', embedding.shape)
#
# embedding_dist = np.zeros((embedding.shape[0],embedding.shape[0]))
# embedded = embedding
# pca = PCA(n_components=2)
# embedded = pca.fit(embedding).transform(embedding)
# plt.switch_backend('agg')
# plt.scatter(embedded[:,0], embedded[:,1])
# for i in range(embedded.shape[0]):
#     plt.text(embedded[i, 0], embedded[i, 1], str(i))
#     for j in list(G.adj[i]):
#         plt.plot([embedded[i,0],embedded[j,0]],[embedded[i,1],embedded[j,1]],
#                  color = 'r', linewidth = 0.5)
# plt.savefig('figures/graph_view_node2vec.png')






def sample_y(y,sample, thresh=0.5):
    # do sampling
    if sample:
        y_thresh = Variable(torch.rand(y.size(0),y.size(1),y.size(2))).cuda(CUDA)
        y_result = torch.gt(y,y_thresh).float()
    # do max likelihood based on some threshold
    else:
        y_thresh = Variable(torch.ones(y.size(0), y.size(1), y.size(2))*thresh).cuda(CUDA)
        y_result = torch.gt(y, y_thresh).float()
    return y_result




def train_epoch(epoch, args, generator, dataset, optimizer, scheduler, thresh, train=True):
    generator.train()
    optimizer.zero_grad()
    generator.hidden = generator.init_hidden()


    x,y,_,_,_ = dataset.sample()
    x = Variable(x).cuda(CUDA)
    y = Variable(y).cuda(CUDA)
    # if train
    if train:
        y_pred = generator(x)

    # if validate, do sampling/threshold each step
    else:
        y_pred = Variable(torch.zeros(x.size(0),x.size(1),x.size(2))).cuda(CUDA)
        y_pred_long = Variable(torch.zeros(x.size(0),x.size(1),x.size(2))).cuda(CUDA)
        x_step = x[:,0:1,:]
        for i in range(x.size(1)):
            y_step = generator(x_step)
            y_pred[:, i, :] = y_step
            y_step = F.sigmoid(y_step)
            x_step = sample_y(y_step, sample=args.sample_when_validate, thresh = 0.45)
            y_pred_long[:,i,:] = x_step
        y_pred_long = y_pred_long.long()



    alpha = 1
    loss = F.binary_cross_entropy_with_logits(y_pred, y, weight=alpha)

    if train:
        loss.backward()
        optimizer.step()
        scheduler.step()



    y_data = y.data
    y_pred_data = F.sigmoid(y_pred).data


    y_data_flat = y_data.view(-1).cpu().numpy()
    y_pred_data_flat = y_pred_data.view(-1).cpu().numpy()



    if epoch % args.epochs_log == 0 and epoch>0:
        fpr, tpr, thresholds = roc_curve(y_data_flat, y_pred_data_flat)
        if train:
            thresh = thresholds[np.nonzero(fpr > 0.05)[0][0]].item()
        ap = average_precision_score(y_data_flat,y_pred_data_flat)
        auc = roc_auc_score(y_data_flat,y_pred_data_flat)
        print('is_train:',train,'ap', ap, 'auc', auc, 'thresh', thresh)


    # if epoch % args.epochs_log == 0:
    #     np.set_printoptions(precision=3)
    #     print('real\n', y_data[0])
    #     print('pred\n', y_pred_data[0])

    real_score_mean = y_data.mean()
    pred_score_mean = y_pred_data.mean()

    # calc accuracy
    # thresh = 0.03
    y_pred_data[y_pred_data>thresh] = 1
    y_pred_data[y_pred_data<=thresh] = 0
    y_data = y_data.long()
    y_pred_data = y_pred_data.long()
    if train==False:
        y_pred_data = y_pred_long.data

    correct = torch.eq(y_pred_data, y_data).long().sum()
    all = y_pred_data.size(0)*y_pred_data.size(1)*y_pred_data.size(2)

    # plot graph
    if epoch % args.epochs_log == 0 and train==False:
        # save graphs as pickle
        G_real_list = []
        G_pred_list = []
        for i in range(y_data.size(0)):
            adj_real = decode_adj(y_data[i].cpu().numpy(), args.max_prev_node)
            adj_pred = decode_adj(y_pred_data[i].cpu().numpy(), args.max_prev_node)
            G_real = get_graph(adj_real)
            G_pred = get_graph(adj_pred)
            G_real_list.append(G_real)
            G_pred_list.append(G_pred)
        # save list of objects
        fname_pred = args.graph_save_path + args.note+ '_'+ args.graph_type + '_' + str(args.graph_node_num) + '_' +\
                     str(epoch) + '_pred_' + str(args.sample_when_validate)+'_'+str(args.num_layers)+'.dat'
        save_graph_list(G_pred_list,fname_pred)
        fname_real = args.graph_save_path + args.note + '_' + args.graph_type + '_' + str(args.graph_node_num) + '_' + \
                     str(epoch) + '_real_' + str(args.sample_when_validate)+'_'+str(args.num_layers)+'.dat'
        save_graph_list(G_real_list, fname_real)

        # adj_real = np.zeros((y_data.size(1)+1,y_data.size(1)+1))
        # adj_real[1:y_data.size(1)+1,0:y_data.size(1)] = np.tril(y_data[0].cpu().numpy(),0)
        # adj_real = adj_real+adj_real.T
        # adj_real = decode_adj(y_data[0].cpu().numpy(),args.max_prev_node)

        # adj_pred = np.zeros((y_data.size(1)+1, y_data.size(1)+1))
        # adj_pred[1:y_data.size(1)+1, 0:y_data.size(1)] = np.tril(y_pred_data[0].cpu().numpy(),0)
        # adj_pred = adj_pred + adj_pred.T
        # adj_pred = decode_adj(y_pred_data[0].cpu().numpy(),args.max_prev_node)

        # decode_graph(adj_real,args.graph_type+'_'+str(args.graph_node_num)+'_'+str(epoch)+'real'+str(args.sample_when_validate)+str(args.num_layers))
        # decode_graph(adj_pred,args.graph_type+'_'+str(args.graph_node_num)+'_'+str(epoch)+'pred'+str(args.sample_when_validate)+str(args.num_layers))

    accuracy = correct/float(all)
    if epoch%args.epochs_log==0 and train==True:
        print('Epoch: {}/{}, train loss: {:.6f}, accuracy: {}/{} {:.4f}, real mean: {:.4f}, pred mean: {:.4f}, sample: {}'.format(
                epoch, args.epochs, loss.data[0], correct, all, accuracy, real_score_mean, pred_score_mean, args.sample_when_validate))
        logging.warning('Epoch: {}/{}, train loss: {:.6f}, accuracy: {}/{} {:.4f}, real mean: {:.4f}, pred mean: {:.4f}, sample: {}'.format(
            epoch, args.epochs, loss.data[0], correct, all, accuracy, real_score_mean, pred_score_mean, args.sample_when_validate))
    elif epoch%args.epochs_log==0 and train==False:
        print('Epoch: {}/{}, test loss: {:.6f}, accuracy: {}/{} {:.4f}, real mean: {:.4f}, pred mean: {:.4f}'.format(
                epoch, args.epochs, loss.data[0], correct, all, accuracy, real_score_mean, pred_score_mean))
        logging.warning('Epoch: {}/{}, test loss: {:.6f}, accuracy: {}/{} {:.4f}, real mean: {:.4f}, pred mean: {:.4f}'.format(
            epoch, args.epochs, loss.data[0], correct, all, accuracy, real_score_mean, pred_score_mean))

    # log_value('train_loss', loss_mean, epoch)
    # log_value('accuracy', accuracy, epoch)
    # log_value('auc_mean', auc_mean, epoch)
    # log_value('ap_mean', ap_mean, epoch)
    return thresh


def train_epoch_GCN(epoch, args, encoder, decoder, dataset, optimizer, scheduler, thresh, train=True):
    encoder.train()
    decoder.train()
    optimizer.zero_grad()


    _,_,y, y_norm,x = dataset.sample()
    y = Variable(y).cuda(CUDA)
    y_norm = Variable(y_norm).cuda(CUDA)
    x = Variable(x).cuda(CUDA)

    z = encoder(x, y_norm)
    y_pred = decoder(z)

    alpha = 1
    loss = F.binary_cross_entropy_with_logits(y_pred, y, weight=alpha)

    if train:
        loss.backward()
        optimizer.step()
        scheduler.step()

    y_data = y.data
    y_pred_data = F.sigmoid(y_pred).data


    y_data_flat = y_data.view(-1).cpu().numpy()
    y_pred_data_flat = y_pred_data.view(-1).cpu().numpy()


    if epoch % args.epochs_log == 0 and epoch>0:
        fpr, tpr, thresholds = roc_curve(y_data_flat, y_pred_data_flat)
        if train:
            thresh = thresholds[np.nonzero(fpr > 0.05)[0][0]].item()
        ap = average_precision_score(y_data_flat,y_pred_data_flat)
        auc = roc_auc_score(y_data_flat,y_pred_data_flat)
        print('is_train:',train,'ap', ap, 'auc', auc, 'thresh', thresh)


    # if epoch % args.epochs_log == 0:
    #     np.set_printoptions(precision=3)
    #     print('real\n', y_data[0])
    #     print('pred\n', y_pred_data[0])

    real_score_mean = y_data.mean()
    pred_score_mean = y_pred_data.mean()

    # calc accuracy
    thresh = 0.45
    if args.sample_when_validate:
        y_thresh = torch.rand(y_pred_data.size(0), y_pred_data.size(1), y_pred_data.size(2)).cuda(CUDA)
        y_pred_data = torch.gt(y_pred_data, y_thresh).long()
    else:
        y_pred_data[y_pred_data>thresh] = 1
        y_pred_data[y_pred_data<=thresh] = 0
    y_data = y_data.long()
    y_pred_data = y_pred_data.long()


    correct = torch.eq(y_pred_data, y_data).long().sum()
    all = y_pred_data.size(0)*y_pred_data.size(1)*y_pred_data.size(2)

    # plot graph
    if epoch % args.epochs_log == 0 and train==False:
        # save graphs as pickle
        G_real_list = []
        G_pred_list = []
        for i in range(y_data.size(0)):
            adj = np.asmatrix(y_data[i].cpu().numpy())
            G_real = nx.from_numpy_matrix(adj)
            adj = np.asmatrix(y_pred_data[i].cpu().numpy())
            G_pred = nx.from_numpy_matrix(adj)
            G_real_list.append(G_real)
            G_pred_list.append(G_pred)
        # save list of objects
        fname_pred = args.graph_save_path + args.note+ '_'+ args.graph_type + '_' + str(args.graph_node_num) + '_' +\
                     str(epoch) + '_pred_' + str(args.sample_when_validate)+'_'+str(args.num_layers)+'.dat'
        save_graph_list(G_pred_list,fname_pred)
        fname_real = args.graph_save_path + args.note + '_' + args.graph_type + '_' + str(args.graph_node_num) + '_' + \
                     str(epoch) + '_real_' + str(args.sample_when_validate)+'_'+str(args.num_layers)+'.dat'
        save_graph_list(G_real_list, fname_real)

        # adj_real = np.zeros((y_data.size(1)+1,y_data.size(1)+1))
        # adj_real[1:y_data.size(1)+1,0:y_data.size(1)] = np.tril(y_data[0].cpu().numpy(),0)
        # adj_real = adj_real+adj_real.T
        # adj_real = decode_adj(y_data[0].cpu().numpy(),args.max_prev_node)

        # adj_pred = np.zeros((y_data.size(1)+1, y_data.size(1)+1))
        # adj_pred[1:y_data.size(1)+1, 0:y_data.size(1)] = np.tril(y_pred_data[0].cpu().numpy(),0)
        # adj_pred = adj_pred + adj_pred.T
        # adj_pred = decode_adj(y_pred_data[0].cpu().numpy(),args.max_prev_node)

        # decode_graph(adj_real,args.graph_type+'_'+str(args.graph_node_num)+'_'+str(epoch)+'real'+str(args.sample_when_validate)+str(args.num_layers))
        # decode_graph(adj_pred,args.graph_type+'_'+str(args.graph_node_num)+'_'+str(epoch)+'pred'+str(args.sample_when_validate)+str(args.num_layers))

    accuracy = correct/float(all)
    if epoch%args.epochs_log==0 and train==True:
        print('Epoch: {}/{}, train loss: {:.6f}, accuracy: {}/{} {:.4f}, real mean: {:.4f}, pred mean: {:.4f}, sample: {}'.format(
                epoch, args.epochs, loss.data[0], correct, all, accuracy, real_score_mean, pred_score_mean, args.sample_when_validate))
        logging.warning('Epoch: {}/{}, train loss: {:.6f}, accuracy: {}/{} {:.4f}, real mean: {:.4f}, pred mean: {:.4f}, sample: {}'.format(
            epoch, args.epochs, loss.data[0], correct, all, accuracy, real_score_mean, pred_score_mean, args.sample_when_validate))
    elif epoch%args.epochs_log==0 and train==False:
        print('Epoch: {}/{}, test loss: {:.6f}, accuracy: {}/{} {:.4f}, real mean: {:.4f}, pred mean: {:.4f}'.format(
                epoch, args.epochs, loss.data[0], correct, all, accuracy, real_score_mean, pred_score_mean))
        logging.warning('Epoch: {}/{}, test loss: {:.6f}, accuracy: {}/{} {:.4f}, real mean: {:.4f}, pred mean: {:.4f}'.format(
            epoch, args.epochs, loss.data[0], correct, all, accuracy, real_score_mean, pred_score_mean))

    # log_value('train_loss', loss_mean, epoch)
    # log_value('accuracy', accuracy, epoch)
    # log_value('auc_mean', auc_mean, epoch)
    # log_value('ap_mean', ap_mean, epoch)
    return thresh







def train_epoch_autoencoder(epoch, args, encoder, generator, dataset, optimizer, scheduler, thresh, train=True):
    encoder.train()
    generator.train()
    optimizer.zero_grad()
    generator.hidden = generator.init_hidden()

    # x: batch*number_nodes*feature
    x,y,adj,adj_norm,feature = dataset.sample()
    x = Variable(x).cuda(CUDA)
    y = Variable(y).cuda(CUDA)
    adj = Variable(adj).cuda(CUDA)
    adj_norm = Variable(adj_norm).cuda(CUDA)
    feature = Variable(feature).cuda(CUDA)

    # hidden shape: num_layers*batch*feature
    hidden = encoder(feature, adj_norm)
    generator.hidden = hidden.contiguous()
    generator.cell = Variable(torch.zeros(hidden.size(0),hidden.size(1),hidden.size(2))).cuda(CUDA)

    # if train
    if train:
        # use encoded vector as the hidden state of generator
        # todo: try to put the vector in multiple/single layer, in/not in cell state
        y_pred = generator(x)

    # if validate, do sampling/threshold each step
    else:
        y_pred = Variable(torch.zeros(x.size(0),x.size(1),x.size(2))).cuda(CUDA)
        y_pred_long = Variable(torch.zeros(x.size(0),x.size(1),x.size(2))).cuda(CUDA)
        x_step = x[:,0:1,:]
        for i in range(x.size(1)):
            y_step = generator(x_step)
            y_pred[:, i, :] = y_step
            y_step = F.sigmoid(y_step)
            x_step = sample_y(y_step, sample=args.sample_when_validate, thresh = 0.45)
            y_pred_long[:,i,:] = x_step
        y_pred_long = y_pred_long.long()



    alpha = 1
    loss = F.binary_cross_entropy_with_logits(y_pred, y, weight=alpha)

    if train:
        loss.backward()
        optimizer.step()
        scheduler.step()
        # print gradient
        # print('conv_first grad', torch.norm(encoder.conv_first.weight.grad.data))
        # print('lstm grad', torch.norm(generator.lstm.weight_ih_l0.grad.data))



    y_data = y.data
    y_pred_data = F.sigmoid(y_pred).data


    y_data_flat = y_data.view(-1).cpu().numpy()
    y_pred_data_flat = y_pred_data.view(-1).cpu().numpy()



    if epoch % args.epochs_log == 0 and epoch>0:
        fpr, tpr, thresholds = roc_curve(y_data_flat, y_pred_data_flat)
        if train:
            thresh = thresholds[np.nonzero(fpr > 0.05)[0][0]].item()
        ap = average_precision_score(y_data_flat,y_pred_data_flat)
        auc = roc_auc_score(y_data_flat,y_pred_data_flat)
        print('is_train:',train,'ap', ap, 'auc', auc, 'thresh', thresh)


    # if epoch % args.epochs_log == 0:
    #     np.set_printoptions(precision=3)
    #     print('real\n', y_data[0])
    #     print('pred\n', y_pred_data[0])

    real_score_mean = y_data.mean()
    pred_score_mean = y_pred_data.mean()

    # calc accuracy
    # thresh = 0.03
    y_pred_data[y_pred_data>thresh] = 1
    y_pred_data[y_pred_data<=thresh] = 0
    y_data = y_data.long()
    y_pred_data = y_pred_data.long()
    if train==False:
        y_pred_data = y_pred_long.data

    correct = torch.eq(y_pred_data, y_data).long().sum()
    all = y_pred_data.size(0)*y_pred_data.size(1)*y_pred_data.size(2)

    # plot graph
    if epoch % args.epochs_log == 0 and train == False:
        # save graphs as pickle
        G_real_list = []
        G_pred_list = []
        for i in range(y_data.size(0)):
            adj_real = decode_adj(y_data[i].cpu().numpy(), args.max_prev_node)
            adj_pred = decode_adj(y_pred_data[i].cpu().numpy(), args.max_prev_node)
            G_real = get_graph(adj_real)
            G_pred = get_graph(adj_pred)
            G_real_list.append(G_real)
            G_pred_list.append(G_pred)
        # save list of objects
        fname_pred = args.graph_save_path + args.note + '_' + args.graph_type + '_' + str(args.graph_node_num) + '_' + \
                     str(epoch) + '_pred_' + str(args.sample_when_validate) + '_' + str(args.num_layers) + '.dat'
        save_graph_list(G_pred_list, fname_pred)
        fname_real = args.graph_save_path + args.note + '_' + args.graph_type + '_' + str(args.graph_node_num) + '_' + \
                     str(epoch) + '_real_' + str(args.sample_when_validate) + '_' + str(args.num_layers) + '.dat'
        save_graph_list(G_real_list, fname_real)

    accuracy = correct/float(all)
    if epoch%args.epochs_log==0 and train==True:
        print('Epoch: {}/{}, train loss: {:.6f}, accuracy: {}/{} {:.4f}, real mean: {:.4f}, pred mean: {:.4f}, sample: {}'.format(
                epoch, args.epochs, loss.data[0], correct, all, accuracy, real_score_mean, pred_score_mean, args.sample_when_validate))
        logging.warning('Epoch: {}/{}, train loss: {:.6f}, accuracy: {}/{} {:.4f}, real mean: {:.4f}, pred mean: {:.4f}, sample: {}'.format(
            epoch, args.epochs, loss.data[0], correct, all, accuracy, real_score_mean, pred_score_mean, args.sample_when_validate))
    elif epoch%args.epochs_log==0 and train==False:
        print('Epoch: {}/{}, test loss: {:.6f}, accuracy: {}/{} {:.4f}, real mean: {:.4f}, pred mean: {:.4f}'.format(
                epoch, args.epochs, loss.data[0], correct, all, accuracy, real_score_mean, pred_score_mean))
        logging.warning('Epoch: {}/{}, test loss: {:.6f}, accuracy: {}/{} {:.4f}, real mean: {:.4f}, pred mean: {:.4f}'.format(
            epoch, args.epochs, loss.data[0], correct, all, accuracy, real_score_mean, pred_score_mean))

    # log_value('train_loss', loss_mean, epoch)
    # log_value('accuracy', accuracy, epoch)
    # log_value('auc_mean', auc_mean, epoch)
    # log_value('ap_mean', ap_mean, epoch)
    return thresh


def train(args, dataset_train, generator):
    if args.load:
        epoch_load = 6100
        fname = args.model_save_path + args.note + '_' + args.graph_type + '_' + str(args.graph_node_num) + \
                '_' + str(epoch_load) + str(args.sample_when_validate) + '_' + str(args.num_layers) + '.dat'
        generator.load_state_dict(torch.load(fname))
        args.lr = 0.00001
        epoch = epoch_load
        print('model loaded!')
    else:
        epoch = 0

    torch.manual_seed(args.seed)
    optimizer = optim.Adam(list(generator.parameters()), lr=args.lr)
    scheduler = MultiStepLR(optimizer, milestones=args.milestones, gamma=args.lr_rate)
    thresh = 0.3


    while epoch<=args.epochs:
        thresh = train_epoch(epoch, args, generator, dataset_train, optimizer, scheduler, thresh, train=True)
        if epoch % args.epochs_test == 0:
            train_epoch(epoch, args, generator, dataset_train, optimizer, scheduler, thresh, train = False)
        if args.save:
            if epoch % args.epochs_save == 0:
                fname = args.model_save_path + args.note + '_' + args.graph_type + '_' + str(args.graph_node_num) +\
                '_' + str(epoch) + str(args.sample_when_validate) +'_'+str(args.num_layers) + '.dat'
                torch.save(generator.state_dict(), fname)
        epoch += 1

def train_AE(args, dataset_train,encoder, generator):
    if args.load:
        epoch_load = 6100
        fname = args.model_save_path + args.note + '_' + args.graph_type + '_' + str(args.graph_node_num) + \
                '_' + str(epoch_load) + str(args.sample_when_validate) + '_' + str(args.num_layers) + '_generator.dat'
        generator.load_state_dict(torch.load(fname))
        fname = args.model_save_path + args.note + '_' + args.graph_type + '_' + str(args.graph_node_num) + \
                '_' + str(epoch_load) + str(args.sample_when_validate) + '_' + str(args.num_layers) + '_encoder.dat'
        encoder.load_state_dict(torch.load(fname))
        args.lr = 0.00001
        epoch = epoch_load
        print('model loaded!')
    else:
        epoch = 0

    torch.manual_seed(args.seed)
    optimizer = optim.Adam(list(generator.parameters()), lr=args.lr)
    scheduler = MultiStepLR(optimizer, milestones=args.milestones, gamma=args.lr_rate)
    thresh = 0.3


    while epoch<=args.epochs:
        thresh = train_epoch_autoencoder(epoch, args,encoder, generator, dataset_train, optimizer, scheduler, thresh, train=True)
        if epoch % args.epochs_test == 0:
            train_epoch_autoencoder(epoch, args,encoder, generator, dataset_train, optimizer, scheduler, thresh, train = False)
        if args.save:
            if epoch % args.epochs_save == 0:
                fname = args.model_save_path + args.note + '_' + args.graph_type + '_' + str(args.graph_node_num) + \
                        '_' + str(epoch) + str(args.sample_when_validate) + '_' + str(args.num_layers) + '_encoder.dat'
                torch.save(encoder.state_dict(), fname)
                fname = args.model_save_path + args.note + '_' + args.graph_type + '_' + str(args.graph_node_num) +\
                '_' + str(epoch) + str(args.sample_when_validate) +'_'+str(args.num_layers) + '_generator.dat'
                torch.save(generator.state_dict(), fname)
        epoch += 1


def train_GCN(args, dataset_train, encoder, decoder):
    if args.load:
        epoch_load = 6100
        fname = args.model_save_path + args.note + '_' + args.graph_type + '_' + str(args.graph_node_num) + \
                '_' + str(epoch_load) + str(args.sample_when_validate) + '_' + str(args.num_layers) + '.dat'
        encoder.load_state_dict(torch.load(fname))
        args.lr = 0.00001
        epoch = epoch_load
        print('model loaded!')
    else:
        epoch = 0

    torch.manual_seed(args.seed)
    optimizer = optim.Adam(list(encoder.parameters()), lr=args.lr)
    scheduler = MultiStepLR(optimizer, milestones=args.milestones, gamma=args.lr_rate)
    thresh = 0.45

    while epoch <= args.epochs:
        thresh = train_epoch_GCN(epoch, args, encoder, decoder, dataset_train, optimizer, scheduler, thresh, train=True)
        if epoch % args.epochs_test == 0:
            train_epoch_GCN(epoch, args, encoder, decoder, dataset_train, optimizer, scheduler, thresh, train=False)
        if args.save:
            if epoch % args.epochs_save == 0:
                fname = args.model_save_path + args.note + '_' + args.graph_type + '_' + str(
                    args.graph_node_num) + '_' + str(epoch) + str(args.sample_when_validate) + '_' + str(args.num_layers) + '.dat'
                torch.save(encoder.state_dict(), fname)
        epoch += 1

################ start test code
if __name__ == '__main__':
    time = strftime("%Y-%m-%d %H:%M:%S", gmtime())
    logging.basicConfig(filename='logs/train' + time + '.log', level=logging.DEBUG)
    args = Args()

    # if using a single graph
    if args.graph_type=='star':
        G = nx.star_graph(args.graph_node_num)
    if args.graph_type=='ladder':
        G = nx.ladder_graph(args.graph_node_num)
    if args.graph_type=='karate':
        G = nx.karate_club_graph()
    if args.graph_type=='tree':
        G = nx.balanced_tree(3,3)
    if args.graph_type=='caveman':
        G = nx.connected_caveman_graph(8,6)
    if args.graph_type=='grid':
        G = nx.grid_2d_graph(6,6)
    if args.graph_type=='barabasi':
        G = nx.barabasi_albert_graph(args.graph_node_num,2)
    # if using a batch of graphs
    if args.graph_type == 'enzymes':
        graphs, max_num_nodes = Graph_load_batch(min_num_nodes=6, name='ENZYMES')
        print('max num nodes', max_num_nodes)
    if args.graph_type == 'protein':
        graphs, max_num_nodes = Graph_load_batch(min_num_nodes=6, name='PROTEINS_full')
        print('max num nodes', max_num_nodes)



    ################## the GraphRNN model #####################
    # # if using a single graph
    # sampler = Graph_sequence_sampler_bfs_permute_truncate(G, batch_size=args.batch_size, max_prev_node=args.max_prev_node)
    # x, y = sampler.sample()


    sampler = Graph_sequence_sampler_bfs_permute_truncate_multigraph(graphs, max_node_num=max_num_nodes,batch_size=args.batch_size, max_prev_node=args.max_prev_node)
    x, y, adj,adj_norm,feature = sampler.sample()


    encoder = GCN_encoder_graph(feature.size(2),args.hidden_dim,args.input_size,args.num_layers).cuda(CUDA)
    generator = Graph_generator_LSTM_graph(feature_size=x.size(2), input_size=args.input_size, hidden_size=args.hidden_size,
                                           output_size=y.size(2), batch_size=args.batch_size, num_layers=args.num_layers).cuda(CUDA)

    train_AE(args,sampler,encoder,generator)


    ################## the GCN baseline model #################
    # sampler = Graph_sequence_sampler_bfs_permute_truncate_multigraph([G], max_node_num=G.number_of_nodes(), batch_size=args.batch_size
    #                                                                  , max_prev_node = args.max_prev_node, feature = True)
    # encoder = GCN_encoder(input_dim=G.number_of_nodes(),hidden_dim=args.hidden_dim,output_dim=args.output_dim)
    # decoder = GCN_decoder()
    #
    # train_GCN(args,sampler,encoder, decoder)