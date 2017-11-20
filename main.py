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
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from time import gmtime, strftime
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from utils import *
from model import *
from data import *
from random import shuffle
import pickle
from tensorboard_logger import configure, log_value






class Args():
    def __init__(self):
        self.seed = 123

        ### data config
        # self.graph_type = 'star'
        self.graph_type = 'ladder'
        # self.graph_type = 'karate'
        # self.graph_type = 'tree'
        # self.graph_type = 'caveman'
        # self.graph_type = 'grid'
        # self.graph_type = 'barabasi'
        # self.graph_type = 'enzymes'
        # self.graph_type = 'protein'
        # self.graph_type = 'DD'


        ## self.graph_node_num = 50 # obsolete

        # self.max_prev_node = 150 # max previous node that looks back
        # self.max_prev_node = 100
        self.max_prev_node = 50
        # self.max_prev_node = 25


        ### network config
        ## GraphRNN
        self.input_size = 128
        self.hidden_size = 128
        self.batch_size = 128
        self.num_layers = 4
        self.is_dilation = True
        self.is_flexible = False # if use flexible input size
        self.is_bn = True
        self.bptt = False # if use truncate back propagation (not very stable)
        self.bptt_len = 20
        ## GCN
        self.output_dim = 64
        self.hidden_dim = 64

        ### training config
        self.lr = 0.01
        self.epochs = 50000
        # self.epochs = 100000
        self.epochs_test = 500
        self.epochs_log = 500
        self.epochs_save = 500
        self.milestones = [4000, 10000, 20000]
        # self.milestones = [16000, 32000]

        self.lr_rate = 0.3
        self.sample_when_validate = True
        self.sample_time = 2
        # self.sample_when_validate = False

        ### output config
        self.model_save_path = 'model_save/'
        self.graph_save_path = 'graphs/'
        self.figure_save_path = 'figures/'
        self.load = False
        self.load_epoch = 8000
        self.save = True
        self.note = 'GraphRNN'
        # self.note = 'GraphRNN_AE'
        # self.note = 'GraphRNN_structure'
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


def sample_y(y,sample, thresh=0.5,multi_sample=1):
    '''
        do sampling over output
    :param y: input
    :param sample: Bool
    :param thresh: if not sample, the threshold
    :param multi_sample: how many times do we sample, if =1, do single sample
    :return: sampled result
    '''
    # do sampling
    if sample:
        if multi_sample>1:
            y_result = Variable(torch.rand(y.size(0),y.size(1),y.size(2))).cuda(CUDA)
            # loop over all batches
            for i in range(y_result.size(0)):
                # do 'multi_sample' times sampling
                for j in range(multi_sample):
                    y_thresh = Variable(torch.rand(y.size(1), y.size(2))).cuda(CUDA)
                    y_result[i] = torch.gt(y[i], y_thresh).float()
                    if (torch.sum(y_result[i]).data>0).any():
                        break
                    # else:
                    #     print('all zero',j)
        else:
            y_thresh = Variable(torch.rand(y.size(0),y.size(1),y.size(2))).cuda(CUDA)
            y_result = torch.gt(y,y_thresh).float()
    # do max likelihood based on some threshold
    else:
        y_thresh = Variable(torch.ones(y.size(0), y.size(1), y.size(2))*thresh).cuda(CUDA)
        y_result = torch.gt(y, y_thresh).float()
    return y_result


def detach_hidden_lstm(hidden):
    # return (Variable(hidden[0].data).cuda(CUDA),Variable(hidden[1].data).cuda(CUDA))
    return (hidden[0].detach(),hidden[1].detach())

############# this is the baseline method used, LSTM
def train_epoch(epoch, args, generator, dataset, optimizer, scheduler, thresh, train=True):
    generator.train()
    optimizer.zero_grad()
    generator.hidden = generator.init_hidden()


    x,y,len = dataset.sample()
    x = Variable(x).cuda(CUDA)
    y = Variable(y).cuda(CUDA)
    # if train
    y_pred = Variable(torch.ones(x.size(0), x.size(1), x.size(2))*-100).cuda(CUDA)
    if train:
        # if do truncate backprop
        if args.bptt:
            start_id = 0
            while start_id<x.size(1):
                print('start id',start_id)
                end_id = max(start_id+args.bptt_len,x.size(1))
                y_pred_temp = generator(x[:,start_id:end_id,:])
                generator.hidden = detach_hidden_lstm(generator.hidden)
                # generator.hidden[0].detach()
                # generator.hidden[1].detach()

                y_pred[:,start_id:end_id,:] = y_pred_temp
                start_id += args.bptt_len
            y_pred_clean = Variable(torch.ones(x.size(0), x.size(1), x.size(2))*-100).cuda(CUDA)
            # before computing loss, cleaning y_pred so that only valid entries are supervised
            y_pred = pack_padded_sequence(y_pred, len, batch_first=True)
            y_pred = pad_packed_sequence(y_pred, batch_first=True)[0]
            y_pred_clean[:, 0:y_pred.size(1), :] = y_pred
            y_pred = y_pred_clean
        # if backprop through the start
        else:
            y_pred_temp = generator(x,pack = True,len=len)
            # before computing loss, cleaning y_pred so that only valid entries are supervised
            y_pred_temp = pack_padded_sequence(y_pred_temp, len, batch_first=True)
            y_pred_temp = pad_packed_sequence(y_pred_temp, batch_first=True)[0]
            y_pred[:, 0:y_pred_temp.size(1), :] = y_pred_temp

    # if validate, do sampling/threshold each step
    else:
        y_pred_long = Variable(torch.zeros(x.size(0),x.size(1),x.size(2))).cuda(CUDA)
        x_step = x[:,0:1,:]
        for i in range(x.size(1)):
            y_step = generator(x_step)
            y_pred[:, i:i+1, :] = y_step
            y_step = F.sigmoid(y_step)
            x_step = sample_y(y_step, sample=args.sample_when_validate, thresh = 0.45, multi_sample=args.sample_time)
            y_pred_long[:,i:i+1,:] = x_step

        y_pred_long = y_pred_long.long()



    # when training, we are packing input with wrong predicitons (which is zero) at the end of the sequence. so the loss may look high, but it shouldn't matter
    loss = F.binary_cross_entropy_with_logits(y_pred, y)

    if train:
        loss.backward()
        optimizer.step()
        scheduler.step()



    y_data = y.data
    y_pred_data = F.sigmoid(y_pred).data


    # y_data_flat = y_data.view(-1).cpu().numpy()
    # y_pred_data_flat = y_pred_data.view(-1).cpu().numpy()



    # if epoch % args.epochs_log == 0 and epoch>0:
    #     fpr, tpr, thresholds = roc_curve(y_data_flat, y_pred_data_flat)
    #     if train:
    #         thresh = thresholds[np.nonzero(fpr > 0.05)[0][0]].item()
    #     ap = average_precision_score(y_data_flat,y_pred_data_flat)
    #     auc = roc_auc_score(y_data_flat,y_pred_data_flat)
    #     print('is_train:',train,'ap', ap, 'auc', auc, 'thresh', thresh)


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
            # adj_error = adj_real-adj_raw
            # print(np.amin(adj_error),np.amax(adj_error))
            G_real = get_graph(adj_real)
            G_pred = get_graph(adj_pred)
            # print('real', G_real.number_of_nodes())
            # print('pred', G_pred.number_of_nodes())
            G_real_list.append(G_real)
            G_pred_list.append(G_pred)
        # save list of objects
        fname_pred = args.graph_save_path + args.note+ '_'+ args.graph_type + '_' +\
                     str(epoch) + '_pred_' +str(args.num_layers)+'_'+str(args.bptt)+'_'+str(args.bptt_len)+'.dat'
        save_graph_list(G_pred_list,fname_pred)
        fname_real = args.graph_save_path + args.note + '_' + args.graph_type + '_' + \
                     str(epoch) + '_real_' +str(args.num_layers)+'_'+str(args.bptt)+'_'+str(args.bptt_len)+'.dat'
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
        print('Epoch: {}/{}, train loss: {:.6f}, accuracy: {}/{} {:.4f}, real mean: {:.4f}, pred mean: {:.4f}, graph type: {}, num_layer: {}, bptt: {}, bptt_len:{}'.format(
                epoch, args.epochs, loss.data[0], correct, all, accuracy, real_score_mean, pred_score_mean, args.graph_type, args.num_layers, args.bptt, args.bptt_len))
        # logging.warning('Epoch: {}/{}, train loss: {:.6f}, accuracy: {}/{} {:.4f}, real mean: {:.4f}, pred mean: {:.4f}, graph type: {}, num_layer: {}, bptt: {}'.format(
        #     epoch, args.epochs, loss.data[0], correct, all, accuracy, real_score_mean, pred_score_mean, args.sample_when_validate))
    elif epoch%args.epochs_log==0 and train==False:
        print('Epoch: {}/{}, test loss: {:.6f}, accuracy: {}/{} {:.4f}, real mean: {:.4f}, pred mean: {:.4f}'.format(
                epoch, args.epochs, loss.data[0], correct, all, accuracy, real_score_mean, pred_score_mean))
        # logging.warning('Epoch: {}/{}, test loss: {:.6f}, accuracy: {}/{} {:.4f}, real mean: {:.4f}, pred mean: {:.4f}'.format(
        #     epoch, args.epochs, loss.data[0], correct, all, accuracy, real_score_mean, pred_score_mean))

    log_value('train_loss', loss.data[0], epoch)
    log_value('accuracy', accuracy, epoch)
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







def train_epoch_AE(epoch, args, encoder, generator, dataset, optimizer, scheduler, thresh, train=True):
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




########## The proposed structure RNN model
def train_epoch_GraphRNN_structure(epoch, args, generator, dataset, optimizer, scheduler, thresh, train=True):
    generator.train()
    optimizer.zero_grad()
    # initialize hidden list

    generator.hidden_all = generator.init_hidden(len=args.max_prev_node)
    # print(len(generator.hidden_all))
    y= dataset.sample()

    # store the output
    y_pred = Variable(torch.zeros(y.size(0), y.size(1), y.size(2))).cuda(CUDA)
    y_pred_long = Variable(torch.zeros(y.size(0), y.size(1), y.size(2))).cuda(CUDA) # in long format (discrete)


    # is_teacher_forcing = True if random.random()<0.8 else False
    is_teacher_forcing = True
    for i in range(y.size(1)):
        if train:
            if is_teacher_forcing:
                y_step = Variable(y[:, i:i + 1, :]).cuda(CUDA)
                y_step_pred,_ = generator(y_step, teacher_forcing=True, temperature=0.1,bptt=args.bptt,flexible=args.is_flexible)
            else:
                y_step_pred,_ = generator(None, teacher_forcing=False, temperature=0.1,bptt=args.bptt,flexible=args.is_flexible)
        else:
            y_step_pred, y_step_pred_sample = generator(None, teacher_forcing=False, temperature=0.1,bptt=args.bptt,flexible=args.is_flexible)
            y_step_pred_sample = sample_y(y_step_pred_sample, sample=False, thresh=0.5) # do threshold
            y_pred_long[:, i:i+1, :] = y_step_pred_sample
        # write the prediction
        y_pred[:,i:i+1, :] = y_step_pred

    y_pred_long = y_pred_long.long()
    y = Variable(y).cuda(CUDA)
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



    # if epoch % args.epochs_log == 0 and epoch>0:
    #     fpr, tpr, thresholds = roc_curve(y_data_flat, y_pred_data_flat)
    #     if train:
    #         thresh = thresholds[np.nonzero(fpr > 0.05)[0][0]].item()
    #     ap = average_precision_score(y_data_flat,y_pred_data_flat)
    #     auc = roc_auc_score(y_data_flat,y_pred_data_flat)
    #     print('is_train:',train,'ap', ap, 'auc', auc, 'thresh', thresh)
    #

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
        # print(y_pred)
        # print(y_pred_long)

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
                     str(epoch) + '_pred_bptt_' + str(args.bptt)+'_'+str(args.num_layers)+'_dilation_'+str(args.is_dilation)+'_flexible_'+str(args.is_flexible)+'_bn_'+str(args.is_bn)+'_lr_'+str(args.lr)+'.dat'
        save_graph_list(G_pred_list,fname_pred)
        fname_real = args.graph_save_path + args.note + '_' + args.graph_type + '_' + str(args.graph_node_num) + '_' + \
                     str(epoch) + '_real_bptt_' + str(args.bptt)+'_'+str(args.num_layers)+'_dilation_'+str(args.is_dilation)+'_flexible_'+str(args.is_flexible)+'_bn_'+str(args.is_bn)+'_lr_'+str(args.lr)+'.dat'
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
        print('Epoch: {}/{}, train loss: {:.6f}, accuracy: {}/{} {:.4f}, real mean: {:.4f}, pred mean: {:.4f},layers:{}, bptt: {}, dilation: {}, bn:{}, lr:{}'.format(
                epoch, args.epochs, loss.data[0], correct, all, accuracy, real_score_mean, pred_score_mean,args.num_layers, args.bptt, args.is_dilation, args.is_bn, args.lr))
        logging.warning('Epoch: {}/{}, train loss: {:.6f}, accuracy: {}/{} {:.4f}, real mean: {:.4f}, pred mean: {:.4f},layers:{}, bptt: {}, dilation{}, bn:{}, lr:{}'.format(
            epoch, args.epochs, loss.data[0], correct, all, accuracy, real_score_mean, pred_score_mean,args.num_layers, args.bptt, args.is_dilation, args.is_bn, args.lr))
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




######### The flexible (batch_size=1) version of the proposed model
def train_epoch_GraphRNN_structure_flexible(epoch, args, generator, dataset, optimizer, scheduler, thresh, train=True):
    generator.train()
    optimizer.zero_grad()
    # initialize hidden list
    if args.is_flexible:
        generator.hidden_all = generator.init_hidden(len=1)
    # print(len(generator.hidden_all))
    y_raw,adj_raw = dataset.sample()

    # store the output
    y_max = max(len(y_raw[i]) for i in range(len(y_raw)))
    y = Variable(torch.zeros(1, len(y_raw), y_max)).cuda(CUDA)
    y_pred = Variable(torch.zeros(1, len(y_raw), y_max)).cuda(CUDA)
    y_pred_long = Variable(torch.zeros(1, len(y_raw), y_max)).cuda(CUDA)  # in long format (discrete)
    # print('y_pred_long',y_pred_long.size())

    # is_teacher_forcing = True if random.random()<0.8 else False
    is_teacher_forcing = True
    for i in range(len(y_raw)):
        y_step = torch.FloatTensor(y_raw[i])
        y_step = y_step.view(1, 1, y_step.size(0))
        y_step = Variable(y_step).cuda(CUDA)
        if train:
            if is_teacher_forcing:
                y_step_pred,_ = generator(y_step, teacher_forcing=True, temperature=0.1,bptt=args.bptt,flexible=args.is_flexible,max_prev_node=y_max)
            else:
                y_step_pred,_ = generator(None, teacher_forcing=False, temperature=0.1,bptt=args.bptt,flexible=args.is_flexible,max_prev_node=y_max)
        else:
            y_step_pred, y_step_pred_sample = generator(None, teacher_forcing=False, temperature=0.1,bptt=args.bptt,flexible=args.is_flexible,max_prev_node=y_max)
            y_step_pred_sample = sample_y(y_step_pred_sample, sample=False, thresh=0.5) # do threshold
            # print('y_step', y_step.size())
            # print('y_step_pred_sample',y_step_pred_sample.size())
            y_pred_long[:, i:i+1, -1*y_step_pred_sample.size(2):] = y_step_pred_sample
        # write the prediction
        # print(i, y_step.size())
        # print(i, y_step_pred.size())
        y[:,i:i+1, -1*y_step.size(2):] = y_step
        y_pred[:,i:i+1, -1*y_step_pred.size(2):] = y_step_pred




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



    # if epoch % args.epochs_log == 0 and epoch>0:
    #     fpr, tpr, thresholds = roc_curve(y_data_flat, y_pred_data_flat)
    #     if train:
    #         thresh = thresholds[np.nonzero(fpr > 0.05)[0][0]].item()
    #     ap = average_precision_score(y_data_flat,y_pred_data_flat)
    #     auc = roc_auc_score(y_data_flat,y_pred_data_flat)
    #     print('is_train:',train,'ap', ap, 'auc', auc, 'thresh', thresh)
    #

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
        # print(y_pred)
        # print(y_pred_long)

    correct = torch.eq(y_pred_data, y_data).long().sum()
    all = y_pred_data.size(0)*y_pred_data.size(1)*y_pred_data.size(2)

    # plot graph
    if epoch % args.epochs_log == 0 and train==False:
        # save graphs as pickle
        G_real_list = []
        G_pred_list = []
        for i in range(y_data.size(0)):
            adj_real = decode_adj(y_data[i].cpu().numpy(), y_max)
            adj_error = adj_raw-adj_real
            print(np.amin(adj_error),np.amax(adj_error))
            adj_pred = decode_adj(y_pred_data[i].cpu().numpy(), y_max)
            G_real = get_graph(adj_real)
            G_pred = get_graph(adj_pred)
            G_real_list.append(G_real)
            G_pred_list.append(G_pred)
        # save list of objects
        fname_pred = args.graph_save_path + args.note+ '_'+ args.graph_type + '_' + str(args.graph_node_num) + '_' +\
                     str(epoch) + '_pred_bptt_' + str(args.bptt)+'_'+str(args.num_layers)+'_dilation_'+str(args.is_dilation)+'_flexible_'+str(args.is_flexible)+'_bn_'+str(args.is_bn)+'_lr_'+str(args.lr)+'.dat'
        save_graph_list(G_pred_list,fname_pred)
        fname_real = args.graph_save_path + args.note + '_' + args.graph_type + '_' + str(args.graph_node_num) + '_' + \
                     str(epoch) + '_real_bptt_' + str(args.bptt)+'_'+str(args.num_layers)+'_dilation_'+str(args.is_dilation)+'_flexible_'+str(args.is_flexible)+'_bn_'+str(args.is_bn)+'_lr_'+str(args.lr)+'.dat'
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
        print('Epoch: {}/{}, train loss: {:.6f}, accuracy: {}/{} {:.4f}, real mean: {:.4f}, pred mean: {:.4f},layers:{}, bptt: {}, dilation: {}, bn:{}, lr:{}'.format(
                epoch, args.epochs, loss.data[0], correct, all, accuracy, real_score_mean, pred_score_mean,args.num_layers, args.bptt, args.is_dilation, args.is_bn, args.lr))
        logging.warning('Epoch: {}/{}, train loss: {:.6f}, accuracy: {}/{} {:.4f}, real mean: {:.4f}, pred mean: {:.4f},layers:{}, bptt: {}, dilation{}, bn:{}, lr:{}'.format(
            epoch, args.epochs, loss.data[0], correct, all, accuracy, real_score_mean, pred_score_mean,args.num_layers, args.bptt, args.is_dilation, args.is_bn, args.lr))
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



########### train function for LSTM
def train(args, dataset_train, generator):
    if args.load:
        epoch_load = args.load_epoch
        fname = args.model_save_path + args.note + '_' + args.graph_type + '_' + \
        str(epoch_load) + str(args.num_layers) + '_' + str(args.bptt)+'_'+str(args.bptt_len)+ '.dat'
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
                fname = args.model_save_path + args.note + '_' + args.graph_type + '_' + \
                        str(epoch) + str(args.num_layers) + '_' + str(args.bptt)+'_'+str(args.bptt_len)+ '.dat'
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
        thresh = train_epoch_AE(epoch, args,encoder, generator, dataset_train, optimizer, scheduler, thresh, train=True)
        if epoch % args.epochs_test == 0:
            train_epoch_AE(epoch, args,encoder, generator, dataset_train, optimizer, scheduler, thresh, train = False)
        if args.save:
            if epoch % args.epochs_save == 0:
                fname = args.model_save_path + args.note + '_' + args.graph_type + '_' + str(args.graph_node_num) + \
                        '_' + str(epoch) + str(args.sample_when_validate) + '_' + str(args.num_layers) + '_encoder.dat'
                torch.save(encoder.state_dict(), fname)
                fname = args.model_save_path + args.note + '_' + args.graph_type + '_' + str(args.graph_node_num) +\
                '_' + str(epoch) + str(args.sample_when_validate) +'_'+str(args.num_layers) + '_generator.dat'
                torch.save(generator.state_dict(), fname)
        epoch += 1


########### Train function for structure RNN, the flexible version is also included
def train_GraphRNN_structure(args, dataset_train, generator):
    if args.load:
        epoch_load = args.load_epoch
        fname = args.model_save_path + args.note + '_' + args.graph_type + '_' + str(args.graph_node_num) + \
                '_' + str(epoch_load) + str(args.bptt) + '_' + str(args.num_layers) + '.dat'
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
        if args.is_flexible:
            thresh = train_epoch_GraphRNN_structure_flexible(epoch, args, generator, dataset_train, optimizer, scheduler, thresh, train=True)
        else:
            thresh = train_epoch_GraphRNN_structure(epoch, args, generator, dataset_train, optimizer, scheduler, thresh, train=True)
        if epoch % args.epochs_test == 0:
            if args.is_flexible:
                train_epoch_GraphRNN_structure_flexible(epoch, args, generator, dataset_train, optimizer, scheduler, thresh, train=False)
            else:
                train_epoch_GraphRNN_structure(epoch, args, generator, dataset_train, optimizer, scheduler, thresh, train = False)
        if args.save:
            if epoch % args.epochs_save == 0:
                fname = args.model_save_path + args.note + '_' + args.graph_type + '_' + str(args.graph_node_num) +\
                '_' + str(epoch) + str(args.bptt) +'_'+str(args.num_layers) + '.dat'
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
    print('CUDA', CUDA)
    time = strftime("%Y-%m-%d %H:%M:%S", gmtime())
    logging.basicConfig(filename='logs/train' + time + '.log', level=logging.DEBUG)

    # if os.path.isdir("tensorboard/run"+time):
    #     shutil.rmtree("tensorboard/run"+time)
    configure("tensorboard/run"+time, flush_secs=5)

    args = Args()
    print(args.graph_type)
    print('bptt', args.bptt)
    # if using a single graph
    if args.graph_type=='star':
        G = nx.star_graph(args.graph_node_num)
        graphs = [G]
        max_num_nodes = G.number_of_nodes()
    if args.graph_type=='ladder':
        graphs = []
        for i in range(100, 201):
            graphs.append(nx.ladder_graph(i))
        max_num_nodes = 400
    if args.graph_type=='karate':
        G = nx.karate_club_graph()
        graphs = [G]
        max_num_nodes = G.number_of_nodes()
    if args.graph_type=='tree':
        graphs = []
        for i in range(2, 5):
            for j in range(2, 5):
                graphs.append(nx.balanced_tree(i, j))
        max_num_nodes = 256
    if args.graph_type=='caveman':
        graphs = []
        for i in range(10,21):
            for j in range(10,21):
                graphs.append(nx.connected_caveman_graph(i, j))
        max_num_nodes = 400
    if args.graph_type=='grid':
        graphs = []
        for i in range(10,21):
            for j in range(10,21):
                graphs.append(nx.grid_2d_graph(i,j))
        max_num_nodes = 20*20
    if args.graph_type=='barabasi':
        graphs = []
        for i in range(100,401):
            graphs.append(nx.barabasi_albert_graph(i,2))
        max_num_nodes = 400
    # if using a list of graphs
    if args.graph_type == 'enzymes':
        graphs, max_num_nodes = Graph_load_batch(min_num_nodes=6, name='ENZYMES')
        print('max num nodes', max_num_nodes)
    if args.graph_type == 'protein':
        graphs, max_num_nodes = Graph_load_batch(min_num_nodes=6, name='PROTEINS_full')
        print('max num nodes', max_num_nodes)
    if args.graph_type == 'DD':
        graphs, max_num_nodes = Graph_load_batch(min_num_nodes=6, max_num_nodes=1000, name='DD',node_attributes=False,graph_labels=True)
        print('max num nodes', max_num_nodes)



    ################## the GraphRNN model #####################
    ### 'Graph_sequence_sampler_rnn' is used for baseline model
    sampler = Graph_sequence_sampler_rnn(graphs, max_node_num=max_num_nodes,batch_size=args.batch_size, max_prev_node=args.max_prev_node)
    x, y, len = sampler.sample()
    ### 'Graph_sequence_sampler_fast' is used for the proposed model
    # sampler = Graph_sequence_sampler_fast(graphs, max_node_num=max_num_nodes,batch_size=args.batch_size, max_prev_node=args.max_prev_node)
    ### 'Graph_sequence_sampler_flexible' is used for the flexible version of the proposed model
    # sampler = Graph_sequence_sampler_flexible(graphs)

    ### Graph RNN structure model
    # if args.is_flexible:
    #     args.batch_size = 1
    # generator = Graph_RNN_structure(hidden_size=args.hidden_size, batch_size=args.batch_size, output_size=args.max_prev_node, num_layers=args.num_layers, is_dilation=args.is_dilation, is_bn=args.is_bn).cuda(CUDA)
    # train_GraphRNN_structure(args,sampler,generator)

    ### Graph RNN baseline model
    generator = Graph_generator_LSTM_graph(feature_size=x.size(2), input_size=args.input_size,
                                           hidden_size=args.hidden_size,
                                           output_size=y.size(2), batch_size=args.batch_size, num_layers=args.num_layers).cuda(CUDA)
    train(args,sampler,generator)



    ### auto encoder model
    # encoder = GCN_encoder_graph(feature.size(2),args.hidden_dim,args.input_size,args.num_layers).cuda(CUDA)
    # generator = Graph_generator_LSTM_graph(feature_size=x.size(2), input_size=args.input_size, hidden_size=args.hidden_size,
    #                                        output_size=y.size(2), batch_size=args.batch_size, num_layers=args.num_layers).cuda(CUDA)
    #
    # train_AE(args,sampler,encoder,generator)


    ### the GCN baseline model
    # sampler = Graph_sequence_sampler_bfs_permute_truncate_multigraph([G], max_node_num=G.number_of_nodes(), batch_size=args.batch_size
    #                                                                  , max_prev_node = args.max_prev_node, feature = True)
    # encoder = GCN_encoder(input_dim=G.number_of_nodes(),hidden_dim=args.hidden_dim,output_dim=args.output_dim)
    # decoder = GCN_decoder()
    #
    # train_GCN(args,sampler,encoder, decoder)