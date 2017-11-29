import networkx as nx0
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
        # self.graph_type = 'ladder'
        # self.graph_type = 'karate'
        # self.graph_type = 'tree'
        # self.graph_type = 'caveman'
        # self.graph_type = 'grid'
        # self.graph_type = 'barabasi'
        self.graph_type = 'enzymes'
        # self.graph_type = 'protein'
        # self.graph_type = 'DD'


        ## self.graph_node_num = 50 # obsolete

        # max previous node that looks back
        # self.max_prev_node = 150
        # self.max_prev_node = 100
        self.max_prev_node = 50 # ladder, protein
        # self.max_prev_node = 25 # enzyme


        ### network config
        ## GraphRNN
        self.input_size = 64
        self.hidden_size = 64
        self.embedding_size = 64
        self.noise_size = 64
        self.noise_level = 0.1
        # self.input_size = 128 # for DD dataset
        # self.hidden_size = 128 # for DD dataset
        self.batch_size = 128
        self.num_layers = 4
        self.is_dilation = True
        self.is_flexible = False # if use flexible input size
        self.is_bn = True
        self.bptt = False # if use truncate back propagation (not very stable)
        self.bptt_len = 20
        self.gumbel = False
        self.has_noise = True
        ## GCN
        self.output_dim = 64
        self.hidden_dim = 64

        ### training config
        self.lr = 0.003
        self.lr_gan = 0.001
        self.epochs = 50000
        self.epochs_gan = 50001 # which means not using GAN
        # self.epochs_gan = 4000 # use GAN from epoch 4000
        self.epochs_test = 500
        self.epochs_log = 500
        self.epochs_save = 500
        self.milestones = [4000, 8000, 16000]
        # self.milestones = [16000, 32000]

        self.lr_rate = 0.3
        self.sample_when_validate = True
        self.sample_time = 1
        # self.sample_when_validate = False

        ### output config
        self.model_save_path = 'model_save_new/'
        # self.graph_save_path = 'graphs/'
        self.graph_save_path = 'graphs_new/'
        self.figure_save_path = 'figures/'
        self.load = False
        # self.load_epoch = 50000
        self.load_epoch = 16000

        self.save = False
        # self.note = 'GraphRNN'
        # self.note = 'GraphRNN_VAE'
        # self.note = 'GraphRNN_VAE_nobn'
        self.note = 'GraphRNN_VAE_simple'
        # self.note = 'GraphRNN_VAE_simple_newdecoder'


        # self.note = 'GraphRNN_GAN'
        # self.note = 'GraphRNN_AE'
        # self.note = 'GraphRNN_structure'
        # self.note = 'GCN'


        # self.clean_tensorboard = True
        self.clean_tensorboard = False



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
    optimizer.zero_grad()
    generator.hidden = generator.init_hidden()

    x,y,y_len = dataset.sample()

    y_len_max = max(y_len)
    x = x[:,0:y_len_max,:]
    y = y[:,0:y_len_max,:]

    x = Variable(x).cuda(CUDA)
    y = Variable(y).cuda(CUDA)

    y_pred = Variable(torch.ones(y.size(0), y.size(1), y.size(2))*-100).cuda(CUDA)

    # if train
    if train:
        # if do truncate backprop
        # todo: finish a memory efficient bptt
        if args.bptt:
            start_id = 0
            while start_id<x.size(1):
                print('start id',start_id)
                end_id = min(start_id+args.bptt_len,x.size(1))
                y_pred_temp = generator(x[:,start_id:end_id,:])
                generator.hidden = detach_hidden_lstm(generator.hidden)
                # generator.hidden[0].detach()
                # generator.hidden[1].detach()

                y_pred[:,start_id:end_id,:] = y_pred_temp
                start_id += args.bptt_len
            y_pred_clean = Variable(torch.ones(x.size(0), x.size(1), x.size(2))*-100).cuda(CUDA)
            # before computing loss, cleaning y_pred so that only valid entries are supervised
            y_pred = pack_padded_sequence(y_pred, y_len, batch_first=True)
            y_pred = pad_packed_sequence(y_pred, batch_first=True)[0]
            y_pred_clean[:, 0:y_pred.size(1), :] = y_pred
            y_pred = y_pred_clean


        # if backprop through the start
        else:
            y_pred_temp = generator(x,pack = True,len=y_len)
            # before computing loss, cleaning y_pred so that only valid entries are supervised
            y_pred_temp = pack_padded_sequence(y_pred_temp, y_len, batch_first=True)
            y_pred_temp = pad_packed_sequence(y_pred_temp, batch_first=True)[0]
            y_pred[:, 0:y_pred_temp.size(1), :] = y_pred_temp

            loss = F.binary_cross_entropy_with_logits(y_pred, y)
            loss.backward()
            optimizer.step()
            scheduler.step()


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

    # log_value('train_loss', loss.data[0], epoch)
    # log_value('accuracy', accuracy, epoch)
    # log_value('auc_mean', auc_mean, epoch)
    # log_value('ap_mean', ap_mean, epoch)
    return thresh


def train_gan_epoch(epoch, args, lstm, output_deterministic, output_generator, output_discriminator, dataset,
                    optimizer_lstm, optimizer_deterministic, optimizer_generator, optimizer_discriminator,
                    scheduler_lstm, scheduler_deterministic, scheduler_generator, scheduler_discriminator,
                    temperature=0.5, train=True, gan=False):
    lstm.zero_grad()
    lstm.hidden = lstm.init_hidden()

    x, y, y_len = dataset.sample()

    y_len_max = max(y_len)
    x = x[:, 0:y_len_max, :]
    y = y[:, 0:y_len_max, :]

    x = Variable(x).cuda(CUDA)
    y = Variable(y).cuda(CUDA)

    # if train
    if train:
        # if do truncate backprop
        # todo: finish a memory efficient bptt
        if args.bptt:
            pass
            # start_id = 0
            # while start_id < x.size(1):
            #     print('start id', start_id)
            #     end_id = min(start_id + args.bptt_len, x.size(1))
            #     y_pred_temp = generator(x[:, start_id:end_id, :])
            #     generator.hidden = detach_hidden_lstm(generator.hidden)
            #     # generator.hidden[0].detach()
            #     # generator.hidden[1].detach()
            #
            #     y_pred[:, start_id:end_id, :] = y_pred_temp
            #     start_id += args.bptt_len
            # y_pred_clean = Variable(torch.ones(x.size(0), x.size(1), x.size(2)) * -100).cuda(CUDA)
            # # before computing loss, cleaning y_pred so that only valid entries are supervised
            # y_pred = pack_padded_sequence(y_pred, y_len, batch_first=True)
            # y_pred = pad_packed_sequence(y_pred, batch_first=True)[0]
            # y_pred_clean[:, 0:y_pred.size(1), :] = y_pred
            # y_pred = y_pred_clean

        # if backprop through the start
        else:
            if gan:
                # if using model's own prediction each time step to train
                if args.gumbel:
                    y_pred = Variable(torch.zeros(y.size(0), y.size(1), y.size(2))).cuda(CUDA)  # normalized score
                    x_step = x[:, 0:1, :]
                    for i in range(x.size(1)):
                        h = lstm(x_step)
                        # 1A: train D on real data
                        output_discriminator.zero_grad()
                        # now we assume that discriminator cannot tune lstm or generator to make decision
                        # todo: try h.detach() vs h (tune lstm vs not)
                        l_real = output_discriminator(h.detach(), y)
                        # clean
                        l_real = pack_padded_sequence(l_real, y_len, batch_first=True)
                        l_real = pad_packed_sequence(l_real, batch_first=True)[0]
                        loss_real_d = F.binary_cross_entropy(l_real, Variable(
                            torch.ones(l_real.size(0), l_real.size(1), l_real.size(2)) * 0.9).cuda(CUDA))
                        loss_real_d.backward()

                        # 1B: train D on fake data
                        # uniform noise
                        n = Variable(torch.rand(h.size(0), h.size(1), args.noise_size)).cuda(CUDA)
                        # generator
                        y_fake = output_generator(h, n, temperature)
                        # discriminator
                        l_fake = output_discriminator(h.detach(), y_fake.detach())
                        # clean
                        l_fake = pack_padded_sequence(l_fake, y_len, batch_first=True)
                        l_fake = pad_packed_sequence(l_fake, batch_first=True)[0]
                        loss_fake_d = F.binary_cross_entropy(l_fake, Variable(
                            torch.zeros(l_real.size(0), l_real.size(1), l_real.size(2))).cuda(CUDA))
                        loss_fake_d.backward()

                        # 1C: update D
                        optimizer_discriminator.step()
                        scheduler_discriminator.step()

                        # 2A: Train G on D's response, do not tune D's parameter, but tune lstm's paremeter
                        output_generator.zero_grad()
                        n = Variable(torch.rand(h.size(0), h.size(1), args.noise_size)).cuda(CUDA)
                        y_fake = output_generator(h, n, temperature)
                        # todo: try h.detach() vs h (tune lstm vs not)
                        l_fake = output_discriminator(h, y_fake)
                        # clean
                        l_fake = pack_padded_sequence(l_fake, y_len, batch_first=True)
                        l_fake = pad_packed_sequence(l_fake, batch_first=True)[0]
                        loss_fake_g = F.binary_cross_entropy(l_fake, Variable(
                            torch.ones(l_real.size(0), l_real.size(1), l_real.size(2)) * 0.9).cuda(CUDA))
                        loss_fake_g.backward()

                        # 2B: update G and lstm
                        optimizer_generator.step()
                        scheduler_generator.step()
                        optimizer_lstm.step()
                        scheduler_lstm.step()
                    pass
                # if using ground truth to train
                else:
                    h = lstm(x, pack=True, input_len=y_len)
                    # 1A: train D on real data
                    output_discriminator.zero_grad()
                    # now we assume that discriminator cannot tune lstm or generator to make decision
                    # todo: try h.detach() vs h (tune lstm vs not)
                    l_real = output_discriminator(h.detach(),y)
                    # clean
                    l_real = pack_padded_sequence(l_real, y_len, batch_first=True)
                    l_real = pad_packed_sequence(l_real, batch_first=True)[0]
                    loss_real_d = F.binary_cross_entropy(l_real, Variable(torch.ones(l_real.size(0),l_real.size(1),l_real.size(2))*0.9).cuda(CUDA))
                    loss_real_d.backward()

                    # 1B: train D on fake data
                    # uniform noise
                    n = Variable(torch.rand(h.size(0), h.size(1), args.noise_size)).cuda(CUDA)
                    # generator
                    y_fake = output_generator(h,n,temperature)
                    # discriminator
                    l_fake = output_discriminator(h.detach(),y_fake.detach())
                    # clean
                    l_fake = pack_padded_sequence(l_fake, y_len, batch_first=True)
                    l_fake = pad_packed_sequence(l_fake, batch_first=True)[0]
                    loss_fake_d = F.binary_cross_entropy(l_fake, Variable(torch.zeros(l_real.size(0), l_real.size(1),l_real.size(2))).cuda(CUDA))
                    loss_fake_d.backward()

                    # 1C: update D
                    optimizer_discriminator.step()
                    scheduler_discriminator.step()

                    # 2A: Train G on D's response, do not tune D's parameter, but tune lstm's paremeter
                    output_generator.zero_grad()
                    n = Variable(torch.rand(h.size(0), h.size(1), args.noise_size)).cuda(CUDA)
                    y_fake = output_generator(h, n, temperature)
                    # todo: try h.detach() vs h (tune lstm vs not)
                    l_fake = output_discriminator(h,y_fake)
                    # clean
                    l_fake = pack_padded_sequence(l_fake, y_len, batch_first=True)
                    l_fake = pad_packed_sequence(l_fake, batch_first=True)[0]
                    loss_fake_g = F.binary_cross_entropy(l_fake, Variable(torch.ones(l_real.size(0),l_real.size(1),l_real.size(2))*0.9).cuda(CUDA))
                    loss_fake_g.backward()

                    # 2B: update G and lstm
                    optimizer_generator.step()
                    scheduler_generator.step()
                    optimizer_lstm.step()
                    scheduler_lstm.step()
                    pass
            else:
                # if using model's own prediction each time step to train
                if args.gumbel:
                    output_deterministic.zero_grad()
                    y_pred = Variable(torch.zeros(y.size(0), y.size(1), y.size(2))).cuda(CUDA) # normalized score
                    x_step = x[:, 0:1, :]
                    for i in range(x.size(1)):
                        h = lstm(x_step)
                        y_pred_step = output_deterministic(h,sigmoid=False)
                        y_pred[:, i:i + 1, :] = F.sigmoid(y_pred_step) # write down score
                        x_step = gumbel_sigmoid(y_pred_step, temperature=temperature)
                    # clean
                    y_pred = pack_padded_sequence(y_pred, y_len, batch_first=True)
                    y_pred = pad_packed_sequence(y_pred, batch_first=True)[0]

                    loss_determinstic = F.binary_cross_entropy(y_pred, y)
                    loss_determinstic.backward()
                    # update deterministic and lstm
                    optimizer_deterministic.step()
                    optimizer_lstm.step()
                    scheduler_deterministic.step()
                    scheduler_lstm.step()
                # if using ground truth to train
                else:
                    h = lstm(x, pack=True, input_len=y_len)
                    output_deterministic.zero_grad()
                    y_pred = output_deterministic(h)
                    # clean
                    y_pred = pack_padded_sequence(y_pred, y_len, batch_first=True)
                    y_pred = pad_packed_sequence(y_pred, batch_first=True)[0]
                    # use cross entropy loss
                    loss_determinstic = F.binary_cross_entropy(y_pred, y)
                    loss_determinstic.backward()
                    # update deterministic and lstm
                    optimizer_deterministic.step()
                    optimizer_lstm.step()
                    scheduler_deterministic.step()
                    scheduler_lstm.step()


    # if validate, do sampling/threshold each step
    else:
        y_pred_long = Variable(torch.zeros(y.size(0), y.size(1), y.size(2))).cuda(CUDA)
        x_step = x[:, 0:1, :]
        for i in range(x.size(1)):
            h = lstm(x_step)
            if gan:
                n = Variable(torch.rand(h.size(0), h.size(1), args.noise_size)).cuda(CUDA)
                x_step = output_generator(h,n,temperature)
                x_step = sample_y(x_step, sample=False, thresh=0.5) # get threshold prediction
            else:
                y_pred_step = output_deterministic(h)
                if args.has_noise: # if output function has noise, then no need to do additional sampling
                    x_step = sample_y(y_pred_step, sample=False, thresh=0.5)
                else:
                    x_step = sample_y(y_pred_step, sample=True)
            y_pred_long[:, i:i + 1, :] = x_step
        loss_determinstic = F.binary_cross_entropy(y_pred_long, y)
        y_pred_long = y_pred_long.long()

        y_data = y.data
        y_pred_data = y_pred_long.data

        real_score_mean = y_data.mean()
        pred_score_mean = y_pred_data.float().mean()

    # plot graph
    if epoch % args.epochs_log == 0 and train == False:
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
        fname_pred = args.graph_save_path + args.note + '_' + args.graph_type + '_' + \
                     str(epoch) + '_pred_' + str(args.num_layers) + '_' + str(args.bptt) + '_' + str(
            args.bptt_len) + '_' + str(args.gumbel) + '_' + str(args.has_noise)+ '_' + str(args.noise_level) + '.dat'
        save_graph_list(G_pred_list, fname_pred)
        fname_real = args.graph_save_path + args.note + '_' + args.graph_type + '_' + \
                     str(epoch) + '_real_' + str(args.num_layers) + '_' + str(args.bptt) + '_' + str(
            args.bptt_len) + '_' + str(args.gumbel) + '_' + str(args.has_noise)+ '_' + str(args.noise_level) + '.dat'
        save_graph_list(G_real_list, fname_real)


    if epoch % args.epochs_log == 0 and train == True:
        if gan:
            print('Epoch: {}/{}, train loss_real_d: {:.6f}, train loss_fake_d: {:.6f}, train loss_fake_g: {:.6f}, graph type: {}, num_layer: {}, bptt: {}, bptt_len:{}, gumbel:{}, temperature:{}, has_noise:{}, noise_level:{}'.format(
                epoch, args.epochs, loss_real_d.data[0], loss_fake_d.data[0], loss_fake_g.data[0], args.graph_type, args.num_layers, args.bptt, args.bptt_len, args.gumbel, temperature, args.has_noise, args.noise_level))
        else:
            print('Epoch: {}/{}, train loss: {:.6f}, graph type: {}, num_layer: {}, bptt: {}, bptt_len:{}, gumbel:{}, temperature:{}, has_noise:{}, noise_level:{}'.format(
                epoch, args.epochs, loss_determinstic.data[0], args.graph_type, args.num_layers, args.bptt, args.bptt_len, args.gumbel, temperature, args.has_noise, args.noise_level))
    elif epoch % args.epochs_log == 0 and train == False:
        print('Epoch: {}/{}, test loss: {:.6f}, real mean: {:.4f}, pred mean: {:.4f}'.format(
            epoch, args.epochs, loss_determinstic.data[0], real_score_mean, pred_score_mean))

    log_value('train_loss_'+ args.note + '_' + args.graph_type + '_' + str(args.num_layers) + '_' + str(args.bptt) + '_' + str(
            args.bptt_len) + '_' + str(args.gumbel)+ '_' + str(args.noise_level), loss_determinstic.data[0], epoch)
    # log_value('temperature_' + args.note + '_' + args.graph_type + '_' + str(args.num_layers) + '_' + str(
    #     args.bptt) + '_' + str(args.bptt_len) + '_' + str(args.gumbel), temperature, epoch)


def train_vae_epoch(epoch, args, lstm, output_vae, dataset,
                    optimizer_lstm, optimizer_vae,
                    scheduler_lstm, scheduler_vae, train=True):
    lstm.zero_grad()
    output_vae.zero_grad()
    lstm.hidden = lstm.init_hidden()

    x, y, y_len = dataset.sample()

    y_len_max = max(y_len)
    x = x[:, 0:y_len_max, :]
    y = y[:, 0:y_len_max, :]

    x = Variable(x).cuda(CUDA)
    y = Variable(y).cuda(CUDA)

    # if train
    if train:
        # if do truncate backprop
        # todo: finish a memory efficient bptt
        if args.bptt:
            pass
            # start_id = 0
            # while start_id < x.size(1):
            #     print('start id', start_id)
            #     end_id = min(start_id + args.bptt_len, x.size(1))
            #     y_pred_temp = generator(x[:, start_id:end_id, :])
            #     generator.hidden = detach_hidden_lstm(generator.hidden)
            #     # generator.hidden[0].detach()
            #     # generator.hidden[1].detach()
            #
            #     y_pred[:, start_id:end_id, :] = y_pred_temp
            #     start_id += args.bptt_len
            # y_pred_clean = Variable(torch.ones(x.size(0), x.size(1), x.size(2)) * -100).cuda(CUDA)
            # # before computing loss, cleaning y_pred so that only valid entries are supervised
            # y_pred = pack_padded_sequence(y_pred, y_len, batch_first=True)
            # y_pred = pad_packed_sequence(y_pred, batch_first=True)[0]
            # y_pred_clean[:, 0:y_pred.size(1), :] = y_pred
            # y_pred = y_pred_clean

        # if backprop through the start
        else:
            # if using model's own prediction each time step to train
            if args.gumbel:
                y_pred = Variable(torch.zeros(y.size(0), y.size(1), y.size(2))).cuda(CUDA) # normalized score
                x_step = x[:, 0:1, :]
                for i in range(x.size(1)):
                    h = lstm(x_step)
                    y_pred_step = output_vae(h,sigmoid=False)
                    y_pred[:, i:i + 1, :] = F.sigmoid(y_pred_step) # write down score
                    x_step = gumbel_sigmoid(y_pred_step, temperature=0.5)
                # clean
                y_pred = pack_padded_sequence(y_pred, y_len, batch_first=True)
                y_pred = pad_packed_sequence(y_pred, batch_first=True)[0]

                loss_determinstic = F.binary_cross_entropy(y_pred, y)
                loss_determinstic.backward()
                # update deterministic and lstm
                optimizer_vae.step()
                optimizer_lstm.step()
                scheduler_vae.step()
                scheduler_lstm.step()
            # if using ground truth to train
            else:
                h = lstm(x, pack=True, input_len=y_len)
                y_pred,z_mu,z_lsgms = output_vae(h)
                # clean
                y_pred = pack_padded_sequence(y_pred, y_len, batch_first=True)
                y_pred = pad_packed_sequence(y_pred, batch_first=True)[0]
                z_mu = pack_padded_sequence(z_mu, y_len, batch_first=True)
                z_mu = pad_packed_sequence(z_mu, batch_first=True)[0]
                z_lsgms = pack_padded_sequence(z_lsgms, y_len, batch_first=True)
                z_lsgms = pad_packed_sequence(z_lsgms, batch_first=True)[0]
                # use cross entropy loss
                loss_bce = F.binary_cross_entropy(y_pred, y)
                loss_kl = -0.5 * torch.sum(1 + z_lsgms - z_mu.pow(2) - z_lsgms.exp())
                loss_kl /= y.size(0)*y.size(1)*sum(y_len) # normalize
                loss = loss_bce + loss_kl
                loss.backward()
                # update deterministic and lstm
                optimizer_vae.step()
                optimizer_lstm.step()
                scheduler_vae.step()
                scheduler_lstm.step()

                z_mu_mean = torch.mean(z_mu.data)
                z_sgm_mean = torch.mean(z_lsgms.mul(0.5).exp_().data)
                z_mu_min = torch.min(z_mu.data)
                z_sgm_min = torch.min(z_lsgms.mul(0.5).exp_().data)
                z_mu_max = torch.max(z_mu.data)
                z_sgm_max = torch.max(z_lsgms.mul(0.5).exp_().data)


    # if validate, do sampling/threshold each step
    else:
        y_pred = Variable(torch.zeros(y.size(0), y.size(1), y.size(2))).cuda(CUDA)
        y_pred_long = Variable(torch.zeros(y.size(0), y.size(1), y.size(2))).cuda(CUDA)
        x_step = x[:, 0:1, :]
        for i in range(x.size(1)):
            h = lstm(x_step)
            y_pred_step,_,_ = output_vae(h)
            y_pred[:, i:i + 1, :] = y_pred_step
            x_step = sample_y(y_pred_step, sample=True)
            y_pred_long[:, i:i + 1, :] = x_step
        loss = F.binary_cross_entropy(y_pred_long, y)
        y_pred_long = y_pred_long.long()

        y_data = y.data
        y_pred_data = y_pred_long.data

        real_score_mean = y_data.mean()
        pred_score_mean = y_pred.data[y_pred.data>1e-6].mean()
        real_score_max = y_data.max()
        pred_score_max = y_pred.data[y_pred.data>1e-6].max()
        real_score_min = y_data.min()
        pred_score_min = y_pred.data[y_pred.data>1e-6].min()

    # plot graph
    if epoch % args.epochs_log == 0 and train == False:
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
        fname_pred = args.graph_save_path + args.note + '_' + args.graph_type + '_' + \
                     str(epoch) + '_pred_' + str(args.num_layers) + '_' + str(args.bptt) + '_' + str(
            args.bptt_len) + '_' + str(args.gumbel)  + '.dat'
        save_graph_list(G_pred_list, fname_pred)
        fname_real = args.graph_save_path + args.note + '_' + args.graph_type + '_' + \
                     str(epoch) + '_real_' + str(args.num_layers) + '_' + str(args.bptt) + '_' + str(
            args.bptt_len) + '_' + str(args.gumbel) + '.dat'
        save_graph_list(G_real_list, fname_real)


    if epoch % args.epochs_log == 0 and train == True:
        print('Epoch: {}/{}, train bce loss: {:.6f}, train kl loss: {:.6f}, graph type: {}, num_layer: {}, bptt: {}, bptt_len:{}, gumbel:{}'.format(
            epoch, args.epochs,loss_bce.data[0], loss_kl.data[0], args.graph_type, args.num_layers, args.bptt, args.bptt_len, args.gumbel))
        print('z_mu_mean', z_mu_mean, 'z_mu_min', z_mu_min, 'z_mu_max', z_mu_max, 'z_sgm_mean', z_sgm_mean, 'z_sgm_min', z_sgm_min, 'z_sgm_max', z_sgm_max)
    elif epoch % args.epochs_log == 0 and train == False:
        print('Epoch: {}/{}, test loss: {:.6f}, real mean: {:.4f}, pred mean: {:.4f}, real min: {:.4f}, pred min: {:.4f}, real max: {:.4f}, pred max: {:.4f}'.format(
            epoch, args.epochs, loss.data[0], real_score_mean, pred_score_mean, real_score_min, pred_score_min, real_score_max, pred_score_max))

    if train:
        log_value('train_bce_loss_' + args.note + '_' + args.graph_type + '_' + str(args.num_layers) + '_' + str(
            args.bptt) + '_' + str(args.bptt_len) + '_' + str(args.gumbel) + '_' + str(args.noise_level), loss_bce.data[0],
                  epoch)
        log_value('train_kl_loss_' + args.note + '_' + args.graph_type + '_' + str(args.num_layers) + '_' + str(
            args.bptt) + '_' + str(args.bptt_len) + '_' + str(args.gumbel) + '_' + str(args.noise_level), loss_kl.data[0],
                  epoch)

    # log_value('temperature_' + args.note + '_' + args.graph_type + '_' + str(args.num_layers) + '_' + str(
    #     args.bptt) + '_' + str(args.bptt_len) + '_' + str(args.gumbel), temperature, epoch)


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
    y = dataset.sample()
    # y_var = Variable(y).cuda(CUDA)

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
        # fname_pred = args.graph_save_path + args.note+ '_'+ args.graph_type + '_' + str(args.graph_node_num) + '_' +\
        #              str(epoch) + '_pred_bptt_' + str(args.bptt)+'_'+str(args.num_layers)+'_dilation_'+str(args.is_dilation)+'_flexible_'+str(args.is_flexible)+'_bn_'+str(args.is_bn)+'_lr_'+str(args.lr)+'.dat'
        # save_graph_list(G_pred_list,fname_pred)
        # fname_real = args.graph_save_path + args.note + '_' + args.graph_type + '_' + str(args.graph_node_num) + '_' + \
        #              str(epoch) + '_real_bptt_' + str(args.bptt)+'_'+str(args.num_layers)+'_dilation_'+str(args.is_dilation)+'_flexible_'+str(args.is_flexible)+'_bn_'+str(args.is_bn)+'_lr_'+str(args.lr)+'.dat'
        # save_graph_list(G_real_list, fname_real)

        fname_pred = args.graph_save_path + args.note + '_' + args.graph_type + '_' + \
                     str(epoch) + '_pred_' + str(args.num_layers) + '_' + str(args.bptt) + '.dat'
        save_graph_list(G_pred_list, fname_pred)
        fname_real = args.graph_save_path + args.note + '_' + args.graph_type + '_' + \
                     str(epoch) + '_real_' + str(args.num_layers) + '_' + str(args.bptt) + '.dat'
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




########## The proposed structure RNN model, bptt version (memory efficient)
def train_epoch_GraphRNN_structure_bptt(epoch, args, generator, dataset, optimizer, scheduler, thresh, train=True):
    generator.train()
    optimizer.zero_grad()
    # initialize hidden list

    generator.hidden_all = generator.init_hidden(len=args.max_prev_node)
    # print(len(generator.hidden_all))
    _,y,y_len = dataset.sample()
    # # get a clean version of y
    y_len_max = max(y_len)
    # print('y_len_max',y_len_max)
    y = y[:,0:y_len_max,:]



    # is_teacher_forcing = True if random.random()<0.8 else False
    is_teacher_forcing = True
    if train:
        start_id = 0
        while start_id < y.size(1):
            optimizer.zero_grad()
            end_id = min(start_id + args.bptt_len, y.size(1))
            # print(start_id,end_id)
            y_pred = Variable(torch.zeros(y.size(0), end_id-start_id, y.size(2))).cuda(CUDA)
            y_var = Variable(y[:,start_id:end_id,:]).cuda(CUDA)
            for i in range(0,end_id-start_id):
                if is_teacher_forcing:
                    y_step_pred, _ = generator(y_var[:, i:i + 1, :], teacher_forcing=True, temperature=0.1, bptt=args.bptt,
                                               flexible=args.is_flexible)
                else:
                    y_step_pred, _ = generator(None, teacher_forcing=False, temperature=0.1, bptt=args.bptt,
                                               flexible=args.is_flexible)
                y_pred[:, i:i+1, :] = y_step_pred

            # for i in range(len(generator.hidden_all)):
            #     generator.hidden_all[i].detach()

            # should clean y_pred, so that only meaningful entries will have loss
            y_len_bptt = [min(item-start_id, args.bptt_len) for item in y_len] # could have len<0
            # 1 roughly clean all items (since packing need all items have len>0)
            y_len_bptt_rough = [max(1,item) for item in y_len_bptt]
            y_pred = pack_padded_sequence(y_pred, y_len_bptt_rough, batch_first=True)
            y_pred = pad_packed_sequence(y_pred, batch_first=True)[0]
            # 2 remove items that have len<=0
            y_len_mask = Variable(torch.squeeze(torch.nonzero(torch.LongTensor(y_len_bptt)<=0))).cuda(CUDA)
            # print(y_len_mask)
            if len(y_len_mask.size())>0: # if there is items that has negative length
                y_pred = y_pred.clone()
                y_pred.index_fill_(0,y_len_mask,-100)

            # calc loss
            loss = F.binary_cross_entropy_with_logits(y_pred, y_var)
            loss.backward()
            optimizer.step()
            generator.hidden_all = [Variable(generator.hidden_all[i].data).cuda(CUDA) for i in range(len(generator.hidden_all))]


            start_id += args.bptt_len
        scheduler.step()



    else:
        generator.eval()
        # store the output
        y_pred_data = torch.zeros(y.size(0), y.size(1), y.size(2)) # in long format (discrete)

        for i in range(y.size(1)):
            generator.zero_grad()
            y_step_pred, y_step_pred_sample = generator(None, teacher_forcing=False, temperature=0.1,bptt=args.bptt,flexible=args.is_flexible)
            # print(y_step_pred_sample.data)
            # y_step_pred_sample = sample_y(y_step_pred_sample, sample=False, thresh=0.5) # do threshold
            y_pred_data[:, i:i+1, :] = y_step_pred_sample.data
            generator.hidden_all = [Variable(generator.hidden_all[i].data).cuda(CUDA) for i in range(len(generator.hidden_all))]

        y_pred_data = y_pred_data.long()
        print('pred_mean', torch.mean(y_pred_data.float()),'real_mean', torch.mean(y.float()))
        # loss = F.binary_cross_entropy_with_logits(Variable(y_pred).cuda(CUDA), Variable(y).cuda(CUDA))
    #
    # y_data = y
    # y_pred_data = F.sigmoid(y_pred).data

    #
    # y_data_flat = y_data.view(-1).cpu().numpy()
    # y_pred_data_flat = y_pred_data.view(-1).cpu().numpy()



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

    # real_score_mean = y_data.mean()
    # pred_score_mean = y_pred_data.mean()

    # # calc accuracy
    # # thresh = 0.03
    # y_pred_data[y_pred_data>thresh] = 1
    # y_pred_data[y_pred_data<=thresh] = 0
    # y_data = y_data.long()
    # y_pred_data = y_pred_data.long()
    # if train==False:
    #     y_pred_data = y_pred_long.data
    #     # print(y_pred)
    #     # print(y_pred_long)

    # correct = torch.eq(y_pred_data, y_data).long().sum()
    # all = y_pred_data.size(0)*y_pred_data.size(1)*y_pred_data.size(2)

    # plot graph
    if epoch % args.epochs_log == 0 and train==False:
        # save graphs as pickle
        G_real_list = []
        G_pred_list = []
        for i in range(y.size(0)):
            adj_real = decode_adj(y[i].cpu().numpy(), args.max_prev_node)
            adj_pred = decode_adj(y_pred_data[i].cpu().numpy(), args.max_prev_node)
            G_real = get_graph(adj_real)
            G_pred = get_graph(adj_pred)
            G_real_list.append(G_real)
            G_pred_list.append(G_pred)
        # save list of objects
        # fname_pred = args.graph_save_path + args.note+ '_'+ args.graph_type + '_' + str(args.graph_node_num) + '_' +\
        #              str(epoch) + '_pred_bptt_' + str(args.bptt)+'_'+str(args.num_layers)+'_dilation_'+str(args.is_dilation)+'_flexible_'+str(args.is_flexible)+'_bn_'+str(args.is_bn)+'_lr_'+str(args.lr)+'.dat'
        # save_graph_list(G_pred_list,fname_pred)
        # fname_real = args.graph_save_path + args.note + '_' + args.graph_type + '_' + str(args.graph_node_num) + '_' + \
        #              str(epoch) + '_real_bptt_' + str(args.bptt)+'_'+str(args.num_layers)+'_dilation_'+str(args.is_dilation)+'_flexible_'+str(args.is_flexible)+'_bn_'+str(args.is_bn)+'_lr_'+str(args.lr)+'.dat'
        # save_graph_list(G_real_list, fname_real)

        fname_pred = args.graph_save_path + args.note + '_' + args.graph_type + '_' + \
                     str(epoch) + '_pred_' + str(args.num_layers) + '_' + str(args.bptt) + '_' + str(args.bptt_len) + '_' + str(args.hidden_size) + '.dat'
        save_graph_list(G_pred_list, fname_pred)
        fname_real = args.graph_save_path + args.note + '_' + args.graph_type + '_' + \
                     str(epoch) + '_real_' + str(args.num_layers) + '_' + str(args.bptt) + '_' + str(args.bptt_len) + '_' + str(args.hidden_size) + '.dat'
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

    # accuracy = correct/float(all)
    if epoch%args.epochs_log==0 and train==True:
        print('Epoch: {}/{}, train loss: {:.6f}, layers:{}, bptt: {}, dilation: {}, bn:{}, lr:{}'.format(
                epoch, args.epochs, loss.data[0], args.num_layers, args.bptt, args.is_dilation, args.is_bn, args.lr))
        # logging.warning('Epoch: {}/{}, train loss: {:.6f}, accuracy: {}/{} {:.4f}, real mean: {:.4f}, pred mean: {:.4f},layers:{}, bptt: {}, dilation{}, bn:{}, lr:{}'.format(
        #     epoch, args.epochs, loss.data[0], correct, all, accuracy, real_score_mean, pred_score_mean,args.num_layers, args.bptt, args.is_dilation, args.is_bn, args.lr))
    # elif epoch%args.epochs_log==0 and train==False:
    #     print('Epoch: {}/{}, test loss: {:.6f}, real mean: {:.4f}, pred mean: {:.4f}'.format(
    #             epoch, args.epochs, loss.data[0], real_score_mean, pred_score_mean))
        # logging.warning('Epoch: {}/{}, test loss: {:.6f}, accuracy: {}/{} {:.4f}, real mean: {:.4f}, pred mean: {:.4f}'.format(
        #     epoch, args.epochs, loss.data[0], correct, all, accuracy, real_score_mean, pred_score_mean))

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

    # torch.manual_seed(args.seed)
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


########### train function for LSTM + GAN
def train_gan(args, dataset_train, lstm, output_deterministic, output_generator, output_discriminator):
    # todo: load new models
    # if args.load:
    #     epoch_load = args.load_epoch
    #     fname = args.model_save_path + args.note + '_' + args.graph_type + '_' + \
    #     str(epoch_load) + str(args.num_layers) + '_' + str(args.bptt)+'_'+str(args.bptt_len)+ '.dat'
    #     generator.load_state_dict(torch.load(fname))
    #     args.lr = 0.00001
    #     epoch = epoch_load
    #     print('model loaded!')
    # else:
    #     epoch = 0

    # torch.manual_seed(args.seed)
    optimizer_lstm = optim.Adam(list(lstm.parameters()), lr=args.lr)
    optimizer_deterministic = optim.Adam(list(output_deterministic.parameters()), lr=args.lr)
    optimizer_generator = optim.Adam(list(output_generator.parameters()), lr=args.lr_gan)
    optimizer_discriminator = optim.Adam(list(output_discriminator.parameters()), lr=args.lr_gan)

    scheduler_lstm = MultiStepLR(optimizer_lstm, milestones=args.milestones, gamma=args.lr_rate)
    scheduler_deterministic = MultiStepLR(optimizer_deterministic, milestones=args.milestones, gamma=args.lr_rate)
    scheduler_generator = MultiStepLR(optimizer_generator, milestones=args.milestones, gamma=args.lr_rate)
    scheduler_discriminator = MultiStepLR(optimizer_discriminator, milestones=args.milestones, gamma=args.lr_rate)

    epoch = 0
    gan = False
    while epoch<=args.epochs:
        # start using GAN loss to train
        if epoch==args.epochs_gan:
            gan = True
            optimizer_lstm = optim.Adam(list(lstm.parameters()), lr=args.lr/100) # almost freeze lstm
            scheduler_lstm = MultiStepLR(optimizer_lstm, milestones=[50000], gamma=args.lr_rate) # and don't change lr afterwards

        # train
        if gan:
            temperature = np.exp((-1e-4) * (epoch-args.epochs_gan))
        else:
            temperature = np.exp((-1e-4) * (epoch))

        # print(temperature)
        train_gan_epoch(epoch, args, lstm, output_deterministic, output_generator, output_discriminator, dataset_train,
                        optimizer_lstm, optimizer_deterministic, optimizer_generator, optimizer_discriminator,
                        scheduler_lstm, scheduler_deterministic, scheduler_generator, scheduler_discriminator,
                        temperature, train=True, gan=gan)
        # test
        if epoch % args.epochs_test == 0:
            train_gan_epoch(epoch, args, lstm, output_deterministic, output_generator, output_discriminator,
                            dataset_train,
                            optimizer_lstm, optimizer_deterministic, optimizer_generator, optimizer_discriminator,
                            scheduler_lstm, scheduler_deterministic, scheduler_generator, scheduler_discriminator,
                            temperature, train=False, gan=gan)
        # todo: load new model
        # if args.save:
        #     if epoch % args.epochs_save == 0:
        #         fname = args.model_save_path + args.note + '_' + args.graph_type + '_' + \
        #                 str(epoch) + str(args.num_layers) + '_' + str(args.bptt)+'_'+str(args.bptt_len)+ '.dat'
        #         torch.save(generator.state_dict(), fname)
        epoch += 1


########### train function for LSTM + VAE
def train_vae(args, dataset_train, lstm, output_vae):
    # todo: load new models
    # if args.load:
    #     epoch_load = args.load_epoch
    #     fname = args.model_save_path + args.note + '_' + args.graph_type + '_' + \
    #     str(epoch_load) + str(args.num_layers) + '_' + str(args.bptt)+'_'+str(args.bptt_len)+ '.dat'
    #     generator.load_state_dict(torch.load(fname))
    #     args.lr = 0.00001
    #     epoch = epoch_load
    #     print('model loaded!')
    # else:
    #     epoch = 0

    # torch.manual_seed(args.seed)
    optimizer_lstm = optim.Adam(list(lstm.parameters()), lr=args.lr)
    optimizer_vae = optim.Adam(list(output_vae.parameters()), lr=args.lr)

    scheduler_lstm = MultiStepLR(optimizer_lstm, milestones=args.milestones, gamma=args.lr_rate)
    scheduler_vae = MultiStepLR(optimizer_vae, milestones=args.milestones, gamma=args.lr_rate)

    epoch = 0
    while epoch<=args.epochs:
        # train

        train_vae_epoch(epoch, args, lstm, output_vae, dataset_train,
                        optimizer_lstm, optimizer_vae,
                        scheduler_lstm, scheduler_vae, train=True)
        # test
        if epoch % args.epochs_test == 0:
            train_vae_epoch(epoch, args, lstm, output_vae, dataset_train,
                            optimizer_lstm, optimizer_vae,
                            scheduler_lstm, scheduler_vae, train=False)
        # todo: load new model
        # if args.save:
        #     if epoch % args.epochs_save == 0:
        #         fname = args.model_save_path + args.note + '_' + args.graph_type + '_' + \
        #                 str(epoch) + str(args.num_layers) + '_' + str(args.bptt)+'_'+str(args.bptt_len)+ '.dat'
        #         torch.save(generator.state_dict(), fname)
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

    # torch.manual_seed(args.seed)
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
        fname = args.model_save_path + args.note + '_' + args.graph_type + '_' + \
                str(epoch_load) + str(args.num_layers) + '_' + str(args.bptt) + '.dat'
        generator.load_state_dict(torch.load(fname))
        args.lr = 0.00001
        epoch = epoch_load
        print('model loaded!')
    else:
        epoch = 0

    # torch.manual_seed(args.seed)
    optimizer = optim.Adam(list(generator.parameters()), lr=args.lr)
    scheduler = MultiStepLR(optimizer, milestones=args.milestones, gamma=args.lr_rate)
    thresh = 0.3


    while epoch<=args.epochs:
        if args.is_flexible:
            thresh = train_epoch_GraphRNN_structure_flexible(epoch, args, generator, dataset_train, optimizer, scheduler, thresh, train=True)
        else:
            thresh = train_epoch_GraphRNN_structure_bptt(epoch, args, generator, dataset_train, optimizer, scheduler, thresh, train=True)
        if epoch % args.epochs_test == 0:
            if args.is_flexible:
                train_epoch_GraphRNN_structure_flexible(epoch, args, generator, dataset_train, optimizer, scheduler, thresh, train=False)
            else:
                train_epoch_GraphRNN_structure_bptt(epoch, args, generator, dataset_train, optimizer, scheduler, thresh, train = False)
        if args.save:
            if epoch % args.epochs_save == 0:
                fname = args.model_save_path + args.note + '_' + args.graph_type + '_' + \
                        str(epoch) + str(args.num_layers) + '_' + str(args.bptt) + '.dat'
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

    # torch.manual_seed(args.seed)
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
    args = Args()
    print('CUDA', CUDA)
    time = strftime("%Y-%m-%d %H:%M:%S", gmtime())
    # logging.basicConfig(filename='logs/train' + time + '.log', level=logging.DEBUG)

    if args.clean_tensorboard:
        if os.path.isdir("tensorboard"):
            shutil.rmtree("tensorboard")
    configure("tensorboard/run"+time, flush_secs=5)

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
        args.max_prev_node = 30
    if args.graph_type=='karate':
        G = nx.karate_club_graph()
        graphs = [G]
        max_num_nodes = G.number_of_nodes()
    if args.graph_type=='tree':
        graphs = []
        for i in range(2, 5):
            for j in range(2, 5):
                graphs.append(nx.balanced_tree(i, j))
        max_num_nodes = 340
        args.max_prev_node = 100
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
        args.max_prev_node = 100
    if args.graph_type=='barabasi':
        graphs = []
        for i in range(100,401):
            graphs.append(nx.barabasi_albert_graph(i,2))
        max_num_nodes = 400
        args.max_prev_node = 100

    # if using a list of graphs
    if args.graph_type == 'enzymes':
        graphs, max_num_nodes = Graph_load_batch(min_num_nodes=6, name='ENZYMES')
        print('max num nodes', max_num_nodes)
        args.max_prev_node = 30
    if args.graph_type == 'protein':
        graphs, max_num_nodes = Graph_load_batch(min_num_nodes=6, name='PROTEINS_full')
        print('max num nodes', max_num_nodes)
        args.max_prev_node = 50
    if args.graph_type == 'DD':
        graphs, max_num_nodes = Graph_load_batch(min_num_nodes=6, max_num_nodes=1000, name='DD',node_attributes=False,graph_labels=True)
        print('max num nodes', max_num_nodes)
        args.max_prev_node = 150

    print('max prev node', args.max_prev_node)
    ################## the GraphRNN model #####################
    ### 'Graph_sequence_sampler_rnn' is used for baseline model
    sampler = Graph_sequence_sampler_rnn(graphs, max_node_num=max_num_nodes,batch_size=args.batch_size, max_prev_node=args.max_prev_node)
    # x, y, len = sampler.sample()
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
    # generator = Graph_generator_LSTM(feature_size=x.size(2), input_size=args.input_size,
    #                                  hidden_size=args.hidden_size,
    #                                  output_size=y.size(2), batch_size=args.batch_size, num_layers=args.num_layers).cuda(CUDA)
    # train(args,sampler,generator)

    ### Graph RNN GAN model
    # lstm = Graph_generator_LSTM_plain(feature_size=args.max_prev_node, input_size=args.input_size,
    #                                  hidden_size=args.hidden_size, batch_size=args.batch_size, num_layers=args.num_layers).cuda(CUDA)
    # output_deterministic = Graph_generator_LSTM_output_deterministic_mlp(h_size=args.hidden_size, y_size=args.max_prev_node, has_noise=args.has_noise, noise_level=args.noise_level).cuda(CUDA)
    # output_generator = Graph_generator_LSTM_output_generator(h_size=args.hidden_size, n_size=args.noise_size, y_size=args.max_prev_node).cuda(CUDA)
    # output_discriminator = Graph_generator_LSTM_output_discriminator(h_size=args.hidden_size, y_size=args.max_prev_node).cuda(CUDA)
    # train_gan(args, sampler, lstm, output_deterministic, output_generator, output_discriminator)

    ### Graph RNN VAE model
    lstm = Graph_generator_LSTM_plain(feature_size=args.max_prev_node, input_size=args.input_size,
                                      hidden_size=args.hidden_size, batch_size=args.batch_size,
                                      num_layers=args.num_layers).cuda(CUDA)
    output_vae = Graph_generator_LSTM_output_vae(h_size=args.hidden_size, embedding_size=args.embedding_size,
                                                                         y_size=args.max_prev_node).cuda(CUDA)

    train_vae(args, sampler, lstm, output_vae)

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