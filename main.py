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
import scipy.misc


### program configuration
class Args():
    def __init__(self):
        ### if clean tensorboard
        # self.clean_tensorboard = True
        self.clean_tensorboard = False

        ### model type
        # self.note = 'GraphRNN_MLP'
        self.note = 'GraphRNN_VAE'
        # self.note = 'GraphRNN_RNN' # todo

        ### data config
        # self.graph_type = 'star' # obsolete
        # self.graph_type = 'karate' # obsolete

        # self.graph_type = 'ladder'
        # self.graph_type = 'tree'
        # self.graph_type = 'caveman'
        # self.graph_type = 'grid'
        # self.graph_type = 'barabasi'
        self.graph_type = 'enzymes'
        # self.graph_type = 'protein'
        # self.graph_type = 'DD'

        # update when initializing dataset
        self.max_num_node = None # max number of nodes in a graph
        self.max_prev_node = None # max previous node that looks back


        ### network config
        ## GraphRNN
        self.hidden_size = 64 # hidden size for main LSTM
        self.embedding_size_lstm = 64 # the size for LSTM input
        self.embedding_size_output = 64 # the embedding size for output (VAE/MLP)
        self.batch_size = 128
        self.test_batch_size = 128
        self.test_total_size = 1000
        self.num_layers = 4
        self.bptt = False # if use truncate back propagation (not very stable)
        self.bptt_len = 20
        self.gumbel = True

        ### training config
        self.num_workers = 4 # num workers to load data
        self.batch_ratio = 32 # how many batches per epoch
        self.epochs = 2000 # now one epoch means 16 x batch_size
        self.epochs_gumbel_start = 100 # at what time start gumbel training
        self.epochs_test_start = 200
        self.epochs_test = 200
        self.epochs_log = 200
        self.epochs_save = 200

        self.lr = 0.003
        self.milestones = [500, 1000, 2000]
        self.lr_rate = 0.3

        self.sample_time = 1 # sample time in each time step, when validating

        ### output config
        self.model_save_path = 'model_save/'
        self.graph_save_path = 'graphs/'
        self.figure_save_path = 'figures/'
        self.figure_prediction_save_path = 'figures_prediction/'

        self.load = False # if load model, default lr is very low
        self.load_epoch = 100
        self.save = True

        ### fname
        self.fname = self.note + '_' + self.graph_type + '_' + str(self.num_layers) + '_' + str(self.bptt) + '_' + \
                          str(self.bptt_len) + '_' + str(self.gumbel) + '_'
        self.fname_pred=self.note+'_'+self.graph_type+'_'+str(self.num_layers)+'_'+str(self.bptt)+'_'+\
                        str(self.bptt_len)+'_'+str(self.gumbel)+'_pred_'
        self.fname_real = self.note + '_' + self.graph_type + '_' + str(self.num_layers) + '_' + str(self.bptt) + '_' + \
                          str(self.bptt_len) + '_' + str(self.gumbel) + '_real_'





def train_vae_epoch(epoch, args, lstm, output, data_loader,
                    optimizer_lstm, optimizer_output,
                    scheduler_lstm, scheduler_output, gumbel=0, temperature=1):
    lstm.train()
    output.train()
    for batch_idx, data in enumerate(data_loader):
        lstm.zero_grad()
        output.zero_grad()
        x_unsorted = data['x'].float()
        y_unsorted = data['y'].float()
        y_len_unsorted = data['len']
        y_len_max = max(y_len_unsorted)
        x_unsorted = x_unsorted[:, 0:y_len_max, :]
        y_unsorted = y_unsorted[:, 0:y_len_max, :]
        # initialize lstm hidden state according to batch size
        lstm.hidden = lstm.init_hidden(batch_size=x_unsorted.size(0))

        # sort input
        y_len,sort_index = torch.sort(y_len_unsorted,0,descending=True)
        y_len = y_len.numpy().tolist()
        x = torch.index_select(x_unsorted,0,sort_index)
        y = torch.index_select(y_unsorted,0,sort_index)
        x = Variable(x).cuda(CUDA)
        y = Variable(y).cuda(CUDA)

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
            if gumbel>np.random.rand(): # prob of using gumbel
                y_pred = Variable(torch.zeros(y.size(0), y.size(1), y.size(2))).cuda(CUDA) # normalized score
                z_mu = Variable(torch.zeros(y.size(0), y.size(1), args.embedding_size_output)).cuda(CUDA) # normalized score
                z_lsgms = Variable(torch.zeros(y.size(0), y.size(1), args.embedding_size_output)).cuda(CUDA) # normalized score

                x_step = x[:, 0:1, :] # all ones
                for i in range(x.size(1)):
                    h = lstm(x_step)
                    y_pred_step,z_mu_step,z_lsgms_step = output(h)
                    y_pred[:, i:i + 1, :] = F.sigmoid(y_pred_step) # write down score
                    z_mu[:, i:i + 1, :] = z_mu_step  # write down score
                    z_lsgms[:, i:i + 1, :] = z_lsgms_step  # write down score
                    x_step = gumbel_sigmoid(y_pred_step, temperature=temperature) # do sampling

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
                loss_kl /= y.size(0) * y.size(1) * sum(y_len)  # normalize
                loss = loss_bce + loss_kl
                loss.backward()

                # update deterministic and lstm
                optimizer_output.step()
                optimizer_lstm.step()
                scheduler_output.step()
                scheduler_lstm.step()
            # if using ground truth to train
            else:
                h = lstm(x, pack=True, input_len=y_len)
                y_pred,z_mu,z_lsgms = output(h)
                y_pred = F.sigmoid(y_pred)
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
                optimizer_output.step()
                optimizer_lstm.step()
                scheduler_output.step()
                scheduler_lstm.step()


            z_mu_mean = torch.mean(z_mu.data)
            z_sgm_mean = torch.mean(z_lsgms.mul(0.5).exp_().data)
            z_mu_min = torch.min(z_mu.data)
            z_sgm_min = torch.min(z_lsgms.mul(0.5).exp_().data)
            z_mu_max = torch.max(z_mu.data)
            z_sgm_max = torch.max(z_lsgms.mul(0.5).exp_().data)


        if epoch % args.epochs_log==0 and batch_idx==0: # only output first batch's statistics
            print('Epoch: {}/{}, train bce loss: {:.6f}, train kl loss: {:.6f}, graph type: {}, num_layer: {}, bptt: {}, bptt_len:{}, gumbel:{}, temperature:{}'.format(
                epoch, args.epochs,loss_bce.data[0], loss_kl.data[0], args.graph_type, args.num_layers, args.bptt, args.bptt_len, args.gumbel, temperature))
            print('z_mu_mean', z_mu_mean, 'z_mu_min', z_mu_min, 'z_mu_max', z_mu_max, 'z_sgm_mean', z_sgm_mean, 'z_sgm_min', z_sgm_min, 'z_sgm_max', z_sgm_max)

        # logging
        log_value('bce_loss_'+args.fname, loss_bce.data[0], epoch*args.batch_ratio+batch_idx)
        log_value('kl_loss_' +args.fname, loss_kl.data[0], epoch*args.batch_ratio + batch_idx)
        log_value('z_mu_mean_'+args.fname, z_mu_mean, epoch*args.batch_ratio + batch_idx)
        log_value('z_mu_min_'+args.fname, z_mu_min, epoch*args.batch_ratio + batch_idx)
        log_value('z_mu_max_'+args.fname, z_mu_max, epoch*args.batch_ratio + batch_idx)
        log_value('z_sgm_mean_'+args.fname, z_sgm_mean, epoch*args.batch_ratio + batch_idx)
        log_value('z_sgm_min_'+args.fname, z_sgm_min, epoch*args.batch_ratio + batch_idx)
        log_value('z_sgm_max_'+args.fname, z_sgm_max, epoch*args.batch_ratio + batch_idx)


def test_vae_epoch(epoch, args, lstm, output, test_batch_size=16, save_histogram=False):
    lstm.hidden = lstm.init_hidden(test_batch_size)
    lstm.eval()
    output.eval()

    # generate graphs
    max_num_node = int(args.max_num_node*1.5) # allowing to generate 1.5x bigger graph
    y_pred = Variable(torch.zeros(test_batch_size, max_num_node, args.max_prev_node)).cuda(CUDA) # normalized prediction score
    y_pred_long = Variable(torch.zeros(test_batch_size, max_num_node, args.max_prev_node)).cuda(CUDA) # discrete prediction
    x_step = Variable(torch.ones(test_batch_size,1,args.max_prev_node)).cuda(CUDA)
    for i in range(max_num_node):
        h = lstm(x_step)
        y_pred_step, _, _ = output(h)
        y_pred[:, i:i + 1, :] = F.sigmoid(y_pred_step)
        x_step = sample_sigmoid(y_pred_step, sample=True)
        y_pred_long[:, i:i + 1, :] = x_step
        lstm.hidden = (Variable(lstm.hidden[0].data).cuda(CUDA), Variable(lstm.hidden[1].data).cuda(CUDA))
    y_pred_data = y_pred.data
    y_pred_long_data = y_pred_long.data.long()

    # save graphs as pickle
    G_pred_list = []
    for i in range(test_batch_size):
        adj_pred = decode_adj(y_pred_long_data[i].cpu().numpy(), args.max_prev_node)
        G_pred = get_graph(adj_pred) # get a graph from zero-padded adj
        G_pred_list.append(G_pred)


    # save list of objects
    fname_pred = args.graph_save_path + args.note + '_' + args.graph_type + '_' + \
                 str(epoch) + '_pred_' + str(args.num_layers) + '_' + str(args.bptt) + '_' + str(
        args.bptt_len) + '_' + str(args.gumbel) + '.dat'
    save_graph_list(G_pred_list, fname_pred)


    # save prediction histograms, plot histogram over each time step
    if save_histogram:
        save_prediction_histogram(y_pred_data.cpu().numpy(),
                              fname_pred=args.figure_prediction_save_path+args.fname_pred+str(epoch)+'.jpg',
                              max_num_node=max_num_node)
    return G_pred_list




def train_mlp_epoch(epoch, args, lstm, output, data_loader,
                    optimizer_lstm, optimizer_output,
                    scheduler_lstm, scheduler_output, gumbel=0, temperature=1):
    lstm.train()
    output.train()
    for batch_idx, data in enumerate(data_loader):
        lstm.zero_grad()
        output.zero_grad()
        x_unsorted = data['x'].float()
        y_unsorted = data['y'].float()
        y_len_unsorted = data['len']
        y_len_max = max(y_len_unsorted)
        x_unsorted = x_unsorted[:, 0:y_len_max, :]
        y_unsorted = y_unsorted[:, 0:y_len_max, :]
        # initialize lstm hidden state according to batch size
        lstm.hidden = lstm.init_hidden(batch_size=x_unsorted.size(0))

        # sort input
        y_len,sort_index = torch.sort(y_len_unsorted,0,descending=True)
        y_len = y_len.numpy().tolist()
        x = torch.index_select(x_unsorted,0,sort_index)
        y = torch.index_select(y_unsorted,0,sort_index)
        x = Variable(x).cuda(CUDA)
        y = Variable(y).cuda(CUDA)

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
            if gumbel>np.random.rand(): # prob of using gumbel
                y_pred = Variable(torch.zeros(y.size(0), y.size(1), y.size(2))).cuda(CUDA) # normalized score
                x_step = x[:, 0:1, :] # all ones
                for i in range(x.size(1)):
                    h = lstm(x_step)
                    y_pred_step = output(h)
                    y_pred[:, i:i + 1, :] = F.sigmoid(y_pred_step) # write down score
                    x_step = gumbel_sigmoid(y_pred_step, temperature=temperature) # do sampling

                # clean
                y_pred = pack_padded_sequence(y_pred, y_len, batch_first=True)
                y_pred = pad_packed_sequence(y_pred, batch_first=True)[0]
                # use cross entropy loss
                loss = F.binary_cross_entropy(y_pred, y)
                loss.backward()

                # update deterministic and lstm
                optimizer_output.step()
                optimizer_lstm.step()
                scheduler_output.step()
                scheduler_lstm.step()
            # if using ground truth to train
            else:
                h = lstm(x, pack=True, input_len=y_len)
                y_pred = output(h)
                y_pred = F.sigmoid(y_pred)
                # clean
                y_pred = pack_padded_sequence(y_pred, y_len, batch_first=True)
                y_pred = pad_packed_sequence(y_pred, batch_first=True)[0]
                # use cross entropy loss
                loss = F.binary_cross_entropy(y_pred, y)
                loss.backward()
                # update deterministic and lstm
                optimizer_output.step()
                optimizer_lstm.step()
                scheduler_output.step()
                scheduler_lstm.step()


        if epoch % args.epochs_log==0 and batch_idx==0: # only output first batch's statistics
            print('Epoch: {}/{}, train loss: {:.6f}, graph type: {}, num_layer: {}, bptt: {}, bptt_len:{}, gumbel:{}, temperature:{}'.format(
                epoch, args.epochs,loss.data[0], args.graph_type, args.num_layers, args.bptt, args.bptt_len, args.gumbel, temperature))

        # logging
        log_value('loss_'+args.fname, loss.data[0], epoch*args.batch_ratio+batch_idx)




def test_mlp_epoch(epoch, args, lstm, output, test_batch_size=16, save_histogram=False):
    lstm.hidden = lstm.init_hidden(test_batch_size)
    lstm.eval()
    output.eval()

    # generate graphs
    max_num_node = int(args.max_num_node*1.5) # allowing to generate 1.5x bigger graph
    y_pred = Variable(torch.zeros(test_batch_size, max_num_node, args.max_prev_node)).cuda(CUDA) # normalized prediction score
    y_pred_long = Variable(torch.zeros(test_batch_size, max_num_node, args.max_prev_node)).cuda(CUDA) # discrete prediction
    x_step = Variable(torch.ones(test_batch_size,1,args.max_prev_node)).cuda(CUDA)
    for i in range(max_num_node):
        h = lstm(x_step)
        y_pred_step = output(h)
        y_pred[:, i:i + 1, :] = F.sigmoid(y_pred_step)
        x_step = sample_sigmoid(y_pred_step, sample=True)
        y_pred_long[:, i:i + 1, :] = x_step
        lstm.hidden = (Variable(lstm.hidden[0].data).cuda(CUDA), Variable(lstm.hidden[1].data).cuda(CUDA))
    y_pred_data = y_pred.data
    y_pred_long_data = y_pred_long.data.long()

    # save graphs as pickle
    G_pred_list = []
    for i in range(test_batch_size):
        adj_pred = decode_adj(y_pred_long_data[i].cpu().numpy(), args.max_prev_node)
        G_pred = get_graph(adj_pred) # get a graph from zero-padded adj
        G_pred_list.append(G_pred)


    # save list of objects
    fname_pred = args.graph_save_path + args.note + '_' + args.graph_type + '_' + \
                 str(epoch) + '_pred_' + str(args.num_layers) + '_' + str(args.bptt) + '_' + str(
        args.bptt_len) + '_' + str(args.gumbel) + '.dat'
    save_graph_list(G_pred_list, fname_pred)


    # save prediction histograms, plot histogram over each time step
    if save_histogram:
        save_prediction_histogram(y_pred_data.cpu().numpy(),
                              fname_pred=args.figure_prediction_save_path+args.fname_pred+str(epoch)+'.jpg',
                              max_num_node=max_num_node)
    return G_pred_list




########### train function for LSTM + VAE
def train_vae(args, dataset_train, lstm, output):
    # check if load existing model
    if args.load:
        fname = args.model_save_path + args.fname + 'lstm_' + str(args.load_epoch) + '.dat'
        lstm.load_state_dict(torch.load(fname))
        fname = args.model_save_path + args.fname + 'output_' + str(args.load_epoch) + '.dat'
        output.load_state_dict(torch.load(fname))

        args.lr = 0.00001
        epoch = args.load_epoch
        print('model loaded!, lr: {}'.format(args.lr))
    else:
        epoch = 0

    # initialize optimizer
    optimizer_lstm = optim.Adam(list(lstm.parameters()), lr=args.lr)
    optimizer_vae = optim.Adam(list(output.parameters()), lr=args.lr)

    scheduler_lstm = MultiStepLR(optimizer_lstm, milestones=args.milestones, gamma=args.lr_rate)
    scheduler_vae = MultiStepLR(optimizer_vae, milestones=args.milestones, gamma=args.lr_rate)

    # start main loop
    while epoch<=args.epochs:
        # gradually using gumbel sigmoid to train the model
        if epoch<args.epochs_gumbel_start:
            gumbel = 0
            temperature = 1
        else:
            gumbel = 1-np.exp((-3e-3)*(epoch-args.epochs_gumbel_start))
            temperature = np.exp((-3e-3) * (epoch - args.epochs_gumbel_start))

        # train
        if args.note == 'GraphRNN_VAE':
            train_vae_epoch(epoch, args, lstm, output, dataset_train,
                            optimizer_lstm, optimizer_vae,
                            scheduler_lstm, scheduler_vae, gumbel=gumbel, temperature=temperature)
        elif args.note == 'GraphRNN_MLP':
            train_mlp_epoch(epoch, args, lstm, output, dataset_train,
                            optimizer_lstm, optimizer_vae,
                            scheduler_lstm, scheduler_vae, gumbel=gumbel, temperature=temperature)

        # test
        if epoch % args.epochs_test == 0 and epoch>=args.epochs_test_start:
            G_pred = []
            while len(G_pred)<args.test_total_size:
                if args.note == 'GraphRNN_VAE':
                    G_pred_step = test_vae_epoch(epoch, args, lstm, output, test_batch_size=args.test_batch_size)
                elif args.note == 'GraphRNN_MLP':
                    G_pred_step = test_mlp_epoch(epoch, args, lstm, output, test_batch_size=args.test_batch_size)
                G_pred.extend(G_pred_step)
            # save graphs
            fname = args.graph_save_path + args.fname_pred + str(epoch) + '.dat'
            save_graph_list(G_pred, fname)

        # save model checkpoint
        if args.save:
            if epoch % args.epochs_save == 0:
                fname = args.model_save_path + args.fname + 'lstm_' + str(epoch) + '.dat'
                torch.save(lstm.state_dict(), fname)
                fname = args.model_save_path + args.fname + 'output_' + str(epoch) + '.dat'
                torch.save(output.state_dict(), fname)
        epoch += 1


if __name__ == '__main__':
    print('CUDA', CUDA)
    ### running log
    args = Args()
    print('File name prefix',args.fname)
    time = strftime("%Y-%m-%d %H:%M:%S", gmtime())
    # logging.basicConfig(filename='logs/train' + time + '.log', level=logging.DEBUG)
    if args.clean_tensorboard:
        if os.path.isdir("tensorboard"):
            shutil.rmtree("tensorboard")
    configure("tensorboard/run"+time, flush_secs=5)

    ### load datasets
    graphs=[]
    # synthetic graphs
    if args.graph_type=='star':
        G = nx.star_graph(args.graph_node_num)
        graphs = [G]
    if args.graph_type=='ladder':
        graphs = []
        for i in range(100, 201):
            graphs.append(nx.ladder_graph(i))
        args.max_prev_node = 10
    if args.graph_type=='karate':
        G = nx.karate_club_graph()
        graphs = [G]
    if args.graph_type=='tree':
        graphs = []
        for i in range(2, 5):
            for j in range(3,5):
                graphs.append(nx.balanced_tree(i,j))
        args.max_prev_node = 256
    if args.graph_type=='caveman':
        graphs = []
        for i in range(2,11):
            for j in range(10,41):
                for k in range(10):
                    graphs.append(nx.relaxed_caveman_graph(i, j, 0.1))
        args.max_prev_node = 340
    if args.graph_type=='grid':
        graphs = []
        for i in range(10,21):
            for j in range(10,21):
                graphs.append(nx.grid_2d_graph(i,j))
        args.max_prev_node = 50
    if args.graph_type=='barabasi':
        graphs = []
        for i in range(100,401):
            for j in range(10):
                graphs.append(nx.barabasi_albert_graph(i,2))
        args.max_prev_node = 250
    # real graphs
    if args.graph_type == 'enzymes':
        graphs= Graph_load_batch(min_num_nodes=10, name='ENZYMES')
        args.max_prev_node = 30
    if args.graph_type == 'protein':
        graphs = Graph_load_batch(min_num_nodes=20, name='PROTEINS_full')
        args.max_prev_node = 80
    if args.graph_type == 'DD':
        graphs = Graph_load_batch(min_num_nodes=50, max_num_nodes=1000, name='DD',node_attributes=False,graph_labels=True)
        args.max_prev_node = 230

    args.max_num_node = max([graphs[i].number_of_nodes() for i in range(len(graphs))])

    # show graphs statistics
    print('total graph num: {}'.format(len(graphs)))
    print('max number node: {}'.format(args.max_num_node))
    print('max previous node: {}'.format(args.max_prev_node))

    # save ground truth graphs
    save_graph_list(graphs,args.graph_save_path + args.fname_real + '0.dat')


    ### dataset initialization
    dataset = Graph_sequence_sampler_truncate_pytorch(graphs, max_prev_node=args.max_prev_node)
    sample_strategy = torch.utils.data.sampler.WeightedRandomSampler([1.0 / len(dataset) for i in range(len(dataset))],
                                                                     num_samples=args.batch_size*args.batch_ratio, replacement=True)
    dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                                               sampler=sample_strategy)


    ### model initialization
    ## Graph RNN VAE model
    lstm = LSTM_plain(input_size=args.max_prev_node, embedding_size=args.embedding_size_lstm,
                      hidden_size=args.hidden_size, num_layers=args.num_layers).cuda(CUDA)
    # if using vae
    if args.note == 'GraphRNN_VAE':
        output = MLP_VAE_plain(h_size=args.hidden_size, embedding_size=args.embedding_size_output, y_size=args.max_prev_node).cuda(CUDA)
        # if using vae
    elif args.note == 'GraphRNN_MLP':
        output = MLP_plain(h_size=args.hidden_size, embedding_size=args.embedding_size_output, y_size=args.max_prev_node).cuda(CUDA)

    train_vae(args, dataset_loader, lstm, output)
