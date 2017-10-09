from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import torch.nn.init as init

USE_CUDA = torch.cuda.is_available()


class DecoderRNN_step(nn.Module):
    def __init__(self, input_size, hidden_size, embedding_size, n_layers=1, is_bidirection = True):
        super(DecoderRNN_step, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.is_bidirection = is_bidirection

        self.gru = nn.GRU(input_size, hidden_size, bidirectional=self.is_bidirection, batch_first=True)
        # init.xavier_uniform(self.gru.weight)
        self.linear = nn.Linear(hidden_size,hidden_size)
        self.softmax = torch.nn.Softmax()

        self.embedding = nn.Embedding(embedding_size, hidden_size)
        self.embedding.weight.data = init.uniform(torch.Tensor(embedding_size, hidden_size))
    def forward(self, input, hidden, modify_input = True):
        # run one time step at a time
        # input = F.relu(input) # maybe a trick, need to try

        # input_shape: (batch, seq_length, input_size)
        # hidden_shape: (batch, n_layers*n_directions, hidden_size)
        # input_shape (before modify): (batch, input_size)
        input = input.view(input.size(0),1,input.size(1))
        output, hidden = self.gru(input, hidden)
        output = output[:, :, :self.hidden_size] + output[:, :, self.hidden_size:]  # Sum bidirectional outputs
        output = torch.squeeze(output,dim=1)
        output = self.linear(output)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        result = Variable(torch.zeros(self.n_layers*self.is_bidirection*2, 1, self.hidden_size))
        if USE_CUDA:
            return result.cuda()
        else:
            return result


# # test DecoderRNN_step
# seq_len = 5
# hidden_size = 4
#
# decoder = DecoderRNN_step(input_size=hidden_size, hidden_size=hidden_size)
#
# hidden = Variable(torch.rand(1,hidden_size),requires_grad = True)
# softmax = nn.Softmax()
# hidden = softmax(hidden).view(1,hidden.size(0),hidden.size(1))
# hidden = torch.cat((hidden,hidden),dim = 0)
# print('hidden -- 0', hidden, torch.sum(hidden))
# input_seqence = Variable(torch.rand(1,seq_len,hidden_size))
#
# loss_f = nn.BCELoss()
#
# output = softmax(input_seqence[:,0,:])
# print('input -- 0',output, torch.sum(hidden))
# loss = 0
# for i in range(seq_len-1):
#     output, hidden = decoder(output,hidden)
#     print('output -- '+str(i), output, torch.sum(output))
#     print('hidden -- '+str(i), hidden, torch.sum(hidden))
#     loss += loss_f(output,input_seqence[:,i+1,:])
#     print('loss -- '+str(i), loss_f(output,input_seqence[:,i+1,:]))
# print('total loss', loss)









# reference code


#
# class EncoderRNN(nn.Module):
#     def __init__(self, input_size, hidden_size, n_layers=1, is_bidirection = True):
#         super(EncoderRNN, self).__init__()
#         self.n_layers = n_layers
#         self.hidden_size = hidden_size
#         self.is_bidirection = is_bidirection
#
#         self.gru = nn.GRU(input_size, hidden_size, bidirectional=self.is_bidirection, batch_first=True)
#
#     def forward(self, input, hidden):
#         # run the whole sequence at a time
#         output, hidden = self.gru(input, hidden)
#         return output, hidden
#
#     def initHidden(self):
#         result = Variable(torch.zeros(self.n_layers*self.is_bidirection*2, 1, self.hidden_size))
#         if USE_CUDA:
#             return result.cuda()
#         else:
#             return result
#
# class DecoderRNN(nn.Module):
#     def __init__(self, input_size, hidden_size, n_layers=1, is_bidirection = True):
#         super(DecoderRNN, self).__init__()
#         self.n_layers = n_layers
#         self.hidden_size = hidden_size
#         self.is_bidirection = is_bidirection
#
#         self.gru = nn.GRU(input_size, hidden_size, bidirectional=self.is_bidirection, batch_first=True)
#
#     def forward(self, input, hidden):
#         # run one time step at a time
#         # input = F.relu(input) # maybe a trick, need to try
#         output, hidden = self.gru(input, hidden)
#         output = self.softmax(output)
#         return output, hidden
#
#     def initHidden(self):
#         result = Variable(1, torch.zeros(self.n_layers*self.is_bidirection*2, self.hidden_size))
#         if USE_CUDA:
#             return result.cuda()
#         else:
#             return result
#
#
# encoder = EncoderRNN(16, 64, 1).cuda()
# decoder = DecoderRNN(16, 64)
# input_dummy = Variable(torch.rand(1,30,16)).cuda()
# hidden = encoder.initHidden()
# output, hidden = encoder(input_dummy,hidden)
# print('output', output.size())
# print('hidden', hidden.size())