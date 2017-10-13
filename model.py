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
from collections import OrderedDict
import math


USE_CUDA = torch.cuda.is_available()

# class deconv1d(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size = 3, stride = 2, padding = 0, output_padding = 0, groups = 1, bias = True, dilation = 1):
#         super().__init__()
#         self.deconv1d = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding, output_padding, groups, bias, dilation)
#
#     def forward(self, input):
#         '''
#
#         :param input: N * in_channels * L_in
#         :return: output: N * out_channels * L_out
#         '''



class CNN_decoder(nn.Module):
    def __init__(self, input_size, output_size):

        super(CNN_decoder, self).__init__()

        self.input_size = input_size
        self.output_size = output_size

        self.deconv1 = nn.ConvTranspose1d(in_channels=int(self.input_size), out_channels=int(self.input_size/2), kernel_size=3, stride=2)
        self.bn1 = nn.BatchNorm1d(int(self.input_size/2))
        self.relu = nn.ReLU()
        self.deconv2 = nn.ConvTranspose1d(in_channels=int(self.input_size/2), out_channels=int(self.input_size/4), kernel_size=3, stride=2)
        self.bn2 = nn.BatchNorm1d(int(self.input_size/4))
        self.deconv3 = nn.ConvTranspose1d(in_channels=int(self.input_size/4), out_channels=int(self.output_size), kernel_size=3, stride=1, padding=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        '''

        :param
        x: batch * channel * length
        :return:
        '''
        x = self.deconv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        print(x.size())


        x = self.deconv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        print(x.size())


        x = self.deconv3(x)
        print(x.size())
        return x


x = Variable(torch.randn(1, 256, 1)).cuda()
decoder = CNN_decoder(256, 16).cuda()
y = decoder(x)



    # # reference code for doing residual connections
    # def _make_layer(self, block, planes, blocks, stride=1):
    #     downsample = None
    #     if stride != 1 or self.inplanes != planes * block.expansion:
    #         downsample = nn.Sequential(
    #             nn.Conv2d(self.inplanes, planes * block.expansion,
    #                       kernel_size=1, stride=stride, bias=False),
    #             nn.BatchNorm2d(planes * block.expansion),
    #         )
    #
    #     layers = []
    #     layers.append(block(self.inplanes, planes, stride, downsample))
    #     self.inplanes = planes * block.expansion
    #     for i in range(1, blocks):
    #         layers.append(block(self.inplanes, planes))
    #
    #     return nn.Sequential(*layers)



class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1, is_bidirection = True):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.is_bidirection = is_bidirection

        self.gru = nn.GRU(input_size, hidden_size, bidirectional=self.is_bidirection, batch_first=True)

    def forward(self, input, hidden):
        # run the whole sequence at a time
        output, hidden = self.gru(input, hidden)
        return output, hidden

    def initHidden(self):
        result = Variable(torch.zeros(self.n_layers*self.is_bidirection*2, 1, self.hidden_size), requires_grad = True)
        if USE_CUDA:
            return result.cuda()
        else:
            return result



class DecoderRNN_step(nn.Module):
    def __init__(self, input_size, hidden_size, embedding_size, n_layers=1, is_bidirection = True, embedding_init_flag = False, embedding_init = 0, hidden_grad = False):
        super(DecoderRNN_step, self).__init__()
        self.n_layers = n_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.is_bidirection = is_bidirection

        self.gru = nn.GRU(input_size, hidden_size, bidirectional=self.is_bidirection, batch_first=True)
        # init.xavier_uniform(self.gru.weight)
        self.linear = nn.Linear(hidden_size,input_size)
        self.softmax = torch.nn.Softmax()
        self.tanh = torch.nn.Tanh()

        self.embedding = nn.Embedding(embedding_size, input_size)
        print(self.embedding.weight.data.size())
        if embedding_init_flag == False:
            self.embedding.weight.data = init.uniform(torch.Tensor(embedding_size, input_size))
        else:
            self.embedding.weight.data[3:,:] = embedding_init

        if self.is_bidirection:
            w = torch.Tensor(self.n_layers*2, 1, self.hidden_size)
            init.xavier_uniform(w, gain=init.calculate_gain('relu'))
            self.hidden = nn.Parameter(w, requires_grad = hidden_grad)

            # self.hidden = nn.Parameter(torch.zeros(self.n_layers*2, 1, self.hidden_size), requires_grad = hidden_grad)
        else:
            w = torch.Tensor(self.n_layers, 1, self.hidden_size)
            init.xavier_uniform(w, gain=init.calculate_gain('relu'))
            self.hidden = nn.Parameter(w, requires_grad = hidden_grad)

            # self.hidden = nn.Parameter(torch.zeros(self.n_layers, 1, self.hidden_size), requires_grad = hidden_grad)

        # for m in self.modules():
        #     print(m)

    def forward(self, input, hidden, modify_input = True):
        # run one time step at a time
        # input = F.relu(input) # maybe a trick, need to try

        # input_shape: (batch, seq_length, input_size)
        # hidden_shape: (batch, n_layers*n_directions, hidden_size)
        # input_shape (before modify): (batch, input_size)
        input = input.view(input.size(0),1,input.size(1))
        self.gru.flatten_parameters() # fix pytorch warning
        output, hidden = self.gru(input, hidden)
        # uncomment if bi-directional

        if self.is_bidirection:
            output = output[:, :, :self.hidden_size] + output[:, :, self.hidden_size:]  # Sum bidirectional outputs
        output = torch.squeeze(output,dim=1)
        output = self.linear(output)
        output = self.tanh(output)
        return output, hidden

    def initHidden(self, requires_grad = False):
        if self.is_bidirection:
            w = torch.Tensor(self.n_layers*2, 1, self.hidden_size)
            init.xavier_uniform(w, gain=init.calculate_gain('relu'))
            result = nn.Parameter(w, requires_grad = requires_grad)

            # result = Variable(torch.zeros(self.n_layers*2, 1, self.hidden_size), requires_grad = requires_grad)
        else:
            w = torch.Tensor(self.n_layers, 1, self.hidden_size)
            init.xavier_uniform(w, gain=init.calculate_gain('relu'))
            result = nn.Parameter(w, requires_grad = requires_grad)

            # result = Variable(torch.zeros(self.n_layers, 1, self.hidden_size), requires_grad = requires_grad)
        self.register_parameter('result', result)
        if USE_CUDA:
            return result.cuda()
        else:
            return result







# # test DecoderRNN_step
# seq_len = 5
# hidden_size = 4
# input_size = 4
#
# decoder = DecoderRNN_step(input_size=input_size, hidden_size=hidden_size, embedding_size=10, is_bidirection=False)
# # hidden = decoder.initHidden()
# hidden = Variable(torch.rand(1,1,hidden_size)).cuda()
#
# print('hidden -- 0', hidden)
# input_seqence = Variable(torch.rand(1,seq_len,hidden_size)).cuda()
#
# loss_f = nn.BCELoss()
#
# output = input_seqence[:,0,:]
# print('input -- 0', output)
# loss = 0
# for i in range(seq_len-1):
#     output, hidden = decoder(output,hidden)
#     print('output -- '+str(i), output)
#     print('hidden -- '+str(i), hidden)
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