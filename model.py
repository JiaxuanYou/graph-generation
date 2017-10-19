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
import numpy as np


USE_CUDA = torch.cuda.is_available()
CUDA = 0

class CNN_decoder(nn.Module):
    def __init__(self, input_size, output_size, stride = 2):

        super(CNN_decoder, self).__init__()

        self.input_size = input_size
        self.output_size = output_size

        self.relu = nn.ReLU()
        self.deconv1_1 = nn.ConvTranspose1d(in_channels=int(self.input_size), out_channels=int(self.input_size/2), kernel_size=3, stride=stride)
        self.bn1_1 = nn.BatchNorm1d(int(self.input_size/2))
        self.deconv1_2 = nn.ConvTranspose1d(in_channels=int(self.input_size/2), out_channels=int(self.input_size/2), kernel_size=3, stride=stride)
        self.bn1_2 = nn.BatchNorm1d(int(self.input_size/2))
        self.deconv1_3 = nn.ConvTranspose1d(in_channels=int(self.input_size/2), out_channels=int(self.output_size), kernel_size=3, stride=1, padding=1)

        self.deconv2_1 = nn.ConvTranspose1d(in_channels=int(self.input_size/2), out_channels=int(self.input_size / 4), kernel_size=3, stride=stride)
        self.bn2_1 = nn.BatchNorm1d(int(self.input_size / 4))
        self.deconv2_2 = nn.ConvTranspose1d(in_channels=int(self.input_size / 4), out_channels=int(self.input_size/4), kernel_size=3, stride=stride)
        self.bn2_2 = nn.BatchNorm1d(int(self.input_size / 4))
        self.deconv2_3 = nn.ConvTranspose1d(in_channels=int(self.input_size / 4), out_channels=int(self.output_size), kernel_size=3, stride=1, padding=1)

        self.deconv3_1 = nn.ConvTranspose1d(in_channels=int(self.input_size / 4), out_channels=int(self.input_size / 8), kernel_size=3, stride=stride)
        self.bn3_1 = nn.BatchNorm1d(int(self.input_size / 8))
        self.deconv3_2 = nn.ConvTranspose1d(in_channels=int(self.input_size / 8), out_channels=int(self.input_size / 8), kernel_size=3, stride=stride)
        self.bn3_2 = nn.BatchNorm1d(int(self.input_size / 8))
        self.deconv3_3 = nn.ConvTranspose1d(in_channels=int(self.input_size / 8), out_channels=int(self.output_size), kernel_size=3, stride=1, padding=1)



        for m in self.modules():
            if isinstance(m, nn.ConvTranspose1d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                m.weight.data = init.xavier_uniform(m.weight.data, gain=nn.init.calculate_gain('relu'))
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()



    def forward(self, x):
        '''

        :param
        x: batch * channel * length
        :return:
        '''
        # hop1
        x = self.deconv1_1(x)
        x = self.bn1_1(x)
        x = self.relu(x)
        # print(x.size())
        x = self.deconv1_2(x)
        x = self.bn1_2(x)
        x = self.relu(x)
        # print(x.size())
        x_hop1 = self.deconv1_3(x)
        # print(x_hop1.size())

        # hop2
        x = self.deconv2_1(x)
        x = self.bn2_1(x)
        x = self.relu(x)
        # print(x.size())
        x = self.deconv2_2(x)
        x = self.bn2_2(x)
        x = self.relu(x)
        x_hop2 = self.deconv2_3(x)
        # print(x_hop2.size())

        # hop3
        x = self.deconv3_1(x)
        x = self.bn3_1(x)
        x = self.relu(x)
        # print(x.size())
        x = self.deconv3_2(x)
        x = self.bn3_2(x)
        x = self.relu(x)
        # print(x.size())
        x_hop3 = self.deconv3_3(x)
        # print(x_hop3.size())



        return x_hop1,x_hop2,x_hop3

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





class CNN_decoder_share(nn.Module):
    def __init__(self, input_size, output_size, stride = 2):

        super(CNN_decoder_share, self).__init__()

        self.input_size = input_size
        self.output_size = output_size

        self.relu = nn.ReLU()
        self.deconv = nn.ConvTranspose1d(in_channels=int(self.input_size), out_channels=int(self.input_size), kernel_size=3, stride=stride)
        self.bn = nn.BatchNorm1d(int(self.input_size))
        self.deconv_out = nn.ConvTranspose1d(in_channels=int(self.input_size), out_channels=int(self.output_size), kernel_size=3, stride=1, padding=1)

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose1d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                m.weight.data = init.xavier_uniform(m.weight.data, gain=nn.init.calculate_gain('relu'))
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()



    def forward(self, x):
        '''

        :param
        x: batch * channel * length
        :return:
        '''
        # hop1
        x = self.deconv(x)
        x = self.bn(x)
        x = self.relu(x)
        # print(x.size())
        x = self.deconv(x)
        x = self.bn(x)
        x = self.relu(x)
        # print(x.size())
        x_hop1 = self.deconv_out(x)
        # print(x_hop1.size())

        # hop2
        x = self.deconv(x)
        x = self.bn(x)
        x = self.relu(x)
        # print(x.size())
        x = self.deconv(x)
        x = self.bn(x)
        x = self.relu(x)
        x_hop2 = self.deconv_out(x)
        # print(x_hop2.size())

        # hop3
        x = self.deconv(x)
        x = self.bn(x)
        x = self.relu(x)
        # print(x.size())
        x = self.deconv(x)
        x = self.bn(x)
        x = self.relu(x)
        # print(x.size())
        x_hop3 = self.deconv_out(x)
        # print(x_hop3.size())



        return x_hop1,x_hop2,x_hop3



class CNN_decoder_attention(nn.Module):
    def __init__(self, input_size, output_size, stride=2):

        super(CNN_decoder_attention, self).__init__()

        self.input_size = input_size
        self.output_size = output_size

        self.relu = nn.ReLU()
        self.deconv = nn.ConvTranspose1d(in_channels=int(self.input_size), out_channels=int(self.input_size),
                                         kernel_size=3, stride=stride)
        self.bn = nn.BatchNorm1d(int(self.input_size))
        self.deconv_out = nn.ConvTranspose1d(in_channels=int(self.input_size), out_channels=int(self.output_size),
                                             kernel_size=3, stride=1, padding=1)
        self.deconv_attention = nn.ConvTranspose1d(in_channels=int(self.input_size), out_channels=int(self.input_size),
                                             kernel_size=1, stride=1, padding=0)

        for m in self.modules():
            if isinstance(m, nn.ConvTranspose1d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                m.weight.data = init.xavier_uniform(m.weight.data, gain=nn.init.calculate_gain('relu'))
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        '''

        :param
        x: batch * channel * length
        :return:
        '''
        # hop1
        x = self.deconv(x)
        x = self.bn(x)
        x = self.relu(x)

        x = self.deconv(x)
        x = self.bn(x)
        x = self.relu(x)

        x_hop1 = self.deconv_out(x)
        x_hop1_attention = self.deconv_attention(x)
        x_hop1_attention = torch.matmul(x_hop1_attention,
                                        x_hop1_attention.view(-1,x_hop1_attention.size(2),x_hop1_attention.size(1)))

        # print(x_hop1.size())

        # hop2
        x = self.deconv(x)
        x = self.bn(x)
        x = self.relu(x)

        x = self.deconv(x)
        x = self.bn(x)
        x = self.relu(x)

        x_hop2 = self.deconv_out(x)
        x_hop2_attention = self.deconv_attention(x)
        x_hop2_attention = torch.matmul(x_hop2_attention,
                                        x_hop2_attention.view(-1, x_hop2_attention.size(2), x_hop2_attention.size(1)))

        # print(x_hop2.size())

        # hop3
        x = self.deconv(x)
        x = self.bn(x)
        x = self.relu(x)

        x = self.deconv(x)
        x = self.bn(x)
        x = self.relu(x)

        x_hop3 = self.deconv_out(x)
        x_hop3_attention = self.deconv_attention(x)
        x_hop3_attention = torch.matmul(x_hop3_attention,
                                        x_hop3_attention.view(-1, x_hop3_attention.size(2), x_hop3_attention.size(1)))

        # print(x_hop3.size())



        return x_hop1, x_hop2, x_hop3, x_hop1_attention, x_hop2_attention, x_hop3_attention






#### test code ####
# x = Variable(torch.randn(1, 256, 1)).cuda(CUDA)
# decoder = CNN_decoder(256, 16).cuda(CUDA)
# y = decoder(x)

class Encoder(nn.Module):
    def __init__(self, feature_size, input_size, layer_num):
        super(Encoder, self).__init__()

        self.linear_projection = nn.Linear(feature_size, input_size)

        self.input_size = input_size

        # linear for hop 3
        self.linear_3_0 = nn.Linear(input_size*(2 ** 0), input_size*(2 ** 1))
        self.linear_3_1 = nn.Linear(input_size*(2 ** 1), input_size*(2 ** 2))
        self.linear_3_2 = nn.Linear(input_size*(2 ** 2), input_size*(2 ** 3))
        # linear for hop 2
        self.linear_2_0 = nn.Linear(input_size * (2 ** 0), input_size * (2 ** 1))
        self.linear_2_1 = nn.Linear(input_size * (2 ** 1), input_size * (2 ** 2))
        # linear for hop 1
        self.linear_1_0 = nn.Linear(input_size * (2 ** 0), input_size * (2 ** 1))
        # linear for hop 0
        self.linear_0_0 = nn.Linear(input_size * (2 ** 0), input_size * (2 ** 1))

        self.linear = nn.Linear(input_size*(2+2+4+8), input_size*(16))


        self.bn_3_0 = nn.BatchNorm1d(self.input_size * (2 ** 1))
        self.bn_3_1 = nn.BatchNorm1d(self.input_size * (2 ** 2))
        self.bn_3_2 = nn.BatchNorm1d(self.input_size * (2 ** 3))

        self.bn_2_0 = nn.BatchNorm1d(self.input_size * (2 ** 1))
        self.bn_2_1 = nn.BatchNorm1d(self.input_size * (2 ** 2))

        self.bn_1_0 = nn.BatchNorm1d(self.input_size * (2 ** 1))

        self.bn_0_0 = nn.BatchNorm1d(self.input_size * (2 ** 1))

        self.bn = nn.BatchNorm1d(input_size*(16))

        self.relu = nn.ReLU()
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = init.xavier_uniform(m.weight.data,gain=nn.init.calculate_gain('relu'))
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self, nodes_list, nodes_count_list):
        '''

        :param nodes: a list, each element n_i is a tensor for node's k-i hop neighbours
                (the first nodes_hop is the furthest neighbor)
                where n_i = N * num_neighbours * features
               nodes_count: a list, each element is a list that show how many neighbours belongs to the father node
        :return:
        '''


        # 3-hop feature
        # nodes original features to representations
        nodes_list[0] = Variable(nodes_list[0]).cuda(CUDA)
        nodes_list[0] = self.linear_projection(nodes_list[0])
        nodes_features = self.linear_3_0(nodes_list[0])
        nodes_features = self.bn_3_0(nodes_features.view(-1,nodes_features.size(2),nodes_features.size(1)))
        nodes_features = nodes_features.view(-1,nodes_features.size(2),nodes_features.size(1))
        nodes_features = self.relu(nodes_features)
        # nodes count from previous hop
        nodes_count = nodes_count_list[0]
        # print(nodes_count,nodes_count.size())
        # aggregated representations placeholder, feature dim * 2
        nodes_features_farther = Variable(torch.Tensor(nodes_features.size(0), nodes_count.size(1), nodes_features.size(2))).cuda(CUDA)
        i = 0
        for j in range(nodes_count.size(1)):
            # mean pooling for each father node
            # print(nodes_count[:,j][0],type(nodes_count[:,j][0]))
            nodes_features_farther[:,j,:] = torch.mean(nodes_features[:, i:i+int(nodes_count[:,j][0]), :], 1, keepdim = False)
            i += int(nodes_count[:,j][0])
        # assign node_features
        nodes_features = nodes_features_farther
        nodes_features = self.linear_3_1(nodes_features)
        nodes_features = self.bn_3_1(nodes_features.view(-1,nodes_features.size(2),nodes_features.size(1)))
        nodes_features = nodes_features.view(-1,nodes_features.size(2),nodes_features.size(1))
        nodes_features = self.relu(nodes_features)
        # nodes count from previous hop
        nodes_count = nodes_count_list[1]
        # aggregated representations placeholder, feature dim * 2
        nodes_features_farther = Variable(torch.Tensor(nodes_features.size(0), nodes_count.size(1), nodes_features.size(2))).cuda(CUDA)
        i = 0
        for j in range(nodes_count.size(1)):
            # mean pooling for each father node
            nodes_features_farther[:,j,:] = torch.mean(nodes_features[:, i:i+int(nodes_count[:,j][0]), :], 1, keepdim = False)
            i += int(nodes_count[:,j][0])
        # assign node_features
        nodes_features = nodes_features_farther
        # print('nodes_feature',nodes_features.size())
        nodes_features = self.linear_3_2(nodes_features)
        nodes_features = self.bn_3_2(nodes_features.view(-1,nodes_features.size(2),nodes_features.size(1)))
        nodes_features = nodes_features.view(-1,nodes_features.size(2),nodes_features.size(1))
        # nodes_features = self.relu(nodes_features)
        # nodes count from previous hop
        nodes_features_hop_3 = torch.mean(nodes_features, 1, keepdim=True)
        # print(nodes_features_hop_3.size())

        # 2-hop feature
        # nodes original features to representations
        nodes_list[1] = Variable(nodes_list[1]).cuda(CUDA)
        nodes_list[1] = self.linear_projection(nodes_list[1])
        nodes_features = self.linear_2_0(nodes_list[1])
        nodes_features = self.bn_2_0(nodes_features.view(-1,nodes_features.size(2),nodes_features.size(1)))
        nodes_features = nodes_features.view(-1,nodes_features.size(2),nodes_features.size(1))
        nodes_features = self.relu(nodes_features)
        # nodes count from previous hop
        nodes_count = nodes_count_list[1]
        # aggregated representations placeholder, feature dim * 2
        nodes_features_farther = Variable(torch.Tensor(nodes_features.size(0), nodes_count.size(1), nodes_features.size(2))).cuda(CUDA)
        i = 0
        for j in range(nodes_count.size(1)):
            # mean pooling for each father node
            nodes_features_farther[:,j,:] = torch.mean(nodes_features[:, i:i+int(nodes_count[:,j][0]), :], 1, keepdim = False)
            i += int(nodes_count[:,j][0])
        # assign node_features
        nodes_features = nodes_features_farther
        nodes_features = self.linear_2_1(nodes_features)
        nodes_features = self.bn_2_1(nodes_features.view(-1,nodes_features.size(2),nodes_features.size(1)))
        nodes_features = nodes_features.view(-1,nodes_features.size(2),nodes_features.size(1))
        # nodes_features = self.relu(nodes_features)
        # nodes count from previous hop
        nodes_features_hop_2 = torch.mean(nodes_features, 1, keepdim=True)
        # print(nodes_features_hop_2.size())


        # 1-hop feature
        # nodes original features to representations
        nodes_list[2] = Variable(nodes_list[2]).cuda(CUDA)
        nodes_list[2] = self.linear_projection(nodes_list[2])
        nodes_features = self.linear_1_0(nodes_list[2])
        nodes_features = self.bn_1_0(nodes_features.view(-1,nodes_features.size(2),nodes_features.size(1)))
        nodes_features = nodes_features.view(-1,nodes_features.size(2),nodes_features.size(1))
        # nodes_features = self.relu(nodes_features)
        # nodes count from previous hop
        nodes_features_hop_1 = torch.mean(nodes_features, 1, keepdim=True)
        # print(nodes_features_hop_1.size())


        # own feature
        nodes_list[3] = Variable(nodes_list[3]).cuda(CUDA)
        nodes_list[3] = self.linear_projection(nodes_list[3])
        nodes_features = self.linear_0_0(nodes_list[3])
        nodes_features = self.bn_0_0(nodes_features.view(-1, nodes_features.size(2), nodes_features.size(1)))
        nodes_features_hop_0 = nodes_features.view(-1, nodes_features.size(2), nodes_features.size(1))
        # print(nodes_features_hop_0.size())



        # concatenate
        nodes_features = torch.cat((nodes_features_hop_0, nodes_features_hop_1, nodes_features_hop_2, nodes_features_hop_3),dim=2)
        nodes_features = self.linear(nodes_features)
        # nodes_features = self.bn(nodes_features.view(-1,nodes_features.size(2),nodes_features.size(1)))
        nodes_features = nodes_features.view(-1,nodes_features.size(2),nodes_features.size(1))
        # print(nodes_features.size())
        return(nodes_features)







class Encoder_share(nn.Module):
    def __init__(self, feature_size, input_size, layer_num):
        super(Encoder, self).__init__()

        self.linear_projection = nn.Linear(feature_size, input_size)

        self.input_size = input_size

        # linear for hop 3
        self.linear = nn.Linear(input_size * 2, input_size)

        self.bn = nn.BatchNorm1d(input_size)

        self.relu = nn.ReLU()
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = init.xavier_uniform(m.weight.data, gain=nn.init.calculate_gain('relu'))
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, nodes_list, nodes_count_list):
        '''

        :param nodes: a list, each element n_i is a tensor for node's k-i hop neighbours
                (the first nodes_hop is the furthest neighbor)
                where n_i = N * num_neighbours * features
               nodes_count: a list, each element is a list that show how many neighbours belongs to the father node
        :return:
        '''

        # 3-hop feature
        # nodes original features to representations
        nodes_list[0] = Variable(nodes_list[0]).cuda(CUDA)
        nodes_list[0] = self.linear_projection(nodes_list[0]) # to match input dimension
        nodes_list[0] = torch.cat((nodes_list[0],nodes_list[0]),2)
        nodes_features = self.linear(nodes_list[0])
        nodes_features = self.bn(nodes_features.view(-1, nodes_features.size(2), nodes_features.size(1)))
        nodes_features = nodes_features.view(-1, nodes_features.size(2), nodes_features.size(1))
        nodes_features = self.relu(nodes_features)
        # nodes count from previous hop
        nodes_count = nodes_count_list[0]
        # print(nodes_count,nodes_count.size())
        # aggregated representations placeholder, feature dim * 2
        nodes_features_farther = Variable(
            torch.Tensor(nodes_features.size(0), nodes_count.size(1), nodes_features.size(2))).cuda(CUDA)
        i = 0
        for j in range(nodes_count.size(1)):
            # mean pooling for each father node
            # print(nodes_count[:,j][0],type(nodes_count[:,j][0]))
            nodes_features_farther[:, j, :] = torch.mean(nodes_features[:, i:i + int(nodes_count[:, j][0]), :], 1,
                                                         keepdim=False)
            i += int(nodes_count[:, j][0])
        # assign node_features
        nodes_features = nodes_features_farther
        nodes_list[1] = Variable(nodes_list[1]).cuda(CUDA)
        nodes_list[1] = self.linear_projection(nodes_list[1])
        nodes_list[1] = torch.cat((nodes_list[0], nodes_list[0]), 2)
        nodes_features = self.linear(nodes_list[0])
        nodes_features = self.bn(nodes_features.view(-1, nodes_features.size(2), nodes_features.size(1)))
        nodes_features = nodes_features.view(-1, nodes_features.size(2), nodes_features.size(1))
        nodes_features = self.relu(nodes_features)
        # nodes count from previous hop
        nodes_count = nodes_count_list[0]
        # print(nodes_count,nodes_count.size())
        # aggregated representations placeholder, feature dim * 2
        nodes_features_farther = Variable(
            torch.Tensor(nodes_features.size(0), nodes_count.size(1), nodes_features.size(2))).cuda(CUDA)
        i = 0
        for j in range(nodes_count.size(1)):
            # mean pooling for each father node
            # print(nodes_count[:,j][0],type(nodes_count[:,j][0]))
            nodes_features_farther[:, j, :] = torch.mean(nodes_features[:, i:i + int(nodes_count[:, j][0]), :], 1,
                                                         keepdim=False)
            i += int(nodes_count[:, j][0])




        nodes_features = self.linear_3_1(nodes_features)
        nodes_features = self.bn_3_1(nodes_features.view(-1, nodes_features.size(2), nodes_features.size(1)))
        nodes_features = nodes_features.view(-1, nodes_features.size(2), nodes_features.size(1))
        nodes_features = self.relu(nodes_features)
        # nodes count from previous hop
        nodes_count = nodes_count_list[1]
        # aggregated representations placeholder, feature dim * 2
        nodes_features_farther = Variable(
            torch.Tensor(nodes_features.size(0), nodes_count.size(1), nodes_features.size(2))).cuda(CUDA)
        i = 0
        for j in range(nodes_count.size(1)):
            # mean pooling for each father node
            nodes_features_farther[:, j, :] = torch.mean(nodes_features[:, i:i + int(nodes_count[:, j][0]), :], 1,
                                                         keepdim=False)
            i += int(nodes_count[:, j][0])
        # assign node_features
        nodes_features = nodes_features_farther
        # print('nodes_feature',nodes_features.size())
        nodes_features = self.linear_3_2(nodes_features)
        nodes_features = self.bn_3_2(nodes_features.view(-1, nodes_features.size(2), nodes_features.size(1)))
        nodes_features = nodes_features.view(-1, nodes_features.size(2), nodes_features.size(1))
        # nodes_features = self.relu(nodes_features)
        # nodes count from previous hop
        nodes_features_hop_3 = torch.mean(nodes_features, 1, keepdim=True)
        # print(nodes_features_hop_3.size())

        # 2-hop feature
        # nodes original features to representations
        nodes_list[1] = Variable(nodes_list[1]).cuda(CUDA)
        nodes_list[1] = self.linear_projection(nodes_list[1])
        nodes_features = self.linear_2_0(nodes_list[1])
        nodes_features = self.bn_2_0(nodes_features.view(-1, nodes_features.size(2), nodes_features.size(1)))
        nodes_features = nodes_features.view(-1, nodes_features.size(2), nodes_features.size(1))
        nodes_features = self.relu(nodes_features)
        # nodes count from previous hop
        nodes_count = nodes_count_list[1]
        # aggregated representations placeholder, feature dim * 2
        nodes_features_farther = Variable(
            torch.Tensor(nodes_features.size(0), nodes_count.size(1), nodes_features.size(2))).cuda(CUDA)
        i = 0
        for j in range(nodes_count.size(1)):
            # mean pooling for each father node
            nodes_features_farther[:, j, :] = torch.mean(nodes_features[:, i:i + int(nodes_count[:, j][0]), :], 1,
                                                         keepdim=False)
            i += int(nodes_count[:, j][0])
        # assign node_features
        nodes_features = nodes_features_farther
        nodes_features = self.linear_2_1(nodes_features)
        nodes_features = self.bn_2_1(nodes_features.view(-1, nodes_features.size(2), nodes_features.size(1)))
        nodes_features = nodes_features.view(-1, nodes_features.size(2), nodes_features.size(1))
        # nodes_features = self.relu(nodes_features)
        # nodes count from previous hop
        nodes_features_hop_2 = torch.mean(nodes_features, 1, keepdim=True)
        # print(nodes_features_hop_2.size())


        # 1-hop feature
        # nodes original features to representations
        nodes_list[2] = Variable(nodes_list[2]).cuda(CUDA)
        nodes_list[2] = self.linear_projection(nodes_list[2])
        nodes_features = self.linear_1_0(nodes_list[2])
        nodes_features = self.bn_1_0(nodes_features.view(-1, nodes_features.size(2), nodes_features.size(1)))
        nodes_features = nodes_features.view(-1, nodes_features.size(2), nodes_features.size(1))
        # nodes_features = self.relu(nodes_features)
        # nodes count from previous hop
        nodes_features_hop_1 = torch.mean(nodes_features, 1, keepdim=True)
        # print(nodes_features_hop_1.size())


        # own feature
        nodes_list[3] = Variable(nodes_list[3]).cuda(CUDA)
        nodes_list[3] = self.linear_projection(nodes_list[3])
        nodes_features = self.linear_0_0(nodes_list[3])
        nodes_features = self.bn_0_0(nodes_features.view(-1, nodes_features.size(2), nodes_features.size(1)))
        nodes_features_hop_0 = nodes_features.view(-1, nodes_features.size(2), nodes_features.size(1))
        # print(nodes_features_hop_0.size())



        # concatenate
        nodes_features = torch.cat(
            (nodes_features_hop_0, nodes_features_hop_1, nodes_features_hop_2, nodes_features_hop_3), dim=2)
        nodes_features = self.linear(nodes_features)
        # nodes_features = self.bn(nodes_features.view(-1,nodes_features.size(2),nodes_features.size(1)))
        nodes_features = nodes_features.view(-1, nodes_features.size(2), nodes_features.size(1))
        # print(nodes_features.size())
        return (nodes_features)







#### test code ####
# embedding_size = 4
# # test
# x0 = Variable(torch.randn(1, 1, embedding_size)).cuda(CUDA)
# x1 = Variable(torch.randn(1, 4, embedding_size)).cuda(CUDA)
# x2 = Variable(torch.randn(1, 12, embedding_size)).cuda(CUDA)
# x3 = Variable(torch.randn(1, 36, embedding_size)).cuda(CUDA)
# print(x3, x2, x1, x0)
# node_list = [x3, x2, x1, x0]
# count1 = torch.Tensor([3]).repeat(4)
# count2 = torch.Tensor([3]).repeat(12)
# node_count_list = [count2, count1]
#
# encoder = Encoder(embedding_size, 3).cuda(CUDA)
# y = encoder(node_list, node_count_list)
# # print(y)
#
# decoder = CNN_decoder(y.size(1), embedding_size).cuda(CUDA)
# x = decoder(y)
# print(x)









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
            return result.cuda(CUDA)
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
            return result.cuda(CUDA)
        else:
            return result





##### reference code #####




# # test DecoderRNN_step
# seq_len = 5
# hidden_size = 4
# input_size = 4
#
# decoder = DecoderRNN_step(input_size=input_size, hidden_size=hidden_size, embedding_size=10, is_bidirection=False)
# # hidden = decoder.initHidden()
# hidden = Variable(torch.rand(1,1,hidden_size)).cuda(CUDA)
#
# print('hidden -- 0', hidden)
# input_seqence = Variable(torch.rand(1,seq_len,hidden_size)).cuda(CUDA)
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
#             return result.cuda(CUDA)
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
#             return result.cuda(CUDA)
#         else:
#             return result
#
#
# encoder = EncoderRNN(16, 64, 1).cuda(CUDA)
# decoder = DecoderRNN(16, 64)
# input_dummy = Variable(torch.rand(1,30,16)).cuda(CUDA)
# hidden = encoder.initHidden()
# output, hidden = encoder(input_dummy,hidden)
# print('output', output.size())
# print('hidden', hidden.size())