
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
import torch.nn.init as init

class GraphVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, output_dim):
        '''
        Args:
            input_dim: input feature dimension for node.
            hidden_dim: hidden dim for 2-layer gcn.
            latent_dim: dimension of the latent representation of graph.
            output_dim: output adj matrix dimension.
        '''
        super(GraphVAE, self).__init__()
        self.conv1 = GraphConv(input_dim=input_dim, output_dim=hidden_dim)
        self.conv2 = GraphConv(input_dim=hidden_dim, output_dim=output_dim)
        self.act = nn.ReLU()

        for m in self.modules():
            if isinstance(m, GraphConv):
                m.weight.data = init.xavier_uniform(m.weight.data, gain=nn.init.calculate_gain('relu'))
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x, adj, input_feature):
        x = self.conv1(x, adj)
        x = self.act(x)
        x = self.conv2(x, adj)

        out,_ = torch.max(x, dim=0, keepdim = True)
        return out


