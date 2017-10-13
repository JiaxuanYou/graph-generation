import torch
import torchvision as tv
import torch.nn as nn
from torch.autograd import Variable
import networkx as nx
import random
from tensorboard_logger import configure, log_value
import shutil
import os
from model import *

# token config
PAD_token = 0
SOS_token = 1
EOS_token = 2
NODE_token = 3
# so the first meaningful entry will become 3

class GraphDataset(torch.utils.data.Dataset):
    """Graph Dataset"""
    def __init__(self, G, shuffle_neighbour = True):
        self.G = G
        self.shuffle_neighbour = shuffle_neighbour
    def __len__(self):
        return len(self.G.nodes())
    def __getitem__(self, idx):
        li = list(self.G.adj[idx])
        if self.shuffle_neighbour:
            random.shuffle(li)
        li_new = [x+NODE_token for x in li] # allow for tokens
        li_new.insert(0, idx+NODE_token) # node itself
        li_new.insert(0, SOS_token) # SOS
        li_new.append(EOS_token) # EOS
        sample = {'nodes':torch.LongTensor(li_new)}
        return sample



