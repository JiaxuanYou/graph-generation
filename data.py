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

if os.path.isdir("logs"):
    shutil.rmtree("logs")
configure("logs/logs_toy", flush_secs=1)

PAD_token = 0
SOS_token = 1
EOS_token = 2
NODE_token = 3
# so the first meaningful entry will become 3

class GraphDataset(torch.utils.data.Dataset):
    """Graph Dataset"""
    def __init__(self, G, shuffle_neighbour = False):
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

# Generate Graph
G = nx.karate_club_graph()
graphdataset = GraphDataset(G, shuffle_neighbour = False)

# Define data loader
dataloader = torch.utils.data.DataLoader(graphdataset, batch_size = 1, shuffle=True, num_workers = 1)

# Initialize Decoder network
hidden_size = 4
decoder = DecoderRNN_step(input_size=hidden_size, hidden_size=hidden_size, embedding_size=100)
softmax = nn.Softmax()
loss_f = nn.BCELoss()
optimizer = torch.optim.Adam(decoder.parameters(), lr=0.001)

# Show usage
input_embedding = 0
for epoch in range(1000):
    print('epoch ', epoch)
    for idx, nodes in enumerate(dataloader):

        # print('sample '+str(idx), Variable(nodes['nodes']))
        input = Variable(nodes['nodes'])
        input_embedding = decoder.embedding(input)
        # print(embedding(input),embedding(input).size())

        seq_len = input_embedding.size(1)
        # Now input_embedding is [SOS, node, node's neighbour, EOS]
        # first hidden is the node itself's embedding, id = 1
        hidden = input_embedding[:,1,:]
        # preprocessing (do softmax first)
        hidden = softmax(hidden).view(1,hidden.size(0),hidden.size(1))
        hidden = torch.cat((hidden,hidden),dim = 0)
        # hidden.requires_grad = True, but should define a new variable to do this
        # hidden = Variable(hidden.data,requires_grad = True)
        # print('hidden requires_grad', hidden.requires_grad)
        assert hidden.requires_grad
        assert input_embedding.requires_grad

        # print('hidden -- 0', hidden, torch.sum(hidden))

        # first input is SOS_token, just name it "output"
        output = softmax(input_embedding[:,0,:])
        loss_total = 0
        for i in range(seq_len-2):
            output, hidden = decoder(output,hidden)
            # print('output -- '+str(i), output, torch.sum(output))
            # print('hidden -- '+str(i), hidden, torch.sum(hidden))
            # fist prediction should be the node's first neighbour, id = 2
            target = input_embedding[:,i+2,:].detach()
            assert target.requires_grad == False
            loss = loss_f(output,target)
            # print('loss -- '+str(i), loss)
            loss_total += loss
        # print('total loss', loss_total.data[0])

        optimizer.zero_grad()
        loss_total.backward()
        optimizer.step()
    print('total loss', loss_total.data[0])
    log_value('Loss', loss_total.data[0], epoch)

