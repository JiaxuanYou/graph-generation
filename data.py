import torch
import torchvision as tv
import torch.nn as nn
from torch.autograd import Variable
import networkx as nx
from model import *



class GraphDataset(torch.utils.data.Dataset):
    """Graph Dataset"""
    def __init__(self, G):
        self.G = G
    def __len__(self):
        return len(self.G.nodes())
    def __getitem__(self, idx):
        li = list(self.G.adj[idx])
        li.insert(0,idx)
        sample = {'nodes':torch.LongTensor(li)}
        return sample

# Generate Graph
G = nx.karate_club_graph()
graphdataset = GraphDataset(G)

# Initialize Embedding
hidden_size = 4
embedding = nn.Embedding(100, hidden_size)

# Define data loader
dataloader = torch.utils.data.DataLoader(graphdataset, batch_size = 1, shuffle=True, num_workers = 1)

# Show usage
for idx, nodes in enumerate(dataloader):
    # print('sample '+str(idx), Variable(nodes['nodes']))
    input = Variable(nodes['nodes'])
    input_embedding = embedding(input)
    print(embedding(input),embedding(input).size())


# test DecoderRNN_step
seq_len = 5
softmax = nn.Softmax()
decoder = DecoderRNN_step(input_size=hidden_size, hidden_size=hidden_size)

# the node itself's embedding
hidden = input_embedding[:,0,:]

hidden = softmax(hidden).view(1,hidden.size(0),hidden.size(1))
hidden = torch.cat((hidden,hidden),dim = 0)
hidden.requires_grad = True
print('hidden -- 0', hidden, torch.sum(hidden))
# input_seqence = Variable(torch.rand(1,seq_len,hidden_size))

loss_f = nn.BCELoss()

output = softmax(input_embedding[:,1,:])
loss = 0
for i in range(seq_len-1):
    output, hidden = decoder(output,hidden)
    print('output -- '+str(i), output, torch.sum(output))
    print('hidden -- '+str(i), hidden, torch.sum(hidden))
    loss += loss_f(output,input_embedding[:,i+2,:])
    print('loss -- '+str(i), loss_f(output,input_embedding[:,i+1,:]))
print('total loss', loss)

