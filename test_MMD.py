import torch
import numpy as np
import time

def compute_kernel(x,y):
    x_size = x.size(0)
    y_size = y.size(0)
    dim = x.size(1)
    x_tile = x.view(x_size,1,dim)
    x_tile = x_tile.repeat(1,y_size,1)
    y_tile = y.view(1,y_size,dim)
    y_tile = y_tile.repeat(x_size,1,1)
    return torch.exp(-torch.mean((x_tile-y_tile)**2,dim = 2)/float(dim))


def compute_mmd(x,y):
    x_kernel = compute_kernel(x,x)
    # print(x_kernel)
    y_kernel = compute_kernel(y,y)
    # print(y_kernel)
    xy_kernel = compute_kernel(x,y)
    # print(xy_kernel)
    return torch.mean(x_kernel)+torch.mean(y_kernel)-2*torch.mean(xy_kernel)


# start = time.time()
# x = torch.randn(4000,1).cuda()
# y = torch.randn(4000,1).cuda()
# print(compute_mmd(x,y))
# end = time.time()
# print('GPU time:', end-start)


start = time.time()
torch.manual_seed(123)
batch = 1000
x = torch.randn(batch,1)
y_baseline = torch.randn(batch,1)
y_pred = torch.zeros(batch,1)

print('MMD baseline', compute_mmd(x,y_baseline))
print('MMD prediction', compute_mmd(x,y_pred))


#
# print('before',x)
# print('MMD', compute_mmd(x,y))
# x_idx = np.random.permutation(x.size(0))
# x = x[x_idx,:]
# print('after permutation',x)
# print('MMD', compute_mmd(x,y))
#
#
# end = time.time()
# print('CPU time:', end-start)