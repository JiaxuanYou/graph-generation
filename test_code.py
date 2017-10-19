import torch

a = torch.ones(2,3)
b = torch.ones(3,2)

a_batch = torch.ones(2,3,2,3)
b_batch = torch.ones(1,3,2)


print(torch.matmul(a,b))
print(torch.matmul(a_batch,b_batch))
print(a)