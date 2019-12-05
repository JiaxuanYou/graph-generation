import torch

# Specifies the device that will be used for 
# all tensor variables. Allows for unified cross
# platform solution.
cuda_id = 1
device = torch.device("cuda:{}".format(cuda_id) if torch.cuda.is_available() else "cpu")
