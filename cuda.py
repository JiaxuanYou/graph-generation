import torch

# Specifies the device that will be used for 
# all tensor variables. Allows for unified cross
# platform solution.
#cuda_id = 0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")