from train import *
import numpy as np
import copy
import torch.multiprocessing as mp


# model configuration
hidden_size = 4 # hidden vector size (for a single layer)
embedding_size = 9 # the number of embedding vocabulary
n_layers = 1
# train configuration
lr = 0.01


# clean logging directory
if os.path.isdir("logs"):
    shutil.rmtree("logs")
configure("logs/logs_toy", flush_secs=1)

# clean saving directory
if not os.path.exists("saves"):
    os.makedirs("saves")

# Generate Graph
# G = nx.karate_club_graph()
G = nx.LCF_graph(6,[3,-3],3)
graphdataset = GraphDataset(G, shuffle_neighbour = False)

# Generate network

decoders = []
for lr in [0.3, 0.1, 0.03, 0.01, 0.003, 0.001]:
    for hidden_size in [8]:
        for run in range(1):
            decoder = DecoderRNN_step(input_size=hidden_size, hidden_size=hidden_size, embedding_size=embedding_size,
                                      n_layers=n_layers).cuda()
            train(graphdataset, decoder, optimizer='adam', epoch_num=2000, lr=lr, weight_decay=1e-5, batch_size=1, shuffle=True,
                  num_workers=1, run = run)
        # embedding = np.load('saves/embedding_lr_'+str(lr)+'.npy')



# def train_wrapper(num_processes, optimizer='adam', epoch_num=400, lr=lr, weight_decay=1e-5, batch_size=1, shuffle=True,
#                   num_workers=1, run = 1):
#     model = DecoderRNN_step(input_size=hidden_size, hidden_size=hidden_size, embedding_size=embedding_size,
#                             n_layers=n_layers).cuda()
#
#     processes = []
#     for rank in range(num_processes):
#         model = copy.deepcopy(model)
#         p = mp.Process(target=train, args=(graphdataset, model, optimizer, epoch_num, lr, weight_decay,
#                                            batch_size, shuffle, num_workers, run))
#         p.start()
#         processes.append(p)
#     for p in processes:
#       p.join()
#
#
# train_wrapper(1)