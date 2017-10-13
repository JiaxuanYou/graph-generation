from train import *
import numpy as np
import copy
import torch.multiprocessing as mp
# from node2vec.src.main import *
import node2vec.src.main as nv



if __name__ == '__main__':

    # model configuration
    # hidden_size = 16 # hidden vector size for a single GRU layer
    input_size = 4  # embedding vector size for each node
    # embedding_size = 3+14 # the number of embedding vocabulary
    n_layers = 1
    # train configuration
    # lr = 0.01
    ############# node2vec config###############
    args = nv.config(dimension=input_size)
    ############################################

    # clean logging directory
    if os.path.isdir("logs"):
        shutil.rmtree("logs")
    # configure("logs/logs_toy", flush_secs=1)

    # clean saving directory
    if not os.path.exists("saves"):
        os.makedirs("saves")

    # Generate Graph
    # G = nx.karate_club_graph()
    G = nx.LCF_graph(14, [5, -5], 7)
    graphdataset = GraphDataset(G, shuffle_neighbour=False)
    # run node2vec
    for edge in G.edges():
        G[edge[0]][edge[1]]['weight'] = 1
    embedding = nv.node2vec_main(G, args)
    print(embedding.shape)

    embedding_size = embedding.shape[0] + 3

    embedding = torch.from_numpy(embedding).float().cuda()
    # normalize
    embedding = embedding / torch.mean(torch.norm(embedding, 2, 1))
    print(embedding)


    ##### parallel
    mp.set_start_method('spawn')
    decoder = DecoderRNN_step(input_size=input_size, hidden_size=64, embedding_size=embedding_size,
                                          n_layers=n_layers, is_bidirection=False, embedding_init_flag=True, embedding_init=embedding).cuda()


    optimizer = "adam"
    epoch_num = 100
    lr = 0.01

    print('****************test****************')
    processes = []
    for rank in range(5):
        decoder = copy.deepcopy(decoder)
        p = mp.Process(target=train, args=(graphdataset, decoder, optimizer, epoch_num, lr))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    # # Generate network
    # decoders = []
    # for lr in [0.01]:
    #     for hidden_size in [64]:
    #         for run in range(5):
    #             decoder = DecoderRNN_step(input_size=input_size, hidden_size=hidden_size, embedding_size=embedding_size,
    #                                       n_layers=n_layers, is_bidirection=False, embedding_init_flag=True, embedding_init=embedding).cuda()
    #             train(graphdataset, decoder, optimizer='adam', epoch_num=2000, lr=lr, weight_decay=1e-5, batch_size=1, shuffle=False,
    #                   num_workers=1, run = run, multi_target= False)
    #         # embedding = np.load('saves/embedding_lr_'+str(lr)+'.npy')



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