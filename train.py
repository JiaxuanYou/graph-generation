from model import *
from data import *
import numpy as np

def train(dataset, decoder, optimizer, epoch_num, lr, weight_decay = 1e-5, batch_size=1, shuffle=True, num_workers = 1, run = 0):
    # define dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    softmax = nn.Softmax()
    loss_f = nn.BCELoss()
    # loss_f = nn.MSELoss()

    # define optimizer
    if optimizer == "adam":
        optimizer = torch.optim.Adam(decoder.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.SGD(decoder.parameters(), lr=lr, weight_decay=weight_decay)

    # train
    for epoch in range(epoch_num):
        loss_summary = 0
        count = 0
        for idx, nodes in enumerate(dataloader):
            # sample an input
            input = Variable(nodes['nodes']).cuda()
            input_embedding = decoder.embedding(input)
            # input_embedding don't need grad
            input_embedding = Variable(input_embedding.data).cuda()
            assert input_embedding.requires_grad == False
            seq_len = input_embedding.size(1)

            # Now input_embedding is [SOS, node, node's neighbour, EOS]
            # first hidden is the node itself's embedding, id = 1
            hidden_first = Variable(input_embedding[:, 1, :].data, requires_grad = True).cuda()
            # preprocessing (do softmax first)
            hidden = softmax(hidden_first).view(1, hidden_first.size(0), hidden_first.size(1))
            hidden = torch.cat((hidden, hidden), dim=0)
            assert hidden.requires_grad

            # Calculate loss
            loss_total = 0
            # first input is SOS_token, just name it "output"
            input_first = softmax(input_embedding[:, 0, :])
            for i in range(seq_len - 2):
                if i==0:
                    output, hidden = decoder(input_first, hidden)
                else:
                    output, hidden = decoder(output.cuda(), hidden.cuda())
                # fist prediction should be the node's first neighbour, id = 2
                if i < seq_len-3:
                    # target = input_embedding[:, i + 2, :].detach()
                    target = softmax(input_embedding[:, i + 2, :])
                elif i == seq_len-3:
                    # target = input_embedding[:, -1, :].detach()
                    target = softmax(input_embedding[:, -1, :])
                assert target.requires_grad == False
                loss = loss_f(output, target)
                loss_total += loss

                # do evaluation
                # print(str(i)+'output', output.cpu().data, 'target', target.cpu().data, 'diff',
                #       output.cpu().data - target.cpu().data)
                # # print('embedding', decoder.embedding(Variable(torch.LongTensor(range(decoder.embedding_size))).cuda()).cpu().data)
                # print(softmax(input_embedding[0]))

            optimizer.zero_grad()
            loss_total.backward()
            optimizer.step()
            # put the optimized hidden_first back
            input_embedding[:, 1, :].data = hidden_first.data

            loss_summary += loss_total
            count+=1
        # print('total loss', loss_summary.cpu().data[0]/(idx+1))
        log_value('Loss: lr = '+str(lr)+' hidden = '+str(decoder.hidden_size)+' run = '+str(run), loss_summary.cpu().data[0]/count, epoch)
        if epoch%10 == 0:
            print('epoch ', epoch, 'lr', lr, 'total loss', loss_summary.cpu().data[0]/count, 'hidden size', decoder.hidden_size, 'run', run)
            # # do evaluation
            # print('output', output.cpu().data, 'target', target.cpu().data, 'diff', output.cpu().data-target.cpu().data)
            # # print('embedding', decoder.embedding(Variable(torch.LongTensor(range(decoder.embedding_size))).cuda()).cpu().data)
            # print(softmax(input_embedding[0]))
    # save
    # save embedding
    np.save('saves/embedding_lr_'+str(lr)+'_hidden_'+str(decoder.hidden_size)+'_run_'+str(run)+'.npy',decoder.embedding(Variable(torch.LongTensor(range(decoder.embedding_size))).cuda()).cpu().data.numpy())
    print('embedding saved')



# # model configuration
# hidden_size = 4  # hidden vector size (for a single layer)
# embedding_size = 100  # the number of embedding vocabulary
# n_layers = 1
# # train configuration
# lr = 0.001
#
# if os.path.isdir("logs"):
#     shutil.rmtree("logs")
# configure("logs/logs_toy", flush_secs=1)
#
# # Generate Graph
# G = nx.karate_club_graph()
# graphdataset = GraphDataset(G, shuffle_neighbour=False)



# # Define data loader
# dataloader = torch.utils.data.DataLoader(graphdataset, batch_size = 1, shuffle=True, num_workers = 1)
#
# # Initialize Decoder network
# decoder = DecoderRNN_step(input_size=hidden_size, hidden_size=hidden_size, embedding_size=embedding_size, n_layers=n_layers).cuda()
# softmax = nn.Softmax()
# loss_f = nn.BCELoss()
# optimizer = torch.optim.Adam(decoder.parameters(), lr=0.001)
#
# # Show usages
# for epoch in range(10000):
#     print('epoch ', epoch)
#     for idx, nodes in enumerate(dataloader):
#         input = Variable(nodes['nodes']).cuda()
#         input_embedding = decoder.embedding(input)
#
#         seq_len = input_embedding.size(1)
#         # Now input_embedding is [SOS, node, node's neighbour, EOS]
#         # first hidden is the node itself's embedding, id = 1
#         hidden = input_embedding[:,1,:]
#         # preprocessing (do softmax first)
#         hidden = softmax(hidden).view(1,hidden.size(0),hidden.size(1))
#         hidden = torch.cat((hidden,hidden),dim = 0)
#         # hidden.requires_grad = True
#         assert hidden.requires_grad
#
#         # first input is SOS_token, just name it "output"
#         output = softmax(input_embedding[:,0,:])
#         loss_total = 0
#         for i in range(seq_len-2):
#             output, hidden = decoder(output.cuda(),hidden.cuda())
#             # fist prediction should be the node's first neighbour, id = 2
#             target = input_embedding[:,i+2,:].detach()
#             assert target.requires_grad == False
#             loss = loss_f(output,target)
#             loss_total += loss
#
#         optimizer.zero_grad()
#         loss_total.backward()
#         optimizer.step()
#     print('total loss', loss_total.cpu().data[0])
#     log_value('Loss', loss_total.cpu().data[0], epoch)






    ## reference code
# def train(input, encoder, decoder, encoder_optimizer,
#           decoder_optimizer, criterion, max_length=MAX_LENGTH, clip = 50):
#     # Zero gradients of both optimizers
#     encoder_optimizer.zero_grad()
#     decoder_optimizer.zero_grad()
#     loss = 0  # Added onto for each word
#
#
#     # Run words through encoder
#     encoder_hidden = encoder.initHidden()
#     encoder_outputs, encoder_hidden = encoder(input, encoder_hidden)
#
#     # Prepare input and output variables
#     decoder_input = Variable(torch.LongTensor([SOS_token]))
#     decoder_hidden = encoder_hidden  # Use last hidden state from encoder
#
#
#     # Move new Variables to CUDA
#     if USE_CUDA:
#         decoder_input = decoder_input.cuda()
#         all_decoder_outputs = all_decoder_outputs.cuda()
#
#     # Run through decoder one time step at a time
#     for t in range(input.length()):
#         decoder_output, decoder_hidden = decoder(
#             decoder_input, decoder_hidden, encoder_outputs
#         )
#
#         all_decoder_outputs[t] = decoder_output
#         decoder_input = target_batches[t]  # Next input is current target
#
#     # Loss calculation and backpropagation
#     loss = masked_cross_entropy(
#         all_decoder_outputs.transpose(0, 1).contiguous(),  # -> batch x seq
#         target_batches.transpose(0, 1).contiguous(),  # -> batch x seq
#         target_lengths
#     )
#     loss.backward()
#
#     # Clip gradient norms
#     ec = torch.nn.utils.clip_grad_norm(encoder.parameters(), clip)
#     dc = torch.nn.utils.clip_grad_norm(decoder.parameters(), clip)
#
#     # Update parameters with optimizers
#     encoder_optimizer.step()
#     decoder_optimizer.step()
#
#     return loss.data[0], ec, dc