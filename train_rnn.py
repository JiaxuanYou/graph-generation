from model import *
from dataset import *
import numpy as np
import copy

def train(dataset, decoder, optimizer, epoch_num, lr, weight_decay = 1e-5, batch_size=1, shuffle=True, num_workers = 1, run = 0, multi_target = False):
    # define dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    softmax = nn.Softmax()
    tanh = nn.Tanh()
    # loss_f = nn.BCELoss()
    loss_f = nn.MSELoss()
    # loss_f = nn.KLDivLoss()

    # filter parameters
    params = []
    for param in decoder.state_dict():
        if param!='hidden':
            params.append(decoder.state_dict()[param])

    # define optimizer
    if optimizer == "adam":
        optimizer = torch.optim.Adam([
            {'params':decoder.hidden, 'lr': lr*10},
            {'params':decoder.gru.parameters()},
            {'params': decoder.linear.parameters()},
            {'params': decoder.embedding.parameters()},
        ], lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.SGD(decoder.parameters(), lr=lr, weight_decay=weight_decay)
    # for param in decoder.parameters():
    #     print(param.size())
    # print(decoder.state_dict())
    # for param in decoder.state_dict():
    #     print(param, type(param))
    #     print(decoder.state_dict()[param])

    hidden_first = decoder.hidden
    print('hidden requires grad', hidden_first.requires_grad)
    # train
    for epoch in range(epoch_num):
        loss_summary = 0
        count = 0
        if epoch==80:
            lr /= 10
        for idx, nodes in enumerate(dataloader):
            # if idx>0:
            #     continue
            # sample an input
            input = Variable(nodes['nodes']).cuda()
            # print('input', input, 'idx', idx)
            input_embedding = decoder.embedding(input)
            # input_embedding don't need grad
            input_embedding = Variable(input_embedding.data).cuda()
            assert input_embedding.requires_grad == False
            seq_len = input_embedding.size(1)

            # Now input_embedding is [SOS, node, node's neighbour, EOS]
            # first hidden is the node itself's embedding, id = 1
            # hidden_first = Variable(input_embedding[:, 1, :].dataset, requires_grad = True).cuda()

            # # uncomment if want bi-directional net
            # # preprocessing (do softmax first)
            # hidden_first = hidden_first.view(1, hidden_first.size(0), hidden_first.size(1))
            # hidden = torch.cat((hidden, hidden), dim=0)
            # assert hidden.requires_grad

            # Calculate loss
            loss_total = 0
            # first input is SOS_token, just name it "output"
            input_first = input_embedding[:, 1, :]
            for i in range(seq_len - 2):
                if i==0:
                    output, hidden = decoder(input_first, hidden_first)
                else:
                    output, hidden = decoder(output.cuda(), hidden.cuda())
                # fist prediction should be the node's first neighbour, id = 2
                target_all = []
                if i < seq_len-3:
                    # target = input_embedding[:, i + 2, :].detach()
                    if multi_target:
                        # try multiple targets
                        for j in range(i, seq_len-3):
                            target = input_embedding[:, j + 2, :]
                            target_all.append(target)
                    else:
                        target = input_embedding[:, i + 2, :]
                        target_all.append(target)

                elif i == seq_len-3:
                    # target = input_embedding[:, -1, :].detach()
                    target = input_embedding[:, -1, :]
                    target_all.append(target)
                assert target.requires_grad == False

                loss_temp = []
                for target in target_all:
                    loss = loss_f(output, target)
                    loss_temp.append(loss)
                # print('loss temp', loss_temp)
                loss_temp = torch.stack(loss_temp, dim=0)
                loss_total += torch.min(loss_temp)

                # if epoch % 10 == 0:
                    # do evaluation
                    # print(str(i)+'output', output.dataset, 'target', target.dataset, 'diff',
                    #       output.dataset - target.dataset, 'loss', loss, 'loss_total', loss_total)

                    # display
                    # print(str(i) + 'output', output.dataset, 'target', target.dataset, 'diff',
                    #       output.dataset - target.dataset, 'loss_total', loss_total)

                    # print('embedding', decoder.embedding(Variable(torch.LongTensor(range(decoder.embedding_size))).cuda()).cpu().dataset)
                    # print(softmax(input_embedding[0]))

            optimizer.zero_grad()
            loss_total.backward()
            # print(hidden.requires_grad)
            # print('hidden_grad', decoder.gru.weight_ih_l0.grad.dataset)
            optimizer.step()
            # put the optimized hidden_first back
            # input_embedding[:, 1, :].dataset =hidden_first.dataset

            loss_summary += loss_total
            count+=1
        # print('total loss', loss_summary.cpu().dataset[0]/(idx+1))
        log_value('Loss: lr = '+str(lr)+' hidden = '+str(decoder.hidden_size)+' run = '+str(run), loss_summary.cpu().data[0]/count, epoch)
        if epoch%10 == 0 and epoch!=0:
            print('epoch ', epoch, 'lr', lr, 'total loss', loss_summary.cpu().data[0]/count, 'hidden size', decoder.hidden_size, 'run', run,
                  'multi-target', multi_target, 'shuffle_neighbor', dataset.shuffle_neighbour, 'hidden_first', decoder.hidden)
            # # do evaluation
            # print('output', output.cpu().dataset, 'target', target.cpu().dataset, 'diff', output.cpu().dataset-target.cpu().dataset)
            # # print('embedding', decoder.embedding(Variable(torch.LongTensor(range(decoder.embedding_size))).cuda()).cpu().dataset)
            # print(softmax(input_embedding[0]))
            print('evaluation')
            match = 0
            for idx, nodes in enumerate(dataloader):
                input = Variable(nodes['nodes']).cuda()
                # print('input',input.dataset)
                input_embedding = decoder.embedding(input)
                # input_embedding don't need grad
                input_embedding = Variable(input_embedding.data).cuda()
                assert input_embedding.requires_grad == False
                seq_len = input_embedding.size(1)

                # Now input_embedding is [SOS, node, node's neighbour, EOS]
                # first hidden is the node itself's embedding, id = 1
                # hidden_first = Variable(input_embedding[:, 1, :].dataset, requires_grad=True).cuda()
                hidden_first = decoder.initHidden()

                # first input is SOS_token, just name it "output"
                input_first = input_embedding[:, 1, :]
                i = 0
                max = 30
                prediction_all = []
                while(True):
                    if i == 0:
                        output, hidden = decoder(input_first, hidden_first)
                    else:
                        output, hidden = decoder(output.cuda(), hidden.cuda())

                    input_all = Variable(torch.LongTensor(range(decoder.embedding_size))).cuda()
                    all_embedding = decoder.embedding(input_all)
                    diff = all_embedding - output.repeat(decoder.embedding_size,1)
                    diff_norm = torch.norm(diff,2,1)
                    min, prediction = torch.min(diff_norm, dim = 0)
                    # print('prediction', prediction.dataset[0], 'min', min.dataset[0])
                    prediction_all.append(prediction.data[0])
                    if prediction.data[0] == input.data[0,-1] or i>max:
                        # print('break')
                        break
                    i += 1
                prediction_all = torch.LongTensor(prediction_all).cuda().view(1, -1)
                # print('input', input.dataset, 'prediction', prediction_all)
                # see if the prediction match input.dataset[2:]
                if input.data.size(1)-2 == prediction_all.size(1):
                    if torch.sum(input.data[0, 2:] - prediction_all[0]) == 0:
                        match += 1
                        print('match!')
            if match == idx+1:
                print('********************************')
                print('all match! ', 'epoch = ', epoch)
                break



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



# # Define dataset loader
# dataloader = torch.utils.dataset.DataLoader(graphdataset, batch_size = 1, shuffle=True, num_workers = 1)
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
#     print('total loss', loss_total.cpu().dataset[0])
#     log_value('Loss', loss_total.cpu().dataset[0], epoch)






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
#     return loss.dataset[0], ec, dc