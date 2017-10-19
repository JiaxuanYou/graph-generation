from model import *
from data import *
import numpy as np
import copy
from torch.optim.lr_scheduler import MultiStepLR
import matplotlib.pyplot as plt


def make_graph(node, hop_list, max_degree, key):
    '''

    :param node:
    :param hop_list: from hop_1 to hop_k
    :param max_degree:
    :return:
    '''
    G=nx.Graph()
    # nodes that are ready to have neighbors
    node_start = node
    start_list = [node]
    for hop in hop_list:
        for idx, node in enumerate(start_list):
            start_idx = idx * max_degree
            end_idx = (idx+1) * max_degree
            G.add_node(node)
            for i in range(start_idx,end_idx):
                G.add_node(hop[i])
                G.add_edge(node, hop[i])

        start_list = hop.cpu().numpy().tolist()
    # remove the dummy node
    G.remove_node(0)
    # print(G.edges())

    node_color = []
    nodes = list(G.nodes())
    for i in range(G.number_of_nodes()):
        if nodes[i]!= node_start:
            node_color.append('black')
        else:
            node_color.append('red')
    plt.switch_backend('agg')
    options = {
        'node_color': node_color,
        'node_size': 10,
        'width': 1,
    }
    plt.figure()
    plt.subplot()
    nx.draw_networkx(G, **options)
    plt.savefig('figures/graph_generated_'+key+'.png', dpi = 300)
    plt.close()
    return G




def train(args, dataset, encoder, decoder):
    torch.manual_seed(args.seed)
    train_loader = torch.utils.data.DataLoader(dataset,
        batch_size=args.batch_size, shuffle=args.shuffle, num_workers=args.num_workers)
    # optimizer = optim.SGD(list(encoder.parameters())+list(decoder.parameters()), lr=args.lr, momentum=args.momentum)
    optimizer = optim.Adam(list(encoder.parameters())+list(decoder.parameters()), lr=args.lr)
    scheduler = MultiStepLR(optimizer, milestones=args.milestones, gamma=0.5)
    for epoch in range(args.epochs):
        train_epoch(epoch, args, encoder, decoder, train_loader, optimizer, scheduler)
        if epoch%args.epochs_log == 0:
            test_epoch(epoch, args, encoder, decoder, train_loader)



def train_epoch(epoch, args, encoder, decoder, data_loader, optimizer, scheduler):
    encoder.train()
    decoder.train()
    pid = os.getpid()
    train_loss = 0
    correct_hop1 = 0
    correct_hop2 = 0
    correct_hop3 = 0
    total_hop1 = 0
    total_hop2 = 0
    total_hop3 = 0
    pred_max_average_hop1 = 0
    pred_max_average_hop2 = 0
    pred_max_average_hop3 = 0

    counter = 0

    # freeze previous layers
    if args.freeze == True:
        if epoch==args.milestones[0]:
            optimizer = torch.optim.Adam([
                {'params': decoder.deconv1_1.parameters(),'lr': args.lr/10},
                {'params': decoder.deconv1_2.parameters(),'lr': args.lr/10},
                {'params': decoder.deconv1_3.parameters(),'lr': args.lr/10},
                {'params': decoder.deconv2_1.parameters()},
                {'params': decoder.deconv2_2.parameters()},
                {'params': decoder.deconv2_3.parameters()},
                {'params': decoder.deconv3_1.parameters()},
                {'params': decoder.deconv3_2.parameters()},
                {'params': decoder.deconv3_3.parameters()},
                {'params': decoder.bn1_1.parameters(),'lr': args.lr/10},
                {'params': decoder.bn1_2.parameters(),'lr': args.lr/10},
                {'params': decoder.bn2_1.parameters()},
                {'params': decoder.bn2_2.parameters()},
                {'params': decoder.bn3_1.parameters()},
                {'params': decoder.bn3_2.parameters()},

                {'params': encoder.linear_3_0.parameters(),'lr': args.lr/10},
                {'params': encoder.linear_3_1.parameters(),'lr': args.lr/10},
                {'params': encoder.linear_3_2.parameters(),'lr': args.lr/10},
                {'params': encoder.linear_2_0.parameters(),'lr': args.lr/10},
                {'params': encoder.linear_2_1.parameters(),'lr': args.lr/10},
                {'params': encoder.linear_1_0.parameters(),'lr': args.lr/10},
                {'params': encoder.linear_0_0.parameters(),'lr': args.lr/10},
                {'params': encoder.linear.parameters(),'lr': args.lr/10},
                {'params': encoder.linear_projection.parameters(),'lr': args.lr/10},
                {'params': encoder.bn_3_0.parameters(),'lr': args.lr/10},
                {'params': encoder.bn_3_1.parameters(),'lr': args.lr/10},
                {'params': encoder.bn_3_2.parameters(),'lr': args.lr/10},
                {'params': encoder.bn_2_0.parameters(),'lr': args.lr/10},
                {'params': encoder.bn_2_1.parameters(),'lr': args.lr/10},
                {'params': encoder.bn_1_0.parameters(),'lr': args.lr/10},
                {'params': encoder.bn_0_0.parameters(),'lr': args.lr/10},
                {'params': encoder.bn.parameters(),'lr': args.lr/10},
            ], lr=args.lr)


            print('freeze!')

        if epoch==args.milestones[1]:
            optimizer = torch.optim.Adam([
                {'params': decoder.deconv1_1.parameters(),'lr': args.lr/10},
                {'params': decoder.deconv1_2.parameters(),'lr': args.lr/10},
                {'params': decoder.deconv1_3.parameters(),'lr': args.lr/10},
                {'params': decoder.deconv2_1.parameters(),'lr': args.lr/10},
                {'params': decoder.deconv2_2.parameters(),'lr': args.lr/10},
                {'params': decoder.deconv2_3.parameters(),'lr': args.lr/10},
                {'params': decoder.deconv3_1.parameters()},
                {'params': decoder.deconv3_2.parameters()},
                {'params': decoder.deconv3_3.parameters()},
                {'params': decoder.bn1_1.parameters(),'lr': args.lr/10},
                {'params': decoder.bn1_2.parameters(),'lr': args.lr/10},
                {'params': decoder.bn2_1.parameters(),'lr': args.lr/10},
                {'params': decoder.bn2_2.parameters(),'lr': args.lr/10},
                {'params': decoder.bn3_1.parameters()},
                {'params': decoder.bn3_2.parameters()},

                {'params': encoder.linear_3_0.parameters(),'lr': args.lr/10},
                {'params': encoder.linear_3_1.parameters(),'lr': args.lr/10},
                {'params': encoder.linear_3_2.parameters(),'lr': args.lr/10},
                {'params': encoder.linear_2_0.parameters(),'lr': args.lr/10},
                {'params': encoder.linear_2_1.parameters(),'lr': args.lr/10},
                {'params': encoder.linear_1_0.parameters(),'lr': args.lr/10},
                {'params': encoder.linear_0_0.parameters(),'lr': args.lr/10},
                {'params': encoder.linear.parameters(),'lr': args.lr/10},
                {'params': encoder.linear_projection.parameters(),'lr': args.lr/10},
                {'params': encoder.bn_3_0.parameters(),'lr': args.lr/10},
                {'params': encoder.bn_3_1.parameters(),'lr': args.lr/10},
                {'params': encoder.bn_3_2.parameters(),'lr': args.lr/10},
                {'params': encoder.bn_2_0.parameters(),'lr': args.lr/10},
                {'params': encoder.bn_2_1.parameters(),'lr': args.lr/10},
                {'params': encoder.bn_1_0.parameters(),'lr': args.lr/10},
                {'params': encoder.bn_0_0.parameters(),'lr': args.lr/10},
                {'params': encoder.bn.parameters(),'lr': args.lr/10},
            ], lr=args.lr)


            print('freeze!')


    # # freeze previous layers
    # if epoch == 20:
    #     optimizer = torch.optim.Adam([
    #         {'params': decoder.deconv.parameters()},
    #         {'params': decoder.deconv_out.parameters()},
    #         {'params': decoder.bn.parameters()},
    #
    #         {'params': encoder.linear_3_0.parameters(), 'lr': args.lr / 10},
    #         {'params': encoder.linear_3_1.parameters(), 'lr': args.lr / 10},
    #         {'params': encoder.linear_3_2.parameters(), 'lr': args.lr / 10},
    #         {'params': encoder.linear_2_0.parameters(), 'lr': args.lr / 10},
    #         {'params': encoder.linear_2_1.parameters(), 'lr': args.lr / 10},
    #         {'params': encoder.linear_1_0.parameters(), 'lr': args.lr / 10},
    #         {'params': encoder.linear_0_0.parameters(), 'lr': args.lr / 10},
    #         {'params': encoder.linear.parameters(), 'lr': args.lr / 10},
    #         {'params': encoder.linear_projection.parameters(), 'lr': args.lr / 10},
    #         {'params': encoder.bn_3_0.parameters(), 'lr': args.lr / 10},
    #         {'params': encoder.bn_3_1.parameters(), 'lr': args.lr / 10},
    #         {'params': encoder.bn_3_2.parameters(), 'lr': args.lr / 10},
    #         {'params': encoder.bn_2_0.parameters(), 'lr': args.lr / 10},
    #         {'params': encoder.bn_2_1.parameters(), 'lr': args.lr / 10},
    #         {'params': encoder.bn_1_0.parameters(), 'lr': args.lr / 10},
    #         {'params': encoder.bn_0_0.parameters(), 'lr': args.lr / 10},
    #         {'params': encoder.bn.parameters(), 'lr': args.lr / 10},
    #     ], lr=args.lr)
    #
    #     print('freeze!')
    #
    # if epoch == 50:
    #     optimizer = torch.optim.Adam([
    #         {'params': decoder.deconv1_1.parameters(), 'lr': args.lr / 10},
    #         {'params': decoder.bn1_1.parameters(), 'lr': args.lr / 10},
    #
    #         {'params': encoder.linear_3_0.parameters(), 'lr': args.lr / 10},
    #         {'params': encoder.linear_3_1.parameters(), 'lr': args.lr / 10},
    #         {'params': encoder.linear_3_2.parameters(), 'lr': args.lr / 10},
    #         {'params': encoder.linear_2_0.parameters(), 'lr': args.lr / 10},
    #         {'params': encoder.linear_2_1.parameters(), 'lr': args.lr / 10},
    #         {'params': encoder.linear_1_0.parameters(), 'lr': args.lr / 10},
    #         {'params': encoder.linear_0_0.parameters(), 'lr': args.lr / 10},
    #         {'params': encoder.linear.parameters(), 'lr': args.lr / 10},
    #         {'params': encoder.linear_projection.parameters(), 'lr': args.lr / 10},
    #         {'params': encoder.bn_3_0.parameters(), 'lr': args.lr / 10},
    #         {'params': encoder.bn_3_1.parameters(), 'lr': args.lr / 10},
    #         {'params': encoder.bn_3_2.parameters(), 'lr': args.lr / 10},
    #         {'params': encoder.bn_2_0.parameters(), 'lr': args.lr / 10},
    #         {'params': encoder.bn_2_1.parameters(), 'lr': args.lr / 10},
    #         {'params': encoder.bn_1_0.parameters(), 'lr': args.lr / 10},
    #         {'params': encoder.bn_0_0.parameters(), 'lr': args.lr / 10},
    #         {'params': encoder.bn.parameters(), 'lr': args.lr / 10},
    #     ], lr=args.lr)
    #
    #     print('freeze!')





    for batch_idx, data in enumerate(data_loader):
        _,node_idx = torch.max(data['node_list'][3][0], 1)
        if node_idx[0] in range(args.start_idx,args.end_idx):
            continue


        x_real_hop1 = Variable(data['node_list_pad'][2]).cuda(CUDA)
        x_real_hop1 = x_real_hop1.view(x_real_hop1.size(1), x_real_hop1.size(2))
        x_real_hop2 = Variable(data['node_list_pad'][1]).cuda(CUDA)
        x_real_hop2 = x_real_hop2.view(x_real_hop2.size(1), x_real_hop2.size(2))
        x_real_hop3 = Variable(data['node_list_pad'][0]).cuda(CUDA)
        x_real_hop3 = x_real_hop3.view(x_real_hop3.size(1), x_real_hop3.size(2))
        # print(x_real_hop1.size(), x_real_hop2.size(), x_real_hop3.size())

        y = encoder(data['node_list'], data['node_count_list'])
        x_pred_hop1, x_pred_hop2, x_pred_hop3 = decoder(y)
        x_pred_hop1 = x_pred_hop1.view(x_pred_hop1.size(2), x_pred_hop1.size(1))
        x_pred_hop2 = x_pred_hop2.view(x_pred_hop2.size(2), x_pred_hop2.size(1))
        x_pred_hop3 = x_pred_hop3.view(x_pred_hop3.size(2), x_pred_hop3.size(1))

        x_pred_hop1 = F.softmax(x_pred_hop1)
        x_pred_hop2 = F.softmax(x_pred_hop2)
        x_pred_hop3 = F.softmax(x_pred_hop3)

        alpha = 5
        loss_hop1 = F.binary_cross_entropy(x_pred_hop1, x_real_hop1, weight=x_real_hop1*alpha)
        loss_hop2 = F.binary_cross_entropy(x_pred_hop2, x_real_hop2, weight=x_real_hop2*alpha)
        loss_hop3 = F.binary_cross_entropy(x_pred_hop3, x_real_hop3, weight=x_real_hop3*alpha)

        if epoch<args.milestones[0]:
            loss = loss_hop1
        elif epoch<args.milestones[1]:
            loss = loss_hop1+loss_hop2
        else:
            loss = loss_hop1+loss_hop2+loss_hop3

        # loss = loss_hop1 + loss_hop2 + loss_hop3


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        train_loss += loss

        _, real_hop1 = torch.max(x_real_hop1, 1)
        pred_max_hop1, pred_hop1 = torch.max(x_pred_hop1, 1)
        _, real_hop2 = torch.max(x_real_hop2, 1)
        pred_max_hop2, pred_hop2 = torch.max(x_pred_hop2, 1)
        _, real_hop3 = torch.max(x_real_hop3, 1)
        pred_max_hop3, pred_hop3 = torch.max(x_pred_hop3, 1)

        # if epoch%10 ==0:
        #     print('node_idx[0]', node_idx[0],'real', real_hop2.view(1,-1)[0:1,0:15])
        #     print('node_idx[0]', node_idx[0],'pred', pred_hop2.view(1,-1)[0:1,0:15])

        # if epoch%20 ==0:
        #     # make graphs
        #     print('node_idx[0]', node_idx[0],'real')
        #     make_graph(node_idx[0], [real_hop1.data, real_hop2.data, real_hop3.data] , max_degree = 9, key='real')
        #     print('node_idx[0]', node_idx[0],'pred')
        #     make_graph(node_idx[0], [pred_hop1.data, pred_hop2.data, pred_hop3.data] , max_degree = 9, key='pred')


        # only select positive samples
        pred_hop1 = pred_hop1[real_hop1 != 0]
        real_hop1 = real_hop1[real_hop1 != 0]
        pred_hop2 = pred_hop2[real_hop2 != 0]
        real_hop2 = real_hop2[real_hop2 != 0]
        pred_hop3 = pred_hop3[real_hop3 != 0]
        real_hop3 = real_hop3[real_hop3 != 0]



        correct_hop1 += torch.eq(real_hop1, pred_hop1).sum().long()
        correct_hop2 += torch.eq(real_hop2, pred_hop2).sum().long()
        correct_hop3 += torch.eq(real_hop3, pred_hop3).sum().long()

        # if epoch%10 ==0:
        #     print('Selected','node_idx[0]', node_idx[0],'real', real_hop2.view(1,-1))
        #     print('Selected','node_idx[0]', node_idx[0],'pred', pred_hop2.view(1,-1))
        #     print('Step', torch.eq(real_hop2, pred_hop2).sum())
        #     print('Totoal', correct_hop2)


        total_hop1 += real_hop1.size(0)
        total_hop2 += real_hop2.size(0)
        total_hop3 += real_hop3.size(0)

        pred_max_average_hop1 += pred_max_hop1
        pred_max_average_hop2 += pred_max_hop2
        pred_max_average_hop3 += pred_max_hop3

        counter += 1

    train_loss /= counter
    pred_max_average_hop1 /= counter
    pred_max_average_hop2 /= counter
    pred_max_average_hop3 /= counter
    correct_hop1 = correct_hop1.data[0]
    correct_hop2 = correct_hop2.data[0]
    correct_hop3 = correct_hop3.data[0]

    accuracy_hop1 = float(correct_hop1 / total_hop1)
    accuracy_hop2 = float(correct_hop2 / total_hop2)
    accuracy_hop3 = float(correct_hop3 / total_hop3)

    print('{}\tTrain Epoch: {} Loss: {:.4f} Accuracy_hop1: {}/{} {:.4f} Accuracy_hop2: {}/{} {:.4f} Accuracy_hop3: {}/{} {:.4f}'.format(
        pid, epoch, train_loss.data[0], correct_hop1, total_hop1, accuracy_hop1, correct_hop2, total_hop2, accuracy_hop2, correct_hop3, total_hop3, accuracy_hop3))
    log_value('input_size'+str(args.input_size)+'train_loss', train_loss.data[0],epoch)
    log_value('input_size'+str(args.input_size)+'train_accuracy_hop1', accuracy_hop1, epoch)
    log_value('input_size'+str(args.input_size)+'train_pred_max_hop1', pred_max_average_hop1.data[0], epoch)
    log_value('input_size'+str(args.input_size)+'train_accuracy_hop2', accuracy_hop2, epoch)
    log_value('input_size'+str(args.input_size)+'train_pred_max_hop2', pred_max_average_hop2.data[0], epoch)
    log_value('input_size'+str(args.input_size)+'train_accuracy_hop3', accuracy_hop3, epoch)
    log_value('input_size'+str(args.input_size)+'train_pred_max_hop3', pred_max_average_hop3.data[0], epoch)


def test_epoch(epoch, args, encoder, decoder, data_loader):
    # the eval() mode seems to be not stable
    encoder.train()
    decoder.train()
    pid = os.getpid()
    test_loss = 0
    correct_hop1 = 0
    correct_hop2 = 0
    correct_hop3 = 0
    total_hop1 = 0
    total_hop2 = 0
    total_hop3 = 0
    pred_max_average_hop1 = 0
    pred_max_average_hop2 = 0
    pred_max_average_hop3 = 0


    counter = 0
    for batch_idx, data in enumerate(data_loader):
        _, node_idx = torch.max(data['node_list'][3][0], 1)
        if node_idx[0] not in range(args.start_idx, args.end_idx):
            continue

        x_real_hop1 = Variable(data['node_list_pad'][2]).cuda(CUDA)
        x_real_hop1 = x_real_hop1.view(x_real_hop1.size(1), x_real_hop1.size(2))
        x_real_hop2 = Variable(data['node_list_pad'][1]).cuda(CUDA)
        x_real_hop2 = x_real_hop2.view(x_real_hop2.size(1), x_real_hop2.size(2))
        x_real_hop3 = Variable(data['node_list_pad'][0]).cuda(CUDA)
        x_real_hop3 = x_real_hop3.view(x_real_hop3.size(1), x_real_hop3.size(2))

        y = encoder(data['node_list'], data['node_count_list'])
        # if epoch%20 == 0:
        #     print('node_idx', node_idx[0], 'size of embedding', y.size())
        #     print('node_idx', node_idx[0], 'embedding', y.view(1,-1))
        x_pred_hop1, x_pred_hop2, x_pred_hop3 = decoder(y)
        x_pred_hop1 = x_pred_hop1.view(x_pred_hop1.size(2), x_pred_hop1.size(1))
        x_pred_hop2 = x_pred_hop2.view(x_pred_hop2.size(2), x_pred_hop2.size(1))
        x_pred_hop3 = x_pred_hop3.view(x_pred_hop3.size(2), x_pred_hop3.size(1))

        x_pred_hop1 = F.softmax(x_pred_hop1)
        x_pred_hop2 = F.softmax(x_pred_hop2)
        x_pred_hop3 = F.softmax(x_pred_hop3)

        alpha = 5
        loss_hop1 = F.binary_cross_entropy(x_pred_hop1, x_real_hop1, weight=x_real_hop1 * alpha)
        loss_hop2 = F.binary_cross_entropy(x_pred_hop2, x_real_hop2, weight=x_real_hop2 * alpha)
        loss_hop3 = F.binary_cross_entropy(x_pred_hop3, x_real_hop3, weight=x_real_hop3 * alpha)
        loss = loss_hop1 + loss_hop2 + loss_hop3


        test_loss += loss

        _, real_hop1 = torch.max(x_real_hop1, 1)
        pred_max_hop1, pred_hop1 = torch.max(x_pred_hop1, 1)
        _, real_hop2 = torch.max(x_real_hop2, 1)
        pred_max_hop2, pred_hop2 = torch.max(x_pred_hop2, 1)
        _, real_hop3 = torch.max(x_real_hop3, 1)
        pred_max_hop3, pred_hop3 = torch.max(x_pred_hop3, 1)

        # plot
        if args.plot ==True:
            if epoch%20 ==0:
                if node_idx[0] == 95:
                    # make graphs
                    # print('node_idx[0]', node_idx[0],'real')
                    make_graph(node_idx[0], [real_hop1.data, real_hop2.data, real_hop3.data] , max_degree = 9, key='real'+str(epoch))
                    # print('node_idx[0]', node_idx[0],'pred')
                    make_graph(node_idx[0], [pred_hop1.data, pred_hop2.data, pred_hop3.data] , max_degree = 9, key='pred'+str(epoch))
                    print('plotted!')

        # only select positive samples
        pred_hop1 = pred_hop1[real_hop1 != 0]
        real_hop1 = real_hop1[real_hop1 != 0]
        pred_hop2 = pred_hop2[real_hop2 != 0]
        real_hop2 = real_hop2[real_hop2 != 0]
        pred_hop3 = pred_hop3[real_hop3 != 0]
        real_hop3 = real_hop3[real_hop3 != 0]

        correct_hop1 += torch.eq(real_hop1, pred_hop1).sum().long()
        correct_hop2 += torch.eq(real_hop2, pred_hop2).sum().long()
        correct_hop3 += torch.eq(real_hop3, pred_hop3).sum().long()

        total_hop1 += real_hop1.size(0)
        total_hop2 += real_hop2.size(0)
        total_hop3 += real_hop3.size(0)

        pred_max_average_hop1 += pred_max_hop1
        pred_max_average_hop2 += pred_max_hop2
        pred_max_average_hop3 += pred_max_hop3

        counter += 1

    test_loss /= counter
    pred_max_average_hop1 /= counter
    pred_max_average_hop2 /= counter
    pred_max_average_hop3 /= counter
    correct_hop1 = correct_hop1.data[0]
    correct_hop2 = correct_hop2.data[0]
    correct_hop3 = correct_hop3.data[0]

    accuracy_hop1 = float(correct_hop1 / total_hop1)
    accuracy_hop2 = float(correct_hop2 / total_hop2)
    accuracy_hop3 = float(correct_hop3 / total_hop3)


    print('{}\tTest Loss: {:.4f} Accuracy_hop1: {}/{} {:.4f} Accuracy_hop2: {}/{} {:.4f} Accuracy_hop3: {}/{} {:.4f}'.format(
            pid, test_loss.data[0], correct_hop1, total_hop1, accuracy_hop1, correct_hop2, total_hop2,
            accuracy_hop2, correct_hop3, total_hop3, accuracy_hop3))

    log_value('input_size'+str(args.input_size)+'test_loss', test_loss.data[0], epoch)
    log_value('input_size'+str(args.input_size)+'test_accuracy_hop1', accuracy_hop1, epoch)
    log_value('input_size'+str(args.input_size)+'test_pred_max_hop1', pred_max_average_hop1.data[0], epoch)
    log_value('input_size'+str(args.input_size)+'test_accuracy_hop2', accuracy_hop2, epoch)
    log_value('input_size'+str(args.input_size)+'test_pred_max_hop2', pred_max_average_hop2.data[0], epoch)
    log_value('input_size'+str(args.input_size)+'test_accuracy_hop3', accuracy_hop3, epoch)
    log_value('input_size'+str(args.input_size)+'test_pred_max_hop3', pred_max_average_hop3.data[0], epoch)
