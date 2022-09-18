# --------------------------------------
# -*- coding: utf-8 -*- 
# @Time : 2022/9/18 9:31 
# @Author : wzy 
# @File : train.py
# ---------------------------------------
import math

import numpy as np
import torch
import torchvision.models
from torch import nn

from datas import val_loader, train_loader
from utils import save_state, adjust_learning_rate
from loss_lr import dist_label, dist_gap, kl_div, lr_optim
from visual import draw
from arg_parse import parse_args

args = parse_args()


def test_T(mode='train'):
    global best_acc_T
    model_T.eval()
    test_loss = 0
    correct = 0

    for data, target in val_loader:
        data, target = data.cuda(), target.cuda()

        output = model_T(data)
        test_loss += criterion(output, target).item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    acc = 100. * correct / len(val_loader.dataset)

    if mode == 'train':
        if acc > best_acc_T:
            best_acc_T = acc
            save_state(model_T, best_acc_T, flag='T')

    test_loss /= len(val_loader.dataset)
    print('Test set: T: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        test_loss * 128., correct, len(val_loader.dataset),
        100. * correct / len(val_loader.dataset)))
    print('Best Accuracy: {:.2f}%\n'.format(best_acc_T))

    return acc, test_loss


def test_S(mode='train'):
    global best_acc_S
    model_S.eval()
    test_loss = 0
    correct = 0

    for data, target in val_loader:
        data, target = data.cuda(), target.cuda()

        output = model_S(data)
        test_loss += criterion(output, target).item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    acc = 100. * correct / len(val_loader.dataset)

    if mode == 'train':
        if acc > best_acc_S:
            best_acc_S = acc
            save_state(model_S, best_acc_S, flag='S')

    test_loss /= len(val_loader.dataset)
    print('Test set: S: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        test_loss * 128., correct, len(val_loader.dataset),
        100. * correct / len(val_loader.dataset)))
    print('Best Accuracy: {:.2f}%\n'.format(best_acc_S))

    return acc, test_loss


def train(epoch):
    model_T.train()
    model_S.train()

    for batch_idx, (data, target) in enumerate(train_loader):

        # forwarding
        target_onehot = torch.zeros(data.size()[0], 10).cuda().scatter_(1, target.view(target.size()[0], 1).cuda(), 1)
        data, target = data.cuda(), target.cuda()

        optimizer_T.zero_grad()
        output_T = model_T(data)

        optimizer_S.zero_grad()
        output_S = model_S(data)

        ps_y = dist_label(target_onehot, output_S.detach())
        pt_y = dist_label(target_onehot, output_T.detach())
        ps_pt = dist_gap(output_T.detach(), output_S.detach(), 1)

        # epsilon = torch.exp(pt_y / (pt_y + ps_y))  # 原本代码
        epsilon = torch.exp(-1 * (pt_y / (pt_y + ps_y)))  # 按照论文
        delta = ps_y - epsilon * pt_y

        if ps_pt > delta:
            loss_S = criterion(output_S, target) + kl_div(output_T.detach(), output_S, 1)
            loss_S.backward()
            optimizer_S.step()
            # print('\tmode --> expert')

        else:
            loss_T = criterion(output_T, target) + kl_div(output_S.detach(), output_T, 1)
            loss_S = criterion(output_S, target) + kl_div(output_T.detach(), output_S, 1)

            loss_T.backward()
            loss_S.backward()

            optimizer_T.step()
            optimizer_S.step()
            # print('\tmode --> learning')

        progress = math.ceil(batch_idx / len(train_loader) * 50)

        print('\rTrain epoch: [{}/{}] {}/{} [{}]{}% \tLoss: {:.6f}\tLR: {}'.format(epoch + 1, args.epoch,
                                                                                   batch_idx * len(data),
                                                                                   len(train_loader.dataset),
                                                                                   '-' * progress + '>', progress * 2,
                                                                                   loss_S.item(),
                                                                                   optimizer_S.param_groups[0][
                                                                                       'lr']), end='')
    return


if __name__ == '__main__':
    print('==> Options:', args)

    # set the seed
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)

    # define the model
    print('==> building model', args.arch, '...')
    if args.arch == 'resnet':
        model_T = torchvision.models.resnet34(num_classes=10)
        model_S = torchvision.models.resnet18(num_classes=10)
    else:
        raise Exception(args.arch + ' is currently not supported')

    # initialize the model
    if not args.pretrained:
        print('==> Initializing model parameters ...')
        best_acc_T = 0
        for m in model_T.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                c = float(m.weight.data[0].nelement())
                m.weight.data = m.weight.data.normal_(0, 1.0 / c)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data = m.weight.data.zero_().add(1.0)

        best_acc_S = 0
        for m in model_S.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                c = float(m.weight.data[0].nelement())
                m.weight.data = m.weight.data.normal_(0, 1.0 / c)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data = m.weight.data.zero_().add(1.0)
    else:
        print('==> Load pretrained model form', args.pretrained, '...')
        pretrained_model = torch.load(args.pretrained)
        best_acc_T = pretrained_model['best_acc']
        model_T.load_state_dict(pretrained_model['state_dict'])

    if not args.cpu:
        model_T.cuda()
        model_T = torch.nn.DataParallel(model_T, device_ids=range(torch.cuda.device_count()))

        model_S.cuda()
        model_S = torch.nn.DataParallel(model_S, device_ids=range(torch.cuda.device_count()))
    # print(model_S)

    optimizer_T, optimizer_S = lr_optim(model_T, model_S, args.lr)
    criterion = nn.CrossEntropyLoss()

    # start training
    teacher_history = []
    student_history = []
    temp = 0
    for epoch in range(0, args.epoch):
        adjust_learning_rate(optimizer_T, epoch)
        adjust_learning_rate(optimizer_S, epoch)

        train(epoch)
        print()

        acc_T, test_loss_T = test_T()
        acc_S, test_loss_S = test_S()

        teacher_history.append((acc_T, test_loss_T))
        student_history.append((acc_S, test_loss_S))

    teacher_history = np.array(torch.tensor(teacher_history, device='cpu'))
    student_history = np.array(torch.tensor(student_history, device='cpu'))

    # visual
    draw(teacher_history, student_history, args.epoch)
