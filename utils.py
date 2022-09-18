# --------------------------------------
# -*- coding: utf-8 -*- 
# @Time : 2022/9/18 9:41 
# @Author : wzy 
# @File : utils.py
# ---------------------------------------
import torch


def save_state(model, best_acc, flag='T'):
    print('==> Saving model ...')
    state = {
        'best_acc': best_acc,
        'state_dict': model.state_dict(),
    }
    for key in list(state['state_dict'].keys()):
        if 'module' in key:
            state['state_dict'][key.replace('module.', '')] = \
                state['state_dict'].pop(key)
    torch.save(state, './pts/model_' + flag + '_best.pth')


def adjust_learning_rate(optimizer, epoch):
    update_list = [140, 200, 250]
    if epoch in update_list:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1
    return
