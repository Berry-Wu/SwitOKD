# --------------------------------------
# -*- coding: utf-8 -*- 
# @Time : 2022/9/18 9:31 
# @Author : wzy 
# @File : loss_lr.py
# ---------------------------------------
import torch
import torch.nn.functional as F


def kl_div(p_logit, q_logit, T):
    p = F.softmax(p_logit / T, dim=-1)
    kl = torch.sum(p * (F.log_softmax(p_logit / T, dim=-1)
                        - F.log_softmax(q_logit / T, dim=-1)), 1)
    return torch.mean(kl)


def dist_label(y, q):
    q = F.softmax(q, dim=-1)
    dist = torch.sum(torch.abs(q - y), 1)

    return torch.mean(dist)


def dist_gap(p_logit, q_logit, T):
    p = F.softmax(p_logit / T, dim=-1)
    q = F.softmax(q_logit / T, dim=-1)
    dist = torch.sum(torch.abs(q - p), 1)

    return torch.mean(dist)


def lr_optim(model_t, model_s, lr):
    # define solver and criterion
    base_lr = float(lr)
    # Teacher
    param_dict_T = dict(model_t.named_parameters())
    params_T = []
    for key, value in param_dict_T.items():
        params_T += [{'params': [value], 'lr': base_lr, 'weight_decay': 1e-4}]
        optimizer_T = torch.optim.SGD(params_T, lr=lr, momentum=0.9, weight_decay=1e-4)

    # Student
    param_dict_S = dict(model_s.named_parameters())
    params_S = []
    for key, value in param_dict_S.items():
        params_S += [{'params': [value], 'lr': base_lr, 'weight_decay': 1e-4}]
        optimizer_S = torch.optim.SGD(params_S, lr=lr, momentum=0.9, weight_decay=1e-4)
    return optimizer_T, optimizer_S