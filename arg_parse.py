# --------------------------------------
# -*- coding: utf-8 -*- 
# @Time : 2022/9/18 14:42 
# @Author : wzy 
# @File : arg_parse.py
# ---------------------------------------
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="The hyper-parameter of SwinOKD")
    parser.add_argument('--cpu', action='store_true',
                        help='set if only CPU is available')
    parser.add_argument('--data', action='store', default='dataset-path',
                        help='dataset path')
    parser.add_argument('--arch', action='store', default='resnet',
                        help='the architecture for the network: nin')
    parser.add_argument('--lr', action='store', default=0.01,
                        help='the initial learning rate')
    parser.add_argument('--pretrained', action='store', default=None,
                        help='the path to the pretrained model')
    parser.add_argument('--evaluate', action='store_true',
                        help='evaluate the model')
    parser.add_argument('--epoch', default=5, help='the training epochs')
    args = parser.parse_args()
    return args
