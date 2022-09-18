# --------------------------------------
# -*- coding: utf-8 -*- 
# @Time : 2022/9/18 9:50 
# @Author : wzy 
# @File : datas.py
# ---------------------------------------
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms

transform_train = transforms.Compose([
    # transforms.RandomCrop(64, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_val = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

train_data = torchvision.datasets.CIFAR10(
    root='../data/',
    train=True,
    transform=transform_train,
    download=True
)

val_data = torchvision.datasets.CIFAR10(
    root='../data/',
    train=False,
    transform=transform_val,
    download=True
)

train_loader = DataLoader(dataset=train_data, batch_size=128, shuffle=True)
val_loader = DataLoader(dataset=val_data, batch_size=128, shuffle=False)
