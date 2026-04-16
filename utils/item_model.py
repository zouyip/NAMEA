import timm
import torch
import numpy as np
import torch.nn as nn
import tensorflow as tf
from torchvision import transforms

import args


args = args.getArgs()
mm = timm.list_models('*mobile*')
# print(mm)
dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


# model_list = timm.list_models('*v2*')
# print(model_list)


def exrators(root='./', item_model=None):
    item_model_list = []
    if item_model is None:  # 'densenet201', 'resnet152', 'densenet169', 'resnet50', 'densenet121', 'vgg16', 'inception_v3', 'inception_v4'
        item_model = ['inception_v3', 'deit_tiny_patch16_224', 'vit_tiny_patch16_224',  'resnet18']
    for name in item_model:
        model = (timm.create_model(
            name,
            pretrained=True,
            num_classes=1000,
        ).cuda())
        if 'vit_' in name or 'inc' in name or 'bit' in name:
            model = torch.nn.Sequential(transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), model)
        else:
            model = torch.nn.Sequential(transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)), model)
        item_model_list.append(model)
    return item_model_list
