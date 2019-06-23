import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import h5py
from Augmentation import RandomCrop, CenterCrop, RandomFlip
from torch.autograd import Variable

cfg = {
    'glosy':[32, 'M2', 32, 'M', 64, 'M', 128, 'M2', 128, 128, 'M2'],
    'VGG9': [64, 'M', 128, 'M', 256, 'M', 512, 'M', 512, 512, 'M'],
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):

    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(9216, 2)


    def forward(self, x):
        out = self.features(x)
        #print(out.shape)
        out = out.view(out.size(0), -1)
        out = F.relu(self.classifier(out))
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 1
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))]
            elif x == 'M2':
                layers += [nn.MaxPool3d(kernel_size=(2, 2, 1), stride=(2, 2, 1))]
            elif x == 32:
                layers += [nn.Conv3d(in_channels, x, kernel_size=(3, 3, 3), stride = (2, 2,1), padding=1),
                           nn.BatchNorm3d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
            else:
                layers += [nn.Conv3d(in_channels, x, kernel_size=(3, 3, 3), stride = (1, 1,1), padding=1),
                           nn.BatchNorm3d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool3d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)
