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

class Glosy(nn.Module):
    def __init__(self, drop_rate=0.2):
        super(Glosy, self).__init__()
        
        self.drop_rate = drop_rate
        self.conv1 = nn.Conv3d(1, 32, (3, 3, 3), stride=(2, 2, 1), padding=1)
        self.conv2 = nn.Conv3d(32, 32, (3, 3, 3), stride=(2, 2, 1), padding=1)
        self.conv3 = nn.Conv3d(32, 64, (3, 3, 3), stride=(1, 1, 1), padding=1)
        self.conv4 = nn.Conv3d(64, 128, (3, 3, 3), stride=(1, 1, 1), padding=1)
        self.conv5 = nn.Conv3d(128, 128, (3, 3, 3), stride=(1, 1, 1), padding=1)
        self.conv6 = nn.Conv3d(128, 128, (3, 3, 3), stride=(1, 1, 1), padding=1)
        self.bn1 = nn.BatchNorm3d(32)
        self.bn2 = nn.BatchNorm3d(64)
        self.bn3 = nn.BatchNorm3d(128)
        self.relu = nn.ReLU()
        self.maxpool1 = nn.MaxPool3d((2, 2, 2), stride=(2, 2, 2))
        self.maxpool2 = nn.MaxPool3d((2, 2, 1), stride=(2, 2, 1))
        self.dropout = nn.Dropout(p=drop_rate)
        self.avepool = nn.AvgPool3d(kernel_size=1, stride=1)
        self.linear = nn.Linear(4608, 2)
        
    def forward(self, img):
        hidden = self.conv1(img)
        hidden = self.relu(self.bn1(hidden))
        hidden = self.maxpool1(hidden)
        
        hidden = self.conv2(hidden)
        hidden = self.relu(self.bn1(hidden))
        hidden = self.maxpool1(hidden)
        
        hidden = self.conv3(hidden)
        hidden = self.relu(self.bn2(hidden))
        hidden = self.maxpool1(hidden)
        
        hidden = self.conv4(hidden)
        hidden = self.relu(self.bn3(hidden))
        hidden = self.maxpool2(hidden)
        
        hidden = self.conv5(hidden)
        hidden = self.conv6(hidden)
        hidden = self.relu(self.bn3(hidden))
        hidden = self.maxpool2(hidden)
        
        #hidden = self.dropout(hidden)
        hidden = self.avepool(hidden)
        
        out = hidden.view(hidden.size(0), -1)
        out = self.linear(out)
        
        return out
