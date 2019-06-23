import os
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import h5py
from Augmentation import RandomCrop, CenterCrop, RandomFlip
from dataloader import OAI_Dataloader
from VGG import VGG

no_cuda = False
log_interval = 1
cuda = not no_cuda and torch.cuda.is_available()
seed = 1
torch.manual_seed(seed)
device = torch.device("cuda" if cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}
learning_rate = 0.00005
num_epochs = 100

csv_root_dir = '/home/hg1196/OA_TKR_imaging/VGG_pytorch/'
root_dir = '/gpfs/data/denizlab/Datasets/OAI/SAG_IW_TSE/'
train_csv_file = 'VGG_train.csv'
val_csv_file = 'VGG_val.csv'

train_params = {'dim': (384,384,32),
          'normalize' : True,
          'randomCrop' : True,
          'randomFlip' : True,
          'flipProbability' : -1,
          'cropDim' : (384,384,32)}

val_params = {'dim': (384,384,32),
          'normalize' : True,
          'randomCrop' : False,
          'randomFlip' : False,
          'flipProbability' : -1,
          'cropDim' : (384,384,32)}

def test_model(loader, model):
    correct = 0
    total = 0
    model.eval()
    for im, labels in loader:
        im, labels = im.to(device), labels.to(device)
        outputs = F.softmax(model(im), dim=1)
        loss = criterion(outputs, labels)
        predicted = outputs.max(1, keepdim=True)[1]

        total += labels.size(0)
        correct += predicted.eq(labels.view_as(predicted)).sum().item()
    return (100 * correct / total), loss

model = VGG('glosy').to(device)
train_accuracy_500_vgg = []
train_accuracy_epoch_vgg = []
train_loss = []
val_accuracy_500_vgg = []
val_accuracy_epoch_vgg = []
val_loss = []
criterion = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


train_dataset = OAI_Dataloader(root_dir, csv_root_dir, train_csv_file, **train_params)
train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
val_dataset = OAI_Dataloader(root_dir, csv_root_dir, val_csv_file, **val_params)
val_dataloader = DataLoader(val_dataset, batch_size=2, shuffle=True)
    
for epoch in range(num_epochs):
    for i, (im, labels) in enumerate(train_dataloader):
        im, labels = im.to(device), labels.to(device)
        model.train()
        optimizer.zero_grad()
            # Forward pass
        outputs = model(im)
        loss = criterion(outputs, labels)

            # Backward and optimize
        loss.backward()
        optimizer.step()
            # validate every 100 iterations
        if i > 0 and i % 1 == 0:
                # validate
            train_acc, train_l = test_model(train_dataloader, model)
            val_acc, val_l = test_model(val_dataloader, model)
            print('Epoch: [{}/{}], Step: [{}/{}], Validation Acc: {}'.format(
                       epoch+1, num_epochs, i+1, len(train_dataloader), val_acc))
            train_accuracy_500_vgg.append(train_acc)
            val_accuracy_500_vgg.append(val_acc)

    train_accuracy_epoch_vgg.append(train_acc)
    train_loss.append(train_l)
    val_accuracy_epoch_vgg.append(val_acc)
    val_loss.append(val_l)
        
out = {"train_accuracy_500_vgg": train_accuracy_500_vgg, 
       "train_accuracy_epoch_vgg": train_accuracy_epoch_vgg, 
       "train_loss": train_loss, 
       "val_accuracy_500_vgg": val_accuracy_500_vgg, 
       "val_accuracy_epoch_vgg": val_accuracy_epoch_vgg, 
       "val_loss": val_loss}

with open('vgg_output_mini.txt', 'w') as file:
     file.write(json.dumps(out))
