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
from torch.autograd import Variable
import h5py
from Augmentation import RandomCrop, CenterCrop, RandomFlip

def normalize_MRIs(image):
    mean = np.mean(image)
    std = np.std(image)
    image -= mean
    #image -= 95.09
    image /= std
    #image /= 86.38
    return image

class OAI_Dataloader(Dataset):
    """OAI dataset. Sequences of images, each sequence labeled by 0 or 1 (TKR or not)"""

    def __init__(self, root_dir, csv_root_dir, csv_file, dim=(384,384,32), 
                 normalize = True, randomCrop = True, 
                 randomFlip = True, flipProbability = -1, cropDim = (384,384,32)):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.data = pd.read_csv(csv_root_dir + csv_file)[:4]
        self.folders = ['00m']
        self.dim = dim
        self.normalize = normalize
        self.randomCrop = randomCrop
        self.randomFlip = randomFlip
        self.flipProbability = flipProbability
        self.cropDim = cropDim
        
        
    def __len__(self):
        return self.data.shape[0]
    

    def __getitem__(self, idx):
        """ returns img, label with : img = [batch_size, 1, X, Y, Z] 
                                      label in {0,1}
        """
        file = self.data['FileName'].iloc[idx]
        
        img_seq = []
        k = 0
        for folder in self.folders:
            pre_image = h5py.File(self.root_dir + file, "r")['data/'].value.astype('float64')
            if self.normalize:
                pre_image = normalize_MRIs(pre_image)
            # Augmentation
            if self.randomFlip:
                pre_image = RandomFlip(image=pre_image,p=0.5).horizontal_flip(p=self.flipProbability)
            if self.randomCrop:
                pre_image = RandomCrop(pre_image).crop_along_hieght_width_depth(self.cropDim)
            else:
                pre_image = CenterCrop(image=pre_image).crop(size = self.cropDim)
                
            img_seq.append(pre_image)
            k += 1

        img_seq = torch.tensor(torch.from_numpy(np.array(img_seq)), dtype=torch.float)

        label = int(self.data['Label'].iloc[idx])
         
        return (img_seq, label)
    
