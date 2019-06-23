
# coding: utf-8

# In[1]:


from __future__ import print_function
import numpy as np
import pickle
import os
import h5py
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from torch.optim import Adam
from torch.nn.parameter import Parameter
import torch.nn as nn
from sklearn import metrics
import matplotlib.pyplot as plt


# In[2]:


def read_labelled_data(root_dir, csv_file):
    
    train_labelled = pd.read_csv(csv_file)
    cropDim = (384, 384, 32)
    image_list = []
    label_list = []
    sample = train_labelled

    for i, name in enumerate(sample['FileName']):
        img = h5py.File(root_dir + name, "r")['data/'].value.astype('float64')
        image = CenterCrop(image=img).crop(size=cropDim)
        image_list.append(image)
        label_list.append(sample.iloc[[i], [9]].values[0][0])

    image_list = np.array(image_list)
    images = image_list[:, np.newaxis, :, :]
    labels = np.array(label_list)
    return images, labels


# In[3]:


class CenterCrop:
    '''
    CenterCrop Images
    '''
    
    def __init__(self, image):
        self.image = image
        self.h, self.w, self.d = image.shape

    def __functional__(self, size):
        '''
        param: size: crop Size
        return: Center Crop Images
        '''
        crop_h = int((self.h - size[0]) / 2)
        crop_w = int((self.w - size[1]) / 2)
        crop_d = int((self.d - size[2]) / 2)
        return self.image[crop_h:crop_h+size[0], crop_w:crop_w+size[1], crop_d:crop_d+size[2]]

    def crop(self,size):
        return self.__functional__(size)


# In[4]:


class SAG_IW_TSE_Dataset(Dataset):
    '''
    SAG_IW_TSE dataset 
    image = (384, 384, 32), label = {-1, 0, 1}
    -1: no label
    0: without TKR
    1: with TKR
    '''

    def __init__(self, root_dir, csv_file):
        '''
        Args:
            root_dir (string): Directory with all the images.
            csv_file (string): Path to the csv file with annotations.
        '''
        self.root_dir = root_dir
        self.data = pd.read_csv(csv_file)
    
    def __len__(self):
        return self.data.shape[0]
    

    def __getitem__(self, idx):
        '''
        return image, label
        '''
        
        cropDim = (384,384,32)
        image_list = []
        label_list = []
        
        name = self.data.iloc[idx]['FileName']
        img = h5py.File(self.root_dir + name, "r")['data/'].value.astype('float64')
        image = CenterCrop(image=img).crop(size=cropDim)
        label = self.data.iloc[[idx], [9]].values[0][0]
        
        image_list.append(image)
        label_list.append(label)
            
        image_list = np.array(image_list)
        labels = np.array(label_list)
        
        return (torch.FloatTensor(image_list), torch.LongTensor(labels))


# In[5]:


class CNN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, if_last_layer, noise_level):

        super(CNN, self).__init__()
        
        #self.maxpool = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        
        if out_channels == 32:
            self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=(3, 3, 3), stride = (2, 2,1), padding=1)
            #self.maxpool = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
            self.maxpool = nn.MaxPool3d(kernel_size=(2, 2, 1), stride=(2, 2, 1))
        elif out_channels == 128:
            self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=(3, 3, 3), stride = (1, 1,1), padding=1)
            #self.maxpool = nn.MaxPool3d(kernel_size=(2, 2, 1), stride=(2, 2, 1))
            self.maxpool = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        else:
            self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=(3, 3, 3), stride = (1, 1,1), padding=1)
            self.maxpool = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        
        self.if_last_layer = if_last_layer
        if if_last_layer:
            self.classifier = nn.Linear(4608,2)
            self.softmax = nn.Softmax(dim=1)
        
        self.activation = torch.nn.ReLU()
        self.noise_level = noise_level
        
        self.bn_normalize_clean = nn.BatchNorm3d(out_channels)
        self.bn_normalize = nn.BatchNorm3d(out_channels)
    
        self.buffer_z_pre = None
        self.buffer_z = None
        self.buffer_tilde_z = None

    def forward_clean(self, h):
        if self.if_last_layer:
            z_pre = self.conv(h)
            z_pre = self.conv(z_pre)
        else:
            z_pre = self.conv(h)
        self.buffer_z_pre = z_pre.detach().clone()
        z = self.bn_normalize_clean(z_pre)
        self.buffer_z = z.detach().clone()
        h = self.activation(z)
        h = self.maxpool(h)
        if self.if_last_layer:
            h = h.view(h.size(0), -1)
            h = self.classifier(h)
            h = self.softmax(h)
        return h

    def forward_noise(self, tilde_h):
        if self.if_last_layer:
            z_pre = self.conv(tilde_h)
            z_pre = self.conv(z_pre)
        else:
            z_pre = self.conv(tilde_h)
        z_pre_norm = self.bn_normalize(z_pre)
        noise = Variable(torch.normal(0.0, self.noise_level*torch.ones(z_pre_norm.size())))
        tilde_z = z_pre_norm + noise.to('cuda')
        self.buffer_tilde_z = tilde_z
        h = self.activation(tilde_z)
        h = self.maxpool(h)
        if self.if_last_layer:
            h = h.view(h.size(0), -1)
            h = self.classifier(h)
        return h


# In[6]:


class StackedEncoders(torch.nn.Module):
    def __init__(self, in_channel, out_channels_list, last_layer, noise_std):
        super(StackedEncoders, self).__init__()
        self.buffer_tilde_z_bottom = None
        self.encoders_ref = []
        self.encoders = torch.nn.Sequential()
        self.noise_level = noise_std
        n_encoders = len(out_channels_list)
        for i in range(n_encoders):
            if i == 0:
                in_channel = in_channel
            else:
                in_channel = out_channels_list[i-1]
                
            out_channel = out_channels_list[i]
            if_last_layer = last_layer[i]
            
            encoder_ref = "encoder_" + str(i)
            encoder = CNN(in_channel, out_channel, if_last_layer, noise_std)
            self.encoders_ref.append(encoder_ref)
            self.encoders.add_module(encoder_ref, encoder)

    def forward_clean(self, x):
        h = x
        for e_ref in self.encoders_ref:
            encoder = getattr(self.encoders, e_ref)
            h = encoder.forward_clean(h)
        return h

    def forward_noise(self, x):
        noise = Variable(torch.normal(0.0, self.noise_level*torch.ones(x.size())))
        h = x + noise.to('cuda')
        self.buffer_tilde_z_bottom = h.clone()
        
        for e_ref in self.encoders_ref:
            encoder = getattr(self.encoders, e_ref)
            h = encoder.forward_noise(h)
        return h

    def get_encoders_tilde_z(self, reverse=True):
        tilde_z_layers = []
        for e_ref in self.encoders_ref:
            encoder = getattr(self.encoders, e_ref)
            tilde_z = encoder.buffer_tilde_z.clone()
            tilde_z_layers.append(tilde_z)
        if reverse:
            tilde_z_layers.reverse()
        return tilde_z_layers

    def get_encoders_z_pre(self, reverse=True):
        z_pre_layers = []
        for e_ref in self.encoders_ref:
            encoder = getattr(self.encoders, e_ref)
            z_pre = encoder.buffer_z_pre.clone()
            z_pre_layers.append(z_pre)
        if reverse:
            z_pre_layers.reverse()
        return z_pre_layers

    def get_encoders_z(self, reverse=True):
        z_layers = []
        for e_ref in self.encoders_ref:
            encoder = getattr(self.encoders, e_ref)
            z = encoder.buffer_z.clone()
            z_layers.append(z)
        if reverse:
            z_layers.reverse()
        return z_layers


# In[7]:


class Decoder(torch.nn.Module):
    def __init__(self, in_channel, out_channel, if_first_layer, if_bottom):
        super(Decoder, self).__init__()

        self.in_channel = in_channel
        self.out_channel = out_channel
        self.if_first_layer = if_first_layer
        self.if_bottom = if_bottom
        
        if if_first_layer == True:
            #self.linear = nn.Linear(2, 18432)
            self.linear = nn.Linear(2, 4608)

        if self.if_bottom:
            self.upconv = nn.ConvTranspose3d(in_channel, 1, kernel_size = (2, 2, 1), stride=(2, 2, 1))
        else:
            if self.in_channel == 32:
                #self.upconv = nn.ConvTranspose3d(in_channel, out_channel, (4,4,2), stride=(4,4,2))
                self.upconv = nn.ConvTranspose3d(in_channel, out_channel, (4,4,1), stride=(4,4,1))
            #elif self.out_channel == 128:
            elif self.in_channel == 64:
                self.upconv = nn.ConvTranspose3d(in_channel, out_channel, (2,2,1), stride=(2,2,1))
            else:
                self.upconv = nn.ConvTranspose3d(in_channel, out_channel, 2, stride=2)
            self.conv = nn.Conv3d(out_channel, out_channel, kernel_size=(3, 3, 3), stride = (1, 1,1), padding=1)
            self.bn_normalize = torch.nn.BatchNorm3d(out_channel, affine=False)
            
        self.buffer_hat_z_l = None

    def g(self, tilde_z_l, u_l):
        #print('input of encoder.g tilde_z_l: ', tilde_z_l.shape)
        #print('input of encoder.g u_l: ', u_l.shape)
        if self.if_first_layer:
            u_l = self.linear(u_l)
            #u_l = u_l.reshape(u_l.shape[0], 512, 6, 6, 1)
            u_l = u_l.reshape(u_l.shape[0], 128, 3, 3, 4)
        
        u_l = self.upconv(u_l) 
        hat_z_l = tilde_z_l + u_l
        return hat_z_l

    def forward(self, tilde_z_l, u_l):
        hat_z_l = self.g(tilde_z_l, u_l)
        self.buffer_hat_z_l = hat_z_l

        if self.if_bottom:
            return None
        else:
            t = self.conv(hat_z_l)
            u_l_below = self.bn_normalize(t)
            return u_l_below


# In[8]:


class StackedDecoders(torch.nn.Module):
    def __init__(self, in_channel, out_channels_list, first_layer):
        super(StackedDecoders, self).__init__()
        self.decoders_ref = []
        self.decoders = torch.nn.Sequential()
        n_decoders = len(out_channels_list)
        for i in range(n_decoders):
            if i == 0:
                in_channel = in_channel
            else:
                in_channel = out_channels_list[i - 1]
            out_channel = out_channels_list[i]
            if_first_layer = first_layer[i]
            decoder_ref = "decoder_" + str(i)
            decoder = Decoder(in_channel, out_channel, if_first_layer, False)
            self.decoders_ref.append(decoder_ref)
            self.decoders.add_module(decoder_ref, decoder)

        #self.bottom_decoder = Decoder(64, 1, False, True)
        self.bottom_decoder = Decoder(32, 1, False, True)

    def forward(self, tilde_z_layers, u_top, tilde_z_bottom):
        hat_z = []
        u = u_top
        for i in range(len(self.decoders_ref)):
            #print('decoder ', i)
            d_ref = self.decoders_ref[i]
            decoder = getattr(self.decoders, d_ref)
            tilde_z = tilde_z_layers[i]
            u = decoder.forward(tilde_z, u)
            hat_z.append(decoder.buffer_hat_z_l)
        self.bottom_decoder.forward(tilde_z_bottom, u)
        hat_z_bottom = self.bottom_decoder.buffer_hat_z_l.clone()
        hat_z.append(hat_z_bottom)
        return hat_z

    '''
    def bn_hat_z_layers(self, hat_z_layers, z_pre_layers):
        assert len(hat_z_layers) == len(z_pre_layers)
        hat_z_layers_normalized = []
        for i, (hat_z, z_pre) in enumerate(zip(hat_z_layers, z_pre_layers)):
            mean = z_pre.view(z_pre.size()[0], -1).mean(-1)
            mean = mean.view(mean.size()[0], 1, 1, 1, 1)
            ones = Variable(torch.ones(z_pre.size())).to('cuda')
            var = Variable(torch.normal(0, (1-1e-10)*torch.ones(z_pre.size()))).to('cuda')
            hat_z_normalized = (hat_z-mean*ones)/torch.sqrt(var+1e-10)
            hat_z_layers_normalized.append(hat_z_normalized)
        return hat_z_layers_normalized
    '''
        
    def bn_hat_z_layers(self, hat_z_layers, z_pre_layers):
        assert len(hat_z_layers) == len(z_pre_layers)
        hat_z_layers_normalized = []
        for i, (hat_z, z_pre) in enumerate(zip(hat_z_layers, z_pre_layers)):
            mean = z_pre.view(z_pre.size()[0], -1).mean(-1)
            mean = mean.view(mean.size()[0], 1, 1, 1, 1)
            ones = Variable(torch.ones(z_pre.size())).to('cuda')
            noise_var = torch.normal(0, (1-1e-10)*torch.ones(z_pre.size())).to('cuda')
            var = torch.arange(0., z_pre.shape[0]).to('cuda')
            
            for var_size in range(z_pre.shape[0]):    
                var[var_size] = torch.var(z_pre[var_size] + noise_var[var_size]).reshape(1)
                
            var = var.view(var.size()[0], 1, 1, 1, 1)

            hat_z_normalized = torch.div((hat_z-mean*ones), (torch.sqrt(var+1e-10))*ones)
            hat_z_layers_normalized.append(hat_z_normalized)
        return hat_z_layers_normalized
    


# In[9]:


class Ladder(torch.nn.Module):
    def __init__(self, encoder_in_channel, encoder_out_channel_list, encoder_if_last_layer, noise_std,
                decoder_in_channel, decoder_out_channel_list, decoder_if_first_layer):
        super(Ladder, self).__init__()
    
        self.se = StackedEncoders(encoder_in_channel, encoder_out_channel_list, encoder_if_last_layer, noise_std)
        self.de = StackedDecoders(decoder_in_channel, decoder_out_channel_list, decoder_if_first_layer)

    def forward_encoders_clean(self, data):
        return self.se.forward_clean(data)

    def forward_encoders_noise(self, data):
        return self.se.forward_noise(data)

    def forward_decoders(self, tilde_z_layers, encoder_output, tilde_z_bottom):
        return self.de.forward(tilde_z_layers, encoder_output, tilde_z_bottom)

    def get_encoders_tilde_z(self, reverse=True):
        return self.se.get_encoders_tilde_z(reverse)

    def get_encoders_z_pre(self, reverse=True):
        return self.se.get_encoders_z_pre(reverse)

    def get_encoder_tilde_z_bottom(self):
        return self.se.buffer_tilde_z_bottom.clone()

    def get_encoders_z(self, reverse=True):
        return self.se.get_encoders_z(reverse)

    def decoder_bn_hat_z_layers(self, hat_z_layers, z_pre_layers):
        return self.de.bn_hat_z_layers(hat_z_layers, z_pre_layers)


# In[10]:


def evaluate_performance(ladder, valid_loader, e, agg_cost_scaled, agg_supervised_cost_scaled,
                         agg_unsupervised_cost_scaled, cuda):
    correct = 0.
    total = 0.
    TP = 0.
    TN = 0.
    FP = 0.
    FN = 0.
    TPR = 0.
    FPR = 0.
    op = []
    for batch_idx, (data, target) in enumerate(valid_loader):
        if cuda:
            data = data.cuda()
        data, target = Variable(data), Variable(target)
        output = ladder.forward_encoders_clean(data)
        #print('batch_idx', batch_idx)
        #if len(data.shape)==2:
            #output = ladder.forward_encoders_clean(data)
        #else:
            #data = data.reshape(data.shape[0], data.shape[2])
            #output = ladder.forward_encoders_clean(data)
        # TODO: Do away with the below hack for GPU tensors.
        if cuda:
            output = output.cpu()
            target = target.cpu()
        
        output = output.data.numpy()
        #print('output', output)
        preds = np.argmax(output, axis=1)
        #print('preds', preds)
        target = target.data.numpy()
        target = target.T[0]
        #print('target', target)
        correct += np.sum(target == preds)
        #print('correct', correct)
        total += target.shape[0]
        #print('total', total)
        
        for val_idx in range(len(preds)):
            op.append(output[val_idx][1]/(output[val_idx][1]+output[val_idx][0]))
            if target[val_idx] == 1:
                if preds[val_idx] == 1:
                    TP += 1
                elif preds[val_idx] == 0:
                    FN += 1
            elif target[val_idx] == 0:
                if preds[val_idx] == 1:
                    FP += 1
                elif preds[val_idx] == 0:
                    TN += 1
                    
        
    TPR = TP/(TP + FN)
    FPR = FP/(FP + TN)
    TPR_list.append(TPR)
    FPR_list.append(FPR)
        
    print("Epoch:", e + 1, "\t",
          "Total Cost:", "{:.4f}".format(agg_cost_scaled), "\t",
          "Supervised Cost:", "{:.4f}".format(agg_supervised_cost_scaled), "\t",
          "Unsupervised Cost:", "{:.4f}".format(agg_unsupervised_cost_scaled), "\t",
          "Validation Accuracy:", correct / total,
          "TP:", TP,
          "FN:", FN,
          "FP:", FP,
          "TN:", TN,
          "Recall:", TPR)
    print("output:", op)


# In[11]:


def evaluate_performance2(ladder, test_loader, cuda):
    correct = 0.
    total = 0.
    TP = 0.
    TN = 0.
    FP = 0.
    FN = 0.
    TPR = 0.
    FPR = 0.
    op = []
    for batch_idx, (data, target) in enumerate(test_loader):
        if cuda:
            data = data.cuda()
        data, target = Variable(data), Variable(target)
        output = ladder.forward_encoders_clean(data)
        
        if cuda:
            output = output.cpu()
            target = target.cpu()
        
        output = output.data.numpy()
        preds = np.argmax(output, axis=1)
        target = target.data.numpy()
        target = target.T[0]
        correct += np.sum(target == preds)
        total += target.shape[0]
        
        for val_idx in range(len(preds)):
            op.append(output[val_idx][1]/(output[val_idx][1]+output[val_idx][0]))
            if target[val_idx] == 1:
                if preds[val_idx] == 1:
                    TP += 1
                elif preds[val_idx] == 0:
                    FN += 1
            elif target[val_idx] == 0:
                if preds[val_idx] == 1:
                    FP += 1
                elif preds[val_idx] == 0:
                    TN += 1
                    
        
    TPR = TP/(TP + FN)
    FPR = FP/(FP + TN)

        
    print("Test Accuracy:", correct / total,
          "TP:", TP,
          "FN:", FN,
          "FP:", FP,
          "TN:", TN,
          "Recall:", TPR,
          "FPR:", FPR)
    
    print("output:", op)


# In[12]:


batch_size = 2
epochs = 8
noise_std = 0.2
seed = 42
decay_epoch = 15
cuda = True

if cuda and not torch.cuda.is_available():
    print("WARNING: torch.cuda not available, using CPU.\n")
    cuda = False

print("=====================")
print("BATCH SIZE:", batch_size)
print("EPOCHS:", epochs)
print("RANDOM SEED:", seed)
print("NOISE STD:", noise_std)
print("LR DECAY EPOCH:", decay_epoch)
print("CUDA:", cuda)
print("=====================\n")

np.random.seed(seed)
torch.manual_seed(seed)
if cuda:
    torch.cuda.manual_seed(seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}


# In[13]:


root_dir = '/gpfs/data/denizlab/Datasets/OAI/SAG_IW_TSE/'
csv_file1 = '/home/ys2542/train_unlabelled3.csv' 
csv_file2 = '/home/ys2542/validation_labelled3.csv'
csv_file3 = '/home/ys2542/train_labelled3.csv' 
csv_file4 = '/home/ys2542/test_labelled3.csv'

#csv_file1 = '/home/ys2542/train_unlabelled4.csv' 
#csv_file2 = '/home/ys2542/validation_labelled4.csv'
#csv_file3 = '/home/ys2542/train_labelled4.csv' 
#csv_file4 = '/home/ys2542/test_labelled4.csv'

unlabelled_dataset = SAG_IW_TSE_Dataset(root_dir, csv_file1)
unlabelled_loader = DataLoader(dataset=unlabelled_dataset, batch_size=batch_size, shuffle=True, **kwargs)

validation_dataset = SAG_IW_TSE_Dataset(root_dir, csv_file2)
validation_loader = DataLoader(dataset=validation_dataset, batch_size=batch_size, shuffle=True, **kwargs)

train_labelled_images, train_labelled_labels = read_labelled_data(root_dir, csv_file3)

test_dataset = SAG_IW_TSE_Dataset(root_dir, csv_file4)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, **kwargs)


# In[14]:


# Configure the Ladder
starter_lr = 0.02
encoder_in_channels = 1
encoder_out_channels_list = [32, 32, 64, 128, 128]
if_last_layer = [False, False, False, False, True]
noise_level = 0.2

decoder_in_channels = 128
decoder_out_channels_list = [128, 128, 64, 32, 32]
if_first_layer = [True, False, False, False, False]

unsupervised_costs_lambda = [(0.1**5)/10, (0.1**5)/10, (0.1**5)/10, (0.1**3)/10, 1/10, 10/10]

ladder = Ladder(encoder_in_channels, encoder_out_channels_list, if_last_layer, noise_level, 
               decoder_in_channels, decoder_out_channels_list, if_first_layer).to('cuda')
optimizer = Adam(ladder.parameters(), lr=starter_lr)
loss_supervised = torch.nn.CrossEntropyLoss()
loss_unsupervised = torch.nn.MSELoss()

assert len(unsupervised_costs_lambda) == len(decoder_out_channels_list) + 1
assert len(encoder_out_channels_list) == len(decoder_out_channels_list)

print("")
print("========NETWORK=======")
print(ladder)
print("======================")

print("")
print("==UNSUPERVISED-COSTS==")
print(unsupervised_costs_lambda)

print("")
print("=====================")
print("TRAINING\n")


# In[15]:


for e in range(epochs):
    TPR_list = []
    FPR_list = []
    #pred_list = []
    agg_cost = 0.
    agg_supervised_cost = 0.
    agg_unsupervised_cost = 0.
    num_batches = 0
    ladder.train()
    # TODO: Add volatile for the input parameters in training and validation
    ind_labelled = 0
    ind_limit = np.ceil(float(train_labelled_images.shape[0]) / batch_size)

    if e > decay_epoch:
        ratio = float(epochs - e) / (epochs - decay_epoch)
        current_lr = starter_lr * ratio
        optimizer = Adam(ladder.parameters(), lr=current_lr)


    for batch_idx, (unlabelled_images, unlabelled_labels) in enumerate(unlabelled_loader):
        if ind_labelled == ind_limit:
            randomize = np.arange(train_labelled_images.shape[0])
            np.random.shuffle(randomize)
            train_labelled_images = train_labelled_images[randomize]
            train_labelled_labels = train_labelled_labels[randomize]
            ind_labelled = 0

        # TODO: Verify whether labelled examples are used for calculating unsupervised loss.

        labelled_start = batch_size * ind_labelled
        labelled_end = batch_size * (ind_labelled + 1)
        ind_labelled += 1
        batch_train_labelled_images = torch.FloatTensor(train_labelled_images[labelled_start:labelled_end])
        batch_train_labelled_labels = torch.LongTensor(train_labelled_labels[labelled_start:labelled_end])
        
        #unlabelled_images = names2images(root_dir, unlabelled_images_names)

        if cuda:
            batch_train_labelled_images = batch_train_labelled_images.cuda()
            batch_train_labelled_labels = batch_train_labelled_labels.cuda()
            unlabelled_images = unlabelled_images.cuda()

        labelled_data = Variable(batch_train_labelled_images, requires_grad=False)
        labelled_target = Variable(batch_train_labelled_labels, requires_grad=False)
        unlabelled_data = Variable(unlabelled_images)

        optimizer.zero_grad()

        # do a noisy pass for labelled data
        #print('input shape: ', labelled_data.shape)
        output_noise_labelled = ladder.forward_encoders_noise(labelled_data)

        # do a noisy pass for unlabelled_data
        # print('Unlabelled_data_size:', unlabelled_data.shape)
        #unlabelled_data = unlabelled_data.reshape(unlabelled_data.shape[0], unlabelled_data.shape[2])
        #print('Unlabelled_data_size: ', unlabelled_data.shape)
        output_noise_unlabelled = ladder.forward_encoders_noise(unlabelled_data)
        tilde_z_layers_unlabelled = ladder.get_encoders_tilde_z(reverse=True)

        # do a clean pass for unlabelled data
        output_clean_unlabelled = ladder.forward_encoders_clean(unlabelled_data)
        z_pre_layers_unlabelled = ladder.get_encoders_z_pre(reverse=True)
        #for z_value in z_pre_layers_unlabelled:
            #print('z_pre', torch.max(z_value))
        z_layers_unlabelled = ladder.get_encoders_z(reverse=True)
        #for z_value in z_layers_unlabelled:
            #print('z', torch.max(z_value))

        tilde_z_bottom_unlabelled = ladder.get_encoder_tilde_z_bottom()
        #print(torch.max(tilde_z_bottom_unlabelled))

        # pass through decoders
        hat_z_layers_unlabelled = ladder.forward_decoders(tilde_z_layers_unlabelled,
                                                          output_noise_unlabelled,
                                                          tilde_z_bottom_unlabelled)
        #for z_value in hat_z_layers_unlabelled:
            #print('hat_z', torch.max(z_value))
        z_pre_layers_unlabelled.append(unlabelled_data)
        z_layers_unlabelled.append(unlabelled_data)

        # TODO: Verify if you have to batch-normalize the bottom-most layer also
        # batch normalize using mean, var of z_pre
        bn_hat_z_layers_unlabelled = ladder.decoder_bn_hat_z_layers(hat_z_layers_unlabelled, z_pre_layers_unlabelled)

        # calculate costs
        cost_supervised = loss_supervised.forward(output_noise_labelled, labelled_target)
        cost_unsupervised = 0.
        assert len(z_layers_unlabelled) == len(bn_hat_z_layers_unlabelled)
        for cost_lambda, z, bn_hat_z in zip(unsupervised_costs_lambda, z_layers_unlabelled, bn_hat_z_layers_unlabelled):
            #print(torch.max(bn_hat_z), torch.max(z))
            c = cost_lambda * loss_unsupervised.forward(bn_hat_z, z)
            #print(c)
            cost_unsupervised += c
            
        #print(ind_labelled)    
        #print("Start Backprop!!!!")
        #print(cost_supervised)
        #print(cost_unsupervised)
        
        # backprop
        cost = cost_supervised + cost_unsupervised
        #print(cost)
        cost.backward()
        optimizer.step()

        agg_cost += cost.data[0]
        agg_supervised_cost += cost_supervised.data[0]
        agg_unsupervised_cost += cost_unsupervised.data[0]
        num_batches += 1

        if ind_labelled == ind_limit:
        #if ind_labelled == 2:
            # Evaluation
            ladder.eval()
            evaluate_performance(ladder, validation_loader, e,
                                 agg_cost / num_batches,
                                 agg_supervised_cost / num_batches,
                                 agg_unsupervised_cost / num_batches,
                                 cuda)
            ladder.train()
            
            
    TPR_array = np.array(TPR_list)        
    FPR_array = np.array(FPR_list)
    #pred_array= np.array(pred_list)
    print('TPR_array:', TPR_array)
    print('FPR_array:', FPR_array)
    #print('pred_array:', pred_array)
    torch.save(ladder.state_dict(), './model4/' + "model" + str(e+1) + ".pth")
    

print('Finish training')


# In[ ]:


ladder.eval()
evaluate_performance2(ladder, test_loader, cuda)
print("=====================\n")
print("Done :)")


# In[ ]:


#ladder.load_state_dict(torch.load('./model/model1.pth'))

