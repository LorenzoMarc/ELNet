import torch
import torch.nn as nn
#from torchvision import models
# Copyright (c) 2019, Adobe Inc. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike
# 4.0 International Public License. To view a copy of this license, visit
# https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode.
 
import torch
import torch.nn.parallel
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from IPython import embed
 
class Downsample(nn.Module):
    def __init__(self, pad_type='reflect', filt_size=3, stride=2, channels=None, pad_off=0):
        super(Downsample, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [int(1.*(filt_size-1)/2), int(np.ceil(1.*(filt_size-1)/2)), int(1.*(filt_size-1)/2), int(np.ceil(1.*(filt_size-1)/2))]
        self.pad_sizes = [pad_size+pad_off for pad_size in self.pad_sizes]
        self.stride = stride
        self.off = int((self.stride-1)/2.)
        self.channels = channels
 
        if(self.filt_size==1):
            a = np.array([1.,])
        elif(self.filt_size==2):
            a = np.array([1., 1.])
        elif(self.filt_size==3):
            a = np.array([1., 2., 1.])
        elif(self.filt_size==4):    
            a = np.array([1., 3., 3., 1.])
        elif(self.filt_size==5):    
            a = np.array([1., 4., 6., 4., 1.])
        elif(self.filt_size==6):    
            a = np.array([1., 5., 10., 10., 5., 1.])
        elif(self.filt_size==7):    
            a = np.array([1., 6., 15., 20., 15., 6., 1.])
 
        filt = torch.Tensor(a[:,None]*a[None,:])
        filt = filt/torch.sum(filt)
        self.register_buffer('filt', filt[None,None,:,:].repeat((self.channels,1,1,1)))
 
        self.pad = get_pad_layer(pad_type)(self.pad_sizes)
 
    def forward(self, inp):
        if(self.filt_size==1):
            if(self.pad_off==0):
                return inp[:,:,::self.stride,::self.stride]    
            else:
                return self.pad(inp)[:,:,::self.stride,::self.stride]
        else:
            return F.conv2d(self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1])
 
def get_pad_layer(pad_type):
    if(pad_type in ['refl','reflect']):
        PadLayer = nn.ReflectionPad2d
    elif(pad_type in ['repl','replicate']):
        PadLayer = nn.ReplicationPad2d
    elif(pad_type=='zero'):
        PadLayer = nn.ZeroPad2d
    else:
        print('Pad type [%s] not recognized'%pad_type)
    return PadLayer
 
class Downsample1D(nn.Module):
    def __init__(self, pad_type='reflect', filt_size=3, stride=2, channels=None, pad_off=0):
        super(Downsample1D, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [int(1. * (filt_size - 1) / 2), int(np.ceil(1. * (filt_size - 1) / 2))]
        self.pad_sizes = [pad_size + pad_off for pad_size in self.pad_sizes]
        self.stride = stride
        self.off = int((self.stride - 1) / 2.)
        self.channels = channels
 
        # print('Filter size [%i]' % filt_size)
        if(self.filt_size == 1):
            a = np.array([1., ])
        elif(self.filt_size == 2):
            a = np.array([1., 1.])
        elif(self.filt_size == 3):
            a = np.array([1., 2., 1.])
        elif(self.filt_size == 4):
            a = np.array([1., 3., 3., 1.])
        elif(self.filt_size == 5):
            a = np.array([1., 4., 6., 4., 1.])
        elif(self.filt_size == 6):
            a = np.array([1., 5., 10., 10., 5., 1.])
        elif(self.filt_size == 7):
            a = np.array([1., 6., 15., 20., 15., 6., 1.])
 
        filt = torch.Tensor(a)
        filt = filt / torch.sum(filt)
        self.register_buffer('filt', filt[None, None, :].repeat((self.channels, 1, 1)))
 
        self.pad = get_pad_layer_1d(pad_type)(self.pad_sizes)
 
    def forward(self, inp):
        if(self.filt_size == 1):
            if(self.pad_off == 0):
                return inp[:, :, ::self.stride]
            else:
                return self.pad(inp)[:, :, ::self.stride]
        else:
            return F.conv1d(self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1])
 
def get_pad_layer_1d(pad_type):
    if(pad_type in ['refl', 'reflect']):
        PadLayer = nn.ReflectionPad1d
    elif(pad_type in ['repl', 'replicate']):
        PadLayer = nn.ReplicationPad1d
    elif(pad_type == 'zero'):
        PadLayer = nn.ZeroPad1d
    else:
        print('Pad type [%s] not recognized' % pad_type)
    return PadLayer
# ELNet architecture definition

class ELNet(nn.Module):
 
   def __init__(self):
    super(ELNet, self).__init__()

    K = 4

    self.conv1 = nn.Conv2d(3,4*K, kernel_size=7, stride= 2, padding=3)
    
    self.norm1 = nn.LayerNorm((4*K, 128,128), eps = 1e-8, elementwise_affine=True)
   
    self.blurpool1 = nn.Sequential(
        nn.Conv2d(4*K,4*K, kernel_size= 7,stride=1,padding=1),
        nn.ReLU(inplace=True),
        Downsample(channels=4*K, filt_size=3, stride=2))
 
    self.block1= nn.Sequential(
        nn.Conv2d(4*K,4*K, kernel_size=5, stride =1, padding = 2),
        nn.LayerNorm((4*K, 62,62), eps = 1e-8, elementwise_affine=True),
        nn.ReLU()
    )
 
    self.conv2 = nn.Conv2d(4*K, 8*K, kernel_size= 5, padding = 2)
   
    self.blurpool2 = nn.Sequential(
        nn.Conv2d(8*K, 8*K, kernel_size= 7,stride=1,padding=1),
        nn.ReLU(),
        Downsample(channels=8*K, filt_size=3, stride=2) #prova filt = 5
    )
 
    self.block2 = nn.Sequential(
        nn.Conv2d(8*K,8*K, kernel_size=3, stride=1, padding=1),
        nn.LayerNorm((8*K, 29,29), eps = 1e-8, elementwise_affine=True),
        nn.ReLU()  
    )
 
    self.conv3 = nn.Conv2d(8*K, 16*K, kernel_size= 3, padding = 1)
 
    self.blurpool3 = nn.Sequential(
        nn.Conv2d(16*K, 16*K, kernel_size= 7,stride=1,padding=1),
        nn.ReLU(inplace=True),
        Downsample(channels=16*K, filt_size=3, stride=2) #prova filt = 5
    )
 
    self.block3 = nn.Sequential(
        nn.Conv2d(16*K,16*K, kernel_size=3, stride=1, padding=1),
        nn.LayerNorm((16*K, 13,13), eps = 1e-8, elementwise_affine=True),
        nn.ReLU()
)
    
    self.block4 = nn.Sequential(
        nn.Conv2d(16*K,16*K, kernel_size=3, stride=1, padding=1),
        nn.LayerNorm((16*K, 5,5), eps = 1e-8, elementwise_affine=True),
        nn.ReLU()         
    )
 
    self.conv4 = nn.Conv2d(16*K, 16*K, kernel_size= 3, padding = 1)
    self.blurpool4 = nn.Sequential(
        nn.Conv2d(16*K, 16*K, kernel_size= 5,stride=1,padding=1),
        nn.ReLU(inplace=True),
        Downsample(channels=16*K, filt_size=5, stride=2) #prova filt = 5
    )
    self.pooling = nn.MaxPool2d(kernel_size=2,stride = 1)

    self.fc1= nn.Sequential(
        nn.Dropout(),
        nn.Linear(16*K, 2),
        nn.Softmax(dim=1)   #PROVA COSì
        )
    

   def forward(self, x):
    x = torch.squeeze(x, dim=0)
    x = self.norm1(self.conv1(x))
    # --> sx4Kx128x128
    x = self.blurpool1(x)
    # --> sx4Kx62x62
    #Block1 [5x5]
    b1 = x
    for i in [1,2]:
      b1 = self.block1(b1)
    #skip connection
    x = x + b1
    
    x = self.conv2(x)
    # --> sx8Kx62x62
    x = self.blurpool2(x)
    # --> sx8Kx29x29
    
    #Block2 [3x3]
    b2 = x
    for i in [1,2]:
      b2 = self.block2(b2)
    #skip connection
    x = x + b2
 
    x = self.conv3(x)
    # --> sx16Kx29x29
 
    x = self.blurpool3(x)
    # --> sx16Kx13x13
 
    #Block3 [3x3]
    b3 = self.block3(x)
    
    #skip connection
    x = x + b3
 
    x = self.conv4(x)
    x= self.blurpool3(x)
    # --> sx16Kx5x5
 
    #Block4 [3x3]
    b4 = self.block4(x)
    #skip connection
    x = x + b4
 
    x = self.conv4(x)
    x = self.blurpool4(x)
    # --> sx16K
    # Feature Extraction
    x = self.pooling(x).view(x.size(0), -1)
  
    x = torch.max(x, 0, keepdim=True)[0]
    # --> 16K
    x = self.fc1(x)
    # --> 2
    return x