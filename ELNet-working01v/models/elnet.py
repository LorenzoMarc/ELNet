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
import antialiased_cnns

# ELNet architecture definition

class ELNet(nn.Module):
 
   def __init__(self, K, norm_type):
    super(ELNet, self).__init__()
    
    self.conv1 = nn.Conv2d(1,4*K, kernel_size=7, stride= 2, padding=3)
    self.blurpool1 = nn.Sequential(
        nn.Conv2d(4*K,4*K, kernel_size= 7,stride=1,padding=1),
        nn.ReLU(inplace=True),
        antialiased_cnns.BlurPool(channels=4*K, filt_size=3, stride=2))
    self.block1=nn.Conv2d(4*K,4*K, kernel_size=5, stride =1, padding = 2)
 

    self.conv2 = nn.Conv2d(4*K, 8*K, kernel_size= 5, padding = 2)
    self.blurpool2 = nn.Sequential(
        nn.Conv2d(8*K, 8*K, kernel_size= 7,stride=1,padding=1),
        nn.ReLU(),
        antialiased_cnns.BlurPool(channels=8*K, filt_size=3, stride=2) #prova filt = 3
    )
    self.block2 = nn.Conv2d(8*K,8*K, kernel_size=3, stride=1, padding=1)
          

 
    self.conv3 = nn.Conv2d(8*K, 16*K, kernel_size= 3, padding = 1)
    self.blurpool3 = nn.Sequential(
        nn.Conv2d(16*K, 16*K, kernel_size= 7,stride=1,padding=1),
        nn.ReLU(inplace=True),
        antialiased_cnns.BlurPool(channels=16*K, filt_size=3, stride=2) #prova filt = 3
    )
    self.block3 = nn.Conv2d(16*K,16*K, kernel_size=3, stride=1, padding=1)

    
    self.conv4 = nn.Conv2d(16*K, 16*K, kernel_size= 3, padding = 1)
    self.blurpool4 = nn.Sequential(
        nn.Conv2d(16*K, 16*K, kernel_size= 5,stride=1,padding=1),
        nn.ReLU(inplace=True),
        antialiased_cnns.BlurPool(channels=16*K, filt_size=5, stride=2) #prova filt = 3
    )
    self.block4 = nn.Conv2d(16*K,16*K, kernel_size=3, stride=1, padding=1)

    self.conv5 = nn.Conv2d(16*K, 16*K, kernel_size= 3, padding = 1)
    self.blurpool5 = nn.Sequential(
        nn.Conv2d(16*K, 16*K, kernel_size= 5,stride=1,padding=1),
        nn.ReLU(inplace=True),
        antialiased_cnns.BlurPool(channels=16*K, filt_size=5, stride=2) #prova filt = 3
    )

    self.pooling = nn.MaxPool2d(kernel_size=2,stride = 1)

    self.fc1= nn.Sequential(
        nn.Dropout(),
        nn.Linear(16*K, 2)
        )
    

   def forward(self, x):
     norm_type='layer'
     K = 4

     if norm_type=='layer':
       x = torch.squeeze(x, dim=0)
       norm1 = nn.LayerNorm((4*K, 128,128), eps = 1e-8, elementwise_affine=True)
       norm2  = nn.LayerNorm((4*K, 62,62), eps = 1e-8, elementwise_affine=True)
       norm3 = nn.LayerNorm((8*K, 29,29), eps = 1e-8, elementwise_affine=True)
       norm4 = nn.LayerNorm((16*K, 13,13), eps = 1e-8, elementwise_affine=True)
       norm5 = nn.LayerNorm((16*K, 5,5), eps = 1e-8, elementwise_affine=True)
     else:
       x= x[-1]
       norm1 = nn.LayerNorm((x.size(1),4*K, 128,128), eps = 1e-8, elementwise_affine=True)
       norm2  = nn.LayerNorm((x.size(1),4*K, 62,62), eps = 1e-8, elementwise_affine=True)
       norm3 = nn.LayerNorm((x.size(1),8*K, 29,29), eps = 1e-8, elementwise_affine=True)
       norm4 = nn.LayerNorm((x.size(1),16*K, 13,13), eps = 1e-8, elementwise_affine=True)
       norm5 = nn.LayerNorm((x.size(1),16*K, 5,5), eps = 1e-8, elementwise_affine=True)
    
     
     x=self.conv1(x)
     x = norm1(x)
     # --> sx4Kx128x128
     x = self.blurpool1(x)
     # --> sx4Kx62x62
     #Block1 [5x5]
     b1 = x
     for i in [1,2]:
       b1 = self.block1(b1)
       b1=norm2(b1)
       b1=F.relu(b1)

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
       b2=norm3(b2)
       b2= F.relu(b2)

     #skip connection
     x = x + b2
  
     x = self.conv3(x)
     # --> sx16Kx29x29
     x = self.blurpool3(x)
     # --> sx16Kx13x13
     #Block3 [3x3]
     b3 = self.block3(x)
     b3 = norm4(b3)
     b3 = F.relu(b3)
      
     #skip connection
     x = x + b3
  
     x = self.conv4(x)
     x= self.blurpool3(x)
     # --> sx16Kx5x5
     #Block4 [3x3]
     b4 = self.block4(x)
     b4=norm5(b4)
     b4=F.relu(b4)
     #skip connection
     x = x + b4
  
     x = self.conv5(x)
     x = self.blurpool5(x)
     # --> sx16K

     # Feature Extraction
     x = self.pooling(x).view(x.size(0), -1)
     x = torch.max(x, dim=0, keepdim=True)[0]
     # --> 16K
     x = self.fc1(x)
     # --> 2
     return x