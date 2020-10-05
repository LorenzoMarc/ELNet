import torch
import torch.nn.parallel
import random
import math
import os
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from IPython import embed
import antialiased_cnns

def ident_block(channels, kernel_size,norm,dilation=1, iter=2):
    block_list = nn.ModuleList([])
    for i in range(iter):
        conv2d = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size,
                           dilation=1, stride=1,
                           padding=(kernel_size + ((dilation - 1) * (kernel_size - 1))) // 2)
        conv_list.append(conv2d)
        conv_list.append(normalization(channels, norm))
        conv_list.append(nn.ReLU())
    return nn.Sequential(*conv_list)

# Group norm può essere visualizzato come una generalizzazione di instanceNorm e LayerNorm
def normalization(channel, norma_type):
    if norma_type == 'constrast':
        layer = nn.GroupNorm(channel, channel)  #nn.InstanceNorm2d()
    else:
        layer = nn.GroupNorm(1, channel)        #nn.LayerNorm()
    return layer

class ELNet(nn.Module):
 
  def __init__(self, K, norm_type):
    super(ELNet, self).__init__()
    
    self.conv1 = nn.Conv2d(1,4*K, kernel_size=7, stride= 2, padding=3)
    self.norm = normalization(4*K, norm_type)

    #Da articolo di Zhang: conversione da blocco convoluzionale in conv_downsample
    self.blurpool1 = nn.Sequential(
        nn.Conv2d(4*K,4*K, kernel_size= 7,stride=1,padding=1),
        nn.ReLU(),
        antialiased_cnns.BlurPool(channels=4*K, filt_size=5, stride=2))

    self.block1= ident_block(channels=4*K, kernel_size=5,norm=norm_type, iter = 2 ) 

    self.conv2 =nn.Conv2d(4*K, 8*K, kernel_size= 5, padding = 2)

    #Da articolo di Zhang: conversione da blocco convoluzionale in conv_downsample
    self.blurpool2 = nn.Sequential(
        nn.Conv2d(8*K, 8*K, kernel_size= 7,stride=1,padding=1),
        nn.ReLU(),
        antialiased_cnns.BlurPool(channels=8*K, filt_size=5, stride=2) #prova filt = 3
    )
    self.block2 =ident_block(8*K, kernel_size= 3,norm=norm_type, iter = 2) 
      
    self.conv3 =nn.Conv2d(8*K, 16*K, kernel_size= 3, padding = 1)

    #Da articolo di Zhang: conversione da blocco convoluzionale in conv_downsample
    self.blurpool3 = nn.Sequential(
        nn.Conv2d(16*K, 16*K, kernel_size= 7,stride=1,padding=1),
        nn.ReLU(),
        antialiased_cnns.BlurPool(channels=16*K, filt_size=5, stride=2) #prova filt = 3
    )
    self.block3 = ident_block(16*K, kernel_size= 3,norm=norm_type, iter = 1)    
    self.conv4 = nn.Conv2d(16*K, 16*K, kernel_size= 3, padding = 1)

    #Da articolo di Zhang: conversione da blocco convoluzionale in conv_downsample
    self.blurpool4 = nn.Sequential(
        nn.Conv2d(16*K, 16*K, kernel_size= 5,stride=1,padding=1),
        nn.ReLU(),
        antialiased_cnns.BlurPool(channels=16*K, filt_size=5, stride=2) #prova filt = 3
    )


    self.block4 =ident_block(16*K, kernel_size= 3,norm=norm_type, iter = 1)

    self.conv5=nn.Conv2d(16*K, 16*K, kernel_size= 3, padding = 1)

    #Da articolo di Zhang: conversione da blocco convoluzionale in conv_downsample    
    self.blurpool5 = nn.Sequential(
        nn.Conv2d(16*K, 16*K, kernel_size= 5,stride=1,padding=1),
        nn.ReLU(),
        antialiased_cnns.BlurPool(channels=16*K, filt_size=5, stride=2) #prova filt = 3
    )

    self.max_pool = nn.AdaptiveMaxPool1d(1)

    self.drop = nn.Dropout()

    self.fc= nn.Linear(16*K, 2)

  #conv_net serve per ridurre la dimensione delle feature map in
  # slice x Channels x H x W, con H,W = 1
  def conv_net(self,x):
        x = x.permute(1,0,2,3)

        x = self.blurpool1(F.relu(self.norm(self.conv1(x))))
        x = x + self.block1(x)

        x = self.blurpool2(F.relu(self.conv2(x)))
        x = x + self.block2(x)

        x = self.blurpool3(F.relu(self.conv3(x)))
        x = x + self.block3(x)

        x = self.blurpool4(F.relu(self.conv4(x)))
        x = x +self.block4(x)

        x = self.blurpool5(F.relu(self.conv5(x)))

        x = nn.AdaptiveMaxPool2d(1)(x)
        x = self.drop(x)

        return x    
   
  def forward(self, x):

     x = self.conv_net(x) # sxCxHxW
     x = x.squeeze(3) #sx16x1
     x = x.permute(2,1,0) # 1x16kxs
     x = self.max_pool(x).squeeze(2) # 1x16K

     res = self.fc(x)
     return res

     '''
     x = self.conv1(x)
     x = self.norm1(x)
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
     '''