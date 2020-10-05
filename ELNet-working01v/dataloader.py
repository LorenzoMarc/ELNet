import os
import pandas as pd
import numpy as np

import torch
import torch.nn.functional as F
import torch.utils.data as data
import utils as ut
import torchvision.transforms as transform
import random

# PyTorch class to load the ELNet dataset

class ELDataset(data.Dataset):
    def __init__(self, root_dir, task, plane, train=True, weights=None):
        super().__init__()
        self.task = task
        self.plane = plane
        self.root_dir = root_dir
        self.train = train
        if self.train:
            self.folder_path = self.root_dir + 'train/{0}/'.format(plane)
            self.records = pd.read_csv(
                self.root_dir + 'train-{0}.csv'.format(task), header=None, names=['id', 'label'])
        else:
            self.folder_path = self.root_dir + 'valid/{0}/'.format(plane)
            self.records = pd.read_csv(
                self.root_dir + 'valid-{0}.csv'.format(task), header=None, names=['id', 'label'])

        self.records['id'] = self.records['id'].map(
            lambda i: '0' * (4 - len(str(i))) + str(i))
        self.paths = [self.folder_path + filename +
                      '.npy' for filename in self.records['id'].tolist()]
        self.labels = self.records['label'].tolist()

        if weights is None:
            neg_weight = np.mean(self.labels)
            self.weights = torch.FloatTensor([neg_weight, 1 - neg_weight])
        else:
            self.weights = torch.FloatTensor(weights)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        array = np.load(self.paths[index])
        
        #label = torch.FloatTensor([self.labels[index]])
        label = torch.LongTensor([self.labels[index]])

        weight = torch.FloatTensor([self.weights[self.labels[index]]])

        if self.train:
          # data augmentation
          array = ut.random_shift(array, 10)
          array = ut.random_rotate(array, 10)
          array = ut.random_flip(array)
        if self.plane == 'axial' or self.plane == 'coronal':
           array = ut.random_rotate(array, 90)
           '''
           l = [90,180,270]
           rand_rotation = transform.RandomRotation(random.choice(l))
           array = rand_rotation(array)
           '''

        # data standardization
        #array = (array - 58.09) / 49.73   # ???? no needs cause LayerNorm layer 
        #array = np.stack((array,), axis=1)

        array = torch.FloatTensor(array) # array size is now [S, 256, 256,1]

        return array, label, weight

