from torch.utils.data.dataset import Dataset

import os
import torch
import torch.nn.functional as F
import fnmatch
import numpy as np
import random
import json


class CityScapes(Dataset):

    def __init__(self, root, mode='train'):
        self.mode = mode
        self.root = os.path.expanduser(root)
        
        # read the data file
        if self.mode == 'train':
            data_len = len(fnmatch.filter(os.listdir(self.root + '/train/image'), '*.npy'))
            self.index_list = list(range(data_len))
            self.data_path = self.root + '/train'
        elif self.mode == 'test':
            data_len = len(fnmatch.filter(os.listdir(self.root + '/val/image'), '*.npy'))
            self.index_list = list(range(data_len))
            self.data_path = self.root + '/val'

    def __getitem__(self, i):
        index = self.index_list[i]
        # load data from the pre-processed npy files
        image = torch.from_numpy(np.moveaxis(np.load(self.data_path + '/image/{:d}.npy'.format(index)), -1, 0))
        semantic = torch.from_numpy(np.load(self.data_path + '/label/{:d}.npy'.format(index)))
        depth = torch.from_numpy(np.moveaxis(np.load(self.data_path + '/depth/{:d}.npy'.format(index)), -1, 0))

        return image.float(), {'segmentation': semantic.float(), 'depth': depth.float()}

    def __len__(self):
        return len(self.index_list)
