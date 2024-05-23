# -*- coding: utf-8 -*-
# -*- author : Vincent Roduit-*-
# -*- date : 2024-05-19 -*-
# -*- Last revision: 2024-05-19 (Vincent Roduit)-*-
# -*- python version : 3.12.3 -*-
# -*- Description: Class functions for different datasets -*-

#import libraries
from torch.utils.data import Dataset

class CoinDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        return image, label
    
class CoinDatasetRadius(Dataset):
    def __init__(self, images, labels, radius_info):
        self.images = images
        self.labels = labels
        self.radius_info = radius_info

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        radius = self.radius_info[idx]
        return image, label, radius

class CoinTestDataset(Dataset):
    def __init__(self, images):
        self.images = images
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx]
        return image

class CoinTestDatasetRadius(Dataset):
    def __init__(self, images, radius_info):
        self.images = images
        self.radius_info = radius_info

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        radius = self.radius_info[idx]
        return image, radius