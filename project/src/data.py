# -*- coding: utf-8 -*-
# -*- authors : Vincent Roduit -*-
# -*- date : 2024-05-03 -*-
# -*- Last revision: 2024-05-03 -*-
# -*- python version : 3.9.18 -*-
# -*- Description: Class to load data -*-


# Importing libraries
import numpy as np
import os
from torch.utils.data import Dataset
from PIL import Image

#import files
import constants

class Coin(Dataset):
    def __init__(self, type = 'train'):
        valide_types = ['train', 'test','ref']
        if type not in valide_types:
            raise ValueError(f"Invalid type. Expected one of {valide_types}")
        else:
            self.type = type
        if self.type == 'train':
            self.path = constants.TRAIN
        elif self.type == 'test':
            self.path = constants.TEST
        else:
            self.path = constants.REF
        
        self.raw_data = self.load_data()
    
    def load_data(self):
        images = []
        for file in os.listdir(self.path):
            if file.endswith('.jpg'):
                img = Image.open(os.path.join(self.path, file))
                print(img)
                if img is not None:
                    images.append(img)
        return images