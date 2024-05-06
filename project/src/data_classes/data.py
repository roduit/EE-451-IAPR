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
from IPython.display import display
import matplotlib.pyplot as plt 

#import files
import constants
import pickle_func
import processing.process_func as pf

class Coin(Dataset):
    """
    Parent class to load data

    """
    def __init__(self, type = 'train'):
        """Constructor
        Args:
            type (str): Type of data to load. Can be 'train', 'test' or 'ref'
        """

        self.raw_data = None
        self.path = None


        valide_types = ['train', 'test', 'ref']
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
    
    def load_data(self):
        """
        Load the data from the folder
        """
        success = False
        if os.path.exists(os.path.join(self.path, 'pickle')):
            try:
                print('Loading data from pickle files')
                self.raw_data = pickle_func.load_pickle(os.path.join(self.path,'pickle', self.type+str('.pkl')))
                self.data_index = pickle_func.load_pickle(os.path.join(self.path,'pickle', self.type+str('_index.pkl')))
                if ((self.raw_data is not None) and (self.data_index is not None)):
                    success = True
            except Exception as e:
                raise Exception('Pickle files note found')
        if not success:
            print('Loading data from folders')
            image_names = []
            images = []
            for filename in os.listdir(self.path):
                if filename.endswith(".JPG"): 
                    img = Image.open(os.path.join(self.path, filename))
                    img_array = np.array(img)
                    if img is not None:
                        images.append(img_array)
                        image_names.append(os.path.splitext(filename)[0].strip())
            self.raw_data = images
            self.data_index = image_names

            if not os.path.exists(os.path.join(self.path, 'pickle')):
                os.makedirs(os.path.join(self.path, 'pickle'))

            pickle_func.save_pickle(self.raw_data, os.path.join(self.path,'pickle',self.type+str('.pkl')))
            pickle_func.save_pickle(self.data_index, os.path.join(self.path,'pickle',self.type+str('_index.pkl')))

    
    def display_img(self, index, raw = True):
        """
        Display the image at the given index
        Args:
            index (int): Index of the image to display
        """
        if raw:
            img = Image.fromarray(self.raw_data[index])
        else:
            img = Image.fromarray(self.processed_data[index])
        fig1 = plt.figure(figsize=(10, 10))
        plt.imshow(img)
    
    def process_data(self):
        """
        Process the data
        """
        self.processed_data = []
        for img in self.raw_data:
            gray_img = pf.rgb_to_gray(img)
            filtered_img = pf.apply_median(gray_img) * 255
            self.processed_data.append(filtered_img)