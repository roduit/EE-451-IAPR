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
from tqdm import tqdm

#import files
import constants
import pickle_func

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
        images = []
        for file in os.listdir(self.path):
            if file.endswith('.JPG'):
                img = Image.open(os.path.join(self.path, file))
                img_array = np.array(img)
                if img is not None:
                    images.append(img_array)
        self.raw_data = np.array(images)
    
    def display_img(self, index):
        """
        Display the image at the given index
        Args:
            index (int): Index of the image to display
        """
        img = Image.fromarray(self.raw_data[index])
        fig1 = plt.figure(figsize=(10, 10))
        plt.imshow(img)

class trainCoin(Coin):
    """
    Class to load the training data
    """
    def __init__(self):
        super().__init__('train')
        self.load_data()

    def load_data(self):
        """
        Load the data from the main folder

        self.data : dict where the key is the class name and the value is a list of images
        self.data_index : dict where the key is the class name and the value is a list of image names
        """
        self.raw_data = {}
        self.data_index = {}

        success = False

        if os.path.exists(os.path.join(self.path, 'pickle')):
            print('Loading data from pickle files')
            try:
                self.raw_data = pickle_func.load_pickle(os.path.join(self.path, 'pickle', 'train.pkl'))
                self.data_index = pickle_func.load_pickle(os.path.join(self.path, 'pickle', 'train_index.pkl'))
                if ((self.raw_data is not None) and (self.data_index is not None)):
                    success = True
            except Exception as e:
                raise Exception('Pickle files note found')
        if not success: 
            print('Loading data from folders')
            folders = os.listdir(self.path)
            for folder in folders:
                folder_path = os.path.join(self.path, folder)
                if os.path.isdir(folder_path):
                    folder_name = folder.split('.')[-1]
                    folder_name = folder_name.strip()
                    self.raw_data[folder_name], self.data_index[folder_name] = self.load_images_from_folder(folder_path)

            if not os.path.exists(os.path.join(self.path, 'pickle')):
                os.makedirs(os.path.join(self.path, 'pickle'))
            pickle_func.save_pickle(self.raw_data, os.path.join(self.path, 'pickle', 'train.pkl'))
            pickle_func.save_pickle(self.data_index, os.path.join(self.path, 'pickle', 'train_index.pkl'))
            print('Data saved in pickle files')


    def load_images_from_folder(self, folder_path):
        images = []
        image_names = []
        for filename in os.listdir(folder_path):
            if filename.endswith(".JPG"): 
                img = Image.open(os.path.join(folder_path, filename))
                img_array = np.array(img)
                if img is not None:
                    images.append(img_array)
                    image_names.append(os.path.splitext(filename)[0])
        return images, image_names
    
    def display_img(self, category, index):
        """
        Display the image at the given index
        Args:
            category (str): Category of the image
            index (int): Index of the image to display
        """
        img = Image.fromarray(self.raw_data[category][index])
        fig1 = plt.figure(figsize=(10, 10))
        plt.imshow(img)
        
class refCoin(Coin):
    """
    Class to load the reference data
    """
    def __init__(self):
        super().__init__('ref')
        self.load_data()

class testCoin(Coin):
    """
    Class to load the test data
    """
    def __init__(self):
        super().__init__('test')
        self.load_data()