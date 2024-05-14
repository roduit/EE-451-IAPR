# -*- coding: utf-8 -*-
# -*- authors : Vincent Roduit -*-
# -*- date : 2024-05-03 -*-
# -*- Last revision: 2024-05-03 -*-
# -*- python version : 3.9.18 -*-
# -*- Description: Class to load data -*-

# Importing libraries
import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import cv2 as cv

# Importing files
from data_classes.data import Coin
import pickle_func
import processing.process_func as pf
import constants

class trainCoin(Coin):
    """
    Class to load the training data
    """
    def __init__(self):

        self.ref_bg = {}
        self.raw_data = {}
        self.data_index = {}

        super().__init__('train')
        self.load_data()
        self.compute_ref_bg()

    def load_data(self):
        """
        Load the data from the main folder

        self.data : dict where the key is the class name and the value is a list of images
        self.data_index : dict where the key is the class name and the value is a list of image names
        """

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
                img = cv.imread(os.path.join(folder_path, filename))
                img = cv.resize(img, (0,0), fx=0.25, fy=0.25)
                if img is not None:
                    images.append(img)
                    image_names.append(os.path.splitext(filename)[0])
        return images, image_names
    
    # def compute_ref_bg(self):
    #     """
    #     Compute the reference background for each class
    #     """
    #     categories = ['hand', 'noisy_bg', 'neutral_bg']

    #     for category in categories:
    #         data = self.raw_data[category] + self.raw_data[f'{category}_outliers']
    #         self.ref_bg[category] = pf.calculate_ref_bg(data)
    #         self.ref_bg[f'{category}_outliers'] = pf.calculate_ref_bg(data)

    def compute_ref_bg(self):
        """
        Compute the reference background for each class
        """

        for category in self.raw_data:
            data = self.raw_data[category]
            self.ref_bg[category] = pf.calculate_ref_bg(data)

    
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

    def process_images(self):
        """
        Process the images to extract the contours
        """
        hand_category = ['hand', 'hand_outliers']
        neutral_category = ['neutral_bg', 'neutral_bg_outliers']
        for category in self.raw_data:
            images_set = self.raw_data[category]
            background = self.ref_bg[category]
            path = os.path.join(constants.RESULT_PATH, category)
            if category in hand_category:
                pf.get_contours_hand(images_set, path)
            elif category in neutral_category:
                pf.get_contours(images_set, background, path)
            else:
                pf.get_contours_noisy(images_set, background, path)

