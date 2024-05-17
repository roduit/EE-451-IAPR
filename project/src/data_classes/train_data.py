# -*- coding: utf-8 -*-
# -*- authors : Vincent Roduit -*-
# -*- date : 2024-05-03 -*-
# -*- Last revision: 2024-05-17 (Vincent) -*-
# -*- python version : 3.9.18 -*-
# -*- Description: Class to load data -*-

# Importing libraries
import os
import matplotlib.pyplot as plt
import cv2 as cv
import pandas as pd

# Importing files
from data_classes.data import Coin
import pickle_func
import processing.process_func as pf
import constants

class trainCoin(Coin):
    """
    Class to load the training data
    """
    def __init__(self, save=False, load_from_pickle=False):

        super().__init__('train')
        
        self.from_pickle = load_from_pickle
        self.pickle_path = os.path.join(constants.RESULT_PATH, 'pickles')
        self.pickle_file_name = 'trainCoin.pkl'
        self.save = save
        self.raw_data = {}
        self.data_index = {}
        self.ref_bg = {}
        self.image_masked = {}
        self.contours = {}
        self.coins = []
        self.coins_labels = []

        self.load_data()

    def load_data(self):
        """
        Load the data from the main folder

        self.data : dict where the key is the class name and the value is a list of images
        self.data_index : dict where the key is the class name and the value is a list of image names
        """
        success = False
        if self.from_pickle:
            print('Loading class from pickle')
            self.load_pickle()
            return
        
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
                #img = cv.resize(img, (0,0), fx=0.25, fy=0.25)
                if img is not None:
                    images.append(img)
                    image_names.append(os.path.splitext(filename)[0])
        return images, image_names

    def process_images(self):
        """
        Process the images to extract the contours
        """
        for category in self.raw_data:
            images_set = self.raw_data[category]
            path = os.path.join(constants.RESULT_PATH, category)
            self.contours[category] = pf.detect_contours(images_set, path, self.save)
    
    def create_masked_images(self):
        """
        Create the masked images
        """
        path = os.path.join(constants.RESULT_PATH, 'masked_img')
        if not os.path.exists(path):
            os.makedirs(path)

        for category in self.raw_data:
            self.image_masked[category] = []
            for idx, img in enumerate(self.raw_data[category]):
                image_name = self.data_index[category][idx]
                img_path = os.path.join(path, f'{image_name}.png')
                circles = self.contours[category][idx][0]
                img_black = pf.detour_coins(img, circles)
                self.image_masked[category].append(img_black)
                if self.save:
                    plt.figure()
                    plt.imshow(img_black)
                    plt.savefig(img_path)
                    plt.close()
                    
    def create_coin_images(self):
        """
        Create the images with only the coins
        """
        path = os.path.join(constants.RESULT_PATH, 'coin_img')
        if not os.path.exists(path):
            os.makedirs(path)
            
        coin_images = []
        coins_labels = []
        coins_contours = []
        for category in self.image_masked:
            for idx1, img in enumerate(self.image_masked[category]):
                image_name = self.data_index[category][idx1]
                img_crops = pf.crop_coins(img, self.contours[category][idx1][0])
                for idx2, coin in enumerate(img_crops):
                    coin_name = f'{image_name}_{idx2}'
                    img_path = os.path.join(path, f'{image_name}_{idx2}.png')
                    coins_labels.append(coin_name)
                    coin_images.append((image_name, coin_name, coin))
                    coins_contours.append((image_name, coin_name, self.contours[category][idx1][0][idx2]))
                    if self.save:
                        plt.figure()
                        plt.imshow(coin)
                        plt.savefig(img_path)
                        plt.close()
        self.coins = coin_images
        self.coins_labels = coins_labels
        self.contours = coins_contours

        #save labels as xls
        df = pd.DataFrame(self.coins_labels)
        df.columns = ['image_name']
        df.sort_values('image_name', inplace=True)
        df.to_excel(os.path.join(path, 'coin_labels.xlsx'), index=False)

    def proceed_data(self):
        """
        Process the data
        """
        print('Finding contours')
        self.process_images()
        print('Creating masked images')
        self.create_masked_images()
        print('Creating coin images')
        self.create_coin_images()
        if self.save:
            print('Saving class')
            self.save_class()

    def save_class(self):
        """
        Save the class as pickle file
        """
        if not os.path.exists(self.pickle_path):
            os.makedirs(self.pickle_path)
        pickle_func.save_pickle(self, os.path.join(self.pickle_path))
    
    def load_pickle(self):
        """
        Load the class from pickle
        """
        try:
            self = pickle_func.load_pickle(os.path.join(self.pickle_path, self.pickle_file_name))
        except Exception as e:
            raise Exception('Pickle file not found')
                

