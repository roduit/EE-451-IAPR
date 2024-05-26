# -*- coding: utf-8 -*-
# -*- authors : Vincent Roduit -*-
# -*- date : 2024-05-03 -*-
# -*- Last revision: 2024-05-17 (Vincent) -*-
# -*- python version : 3.12.3 -*-
# -*- Description: Class to load train data -*-

# Importing libraries
import os
import matplotlib.pyplot as plt
import cv2 as cv
import pandas as pd

# Importing files
from data_classes.data import Coin
import pickle_func
import pre_processing.process_func as pf
import constants

class trainCoin(Coin):
    """
    Class to load the training data
    """
    def __init__(self, save=False):

        super().__init__(type='train', save=save)
        
        self.pickle_file_name = 'trainCoin.pkl'
        self.raw_data_pkl_name = 'raw_data.pkl'
        self.data_index_pkl_name = 'data_index.pkl'
        self.raw_data = {}
        self.data_index = {}
        self.image_masked = {}
        self.contours = {}
        self.coins = []
        self.coins_labels = []
        self.contours_tuple = []

        self.load_data()

    def load_data(self):
        """
        Load the data from the main folder

        self.data : dict where the key is the class name and the value is a list of images
        self.data_index : dict where the key is the class name and the value is a list of image names
        """
        success = False
        
        if os.path.exists(self.pickle_path):
            print('Loading data from pickle files')
            try:
                self.raw_data = pickle_func.load_pickle(file_name=self.raw_data_pkl_name, load_path=os.path.join(self.pickle_path))
                self.data_index = pickle_func.load_pickle(file_name=self.data_index_pkl_name, load_path=os.path.join(self.pickle_path))
                if ((self.raw_data is not None) and (self.data_index is not None)):
                    success = True
            except Exception as e:
                raise Exception('Pickle files not found')
        if not success: 
            print('Loading data from folders')
            folders = os.listdir(self.path)
            for folder in folders:
                folder_path = os.path.join(self.path, folder)
                if os.path.isdir(folder_path):
                    folder_name = folder.split('.')[-1]
                    folder_name = folder_name.strip()
                    self.raw_data[folder_name], self.data_index[folder_name] = self.load_images_from_folder(folder_path)

            if not os.path.exists(self.pickle_path):
                os.makedirs(os.path.join(self.pickle_path))
            pickle_func.save_pickle(result=self.raw_data, file_name=self.raw_data_pkl_name, save_path=self.pickle_path)
            pickle_func.save_pickle(result=self.data_index, file_name=self.data_index_pkl_name, save_path=self.pickle_path)
            print(f'Data saved in pickle files at {self.pickle_path}')


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
            path = os.path.join(constants.RESULT_PATH,self.type,'contours', category)
            images_names = self.data_index[category]
            self.contours[category] = pf.detect_contours(images_set, path, images_names, self.save)
    
    def create_masked_images(self):
        """
        Create the masked images
        """
        path = os.path.join(constants.RESULT_PATH,self.type, 'masked_img')
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
        path = os.path.join(constants.RESULT_PATH,self.type, 'coin_img')
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
        self.contours_tuple = coins_contours

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
                

