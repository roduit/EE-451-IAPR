# -*- coding: utf-8 -*-
# -*- authors : Vincent Roduit -*-
# -*- date : 2024-05-03 -*-
# -*- Last revision: 2024-05-14 -*-
# -*- python version : 3.12.3 -*-
# -*- Description: Parent class to load data -*-


# Importing libraries
import os
from torch.utils.data import Dataset
import matplotlib.pyplot as plt 
import cv2 as cv
import pandas as pd

#import files
import constants
import pickle_func
import pre_processing.process_func as pf

class Coin(Dataset):
    """
    Parent class to load data

    """
    def __init__(self, save=False, type = 'train'):
        """Constructor
        Args:
            type (str): Type of data to load. Can be 'train', 'test' or 'ref'
        """
        self.save = save
        self.raw_data = None
        self.path = None
        self.pickle_path = None
        self.data_index = None
        self.contours = []
        self.contours_tuple = []
        self.image_masked = []


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
        
        self.raw_data_file_name = self.type+str('_raw_data.pkl')
        self.data_index_file_name = self.type+str('_index.pkl')
        self.pickle_path = os.path.join(self.path, 'pickle_files')
    
    def load_data(self):
        """
        Load the data from the folder
        """
        success = False
        if os.path.exists(self.pickle_path):
            try:
                print('Loading data from pickle files')
                self.raw_data = pickle_func.load_pickle(file_name=self.raw_data_file_name , load_path=self.pickle_path)
                self.data_index = pickle_func.load_pickle(file_name=self.data_index_file_name, load_path=self.pickle_path)
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
                    img = cv.imread(os.path.join(self.path, filename))
                    img_array = img #np.array(img)
                    if img is not None:
                        images.append(img_array)
                        image_names.append(os.path.splitext(filename)[0].strip())
            self.raw_data = images
            self.data_index = image_names

            if not os.path.exists(self.pickle_path):
                os.makedirs(self.pickle_path)

            pickle_func.save_pickle(result=self.raw_data, file_name=self.raw_data_file_name, save_path=self.pickle_path)
            pickle_func.save_pickle(result=self.data_index, file_name=self.data_index_file_name, save_path=self.pickle_path)
    
    def process_images(self):
        """
        Process the images to extract the contours
        """
        images_set = self.raw_data
        path = os.path.join(constants.RESULT_PATH, self.type, 'contours')
        image_names = self.data_index
        self.contours = pf.detect_contours(images_set, path, image_names, self.save)
    
    def create_masked_images(self):
        """
        Create the masked images
        """
        path = os.path.join(constants.RESULT_PATH,self.type, 'masked_img')
        if not os.path.exists(path):
            os.makedirs(path)
        img_masked = []
        for idx, img in enumerate(self.raw_data):
            image_name = self.data_index[idx]
            img_path = os.path.join(path, f'{image_name}.png')
            circles = self.contours[idx][0]
            img_black = pf.detour_coins(img, circles)
            img_masked.append(img_black)
            if self.save:
                plt.figure()
                plt.imshow(img_black)
                plt.savefig(img_path)
                plt.close()
        self.image_masked = img_masked
                    
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
        for idx1, img in enumerate(self.image_masked):
            image_name = self.data_index[idx1]
            img_crops = pf.crop_coins(img, self.contours[idx1][0])
            for idx2, coin in enumerate(img_crops):
                coin_name = f'{image_name}_{idx2}'
                img_path = os.path.join(path, f'{image_name}_{idx2}.png')
                coins_labels.append(coin_name)
                coin_images.append((image_name, coin_name, coin))
                coins_contours.append((image_name, coin_name, self.contours[idx1][0][idx2]))
                if self.save:
                    print(coin.shape)
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
