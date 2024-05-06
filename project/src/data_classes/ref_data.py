# -*- coding: utf-8 -*-
# -*- authors : Vincent Roduit -*-
# -*- date : 2024-05-03 -*-
# -*- Last revision: 2024-05-03 -*-
# -*- python version : 3.9.18 -*-
# -*- Description: Class to load data -*-

# Importing files
from data_classes.data import Coin
import processing.process_func as pf
import processing.morphology as morph
import constants

class refCoin(Coin):
    """
    Class to load the reference data
    """
    def __init__(self):
        super().__init__('ref')
        self.load_data()

        self.gray_imgs = []
        self.thresholds = []
        self.th_imgs = []
    
    def create_gray_imgs(self):
        for img in self.raw_data:
            gray_img = pf.rgb_to_gray(img)
            filtered_img = pf.apply_median(gray_img) * 255
            self.gray_imgs.append(filtered_img)

    def find_thresholds(self):
        self.threshold = []
        for img in self.gray_imgs:
            img_th, threshold = pf.find_threshold(img)
            self.th_imgs.append(img_th)
            self.thresholds.append(threshold)

    def create_contours(self):
        for img in self.th_imgs:
            closed_img = morph.apply_opening(img, constants.DISK_SIZE)
            self.processed_data.append(closed_img)
        self.contours = pf.find_contour(self.processed_data)