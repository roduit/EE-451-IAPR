# -*- coding: utf-8 -*-
# -*- authors : Vincent Roduit -*-
# -*- date : 2024-05-03 -*-
# -*- Last revision: 2024-05-14 (Vincent) -*-
# -*- python version : 3.9.18 -*-
# -*- Description: Function used for preprocessing images -*-

# Importing libraries
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os
# Importing files
from processing.morphology import apply_closing, remove_objects

def detect_contours_single_img(img, path, save):

    std = np.std(cv.cvtColor(255 -img, cv.COLOR_BGR2GRAY))
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = cv.resize(img, (0,0), fx=0.25, fy=0.25)
    img_copy = img.copy()

    img[:,:,0] = img[:,:,0]*0.2 #red
    img[:,:,1] = img[:,:,1]*0.2 #green
    img[:,:,2] = img[:,:,2]*1 #blue 

    red_mean = np.mean(img_copy[:,:,0]) 

    imgGray = cv.cvtColor(255 - img, cv.COLOR_BGR2GRAY)

    if std < 18: #neutral
        remove_objects_size = 10
        thres2 = 1
        sigma = 2.7
        p2 = 32
        th_sigma = 2
        open_th = 3
    
    else: #hand
        remove_objects_size = 120
        if red_mean > 184:
            thres2 = 4
            sigma = 2.7
            p2 = 32
            th_sigma = 2.5
            open_th = 6
        else: # noisy
            thres2 = 1
            sigma = 2.7
            p2 = 31
            th_sigma = 1.8
            open_th = 4

    low_pass = cv.GaussianBlur(imgGray, (45, 45), sigma)
    high_pass = cv.subtract(imgGray, low_pass) 
    _, thresholded = cv.threshold(high_pass, thres2, 255, cv.THRESH_BINARY)

    thresholded = np.uint8(remove_objects(thresholded,remove_objects_size) * 255)
    thresholded_open = apply_closing(thresholded, open_th) # 3
    thresholded_open = cv.GaussianBlur(thresholded_open, (45, 45), th_sigma) # 2

    circles = cv.HoughCircles(thresholded_open, cv.HOUGH_GRADIENT, dp=1, minDist=50, param1=15, param2=p2, minRadius=40, maxRadius=120) # 32
    if save and circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            cv.circle(img_copy, (i[0], i[1]), i[2], (0, 0, 0), 10)
        
        #save figure
        plt.figure()
        plt.imshow(img_copy, interpolation='nearest', cmap='gray')
        plt.savefig(path)
        plt.close()
        

    return circles

def detect_contours(imgs, path, save):
    """Detect the contours of the coins in the images
    Args:
        imgs: list of np.array (M, N) List of images
        path: str Path to save the images
        save: bool Save the images or not
    
    Returns:
        circles: list of np.array (N, 3) List of circles coordinates
    """
    if not os.path.exists(path):
        os.makedirs(path)

    all_contours = []
    for i, img in enumerate(imgs):
        image_path = os.path.join(path,f'_img_{i}.png')
        contours = detect_contours_single_img(img, image_path, save)
        all_contours.append(contours)
    return all_contours

def detour_coins(img, circles):
    """Detour the coins in the image
    Args:
        img: np.array (M, N) Image
        circles: np.array (N, 3) Circles coordinates
    
    Returns:
        img_black: np.array (M, N) Image with the coins detoured
    """
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    all_center_coordinates = circles[:,:2] * 4
    all_radius = circles[:,2] * 4

    mask = np.zeros_like(img)

    # Draw a filled white circle in the mask
    for center_coordinates, radius in zip(all_center_coordinates, all_radius):
        center_coordinates = tuple(map(int, center_coordinates))
        radius = int(radius)
        cv.circle(mask, center_coordinates, radius, (255,255,255), thickness=-1)

    mask = mask.astype(bool)

    img_black = np.zeros_like(img)

    np.copyto(img_black, img, where=mask)

    return img_black

def crop_coins(img, circles):
    """Crop the coins from the image
    Args:
        img: np.array (M, N) Image
        circles: np.array (N, 3) Circles coordinates
    
    Returns:
        img_crops: list of np.array (M, N) List of cropped images
    """
    all_center_coordinates = circles[:,:2] * 4
    all_radius = circles[:,2] * 4

    img_crops = []

    for center_coordinates, radius in zip(all_center_coordinates, all_radius):
        x1 = center_coordinates[0] - radius
        x2 = center_coordinates[0] + radius
        y1 = center_coordinates[1] - radius
        y2 = center_coordinates[1] + radius
        img_crop = img[y1:y2, x1:x2]
        img_crops.append(img_crop)

    return img_crops