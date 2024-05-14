# -*- coding: utf-8 -*-
# -*- authors : Vincent Roduit -*-
# -*- date : 2024-05-03 -*-
# -*- Last revision: 2024-05-14 (Vincent) -*-
# -*- python version : 3.9.18 -*-
# -*- Description: Function used for preprocessing images -*-

# Importing libraries
import numpy as np
import cv2
import os
from skimage.morphology import closing, opening, disk, remove_small_holes, remove_small_objects
from scipy import ndimage
from copy import deepcopy

# Importing files
from processing.thresholding import *

def remove_large_objects(img, max_size):
    """Remove large objects from the image
    Args: 
        img: np.array (M, N) Image
        max_size: int Maximum size of the object

    Returns:
        img_clean: np.array (M, N) Cleaned image
    """

    labeled, num_objects = ndimage.label(img)
    sizes = ndimage.sum(img, labeled, range(num_objects + 1))
    mask_size = (sizes < max_size) & (sizes > 0)
    img_clean = mask_size[labeled]
    return img_clean

def get_contours(image_set, ref_bg, path):
    """Get the contours of the coins with neutral background
    Args:
        image_set: list of np.array (M, N, 3) List of images
        ref_bg: np.array (M, N, 3) Reference background
        path: str Path to save the images
    """
    image_set_arr = np.array(image_set)
    ref_bg = np.array(ref_bg).astype(np.uint8)
    contours_list = []

    if not os.path.exists(path):
        os.makedirs(path)

    for idx, img in enumerate(image_set_arr):
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img_det, contours = detect_coin(imgRGB, min_radius=30, max_radius=100, median_tresh=31, param1=10, param2=30)
        img_path = os.path.join(path, f'img_{idx}')
        contours_list.append(contours)
        plt.figure()
        plt.imshow(img_det, interpolation='nearest', cmap='gray')
        plt.savefig(img_path)
        plt.close()

    return contours_list

def get_contours_hand(image_set, path):
    """Get the contours of the coins with hand background
    Args:
        image_set: list of np.array (M, N, 3) List of images
        path: str Path to save the images
    """

    image_set_arr = np.array(image_set)
    contours_list = []

    if not os.path.exists(path):
        os.makedirs(path)

    for idx, img in enumerate(image_set_arr):
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img_original = deepcopy(img)
        img_final = apply_hsv_threshold(img, hand_threshold)
        img_final = closing(img_final, disk(3))
        img_final = remove_small_holes(img_final, 5000)
        img_final = remove_small_objects(img_final, 5000)
        img_final = remove_large_objects(img_final, max_size=60000)
        img_final = img_final.astype(np.uint8)
        masked_img = cv2.bitwise_and(img_original, img_original, mask=img_final)
        img_contours, contours = detect_coin(masked_img, 20, 100, 3)
        plt.figure()
        plt.imshow(img_contours, interpolation='nearest', cmap='gray')
        img_path = os.path.join(path, f'img_{idx}.png')
        plt.savefig(img_path)
        plt.close()
        contours_list.append(contours)
    return contours_list

def get_contours_noisy(image_set, ref_bg, path):
    """Get the contours of the coins with noisy background
    Args:
        image_set: list of np.array (M, N, 3) List of images
        ref_bg: np.array (M, N, 3) Reference background
        path: str Path to save the images
    """
    if not os.path.exists(path):
        os.makedirs(path)

    contours_list = []

    for idx, img in enumerate(image_set):
        img = img.astype(np.uint8)
        img_original = deepcopy(img)
        img_original = cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB)

        #Processing 
        img = img - ref_bg
        img_final = rgb_threshold_noisy(img, 150, 60, 150)
        img_final = remove_small_objects(img_final, 1000)
        img_final = closing(img_final, disk(1))
        img_final = remove_small_holes(img_final, 1000)
        img_final = opening(img_final, disk(5))
        img_final = np.logical_not(img_final)
        img_final = remove_large_objects(img_final, max_size=100000)
        img_final = img_final.astype(np.uint8)
        masked_img = cv2.bitwise_and(img_original, img_original, mask=img_final)
        img_final, contours = detect_coin(masked_img, 30, 100, 5)
        contours_list.append(contours)

        # Save the image
        img_path = os.path.join(path, f'img_{idx}.png')

        plt.figure()
        plt.imshow(img_final, interpolation='nearest', cmap='gray')
        plt.savefig(img_path)
        plt.close()

    return contours_list

def calculate_ref_bg(set_images):
    """Calculate the reference background
    Args:
        set_images: list of np.array (M, N, 3) List of images

    Returns:
        np.array (M, N, 3) Reference background
    """
    imgs_array = np.array(set_images)
    imgs_mean = np.mean(imgs_array, axis=0)
    imgs_mean = imgs_mean.astype(np.uint8)
    return imgs_mean

def detect_coin(
        img: np.array, 
        min_radius: int, 
        max_radius: int, 
        median_tresh: int,
        param1: int = 35,
        param2: int = 40,
        ) -> np.array:
    """
    Detect the coins in the image

    Parameters:
    img (numpy array): Image
    min_radius (int): Minimum radius
    max_radius (int): Maximum radius

    Returns:
    numpy array: Image with the detected coins
    """
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray = cv.medianBlur(gray, median_tresh)
    circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, dp=1, minDist=120, param1=param1, param2=param2, minRadius=min_radius, maxRadius=max_radius)
    img_circles = img.copy()
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            cv.circle(img_circles, (i[0], i[1]), i[2], (255, 0, 0), 5)
    
    return img_circles, circles

def detour_coins(img, circles):
    """Detour the coins in the image
    Args:
        img: np.array (M, N) Image
        circles: np.array (N, 3) Circles coordinates
    
    Returns:
        img_black: np.array (M, N) Image with the coins detoured
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    all_center_coordinates = circles[:,:2]
    all_radius = circles[:,2]

    mask = np.zeros_like(img)

    # Draw a filled white circle in the mask
    for center_coordinates, radius in zip(all_center_coordinates, all_radius):
        center_coordinates = tuple(map(int, center_coordinates))
        radius = int(radius)
        cv2.circle(mask, center_coordinates, radius, (255,255,255), thickness=-1)

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
    all_center_coordinates = circles[:,:2]
    all_radius = circles[:,2]

    img_crops = []

    for center_coordinates, radius in zip(all_center_coordinates, all_radius):
        x1 = center_coordinates[0] - radius
        x2 = center_coordinates[0] + radius
        y1 = center_coordinates[1] - radius
        y2 = center_coordinates[1] + radius
        img_crop = img[y1:y2, x1:x2]
        img_crops.append(img_crop)

    return img_crops