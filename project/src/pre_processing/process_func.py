# -*- coding: utf-8 -*-
# -*- authors : Filippo Quadri -*-
# -*- date : 2024-05-03 -*-
# -*- Last revision: 2024-05-17 (Filippo Quadri) -*-
# -*- python version : 3.12.3 -*-
# -*- Description: Function used for preprocessing images -*-

# Importing libraries
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os

# Importing files
from pre_processing.morphology import apply_closing, remove_objects


def enhance_blue_channel(image):
    """
    Enhance the blue channel and suppress other channels.

    Args
    ----
    image: np.ndarray (M, N, 3)
        BGR image of size MxN.

    Return
    ------
    enhanced_image: np.ndarray (M, N, 3)
        Image with enhanced blue channel
    """
    enhanced_image = image.copy()
    enhanced_image[:,:,0] = np.clip(image[:,:,0] * 1.3, 0, 255)  # Enhance Blue channel
    enhanced_image[:,:,2] = np.clip(image[:,:,2] * 1, 0, 255)  # Suppress Red channel
    enhanced_image[:,:,1] = np.clip(image[:,:,1] * 1, 0, 255)  # Suppress Green channel

    return enhanced_image

def generate_mask(img):

    image_mask = img.copy()
    image_mask = cv.cvtColor(image_mask, cv.COLOR_RGB2BGR)

    lower_blue_range = [30, 80, 80]  # Lower bound of the HSV range for blue 20, 80, 80
    upper_blue_range = [160, 255, 255]  # Upper bound of the HSV range for blue
    
    enhanced_image = enhance_blue_channel(image_mask)
    
    # Convert to HSV color space
    #hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the lower and upper range of the blue color in HSV
    lower_range = np.array(lower_blue_range, dtype=np.uint8)
    upper_range = np.array(upper_blue_range, dtype=np.uint8)
    
    # Create a mask that identifies the blue coins
    mask = cv.inRange(enhanced_image, lower_range, upper_range)

    # Apply opening to the mask
    mask = np.uint8(remove_objects(mask, 100) * 255)

    kernel = np.ones((2, 2), np.uint8)
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)

    # apply closing to the mask
    kernel = np.ones((7, 7), np.uint8)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)

    # Apply Gaussian blur to the mask
    mask = cv.GaussianBlur(mask, (13, 13), 0)

    circles = cv.HoughCircles(mask, cv.HOUGH_GRADIENT, dp=1, minDist=50, param1=10, param2=25, minRadius=40, maxRadius=120) # 32
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            cv.circle(image_mask, (i[0], i[1]), i[2], (0, 0, 0), 2)
    
    image_mask = cv.cvtColor(image_mask, cv.COLOR_BGR2RGB)

    return image_mask


def detect_contours_single_img(img, path, save, size = 45):

    std = np.std(cv.cvtColor(255 - img, cv.COLOR_BGR2GRAY))
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = cv.resize(img, (0,0), fx=0.25, fy=0.25)
    img_copy = img.copy()

    # canny edge detection
    img_edges_std = np.std(cv.Canny(cv.cvtColor(img, cv.COLOR_BGR2GRAY), 50, 200))

    if std < 18: #neutral
        title = "Neutral"
        remove_objects_size = 10
        thres2 = 1
        sigma = 2.7
        p2 = 32
        th_sigma = 2.5
        open_th = 3
    
    else:
        remove_objects_size = 120
        if img_edges_std < 50: #hand
            title = "Hand"
            thres2 = 3
            sigma = 2.7
            p2 = 38
            th_sigma = 2.7 # 2.7
            open_th = 6 # 6
        else: # noisy
            title = "Noisy"
            thres2 = 2 # 1
            sigma = 2.7 # 2.7
            p2 = 32 # 30
            th_sigma = 2.2 # 2.2
            open_th = 5 # 3
    
    if title == "Noisy":
        img = generate_mask(img)

    #print(idx, " -", title, ": edges_std:", img_edges_std, "edges_mean:", img_edges_mean, "compactness:", compactness)

    img[:,:,0] = img[:,:,0]*0.25 #red
    img[:,:,1] = img[:,:,1]*0.25 #green
    img[:,:,2] = img[:,:,2]*1 #blue 

    imgGray = cv.cvtColor(255 - img, cv.COLOR_BGR2GRAY)

    low_pass = cv.GaussianBlur(imgGray, (size, size), sigma)
    high_pass = cv.subtract(imgGray, low_pass)
    _, thresholded = cv.threshold(high_pass, thres2, 255, cv.THRESH_BINARY)

    #thresholded = np.uint8(remove_objects(thresholded, 16) * 255)
    thresholded = np.uint8(remove_objects(thresholded,remove_objects_size) * 255)
    thresholded_closing = apply_closing(thresholded, open_th) # 3
    thresholded_blur = cv.GaussianBlur(thresholded_closing, (size, size), th_sigma) # 2

    circles = cv.HoughCircles(thresholded_blur, cv.HOUGH_GRADIENT, dp=1, minDist=50, param1=15, param2=p2, minRadius=40, maxRadius=120) # 32
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

# apply the function to all the images in the folder
def detect_contours(imgs, path, images_names, save):
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
        image_name = images_names[i]
        image_path = os.path.join(path,f'{image_name}.png')
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

import numpy as np

def crop_coins(img, circles):
    """Crop the coins from the image.
    Args:
        img: np.array (H, W, C) Image
        circles: np.array (N, 3) Circles coordinates
    
    Returns:
        img_crops: list of np.array (h, w, C) List of cropped images
    """
    all_center_coordinates = circles[:, :2] * 4
    all_radius = circles[:, 2] * 4

    img_crops = []
    img_height, img_width = img.shape[:2]

    for center_coordinates, radius in zip(all_center_coordinates, all_radius):
        x1 = max(int(center_coordinates[0] - radius), 0)
        x2 = min(int(center_coordinates[0] + radius), img_width)
        y1 = max(int(center_coordinates[1] - radius), 0)
        y2 = min(int(center_coordinates[1] + radius), img_height)
        
        # Ensure the dimensions are valid (x2 > x1 and y2 > y1)
        if x2 > x1 and y2 > y1:
            img_crop = img[y1:y2, x1:x2]
            img_crops.append(img_crop)

    return img_crops
