# -*- coding: utf-8 -*-
# -*- authors : Vincent Roduit -*-
# -*- date : 2024-05-05 -*-
# -*- Last revision: 2024-05-05 -*-
# -*- python version : 3.12.3 -*-
# -*- Description: Function to visualize results -*-

# Importing libraries
import os
import matplotlib.pyplot as plt
import cv2 as cv
from copy import deepcopy
import numpy as np
# Importing files
import constants
from post_processing.data_augmentation import *
from pre_processing.process_func import generate_mask
from pre_processing.morphology import apply_closing, remove_objects



def plot_side_by_side(original_images, processed_images, index, title1, title2):
    """Plot side by side the original and processed images.
    
    Args:
        Original_images: dict Dictionary containing the original images.
        Processed_images: dict Dictionary containing the processed images.
        Index: int Index of the image to plot.
        Title1: str Title of the first image.
        Title2: str Title of the second image.
    """

    categories = get_single_cat(original_images)

    fig, axs = plt.subplots(len(categories), 2, figsize=(10, 8))

    for i, category in enumerate(categories):
        axs[i, 0].imshow(cv.cvtColor(original_images[category][index], cv.COLOR_BGR2RGB))
        axs[i, 0].set_title(title1)
        axs[i, 0].axis('off')

        axs[i, 1].imshow(processed_images[category][index])
        axs[i, 1].set_title(title2)
        axs[i, 1].axis('off')

def save_coins_classified(df_images_labels, images, type='train'):
    """ Save the classified coins in the corresponding folder.
    
    Args:
        df_images_labels: pd.DataFrame Dataframe containing the images and labels.
        Images: list List of images.
        Type: str Type of the images (train or test).
    """
    classification_path = os.path.join(constants.RESULT_PATH, type, 'coins_classified')

    for category in df_images_labels['label'].unique():
        category_for_path = category.replace('.', '_')
        category_path = os.path.join(classification_path, category_for_path)
        if not os.path.exists(category_path):
            os.makedirs(category_path)

        for idx, img in df_images_labels[df_images_labels['label'] == category].iterrows():
            img_path = os.path.join(category_path, img['coin_name'])
            plt.imshow(images[idx])
            plt.savefig(img_path)
            plt.close()

def get_single_cat(raw_data):
    """Get the categories of images and remove the outliers categories.
    
    Args:
        raw_data: dict Dictionary containing the images.
    
    Returns:
        keys: list List of categories.
    """
    keys = list(raw_data.keys())
    new_keys = []
    for key in keys:
        if not '_outliers' in key:
            new_keys.append(key)                
    return new_keys

def construct_contour_images(images, contours):
    """Construct the images with the contours.

    Args:
        images: dict Dictionary containing the images.
        contours: dict Dictionary containing the contours.
    
    Returns:
        contour_images: dict Dictionary containing the images with the contours.
    """
    contour_images = {}
    raw_data = deepcopy(images)
    contours = deepcopy(contours)
    for category in raw_data:
        contours_cat = []
        for idx, img in enumerate(raw_data[category]):
            contours_img = contours[category][idx]
            contours_img = [np.array(contour, dtype=np.int32) for contour in contours_img]
            img_with_contour = draw_circles(image=img, contours=contours_img)
            contours_cat.append(img_with_contour)
        contour_images[category] = contours_cat
    return contour_images

def draw_circles(image, contours):
    """Draw the circles in the image.
    
    Args:
        image: np.array (M, N) Image
        contours: list of np.array (N, 3) List of circles coordinates
    
    Returns:
        image: np.array (M, N) Image with the circles drawn
    """
    for contour in contours:
        for i in contour:
            x, y, radius = i * 4
            cv.circle(image, (x, y), radius, (0, 0, 255), 15)
    return cv.cvtColor(image, cv.COLOR_BGR2RGB)

def plot_crop_coins(coin_images):

    indexes = np.random.randint(0, len(coin_images), 4)

    fig, axs = plt.subplots(1, 4, figsize=(15, 5))

    for i, idx in enumerate(indexes):
        axs[i].imshow(coin_images[idx][2])
        axs[i].axis('off')
        axs[i].set_title(f'Coin {idx}')
    fig.suptitle('Cropped coins', fontsize=20)
    plt.show()

def display_augmentation(image):
    """display the different augmentations
    
    Args:
        image: np.array (M, N) Image
    """

    image_rotated = rotate_image(image, 45)
    image_blurred = blur_image(image, 5)
    image_hist_eq = histogram_equalization(image)
    image_contrast = gamma_correction(image, 1.5)

    fig, axs = plt.subplots(1, 4, figsize=(15, 5))

    axs[0].imshow(image_rotated)
    axs[0].axis('off')
    axs[0].set_title('Rotated')

    axs[1].imshow(image_blurred)
    axs[1].axis('off')
    axs[1].set_title('Blurred')

    axs[2].imshow(image_hist_eq)
    axs[2].axis('off')
    axs[2].set_title('Histogram Equalization')

    axs[3].imshow(image_contrast)
    axs[3].axis('off')
    axs[3].set_title('Gamma Correction')

    plt.show()

def display_cluster(images, labels, index):

    images_cluster = [coin for i, coin in enumerate(images) if labels[i] == index]

    # Calculate the number of rows needed
    num_images = len(images_cluster)
    num_cols = 5
    num_rows = (num_images + num_cols - 1) // num_cols  # Ensure all images fit into the grid

    # Create subplots
    fig, ax = plt.subplots(num_rows, num_cols, figsize=(20, 20))

    # Plot the images
    for i in range(num_rows * num_cols):
        if i < num_images:
            ax[i // num_cols, i % num_cols].imshow(images_cluster[i], cmap='gray')
            ax[i // num_cols, i % num_cols].axis('off')
            ax[i // num_cols, i % num_cols].set_title(f'Coin {i+1}')
        else:
            ax[i // num_cols, i % num_cols].axis('off')  # Turn off any unused subplots

    # Adjust layout to avoid overlap
    plt.tight_layout()
    plt.show()

def display_generate_mask(img):
    """Display the generated mask

    Args:
        img: np.array (M, N) Image
    """
    oringinal_img = img.copy()  
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = cv.resize(img, (0,0), fx=0.25, fy=0.25)
    mask = generate_mask(img)

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    axs[0].imshow(cv.cvtColor(oringinal_img, cv.COLOR_BGR2RGB))
    axs[0].axis('off')
    axs[0].set_title('Original Image')

    axs[1].imshow(mask)
    axs[1].axis('off')
    axs[1].set_title('Generated Mask')

    plt.show()

def display_img_processing(img):
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

    original_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    imgGray = cv.cvtColor(255 - img, cv.COLOR_BGR2GRAY)

    low_pass = cv.GaussianBlur(imgGray, (45, 45), sigma)
    high_pass = cv.subtract(imgGray, low_pass)
    _, thresholded = cv.threshold(high_pass, thres2, 255, cv.THRESH_BINARY)

    #thresholded = np.uint8(remove_objects(thresholded, 16) * 255)
    thresholded_rm = np.uint8(remove_objects(thresholded,remove_objects_size) * 255)
    thresholded_closing = apply_closing(thresholded_rm, open_th) # 3
    thresholded_blur = cv.GaussianBlur(thresholded_closing, (45, 45), th_sigma) # 2

    fig, axs = plt.subplots(2, 3, figsize=(10, 5))

    axs[0, 0].imshow(original_gray, cmap='gray')
    axs[0, 0].axis('off')
    axs[0, 0].set_title('Gray Image')

    axs[0, 1].imshow(low_pass, cmap='gray')
    axs[0, 1].axis('off')
    axs[0, 1].set_title('Low Pass')

    axs[0, 2].imshow(high_pass, cmap='gray')
    axs[0, 2].axis('off')
    axs[0, 2].set_title('High Pass')

    axs[1, 0].imshow(thresholded, cmap='gray')
    axs[1, 0].axis('off')
    axs[1, 0].set_title('Thresholded')

    axs[1, 1].imshow(thresholded_rm, cmap='gray')
    axs[1, 1].axis('off')
    axs[1, 1].set_title('Object Removed')

    axs[1, 2].imshow(thresholded_blur, cmap='gray')
    axs[1, 2].axis('off')
    axs[1, 2].set_title('Thresholded Blur')

    plt.show()
