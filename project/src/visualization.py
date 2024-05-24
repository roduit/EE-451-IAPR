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
    for key in keys:
        if 'outliers' in key:
            non_outlier_key = key.replace('_outliers', '')
            if non_outlier_key in raw_data:
                del raw_data[key]
    return keys

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
    axs[3].set_title('Contrast')

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

