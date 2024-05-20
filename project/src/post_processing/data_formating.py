# -*- coding: utf-8 -*-
# -*- author : Vincent Roduit-*-
# -*- date : 2024-05-19 -*-
# -*- Last revision: 2024-05-19 (Vincent Roduit)-*-
# -*- python version : 3.9.18 -*-
# -*- Description: Functions used to format data suitable for Neural Net -*-

#import libraries
import os
import pandas as pd
import numpy as np
import cv2 as cv

#import files
import constants

def get_classes_conv_table():
    """Create a conversion table for the classes
    Returns:

        DataFrame: classes conversion table
    """
    path_train_labels = os.path.join(constants.DATA, 'train_labels.csv')
    df_train_labels = pd.read_csv(path_train_labels)
    classes = df_train_labels.columns[1:]
    classes_conversion_table = {classes[i].strip(): i for i in range(len(classes))}
    return classes_conversion_table

def get_coin_labels():
    """Get the coin labels
    Returns:

        DataFrame: DF containing the coin labels with the coin name (image_name_idx)
    """
    coin_labels_path = os.path.join(constants.RESULT_PATH, 'coin_img', 'coin_labels_complete.xlsx')
    coin_labels = pd.read_excel(coin_labels_path)
    coin_labels.dropna(subset=['Label'], inplace=True)
    return coin_labels

def create_train_data_structure(coins, contours, coin_labels, conversion_table):
    """Create the train data structure for the model
    Args:
    coins (list): list of coins
    contours (list): list of contours
    coin_labels (DataFrame): coin labels
    conversion_table (DataFrame): conversion table for the classes
    Returns:

        tuple: images, labels, df_images_labels
    """
    images= []
    images_names = []
    coin_names = []
    images_labels = []
    for image_name, coin_name, image in coins:
        if coin_name in coin_labels['image_name'].values:
            images.append(image)
            images_names.append(image_name)
            coin_names.append(coin_name)
            images_labels.append(coin_labels[coin_labels['image_name'] == coin_name]['Label'].values[0])
            

    images = pad_images(images) 
    images = resize_images(images, constants.IMG_SIZE)     


    df_images_labels = pd.DataFrame({'image_name': images_names,'coin_name':coin_names, 'label': images_labels})
    df_images_labels['label_int'] = df_images_labels['label'].apply(lambda x: conversion_table[x])
    df_contours = pd.DataFrame(contours, columns=['image_name', 'coin_name', 'contour'])
    df_images_labels = df_images_labels.merge(df_contours)
    df_images_labels['radius'] = df_images_labels['contour'].apply(lambda x: x[2])
    labels = df_images_labels['label_int'].values
    radius_info = df_images_labels['radius'].values


    return np.array(images), np.array(radius_info),np.array(labels), df_images_labels

def create_test_data_structure(coins, contours):
    """Create the train data structure for the model
    Args:
    coins (list): list of coins
    contours (list): list of contours
    conversion_table (DataFrame): conversion table for the classes
    Returns:

        tuple: images, radius_info, df_images_labels
    """
    images= []
    images_names = []
    coin_names = []
    for image_name, coin_name, image in coins:
        images.append(image)
        images_names.append(image_name)
        coin_names.append(coin_name)
        
    images = pad_images(images)    
    images = resize_images(images, constants.IMG_SIZE)

    df_images = pd.DataFrame({'image_name': images_names,'coin_name':coin_names})
    df_contours = pd.DataFrame(contours, columns=['image_name', 'coin_name', 'contour'])
    df_images = df_images.merge(df_contours)
    df_images['radius'] = df_images['contour'].apply(lambda x: x[2])
    radius_info = df_images['radius'].values
    return np.array(images), np.array(radius_info), df_images

def resize_images(images, size):
    """Resize the images
    Args:
        images (list): list of images
        size (tuple): size of the images
    Returns:

        np.array: resized images
    """
    resized_images = [cv.resize(image, (size,size)) for image in images]
    return np.array(resized_images)

def pad_images(images):
    """Pad the images to the biggest image
    Args:
        images (list): list of images
    
    Returns:
        images_padded : list of padded images
    """

    # Find the dimensions of the biggest image
    biggest_dim = max(max(image.shape[:2]) for image in images)
    images_padded = []
    for img in images:
        # Calculate padding dimensions
        height, width = img.shape[:2]
        pad_height = biggest_dim - height
        pad_width = biggest_dim - width

        # Distribute padding evenly on both sides of the image
        pad_top = pad_height // 2
        pad_bottom = pad_height - pad_top
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left

        # Pad image with black
        img_padded = cv.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right, cv.BORDER_CONSTANT, value=[0, 0, 0])
        images_padded.append(img_padded)
    
    return images_padded

def create_splits(
        images, 
        radius_info,
        labels,  
        ratio=constants.RATIO):
    """Create training and validation datasets
    Args:
        images (list): list of images
        labels (list): list of labels
        ratio (float): ratio of the training set
    Returns:
        train_images, train_labels, val_images, val_labels
    """
    
    # Split data into training and validation sets
    split = int(len(images) * ratio)
    images, labels = np.array(images), np.array(labels)
    train_images, val_images = images[:split], images[split:]
    train_radius, val_radius = radius_info[:split], radius_info[split:]
    train_labels, val_labels = labels[:split], labels[split:]

    return train_images, train_radius, train_labels, val_images, val_radius, val_labels