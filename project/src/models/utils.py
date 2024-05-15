# -*- coding: utf-8 -*-
# -*- author : Vincent Roduit-*-
# -*- date : 2024-05-15 -*-
# -*- Last revision: 2024-05-15 (Vincent Roduit)-*-
# -*- python version : 3.9.18 -*-
# -*- Description: Util functions for models -*-

#import libraries
import os
import pandas as pd
import numpy as np
import cv2 as cv
from torch.utils.data import DataLoader

#import files
import constants

def get_classes_conv_table():
    path_train_labels = os.path.join(constants.DATA, 'train_labels.csv')
    df_train_labels = pd.read_csv(path_train_labels)
    classes = df_train_labels.columns[1:]
    classes_conversion_table = {classes[i].strip(): i for i in range(len(classes))}
    return classes_conversion_table

def get_coin_labels():
    coin_labels_path = os.path.join(constants.RESULT_PATH, 'coin_img', 'coin_labels_complete.xlsx')
    coin_labels = pd.read_excel(coin_labels_path)
    coin_labels.dropna(subset=['Label'], inplace=True)
    return coin_labels

def create_data_structure(coins, coin_labels, conversion_table):
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
    df_images_labels = pd.DataFrame({'image_name': images_names,'coin_name':coin_names, 'label': images_labels})
    df_images_labels['label_int'] = df_images_labels['label'].apply(lambda x: conversion_table[x])
    labels = df_images_labels['label_int'].values

    return np.array(images), np.array(labels), df_images_labels

def resize_images(images):
    #find biggest image 
    biggest_dim = max([image.shape[0] for image in images])

    images_resized = []
    for img in images:
        #resize image
        img = cv.resize(img, (biggest_dim, biggest_dim))
        images_resized.append(img)
    return images_resized

def pad_images(images):

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

def create_datasets(
        images, 
        labels,  
        ratio=constants.RATIO):
    
    # Split data into training and validation sets
    split = int(len(images) * ratio)
    train_images, val_images = images[:split], images[split:]
    train_labels, val_labels = labels[:split], labels[split:]

    return train_images, train_labels, val_images, val_labels

def create_dataloader(
        train_images, 
        train_labels, 
        val_images, 
        val_labels, 
        batch_size=constants.BATCH_SIZE, 
        num_workers=constants.NUM_WORKERS):
    
    #transpose images for pytorch
    train_images = np.transpose(train_images, (0, 3, 1, 2)).astype(np.float32) / 255
    val_images = np.transpose(val_images, (0, 3, 1, 2)).astype(np.float32) / 255

    # Create dataloaders
    train_dataloader = DataLoader(
    dataset = list(zip(train_images, train_labels)),
    batch_size = batch_size,
    shuffle = False,
    num_workers = num_workers
    )

    val_dataloader = DataLoader(
        dataset = list(zip(val_images, val_labels)),
        batch_size = constants.BATCH_SIZE,
        shuffle = False,
        num_workers = num_workers
        )

    return train_dataloader, val_dataloader