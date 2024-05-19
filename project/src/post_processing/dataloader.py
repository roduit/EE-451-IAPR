# -*- coding: utf-8 -*-
# -*- author : Vincent Roduit-*-
# -*- date : 2024-05-19 -*-
# -*- Last revision: 2024-05-19 (Vincent Roduit)-*-
# -*- python version : 3.9.18 -*-
# -*- Description: Function to create the Dataloaders  -*-

# Import libraries
from torch.utils.data import DataLoader
import numpy as np

# Import files
from post_processing.datasets import CoinDataset, CoinDatasetRadius
import constants

def create_dataloader(
        train_images, 
        train_labels, 
        val_images, 
        val_labels, 
        train_radius=None,
        val_radius=None,
        batch_size=constants.BATCH_SIZE, 
        num_workers=constants.NUM_WORKERS):
    """Create dataloaders
    Args:
        train_images (list): list of training images
        train_labels (list): list of training labels
        val_images (list): list of validation images
        val_labels (list): list of validation labels
        train_radius (list or None): list of training radius information
        val_radius (list or None): list of validation radius information
        batch_size (int): batch size
        num_workers (int): number of workers
    
    Returns:
        train_dataloader, val_dataloader
    """
    
    # Transpose and normalize images for PyTorch
    train_images = np.transpose(train_images, (0, 3, 1, 2)).astype(np.float32) / 255
    val_images = np.transpose(val_images, (0, 3, 1, 2)).astype(np.float32) / 255

    # Create datasets
    if train_radius is None:
        train_dataset = CoinDataset(train_images, train_labels)
        val_dataset = CoinDataset(val_images, val_labels)
    else:
        train_dataset = CoinDatasetRadius(train_images, train_labels, train_radius)
        val_dataset = CoinDatasetRadius(val_images, val_labels, val_radius)

    # Create dataloaders
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,  # Shuffle training data
        num_workers=num_workers
    )

    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,  # Use the provided batch_size
        shuffle=False,
        num_workers=num_workers
    )

    return train_dataloader, val_dataloader