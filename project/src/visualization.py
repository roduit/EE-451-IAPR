# -*- coding: utf-8 -*-
# -*- authors : Vincent Roduit -*-
# -*- date : 2024-05-03 -*-
# -*- Last revision: 2024-05-03 -*-
# -*- python version : 3.9.18 -*-
# -*- Description: Function to visualize results -*-

# Importing libraries
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable


def display_gray(original, img):
    fig = plt.figure()

    fig.add_subplot(1, 2, 1)
    plt.axis('off')
    plt.title("Original")
    plt.imshow(original)

    fig.add_subplot(1, 2, 2)
    plt.axis('off')
    plt.title("Gray/Blur")
    plt.imshow(img, cmap='gray')

    plt.show()

def visualize_threshold(img,img_thresh, threshold):
    fig = plt.figure(figsize=(10, 5))
    fig.add_subplot(1, 2, 1)
    plt.axis('off')
    plt.imshow(img_thresh, interpolation='nearest')

    fig.add_subplot(1, 2, 2)
    plt.title("Threshold: ({}/255)".format(threshold))
    plt.axvline(x=threshold, color='red')
    plt.hist(img.ravel(), bins=256, range=[0, 256])
    plt.show()

def display_contours(img, contours):
    fig = plt.figure()
    plt.axis('off')
    plt.imshow(img, interpolation='nearest')
    for contour in contours:
        plt.plot(contour[:, 0], contour[:, 1], 'r',linewidth=2)
    plt.show()

# Plot color space distribution 
def plot_colors_histo(
    img: np.ndarray,
    func: Callable,
    labels: list[str],
):
    """
    Plot the original image (top) as well as the channel's color distributions (bottom).

    Args
    ----
    img: np.ndarray (M, N, C)
        Input image of shape MxN and C channels.
    func: Callable
        A callable function that extracts D channels from the input image
    labels: list of str
        List of D labels indicating the name of the channel
    """

    # Extract colors
    channels = func(img=img)
    C2 = len(channels)
    M, N, C1 = img.shape
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, C2)

    # Use random seed to downsample image colors (increase run speed - 10%)
    mask = np.random.RandomState(seed=0).rand(M, N) < 0.1
    
    # Plot base image
    ax = fig.add_subplot(gs[:2, :])
    ax.imshow(img)
    # Remove axis
    ax.axis('off')
    ax1 = fig.add_subplot(gs[2, 0])
    ax2 = fig.add_subplot(gs[2, 1])
    ax3 = fig.add_subplot(gs[2, 2])

    # Plot channel distributions
    ax1.scatter(channels[0][mask].flatten(), channels[1][mask].flatten(), c=img[mask]/255, s=1, alpha=0.1)
    ax1.set_xlabel(labels[0])
    ax1.set_ylabel(labels[1])
    ax1.set_title("{} vs {}".format(labels[0], labels[1]))
    ax2.scatter(channels[0][mask].flatten(), channels[2][mask].flatten(), c=img[mask]/255, s=1, alpha=0.1)
    ax2.set_xlabel(labels[0])
    ax2.set_ylabel(labels[2])
    ax2.set_title("{} vs {}".format(labels[0], labels[2]))
    ax3.scatter(channels[1][mask].flatten(), channels[2][mask].flatten(), c=img[mask]/255, s=1, alpha=0.1)
    ax3.set_xlabel(labels[1])
    ax3.set_ylabel(labels[2])
    ax3.set_title("{} vs {}".format(labels[1], labels[2]))
        
    plt.tight_layout()