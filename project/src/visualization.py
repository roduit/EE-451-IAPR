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