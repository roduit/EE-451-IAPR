"""
Module: visualization.py
Description: This module provides functions for visualizing data.
Author: Vincent Roduit, Filippo Quadri
Date: 07.05.2024
"""

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def detect_coin(img: np.array, min_radius: int, max_radius: int, median_tresh: int) -> np.array:
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
    circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, dp=1, minDist=50, param1=10, param2=30, minRadius=min_radius, maxRadius=max_radius)
    img_circles = img.copy()
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            cv.circle(img_circles, (i[0], i[1]), i[2], (255, 0, 0), 5)
            
            # Center of the circle
            # cv.circle(img_circles, (i[0], i[1]), 2, (0, 0, 0), 5)
    
    return img_circles