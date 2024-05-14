import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2hsv
from typing import Callable
import cv2 as cv

def extract_rgb_channels(img):
    """
    Extract RGB channels from the input image.

    Args
    ----
    img: np.ndarray (M, N, C)
        Input image of shape MxN and C channels.
    
    Return
    ------
    data_red: np.ndarray (M, N)
        Red channel of input image
    data_green: np.ndarray (M, N)
        Green channel of input image
    data_blue: np.ndarray (M, N)
        Blue channel of input image
    """

    # Get the shape of the input image
    M, N, C = np.shape(img)

    # Define default values for RGB channels
    data_red = np.zeros((M, N))
    data_green = np.zeros((M, N))
    data_blue = np.zeros((M, N))

    # ------------------
    # Your code here ...
    data_red = img[:, :, 0]
    data_green = img[:, :, 1]
    data_blue = img[:, :, 2] 
    # ------------------
    
    return data_red, data_green, data_blue


def extract_hsv_channels(img):
    """
    Extract HSV channels from the input image.

    Args
    ----
    img: np.ndarray (M, N, C)
        Input image of shape MxN and C channels.
    
    Return
    ------
    data_h: np.ndarray (M, N)
        Hue channel of input image
    data_s: np.ndarray (M, N)
        Saturation channel of input image
    data_v: np.ndarray (M, N)
        Value channel of input image
    """

    # Get the shape of the input image
    M, N, C = np.shape(img)

    # Define default values for HSV channels
    data_h = np.zeros((M, N))
    data_s = np.zeros((M, N))
    data_v = np.zeros((M, N))

    # ------------------
    # Your code here ...
    hsv_image =  rgb2hsv(img)
    data_h = hsv_image[:, :, 0]
    data_s = hsv_image[:, :, 1]
    data_v = hsv_image[:, :, 2]
    # ------------------
    
    return data_h, data_s, data_v

def rgb_neutral_threshold(data_red, data_green, data_blue):
    th_red = 150
    th_green = 200
    th_blue = 40
    data_red = (data_red > th_red)
    data_green = (data_green > th_green)
    data_blue = (data_blue < th_blue)

    return data_red, data_green, data_blue

def rgb_noisy_bg_threshold(data_red, data_green, data_blue):
    th_red = 230
    th_green = 235
    th_blue = 0
    data_red = (data_red < th_red)
    data_green = (data_green > th_green)
    data_blue = (data_blue < th_blue)

    return data_red, data_green, data_blue


def apply_rgb_threshold(img, func=rgb_neutral_threshold):
    """
    Apply threshold to RGB input image.

    Args
    ----
    img: np.ndarray (M, N, C)
        Input image of shape MxN and C channels.
    
    Return
    ------
    img_th: np.ndarray (M, N)
        Thresholded image.
    """

    # Define the default value for the input image
    M, N, C = np.shape(img)
    img_th = np.zeros((M, N))

    # Use the previous function to extract RGB channels
    data_red, data_green, data_blue = extract_rgb_channels(img=img)
    
    # ------------------
    # Your code here ...
    data_red, data_green, data_blue = func(data_red, data_green, data_blue)
    img_th = np.logical_and(data_red, data_green, data_blue) * 255
    # ------------------
    return  img_th 

def hand_threshold(data_h,data_s,data_v):
    """Apply threshold to HSV input image.
    Args
    data_h: np.ndarray (M, N) Hue channel of input image
    data_s: np.ndarray (M, N) Saturation channel of input image
    data_v: np.ndarray (M, N) Value channel of input image

    Return
    data_h: np.ndarray (M, N) Hue channel of input image
    data_s: np.ndarray (M, N) Saturation channel of input image
    data_v: np.ndarray (M, N) Value channel of input image

    """
    th_h_low = 0.058
    th_h_high = 0.19
    th_s_low = 0.05
    th_s_high = 1
    th_v_low = 0
    th_v_high = 1
    data_h = (data_h > th_h_low) & (data_h < th_h_high)
    data_s = ((data_s > th_s_low) & (data_s < th_s_high))
    data_v = (data_v > th_v_low) & (data_v < th_v_high)
    return data_h, data_s, data_v

def apply_hsv_threshold(img, func=hand_threshold):
    """
    Apply threshold to the input image in hsv colorspace.

    Args
    ----
    img: np.ndarray (M, N, C)
        Input image of shape MxN and C channels.
    
    Return
    ------
    img_th: np.ndarray (M, N)
        Thresholded image.
    """

    # Define the default value for the input image
    M, N, C = np.shape(img)
    img_th = np.zeros((M, N))

    # Use the previous function to extract HSV channels
    data_h, data_s, data_v = extract_hsv_channels(img=img)
    
    # ------------------
    # Your code here ... 
    data_h, data_s, data_v = func(data_h, data_s, data_v)
    img_th = np.logical_and(data_h, data_s, data_v) * 255
    # ------------------
    return  img_th

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
    circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, dp=1, minDist=120, param1=25, param2=40, minRadius=min_radius, maxRadius=max_radius)
    img_circles = img.copy()
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            cv.circle(img_circles, (i[0], i[1]), i[2], (255, 0, 0), 5)
            
            # Center of the circle
            # cv.circle(img_circles, (i[0], i[1]), 2, (0, 0, 0), 5)
    
    return img_circles