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

def apply_rgb_threshold(img, th_red, th_green, th_blue):
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
    data_red = (data_red < th_red)
    data_green = (data_green < th_green)
    data_blue = (data_blue < th_blue)
    img_th = np.logical_and(data_red, data_green, data_blue) * 255
    # ------------------
    return  img_th 


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

def my_trheshold_func(data_h,data_s,data_v):
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

def apply_hsv_threshold(img, func=my_trheshold_func):
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