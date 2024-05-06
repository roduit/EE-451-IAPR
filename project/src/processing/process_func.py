# -*- coding: utf-8 -*-
# -*- authors : Vincent Roduit -*-
# -*- date : 2024-05-03 -*-
# -*- Last revision: 2024-05-03 -*-
# -*- python version : 3.9.18 -*-
# -*- Description: Function used for preprocessing images -*-

# Importing libraries
import numpy as np
import skimage as sk

def rgb_to_gray(img):
    """
    Convert an RGB image to grayscale
    
    Args:
    img: np.ndarray (H, W, 3)
        RGB image to convert
    
    Returns:
    gray_img: np.ndarray (H, W)
        Grayscale image
    """
    gray_img = sk.color.rgb2gray(img)
    
    return gray_img

def apply_median(img, kernel_size=5):
    """
    Apply a median filter to an image
    
    Args:
    img: np.ndarray (H, W)
        Image to filter
    kernel_size: int
        Size of the kernel
    
    Returns:
    filtered_img: np.ndarray (H, W)
        Filtered image
    """
    filtered_img = sk.filters.median(img, sk.morphology.disk(kernel_size))
    
    return filtered_img

def find_threshold(gray_img):
    """
    Find the threshold of an image
    
    Args:
    img: np.ndarray (H, W)
        Image to threshold
    
    Returns:
    threshold: float
        Threshold value
    """

    normalized_img = sk.exposure.rescale_intensity(gray_img, in_range=(gray_img.min(), gray_img.max()), out_range=(0, 255))

    otsu_threshold = sk.filters.threshold_otsu(normalized_img)

    adjusted_threshold = otsu_threshold - 15
    final_binary_img = normalized_img > adjusted_threshold
    final_binary_img = final_binary_img.astype(np.uint8) * 255
    
    return final_binary_img, adjusted_threshold
    