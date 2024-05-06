# -*- coding: utf-8 -*-
# -*- authors : Vincent Roduit -*-
# -*- date : 2024-05-03 -*-
# -*- Last revision: 2024-05-03 -*-
# -*- python version : 3.9.18 -*-
# -*- Description: Function used for preprocessing images -*-

# Importing libraries
import numpy as np
import skimage as sk
import cv2
from skimage.measure import regionprops, approximate_polygon

# Importing files
import constants

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

def apply_median(img, kernel_size=constants.KERNEL_SIZE):
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

    adjusted_threshold = otsu_threshold - constants.ADJ_THRESHOLD
    final_binary_img = normalized_img > adjusted_threshold
    final_binary_img = final_binary_img.astype(np.uint8) * 255
    
    return final_binary_img, adjusted_threshold

def is_contour_closed(contour, tolerance=1e-6):
    if np.linalg.norm(contour[0] - contour[-1]) < tolerance:
        return True
    else:
        return False

def find_contour(images: np.ndarray):
    """
    Find the contours for the set of images
    
    Args
    ----
    images: np.ndarray (N, 28, 28)
        Source images to process

    Return
    ------
    contours: list of np.ndarray
        List of N arrays containing the coordinates of the contour. Each element of the 
        list is an array of 2d coordinates (K, 2) where K depends on the number of elements 
        that form the contour. 
    """

    # Get number of images to process
    N, _, _ = np.shape(images)
    # Fill in dummy values (fake points)
    contours = [np.array([[0, 0], [1, 1]]) for i in range(N)]

    # ------------------
    # Your code here ... 
    for idx, i in enumerate(images):
        cnts = sk.measure.find_contours(np.transpose(i))
        #filter contours
        filtered_cnts = []
        for cnt in cnts:
            if len(cnt) > constants.MIN_CONTOUR_LEN and is_contour_closed(cnt):
                    filtered_cnts.append(cnt)
        contours[idx] = filtered_cnts
    # ------------------
    
    return contours


def compute_features(imgs: np.ndarray):
    """
    Compute compacity for each input image.
    
    Args
    ----
    imgs: np.ndarray (N, 28, 28)
        Source images
        
    Return
    ------
    f_peri: np.ndarray (N,)
        Estimated perimeter length for each image
    f_area: np.ndarray (N,)
        Estimated area for each image
    f_comp: np.ndarray (N,)
        Estimated compacity for each image
    f_rect: np.ndarray (N,)
        Estimated rectangularity for each image
    """

    f_peri = np.zeros(len(imgs))
    f_area = np.zeros(len(imgs))
    f_comp = np.zeros(len(imgs))
    f_rect = np.zeros(len(imgs))
    
    # ------------------
    # Your code here ... 
    for idx, img in enumerate(imgs):
        properties = regionprops(img.astype(int))[0]
        f_peri[idx] = properties.perimeter_crofton
        f_area[idx] = properties.area
        f_comp[idx] = f_peri[idx]**2 / f_area[idx]
        f_rect[idx] = properties.area / properties.area_bbox
    # ------------------

    return f_peri, f_area, f_comp, f_rect