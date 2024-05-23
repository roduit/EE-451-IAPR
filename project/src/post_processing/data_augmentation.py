# -*- coding: utf-8 -*-
# -*- author : Vincent Roduit-*-
# -*- date : 2024-05-15 -*-
# -*- Last revision: 2024-05-15 (Vincent Roduit)-*-
# -*- python version : 3.12.3 -*-
# -*- Description: Functions for data augmentation -*-

# import libraries
import cv2 as cv
import numpy as np
import constants

def rotate_image(image, angle):
    """
    Rotate an image by a given angle.

    Args:
        image (numpy.ndarray): Image to rotate.
        angle (int): Angle in degrees.

    Returns:
        numpy.ndarray: Rotated image.
    """
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv.INTER_LINEAR)
    return result

def rotate_by_set_angles(image):
    """
    Rotate an image by a set of angles.

    Args:
        image (numpy.ndarray): Image to rotate.

    Returns:
        list: List of rotated images.
    """
    angles = constants.ANGLES_SET
    rotated_images = [rotate_image(image, angle) for angle in angles]
    return rotated_images

def augment_set_rotations(train_images, radius, train_labels):
    """
    Augment a set of images by rotating them by a set of angles.

    Args:
        train_images (list or np.ndarray): List or array of images to augment.
        train_labels (list or np.ndarray): List or array of labels corresponding to the images.

    Returns:
        tuple: Tuple containing arrays of augmented images and corresponding labels.
    """
    num_angles = len(constants.ANGLES_SET)
    num_images = len(train_images)
    img_shape = train_images[0].shape

    train_images_aug = np.zeros((num_images * num_angles, *img_shape), dtype=train_images.dtype)
    train_labels_aug = np.zeros(num_images * num_angles, dtype=train_labels.dtype)
    train_radius_aug = np.zeros(num_images * num_angles, dtype=radius.dtype)

    for idx, img in enumerate(train_images):
        imgs_rotated = rotate_by_set_angles(img)
        start_idx = idx * num_angles
        end_idx = start_idx + num_angles
        train_images_aug[start_idx:end_idx] = imgs_rotated
        train_labels_aug[start_idx:end_idx] = train_labels[idx]
        train_radius_aug[start_idx:end_idx] = radius[idx]
    
    return train_images_aug,train_radius_aug, train_labels_aug


def blur_image(image, kernel_size):
    """
    Apply a Gaussian blur to an image.

    Args:
        image (numpy.ndarray): Image to blur.
        kernel_size (int): Kernel size for the Gaussian blur.

    Returns:
        numpy.ndarray: Blurred image.
    """
    return cv.GaussianBlur(image, (kernel_size, kernel_size), 0)


def augment_blur(train_images, train_radius, train_labels, blur_prob=constants.BLUR_PROB):
    """
    Augment a set of images by applying a Gaussian blur.

    Args:
        images (list): List of images to augment.
        kernel_size (int): Kernel size for the Gaussian blur.
        blur_prob (float): Probability of applying the blur.

    Returns:
        list: List of augmented images.
    """
    train_images_aug = []
    train_images_labels = []
    train_images_radius = []

    for idx, img in enumerate(train_images):
        if np.random.rand() < blur_prob:
            kernel_size = np.random.choice(constants.KERNEL_SIZE_SET)
            img_blurred = blur_image(img, kernel_size)
            train_images_aug.append(img_blurred)
            train_images_aug.append(img)
            train_images_labels.extend([train_labels[idx]] * 2)
            train_images_radius.extend([train_radius[idx]] * 2)
        else:
            train_images_aug.append(img)
            train_images_labels.append(train_labels[idx])
            train_images_radius.append(train_radius[idx])

    return train_images_aug, train_images_radius, train_images_labels

def augment_gamma_correction(train_images, train_radius, train_labels):
    """
    Augment a set of images by applying gamma correction.

    Args:
        train_images (list): List of images to augment.
        train_labels (list): List of labels corresponding to the images.

    Returns:
        list: List of augmented images.
    """
    train_images_aug = []
    train_images_labels = []
    train_images_radius = []

    for idx, img in enumerate(train_images):
        if np.random.rand() < constants.GAMMA_CORRECTION_PROB:
            gamma = np.random.choice(constants.GAMMA_SET)
            img_corrected = gamma_correction(img, gamma)
            train_images_aug.append(img_corrected)
            train_images_aug.append(img)
            train_images_labels.extend([train_labels[idx]] * 2)
            train_images_radius.extend([train_radius[idx]] * 2)
        else:
            gamma = np.random.choice(constants.GAMMA_SET)
            img_corrected = gamma_correction(img, gamma)
            train_images_aug.append(img_corrected)
            train_images_labels.append(train_labels[idx])
            train_images_radius.append(train_radius[idx])

    return train_images_aug, train_images_radius, train_images_labels

def augment_histogram_equalization(train_images, train_radius, train_labels):
    """
    Augment a set of images by applying histogram equalization.

    Args:
        train_images (list): List of images to augment.
        train_labels (list): List of labels corresponding to the images.

    Returns:
        list: List of augmented images.
    """
    train_images_aug = []
    train_images_labels = []
    train_images_radius = []
    for idx, img in enumerate(train_images):
        if np.random.rand() < constants.HISTO_PROB:
            img_equalized = histogram_equalization(img)
            train_images_aug.append(img_equalized)
            train_images_aug.append(img)
            train_images_labels.extend([train_labels[idx]] * 2)
            train_images_radius.extend([train_radius[idx]] * 2)
        else:
            train_images_aug.append(img)
            train_images_labels.append(train_labels[idx])
            train_images_radius.append(train_radius[idx])

    return train_images_aug, train_images_radius, train_images_labels


def histogram_equalization(image):
    # Convert to YUV color space
    img_yuv = cv.cvtColor(image, cv.COLOR_BGR2YUV)
    
    # Apply histogram equalization to the Y channel
    img_yuv[:, :, 0] = cv.equalizeHist(img_yuv[:, :, 0])
    
    # Convert back to BGR color space
    equalized_image = cv.cvtColor(img_yuv, cv.COLOR_YUV2BGR)
    return equalized_image

def gamma_correction(image, gamma=1.0):
    # Ensure the input is a numpy array
    if not isinstance(image, np.ndarray):
        image = np.array(image)
    
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype("uint8")
    return cv.LUT(image, table)
