# -*- coding: utf-8 -*-
# -*- author : Vincent Roduit-*-
# -*- date : 2024-05-15 -*-
# -*- Last revision: 2024-05-15 (Vincent Roduit)-*-
# -*- python version : 3.9.18 -*-
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

def augment_set(train_images, train_labels):
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

    for idx, img in enumerate(train_images):
        imgs_rotated = rotate_by_set_angles(img)
        start_idx = idx * num_angles
        end_idx = start_idx + num_angles
        train_images_aug[start_idx:end_idx] = imgs_rotated
        train_labels_aug[start_idx:end_idx] = train_labels[idx]

    return train_images_aug, train_labels_aug


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


def augment_blur(train_images, train_labels, blur_prob=constants.BLUR_PROB):
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

    for idx, img in enumerate(train_images):
        if np.random.rand() < blur_prob:
            kernel_size = np.random.choice(constants.KERNEL_SIZE_SET)
            img_blurred = blur_image(img, kernel_size)
            train_images_aug.append(img_blurred)
            train_images_aug.append(img)
            train_images_labels.extend([train_labels[idx]] * 2)
        else:
            train_images_aug.append(img)
            train_images_labels.append(train_labels[idx])

    return train_images_aug, train_images_labels
