# -*- coding: utf-8 -*-
# -*- authors : Vincent Roduit -*-
# -*- date : 2024-21-05 -*-
# -*- Last revision: 2024-12-05 -*-
# -*- python version : 3.9.18 -*-
# -*- Description: Defines all the constants used in the project -*-

# Importing libraries
import os
import numpy as np

# ----------------------------------
# -----    Defining paths  ---------
# ----------------------------------

# Path to the data folder
try:
    DATA = os.path.join('..', 'data')
    if not os.path.exists(DATA):
        raise FileNotFoundError("Folder data does not exist.", DATA)
except Exception as e:
    print("An error occured: ", e)

# Path to the reference folder
try:
    REF = os.path.join(DATA, 'ref')
    if not os.path.exists(REF):
        raise FileNotFoundError("Folder ref does not exist.", REF)
except Exception as e:
    print("An error occured: ", e)

# Path to the test folder
try:
    TEST = os.path.join(DATA, 'test')
    if not os.path.exists(TEST):
        raise FileNotFoundError("Folder test does not exist.", TEST)
except Exception as e:
    print("An error occured: ", e)

# Path to the train folder
try:
    TRAIN = os.path.join(DATA, 'train')
    if not os.path.exists(TRAIN):
        raise FileNotFoundError("Folder train does not exist.", TRAIN)
except Exception as e:
    print("An error occured: ", e)

# Path to the results folder
try:
    RESULT_PATH = os.path.join(DATA, 'results')
    if not os.path.exists(RESULT_PATH):
        raise FileNotFoundError("Folder results does not exist.", RESULT_PATH)
except Exception as e:
    print("An error occured: ", e)


# Path to the submission folder
try:
    SUBMISSION_PATH = os.path.join('..', 'submissions')
    if not os.path.exists(SUBMISSION_PATH):
        raise FileNotFoundError("Folder submission does not exist.", SUBMISSION_PATH)
except Exception as e:
    print("An error occured: ", e)


# ----------------------------------
# -----  PROCESSING CONST  ---------
# ----------------------------------

# Disk size for opening
DISK_SIZE = 15

# Median filter kernel size
KERNEL_SIZE = 5

# Adjust the threshold
ADJ_THRESHOLD = 15

# Contour len
MIN_CONTOUR_LEN = 1500

#Number of clusters
N_CLUSTERS = 16

# ----------------------------------
# ---------  MODEL CONST  ----------
# ----------------------------------

BATCH_SIZE = 32

NUM_WORKERS = 8

RATIO = 0.8

ANGLES_SET = np.linspace(0, 360, 18, endpoint=False)

BLUR_PROB = 0.5

KERNEL_SIZE_SET = [3, 5, 7, 9]

IMG_SIZE = 400

