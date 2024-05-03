# -*- coding: utf-8 -*-
# -*- authors : Vincent Roduit -*-
# -*- date : 2024-05-03 -*-
# -*- Last revision: 2024-05-03 -*-
# -*- python version : 3.9.18 -*-
# -*- Description: Defines all the constants used in the project -*-

# Importing libraries
import os

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