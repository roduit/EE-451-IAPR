# -*- coding: utf-8 -*-
# -*- authors : Vincent Roduit -*-
# -*- date : 2024-05-05 -*-
# -*- Last revision: 2024-05-05 -*-
# -*- python version : 3.12.3 -*-
# -*- Description: Function to save results as pickle -*-

# import libraries
import pickle
import os

# import files
import constants

def save_pickle(result, file_name="pickle", save_path=None):
    """Save a variable in a binary format

    Args:
        result: dataFrame
        file_path: file path where to store this variable

    Returns:
    """
    if save_path is None:
        save_path = os.path.join(constants.RESULT_PATH, 'pickle_files')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
    file_path = os.path.join(save_path, file_name)
    with open(file_path, "wb") as file:
        pickle.dump(result, file)


def load_pickle(file_name, load_path=None):
    """Load a variable from a binary format path

    Args:
        file_path: the file path where the file is stored

    Returns:
        return the content of the file, generally a dataFrame here.
    """
    if load_path is None:
        load_path = os.path.join(constants.RESULT_PATH, 'pickle_files')
        if not os.path.exists(load_path):
            os.makedirs(load_path)
    file_path = os.path.join(load_path, file_name)
    with open(file_path, "rb") as file:
        return pickle.load(file)