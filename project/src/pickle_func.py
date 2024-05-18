
# -*- coding: utf-8 -*-
# -*- authors : Vincent Roduit -*-
# -*- date : 2024-05-03 -*-
# -*- Last revision: 2024-05-03 -*-
# -*- python version : 3.9.18 -*-
# -*- Description: Function to save results as pickle -*-

# import libraries
import pickle
import os

# import files
import constants

def save_pickle(result, file_name="pickle"):
    """Save a variable in a binary format

    Args:
        result: dataFrame
        file_path: file path where to store this variable

    Returns:
    """
    save_pickle_path = os.path.join(constants.RESULT_PATH, 'pickle_files')
    if not os.path.exists(save_pickle_path):
        os.makedirs(save_pickle_path)
    file_path = os.path.join(save_pickle_path, file_name)
    with open(file_path, "wb") as file:
        pickle.dump(result, file)


def load_pickle(file_path):
    """Load a variable from a binary format path

    Args:
        file_path: the file path where the file is stored

    Returns:
        return the content of the file, generally a dataFrame here.
    """
    with open(file_path, "rb") as file:
        return pickle.load(file)