
# -*- coding: utf-8 -*-
# -*- authors : Vincent Roduit -*-
# -*- date : 2024-05-03 -*-
# -*- Last revision: 2024-05-03 -*-
# -*- python version : 3.9.18 -*-
# -*- Description: Function to save results as pickle -*-

import pickle

def save_pickle(result, file_path="pickle"):
    """Save a variable in a binary format

    Args:
        result: dataFrame
        file_path: file path where to store this variable

    Returns:
    """
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