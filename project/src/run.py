# -*- coding: utf-8 -*-
# -*- author : Vincent Roduit -*-
# -*- date : 2024-05-22 -*-
# -*- Last revision: 2024-05-22 -*-
# -*- python version : 3.9.18 -*-
# -*- Description: Run the best solution -*-

#import libraries
import time
import argparse
import os

#import files
from data_classes.train_data import trainCoin
import constants


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the best solution")
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to the model file",
    )
    parser.add_argument(
        "--data_path", 
        type=str, 
        default=constants.DATA,
        help="path to the dataset", 
        required=False
    )
    parser.add_argument(
        "--output_csv_path", 
        type=str, 
        help="submission file path", 
        default=constants.SUBMISSION_PATH,
        required=False
    )
    parser.add_argument(
        "--save",
        type=bool,
        default=False,
        help="Save the images",
    )
    parser.add_argument(
        "--img_size",
        type=int,
        default=constants.IMG_SIZE,
        help="Size of the images",
    )
    args = parser.parse_args()

    model_path = args.model_path
    data_path = parser.data_path
    output_csv_path = parser.output_csv_path
    save = parser.save
    img_size = parser.img_size

    st = time.time()


def run_best_solution(model_path, data_path, output_csv_path, img_size, save):
    # Load the model
    model = load_model(model_path, save, img_size)

    print(f"Time elapsed: {time.time() - st:.2f} s")

def load_model(model_path, save, img_size):
    if model_path is None:

        #Load train data
        train_data = trainCoin(save=save)
        train_data.proceed_data()