# -*- coding: utf-8 -*-
# -*- author : Vincent Roduit -*-
# -*- date : 2024-05-22 -*-
# -*- Last revision: 2024-05-23 -*-
# -*- python version : 3.12.3 -*-
# -*- Description: Run the best solution -*-

#import libraries
import time
import argparse
import os
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import timm

#import files
from data_classes.train_data import trainCoin
from data_classes.test_data import testCoin
import constants
from post_processing.data_formating import get_classes_conv_table
from post_processing.data_formating import get_coin_labels
from post_processing.data_formating import create_train_data_structure
from post_processing.data_formating import create_test_data_structure
from post_processing.data_formating import create_splits
from post_processing.data_augmentation import *
from post_processing.dataloader import create_dataloader, create_test_dataloader
from post_processing.submission import create_submission_file
from models.efficient_net import *


def handle_path(model_path,output_csv_path):
    """Handle the model and output path
    Args:
        model_path (str): Path to the model file
        output_csv_path (str): Path to the output csv file
    
    Returns:
        model_path (str): Path to the model file
        output_csv_path (str): Path to the output csv file
    """
    if model_path is not None and not os.path.exists(model_path):
        raise ValueError("Model path does not exist")
    if output_csv_path is None:
        output_csv_path = constants.SUBMISSION_PATH
    elif not os.path.exists(output_csv_path):
        os.makedirs(output_csv_path)
    return model_path,output_csv_path

def run_best_solution(model_path, output_csv_path, save):
    """Run the best solution
    Args:
        model_path (str): Path to the model file
        output_csv_path (str): Path to the output csv file
        save (bool): Save the images
    
    Returns:
        None
    """
    # Load the model
    model_path, output_csv_path = handle_path(model_path, output_csv_path)
    make_prediction(model_path, output_csv_path, save)

    print(f"Time elapsed: {time.time() - st:.2f} s")

def make_prediction(model_path,output_csv_path, save):
    """Make predictions using the best model and create the submission file
    Args:
        model_path (str): Path to the model file
        output_csv_path (str): Path to the output csv file
        save (bool): Save the images
    
    Returns:
        None
    """
    if model_path is None:
        print("No model path provided: training the model...")

        print("processing train data...")
        train_data = trainCoin(save=save)
        train_data.proceed_data()
        print("processing test data...")
        test_data = testCoin(save=save)
        test_data.proceed_data()
        print("creating data loaders...")
        conversion_table = get_classes_conv_table()
        coin_labels = get_coin_labels()
        train_images_raw, train_radius_infos, labels, _ = create_train_data_structure(train_data.coins, train_data.contours_tuple, coin_labels,conversion_table)
        test_imgs, _, df_test_images = create_test_data_structure(test_data.coins, test_data.contours_tuple)
        train_images, train_radius, train_labels, val_images, _, val_labels = create_splits(
                                                                                        train_images_raw, 
                                                                                        train_radius_infos, 
                                                                                        labels,
                                                                                        ratio=constants.RATIO)
        # deleting the raw images to free up memory
        del train_data
        del test_data
        print("augmenting data...")
        # Augment the training set with rotations
        train_images_aug, train_radius_aug, train_labels_aug = augment_set_rotations(train_images, train_radius, train_labels)

        # Augment the training set with Gaussian blur
        train_images_aug, train_radius_aug, train_labels_aug = augment_blur(train_images_aug,train_radius_aug, train_labels_aug)

        # Augment the training set with histogram equalization
        train_images_aug, train_radius_aug, train_labels_aug = augment_histogram_equalization(train_images_aug, train_radius_aug, train_labels_aug)

        # Augment the training set with gamma correction
        train_images_aug, train_radius_aug, train_labels_aug = augment_gamma_correction(train_images_aug, train_radius_aug, train_labels_aug)

        train_dataloader, val_dataloader = create_dataloader(
                                                train_images=train_images_aug,
                                                train_labels=train_labels_aug,
                                                train_radius=None,
                                                val_images=val_images,
                                                val_labels=val_labels,
                                                val_radius=None)
        
        test_dataloader = create_test_dataloader(test_imgs, None)
        num_classes = len(conversion_table)

        #delete datasets to free up memory
        del train_images
        del train_radius
        del train_labels
        del val_images
        del val_labels
        del train_images_aug
        del train_radius_aug
        del train_labels_aug

        print("training model...")
        # Define the model
        model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=num_classes)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

        # Train the model
        model = train_model(model, optimizer, scheduler, train_dataloader, val_dataloader, num_epochs=10)

        print("making predictions...")
        predictions = predict(model, test_dataloader)
        _ = create_submission_file(predictions, df_test_images, conversion_table, name='submission_from_run.csv', path=output_csv_path)
        print("Model trained and submission file created")
    else:
        print("Model path provided: loading the model...")
        # Load the model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = torch.load(model_path, map_location=device)
        model.eval()
        # Load test data
        test_data = testCoin(save=save)
        print("processing test data...")
        test_data.proceed_data()
        print("creating data loaders...")
        test_imgs, _, df_test_images = create_test_data_structure(test_data.coins, test_data.contours_tuple)
        test_dataloader = create_test_dataloader(test_imgs, None)
        print("making predictions...")
        predictions = predict(model, test_dataloader)
        conversion_table = get_classes_conv_table()
        _ = create_submission_file(predictions, df_test_images, conversion_table, name='submission_from_run.csv', path=output_csv_path)
        print("Submission file created")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the best solution")
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to the model file",
        required=False,
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
        required=False,
    )
    args = parser.parse_args()

    model_path = args.model_path
    output_csv_path = args.output_csv_path
    save = args.save

    st = time.time()

    run_best_solution(model_path, output_csv_path, save)
    # get the end time
    et = time.time()
    # get the execution time in minutes
    elapsed_time = et - st
    minutes, seconds = divmod(elapsed_time, 60)

    print(f"Process finished in {int(minutes)} min {int(seconds)} sec")