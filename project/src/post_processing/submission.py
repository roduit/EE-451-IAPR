# -*- coding: utf-8 -*-
# -*- authors : Vincent Roduit -*-
# -*- date : 2024-05-19 -*-
# -*- Last revision: 2024-05-19 -*-
# -*- python version : 3.9.18 -*-
# -*- Description: Functions to create the csv submision-*-

#import libraries
import pandas as pd
import os

#Import files 
import constants

def create_submission_file(predictions, df_test_images, conversion_table):
    """
    Create the submission file
    
    Args:
        predictions (list): List of predictions
        df_test_images (dataframe): Dataframe with the test images
        conversion_table (dict): Dictionary with the conversion table
    
    Returns:
        df_test_images (dataframe): Dataframe with the test images
    """
    # Add the predictions and convert the labels
    df_test_images['label_int'] = predictions
    df_test_images['label'] = df_test_images['label_int'].apply(lambda x: list(conversion_table.keys())[list(conversion_table.values()).index(x)])
    labels_formated = pd.get_dummies(df_test_images['label']).astype(int)
    df_test_images = pd.concat([df_test_images, labels_formated], axis=1)

    # Drop the unnecessary columns
    df_test_images.drop(columns=['label_int','label','coin_name','contour','radius'], inplace=True)

    # Format the dataframe
    df_test_images = df_test_images.groupby('image_name').sum()
    df_test_images.reset_index(inplace=True)
    df_test_images.rename(columns={'image_name':'id'}, inplace=True)
    train_labels = pd.read_csv(os.path.join(constants.DATA, 'train_labels.csv'))
    cols = train_labels.columns
    cols = [x.strip() for x in cols]
    df_test_images = df_test_images[cols]

    return df_test_images

