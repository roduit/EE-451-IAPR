# -*- coding: utf-8 -*-
# -*- authors : Vincent Roduit -*-
# -*- date : 2024-05-03 -*-
# -*- Last revision: 2024-05-03 -*-
# -*- python version : 3.9.18 -*-
# -*- Description: Function to visualize results -*-

# Importing libraries
import os
import matplotlib.pyplot as plt

# Importing files
import constants



def display_gray(original, img):
    fig = plt.figure()

    fig.add_subplot(1, 2, 1)
    plt.axis('off')
    plt.title("Original")
    plt.imshow(original)

    fig.add_subplot(1, 2, 2)
    plt.axis('off')
    plt.title("Gray/Blur")
    plt.imshow(img, cmap='gray')

    plt.show()

def save_coins_classified(df_images_labels, images):
    classification_path = os.path.join(constants.RESULT_PATH, 'coins_classified')

    for category in df_images_labels['label'].unique():
        category_for_path = category.replace('.', '_')
        category_path = os.path.join(classification_path, category_for_path)
        if not os.path.exists(category_path):
            os.makedirs(category_path)

        for idx, img in df_images_labels[df_images_labels['label'] == category].iterrows():
            img_path = os.path.join(category_path, img['coin_name'])
            plt.imshow(images[idx])
            plt.savefig(img_path)
            plt.close()