# -*- coding: utf-8 -*-
# -*- author : Vincent Roduit-*-
# -*- date : 2024-05-19 -*-
# -*- Last revision: 2024-05-19 (Vincent Roduit)-*-
# -*- python version : 3.12.3 -*-
# -*- Description: Function to extract features for automatic labelisation -*-

# Import libraries
import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import silhouette_score
import pandas as pd
import os
import timm
from torch import nn
import torch

# Import files
import constants
from post_processing.data_formating import resize_images

class FeatureExtractor(nn.Module):
    def __init__(self, model):
        super(FeatureExtractor, self).__init__()
        self.features = nn.Sequential(*list(model.children())[:-1])
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return x

def extract_combined_features(image):
    """Extract combined features from an image.
    Args:
        image (np.array): The image to extract features from.
    
    Returns:
        np.array: The extracted features.
    """
    features = []

    # LBP feature
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray_image, P=20, R=3, method='ror')
    n_bins = int(lbp.max() + 1)
    lbp_hist, _ = np.histogram(lbp, bins=n_bins, range=(0, n_bins), density=True)
    lbp_hist = lbp_hist / np.sum(lbp_hist)
    features.extend(lbp_hist)
    
    # Color feature
    avg_color = cv2.mean(image)[:3]
    features.extend(avg_color)

    # image size
    features.append(image.shape[0])

    return np.array(features)

def find_gmm_labels(images, n_clusters, handcrafted=False):
    """Find GMM labels for a list of images.
    Args:
        images (List[np.array]): The images to find labels for.
        n_clusters (int): The number of clusters to use.
    
    Returns:
        List[int]: The labels for the images.
    """
    if handcrafted:
        features = [extract_combined_features(image) for image in images]
    else:
        model = timm.create_model('efficientnet_b0', pretrained=True)
        feature_extractor = FeatureExtractor(model)
        biggest_dim = max(max(image.shape[:2]) for image in images)
        coin_images_reshaped = resize_images(images, biggest_dim)
        coin_images_reshaped = resize_images(coin_images_reshaped, 224)
        coin_images_transposed = [np.transpose(image, (2,0,1)).astype(np.float32) for image in coin_images_reshaped]
        coin_images_tensor = torch.tensor(coin_images_transposed)
        features = feature_extractor(coin_images_tensor).detach().numpy()

    # Reduce dimension
    pca = PCA(n_components=30)
    features_pca = pca.fit_transform(features)

    # Perform Clustering
    gmm = GaussianMixture(n_components=n_clusters, random_state=0)
    labels = gmm.fit_predict(features_pca)

    score = silhouette_score(features_pca, labels)
    print(f'Silhouette Score: {score}')

    return labels

def associate_labels(images):
    """Associate labels to the clusters found by the GMM.
    Args:
        images (List[np.array]): The images to associate labels to.
    
    Returns:
        List[int]: The labels for the images.
    """

    labels = find_gmm_labels(images, n_clusters=constants.N_CLUSTERS)

    df_train_labels = pd.read_csv(os.path.join(constants.DATA, 'train_labels.csv'))
    known_counts = (df_train_labels.sum()[1:].values).astype(int)
    coin_label_name = df_train_labels.columns[1:]
    cluster_counts = np.bincount(labels, minlength=len(known_counts))

    # perform Optimization Search
    cost_matrix = np.abs(cluster_counts[:, np.newaxis] - known_counts)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Maapping
    cluster_to_coin_type = {row: col for row, col in zip(row_ind, col_ind)}
    for cluster, coin_type in cluster_to_coin_type.items():
        print(f"Cluster {cluster} is mapped to coin type {coin_type + 1}, which is a {coin_label_name[coin_type]}")
    
    labels = [cluster_to_coin_type[label] for label in labels]
    
    return labels