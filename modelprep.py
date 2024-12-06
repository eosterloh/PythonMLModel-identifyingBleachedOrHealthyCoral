import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from tqdm import tqdm


# skimage imports
from skimage.io import imread
from skimage.filters import sobel, sato, gaussian
from skimage.feature import hog
from skimage.transform import resize

# sklearn imports
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression, RidgeClassifier, Lasso
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.naive_bayes import GaussianNB


# initializing empty lists to store features and labels as we iterate through different image folders
features = []
labels = []
# creating a variable for a standard image size
size = 64


# dictionary to store the pathway of the different datasets downloaded from Kaggle
directories = {"testing":"/Users/erickosterloh/Downloads/AppliedPython1/BleachedCoralsAndHealthyCoralsClassification/Testing",
                "training":"/Users/erickosterloh/Downloads/AppliedPython1/BleachedCoralsAndHealthyCoralsClassification/Training",
                "validation":"/Users/erickosterloh/Downloads/AppliedPython1/BleachedCoralsAndHealthyCoralsClassification/Validation"}


# iterate over each of those datasets to define paths to each image
for set, dir in directories.items():
    for classification in ['bleached_corals', 'healthy_corals']:
        # need to define the folder path for each category (bleached, ubleached for each data set)
        class_folder = os.path.join(dir, classification) # .join allows us to concatenate pathways into one path
        # we are defining the pathway for each sub-folder (ie testing--> unbleached vs testing--> healthy)

        # need to get all those images in that class class folder
        for image_name in tqdm(os.listdir(class_folder)): #listdir creates a list of all of the files stored in class_folder
            # get image path:
            image_path = os.path.join(class_folder, image_name) # creates a unique path for each image in each folder

            # load image using skimage
            image = imread(image_path) # imread converts an image into a numpy array
            image_resized = resize(image, (size, size), mode='reflect')

            # create color channels:
            red_channel = image_resized[:, :, 0] # each are an array of shape (size, size, 1)
            green_channel = image_resized[:, :, 1]

            # get mean pixel value for each color channel
            red_mean = np.mean(red_channel)
            green_mean = np.mean(green_channel)
            #blue_mean = np.mean(blue_channel)

            # applying skimage filters and getting average pixel value of filtered image
            #gaus_avg = np.mean(gaussian(image_resized))
            gaus_max = np.max(gaussian(image_resized))
            gaus_min = np.min(gaussian(image_resized))
            gaus_std = np.std(gaussian(image_resized))

            #sob_avg = np.mean(sobel(image_resized))
            sob_max = np.max(sobel(image_resized))
            #sob_min = np.min(sobel(image_resized))
            sob_std = np.std(sobel(image_resized))

            sato_avg = np.mean(sato(image_resized))
            #sato_max = np.max(sato(image_resized))
            #sato_min = np.min(sato(image_resized))
            sato_std = np.std(sato(image_resized))

            #im_right_avg = np.mean(image_resized[:,size//2:, :])
            im_left_avg = np.mean(image_resized[:,:size//2, :])
            im_max = np.max(image_resized)
            im_min = np.min(image_resized)
            #im_std = np.std(image_resized)

            # storing each feature in a vector / list
        #    feature_vector = [float(red_mean), float(green_mean), float(sob_avg), float(sato_avg),
        #    float(gaus_max), float(gaus_min), float(gaus_std), float(sob_max), float(sob_min), float(sob_std),
        #float(sato_std), float(im_left_avg), float(im_max), float(im_min)]
            feature_vector = [float(red_mean), float(green_mean), float(blue_mean), float(gaus_avg), float(sob_avg), float(sato_avg),
            float(gaus_max), float(gaus_min), float(gaus_std), float(sob_max), float(sob_min), float(sob_std), float(sato_max),
        float(sato_min), float(sato_std), float(im_right_avg), float(im_left_avg), float(im_max), float(im_min), float(im_std)]

            # appending the feature vector for each image to a cumulative features list
            features.append(feature_vector)

            # appending the label associated with each image to a labels list
            labels.append(classification)


# converting the features and labels list into numpy arays
X = np.array(features) # has shape (9292x20)
Y = np.array(labels) # has shape (9292x1)


# saving arrays to .npy files
np.save('X_array', X)
np.save('Y_array', Y)
