# Created by Rubi Dionisio
# Date: 05/05/2024

import cv2
import numpy as np
import math
from sklearn import svm
from sklearn.cluster import KMeans

def load_img(file_name):
    """
    Returns an image from a file
    :param file_name:
    :return:
    """
    # https://www.geeksforgeeks.org/python-opencv-cv2-imread-method/
    # used for greyscaling images
    return cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)

def display_img(image):
    """
    Displays an image
    :param image:
    :return:
    """
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def generate_vocabulary(train_data_file):
    """
    This function takes a list of training images in txt format. An example is provided with the test script. This function should return a set of visual vocabulary “words” in the form of vectors. Each of these vectors will be used as a bin in our histogram of features.
    :param train_data_file:
    :return:
    """
    # open and read file
    file_paths = open(train_data_file, 'r').readlines()
    file_paths = [path.strip().split() for path in file_paths]

    # extract features from each image
    descriptors = []
    for path in file_paths:
        image = load_img(path[0])

        # use SIFT to extract features
        sift = cv2.SIFT_create()

        _, desc = sift.detectAndCompute(image, None) # get descriptors

        descriptors.extend(desc)

    # perform k-means clustering to create visual vocabulary
    kmeans = KMeans(n_clusters=100)
    kmeans.fit(descriptors)

    # the cluster centers are our visual words
    vocabulary = kmeans.cluster_centers_

    return vocabulary

def extract_features(image, vocabulary):
    """
    This function takes an image and the vocabulary as input and extracts features, generating a BOW count vector.
    :param image:
    :param vocabulary:
    :return:
    """
    sift = cv2.SIFT_create()

    _, descriptors = sift.detectAndCompute(image, None) # get descriptors

    # create a BOW count vector
    bow_vector = np.zeros(len(vocabulary))
    for desc in descriptors:
        min_dist = np.inf
        min_index = 0
        for i, word in enumerate(vocabulary):
            dist = np.linalg.norm(desc - word)
            if dist < min_dist:
                min_dist = dist
                min_index = i
        bow_vector[min_index] += 1

    return bow_vector

def train_classifier(train_data_file, vocab):
    """
    This function takes the training data file and the vocabulary, extracts the features from each training image, and trains a classifier (perceptron, KNN, SVM) on the data. You can choose which classifier you’d like to use. You can use scikit learn for this.
    :param train_data_file:
    :param vocab:
    :return:
    """
    # open and read file
    file_paths = open(train_data_file, 'r').readlines()
    file_paths = [path.strip().split() for path in file_paths]

    # train classifier
    classifier = svm.SVC()
    features = []
    labels = []
    for path in file_paths:
        image = load_img(path[0])
        features.append(extract_features(image, vocab))
        labels.append(int(path[1]))  # ensure labels are numerical

    classifier.fit(features, labels)
    return classifier

def classify_image(classifier, test_img, vocabulary):
    """
    This function takes the trained classifier, a test image and the vocabulary as inputs. It generates the feature vector for the test image using the vocabulary and runs it through the classifier, returning the output classification
    :param classifier:
    :param test_img:
    :param vocabulary:
    :return:
    """
    # extract features from test image
    features = extract_features(test_img, vocabulary)

    # classify image
    prediction = classifier.predict([features])

    # purpose of returning a string value, much nicer to see
    # class_names = {0: 'Dog', 1: 'Cat'}
    # return class_names[prediction[0]]

    return prediction[0] # return numerical value, 0 = Dog, 1 = Cat

# ~~~~~~~~~~~ Image Segmentation ~~~~~~~~~~~
def threshold_image(image, low_thresh, high_thresh):
    """
    This function will take an image and two thresholds and perform hysteresis thresholding, producing a black and white image.
    :param image:
    :param low_thresh:
    :param high_thresh:
    :return:
    """
    for i in range(len(image)):
        for j in range(len(image[i])):
            if image[i][j] < low_thresh:
                image[i][j] = 0 # object
            elif image[i][j] > high_thresh:
                image[i][j] = 255 # background
            else:
                # if pixel is between low and high threshold, check if any of the 8 neighbors are above high threshold, if adjacent pixels are "object" (<low thresh) then pixel is also "object"
                # if any of the 8 neighbors are above high threshold, set pixel to 255
                #pad image with 0s so it doesn't go out of bounds
                temp_img = np.pad(image, 1, mode='constant', constant_values=0)
                if temp_img[i-1][j-1] == 255 or temp_img[i-1][j] == 255 or temp_img[i-1][j+1] == 255 or temp_img[i][j-1] == 255 or temp_img[i][j+1] == 255 or temp_img[i+1][j-1] == 255 or temp_img[i+1][j] == 255 or temp_img[i+1][j+1] == 255:
                    image[i][j] = 255
                else:
                    image[i][j] = 0

    return image

def grow_regions(image):
    """
    This function will take an image as input. Use one of the techniques from class to perform region growing, returning the output region map.
    :param image:
    :return:
    """
    # have an empty image to store the region map
    region_map = np.zeros_like(image)

    # create label for each region
    label = 1

    # iterate through each pixel in the image
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if region_map[i][j] == 0:
                stack = [(i, j)]  # start a new region with current pixel as seed if pixel is not in the region map

                while stack:
                    x, y = stack.pop()
                    region_map[x][y] = label  # add pixel to region map

                    # get neighbors
                    neighbors = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
                    # make sure no neighbors are out of bounds
                    neighbors = [(nx, ny) for nx, ny in neighbors if
                                 0 <= nx < image.shape[0] and 0 <= ny < image.shape[1]]

                    for nx, ny in neighbors:
                        # P = |I(x, y) - I(x', y')| < 0.1 * (max(I) - min(I))
                        if region_map[nx][ny] == 0 and abs(int(image[nx][ny]) - int(image[x][y])) < 0.1 * (
                                np.max(image) - np.min(image)):
                            stack.append((nx, ny))
            label += 1

    return region_map

def split_regions(image):
    """
    This function will take an image as input. Use one of the techniques from class to perform region splitting, returning the output region map.
    :param image:
    :return:
    """
    # have an empty image to store the region map
    region_map = np.zeros_like(image)

    # we want a queue to store the regions that need to be processed
    queue = [(0, 0, image.shape[0], image.shape[1])]

    # create label for each region
    label = 1

    while queue:
        x, y, h, w = queue.pop(0)
        region = image[x:x+h, y:y+w]

        # check if region is homogeneous, if not split region into 4 quadrants
        if np.std(region) > 10 and h > 2 and w > 2:
            queue.append((x, y, h // 2, w // 2))
            queue.append((x + h // 2, y, h // 2, w // 2))
            queue.append((x, y + w // 2, h // 2, w // 2))
            queue.append((x + h // 2, y + w // 2, h // 2, w // 2))
        else:
            # if region is homogeneous, assign label to region
            region_map[x:x + h, y:y + w] = label
            label += 1

    return region_map

def merge_regions(image):
    """
    This function will take an image as input. Use one of the techniques from class to perform region merging, returning the output region map.
    :param image:
    :return:
    """
    # make region map
    region_map = np.zeros_like(image, dtype=int)

    # assign a unique label to each pixel
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region_map[i, j] = i * image.shape[1] + j

    # iterate over each pixel in the image
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):

            # iterate over the 4-connected neighborhood
            for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                ni, nj = i + di, j + dj
                # check if the neighbor is within the image bounds
                if 0 <= ni < image.shape[0] and 0 <= nj < image.shape[1]:
                    if region_map[i, j] != region_map[ni, nj]: # check if the neighbor belongs to a different region
                        # calculate the intensity difference
                        diff = abs(int(image[i, j]) - int(image[ni, nj]))

                        if diff <= 10: # if the difference is below the threshold, merge the regions
                            region_map[region_map == region_map[ni, nj]] = region_map[i, j]

    region_map = region_map.astype(np.uint8) # convert to uint8 because i was getting an error
    return region_map

def segment_image(image):
    """
    This function will take an image as input. Using different combinations of the above methods, extract three segmentation maps with labels to indicate the approach.
    :param image:
    :return:
    """

    # Initialize an empty dictionary to store the segmentation maps
    segmentation_maps = {}

    # Apply thresholding
    thresholded = threshold_image(image, 150, 200)
    segmentation_maps['Thresholding'] = thresholded

    # Apply region growing
    region_grown = grow_regions(image)
    segmentation_maps['Region Growing'] = region_grown

    # Apply region merging
    # region_merged = split_regions(image)
    region_merged = merge_regions(image)
    segmentation_maps['Region Merging'] = region_merged

    return thresholded, region_grown, region_merged

# ~~~~~~~~~~~ Image Segmentation with K-Means ~~~~~~~~~~~
def kmeans_segment(image):
    """
    Use Kmeans to perform image segmentation. You’re free to do this however you’d like. Do not assume the number of classes is 2. So you’ll want to implement a method for determining what k should be.
    :param image:
    :return:
    """
    # reshape the image to a 2D array of pixels
    pixels = np.float32(image.reshape(-1, 1))

    # define the criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

    # determine the optimal number of clusters
    optimal_k = 3 # the higher the number, the better it looks, though 3 is a good ground

    # apply kmeans
    _, labels, centers = cv2.kmeans(pixels, optimal_k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    centers = np.uint8(centers) # convert the centers to 8-bit values
    labels = labels.flatten() # flatten the labels array

    # convert the labels to the original shape
    segmented_image = centers[labels.flatten()]
    segmented_image = segmented_image.reshape(image.shape)

    return segmented_image