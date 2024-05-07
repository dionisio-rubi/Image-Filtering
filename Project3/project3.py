# Created by Rubi Dionisio
# Date: 05/05/2024

import cv2
import numpy as np
import math
# import scikit

def load_img(file_name):
    """
    Returns an image from a file
    :param file_name:
    :return:
    """
    # https://www.geeksforgeeks.org/python-opencv-cv2-imread-method/
    # used for greyscaling images
    return cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
    # not greyscaling images
    # return cv2.imread(file_name)

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
    features = []
    for path in file_paths:
        image = load_img(path[0])

        # use sift or corner detection to extract features
        sift = cv2.xfeatures2d.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(image, None)

        #create visual vocabulary: taking features from sift create a small window around feature, get feature vector, add to features
        for i in range(len(keypoints)):
            x, y = keypoints[i].pt
            x, y = int(x), int(y)
            window = image[x-4:x+4, y-4:y+4]
            features[0].append(window.flatten()) if path[1] == '0' else features[1].append(window.flatten())

    # convert features to numpy array
    features = np.array(features)
    print(features)
    return features


def extract_features(image, vocabulary):
    """
    This function takes an image and the vocabulary as input and extracts features, generating a BOW count vector.
    :param image:
    :param vocabulary:
    :return:
    """

    pass

def train_classifier(train_data_file, vocab):
    """
    This function takes the training data file and the vocabulary, extracts the features from each training image, and trains a classifier (perceptron, KNN, SVM) on the data. You can choose which classifier you’d like to use. You can use scikit learn for this.
    :param train_data_file:
    :param vocab:
    :return:
    """

    pass

def classify_image(classifier, test_img, vocabulary):
    """
    This function takes the trained classifier, a test image and the vocabulary as inputs. It generates the feature vector for the test image using the vocabulary and runs it through the classifier, returning the output classification
    :param classifier:
    :param test_img:
    :param vocabulary:
    :return:
    """

    pass

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

        # check if region is homogeneous
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
    # have an empty image to store the region map
    region_map = np.zeros_like(image)

    # build region adjacency graph
    region_adjacency = {}
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region_adjacency[(i, j)] = []
            if i > 0:
                region_adjacency[(i, j)].append((i - 1, j))
            if i < image.shape[0] - 1:
                region_adjacency[(i, j)].append((i + 1, j))
            if j > 0:
                region_adjacency[(i, j)].append((i, j - 1))
            if j < image.shape[1] - 1:
                region_adjacency[(i, j)].append((i, j + 1))

    return region_map

def segment_image(image):
    """
    This function will take an image as input. Using different combinations of the above methods, extract three segmentation maps with labels to indicate the approach.
    :param image:
    :return:
    """

    pass

# ~~~~~~~~~~~ Image Segmentation with K-Means ~~~~~~~~~~~
def kmeans_segment(image):
    """
    Use Kmeans to perform image segmentation. You’re free to do this however you’d like. Do not assume the number of classes is 2. So you’ll want to implement a method for determining what k should be..
    :param image:
    :return:
    """

    pass