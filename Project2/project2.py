import cv2
import numpy as np
import math

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

def moravec_detector(image):
    """
    Assume you’re working with a 3x3 window. This function should return a list of keypoints ((x,y) coordinates) in the image
    :param image:
    :return list:
    """
    keypoints = []
    height, width = image.shape
    window_size = 3
    threshold = 1000 # might change later

    # reason we start at 1 is so that we don't go out of bounds
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            sw_left = np.sum((image[y, x] - image[y, x - 1]) ** 2)
            sw_right = np.sum((image[y, x] - image [y, x + 1]) ** 2)
            sw_up = np.sum((image[y, x] - image[y - 1, x]) ** 2)
            sw_down = np.sum((image[y, x] - image[y + 1, x]) ** 2)
            sw_up_left = np.sum((image[y, x] - image[y - 1, x - 1]) ** 2)
            sw_up_right = np.sum((image[y, x] - image[y - 1, x + 1]) ** 2)
            sw_down_left = np.sum((image[y, x] - image[y + 1, x - 1]) ** 2)
            sw_down_right = np.sum((image[y, x] - image[y + 1, x + 1]) ** 2)

            min_sw = min(sw_left, sw_right, sw_up, sw_down, sw_up_left, sw_up_right, sw_down_left, sw_down_right)

            if min_sw > threshold:
                keypoints.append((x, y))

    return keypoints

def plot_keypoints(image, keypoints):
    """
    Plots keypoints on an image in red
    :param image:
    :param keypoints:
    :return new_image:
    """
    return image

def extract_LBP(image, keypoint):
    """
    Assume you are working with one keypoint. This function should return a single feature vector
    :param image:
    :param keypoint:
    :return feature_vector:
    """
    return []

def extract_HOG(image, keypoint):
    """
    Assume you are working with one keypoint. This function should return a single feature vector
    :param image:
    :param keypoint:
    :return feature_vector:
    """
    return []

def feature_matching(image1, image2, detector, extractor):
    """
    detector should either be “Moravec” or “Harris”. Anything other than these two options should result in an error. extractor should either be “LBP” or “HOG”. Any other value should result in an error. This function should return two lists of matching keypoints. For example, list 1 would have its first element (x1,y1) and list 2 would have its first element (x2,y2) and this would mean that (x1,y1) in image 1 matches with (x2,y2) in image 2. This function should call your moravec_detectoror harris_detector and your extract_LBP or extract_HOG functions.
    :param image1:
    :param image2:
    :param detector:
    :param extractor:
    :return list1, list2:
    """
    return [] , []

def plot_matches(image1, image2, matches):
    """
    matches should be the result of the above feature_matching function, that is, two lists. You can pass this however you want, as you’re the one who will use it in the function. Just make sure it is only one parameter so that my test script works. This function should return a new image, where image1 is on the left and image 2 is on the right, and there are red points marking keypoints in image1 and image2 and red lines connecting matching keypoints.
    :param image1:
    :param image2:
    :param matches:
    :return new_image:
    """
    return image1

def harris_detector(image):
    """
    This function should return a list of keypoints ((x,y) coordinates) in the image
    :param image:
    :return list:
    """
    return []