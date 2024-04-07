# Created by Rubi Dionisio
# Date: 04/01/2024

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

# def moravec_detector(image):
#     """
#     Assume you’re working with a 3x3 window. This function should return a list of keypoints ((x,y) coordinates) in the image
#     :param image:
#     :return list:
#     """
#     keypoints = []
#     threshold = 55000 # might change later, 10000 is a good starting point
#
#     padded_image = np.pad(image, ((1, 1), (1, 1)), mode='constant') # pad the image so it doesn't go out of bounds
#     height, width = padded_image.shape
#
#     # iterate over pixels in the new image
#     for y in range(1, height - 1):
#         for x in range(1, width - 1):
#             # need to keep track of all sw
#             sw_values = []
#
#             # will be used to calculate the sum of squared differences in all 8 directions
#             for dx in [-1, 0, 1]:
#                 for dy in [-1, 0, 1]:
#                     if dx == 0 and dy == 0: # skip the center pixel
#                         continue
#                     # sw = np.sum((padded_image[y, x] - padded_image[y + dy, x + dx]) ** 2)
#                     # sw_values.append(sw)
#                     # Compute SSD within the 3x3 window
#                     center_pixel = padded_image[y, x]
#                     neighbor_pixel = padded_image[y + dy, x + dx]
#                     diff = center_pixel - neighbor_pixel
#                     sw_values.append(diff ** 2)
#
#             min_sw = min(sw_values)
#
#             if min_sw > threshold:
#                 keypoints.append((x - 1, y - 1)) # subtract 1 to get the original image coordinates
#
#     return keypoints

def moravec_detector(img):
    feature_coords = []
    pad_amt = 2
    img_padded = np.pad(img, ((pad_amt,pad_amt),(pad_amt,pad_amt)), mode="constant")
    for y in range(pad_amt+2,len(img_padded)-pad_amt-2):
        for x in range(pad_amt+2,len(img_padded[0])-pad_amt-2):
            s = 0
            # grab the window
            center_window = img_padded[y-1:y+2, x-1:x+2]
            for window_y in (-1,0,1):
                for window_x in (-1,0,1):
                    new_window = img_padded[y+window_y-1:y+window_y+2, x+window_x-1:x+window_x+2]

                    total = 0
                    for r_index in range(len(center_window)):
                        for c_index in range(len(new_window)):
                            total += float(center_window[r_index][c_index]) - float(new_window[r_index][c_index])
                    s_xy = total**2
                    if s_xy > s:
                        s = s_xy
            if s > 55000:
                feature_coords.append([x-pad_amt,y-pad_amt])
    return feature_coords

def plot_keypoints(image, keypoints):
    """
    Plots keypoints on an image in red
    :param image:
    :param keypoints:
    :return new_image:
    """
    # create a copy of the image so that original isn't modified
    return_image = image.copy()

    # make return_image a color image
    return_image = cv2.cvtColor(return_image, cv2.COLOR_GRAY2BGR)

    for x, y in keypoints:
        cv2.circle(return_image, (x, y), radius=1, color=(0, 0, 255), thickness=-1) # plot the circle

    return return_image

def extract_LBP(image, keypoint):
    """
    Assume you are working with one keypoint. This function should return a single feature vector
    :param image:
    :param keypoint:
    :return feature_vector:
    """
    feature_vector = []

    # get a 16x16 window around the keypoint, padding if necessary
    x, y = keypoint
    # pad image first to avoid out of bounds error
    padded_image = np.pad(image, ((8, 8), (8, 8)), mode='constant')
    window = padded_image[y-8:y+8, x-8:x+8]

    # iterate over the window
    for i in range(16):
        for j in range(16):
            # get the center pixel
            center_pixel = window[i, j]
            binary_string = ""
            # iterate over the 3x3 window
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    # get the pixel value
                    pixel_y = i + dy + 8
                    pixel_x = j + dx + 8
                    # print(pixel_y, pixel_x, window.shape, window[pixel_y, pixel_x])
                    pixel = window[pixel_y, pixel_x]

                    if pixel >= center_pixel:
                        binary_string += "1"
                    else:
                        binary_string += "0"
            # convert the binary string to an integer
            feature_vector.append(int(binary_string, 2))

    # create a histogram
    histogram = np.zeros(256)
    for value in feature_vector:
        histogram[value] += 1

    # normalize the histogram
    histogram = histogram / np.sum(histogram)

    return histogram

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
    # check if detector is valid
    if detector.lower() not in ["moravec", "harris"]:
        raise ValueError("Invalid detector! Must be either Moravec or Harris")

    # check if extractor is valid
    if extractor.lower() not in ["lbp", "hog"]:
        raise ValueError("Invalid extractor! Must be either LBP or HOG")

    # get keypoints
    if detector.lower() == "moravec":
        keypoints1 = moravec_detector(image1)
        keypoints2 = moravec_detector(image2)
    else:
        keypoints1 = harris_detector(image1)
        keypoints2 = harris_detector(image2)

    # extract features
    if extractor.lower() == "lbp":
        features1 = [extract_LBP(image1, keypoint) for keypoint in keypoints1]
        features2 = [extract_LBP(image2, keypoint) for keypoint in keypoints2]
    else:
        features1 = [extract_HOG(image1, keypoint) for keypoint in keypoints1]
        features2 = [extract_HOG(image2, keypoint) for keypoint in keypoints2]

    # perform feature matching
    matches = []
    threshold = 0.1 # might change later

    for i, feature1 in enumerate(features1):
        best_match_index = None
        best_match_distance = float('inf')

        for j, feature2 in enumerate(features2):
            # Compute distance between features
            distance = abs(feature1 - feature2) # might change later

            # Check if this distance is the best match so far
            if distance < best_match_distance:
                best_match_distance = distance
                best_match_index = j

        # Check if the best match is below the threshold
        if best_match_distance < threshold:
            matches.append((i, best_match_index))

    # return the matching keypoints as one 2d array
    matched_keypoints1 = [keypoints1[i] for i, _ in matches]
    matched_keypoints2 = [keypoints2[j] for _, j in matches]

    combined_matches = [matched_keypoints1, matched_keypoints2]

    return combined_matches

def plot_matches(image1, image2, matches):
    """
    matches should be the result of the above feature_matching function, that is a 2d array. This function should return a new image, where image1 is on the left and image 2 is on the right, and there are red points marking keypoints in image1 and image2 and red lines connecting matching keypoints.
    :param image1:
    :param image2:
    :param matches:
    :return new_image:
    """
    # create a new image, with image1 on the left and image2 on the right
    result_image = np.zeros((max(image1.shape[0], image2.shape[0]), image1.shape[1] + image2.shape[1], 3), dtype=np.uint8)

    return image1

def harris_detector(image):
    """
    This function should return a list of keypoints ((x,y) coordinates) in the image
    :param image:
    :return list:
    """
    return []