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

def moravec_detector(image):
    """
    Assume you’re working with a 3x3 window. This function should return a list of keypoints ((x,y) coordinates) in the image
    :param image:
    :return list:
    """
    keypoints = []
    threshold = 100_000 # might change later, 10000 is a good starting point

    padded_image = np.pad(image, ((2, 2), (2, 2)), mode='constant') # pad the image so it doesn't go out of bounds, 2 is good because of the 3x3 window, so that corners can be calculated for as well
    height, width = padded_image.shape

    # iterate over pixels in the new image
    for y in range(4, height-4):
        for x in range(4, width-4):
            sw = 0

            # my window
            window = padded_image[y - 1:y + 2, x - 1:x + 2]

            # will be used to calculate the sum of squared differences in all 8 directions
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0: # skip the center pixel
                        continue

                    # get the window for the new pixel
                    new_window = padded_image[y + dy - 1:y + dy + 2, x + dx - 1:x + dx + 2]

                    # calculate the sum of squared differences
                    ssd = 0
                    for r_index in range(len(window)):
                        for c_index in range(len(window)):
                            ssd += (float(window[r_index][c_index]) - float(new_window[r_index][c_index]))
                    ssd = ssd ** 2 # square the result

                    if ssd > sw: # if the ssd is greater than the current sw, update sw
                        sw = ssd

            if sw > threshold:
                keypoints.append([x - 2, y - 2]) # subtract 2 to get the original image coordinates

    return keypoints

def plot_keypoints(image, keypoints):
    """
    Plots keypoints on an image in red
    :param image:
    :param keypoints:
    :return new_image:
    """
    # make image a color image
    return_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

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
    LBP_neighbors = [(1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1)]

    feature_vector = []
    x, y = keypoint

    # pad the image
    padded_image = np.pad(image, ((8, 8), (8, 8)), mode='edge')

    # get the 16x16 window
    window = padded_image[y - 8:y + 8, x - 8:x + 8]

    # pad the window for when calculating around the window and it doesn't give us an out of bounds error
    padded_window = np.pad(window, ((1, 1), (1, 1)), mode='edge')

    for i in range(16):
        for j in range(16):
            center = window[i, j]
            binary_string = ""

            for index, (dx, dy) in enumerate(LBP_neighbors):
                padded_y = i + dy + 1
                padded_x = j + dx + 1
                pixel = padded_window[padded_y, padded_x]

                if pixel >= center:
                    binary_string += "1"
                else:
                    binary_string += "0"

            feature_vector.append(int(binary_string, 2))

    histogram = np.zeros(256)
    for value in feature_vector:
        histogram[value] += 1

    # normalize the histogram
    histogram = histogram / np.sum(histogram)

    return np.array(histogram)

def extract_HOG(image, keypoint):
    """
    Assume you are working with one keypoint. This function should return a single feature vector
    :param image:
    :param keypoint:
    :return feature_vector:
    """
    x, y = keypoint
    cell_size = 8
    block_size = 2 # num of cells in a block
    num_bins = 9 # number of bins in the histogram

    # calculate image gradients
    i_x = np.gradient(image)[1]
    i_y = np.gradient(image)[0]

    # calculate the histogram
    descriptor = []

    # iterate over the window
    for i in range(0, 16, cell_size):
        for j in range(0, 16, cell_size):
            cell_magnitude = []
            cell_orientation = []

            # Calculate gradients within cell
            for k in range(cell_size):
                for z in range(cell_size):
                    cell_magnitude.append(math.sqrt(i_x[y + i + k, x + j + z] ** 2 + i_y[y + i + k, x + j + z] ** 2))
                    cell_orientation.append(math.atan2(i_y[y + i + k, x + j + z], i_x[y + i + k, x + j + z]))

            histogram = np.zeros(num_bins)
            for magnitude, orientation in zip(cell_magnitude, cell_orientation):
                bin_index = int(orientation / (2 * np.pi / num_bins))
                histogram[bin_index] += magnitude

            histogram /= np.linalg.norm(histogram) + 1e-5
            descriptor.extend(histogram)

    # normalize the descriptor
    normalized_descriptor = []
    for i in range(0, len(descriptor), block_size):
        block = descriptor[i:i+block_size]
        block /= np.linalg.norm(block) + 1e-5
        normalized_descriptor.extend(block)

    return np.array(normalized_descriptor)

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
        padded_image1 = np.pad(image1, ((8, 8), (8, 8)), mode='constant')
        padded_image2 = np.pad(image2, ((8, 8), (8, 8)), mode='constant')
        features1 = [extract_HOG(padded_image1, keypoint) for keypoint in keypoints1]
        features2 = [extract_HOG(padded_image2, keypoint) for keypoint in keypoints2]

    # perform feature matching
    matches = []
    threshold = 2 # might change later

    for i, feature1 in enumerate(features1):
        best_match_index = None
        best_match_distance = float('inf')

        for j, feature2 in enumerate(features2):
            # Compute euclidean distance between the two features
            distance = np.linalg.norm(feature1 - feature2)

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
    # plot keypoints on each image
    img1 = plot_keypoints(image1, matches[0])
    img2 = plot_keypoints(image2, matches[1])

    #rezise the images to the same size
    max_height = max(img1.shape[0], img2.shape[0])

    # blank image
    result_image = np.zeros((max_height, img1.shape[1] + img2.shape[1], 3), dtype=np.uint8)

    # combine the two images
    result_image[:img1.shape[0], :img1.shape[1]] = img1

    # plot black for blank space depending on the size of the images
    if img1.shape[0] < img2.shape[0]:
        result_image[:img2.shape[0], img1.shape[1]:] = img2
        result_image[img1.shape[0]:, :img1.shape[1]] = (0, 0, 0)
    else:
        result_image[:img2.shape[0], img1.shape[1]:] = img2
        result_image[img2.shape[0]:, img1.shape[1]:] = (0,0,0)

    # plot the lines connecting the keypoints
    for (x1, y1), (x2, y2) in zip(matches[0], matches[1]):
        x2 += image1.shape[1] # shift the x coordinate of the second image
        cv2.line(result_image, (x1, y1), (x2, y2), (0, 0, 255), 1) # plot the line

    return result_image

def harris_detector(image):
    """
    This function should return a list of keypoints ((x,y) coordinates) in the image
    :param image:
    :return list:
    """
    # smooth the image first
    smoothed_image = cv2.GaussianBlur(image, (5, 5), 0)

    # compute the horizontal and vertical derivatives
    i_x = cv2.Sobel(smoothed_image, cv2.CV_64F, 1, 0, ksize=3)
    i_y = cv2.Sobel(smoothed_image, cv2.CV_64F, 0, 1, ksize=3)

    # compute the products of the derivatives
    i_x_squared = i_x ** 2
    i_y_squared = i_y ** 2
    i_x_i_y = i_x * i_y

    # convolve the products with a Gaussian window
    i_x_squared = cv2.GaussianBlur(i_x_squared, (5, 5), 0)
    i_y_squared = cv2.GaussianBlur(i_y_squared, (5, 5), 0)
    i_x_i_y = cv2.GaussianBlur(i_x_i_y, (5, 5), 0)

    # compute R(aw)
    R = np.zeros(image.shape) # initialize the R matrix
    k = 0.04 # empirical constant
    for y in range(image.shape[0]):
        for x in range(image.shape[1]):
            M = np.array([[i_x_squared[y, x], i_x_i_y[y, x]], [i_x_i_y[y, x], i_y_squared[y, x]]]) # create the M matrix
            R[y, x] = np.linalg.det(M) - k * (np.trace(M) ** 2) # calculate the R value

    # find the local maxima
    keypoints = []
    threshold = 0.5 * np.max(R) # threshold value
    for y in range(1, image.shape[0] - 1):
        for x in range(1, image.shape[1] - 1):
            if R[y, x] > threshold and R[y, x] > R[y - 1, x - 1] and R[y, x] > R[y - 1, x] and R[y, x] > R[y - 1, x + 1] and R[y, x] > R[y, x - 1] and R[y, x] > R[y, x + 1] and R[y, x] > R[y + 1, x - 1] and R[y, x] > R[y + 1, x] and R[y, x] > R[y + 1, x + 1]:
                keypoints.append([x, y])

    return keypoints