# from PIL import Image
import cv2
import numpy as np
import math

def load_img(file_name):
    """
    Returns an image from a file
    :param file_name:
    :return:
    """
    # https://www.geeksforgeeks.org/python-pil-imageops-greyscale-method/
    # https://stackoverflow.com/questions/52307290/what-is-the-difference-between-images-in-p-and-l-mode-in-pil
    # img = Image.open(file_name)
    # img = img.convert('L')
    # the 'L' converts to grayscale even if image is in color
    # return Image.open(file_name).convert('L')

    # https://www.geeksforgeeks.org/python-opencv-cv2-imread-method/
    # used for greyscaling images
    return cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
    # return cv2.imread(file_name)

def display_img(image):
    """
    Displays an image
    :param image:
    :return:
    """
    # image.show() # for PIL

    cv2.imshow('grey_image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def generate_gaussian(sigma, filter_w, filter_h):
    """
    If either filter_w or filter_h is 1, generate a 1D filter, otherwise 2D
    :param sigma:
    :param filter_w:
    :param filter_h:
    :return gaussian_filter:
    """

    if (filter_w == 1 or filter_h == 1) and filter_w != filter_h: # 1D filter
        length = max(filter_w, filter_h) # in case filter_h or filter_w is 1
        gaussian_filter = np.zeros(length) # 1D array with fixed size, fixed so sum() works
        middle = length // 2

        for n in range(length):
            gaussian_filter[n] = math.exp(- ((n - middle)**2) / (2 * sigma**2))

        return gaussian_filter / sum(gaussian_filter)

    else: # 2D filter
        if filter_w != filter_h: # just in case
            raise ValueError('filter_w and filter_h must be equal for 2D filter')

        length = filter_w # doesn't matter if filter_w or filter_h
        middle = length // 2
        gaussian_filter = np.zeros((length, length))
        for y in range(length):
            for x in range(length):
                gaussian_filter[y, x] = math.exp(- ((y - middle)**2 + (x - middle)**2) / (2 * sigma**2))
        return gaussian_filter / sum(gaussian_filter)

def apply_filter(image, filter, pad_pixels, pad_value):
    """
    Takes, image, filter, number of pixels to pad on each side, and the value with which to pad.
    :param image:
    :param filter:
    :param pad_pixels:
    :param pad_value:
    :return:
    """

    # first pad the image
    padded_image = []
    if pad_value == 0:
        padding = [[0] * pad_pixels for _ in range(len(image[0]) + 2 * pad_pixels)]
    else:
        padding = [image[0][:pad_pixels] for _ in range(pad_pixels)]

    for row in image:
        padded_row = padding + row + padding
        padded_image.append(padded_row)

    # apply the filter
    filtered_image = []
    filter_h = len(filter)
    filter_w = len(filter[0])

    for y in range(pad_pixels, len(padded_image) - pad_pixels):
        filtered_row = []
        for x in range(pad_pixels, len(padded_image[y]) - pad_pixels):
            pixel_val = 0
            for fy in range(filter_h):
                for fx in range(filter_w):
                    pixel_val += filter[fy][fx] * padded_image[y + fy - pad_pixels][x + fx - pad_pixels]
            filtered_row.append(pixel_val)
        filtered_image.append(filtered_row)

    return filtered_image

def median_filtering(image, filter_w, filter_h):
    """
    Takes an image and the width and height of the filter
    :param image:
    :param filter_w:
    :param filter_h:
    :return filtered_image:
    """

    # get dimensions of the image
    numDimensions = len(image.shape) # check how many dimensions the image has
    pad_w, pad_h = filter_w // 2, filter_h // 2

    if numDimensions == 2: # grayscale images only
        height, width = image.shape

        # initiate a filtered image
        filtered_image = np.zeros((height, width), dtype=image.dtype)

        for y in range(height):
            for x in range(width):
                    neighbors = []
                    for j in range(-pad_h, pad_h + 1):
                        for i in range(-pad_w, pad_w + 1):
                            if 0 <= y + j < height and 0 <= x + i < width:
                                neighbors.append(image[y + j, x + i])
                    neighbors.sort()
                    median_val = neighbors[len(neighbors) // 2]
                    filtered_image[y, x] = median_val
    else: # colored images
        height, width, channels = image.shape

        # initiate a filtered image
        filtered_image = np.zeros((height, width, channels), dtype=image.dtype)

        # pad the image
        for y in range(height):
            for x in range(width):
                for c in range(channels):
                    neighbors = []
                    for j in range(-pad_h, pad_h + 1):
                        for i in range(-pad_w, pad_w + 1):
                            if 0 <= y + j < height and 0 <= x + i < width:
                                neighbors.append(image[y + j, x + i, c])
                    neighbors.sort()
                    median_val = neighbors[len(neighbors) // 2]
                    filtered_image[y, x, c] = median_val

    return filtered_image

def hist_eq(image):
    """
    Takes an image and returns the histogram equalized image
    :param image:
    :return filtered_image:
    """

    if len(image.shape) == 3: # this is for colored images only
        # convert image to grayscale
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # this is to calculate the histogram
    histogram = [0] * 256
    total_pixels = 0
    for row in image:
        for pixel in row:
            histogram[pixel] += 1
            total_pixels += 1

    # calculate the cdf
    cdf = [sum(histogram[:i+1]) for i in range(256)]

    # normalization
    cdf_min = min(cdf)
    cdf_normalized = [int((cdf[i] - cdf_min) / (total_pixels - cdf_min) * 255) for i in range(256)]

    # apply the cdf to the image
    hist_image = np.zeros(image.shape, dtype=image.dtype)
    for y in range(len(image)):
        for x in range(len(image[y])):
            hist_image[y, x] = cdf_normalized[image[y, x]]
    return hist_image

def rotate(image, theta):
    """
    Takes an image and a rotation angle in radians
    :param image:
    :param theta:
    :return transformed_image:
    """

    if len(image.shape) == 3: # colored images only
        # dimensions of the image
        height, width, channels = image.shape

        # initiate a rotated image
        rotated_image = np.zeros((height, width, channels), dtype=image.dtype)

    else: # greyscale images
        height, width = image.shape
        rotated_image = np.zeros((height, width), dtype=image.dtype)

    # center of the image
    center_x, center_y = width // 2, height // 2

    # rotate the image
    for y in range(height):
        for x in range(width):
            # equations from powerpoint 4 slide 90
            new_x = int(center_x + (x - center_x) * math.cos(theta) - (y - center_y) * math.sin(theta))
            new_y = int(center_y + (x - center_x) * math.sin(theta) + (y - center_y) * math.cos(theta))

            # make sure new coordinates are withing image dimensions
            if 0 <= new_x < width and 0 <= new_y < height:
                rotated_image[y, x] = image[new_y, new_x]

    return rotated_image

def edge_detection(image):
    """
    Takes an image and returns the edge detected image
    :param image:
    :return filtered_image:
    """

    # Step 1: Smoothing (Gaussian filter)

    # Step 2: Enhancement

    # Step 3: Detection/Thresholding

    # Step 4: Localization

    return