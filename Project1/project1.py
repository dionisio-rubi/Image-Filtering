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
    # return cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
    return cv2.imread(file_name)

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

    if (filter_w == 1 or filter_h == 1): # 1D filter
        length = max(filter_w, filter_h) # in case filter_h or filter_w is 1
        gaussian_filter = np.zeros((length, 1)) # 1D array with fixed size, fixed so sum() works
        middle = length // 2

        for n in range(length):
            gaussian_filter[n] = math.exp(- ((n - middle)**2) / (2 * sigma**2))

        return gaussian_filter / np.sum(gaussian_filter)

    else: # 2D filter
        gaussian_filter = np.zeros((filter_w, filter_h))
        middle_h, middle_w = filter_h // 2, filter_w // 2

        for y in range(filter_w):
            for x in range(filter_h):
                gaussian_filter[y, x] = math.exp(- ((y - middle_w)**2 + (x - middle_h)**2) / (2 * sigma**2))
        return gaussian_filter / np.sum(gaussian_filter)

def apply_filter(image, filter, pad_pixels, pad_value):
    """
    Takes, image, filter, number of pixels to pad on each side, and the value with which to pad.
    :param image:
    :param filter:
    :param pad_pixels:
    :param pad_value:
    :return:
    """
    if len(image.shape) == 3: # colored images only
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # is_color_image = len(image.shape) == 3
    # pad the image
    if pad_value == 0:
        padded_img = np.pad(image, pad_pixels, mode='constant', constant_values=pad_value)
    else:
        padded_img = np.pad(image, pad_pixels, mode='edge')

    # determine the dimensions of the filter
    filter_h, filter_w = filter.shape

    # determine the dimensions of the image
    height, width = image.shape
    # filtered_image = np.zeros((height, width), dtype=image.dtype)
    filtered_image = np.zeros_like(padded_img)

    # apply filter
    if filter_h == 1 or filter_w == 1: # 1D filter
        filter_size = len(filter)
        pad = filter_size // 2

        for y in range(pad, height - pad):
            for x in range(pad, width - pad):
                filtered_image[y, x] = np.sum(padded_img[y - pad:y + pad + 1, x] * filter)
        filter_size = len(filter)
        pad = filter_size // 2

        # # Perform 1D convolution
        # h_padded, _ = padded_img.shape
        # mask_length = filter.shape[0] - 1
        # total_padding = 2 * pad_pixels
        # total_trim_length = max(total_padding, mask_length)
        #
        # output_height = h_padded - total_trim_length
        # output = np.zeros((output_height, width), dtype=float)
        #
        # trim_length = total_trim_length // 2
        # for i in range(trim_length, h_padded - trim_length):
        #     for j in range(pad, width - pad):
        #         output[i - trim_length, j - pad] = np.sum(
        #             padded_img[i, j - pad:j + filter.shape[0] - pad] * filter)
        #
        # # Adjust dimensions of filtered_image
        # filtered_height = output_height
        # filtered_width = width
        # filtered_image = np.zeros((filtered_height, filtered_width), dtype=padded_img.dtype)
        #
        # # Assign output to filtered_image
        # filtered_image[:, :] = output

    else: # 2D filter
        pad_h, pad_w = filter_h // 2, filter_w // 2

        for y in range(pad_h, height - pad_h):
            for x in range(pad_w, width - pad_w):
                filtered_image[y, x] = np.sum(padded_img[y - pad_h:y + pad_h + 1, x - pad_w:x + pad_w + 1] * filter)

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
    gaussian_filter = generate_gaussian(1, 5, 5) # similar to LOG ?
    gaus1d = generate_gaussian(1, 5, 1)
    smoothed_image = apply_filter(image, gaussian_filter, 2, 0)


    # display_img(smoothed_image)

    # Step 2: Enhancement (gradient magnitude)
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    # calculate the gradient magnitude
    gradient_x = apply_filter(smoothed_image, sobel_x, 1, 0)
    gradient_y = apply_filter(smoothed_image, sobel_y, 1, 0)
    gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
    gradient_magnitude = (gradient_magnitude*255).astype(np.uint8)
    # display_img(gradient_magnitude)

    # Step 3: Detection/Thresholding
    threshold = 100 # can be changed
    bw_image = np.where(smoothed_image > threshold, 0, 255)
    filtered_image = bw_image.astype(np.uint8)

    # display_img(filtered_image)

    # Step 4: Localization (Canny edge detector, hysteresis thresholding)
    height, width = filtered_image.shape
    edge_image = np.zeros((height, width))

    # neighborhood for connectivity
    neighborhood = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])

    # locate string edge pixels
    high_threshold = 255 # can be changed
    low_threshold = 200 # can be changed
    strong_pixels = np.where(filtered_image >= high_threshold, 1, 0)

    # perform hysteresis thresholding
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            if strong_pixels[y, x] == 1:
                edge_image[y, x] = 255
                edge_image[y-1:y+2, x-1:x+2] += neighborhood * (filtered_image[y-1:y+2, x-1:x+2] >= low_threshold)
    final_image = np.where(edge_image >= 255, 255, 0).astype(np.uint8)
    # final_image = apply_filter(image, gaus1d, 0, 0)
    return final_image