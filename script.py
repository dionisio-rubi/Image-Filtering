import cv2
import PIL
import matplotlib
import skimage
import numpy as np
import math
import project1 as p1

filter_size = 5
sigma = 1
theta = math.pi/2
#Load and display image
img = p1.load_img("images/test_img.jpg")
# img = p1.load_img("images/image.jpg") # colored image
# img = p1.load_img("images/sp.jpg")
# p1.display_img(img)

#Generate 1D gaussian filter
gaussian1D = p1.generate_gaussian(sigma, filter_size, 1)
# print(gaussian1D, sum(gaussian1D))

#Filter image with 1D gaussian
filtered_img = p1.apply_filter(img, gaussian1D, 0, 0)
# exam = cv2.GaussianBlur(img, (filter_size, 1), sigma)
# compare = np.concatenate((img, filtered_img, exam), axis=1)
# p1.display_img(compare)
p1.display_img(filtered_img)

#Generate 2D gaussian filter
gaussian2D = p1.generate_gaussian(sigma, filter_size, filter_size)
# print(gaussian2D, sum(gaussian2D))

#Filter image with 2D gaussian
filtered_img = p1.apply_filter(img, gaussian2D, 0, 0)
# exam = cv2.GaussianBlur(img, (filter_size, filter_size), sigma)
# compare = np.concatenate((img, filtered_img, exam), axis=1)
# p1.display_img(compare)
p1.display_img(filtered_img)

#Noise removal with median filter
# filtered_img = p1.median_filtering(img, filter_size, filter_size)
# median = cv2.medianBlur(img, filter_size)
# compare = np.concatenate((img, filtered_img, median), axis=1)
# p1.display_img(compare)
# p1.display_img(filtered_img)

#Histogram Equalization
# filtered_img = p1.hist_eq(img)
# median = cv2.calcHist(img, [0, 1, 2], None, [256], [0, 256])
# compare = np.concatenate((img, filtered_img, median), axis=1)
# p1.display_img(compare)
# p1.display_img(filtered_img)

#Rotate Image
# transformed_img = p1.rotate(img, theta)
# p1.display_img(transformed_img)

#Edge Detection
filtered_img = p1.edge_detection(img)
p1.display_img(filtered_img)
