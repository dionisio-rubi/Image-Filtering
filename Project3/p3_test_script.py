import cv2
import PIL
import matplotlib
import skimage
import numpy as np
import math
import sklearn
import project3 as p3

#Extract Vocab
vocab = p3.generate_vocabulary("train_data.txt")

#Train Object Classifier
classifier = p3.train_classifier("train_data.txt", vocab)

#Test Object Classifier
test_img = p3.load_img("test_img.jpg")
out = p3.classify_image(classifier, test_img, vocab)
print(out)

#Segment an Image
img = p3.load_img("test_img.jpg")

# img_thresh = p3.threshold_image(img, 150, 200)
# p3.display_img(img_thresh)
#
# img_grow = p3.grow_regions(img)
# p3.display_img(img_grow)
#
# img_split = p3.split_regions(img)
# p3.display_img(img_split)
#
# img_merge = p3.merge_regions(img)
# p3.display_img(img_merge)

im1, im2, im3 = p3.segment_image(img)
p3.display_img(im1)
p3.display_img(im2)
p3.display_img(im3)

img = p3.load_img("test_img.jpg")
img_out = p3.kmeans_segment(img)
p3.display_img(img_out)

