PS: a COLORED IMAGE CAN BE INPUT
beware: for some functions, since I couldn't finish implementing color, it will just output a black and white image, otherwise you get a colored image :)

Load and Display Images:
    I tested how to display and load images using the open cv and PIL packages (can be seen in my comments). Both are simple to use and I noticed that I can use less lines of code when using PIL, however, I opted to go with open cv because cv2 is more common from what I've heard and I wanted to understand it better.

    As I kept progressing through the assignment I also noticed that using open cv was a lot more convenient for implementing the other functions in the assignment and also getting to display the image after doing some sort of filtering or transformation.

1D and 2D Filtering:
    generate_gaussian():
        To generate a 1D array, first I made sure that only either filter_h or filter_w was 1. I assigned the length of the array to be whichever was not 1. I then used the formula e^(-n/2*sigma^2)  initialize an array with values calculated based on the Gaussian function. Then, for every element in the array, I divided each value by the sum of all the values in the array to normalize it and returned it.

        To generate the 2D array I used a similar implementation as the 1D array, but I used the formula e^(-((x^2 + y^2)/2*sigma^2)) to initialize the array with values calculated based on the Gaussian function. Then, for every element in the array, I divided each value by the sum of all the values in the array to normalize it and returned it.

Histogram Equalization:
    Can take a colored or black and white image, regardless, it returns a black and white image.

Image Transformation:
    To get a greyscale image to rotate I originally converted the image to greyscale using the cv2.cvtColor() function. Then I got the dimensions of the image using image.shape, I found the center of the image and then created a new image with a matrix of zeros with the same dimensions as the original image. I then used a nested for loop to iterate through the new image and for every pixel in the new image, I used the rotation formula given in lecture 4 slide 90 to find the new x and y coordinates of the pixel in the original image. I then checked if the new x and y coordinates were within the bounds of the original image and if they were, I assigned the pixel in the new image to the pixel in the original image. I then returned the new image.

    I figured that rotating a colored image was actually not that difficult, so I commented out the line that converted the image to greyscale and then just added the channels variable to the dimensions of the image to get the new image to have the same number of channels as the original image. I then used the same rotation formula as the greyscale image to find the new x and y coordinates of the pixel in the original image. I then checked if the new x and y coordinates were within the bounds of the original image and if they were, I assigned the pixel in the new image to the pixel in the original image. I then returned the new image. (there was almost no difference in the code for rotating a colored image and a greyscale image)

    The function that I submitted takes a colored or not colored image and rotates it by a given angle. It then returns the rotated image.

Edge Detection:
    Step 1: Smoothing
        I for this step, I decided to use a 2d Gaussian filter to smooth the image. I utilized the functions previously implemented to smooth my image first. I decided to use this because, from the slides, using a 2d Gaussian filter is a good way to smooth an image and since I already had implemented how to do this, I decided to use it for this step.

    Step 2: Enhancement
        I used the gradient magnitude way to enhance my image. I used the Sobel filter to get the gradient in the x and y direction and then applied the filter to my smoothed image using hte apply_filter function that I implemented earlier. Based oin the slides, this technique for enhancing is a good way to help detect edges in both the x and y directions.

    Step 3: Thresholding
        I used the thresholding technique to get the edges of the image. I used the threshold value of 100 just to test it out. I used np.where to check if the pixel value was greater than the threshold value and if it was, I assigned the pixel value to 255, otherwise I assigned the pixel value to 0, which returns an image in black and white,

    Step 4: Localization
        For this step, I decided to use the canny edge detection technique. More specifically I opted for using the hysteresis thresholding technique to find the stronger pixels and connect the lines with weaker pixels to crete a more continuous line. I used two threshold values, a low and a high, to locate my stronger and weaker pixels. I figured that this technique would be simpler to implement and still produce similar results as other techniques.