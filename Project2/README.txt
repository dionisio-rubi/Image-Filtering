PS: a COLORED IMAGE CAN BE INPUT
beware: for some functions, since I couldn't finish implementing color, it will just output a black and white image, otherwise you get a colored image :)

Load and Display Images:
    I am adopting these functions from project 1:
        I tested how to display and load images using the open cv and PIL packages (can be seen in my comments). Both are simple to use and I noticed that I can use less lines of code when using PIL, however, I opted to go with open cv because cv2 is more common from what I've heard and I wanted to understand it better.

        As I kept progressing through the assignment I also noticed that using open cv was a lot more convenient for implementing the other functions in the assignment and also getting to display the image after doing some sort of filtering or transformation.

Moravec Detector:
    First I started off by creating a list that would store all the keypoints, set a certain threshold, and got the dimensions of the image. I then iterated over each pixel in the image, excluding the border pixels, to calculate corner response function (SW) at each pixel location. The SW value is computed by summing the squared differences in intensity between the central pixel and its neighbors in all eight directions within a 3x3 window, denoted in the second set of nested for-loops found inside the first nested for-loops. After calculating SW values for all directions, the minimum SW value is determined. If this minimum value exceeds the predefined threshold, the pixel coordinates are added to the list of keypoints.The function returns the list of detected keypoints at the end.

LBP Features:


HOG Features:

Feature Matching:
