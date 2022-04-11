import cv2
import numpy as np

# Erosion
# Copied from my lab 4
def erosion(size, image, iter=1, shape=cv2.MORPH_ELLIPSE):
    ''' Erode the image to remove noise.
    Got help from https://docs.opencv.org/3.4.15/db/df6/tutorial_erosion_dilatation.html
    '''
    erosion_shape = shape
    kernel = cv2.getStructuringElement(erosion_shape, (2 * size + 1, 2 * size + 1),
                                       (size, size))
    return cv2.erode(image, kernel, iterations=iter)

# Dilation
# Copied from my lab 4
def dilation(size, image, iter=1, shape=cv2.MORPH_ELLIPSE):
    ''' Perform dilation to reconstruct the eroded parts of the image of interest.
    Got help from https://docs.opencv.org/3.4.15/db/df6/tutorial_erosion_dilatation.html
    '''
    dilation_shape = shape
    kernel = cv2.getStructuringElement(dilation_shape, (2 * size + 1, 2 * size + 1),
                                       (size, size))
    return cv2.dilate(image, kernel, iterations=iter)

# Thresholding function.
# Used to display a white pixel if the difference in intensity is greater than or equal to the threshold,
# else display a black pixel.
def thresholding(v, t):
    if v >= t:
        return [255]
    return [0]

# Applies the thresholding function while performing the necessary modifications
def apply_thresholding(img, threshold):
    img_filtered = img.reshape((-1, 1))
    img_filtered = np.array(list(map(lambda a: thresholding(a, threshold), img_filtered))) # Intensity thresholding
    img_filtered = img_filtered.reshape(img.shape)
    img_filtered = np.uint8(img_filtered)
    return img_filtered