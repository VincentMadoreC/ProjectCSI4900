import cv2
import numpy as np
from numpy.core.fromnumeric import shape
import filters as flt
import time
import os
import sys


# Settings
IMAGE_FOLDER = "./images"
INTENSITY_THRESHOLD = 129
SCALING = 1


def find_limits(image):
    '''
    Find the limits of the specified image.
    Return the coordinates of a rectangle that encompasses the image (top-left corner, bottom-right corner)
    in the form ((first_col, first_row), (last_col, last_row))
    '''
    # Get all non-empty rows
    non_empty_rows = []
    first_row = 0
    last_row = 0
    found_first_row = False
    for row_id, row in enumerate(image):
        for pixel in row:
            if pixel != 0:
                if not found_first_row:
                    first_row = row_id
                    found_first_row = True
                last_row = row_id
                non_empty_rows.append(row)
                break

    # Get the first and last non-empty columns
    first_col = 0
    last_col = 0
    found_first_col = False
    for col_id, col in enumerate(np.transpose(non_empty_rows)):
        for pixel in col:
            if pixel != 0:
                if not found_first_col:
                    first_col = col_id
                    found_first_col = True
                last_col = col_id
                break
    return ((first_col, first_row), (last_col, last_row))


def isolate_image(image):
    '''
    Trim the image to only leave the delimited part
    '''
    limits = find_limits(image)
    return np.array(image)[
        limits[0][1]:limits[1][1], limits[0][0]:limits[1][0]]


def compare_images(image1, image2):
    '''
    Inspired by this https://www.pyimagesearch.com/2014/09/15/python-compare-two-images/
    Use the Mean Squared Error (MSE) to compare the two digits. 
    The two images must be the same size, so we match their height while keeping the aspect ratio,
    then we stretch the most narrow one to match their width without changing the height. 
    Both images need to have the same dimensions to be compared.
        It is okay to stretch the image without keeping the aspect ratio, 
        because two equal digits will have roughly the same shape, so they won’t be stretched too much, 
        while different digits will be stretched more, giving a worse (higher) MSE value. 
        This makes it less likely to get a match where we don’t want to.
    '''

    # Isolate the number part of each image
    img_isolated1 = isolate_image(np.array(image1))
    img_isolated2 = isolate_image(np.array(image2))

    # Resize the smallest one to match the dimensions of the biggest one
    max_height = max(img_isolated1.shape[0], img_isolated2.shape[0])
    # Adjust the height of the smallest image to match the height of the biggest, while keeping proportions
    if (img_isolated1.shape[0] < img_isolated2.shape[0]):
        ratio = max_height/img_isolated1.shape[0]
        img_isolated1 = cv2.resize(img_isolated1, (0, 0), fx=ratio, fy=ratio)
    else:
        ratio = max_height/img_isolated2.shape[0]
        img_isolated2 = cv2.resize(img_isolated2, (0, 0), fx=ratio, fy=ratio)

    # Adjust the width of the most narrow image to match the width of the widest, without changing the height
    max_width = max(img_isolated1.shape[1], img_isolated2.shape[1])
    if (img_isolated1.shape[1] < img_isolated2.shape[1]):
        ratio = max_width/img_isolated1.shape[1]
        img_isolated1 = cv2.resize(img_isolated1, (0, 0), fx=ratio, fy=1)
    else:
        ratio = max_width/img_isolated2.shape[1]
        img_isolated2 = cv2.resize(img_isolated2, (0, 0), fx=ratio, fy=1)

    # Calculate the Mean Squared Error
    # https://www.pyimagesearch.com/2014/09/15/python-compare-two-images/
    img_isolated1 = img_isolated1/255
    img_isolated2 = img_isolated2/255
    mse = np.sum((img_isolated1 - img_isolated2) ** 2)
    mse /= float(img_isolated1.shape[0] * img_isolated1.shape[1])

    return mse


def compare_against_dataset(image):
    '''
    Compare the specified image against the dataset.
    Store the average MSE for each number.
    Check which number has the best (lowest) MSE on average when compared to the image.
    '''
    mse_list = []
    path_dataset = IMAGE_FOLDER + "/standard"
    for folder in os.listdir(path_dataset):
        # Skip if it's just an unclassified image
        if folder.endswith(".png") or folder.endswith(".jpg"):
            continue
        path_folder = path_dataset + "/" + folder
        count = 0
        sum_mse = 0
        for example in os.listdir(path_folder):
            path_example = path_folder + "/" + example
            standard = cv2.imread(path_example, cv2.IMREAD_GRAYSCALE)
            sum_mse += compare_images(standard, image)
            count += 1
        mse_list.append(sum_mse/count)

    best_mse = 1
    best_mse_index = 0
    # Find the best (lowest) MSE and the corresponding folder
    for i, mse in enumerate(mse_list):
        if mse < best_mse:
            best_mse = mse
            best_mse_index = i
    return best_mse_index


if __name__ == "__main__":
    img_path = "./images/270696.png"
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
    img_original = cv2.imread(img_path)

    # Downscale the image to reduce processing time
    img_downscale = cv2.resize(img_original, (0, 0), fx=SCALING, fy=SCALING)
    img_grayscale = cv2.cvtColor(img_downscale, cv2.COLOR_RGB2GRAY)

    # Filtering
    ret, img_filtered = cv2.threshold(img_grayscale, INTENSITY_THRESHOLD, 255, cv2.THRESH_BINARY)
    img_filtered = flt.erosion(1, img_filtered)
    img_filtered = flt.dilation(1, img_filtered)

    # Draw a rectangle around the numbers
    limits_img_filtered = find_limits(img_filtered)
    img_drawn = cv2.cvtColor(img_filtered, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(img_drawn, (limits_img_filtered[0][0], limits_img_filtered[0][1]),
                  (limits_img_filtered[1][0], limits_img_filtered[1][1]), (0, 255, 0), 1)
    cv2.imshow("Image limits", img_drawn)

    # Isolate the number part
    img_isolated = isolate_image(img_filtered)

    # Draw rectangles around each number for debugging
    numbers_shape = img_isolated.shape

    # Separate each number
    padding = 3
    img_number_1 = img_isolated[:, 0:int(padding + 1 * numbers_shape[1] / 4)]
    img_number_2 = img_isolated[:, int(
        padding + 1 * numbers_shape[1] / 4):int(padding + 2 * numbers_shape[1] / 4)]
    img_number_3 = img_isolated[:, int(
        padding + 2 * numbers_shape[1] / 4):int(padding + 3 * numbers_shape[1] / 4)]
    img_number_4 = img_isolated[:, int(
        padding + 3 * numbers_shape[1] / 4):int(4 * numbers_shape[1] / 4)]

    # Identify each number
    number_1 = compare_against_dataset(img_number_1)
    number_2 = compare_against_dataset(img_number_2)
    number_3 = compare_against_dataset(img_number_3)
    number_4 = compare_against_dataset(img_number_4)

    # Print the result
    numbers_found = "{}{}{}{}".format(number_1, number_2, number_3, number_4)
    file_number = img_path.split("/").pop().replace(".png", "")
    print("<{}, {}>".format(file_number, numbers_found))

    cv2.waitKey(0)
