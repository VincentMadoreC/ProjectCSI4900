import cv2
import numpy as np
from numpy.core.fromnumeric import shape
import filters as flt
import time
import os

image_folder = "./images"

# Settings
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

# Separate each number
for image in os.listdir(image_folder):
    if not image.endswith(".png"):
        continue
    path_image = image_folder + "/" + image

    img_original = cv2.imread(path_image)
    img_grayscale = cv2.cvtColor(img_original, cv2.COLOR_RGB2GRAY)
    img_filtered = flt.apply_thresholding(img_grayscale, INTENSITY_THRESHOLD)

    img_isolated = isolate_image(img_filtered)

    numbers_shape = img_isolated.shape
    img_number_1 = img_isolated[:, 0:int(1 * numbers_shape[1] / 4)]
    img_number_2 = img_isolated[:, int(
        1 * numbers_shape[1] / 4):int(2 * numbers_shape[1] / 4)]
    img_number_3 = img_isolated[:, int(
        2 * numbers_shape[1] / 4):int(3 * numbers_shape[1] / 4)]
    img_number_4 = img_isolated[:, int(
        3 * numbers_shape[1] / 4):int(4 * numbers_shape[1] / 4)]

    cv2.imwrite(image_folder + "/standard/" + image.replace(".png", "") + "_1.png", img_number_1)
    cv2.imwrite(image_folder + "/standard/" + image.replace(".png", "") + "_2.png", img_number_2)
    cv2.imwrite(image_folder + "/standard/" + image.replace(".png", "") + "_3.png", img_number_3)
    cv2.imwrite(image_folder + "/standard/" + image.replace(".png", "") + "_4.png", img_number_4)
    print("DONE - " + image)














# cv2.imshow("Number 1", img_number_1)
# cv2.imshow("Number 2", img_number_2)
# cv2.imshow("Number 3", img_number_3)
# cv2.imshow("Number 4", img_number_4)

# # Used to generate the standards for comparison
# cv2.imwrite("./images/standard/000001.jpg", img_number_1)
# cv2.imwrite("./images/standard/000002.jpg", img_number_2)
# cv2.imwrite("./images/standard/000003.jpg", img_number_3)
# cv2.imwrite("./images/standard/000004.jpg", img_number_4)


def compare_images(image1, image2):
    '''
    Use the Mean Squared Error (MSE) to compare the two digits.
    The two images must be the same size, so we match their height while keeping the aspect ratio,
    then we stretch the most narrow one to match their width without changing the height. 
    '''

    # Isolate the number part of each image
    img_isolated1 = isolate_image(np.array(image1))
    img_isolated2 = isolate_image(np.array(image2))

    # cv2.imshow("isol1", img_isolated1)
    # cv2.imshow("isol2", img_isolated2)

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

    # print(img_isolated1.shape)
    # print(img_isolated2.shape)

    # array = np.array([255,0,128])
    # new_array = array/255
    # print(new_array)

    # print(img_isolated1)

    img_isolated1 = img_isolated1/255
    img_isolated2 = img_isolated2/255
    # picture1_norm = picture1/np.sqrt(np.sum(picture1**2))
    # picture2_norm = picture2/np.sqrt(np.sum(picture2**2))
    # cv2.imshow("pic1", picture1)
    # cv2.imshow("pic2", picture2)
    # cv2.imshow("pic1norm", picture1_norm)
    # cv2.imshow("pic2norm", picture2_norm)

    # print(np.sum(picture1_norm*picture2_norm))

    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!Pad with zeros if necessary!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    # Calculate the Mean Squared Error
    # https://www.pyimagesearch.com/2014/09/15/python-compare-two-images/
    mse = np.sum((img_isolated1 - img_isolated2) ** 2)
    mse /= float(img_isolated1.shape[0] * img_isolated1.shape[1])

    # img_isolated1 = np.resize(img_isolated1, (max_height, max_width))

    # cv2.imshow("isol1", img_isolated1)
    # cv2.imshow("isol2", img_isolated2)

    # print("MSE: ", mse)
    return mse


# compare_images(img_number_1,cv2.imread("images/standard/1/000002.jpg", cv2.IMREAD_GRAYSCALE))
# start_time = time.time()
# compare_images(img_number_1, cv2.imread(
#     "images/standard/0/000003.jpg", cv2.IMREAD_GRAYSCALE))
# end_time = time.time()
# print("Time to compare images: ", str(end_time - start_time))


def compare_against_dataset(image):
    '''
    Compare the specified image against the dataset.
    Store the average MSE for each number.
    Check which number has the best (lowest) MSE on average when compared to the image.
    '''
    mse_list = []
    path_dataset = image_folder + "/standard"
    for folder in os.listdir(path_dataset):
        path_folder = path_dataset + "/" + folder
        if len(os.listdir(path_folder)) == 0:
            mse_list.append(1)
            continue
        count = 0
        sum_mse = 0
        for example in os.listdir(path_folder):
            path_example = path_folder + "/" + example
            standard = cv2.imread(path_example, cv2.IMREAD_GRAYSCALE)
            sum_mse += compare_images(standard, image)
            count += 1
        mse_list.append(sum_mse/count)
    print(mse_list)

# compare_against_dataset(img_number_1)
# compare_against_dataset(img_number_2)
# compare_against_dataset(img_number_3)
# compare_against_dataset(img_number_4)

# Comparison
# see https://stackoverflow.com/questions/11816203/compare-images-in-python
# or https://stackoverflow.com/questions/11541154/checking-images-for-similarity-with-opencv
# https://www.pyimagesearch.com/2014/09/15/python-compare-two-images/

# no need to be a perfect match, only has to be more similar than the other numbers
# standard = cv2.imread("images/standard/1/000003.jpg", cv2.IMREAD_GRAYSCALE)
# print(standard.shape)
# print(img_number_2.shape)
# # zero_array = np.zeros(img_number_2.shape)
# # print(zero_array.shape)
# # added_arrays = np.add(zero_array,standard)
# # np.pad()
# resized = np.array(img_downscale)
# resized.resize(img_number_2.shape)
# print(resized)
# print(resized.shape)
# # for row in standard:
# #     print(row)
# # print("RESIZED!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
# # for row in resized:
# #     print(row)
# # cv2.imshow("img_number_2",img_number_2)
# # cv2.imshow("standard",standard)
# # cv2.imshow("resized",resized)
# mse = np.square(np.subtract(img_number_2, resized)).mean()
# print(mse)
# cv2.imshow("size1", standard)
# standard2 = cv2.resize(standard, (0, 0), fx=1.33, fy=1.33)
# cv2.imshow("size2", standard2)
# To ensure the 2 extracted numbers are the same size,
# 1. find the biggest of the 2 images
# 2. initialize an array of zero of the same size
# 3. add the smaller one to it. (hopefully it will add corresponding pixels)
cv2.waitKey(0)
