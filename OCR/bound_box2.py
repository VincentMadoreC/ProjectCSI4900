
# Make the code runnable from command line

# Still need to get a more robust way to isolate the numbers with having to change the settings for each image (add 1 erosion/dilation cycle)
# also, when finding the "first_row" and "first column", might want to check surrounding pixels to make sure it's not just a stray noise pixel

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
SCALING = 0.3


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

    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!Pad with zeros if necessary!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # cv2.imshow("isol1_resized", img_isolated1)
    # cv2.imshow("isol2_resized", img_isolated2)
    # Calculate the Mean Squared Error
    # https://www.pyimagesearch.com/2014/09/15/python-compare-two-images/
    img_isolated1 = img_isolated1/255
    img_isolated2 = img_isolated2/255
    mse = np.sum((img_isolated1 - img_isolated2) ** 2)
    mse /= float(img_isolated1.shape[0] * img_isolated1.shape[1])

    # cv2.imshow("isol1", img_isolated1)
    # cv2.imshow("isol2", img_isolated2)
    # print("MSE: ", mse)
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
        # skip if it's just an unclassified image
        if folder.endswith(".png") or folder.endswith(".jpg"):
            continue
        path_folder = path_dataset + "/" + folder
        # if len(os.listdir(path_folder)) == 0:
        #     mse_list.append(1)
        #     continue
        count = 0
        sum_mse = 0
        for example in os.listdir(path_folder):
            path_example = path_folder + "/" + example
            standard = cv2.imread(path_example, cv2.IMREAD_GRAYSCALE)
            sum_mse += compare_images(standard, image)
            count += 1
        mse_list.append((sum_mse/count, folder))
        print("Average MSE for {}: {}".format(folder, sum_mse/count))
    best_mse = 1
    # best_mse_index = 0
    # Find the best (lowest) MSE and the corresponding folder
    for i, (mse, folder) in enumerate(mse_list):
        if mse < best_mse:
            best_mse = mse
            best_mse_index = folder
    print("Best: {} ({})".format(best_mse_index, best_mse))
    return best_mse_index


if __name__ == "__main__":
    # img_path = "./images/402408.png"
    img_path = "./images/cfva648.jpg"
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
    img = cv2.imread(img_path)
    img = cv2.resize(img, (0, 0), fx=SCALING, fy=SCALING)
    # img_original = img.copy()
    cv2.imshow("Original", img)
    
    # Find the contours of the characters
    # https://stackoverflow.com/questions/21104664/extract-all-bounding-boxes-using-opencv-python
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_original = img_gray.copy()
    cv2.imshow("Grayscale", img_gray)
    cv2.imwrite('Grayscale', img_gray)

    img_filtered = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    cv2.imshow("Threshold", img_filtered)
    cv2.imwrite('Threshold', img_filtered)

    img_filtered = cv2.erode(img_filtered, None, iterations=3)
    cv2.imshow("Eroded", img_filtered)
    cv2.imwrite('Eroded', img_filtered)

    img_filtered = cv2.dilate(img_filtered, None, iterations=3)
    cv2.imshow("Dilated", img_filtered)
    cv2.imwrite('Dilated', img_filtered)

    # Find contours, obtain bounding box, extract and save region of interest (ROI)
    ROI_number = 0
    cnts = cv2.findContours(img_filtered, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    img_height, img_width, _ = img.shape
    min_height_ratio = 0.2 # % of the image's height a contour must be to be considered relevant
    max_height_ratio = 0.5
    min_width_ratio = 0.05 # % of the image's width a contour must be to be considered relevant 
    max_width_ratio = 0.2

    # Isolate only relevant characters based on size

    characters = []
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        # Valid contour rules
        rule_min_width = (w >= min_width_ratio*img_width)
        rule_max_width = (w <= max_width_ratio*img_width)
        rule_min_height = (h >= min_height_ratio*img_height)
        rule_max_height = (h <= max_height_ratio*img_height)
        rule_ratio = (h > w)
        # Check if the contour is likely to be a character based on the dimensions
        if (rule_min_width and rule_max_width and rule_min_height and rule_max_height and rule_ratio):

        # if (h >= min_height_ratio*img_height):
            cv2.rectangle(img, (x, y), (x + w, y + h), (0,255,0), 2)
            ROI = img_filtered[y:y+h, x:x+w]
            characters.append((ROI, x)) # store the horizontal position to sort them later
            cv2.imwrite('ROI_{}.png'.format(ROI_number), ROI)
            ROI_number += 1
    cv2.imshow('Bounding boxes', img)

    # Sort the bounding boxes from left to right
    sorted_characters = sorted(characters, key=lambda x: x[1])

    # print the final result with the highest confidence.
    # TODO make a list of the most likely, rather than only one. 
    final_string = ""
    for (char, x) in sorted_characters:
        c = compare_against_dataset(char)
        final_string = final_string + c
    print(final_string)
    # char_0 = compare_against_dataset(characters[0])
    # cv2.imshow('character', characters[0])
    # print("CHAR0")
    # print(char_0)




    # contours, hierarchy = cv2.findContours(img_filtered,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[-2:]
    # idx =0 
    # for cnt in contours:
    #     cv2.drawContours(img_original, cnt, -1, (0, 255, 0), 3)
    #     # idx += 1
    #     # x,y,w,h = cv2.boundingRect(cnt)
    #     # roi=img_original[y:y+h,x:x+w]
    #     # cv2.imwrite(str(idx) + '.jpg', roi)
    #     #cv2.rectangle(im,(x,y),(x+w,y+h),(200,0,0),2)
    # cv2.imshow('img',img_original)
    # cv2.waitKey(0)    

    # Get bounding boxes



    # img_filtered = flt.apply_thresholding(img_grayscale, INTENSITY_THRESHOLD)
    # img_filtered = flt.erosion(1, img_filtered)
    # img_filtered = flt.dilation(1, img_filtered)
    

    # # Draw a rectangle around the numbers
    # limits_img_filtered = find_limits(img_filtered)
    # img_drawn = cv2.cvtColor(img_filtered, cv2.COLOR_GRAY2BGR)
    # cv2.rectangle(img_drawn, (limits_img_filtered[0][0], limits_img_filtered[0][1]),
    #               (limits_img_filtered[1][0], limits_img_filtered[1][1]), (0, 255, 0), 1)
    # cv2.imshow("Image limits", img_drawn)

    # # Isolate the number part
    # img_isolated = isolate_image(img_filtered)
    # cv2.imshow("Isolated numbers", img_isolated)

    # # Draw rectangles around each number for debugging
    # numbers_shape = img_isolated.shape
    # img_isolated_drawn = img_isolated.copy()
    # padding = 3
    # cv2.rectangle(img_isolated_drawn, (0, 0), (int(padding +
    #                                                1 * numbers_shape[1] / 4), numbers_shape[0]), (255, 255, 255), 1)
    # cv2.rectangle(img_isolated_drawn, (int(padding + 1 * numbers_shape[1] / 4), 0), (int(
    #     padding + 2 * numbers_shape[1] / 4), numbers_shape[0]), (255, 255, 255), 1)
    # cv2.rectangle(img_isolated_drawn, (int(padding + 2 * numbers_shape[1] / 4), 0), (int(
    #     padding + 3 * numbers_shape[1] / 4), numbers_shape[0]), (255, 255, 255), 1)
    # cv2.rectangle(img_isolated_drawn, (int(padding + 3 * numbers_shape[1] / 4), 0), (int(
    #     padding + 4 * numbers_shape[1] / 4), numbers_shape[0]), (255, 255, 255), 1)

    # cv2.imshow("Drawn", img_isolated_drawn)
    # # Separate each number
    # img_number_1 = img_isolated[:, 0:int(padding + 1 * numbers_shape[1] / 4)]
    # img_number_2 = img_isolated[:, int(
    #     padding + 1 * numbers_shape[1] / 4):int(padding + 2 * numbers_shape[1] / 4)]
    # img_number_3 = img_isolated[:, int(
    #     padding + 2 * numbers_shape[1] / 4):int(padding + 3 * numbers_shape[1] / 4)]
    # img_number_4 = img_isolated[:, int(
    #     padding + 3 * numbers_shape[1] / 4):int(padding + 4 * numbers_shape[1] / 4)]
    # cv2.imshow("Number 1", img_number_1)
    # cv2.imshow("Number 2", img_number_2)
    # cv2.imshow("Number 3", img_number_3)
    # cv2.imshow("Number 4", img_number_4)

    # # Identify each number
    # number_1 = compare_against_dataset(img_number_1)
    # number_2 = compare_against_dataset(img_number_2)
    # number_3 = compare_against_dataset(img_number_3)
    # number_4 = compare_against_dataset(img_number_4)

    # # Print the result
    # numbers_found = "{}{}{}{}".format(number_1, number_2, number_3, number_4)
    # file_number = img_path.split("/").pop().replace(".png", "")
    # print("<{}, {}>".format(file_number, numbers_found))

    cv2.waitKey(0)
