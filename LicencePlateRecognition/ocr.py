
import cv2
import numpy as np
from numpy.core.fromnumeric import shape
import os
import math


# Settings
IMAGE_FOLDER = "./images"

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


def compare_against_dataset(img, debug_mode):
    '''
    Compare the specified image against the dataset.
    Store the average MSE for each number.
    Check which number has the best (lowest) MSE on average when compared to the image.
    '''
    # TODO maybe lower threshold more
    CONFIDENCE_THRESHOLD = 0.2 # all matches with an MSE <= CONFIDENCE_THRESHOLD will be considered as possibilities
    possibilities = []
    mse_list = []
    path_dataset = IMAGE_FOLDER + "/standard"
    for folder in os.listdir(path_dataset):
        # skip if it's just an unclassified image
        if folder.endswith(".png") or folder.endswith(".jpg"):
            continue
        path_folder = path_dataset + "/" + folder
        count = 0
        sum_mse = 0
        for example in os.listdir(path_folder):
            path_example = path_folder + "/" + example
            standard = cv2.imread(path_example, cv2.IMREAD_GRAYSCALE)
            sum_mse += compare_images(standard, img)
            count += 1
        mse_list.append((sum_mse/count, folder))
        if debug_mode:
            print("Average MSE for {}: {}".format(folder, sum_mse/count))
    best_mse = 1

    # Find the best (lowest) MSE and the corresponding folder
    for i, (mse, folder) in enumerate(mse_list):
        if mse < best_mse:
            best_mse = mse
            best_mse_index = folder
        if mse <= CONFIDENCE_THRESHOLD:
            possibilities.append((mse, folder))
    if debug_mode:
        print("Best: {} ({})".format(best_mse_index, best_mse))
        print("Possibilities: {}".format(possibilities))

    return sorted(possibilities, key=lambda x: x[0])


def ocr(img, debug_mode=False):
    print("Starting optical character recognition...")
    # img = cv2.imread(img_path)

    # Scale the image up if too small, or down if too big. The image will have a fixed height, the width will be proportional.
    FIXED_HEIGHT = 500
    width = math.floor(FIXED_HEIGHT * img.shape[1] / img.shape[0])
    img = cv2.resize(img, (width, FIXED_HEIGHT))
    if debug_mode:
        cv2.imshow("Original", img)
    
    # Find the contours of the characters
    # https://stackoverflow.com/questions/21104664/extract-all-bounding-boxes-using-opencv-python
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if debug_mode:
        cv2.imshow("Grayscale", img_gray)

    img_filtered = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    if debug_mode:
        cv2.imshow("Threshold", img_filtered)

    img_filtered = cv2.erode(img_filtered, None, iterations=3)
    if debug_mode:
        cv2.imshow("Eroded", img_filtered)

    img_filtered = cv2.dilate(img_filtered, None, iterations=3)
    if debug_mode:
        cv2.imshow("Dilated", img_filtered)

    # Find contours, obtain bounding box, extract and save region of interest (ROI)
    ROI_number = 0
    cnts = cv2.findContours(img_filtered, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    img_height, img_width, _ = img.shape
    min_height_ratio = 0.2 # % of the image's height a contour must be to be considered relevant
    max_height_ratio = 0.5
    min_width_ratio = 0.03 # % of the image's width a contour must be to be considered relevant 
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
            cv2.rectangle(img, (x, y), (x + w, y + h), (0,255,0), 2)
            ROI = img_filtered[y:y+h, x:x+w]
            characters.append((ROI, x)) # store the horizontal position to sort them later
            cv2.imwrite('ROI_{}.png'.format(ROI_number), ROI)
            ROI_number += 1
    if debug_mode:
        cv2.imshow('Bounding boxes', img)

    # Sort the bounding boxes from left to right
    sorted_characters = sorted(characters, key=lambda x: x[1])

    final_string = ""
    possibilities = []
    for (char, _) in sorted_characters:
        possibilities.append(compare_against_dataset(char, debug_mode))
    
    if debug_mode:
        for char_possibilities in possibilities:
            print(char_possibilities)
    
    if len(possibilities) > 0:
        NUMBER_OF_RESULTS = 10
        stop_flag = False
        for i in range(NUMBER_OF_RESULTS):
            final_string = ''
            sum_confidence = 0
            num_of_chars = 0
            difference = 1
            most_similar_char = (0, 'c') # (mse, character) pair
            for char_possibilities in possibilities:
                # Add the character of the first possibility (those are already sorted)
                # char_possibilities is in the form [(mse, 'c'), (mse, 'c'), (mse, 'c')]
                final_string = final_string + char_possibilities[0][1]
                sum_confidence += (1 - char_possibilities[0][0])
                num_of_chars += 1
                if len(char_possibilities) > 1:
                    new_difference = char_possibilities[1][0] - char_possibilities[0][0]
                    if new_difference < difference:
                        most_similar_char = char_possibilities[0]
                        difference = new_difference
            avg_confidence = sum_confidence * 100 / num_of_chars
            print("{} (confidence: {} %)".format(final_string, format(avg_confidence, ".3f")))

            # Stop the loop if there are no more possible permutations (i.e. all characters have only 1 possibility left)
            for char_possibilities in possibilities:
                if len(char_possibilities) > 1:
                    stop_flag = False
                    break
                else:
                    stop_flag = True

            # remove the (mse, char) pair that is the most similar to the next one
            for char_possibilities in possibilities:
                if char_possibilities[0] == most_similar_char:
                    char_possibilities.pop(0)
                    break

            if stop_flag:
                break

    print("OCR done.{}".format(" Press any key to continue..." if debug_mode else ""))
    cv2.waitKey(0)
