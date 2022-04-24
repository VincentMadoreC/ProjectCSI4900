# Help from https://stackoverflow.com/questions/22656698/perspective-correction-in-opencv-using-python

import cv2
import math
import numpy as np

# Note
# This is a VERY SPECIFIC image. The bounding boxed have been HARDCODED
# to show that the code works when the bounding boxes are properly identified, which is the part that is yet to be completed.
# This will ONLY work with this image AND this scaling
def correct(img_path, debug_mode=False):

    print("Starting correction{}...".format(" in debug mode " if debug_mode else ""))
    SCALING = 0.5
    # img = cv2.imread('images/mycar_cropped.jpg')
    img = cv2.imread(img_path)
    img = cv2.resize(img, (0, 0), fx=SCALING, fy=SCALING)

    if debug_mode:
        cv2.imshow("Original", img)

    # Hardcoded coordinates of the corners
    pts1 = np.float32([[430,90],[735,224],[693,477],[410,300]])

    ratio=1.6
    img_height = math.sqrt((pts1[2][0]-pts1[1][0])*(pts1[2][0]-pts1[1][0])+(pts1[2][1]-pts1[1][1])*(pts1[2][1]-pts1[1][1]))
    img_width = ratio*img_height
    pts2 = np.float32([[pts1[0][0],pts1[0][1]], [pts1[0][0]+img_width, pts1[0][1]], [pts1[0][0]+img_width, pts1[0][1]+img_height], [pts1[0][0], pts1[0][1]+img_height]])

    M = cv2.getPerspectiveTransform(pts1,pts2)

    offsetSize = 500
    transformed = np.zeros((int(img_width+offsetSize), int(img_height+offsetSize)), dtype=np.uint8)
    dst = cv2.warpPerspective(img, M, transformed.shape)

    if debug_mode:
        cv2.imshow("Corrected",dst)

    cv2.line(img,(430,90),(735,224),(0,255,0), 2)
    cv2.line(img,(735,224),(693,477),(0,255,0), 2)
    cv2.line(img,(693,477),(410,300),(0,255,0), 2)
    cv2.line(img,(410,300),(430,90),(0,255,0), 2)

    if debug_mode:
        cv2.imshow("Bounding box",img)

    print("Correction done.{}".format(" Press any key to continue..." if debug_mode else ""))
    cv2.waitKey(0)