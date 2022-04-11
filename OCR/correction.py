# https://stackoverflow.com/questions/22656698/perspective-correction-in-opencv-using-python

import cv2
import math
import numpy as np
import matplotlib.pyplot as plt

SCALING = 0.5
img = cv2.imread('images/mycar_cropped.jpg')
img = cv2.resize(img, (0, 0), fx=SCALING, fy=SCALING)
rows,cols,ch = img.shape

cv2.imshow("Original", img)

pts1 = np.float32([[430,90],[735,224],[693,477],[410,300]])

ratio=1.6
cardH=math.sqrt((pts1[2][0]-pts1[1][0])*(pts1[2][0]-pts1[1][0])+(pts1[2][1]-pts1[1][1])*(pts1[2][1]-pts1[1][1]))
cardW=ratio*cardH
pts2 = np.float32([[pts1[0][0],pts1[0][1]], [pts1[0][0]+cardW, pts1[0][1]], [pts1[0][0]+cardW, pts1[0][1]+cardH], [pts1[0][0], pts1[0][1]+cardH]])

M = cv2.getPerspectiveTransform(pts1,pts2)

offsetSize=500
transformed = np.zeros((int(cardW+offsetSize), int(cardH+offsetSize)), dtype=np.uint8)
dst = cv2.warpPerspective(img, M, transformed.shape)

cv2.imshow("Corrected",dst)



# cv2.circle(img,(430,90), 3, (255,0,255), -1) # !!!!!!!!!!!
# cv2.circle(img,(735,224), 3, (0,0,255), -1) # !!!!!!!!!!!!!!!
# cv2.circle(img,(410,300), 3, (0,255,255), -1) # !!!!!!!!!!!
# cv2.circle(img,(693,477), 3, (255,0,0), -1) # !!!!!!!!!!!

cv2.line(img,(430,90),(735,224),(0,255,0), 2)
cv2.line(img,(735,224),(693,477),(0,255,0), 2)
cv2.line(img,(693,477),(410,300),(0,255,0), 2)
cv2.line(img,(410,300),(430,90),(0,255,0), 2)


cv2.imshow("Bounding box",img)

cv2.waitKey(0)
# plt.subplot(121),plt.imshow(img),plt.title('Input')
# plt.subplot(122),plt.imshow(dst),plt.title('Output')
# plt.show()