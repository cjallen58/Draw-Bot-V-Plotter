import cv2
import numpy as np
from matplotlib import pyplot as plt
import utils

img = cv2.imread('lake.jpg', 1)
print(f"{img.shape[0]} {img.shape[1]}")
imgrez = utils.rez(img)
imgGray = cv2.cvtColor(imgrez,cv2.COLOR_BGR2GRAY)
cv2.imshow('resize', imgrez)

print(f"{imgrez.shape[0]} {imgrez.shape[1]}")
#print(imgGray[:3,:3])
cv2.waitKey(0)
"""
arr = imgGray[:3,:3]
for i in range(3):
    for j in range(3):
        if arr[i, j] > 130:
            arr[i, j] = 255
        else:
            arr[i, j] = 0

print(arr)

xy = utils.pointConversion(arr, 'C')

print(xy[0])
print(xy[1])
"""

