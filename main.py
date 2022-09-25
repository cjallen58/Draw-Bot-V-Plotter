from time import altzone
import cv2
import numpy as np
from matplotlib import pyplot as plt
import utils

img = cv2.imread('statue.jpg', 1)
# at some point check image ratio to make sure it actually makes sense to use
imgresize = utils.rez(img)

imgGray = cv2.cvtColor(imgresize, cv2.COLOR_BGR2GRAY)

imgDither = utils.dithering(imgGray, .2)

ID = utils.pointConversion(imgDither)
Points = utils.scale(ID, 2)

plt.subplot(1, 2, 1)
plt.plot(ID[0], ID[1], '.')
plt.title('Dither Image')

plt.subplot(1, 2, 2)
plt.plot(Points[0], Points[1], '.')
plt.title('Scaled Image')

cv2.imshow('Original', img)
cv2.imshow('resize', imgresize)
cv2.imshow('Gray', imgGray)
cv2.imshow('Dith', imgDither)

plt.show()
cv2.waitKey(0)