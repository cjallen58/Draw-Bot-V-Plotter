from time import altzone
import cv2
import numpy as np
from matplotlib import pyplot as plt
import utils
"""
change this to use a command line argument
for picture selection and paper size
also add a way to track performance and sorting time
"""
img = cv2.imread('Images/statue.jpg', 1)
# at some point check image ratio to make sure it actually makes sense to use
imgresize = utils.rez(img)

imgGray = cv2.cvtColor(imgresize, cv2.COLOR_BGR2GRAY)

imgDither = utils.dithering(imgGray, .2)

ID = utils.pointConversion(imgDither)
final = utils.order(ID)

final = np.hsplit(final, 2)
Points = utils.scale(final, 2)

"""
plt.subplot(1, 2, 1)
plt.plot(final[0], final[1], '.')
plt.title('Dither Image')


plt.subplot(1, 2, 2)
plt.plot(Points[0], Points[1], '.')
plt.title('Scaled Image')

cv2.imshow('Original', img)
cv2.imshow('resize', imgresize)
cv2.imshow('Gray', imgGray)
cv2.imshow('Dith', imgDither)
"""
plt.plot(Points[0], Points[1])
plt.show()
cv2.waitKey(0)