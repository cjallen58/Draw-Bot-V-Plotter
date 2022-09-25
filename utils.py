import cv2
import numpy as np
from matplotlib import pyplot as pltt


def dithering(img, threshold):
    # usable variables needed for performing the Dithering
    height =  img.shape[0]
    width = img.shape[1]
    if threshold >= 1 or threshold <= 0 :
        return print('Incorrect usage of dithering function')
    
    # create 2 new images one is the error image and the other is the dither image
    # error image contains new pixel values with error applied
    # dither image will be the output image
    errimg = img/255
    imgDither = np.zeros((height, width, 1), dtype = np.uint8)

    # for every pixel
    for i in range(height):
        for j in range(width):
            old = errimg[i, j]
            if old > 1:
                old = 1
            elif old < 0:
                old = 0

            # if pixel is above threshold set to 255 get error
            if old > threshold:
                imgDither[i, j] = 255
                err = int(img[i, j]) - int(imgDither[i, j])
            # else set to 0 get error
            else:
                imgDither[i, j] = 0
                err = int(img[i, j]) - int(imgDither[i, j])
            # for every pixel within 1 pixel of current pixel
            
            if j + 1 <= width - 1:
                errimg[i, j + 1] = errimg[i, j + 1] + ((err * 7/16)/255)
            if i + 1 <= height - 1 and j - 1 >= 0:
                errimg[i + 1, j - 1] = errimg[i + 1, j - 1] + ((err * 3/16)/255)
            if i + 1 <= height - 1:
                errimg[i + 1, j] = errimg[i + 1, j] + ((err * 5/16)/255)
            if i + 1 <= height - 1 and j + 1 <= width - 1:
                errimg[i + 1, j + 1] = errimg[i + 1, j + 1] + ((err * 1/16)/255)

    return imgDither

def contrast(img):
    lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)

    # Applying CLAHE to L-channel
    # feel free to try different values for the limit and grid size:
    clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(8,8))
    cl = clahe.apply(l_channel)

    # merge the CLAHE enhanced L-channel with the a and b channel
    limg = cv2.merge((cl,a,b))

    # Converting image from LAB Color model to BGR color spcae
    enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return enhanced_img

def pointConversion(dith):
    
    h =  dith.shape[0]
    w = dith.shape[1]
    points = np.zeros((1, 2))
    
    for i in range(h):
        for j in range(w):
            if dith[i, j] == 0:
                x = j
                y = i
                points = np.append(points,[[x, -y]], 0)
    points = np.delete(points, 0, 0)
    points = np.hsplit(points, 2)
    return points

def rez(img):
    h = img.shape[0]
    w = img.shape[1]
    r = round(h/w, 3)

    if r < 1.332:
        wrez = 150
        hrez = round(wrez * r)
    else:
        hrez = 200
        wrez = round(hrez / r)
    
    return cv2.resize(img, (wrez, hrez))

def scale(points, Psize):
    # determine paper size used on machine
    if Psize == 4:
        m = 10
        s = 1.267
        xpmax = 210
        ypmax = 297
    elif Psize == 3:
        m = 15
        s = 1.780
        xpmax = 297
        ypmax = 420
    elif Psize == 2:
        m = 20
        s = 2.533
        xpmax = 420
        ypmax = 594
    else:
        print('Incorrect paper size')
        return
    
    points = np.multiply(points, s)
    
    xmax = np.amax(points[0])
    ymax = np.amax(-points[1])

    xdif = xpmax - xmax
    ydif = ypmax - ymax

    points[0] = points[0] + (xdif / 2)
    points[1] = points[1] - (ydif / 2)
    
    return points


#def drawline(x_vals, y_vals)
    # this is the traveling sales man problem
    # look at values within a small range of the paper (x,y)
    # start at point closest to 0,0
    # rearrange into new arrays with values closest to eachother
    # once every value has been rearranged look at next chunk of points
        
    

                        

            

            




