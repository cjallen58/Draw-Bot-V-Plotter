from math import sqrt
import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial.distance import pdist, squareform
from tsp_solver.greedy_numpy import solve_tsp


import sys
import time

def dithering(img, threshold=.35):
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
    """This algorithm is called Floyd-Steinberg Dithering. It was taken from wikipedia
    https://en.wikipedia.org/wiki/Floyd%E2%80%93Steinberg_dithering"""
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
    # converts image to lab (whatever that is)
    # splits the data items
    lab= cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a, b = cv2.split(lab)

    # Applying CLAHE to L-channel
    # feel free to try different values for the limit and grid size:
    clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(8,8))
    cl = clahe.apply(l_channel)

    # merge the CLAHE enhanced L-channel with the a and b channel
    limg = cv2.merge((cl,a,b))

    # Converting image from LAB Color model to BGR color space
    enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return enhanced_img

def pointConversion(dith):
    
    h =  dith.shape[0]
    w = dith.shape[1]
    # creates empty numpy array for points
    points = np.zeros((1, 2))
    # for every point turn it into an x and y coordinate and add to new array
    for i in range(h):
        for j in range(w):
            if dith[i, j] == 0:
                x = j
                y = i
                points = np.append(points,[[x, -y]], 0)
    # remove first 0,0 point
    points = np.delete(points, 0, 0)
    # splits the 2 columns into 2 seperate arrays inside a single array
    # points = np.hsplit(points, 2)
    return points

def rez(img):
    # this function resizes the image while maintaining the original ratio
    h = img.shape[0]
    w = img.shape[1]
    r = round(h/w, 3)
    # depending on the ratio either the height or width is changed first and the other is scaled to match
    if r < 1.332:
        wrez = 600
        hrez = round(wrez * r)
    else:
        hrez = 800
        wrez = round(hrez / r)
    
    return cv2.resize(img, (wrez, hrez))

def scale(points, Psize):
    # determine paper size used on machine and its max x and y values
    if Psize == 4:
        s = 1.267
        xpmax = 210
        ypmax = 297
    elif Psize == 3:
        s = 1.780
        xpmax = 297
        ypmax = 420
    elif Psize == 2:
        s = 2.533
        xpmax = 420
        ypmax = 594
    else:
        print('Incorrect paper size')
        return
    
    points = np.multiply(points, s)

    xdif = round(xpmax - np.amax(points[0]), 3)
    ydif = round(ypmax - np.amax(-points[1]), 3)

    points[0] = points[0] + (xdif / 2)
    points[1] = points[1] - (ydif / 2)
    
    return points

def order(points):
    # points will be a numpy array
    # for now i just want it to work
    # also need to increase my point density
    # time to rewrite the entire thing using dijkstras algorithm
    # didnt quite do that but im getting closer
    # it seems to bounce back and forth as it hits every point
    # need to filter a small subsection of the graph first and then iterate over a range


    #initialize final point order and make 0,0 the first point
    start_time = time.time()
    final_order = np.zeros((1, 2))
    lcount = 0
    print(f"points: {points.size / 2}")

    while points.size > 0:
        
        #dictionary to store the closest point
        shortest = {}
        for row in points:
            
            #calculate distance
            dist = sqrt((row[0]**2) + (row[1]**2))
            
            #compare distances and update shortest
            if 'distance' in shortest:
                if dist < shortest['distance']:
                    shortest['distance'] = dist
                    shortest['point'] = row
                else:
                    continue
            else:
                shortest['distance'] = dist
                shortest['point'] = row
        
        #track run time
        run_time = round(time.time() - start_time, 3)
        lcount += 1
        
        #debug 
        if lcount % 1000 == 0:
            print(f"\nrunning")
            print(f"points remaining: {points.size / 2}")
            print(f"run time: {run_time}")

        #appened short point coordinates to final order
        final_order = np.append(final_order, [shortest['point']], axis = 0)
        location = np.where(points == shortest['point'])
        points = np.delete(points,location[0][0], 0)
    
    return final_order
    
def mid_point_search(points):
   
    """
    this is gonna be interesting.
    General Idea:
    1: define first point and end point in a new array
        start will be 0,0 end will be xmax, ymax or the bottom right point
    2: from list of points find the point mid way between start and end 
    3: recursivly do this moving towards the start point
    4: when no points between start and closest point from recursion and do the same moving away from start.
        this is similar to the sort left then right in a recursive merge sort
        the jumping to mid points was inspisred by binary seraching
    5: please for the love of god i hope this works
    """
    start_time = time.time()
    final_order = np.zeros((1, 2))
    lcount = 0                     
    final_order = np.append(final_order, [points[-1]], axis = 0)
    points = np.delete(points, -1, 0)
    #i need to organize these calculations to work properly with recursion
    length = sqrt(final_order[0][0]**2 + final_order[0][1]**2) - sqrt(final_order[1][0]**2 + final_order[1][1]**2)
    
    # i now have my starting point and end point in final_order as well as the length between them
    # now find find mid-point
    list = []
    for row in points:
        # this is going to be tricky and very calculation intensive 
        
        distance = sqrt(row[0]** + row[1]**2)
        location = np.where(points == row)

def tspsort(Dithered_image):
    """apparently i just learned, months after starting this, 
    that someone else was nice enough to solve this exact problem for me
    im so happy that after all this time and ripping my hair out over my own
    algorithm that I found exactly what i need at 
    https://randalolson.com/2018/04/11/traveling-salesman-portrait-in-python/"""
    
    """Alright my newest idea is to seperate the dithered image into sub sections
    to reduce the amount of calculations needed for the tsp solver. Apply the tsp
    to these sub sections and then within the section just do a nearest search.
    Hopefully this wont make the program take 40 hours..."""
    # I should have commented this more

    # First seperate dithiered image into a more condensed array
    x = 5
    y = 5
    hieght = Dithered_image.shape[0]
    width = Dithered_image.shape[1]
    
    black_points = np.argwhere(Dithered_image == 0)
    black_points = np.delete(black_points, 2, 1)
    distances = pdist(black_points)
    dist_matrix = squareform(distances)
    path = solve_tsp(dist_matrix)
    final_points = [black_points[x] for x in path]
    x_vals = [x[1] for x in final_points]
    y_vals = [-x[0] for x in final_points]
    return [x_vals, y_vals]


            

            





