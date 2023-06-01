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

def WVS(img):
    """
    This is for weighted voronoi stippling:
    1. generate random points in a new image the same size as input image
    2. check pixels around points and move points to nearest darkest pixel
    3. iterate this process a set amount of times

    Honestly this would be better to do in C or C++ butt fuck it 
    """
    # making blank image and generating random points
    height =  img.shape[0]
    width = img.shape[1]
    imgStipple = np.zeros((height, width, 1), dtype = np.uint8)
    
    points = 10000
    iterations = 100
    rand_points = np.random.rand(points, 2)
    rand_points[:, 0] *= width
    rand_points[:, 1] *= height
    rand_points = np.round(rand_points)

    for i in height:
        for j in width:
            pass




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

def rez(img):
    # this function resizes the image while maintaining the original ratio
    h = img.shape[0]
    w = img.shape[1]
    r = round(h/w, 3)
    # depending on the ratio either the height or width is changed first and the other is scaled to match
    if r < 1.332:
        wrez = 450
        hrez = round(wrez * r)
    else:
        hrez = 600
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

def dist_matrix(points_array):
    num = points_array.shape[0]
    dist = np.zeros((num, num))
    for i in range(num):
        for j in range(num):
            if i == j:
                continue
            dist[i][j] = np.linalg.norm(points_array[i] - points_array[j])
    return dist

def nearest_neighbor_tsp(distance_matrix):
    num_cities = len(distance_matrix)
    unvisited_cities = set(range(num_cities))
    start_city = np.random.choice(num_cities)
    tour = [start_city]
    unvisited_cities.remove(start_city)
    while unvisited_cities:
        nearest_city = min(unvisited_cities, key=lambda city: distance_matrix[tour[-1]][city])
        tour.append(nearest_city)
        unvisited_cities.remove(nearest_city)
    return tour #, sum(distance_matrix[tour[i - 1]][tour[i]] for i in range(num_cities))

def nearest_neighbor_it(points_array):
    tour = np.zeros((1, 2))
    while points_array.size > 0:
        current = tour[-1]
        nearest_city = None
        nearest_dist = float('inf')
        for i in range(len(points_array)):
            if np.array_equal(points_array[i], current):
                continue
            distance = np.linalg.norm(current - points_array[i])
            if distance < nearest_dist:
                nearest_city = i
                nearest_dist = distance
        tour = np.append(tour, [points_array[nearest_city]], axis=0)
        location = np.where(points_array == tour[-1])
        points_array = np.delete(points_array, location[0][0], 0)

    return tour


            

            





