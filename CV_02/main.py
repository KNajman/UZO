import numpy as np
import matplotlib.pyplot as plt
import cv2
import math
import os

plt.ion()


def clear():
    return os.system('cls')


def calc_center_of_gravity(P):
    x, y = np.meshgrid(np.arange(P.shape[1]), np.arange(P.shape[0]))
    x_t = np.sum(x * P) / np.sum(P)
    y_t = np.sum(y * P) / np.sum(P)
    return x_t, y_t


def calc_hist_and_normalize(hue):
    hist = cv2.calcHist([hue], [0], None, [180], [0, 180])
    hist = cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist


clear()
plt.close('all')

# Read in the image of the object to be tracked
image = cv2.imread('cv02_vzor_hrnecek.bmp')
x = image.shape[0]
y = image.shape[1]

# open video where the object is to be tracked
cap = cv2.VideoCapture('cv02_hrnecek.mp4')

# convert to HSV color space
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# extract hue component
hue = hsv[:, :, 0]

# calculate histogram of hue component
histogram = calc_hist_and_normalize(hue)

# Backprojection of the image P(x, y) = h(G(x, y))
P = cv2.calcBackProject([hue], [0], histogram, [0, 180], 1)

# calc center of gravity
x_t, y_t = calc_center_of_gravity(P)

# draw rectangle in size of image around center of gravity
s = image.shape[0]/2
v = image.shape[1]/2

x1 = int(x_t - v)
y1 = int(y_t - s)

x2 = int(x_t + v)
y2 = int(y_t + s)


while True:
    ret, bgr = cap.read()
    if not ret:
        break

    cv2.rectangle(bgr, (x1, y1), (x2, y2), (0, 255, 0))
    cv2.imshow('Image', bgr)

    # extract region of interest from the next frame
    roi = bgr[int(y_t-s):int(y_t+s), int(x_t-v):int(x_t+v)]
    
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]

    # calcBackProject
    P = cv2.calcBackProject([hue], [0], histogram, [0, 180], 1)

    # calc center of gravity
    x_t, y_t = calc_center_of_gravity(P)

    x1 = int(x_t - v)
    y1 = int(y_t - s)

    x2 = int(x_t + v)
    y2 = int(y_t + s)

    key = 0xFF & cv2.waitKey(30)
    if key == 27:
        break

cv2.destroyAllWindows()
