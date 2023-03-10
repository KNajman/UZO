import numpy as np
import matplotlib.pyplot as plt
import cv2
import math
import os
plt.ion()
def clear(): return os.system('cls')

clear()
plt.close('all')

# Read in the image
image = cv2.imread('cv02_vzor_hrnecek.bmp')

# convert to HSV
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# extract hue
hue = hsv[:, :, 0]

# calculate histogram of hue
hist = cv2.calcHist([hue], [0], None, [180], [0, 180])

# normalize the histogram
normalized_hist = cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)

# plot the histogram
plt.plot(normalized_hist)

cap = cv2.VideoCapture('cv02_hrnecek.mp4')

while True:
    ret, bgr = cap.read()
    if not ret:
        break
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    hist = cv2.calcHist([hue], [0], None, [180], [0, 180])
    hist = cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)

    # calcBackProject
    P = cv2.calcBackProject([hue], [0], hist, [0, 180], 1)

    # calc center of gravity with meshgrid
    x_p = 0
    y_p = 0
    for i in range(P.shape[0]):
        for j in range(P.shape[1]):
            x_p += i * P[i, j]
            y_p += j * P[i, j]

    x_t = x_p / P.sum()
    y_t = y_p / P.sum()

    s = 0.7 * P.sum()
    v = 0.9 * P.sum()

    # draw rectangle in size of image
    x1 = int (s - x_t)
    y1 = int (v - y_t)
    x2 = int (s + x_t)
    y2 = int (v + y_t)
    

    cv2.rectangle(bgr, (x1, y1), (x2, y2), (0, 255, 0))
    cv2.imshow('Image', bgr)

    key = 0xFF & cv2.waitKey(30)
    if key == 27:
        break

cv2.destroyAllWindows()
