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
HSV = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# extract the Hue channel
Hue = HSV[:,:,0]

# search for the 

# histogram of Hue
# hist, b = np.histogram(hsv[:,:,0], 256, (0, 256))
hist = cv2.calcHist([Hue], [0], None, [180], [0, 180])
# normalize the histogram
normalized_hist = cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)

# plot the histogram
plt.plot(normalized_hist)

cap = cv2.VideoCapture('cv02_hrnecek.mp4')

while True:
    ret, bgr = cap.read()
    if not ret:
        break
    hsv = cv2.cvtColor(bgr, cv2.COLOR_RGB2HSV)    
    #calcBackProject
    P = cv2.calcBackProject([hsv], [0], normalized_hist, [0, 180], 1)
    
    #calc center of gravity
    x_p, y_p = 0, 0
    for i in range(P.shape[0]):
        for j in range(P.shape[1]):
            x_p += i * P[i,j]
            y_p += j * P[i,j]
            
    x_t = x_p / P.sum()
    y_t = y_p / P.sum()
        
        
    # draw rectangle
    s = 0.7 * math.sqrt(P.sum())
    v = 0.9 * math.sqrt(P.sum())
    
    cv2.rectangle(bgr, (int(y_t-s/2), int(x_t-s/2)), (int(y_t+v/2), int(x_t+v/2)), (0, 255, 0))
    cv2.imshow('Image', bgr)
    
    key = 0xFF & cv2.waitKey(30)
    if key == 27: 
        break

cv2.destroyAllWindows()
