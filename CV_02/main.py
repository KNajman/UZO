import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

plt.ion()


def clear():
    # replace 'cls' with 'clear' if running on Linux/Mac
    return os.system('clear')


def calc_center_of_gravity(P):  
    x_t = 0
    y_t = 0
    for i in range(P.shape[0]):
        for j in range(P.shape[1]):
            x_t += j * P[i, j]
            y_t += i * P[i, j]
    x_t /= np.sum(P)
    y_t /= np.sum(P)
    return x_t, y_t


def calc_hist_and_normalize(hue):
    # Calculate the histogram of the hue component of the image and normalize it
    hist = cv2.calcHist([hue], [0], None, [180], [0, 180])
    hist = cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    plt.plot(hist)
    return hist


clear()
plt.close('all')

# Read in the image of the object to be tracked
image = cv2.imread('cv02_vzor_hrnecek.bmp')

# Convert to HSV color space and extract hue component
roi_hue = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:, :, 0]

# Calculate histogram of hue component and normalize it
hist = calc_hist_and_normalize(roi_hue)

# zpětná projekce histogramu
P = cv2.calcBackProject([roi_hue], [0], hist, [0, 180], 1)

#výpočet těžiště obrazu
x_t, y_t = calc_center_of_gravity(P)

s = 0.7 * np.sum(P)
v = 0.9 * np.sum(P)

x1 = int(x_t - s)
y1 = int(y_t - s)
x2 = int(x_t + s)
y2 = int(y_t + s)

# Open video where the object is to be tracked
cap = cv2.VideoCapture('cv02_hrnecek.mp4')

while True:
    # Read a frame from the video
    ret, bgr = cap.read()
    if not ret:
        break
    
    roi = bgr[y1:y2, x1:x2]
    roi_hue = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)[:, :, 0]
    
    # zpětná projekce histogramu
    P = cv2.calcBackProject([roi_hue], [0], hist, [0, 180], 1)
    
    #výpočet těžiště obrazu
    x_t, y_t = calc_center_of_gravity(P)
    
    # výběr ROI
    s = 0.7 * np.sum(P)
    v = 0.9 * np.sum(P)
    
    x1 = int(x_t - image.shape[1] / 2)
    y1 = int(y_t - image.shape[0] / 2)
    x2 = int(x_t + image.shape[1] / 2)
    y2 = int(y_t + image.shape[0] / 2)

    cv2.rectangle(bgr, (x1, y1), (x2, y2), (255, 255, 255))

    cv2.imshow('Image', bgr)

    key = 0xFF & cv2.waitKey(30)
    if key == 27:
        break

cv2.destroyAllWindows()
