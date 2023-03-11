import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

plt.ion()


def clear():
    # replace 'cls' with 'clear' if running on Linux/Mac
    return os.system('clear')


def calc_center_of_gravity(P):
    # Calculate the center of gravity of the image P
    x, y = np.meshgrid(np.arange(P.shape[1]), np.arange(P.shape[0]))
    sum_P = np.sum(P)
    x_t = np.sum(x * P) / sum_P
    y_t = np.sum(y * P) / sum_P
    return x_t, y_t


def calc_hist_and_normalize(hue):
    # Calculate the histogram of the hue component of the image and normalize it
    hist = cv2.calcHist([hue], [0], None, [180], [0, 180])
    hist = cv2.normalize(hist, hist, 0, 180, cv2.NORM_MINMAX)
    return hist


clear()
plt.close('all')

# Read in the image of the object to be tracked
image = cv2.imread('cv02_vzor_hrnecek.bmp')

# Convert to HSV color space and extract hue component
hue = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:, :, 0]

# Calculate histogram of hue component and normalize it
histogram = calc_hist_and_normalize(hue)

plt.plot(histogram)

# Backprojection of the image P(x, y) = h(G(x, y))
P = cv2.calcBackProject([hue], [0], histogram, [0, 180], 1)

# Calculate initial center of gravity
x_t, y_t = calc_center_of_gravity(P)

# Set initial size of the object to be tracked
height, width, _ = image.shape
s = 0.7 * height
v = 0.9 * width

# Open video where the object is to be tracked
cap = cv2.VideoCapture('cv02_hrnecek.mp4')

while True:
    # Read a frame from the video
    ret, bgr = cap.read()
    if not ret:
        break

    # Select region of interest around the center of gravity
    x1 = int(x_t - s / 2)
    y1 = int(y_t - v / 2)
    x2 = int(x_t + s / 2)
    y2 = int(y_t + v / 2)

    roi = bgr[y1:y2, x1:x2]

    # Convert the region of interest to HSV color space and extract hue component
    roi_hue = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)[:, :, 0]

    # Backprojection of the image P(x, y) = h(G(x, y))
    P = cv2.calcBackProject([roi_hue], [0], histogram, [0, 180], 1)

    # calc center of gravity
    x_t, y_t = calc_center_of_gravity(P)

    s = 0.7 * np.sum(P)
    v = 0.9 * np.sum(P)

    # draw rectangle in size of image around center of gravity
    x1 = int(x_t - image.shape[1] / 2)
    y1 = int(y_t - image.shape[0] / 2)

    x2 = int(x_t + image.shape[1] / 2)
    y2 = int(y_t + image.shape[0] / 2)

    cv2.rectangle(bgr, (x1, y1), (x2, y2), (0, 255, 0), thickness=1)
    cv2.imshow('Image', bgr)

    key = 0xFF & cv2.waitKey(30)
    if key == 27:
        break

cv2.destroyAllWindows()
