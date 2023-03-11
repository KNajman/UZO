import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
def to_hue(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    return hsv[:,:,0]

def mean_shift(img, x1, y1, x2, y2, hist):
    # transform to hue
    hue = to_hue(img)
    # while not converged
    while True:
        # Select pixels
        mask = np.zeros_like(hue)
        mask[y1:y2, x1:x2] = 1
        # Compute histogram
        hist, b = np.histogram(hue[mask==1], 256, (0, 256))
        # Compute mean
        mean = np.sum(hist * np.arange(256)) / np.sum(hist)
        # Compute variance
        var = np.sum(hist * (np.arange(256) - mean)**2) / np.sum(hist)
        # Compute new window
        x1 = int(mean - 2 * np.sqrt(var))
        x2 = int(mean + 2 * np.sqrt(var))
        y1 = int(mean - 2 * np.sqrt(var))
        y2 = int(mean + 2 * np.sqrt(var))
        # Check convergence
        if np.abs(mean - var) < 1:
            break
    return x1, y1, x2, y2




# main if
if __name__ == '__main__':
    plt.ion()
    clear = lambda: os.system('cls')
    clear()
    plt.close('all')

    cap = cv2.VideoCapture('cv02_hrnecek.mp4')

    x1 = 100
    y1 = 100
    x2 = 200
    y2 = 200


    while True:
        ret, bgr = cap.read()
        if not ret:
            break
        

        cv2.rectangle(bgr, (x1, y1), (x2, y2), (0, 255, 0))
        cv2.imshow('Image', bgr)
        key = 0xFF & cv2.waitKey(30)
        if key == 27:
            break
        
    cv2.destroyAllWindows()