import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

plt.ion()
def clear(): return os.system('cls')

clear()
plt.close('all')

tracked_object = cv2.imread('cv02_vzor_hrnecek.bmp')
tracked_object_hsv = cv2.cvtColor(tracked_object, cv2.COLOR_BGR2HSV)
roi_hist = cv2.calcHist([tracked_object_hsv],[0],None,[180],[0,180])
cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

cap = cv2.VideoCapture('cv02_hrnecek.mp4')

# initialize the ROI of the tracked object in the first frame
ret, frame = cap.read()
x1, y1, x2, y2 = 100, 100, 200, 200
roi = frame[y1:y2, x1:x2]
roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
roi_hist = cv2.calcHist([roi_hsv],[0],None,[180],[0,180])
cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # backproject the histogram onto the current frame
    dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)
    
    # perform the CamShift algorithm to find the new position of the object
    ret, track_window = cv2.CamShift(dst, (x1, y1, x2-x1, y2-y1), term_crit)
    
    # update the ROI with the new position and size
    x1, y1, w, h = track_window
    x2 = x1 + w
    y2 = y1 + h
    
    # draw the new bounding box around the tracked object
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    cv2.imshow('Image', frame)
    
    key = 0xFF & cv2.waitKey(30)
    if key == 27:
        break

cv2.destroyAllWindows()
