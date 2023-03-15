import numpy as np
import cv2

def normalize_histogram(hist):
    """Normalize a histogram to have unit sum."""
    return hist / np.sum(hist)

def compute_similarity(q, p):
    """Compute the similarity between two histograms."""
    rho = np.sum(np.sqrt(q * p))
    return rho

def meanshift_track(target_hsv, frame, maxit):

    hist = cv2.calcHist([target_hsv], [0], None, [180], [0, 180])
    # Normalize the histogram
    q = normalize_histogram(hist)

    while True:
        it = it + 1
        # Convert the frame to HSV color space
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Compute the back projection of the histogram onto the frame
        P = cv2.calcBackProject([frame_hsv], [0], hist, [0, 180], 1)
        # Compute the weights for the pixels in the target region
        weights = np.zeros_like(P)
        for u in range(q.shape[0]):
            bu = np.zeros_like(P)
            bu[target_hsv[:,:,0] == u] = 1
            pu = cv2.calcHist([target_hsv], [0], bu, [180], [0, 180])
            wu = np.zeros_like(P)
            wu[P > 0] = pu[P > 0] / P[P > 0]
            weights += wu * q[u]
        # Compute the new location of the target
        x_mean = np.sum(weights * np.arange(P.shape[1])) / np.sum(weights)
        y_mean = np.sum(weights * np.arange(P.shape[0])) / np.sum(weights)
        return x_mean, y_mean
    
