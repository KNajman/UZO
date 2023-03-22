import cv2
import numpy as np


def brightness_correction(image : np.ndarray, brightness_failure: np.ndarray, c :int):
    # Brightness correction of image with brighttness failure of image and constant c
    # image - image with brighttness failure
    # brighttness_failure - image with brighttness failure
    # c - constant
    # return - image with corrected brightness
    
    # Validate input parameters
    if not isinstance(image, np.ndarray):
        raise TypeError("Input parameter 'image' must be a NumPy array")
    if not isinstance(brightness_failure, np.ndarray):
        raise TypeError("Input parameter 'brightness_failure' must be a NumPy array")
    if image.shape != brightness_failure.shape:
        raise ValueError("Input parameters 'image' and 'brightness_failure' must have the same shape")
    if not np.issubdtype(image.dtype, np.uint8):
        raise TypeError("Input parameter 'image' must be of type uint8")
    
    x, y = image.shape[:2]
    original_image = np.zeros_like(image)
    for i in range(x):
        for j in range(y):
            if brightness_failure[i, j] == 0:
                original_image[i, j] = image[i, j]
            else:   
                original_image[i, j] = (c*image[i, j])/brightness_failure[i, j]
    return original_image


def histogram_equalization(image):
    x, y = image.shape[:2]
    new_image = np.zeros((x, y), np.uint8)
    histogram = np.zeros(256)
    for i in range(x):
        for j in range(y):
            histogram[image[i, j]] += 1
    for i in range(1, 256):
        histogram[i] += histogram[i-1]
    for i in range(x):
        for j in range(y):
            new_image[i, j] = histogram[image[i, j]]*255/(x*y)
    return new_image


def D2_DFT(image):
    # 2D Discrete Fourier Transform
    x, y = image.shape[:2]
    new_image = np.zeros((x, y), np.complex)
    for i in range(x):
        for j in range(y):
            for k in range(x):
                for l in range(y):
                    new_image[i, j] += image[k, l] * \
                        np.exp(-2j*np.pi*(i*k/x + j*l/y))
    return new_image


# 1. Na základě využití jasové korekce odstraňte z obrázků cv04_f01.bmp
# a cv04_f02.bmp poruchy cv04_e01.bmp a cv04_e02.bmp. c = 255.

cv04_f01 = cv2.imread('cv04_f01.bmp')
cv04_f02 = cv2.imread('cv04_f02.bmp')
cv04_e01 = cv2.imread('cv04_e01.bmp')
cv04_e02 = cv2.imread('cv04_e02.bmp')

# 1. Brightness correction
c = 255
# cv04_f01 = brightness_correction(cv04_f01, cv04_e01, c)
# cv04_f02 = brightness_correction(cv04_f02, cv04_e02, c)

cv2.imshow('cv04_f01', cv04_f01)
cv2.imshow('cv04_f02', cv04_f02)

# 2. pro obrázek cv04_rentgen.bmp aplikujte histogramovou ekvalizaci dle vzorce q = T(p) = (q_k - q_0)/(N*M)
# zobrazte původní a ekvalizovaný obrázek spolu s histogramy

rentgen = cv2.imread('cv04_rentgen.bmp', 0)
rentgen_eq = cv2.calcHist([rentgen], [0], None, [256], [0, 256])