import cv2
import numpy as np
import math

def naive_image_rotate(image, degree):
    '''
    This function rotates the image around its center by amount of degrees
    provided.
    '''
    
    height = image.shape[0]
    width  = image.shape[1]

    old_mid_x,old_mid_y = (width//2, height//2)
    
    # move degree to the range 0-360
    if degree >= 0:
        degree = degree%360
    else:
        degree = - 360 - degree%360
    tmp_rads = math.radians(degree % 90)
    rads = math.radians(degree)
        
    new_width = int(width*math.cos(tmp_rads)+height*math.sin(tmp_rads))
    new_height = int(width*math.sin(tmp_rads)+height*math.cos(tmp_rads))
    
    print(degree,new_width,new_height)

    # calculate the new center of the image
    midx,midy = (new_width//2, new_height//2)
    
    
    rot_img = np.uint8(np.zeros([new_height,new_width,3]))
    for i in range(rot_img.shape[0]):
        for j in range(rot_img.shape[1]):
            x= (i-midx)*math.cos(rads)+(j-midy)*math.sin(rads)
            y= -(i-midx)*math.sin(rads)+(j-midy)*math.cos(rads)

            x=round(x)+old_mid_x
            y=round(y)+old_mid_y

            if (x>=0 and y>=0 and x<image.shape[0] and  y<image.shape[1]):
                rot_img[i,j,:] = image[x,y,:]

    return rot_img

image = cv2.imread("cv03_robot.bmp")

for i in range(0,-360,-20):
    rotated_image = naive_image_rotate(image, i)
    cv2.imshow("rotated image",rotated_image)
    cv2.waitKey(0)

#cv2.imshow("original image", image)
#cv2.imshow("rotated image",rotated_image)
cv2.waitKey(0)