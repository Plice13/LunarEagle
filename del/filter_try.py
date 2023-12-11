import cv2
import numpy as np

def remove_orange_part(im):
    Llist = [np.array([0, 0, 0]), np.array([0, 10, 15]), np.array([0, 35, 50]), np.array([0, 50, 80]), np.array([0, 0, 200])]
    for LOWER in Llist:
        UPPER = np.array([255, 255, 255])  # Adjust upper range to cover more shades of orange
        im_hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(im_hsv, LOWER, UPPER)
        cv2.imshow(str(LOWER), mask)

        # Replace orange regions with white
        im_filtered = im.copy()
        im_filtered[mask > 0] = [255, 255, 255]

        cv2.imshow('filtered', im_filtered)
        
    return im_filtered

def martin_remove(im):
    LOWER = np.array([0, 50, 80]) 
    UPPER = np.array([225, 200, 255]) 

    im_hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    mask = cv2.bitwise_not(cv2.inRange(im_hsv, LOWER, UPPER))
    im_inv = cv2.bitwise_not(im)
    im_filtered = cv2.bitwise_and(im_inv, im_inv, mask=mask)
    

    cv2.imshow('im_filtered', im_filtered)
    

image = cv2.imread(r'c:\Users\PlicEduard\AI2\classes\F\19880414053900_593,1304,220,176__cLenght=638.8670928478241_cArea=20023.0__tLoustka=[103.31021247]_uhel=(-0.7430179978215028, -0.14103740881192947).png__Q=31_rho=0.78__b=22_l=280_min_dist=6.png')
cv2.imshow('image', image)
martin_remove(image)
result = remove_orange_part(image)
cv2.imshow('result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
