import cv2
import numpy as np

def remove_orange_part(im):
    LOWER = np.array([0, 10, 35])
    UPPER = np.array([20, 255, 255])  # Adjust upper range to cover more shades of orange
    im_hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(im_hsv, LOWER, UPPER)
    cv2.imshow('mask', mask)

    # Replace orange regions with white
    im_filtered = im.copy()
    im_filtered[mask > 0] = [255, 255, 255]

    cv2.imshow('filtered', im_filtered)
    
    return im_filtered

def martin_remove(im):
    LOWER = np.array([0, 10, 35]) 
    UPPER = np.array([225, 200, 255]) 

    im_hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    mask = cv2.bitwise_not(cv2.inRange(im_hsv, LOWER, UPPER))
    im_inv = cv2.bitwise_not(im)
    im_filtered = cv2.bitwise_and(im_inv, im_inv, mask=mask)
    

    cv2.imshow('im_filtered', im_filtered)
    

image = cv2.imread(r'C:\Users\PlicEduard\AI2\classes\F/19880414053900_587,1297,232,188__cLenght=676.9087228775024_cArea=23024.5__tLoustka=[111.36426716]_uhel=(-1.596063857975726, -0.5268101643941692).png__Q=31_rho=0.78__b=22_l=280_min_dist=6.png')
cv2.imshow('image', image)
martin_remove(image)
result = remove_orange_part(image)
cv2.imshow('result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()
