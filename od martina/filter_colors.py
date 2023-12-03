from os.path import join, dirname, basename
from glob import glob
import numpy as np
import cv2

BASE = r'C:\Users\PlicEduard\AI\more\runs_martin\Axx_Hsx_Cso_Ekc_250'
GROUPS = ['train', 'val', 'test']  # Use a list for GROUPS

# Threshold of orange in HSV space 
LOWER = np.array([0, 10, 35]) 
UPPER = np.array([225, 200, 255]) 

if __name__ == '__main__':
    labels = BASE.split('\\')[-1].split('_')[:-1]  # Use '\\' for Windows path separation
    print(f'Processing {BASE}, labels: {labels}')

    for group in GROUPS:
        for label in labels:
            for im_path in glob(join(BASE, group, label, '*.png')):
                im = cv2.imread(im_path)
                im_hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
                mask = cv2.bitwise_not(cv2.inRange(im_hsv, LOWER, UPPER))
                im_inv = cv2.bitwise_not(im)
                im_filtered = cv2.bitwise_and(im_inv, im_inv, mask=mask)
                
                # Save the filtered image to a different directory or with a different filename
                cv2.imwrite(im_path, im_filtered)

                # cv2.imshow('im', im)
                # cv2.imshow('im_hsv', im_hsv)
                # cv2.moveWindow('im_hsv', 350, -300)
                # cv2.imshow('mask', mask)
                # cv2.moveWindow('mask', 700, -300)
                # cv2.imshow('im_inv', im_inv)
                # cv2.moveWindow('im_inv', 1050, -300)
                # cv2.imshow('im_filtered', im_filtered)
                # cv2.moveWindow('im_filtered', 1400, -300)
                # cv2.waitKey(0)
            
    # cv2.destroyAllWindows()
    # cv2.release()
    # exit()
