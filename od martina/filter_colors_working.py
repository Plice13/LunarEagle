from os.path import join
from glob import glob
import numpy as np
import cv2
from tqdm import tqdm

BASE = r'C:\Users\PlicEduard\AI\more\runs_martin\Axx Hsx 600'
GROUPS = ['train', 'val', 'test']  # Use a list for GROUPS

# Threshold of orange in HSV space 
LOWER = np.array([0, 10, 35]) 
UPPER = np.array([225, 200, 255]) 

if __name__ == '__main__':
    labels = BASE.split('\\')[-1].split()[:-1]  # Use '\\' for Windows path separation
    print(f'Processing {BASE}, labels: {labels}')

    for group in tqdm(GROUPS):
        if group != 'test':
            for label in tqdm(labels):
                for im_path in glob(join(BASE, group, label, '*.png')):
                    im = cv2.imread(im_path)
                    im_hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
                    mask = cv2.bitwise_not(cv2.inRange(im_hsv, LOWER, UPPER))
                    im_inv = cv2.bitwise_not(im)
                    im_filtered = cv2.bitwise_and(im_inv, im_inv, mask=mask)
                    #im_inverted = 255-im_filtered
                    im_inverted = im_filtered
                    
                    # Warning: rewriting the original image
                    cv2.imwrite(im_path, im_inverted)

                    # Your visualization code (commented out) can be added back if needed
                    '''
                    cv2.imshow('im', im)
                    cv2.imshow('im_hsv', im_hsv)
                    cv2.moveWindow('im_hsv', 350, 300)
                    cv2.imshow('mask', mask)
                    cv2.moveWindow('mask', 700, 300)
                    cv2.imshow('im_inv', im_inv)
                    cv2.moveWindow('im_inv', 1050, 300)
                    cv2.imshow('im_filtered', im_filtered)
                    cv2.moveWindow('im_filtered', 100, 300)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                    '''
        else:
            for im_path in glob(join(BASE, group, '*.png')):
                    im = cv2.imread(im_path)
                    im_hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
                    mask = cv2.bitwise_not(cv2.inRange(im_hsv, LOWER, UPPER))
                    im_inv = cv2.bitwise_not(im)
                    im_filtered = cv2.bitwise_and(im_inv, im_inv, mask=mask)
                    #im_inverted = 255-im_filtered
                    im_inverted = im_filtered
                    # Warning: rewriting the original image
                    cv2.imwrite(im_path, im_inverted)
