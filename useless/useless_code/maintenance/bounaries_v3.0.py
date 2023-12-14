import os
from tqdm import tqdm
import cv2
import numpy as np
import statistics
from PIL import Image, ImageDraw
import pandas as pd
import math
import sunpy.coordinates
from datetime import datetime
import csv

def enhance_image_cv2(img, visualisation=False):
    # if not too dark then make it white
    k=220
    low = (0,0,0)
    high = (k,k,k)

    mask = cv2.inRange(img, low, high)
    mask = 255 - mask
    if visualisation == True:
        cv2.imshow("mask", mask)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    blured = cv2.GaussianBlur(mask,(3,3),cv2.BORDER_DEFAULT)
    if visualisation == True:
        cv2.imshow("blured", blured)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    mask2 = cv2.inRange(blured, 0, 200)

    mask2 = 255 - mask2
    if visualisation == True:
        cv2.imshow("mask2", mask2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    return mask2

def find_rectangles(enhanced, base, file, path):
    finding_status = True
    # some more adjustments
    thresh = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # scanning contours
    for i, cnt in enumerate(contours):
        approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            if 10 < w < 300 and 10 < h < 300:
                # Extract the region of interest
                contour_length = cv2.arcLength(cnt, True)
                contour_area = cv2.contourArea(cnt)
                pravdivovost, uhel = is_parallel(approx)
                if pravdivovost == True:
                    pravdivost, value = is_thin(approx)
                    if pravdivost == False:
                        if finding_status == True:
                            finding_status = False
                        else:
                            log_file = open(log_path, 'a', encoding='utf-8')
                            log_file.write(f'Obrázek {file} má více obdélníků.\n')
                            log_file.close()
                        mask = np.zeros_like(base, dtype=np.uint8)
                        cv2.drawContours(mask, [approx], -1, color=(255,255,255), thickness=cv2.FILLED)                    
                        mask_inverted = ~mask
                        roi_whole_image = cv2.bitwise_or(base, mask_inverted)
                        cv2.imwrite(path, roi_whole_image)
                        
            # Extract the region of interest
    if finding_status == True:
        log_file = open(log_path, 'a', encoding='utf-8')
        log_file.write(f'Obrázek {file} nemá obdélník.\n')
        log_file.close()
def is_thin(approx, boundary=28):
    try:
        appr = sorted(approx, key=lambda c: c[0][0])
        pa, pb = sorted(appr[:2], key=lambda c: c[0][1])
        pc, pd = sorted(appr[2:], key=lambda c: c[0][1])

        provided_points = np.array([pa, pb, pc, pd])

        # Calculate pairwise distances between all provided points
        distances = np.sqrt(np.sum((provided_points[:, None] - provided_points) ** 2, axis=-1))
        min_distance_list = []
        for i in range(len(provided_points)):
            for j in range(i + 1, len(provided_points)):
                distance = distances[i, j]
                min_distance_list.append(distance)
                if distance < boundary:
                    print('HIT')
                    return True, min(min_distance_list)
        return False, min(min_distance_list)
    except:
        print('------ERROR------')

def is_parallel(approx, boundary=8):
    # Ensure there are exactly 4 points
    if len(approx) != 4:
        return False

    # Sort the points
    sorted_approx = sorted(approx, key=lambda c: (c[0][0], c[0][1]))
    pa, pb = sorted(sorted_approx[:2], key=lambda c: c[0][1])
    pc, pd = sorted(sorted_approx[2:], key=lambda c: c[0][1])

    # Calculate angles between lines formed by the points
    angle1 = math.atan2(pa[0][1] - pb[0][1], pa[0][0] - pb[0][0]) - math.atan2(pc[0][1] - pd[0][1], pc[0][0] - pd[0][0])
    angle2 = math.atan2(pa[0][1] - pc[0][1], pa[0][0] - pc[0][0]) - math.atan2(pb[0][1] - pd[0][1], pb[0][0] - pd[0][0])

    # Check if the absolute degrees of the angles are within the boundary
    if abs(math.degrees(angle1)) < boundary and abs(math.degrees(angle2)) < boundary:
        return True, (math.degrees(angle1),math.degrees(angle2))
    else:
        return False, (math.degrees(angle1),math.degrees(angle2))

def process_images_in_folder(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(root, file)
                picture = cv2.imread(image_path)
                
                if picture is not None:
                    enhanced_picture = enhance_image_cv2(picture)
                    find_rectangles(enhanced_picture, picture, file, image_path)

# Set the folder path where your images are located
folder_path = r'C:\Users\PlicEduard\AI\more\classes\first_letter'
log_path = os.path.join(folder_path, 'log.txt')

# Call the function to process images in the specified folder
process_images_in_folder(folder_path)
