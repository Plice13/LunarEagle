import csv
import cv2
import ast
import os
import numpy as np


def apply_polygon_mask(image, coordinates):
    # Create a black background
    result = np.zeros_like(image, dtype=np.uint8)

    # Convert the coordinates to NumPy array format
    coordinates_np = np.array(coordinates)

    # Draw a filled polygon on the black background
    cv2.fillPoly(result, [coordinates_np], (255, 255, 255))

    # Apply the mask to the original image
    image=255-image
    result = cv2.bitwise_and(image, result)
    result=255-result

    return result

def get_coordinates_from_csv(csv_path, image_filename_to_check):
    # Check if the image filename exists in the CSV
    with open(csv_path, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)  # Skip header if exists
        for row in csvreader:
            # Extract information from the CSV row
            current_image_filename, coordinates_str = row

            # Check if the current row's image filename matches the one to check
            if current_image_filename == image_filename_to_check:
                # Return the coordinates associated with the image filename
                coordinates = ast.literal_eval(coordinates_str)
                return coordinates

    # Return None if the image filename is not found in the CSV
    return None

def remoove_orange_part(im):
    im_hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
    mask = cv2.bitwise_not(cv2.inRange(im_hsv, LOWER, UPPER))
    im_inv = cv2.bitwise_not(im)
    im_filtered = cv2.bitwise_and(im_inv, im_inv, mask=mask)
    return im_filtered

# Example usage
LOWER = np.array([0, 10, 35]) 
UPPER = np.array([225, 200, 255]) 


csv_path = "bounding_boxes.csv"
image_filename_to_check = "19930715101800_1525,954,83,97"
image_path = '19930715101800_1525,954,83,97__cLenght=343.3553384542465_cArea=6931.5__tLoustka=[75.16648189]_uhel=(-0.6264217456156858, -0.05003996938464458).png__Q=280_rho=0.78__b=8_l=256_min_dist=0.png'

# Check if the image filename exists in the CSV and get the coordinates
coordinates = get_coordinates_from_csv(csv_path, image_filename_to_check)

if coordinates is not None:
    # Read the image
    image = cv2.imread(image_path)

    # Apply the polygon mask
    result_image = apply_polygon_mask(image, coordinates)
    result_image = remoove_orange_part(result_image)

    # Display or save the result
    cv2.imshow("Original Image", image)
    cv2.imshow("Result Image", result_image)
    cv2.imwrite('after_cord.png', result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print(f"The image filename {image_filename_to_check} does not exist in the CSV.")
