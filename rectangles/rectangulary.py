import cv2
import numpy as np

def show_image(picture, screen=1):
    #screen 0 notebook, screen 1 monitor
    height, width = picture.shape[:2]
    if screen == 0:
        final_height=750
    else:
        final_height=1000
    final_dimension=(round((final_height/height)*width),final_height)
    resized_picture = cv2.resize(picture, dsize=final_dimension)
    cv2.imshow('resized_picture', resized_picture)
    cv2.waitKey()
    cv2.destroyWindow('resized_picture')

# Load the image
image = cv2.imread('lena_black_spots_mask.png')

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise
gray = cv2.GaussianBlur(gray, (5, 5), 0)

# Perform edge detection using the Canny function
edges = cv2.Canny(gray, 50, 150, apertureSize=3)

show_image(edges)
# Find contours in the edge-detected image
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Create a directory to save the extracted rectangles
import os
if not os.path.exists("extracted_rectangles"):
    os.mkdir("extracted_rectangles")

# Loop through the detected contours and find rectangles
for i, contour in enumerate(contours):
    cv2.drawContours(image, [contour], 0, (0,255,0), 3)
    #show_image(image)
    # Approximate the contour to a polygon
    epsilon = 0.04 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    
    if len(approx) == 4:
        # If the polygon has 4 corners, it's a rectangle
        # Crop the rectangular region
        x, y, w, h = cv2.boundingRect(approx)
        if w+h<20:
            break
        roi = image[y:y+h, x:x+w]
        
        # Save the extracted rectangle
        filename = f"rectangles\extracted_rectangles/rectangle_{i}.jpg"
        cv2.imwrite(filename, roi)

# Display the image with detected rectangles (optional)
cv2.imshow("Rectangles", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
