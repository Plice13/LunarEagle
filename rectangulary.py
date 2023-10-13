import cv2
import numpy as np

# Load the image
image = cv2.imread('230926dr.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise
gray = cv2.GaussianBlur(gray, (5, 5), 0)

# Perform edge detection using the Canny function
edges = cv2.Canny(gray, 50, 150, apertureSize=3)

# Find contours in the edge-detected image
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Create a directory to save the extracted rectangles
import os
if not os.path.exists("extracted_rectangles"):
    os.mkdir("extracted_rectangles")

# Loop through the detected contours and find rectangles
for i, contour in enumerate(contours):
    # Approximate the contour to a polygon
    epsilon = 0.04 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    
    if len(approx) == 4:
        # If the polygon has 4 corners, it's a rectangle
        # Crop the rectangular region
        x, y, w, h = cv2.boundingRect(approx)
        roi = image[y:y+h, x:x+w]
        
        # Save the extracted rectangle
        filename = f"extracted_rectangles/rectangle_{i}.jpg"
        cv2.imwrite(filename, roi)

# Display the image with detected rectangles (optional)
cv2.imshow("Rectangles", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
