import cv2
import numpy as np
import math

# Load the image
image = cv2.imread('OCR/line.jpg')  # Replace 'your_image.jpg' with your image file

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply a Canny edge detection to the grayscale image
edges = cv2.Canny(gray, 50, 150, apertureSize=3)

# Use the Hough Line Transform to detect lines in the image
lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)

# Draw the detected lines on the original image
for rho, theta in lines[:, 0]:
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))
    cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

# Calculate the angle between each detected line and the y-axis
for rho, theta in lines[:, 0]:
    angle = theta * 180 / np.pi  # Convert radians to degrees
    print(f"Angle with Y-axis: {angle} degrees")

# Show the original image with detected lines
cv2.imshow('Detected Lines', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
