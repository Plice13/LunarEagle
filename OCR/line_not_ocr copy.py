import cv2
import numpy as np
import math


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

def calculate_y_in_x_middle_from_polar(r, theta, x_value):
    theta=theta/math.pi*180
    # Convert polar coordinates to Cartesian coordinates
    x = r * math.cos(math.radians(theta))
    y = r * math.sin(math.radians(theta))
    
    # Calculate the closest y-intercept
    a = -x / y  # Calculate the slope using the provided point
    y_intersect = y - a * x
    
    # Calculate the y-value when x = x_value
    y_value = a * x_value + y_intersect
    return y_value, y_intersect

# Load the image
image = cv2.imread('230926dr.jpg')  # Replace 'your_image.jpg' with your image file

# Define the region of interest (ROI) as 20% to 80% of the image width and 20% to 80% of the image height
height, width = image.shape[:2]
roi_x_start = int(0.2 * width)
roi_x_end = int(0.8 * width)
roi_y_start = int(0.2 * height)
roi_y_end = int(0.8 * height)
roi = image[roi_y_start:roi_y_end, roi_x_start:roi_x_end]

# Convert the ROI to grayscale
gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

# Apply a Canny edge detection to the grayscale ROI
edges = cv2.Canny(gray, 50, 150, apertureSize=3)

show_image(edges)
# Use the Hough Line Transform to detect lines in the ROI
lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)

# Draw the detected lines and points on the original ROI and calculate angles
for rho, theta in lines[:, 0]:
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = int(a * rho)
    y0 = int(b * rho)
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))

    # Check if the line is close to the center within a specified range (e.g., 100 pixels)
    roi_size = roi_y_end-roi_y_start
    middle_of_y_roi = 0.5*roi_size
    target_accuraci_px = 100
    y_middle, y_intersect=calculate_y_in_x_middle_from_polar(rho, theta, (roi_x_end-roi_x_start)/2)

    if middle_of_y_roi-target_accuraci_px<y_middle<middle_of_y_roi+target_accuraci_px:
        if y_intersect>roi_size or y_intersect<0:
            cv2.line(roi, (x1, y1), (x2, y2), (255, 0, 0), 3)

            # Plot points (x0, y0), (x1, y1), and (x2, y2) on the ROI
            cv2.circle(roi, (x0, y0), 5, (0, 255, 0), -1)

            # Calculate the angle between the detected line and the y-axis
            angle = theta * 180 / np.pi  # Convert radians to degrees
            print(f"Angle with Y-axis: {angle} degrees")
        else:
            cv2.line(roi, (x1, y1), (x2, y2), (0, 0, 255), 1)


show_image(image)