import cv2
import numpy as np
import math
import statistics

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

def main(image_path, mode=0, ratio_of_interest = 0.8):
    image = cv2.imread(image_path)
    
    # ratio_of_interest = in what area around middle I want to search for line

    height, width = image.shape[:2]
    roi_x_start = int((1-ratio_of_interest)/2 * width) # (1+-ratio_of_interest)/2 is for focusing the Region Of Interest arround the middle
    roi_x_end = int((1+ratio_of_interest)/2* width)
    roi_y_start = int((1-ratio_of_interest)/2 * height)
    roi_y_end = int((1+ratio_of_interest)/2 * height)
    roi = image[roi_y_start:roi_y_end, roi_x_start:roi_x_end] # taking only 0.6 picture around middle

    # Some converts for better recognition
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    if mode==1:
        show_image(edges)
    
    # calculation of lines in ROI (Region Of Interest)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)

    # for storing all rotate angle of each line 
    angles = []

    # analysing every single line
    for rho, theta in lines[:, 0]:
        # rho and theta are polar coordination for the closest point of line to intersection of both axis
        
        # Check if the line is close to the center within a specified range (e.g., 100 pixels)
        roi_size = ratio_of_interest*width # size of y axis of ROI in px
        middle_of_y_roi = 0.5*roi_size # what y-coordination have middle of ROI
        target_accuraci_px = 100 # how much is center of Sun in center of image?

        # calculate what y-value have the line when the x-value is middle of ROI (to see if the lines go thought the middle of image) and calculate the intersect with y-axis
        y_middle, y_intersect=calculate_y_in_x_middle_from_polar(rho, theta, (roi_x_end-roi_x_start)/2)

        if middle_of_y_roi-target_accuraci_px<y_middle<middle_of_y_roi+target_accuraci_px: # is line near middle?
            # if the lines is near middle then we need to check if the line is cca between -45° and +45°
            
            # it is done with y-intersect
            # if the intersect is in the range of ROI (we would see the intersect on left corner of ROI), then the line is more horizontal
            # if not, then the angle is between -45° and +45°
            if y_intersect>roi_size or y_intersect<0:
                # calculate the angle between the detected line and the y-axis
                angle = theta * 180 / np.pi 
                angles.append(angle)
                if mode == 1:
                    print(f"Angle with Y-axis: {angle} degrees")
                    a = np.cos(theta)
                    b = np.sin(theta)
                    x0 = int(a * rho)
                    y0 = int(b * rho)
                    x1 = int(x0 + 1000 * (-b))
                    y1 = int(y0 + 1000 * (a))
                    x2 = int(x0 - 1000 * (-b))
                    y2 = int(y0 - 1000 * (a))
                    cv2.line(roi, (x1, y1), (x2, y2), (0, 255, 0), 2)
            else:
                if mode == 1:
                    a = np.cos(theta)
                    b = np.sin(theta)
                    x0 = int(a * rho)
                    y0 = int(b * rho)
                    x1 = int(x0 + 1000 * (-b))
                    y1 = int(y0 + 1000 * (a))
                    x2 = int(x0 - 1000 * (-b))
                    y2 = int(y0 - 1000 * (a))
                    cv2.line(roi, (x1, y1), (x2, y2), (0, 0, 255), 1)
    if mode == 1:
        show_image(image)
    return statistics.median(angles)    

if __name__ == "__main__":
    mode = 1 # 0 (for nonvisualization) or 1 (visualization)
    image_path = '230926dr - Copy.jpg' #path for image
    print(f"Úhel natočení Slunce je {main(image_path, mode)} stupňů")