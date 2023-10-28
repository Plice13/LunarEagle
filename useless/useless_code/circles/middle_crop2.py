import cv2
import numpy as np
import statistics
import os
from tqdm import tqdm

path = '230926dr.jpg'

image = cv2.imread(path)
height, width = image.shape[:2]

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (9, 9), 2, 2)

x_list = []
y_list = []

circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=2, param1=1, param2=40, minRadius=700, maxRadius=800)
if circles is not None:
    circles = np.round(circles[0, :]).astype("int")
    for (x, y, r) in circles:
        x_list.append(x)
        y_list.append(y)       

circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=2, param1=1, param2=40, minRadius=300, maxRadius=400)
if circles is not None:
    circles = np.round(circles[0, :]).astype("int")
    for (x, y, r) in circles:
        x_list.append(x)
        y_list.append(y)    

x_med,y_med = (int(statistics.median(x_list)), int(statistics.median(y_list)))
move_x = int(width/2-x_med)
move_y = int(height/2-y_med)
print(move_x, move_y)

from PIL import Image, ImageDraw

# Open the image
image = Image.open(path)

# Create a new image with the same size
new_image = Image.new('RGB', image.size)

# Paste the original image with a 5-pixel offset
new_image.paste(image, (move_x, move_y))

draw = ImageDraw.Draw(new_image)
size_pill=1
draw.ellipse((width/2-size_pill, height/2-size_pill, width/2+size_pill, height/2+size_pill), fill=(255, 0, 0), outline=(0, 0, 0))

# Save the new image
new_image.save('moved_image.png')