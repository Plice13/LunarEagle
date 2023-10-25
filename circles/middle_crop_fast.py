import cv2
import numpy as np
import statistics
import os
from tqdm import tqdm

def process_image(path):
    # Load the image
    image = cv2.imread(path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise and improve circle detection
    gray = cv2.GaussianBlur(gray, (9, 9), 2, 2)

    # Detect circles using the Hough Circle Transform
    circles = cv2.HoughCircles(
        gray, 
        cv2.HOUGH_GRADIENT, dp=1, minDist=2, param1=1, param2=40, minRadius=700, maxRadius=800
    )

    if circles is not None:
        # Convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int")
        x_list = circles[:, 0].tolist()
        y_list = circles[:, 1].tolist()

    # Detect smaller circles
    smaller_circles = cv2.HoughCircles(
        gray, 
        cv2.HOUGH_GRADIENT, dp=1, minDist=2, param1=1, param2=40, minRadius=300, maxRadius=400
    )

    if smaller_circles is not None:
        smaller_circles = np.round(smaller_circles[0, :]).astype("int")
        x_list += smaller_circles[:, 0].tolist()
        y_list += smaller_circles[:, 1].tolist()

    if x_list and y_list:
        x_med, y_med = int(statistics.median(x_list)), int(statistics.median(y_list))
        middle = (x_med, y_med)
        cv2.circle(image, middle, radius=3, color=(0, 0, 255), thickness=-1)

        height, width = image.shape[:2]
        cv2.circle(image, (int(width/2), int(height/2)), radius=3, color=(0, 255, 0), thickness=-1)

        roi_small = image[y_med-30:y_med+30, x_med-30:x_med+30]

        # Save the extracted region
        filename = f"circles/extracted_rectangles/middle_{os.path.basename(path)}"
        cv2.imwrite(filename, roi_small)

if __name__ == '__main__':
    slozka_s_obrazky = r"C:\Users\PlicEduard\ondrejov/"
    output_folder = "circles/extracted_rectangles"
    os.makedirs(output_folder, exist_ok=True)

    # Process all files in the directory
    files = [os.path.join(slozka_s_obrazky, filename) for filename in os.listdir(slozka_s_obrazky)]
    for file in tqdm(files, total=len(files)):
        process_image(file)
