import cv2
import numpy as np
import statistics

# Load the image
image = cv2.imread('230926dr.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise and improve circle detection
gray = cv2.GaussianBlur(gray, (9, 9), 2, 2)

x_list = []
y_list = []

# Detect circles using the Hough Circle Transform
circles = cv2.HoughCircles(
    gray, 
    cv2.HOUGH_GRADIENT, dp=1, minDist=2, param1=1, param2=40, minRadius=700, maxRadius=800
)

if circles is not None:
    # Convert the (x, y) coordinates and radius of the circles to integers
    circles = np.round(circles[0, :]).astype("int")
    for (x, y, r) in circles:
        x_list.append(x)
        y_list.append(y)
        # Draw the circle on the image
        cv2.circle(image, (x, y), r, (0, 255, 0), 1)
        cv2.circle(image, (x, y), radius=4, color=(0, 128, 255), thickness=-1)


# Detect circles using the Hough Circle Transform
circles = cv2.HoughCircles(
    gray, 
    cv2.HOUGH_GRADIENT, dp=1, minDist=2, param1=1, param2=40, minRadius=300, maxRadius=400
)

if circles is not None:
    # Convert the (x, y) coordinates and radius of the circles to integers
    circles = np.round(circles[0, :]).astype("int")
    for (x, y, r) in circles:
        x_list.append(x)
        y_list.append(y)
        # Draw the circle on the image
        cv2.circle(image, (x, y), r, (255, 0, 0), 1)
        cv2.circle(image, (x, y), radius=4, color=(255, 128, 255), thickness=-1)


# Display the image with detected circles
cv2.imwrite("cir_output_image.jpg", image)

# Get the original dimensions of the image
height, width = image.shape[:2]

# Calculate the new dimensions for resizing by 25 percent
new_width = int(width * 1)
new_height = int(height * 1)

middle = (int(statistics.median(x_list)), int(statistics.median(y_list)))

print(f'Střed kružnice vypadá jakože bude {middle}')

image = cv2.circle(image, middle, radius=3, color=(0, 0, 255), thickness=-1)

# Resize the image
resized_image = cv2.resize(image, (new_width, new_height))

cv2.imshow("Circles", resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
