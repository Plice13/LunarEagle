import cv2
import numpy as np

# Load the image
image = cv2.imread('230926dr.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur to reduce noise and improve circle detection
gray = cv2.GaussianBlur(gray, (9, 9), 2, 2)

# Detect circles using the Hough Circle Transform
circles = cv2.HoughCircles(
    gray, 
    cv2.HOUGH_GRADIENT, dp=1, minDist=10, param1=10, param2=30, minRadius=700, maxRadius=800
)

if circles is not None:
    # Convert the (x, y) coordinates and radius of the circles to integers
    circles = np.round(circles[0, :]).astype("int")

    for (x, y, r) in circles:
        # Draw the circle on the image
        cv2.circle(image, (x, y), r, (0, 255, 0), 4)
        cv2.rectangle(image, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)

# Display the image with detected circles
cv2.imwrite("cir_output_image.jpg", image)

# Get the original dimensions of the image
height, width = image.shape[:2]

# Calculate the new dimensions for resizing by 25 percent
new_width = int(width * 0.25)
new_height = int(height * 0.25)

# Resize the image
resized_image = cv2.resize(image, (new_width, new_height))

cv2.imshow("Circles", resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
