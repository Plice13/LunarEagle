import cv2
import numpy as np

def resize_image(image_path, output_path):
    # Load the image
    image = cv2.imread(image_path)

    # Get the image dimensions
    height, width = image.shape[:2]

    # Calculate the new dimensions using a scaling factor of 0.5
    new_width = int(width * 0.5)
    new_height = int(height * 0.5)

    # Resize the image
    resized_image = cv2.resize(image, (new_width, new_height))

    # Save the resized image
    cv2.imwrite(output_path, resized_image)

def detect_stars(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Preprocessing
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Gray", gray)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    cv2.imshow("blurred", blurred)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Thresholding
    _, thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)
    cv2.imshow("thresh", thresh)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    # Hough Circle Transform
    circles = cv2.HoughCircles(thresh, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=0, maxRadius=0)


    # Draw Circles
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            cv2.circle(image, (x, y), r, (0, 255, 0), 2)

    # Display the image
    cv2.imshow("Detected Stars", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
#    image_path = 'IMG_1124(2).JPG_small.jpg_small.jpg'
#    output_path = image_path+str('_small.jpg')
#    resize_image(image_path, output_path)
    image_path = 'star1.jpg'
    detect_stars(image_path)
