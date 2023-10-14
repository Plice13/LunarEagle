import cv2
import numpy as np

def adjust_image_parameters(image, brightness=100, contrast=100, exposure=100, highlights=100, shadows=100):
    if image is None:
        print("Error: Unable to load the image.")
        return None

    # Convert image to float32 for accuracy
    image = image.astype(np.float32)

    # Adjust brightness and contrast
    image = image * (brightness / 100)
    image = np.clip(image, 0, 255)

    # Adjust exposure
    image = image ** (exposure / 100)

    # Adjust highlights and shadows
    image = np.interp(image, (0, 255), (highlights, shadows))

    # Convert back to uint8
    image = image.astype(np.uint8)

    return image

# Load your image
image = cv2.imread('OCR\crop.jpg')
cv2.imshow('Original Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Check if the image was loaded successfully
if image is not None:
    # Adjust the image parameters
    adjusted_image = adjust_image_parameters(image, brightness=100, contrast=100, exposure=100, highlights=100, shadows=100)

    # Display the original and adjusted images
    cv2.imshow('Original Image', image)
    cv2.imshow('Adjusted Image', adjusted_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Image not loaded. Please check the file path.")
