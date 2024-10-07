import cv2
import numpy as np
from rembg import remove

# Load the input image
input_path = 'sunspot.jpg'  # Path to your image
output_path_faded = 'sunspot_faded_with_transparency.png'
output_path_hard_border = 'sunspot_hard_border_with_transparency.png'

# Read the image
input_image = cv2.imread(input_path)

# Convert the image from BGR (OpenCV format) to RGB (required by rembg)
rgb_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)

# Remove the background
result = remove(rgb_image)

# Convert the result back to BGR for OpenCV processing
result_bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

# Create a binary mask (white for object, black for background)
gray_result = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2GRAY)
_, mask = cv2.threshold(gray_result, 1, 255, cv2.THRESH_BINARY)

# ---- Optional: Expand surroundings by dilating the mask ----
kernel = np.ones((50, 50), np.uint8)  # Adjust this kernel size for more or less surroundings
dilated_mask = cv2.dilate(mask, kernel, iterations=1)

# ---- Faded Version (Smooth Transition) ----
# Blur the dilated mask for smooth fade into transparency
blurred_mask = cv2.GaussianBlur(dilated_mask, (111, 111), 0)

# Normalize the mask to be in range [0, 1] for smooth blending
mask_normalized = blurred_mask.astype(float) / 255.0

# Create a 4-channel output (R, G, B, Alpha)
faded_output = np.zeros((input_image.shape[0], input_image.shape[1], 4), dtype=np.uint8)

# Apply the mask to the original image
faded_output[:, :, :3] = input_image  # Keep original RGB
faded_output[:, :, 3] = (mask_normalized * 255).astype(np.uint8)  # Apply the blurred alpha mask

# Save the faded version with transparency
cv2.imwrite(output_path_faded, faded_output)

# ---- Hard Border Version (Sharp Cutoff) ----
# Use the dilated mask directly for sharp edges (no blur)
hard_border_output = np.zeros((input_image.shape[0], input_image.shape[1], 4), dtype=np.uint8)

# Apply the sharp alpha mask
hard_border_output[:, :, :3] = input_image  # Keep original RGB
hard_border_output[:, :, 3] = dilated_mask  # Use the hard-edge alpha mask

# Save the hard border version with transparency
cv2.imwrite(output_path_hard_border, hard_border_output)

# Display images for comparison
cv2.imshow('Original Image', input_image)
cv2.imshow('Faded Mask', blurred_mask)
cv2.imshow('Faded Image with Transparency', faded_output)
cv2.imshow('Hard Border Image with Transparency', hard_border_output)

cv2.waitKey(0)
cv2.destroyAllWindows()