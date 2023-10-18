import cv2
import os
from tqdm import tqdm

# Path to the subfolder containing your images
subfolder_path = r'C:\Users\PlicEduard\ondrejov\small'

# Output folder for saving the blended images
output_folder = '.'

# Ensure the output folder exists
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# List all files in the subfolder
image_files = [f for f in os.listdir(subfolder_path) if os.path.isfile(os.path.join(subfolder_path, f))]

# Set the transparency level (alpha value) for blending
alpha = 1-1/len(image_files)  # Adjust this value as needed

# Load the base image (the first image in the subfolder)
base_image = cv2.imread(os.path.join(subfolder_path, image_files[0]))

# Initialize a progress bar
pbar = tqdm(total=len(image_files) - 1, desc="Processing images")

# Loop through the rest of the images and blend them sequentially
for image_file in image_files[1:]:
    next_image = cv2.imread(os.path.join(subfolder_path, image_file))
    next_image = cv2.resize(next_image, (base_image.shape[1], base_image.shape[0]))
    base_image = cv2.addWeighted(base_image, 0.9, next_image, 0.1, 0)
    pbar.update(1)

pbar.close()

# Save the final blended image
output_path = os.path.join(output_folder, 'blended_image.jpg')
cv2.imwrite(output_path, base_image)

# Display the blended image (optional)
cv2.imshow('Blended Image', base_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
