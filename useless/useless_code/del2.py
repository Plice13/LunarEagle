from PIL import Image
import os

def resize_and_check_dimensions(directory):
    target_width = 2000
    target_height = 1750
    tolerance = 50
    count = 0  # Counter for images not meeting the dimensions

    for filename in os.listdir(directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            file_path = os.path.join(directory, filename)
            with Image.open(file_path) as img:
                width, height = img.size
                new_width = target_width
                new_height = int((target_width / width) * height)

                if abs(new_height - target_height) > tolerance:
                    count += 1
                    print(f"Resized image '{filename}' has dimensions {new_width}x{new_height}, which are not within +- {tolerance}px of {target_width}x{target_height}")

    print(f"Total number of resized images with dimensions outside the specified range: {count}")

# Replace 'path_to_directory' with the path to your directory containing the images
directory_path = r'C:\Users\PlicEduard\ondrejov'
resize_and_check_dimensions(directory_path)
