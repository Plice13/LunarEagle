from PIL import Image
import os

def calculate_aspect_ratio(image_path):
    try:
        img = Image.open(image_path)
        width, height = img.size
        aspect_ratio = width / height
        return aspect_ratio
    except:
        return None

def calculate_aspect_ratios_in_folder(folder_path):
    if not os.path.exists(folder_path):
        print("Folder path does not exist.")
        return

    image_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

    if not image_files:
        print("No image files found in the folder.")
        return

    print("Aspect ratios for images in the folder:")
    for index, image_file in enumerate(image_files, 1):
        image_path = os.path.join(folder_path, image_file)
        aspect_ratio = calculate_aspect_ratio(image_path)
        if aspect_ratio is not None:
            if aspect_ratio >= 1.13:
                pass
            else:
                print(f"{index}: {aspect_ratio} - {image_file}")
        else:
            print(f"{index}: Unable to process {image_file}")

# Provide the path to the folder containing images
folder_path = r'C:\Users\PlicEduard\ondrejov'

calculate_aspect_ratios_in_folder(folder_path)
