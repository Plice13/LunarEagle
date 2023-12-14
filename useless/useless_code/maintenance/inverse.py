import os
from PIL import Image

def inverse_image(image_path, output_path):
    try:
        # Open the image
        img = Image.open(image_path)

        # Invert the image
        inverted_img = Image.eval(img, lambda x: 255 - x)

        # Save the inverted image
        inverted_img.save(output_path)

        print(f"Inverted: {output_path}")
    except Exception as e:
        print(f"Error processing {image_path}: {str(e)}")

def process_folder(folder_path, output_folder):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Traverse through all files and subdirectories
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            # Check if the file is an image (you can add more image extensions if needed)
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                input_path = os.path.join(root, file)
                relative_path = os.path.relpath(input_path, folder_path)
                output_path = os.path.join(output_folder, relative_path)

                # Process and inverse the image
                inverse_image(input_path, output_path)

if __name__ == "__main__":
    # Replace 'input_folder' with the path to your input folder
    input_folder = r'C:\Users\PlicEduard\AI\more\runs_martin\Axx_Hsx_600_inv'

    # Replace 'output_folder' with the path to your output folder
    output_folder = input_folder

    process_folder(input_folder, output_folder)
