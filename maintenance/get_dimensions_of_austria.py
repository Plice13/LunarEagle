import os
from PIL import Image
from collections import defaultdict

def collect_image_dimensions(main_folder):
    # Dictionary to hold dimension counts and their file paths
    dimension_files = defaultdict(list)
    
    # Iterate through all subfolders and files
    for subdir, _, files in os.walk(main_folder):
        for file in files:
            try:
                # Construct full file path
                file_path = os.path.join(subdir, file)
                
                # Try to open the image file
                with Image.open(file_path) as img:
                    # Get dimensions
                    dimensions = img.size  # (width, height)
                    dimension_files[dimensions].append(file_path)
            
            except Exception as e:
                # Print error if file is not an image or cannot be opened
                print(f"Error processing {file_path}: {e}")

    return dimension_files

def main():
    # Specify the path to your main folder
    main_folder = r'C:\Users\PlicEduard\austria\sunspots-austria'  # Change this to your folder path
    
    # Collect dimensions and file paths
    dimension_files = collect_image_dimensions(main_folder)
    
    # Define the target dimensions you are looking for
    target_dimensions = (2550, 3510)
    
    # Print the results for target dimensions
    if target_dimensions in dimension_files:
        print(f"Files with dimensions {target_dimensions}:")
        for file_path in dimension_files[target_dimensions]:
            print(file_path)
        print(f"Total count: {len(dimension_files[target_dimensions])} times")
    else:
        print(f"No files found with dimensions {target_dimensions}.")

if __name__ == "__main__":
    main()
