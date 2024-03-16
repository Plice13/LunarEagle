import os
import random
import shutil

def copy_random_files(source_dir, dest_dir, num_files_per_folder):
    # Walk through the source directory and get all folders
    for root, dirs, files in os.walk(source_dir):
        for folder in dirs:
            folder_path = os.path.join(root, folder)
            # Get all files in the current folder
            files_in_folder = [os.path.join(folder_path, file) for file in os.listdir(folder_path)]
            # Choose random files from the current folder
            selected_files = random.sample(files_in_folder, min(num_files_per_folder, len(files_in_folder)))
            # Copy selected files to the destination directory while preserving directory structure
            for file_path in selected_files:
                relative_path = os.path.relpath(file_path, source_dir)
                dest_path = os.path.join(dest_dir, relative_path)
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                shutil.copy(file_path, dest_path)

# Example usage:
source_directory = r"C:\Users\PlicEduard\AI4_SOC\classes\every2"
destination_directory = r"C:\Users\PlicEduard\AI4_SOC\models\test4"
number_of_files_per_folder = 4  # Change this to the desired number of random files to copy from each folder

copy_random_files(source_directory, destination_directory, number_of_files_per_folder)
