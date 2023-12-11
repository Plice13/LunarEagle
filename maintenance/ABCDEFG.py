import os
import shutil

def organize_files(source_folder, destination_folder):
    for item in os.listdir(source_folder):
        item_path = os.path.join(source_folder, item)

        # Check if it's a directory
        if os.path.isdir(item_path):
            # Get the first letter of the directory
            first_letter = item[0].upper()

            # Create a destination directory if it doesn't exist
            dest_dir = os.path.join(destination_folder, first_letter)
            if not os.path.exists(dest_dir):
                os.makedirs(dest_dir)

            # Copy all files from the subfolder to the destination directory
            for sub_item in os.listdir(item_path):
                sub_item_path = os.path.join(item_path, sub_item)
                shutil.copy(sub_item_path, dest_dir)

if __name__ == "__main__":
    source_folder = r"C:\Users\PlicEduard\clasifics\znovu"  # Replace with the path to your source folder
    destination_folder = r"C:\Users\PlicEduard\AI2\classes"  # Replace with the path to your destination folder

    organize_files(source_folder, destination_folder)
