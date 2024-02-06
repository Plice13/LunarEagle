import os
import random
import shutil

def split_data(source, dest_folder, class_name, train_ratio=0.8, val_ratio=0.1):
    dest_train = os.path.join(dest_folder, 'train', class_name)
    dest_val = os.path.join(dest_folder, 'val', class_name)
    dest_test = os.path.join(dest_folder, 'test', class_name)

    # Get all files in the source folder
    files = os.listdir(source)

    # Shuffle the files randomly
    random.shuffle(files)

    # Calculate the split indices
    total_samples = len(files)
    train_split = int(train_ratio * total_samples)
    val_split = train_split + int(val_ratio * total_samples)

    # Copy files to train, val, and test directories
    for i, file in enumerate(files):
        source_file = os.path.join(source, file)
        if i < train_split:
            dest_folder = dest_train
        elif i < val_split:
            dest_folder = dest_val
        else:
            dest_folder = dest_test

        if not os.path.exists(dest_folder):
            os.makedirs(dest_folder)

        dest_file = os.path.join(dest_folder, file)
        shutil.copyfile(source_file, dest_file)

def process_folders(main_folder, dest_prefolder):
    # Get name of destination folder
    subfolders_list = []
    for subfolder in os.listdir(main_folder):
        subfolder_path = os.path.join(main_folder, subfolder)
        if os.path.isdir(subfolder_path):
            subfolders_list.append(subfolder)

    subfolders = '_'.join(subfolders_list) + f'_0_0'
    dest_folder = os.path.join(dest_prefolder, subfolders)

    # Run script
    for subfolder in os.listdir(main_folder):
        subfolder_path = os.path.join(main_folder, subfolder)
        if os.path.isdir(subfolder_path):
            print(f"Processing subfolder: {subfolder}")
            split_data(subfolder_path, dest_folder, subfolder)

main_folder = r"C:\Users\PlicEduard\AI4_SOC\classes\axx-bxo"
dest_prefolder = r"C:\Users\PlicEduard\AI4_SOC"

process_folders(main_folder, dest_prefolder)
