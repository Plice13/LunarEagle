import os
import random
import shutil

def split_data(source, dest_folder, class_name, samples_for_test, train_ratio=0.89, val_ratio=0.11):
    dest_train = os.path.join(dest_folder, 'train', class_name)
    dest_val = os.path.join(dest_folder, 'val', class_name)
    dest_test = os.path.join(dest_folder, 'test', class_name)

    # Get all files in the source folder
    files = os.listdir(source)

    # Shuffle the files randomly
    random.shuffle(files)

    # Calculate the split indices
    total_samples_for_train_and_val = len(files)-samples_for_test
    train_split = int(train_ratio * total_samples_for_train_and_val)
    val_split = train_split + int(val_ratio * total_samples_for_train_and_val)

    for i, file in enumerate(files):
        source_file = os.path.join(source, file)
        if i < samples_for_test:
            dest_folder = dest_test
        elif i < samples_for_test+train_split:
            dest_folder = dest_train
        else:
            dest_folder = dest_val

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

    subfolders = '_'.join(subfolders_list)
    dest_folder = os.path.join(dest_prefolder, subfolders)

    numbers_of_samples = []
    #Getting number of all samples
    for subfolder in subfolders_list:
        numbers_of_samples.append(len(os.listdir(os.path.join(main_folder, subfolder))))
    print(numbers_of_samples)
    samples_for_test = min(numbers_of_samples)/10

    # Run script
    for subfolder in os.listdir(main_folder):
        subfolder_path = os.path.join(main_folder, subfolder)
        if os.path.isdir(subfolder_path):
            print(f"Processing subfolder: {subfolder}")
            split_data(subfolder_path, dest_folder, subfolder, samples_for_test)

main_folder = r"C:\Users\PlicEduard\AI4_SOC\classes\xrshak"
dest_prefolder = r"C:\Users\PlicEduard\AI4_SOC"

process_folders(main_folder, dest_prefolder)
