import os
import random
import shutil

def random_sample_files(source, num_samples):
    files = os.listdir(source)
    return random.sample(files, num_samples)

def split_data(source, dest_folder, class_name, num_samples, train_ratio=0.8, val_ratio=0.1):
    dest_train = os.path.join(dest_folder, 'train')
    dest_val = os.path.join(dest_folder, 'val')
    dest_test = os.path.join(dest_folder, 'test')
    sampled_files = random_sample_files(source, num_samples)

    # Shuffle the sampled files
    random.shuffle(sampled_files)

    # Calculate the split indices
    train_split = int(train_ratio * num_samples)
    val_split = train_split + int(val_ratio * num_samples)

    # Copy sampled files to train, val, and test directories
    for i, file in enumerate(sampled_files):
        source_file = os.path.join(source, file)
        if i < train_split:
            dest_folder = os.path.join(dest_train, class_name)
        elif i < val_split:
            dest_folder = os.path.join(dest_val, class_name)
        else:
            dest_folder = os.path.join(dest_test, class_name)

        if not os.path.exists(dest_folder):
            os.makedirs(dest_folder)

        dest_file = os.path.join(dest_folder, file)
        shutil.copyfile(source_file, dest_file)

def process_folders(main_folder, dest_prefolder):
    #get name of destination folder
    subfolders_list = []
    for subfolder in os.listdir(main_folder):
        subfolder_path = os.path.join(main_folder, subfolder)
        if os.path.isdir(subfolder_path):
            subfolders_list.append(subfolder)
        subfolders=('_'.join(subfolders_list))+f'_{num_samples_per_class}_{int(num_samples_per_class*0.1)}'
        dest_folder = os.path.join(dest_prefolder, subfolders)

    # run script    
    for subfolder in os.listdir(main_folder):
        subfolder_path = os.path.join(main_folder, subfolder)
        if os.path.isdir(subfolder_path):
            print(f"Processing subfolder: {subfolder}")
            split_data(subfolder_path, dest_folder, subfolder, num_samples_per_class)

main_folder = r"C:\Users\PlicEduard\AI2\classes\sdfghj"
num_samples_per_class = 260
dest_prefolder = r"C:\Users\PlicEduard\AI2"

process_folders(main_folder, dest_prefolder)
