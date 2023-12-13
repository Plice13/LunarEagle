import os
import random
import shutil

def random_sample_files(source, num_samples):
    files = os.listdir(source)
    return random.sample(files, num_samples)

def split_data(source, dest_train, dest_val, dest_test, class_name, num_samples=1600, train_ratio=0.8, val_ratio=0.1):
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

# Example usage
source_folder_class1 = r"C:\Users\PlicEduard\AI2\classes\none\Axx"
source_folder_class2 = r"C:\Users\PlicEduard\AI2\classes\none\Bxo"
dest_folder_train = r"C:\Users\PlicEduard\AI2\Axx_Bxx_1600_160\train"
dest_folder_val = r"C:\Users\PlicEduard\AI2\Axx_Bxx_1600_160\val"
dest_folder_test = r"C:\Users\PlicEduard\AI2\Axx_Bxx_1600_160\test"

# Set the number of samples you want to randomly select
num_samples_per_class = 1600

# Randomly sample and split data into train, val, and test sets
split_data(source_folder_class1, dest_folder_train, dest_folder_val, dest_folder_test, "Axx", num_samples_per_class)
split_data(source_folder_class2, dest_folder_train, dest_folder_val, dest_folder_test, "Bxo", num_samples_per_class)
