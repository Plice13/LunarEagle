import os
from PIL import Image
import imagehash


def process_folders(main_folder):
    for subfolder in os.listdir(main_folder):
        subfolder_path = os.path.join(main_folder, subfolder)
        if os.path.isdir(subfolder_path):
            print(f"Processing subfolder: {subfolder}")
            process_subfolder(subfolder_path)

def process_subfolder(subfolder_path):
    image_folder = subfolder_path
    similar_pairs = find_similar_images(image_folder)

    if similar_pairs:
        delete_first_image_in_pairs(image_folder, similar_pairs)
    else:
        print(f"No similar pairs found in subfolder: {subfolder_path}")


def image_similarity(image1, image2):
    hash1 = imagehash.average_hash(Image.open(image1))
    hash2 = imagehash.average_hash(Image.open(image2))
    return hash1 - hash2

def find_similar_images(image_folder):
    image_files = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]
    similar_pairs = []

    for i, base_filename in enumerate(image_files):
        base_image_path = os.path.join(image_folder, base_filename)
        base_date = base_filename[:14]  # Extract the date from the filename

        for compare_filename in image_files[i + 1:]:
            compare_image_path = os.path.join(image_folder, compare_filename)
            compare_date = compare_filename[:14]  # Extract the date from the filename

            if base_date == compare_date:
                similarity = image_similarity(base_image_path, compare_image_path)

                # Adjust the threshold as needed
                threshold = 5
                if similarity < threshold:
                    similar_pairs.append((base_filename, compare_filename))

    return similar_pairs

def delete_first_image_in_pairs(image_folder, similar_pairs):
    for base_filename, compare_filename in similar_pairs:
        base_image_path = os.path.join(image_folder, base_filename)
        os.remove(base_image_path)
        print(f"The image {base_filename.split('__')[0]} has been deleted.")

# Replace 'path_to_image_folder' with the actual path
main_folder = r"C:\Users\PlicEduard\clasifics\znovu_znovu"

process_folders(main_folder)

