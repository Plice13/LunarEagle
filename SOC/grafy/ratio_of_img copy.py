import cv2
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from random import randint, sample


from collections import Counter

def process_image(file_path):
    if file_path.lower().endswith(('.jpg', '.png')):
        img = cv2.imread(file_path)
        return img.shape[:2]

def main():
    image_folder_path = r'C:\Users\PlicEduard\ondrejov_base'
    list_of_shapes = list()

    list_of_file_names = os.listdir(image_folder_path)
    
    # Randomly sample a subset of files for processing
    files_to_process = list_of_file_names
    
    with ProcessPoolExecutor() as executor:
        for shape in tqdm(executor.map(process_image, [os.path.join(image_folder_path, file_name) for file_name in files_to_process]), total=len(files_to_process)):
            list_of_shapes.append(shape)

    heights, widths = zip(*list_of_shapes)

    
    d_heights = Counter(heights)
    print('d_heights',d_heights)
    
    d_widths = Counter(widths)
    print('d_widths',d_widths)

    
    # Create a bar graph for widths
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.hist(widths, bins=20, color='blue', alpha=0.7)
    plt.title('Widths Distribution')
    plt.xlabel('Width')
    plt.ylabel('Frequency')

    # Create a bar graph for heights
    plt.subplot(1, 2, 2)
    plt.hist(heights, bins=20, color='green', alpha=0.7)
    plt.title('Heights Distribution')
    plt.xlabel('Height')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
