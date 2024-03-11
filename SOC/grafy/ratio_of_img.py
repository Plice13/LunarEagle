import cv2
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from random import randint

image_folder_path = r'C:\Users\PlicEduard\ondrejov_base'
list_of_shapes = list()

list_of_file_names =  os.listdir(image_folder_path)
for file_name in tqdm(list_of_file_names):
    if True:
    #if randint(0,213) == 13:
        if file_name.lower().endswith(('.jpg','.png')): # pokud se jedná o obrázek
            img = cv2.imread(os.path.join(image_folder_path, file_name))
            list_of_shapes.append((img.shape[:2]))
print(list_of_shapes)

widths, heights = zip(*list_of_shapes)

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