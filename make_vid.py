import cv2
import os
from tqdm import tqdm  # Importujeme tqdm pro progress bar

image_folder = r'C:\Users\PlicEduard\ondrejov\extracted_rectangles'
video_name = 'video.mp4'

images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, 0, 60, (width, height))

# Použijeme tqdm k vytvoření progress baru
for image in tqdm(images, desc='Vytváření videa', unit='snímek'):
    video.write(cv2.imread(os.path.join(image_folder, image)))

cv2.destroyAllWindows()
video.release()
