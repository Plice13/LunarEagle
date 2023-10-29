import os
from PIL import Image

def get_image_dimensions(folder_path):
    widths = []
    heights = []

    for filename in os.listdir(folder_path):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.gif')):
            with Image.open(os.path.join(folder_path, filename)) as img:
                width, height = img.size
                widths.append(width)
                heights.append(height)

    return widths, heights

def calculate_mean(data):
    return sum(data) / len(data) if len(data) > 0 else 0

def calculate_median(data):
    sorted_data = sorted(data)
    data_len = len(sorted_data)
    if data_len % 2 == 0:
        return (sorted_data[data_len // 2 - 1] + sorted_data[data_len // 2]) / 2
    else:
        return sorted_data[data_len // 2]

folder_path = r'C:\Users\PlicEduard\ondrejov'

widths, heights = get_image_dimensions(folder_path)

mean_width = calculate_mean(widths)
mean_height = calculate_mean(heights)

median_width = calculate_median(widths)
median_height = calculate_median(heights)

print(f"Mean Width: {mean_width}, Mean Height: {mean_height}")
print(f"Median Width: {median_width}, Median Height: {median_height}")
