from PIL import Image
import os

def resize_and_paste(folder_path):
    if not os.path.exists(folder_path):
        print("Folder path does not exist.")
        return

    image_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

    if not image_files:
        print("No image files found in the folder.")
        return

    for index, image_file in enumerate(image_files, 1):
        image_path = os.path.join(folder_path, image_file)
        img = Image.open(image_path)

        # Resize image to width 2000 while maintaining the aspect ratio
        basewidth = 2000
        wpercent = (basewidth / float(img.size[0]))
        hsize = int((float(img.size[1]) * float(wpercent)))
        img = img.resize((basewidth, hsize), Image.LANCZOS)

        # Create a white canvas of size 2000x1800
        background = Image.new('RGB', (2000, 1800), (255, 255, 255))

        # Calculate position to paste the resized image at the center
        paste_x = (2000 - img.size[0]) // 2
        paste_y = (1800 - img.size[1]) // 2

        # Paste the resized image onto the white canvas
        background.paste(img, (paste_x, paste_y))

        # Save the new image
        save_path = os.path.join(folder_path, f"resized_{image_file}")
        background.save(save_path)
        print(f"Image {index} processed and saved as {save_path}")

# Provide the path to the folder containing images
folder_path = "./"

resize_and_paste(folder_path)
