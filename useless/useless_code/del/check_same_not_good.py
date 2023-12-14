import os
from PIL import Image, ImageTk
import imagehash
import tkinter as tk
from tkinter import messagebox

def image_similarity(image1, image2):
    hash1 = imagehash.average_hash(Image.open(image1))
    hash2 = imagehash.average_hash(Image.open(image2))
    return hash1 - hash2

class ImageComparisonApp:
    def __init__(self, root, image_folder, pairs):
        self.root = root
        self.image_folder = image_folder
        self.pairs = pairs
        self.current_index = 0
        self.threshold = 5

        # Create GUI components
        self.image_label_1 = tk.Label(root)
        self.image_label_1.pack(side=tk.LEFT, padx=5)

        self.image_label_2 = tk.Label(root)
        self.image_label_2.pack(side=tk.RIGHT, padx=5)

        self.next_button = tk.Button(root, text="Next", command=self.next_pair)
        self.next_button.pack()

        self.delete_button = tk.Button(root, text="Delete", command=self.delete_pair)
        self.delete_button.pack()

        # Load the first pair after creating GUI components
        self.load_pair()

    def load_pair(self):
        if self.current_index < len(self.pairs):
            base_filename, compare_filename = self.pairs[self.current_index]
            base_image_path = os.path.join(self.image_folder, base_filename)
            compare_image_path = os.path.join(self.image_folder, compare_filename)

            print(f"Loading pair: {base_image_path}, {compare_image_path}")

            pil_image_1 = Image.open(base_image_path)
            pil_image_2 = Image.open(compare_image_path)

            self.tk_image_1 = ImageTk.PhotoImage(pil_image_1)
            self.tk_image_2 = ImageTk.PhotoImage(pil_image_2)

            self.image_label_1.configure(image=self.tk_image_1)
            self.image_label_2.configure(image=self.tk_image_2)
        else:
            # All pairs have been processed
            self.tk_image_1 = None
            self.tk_image_2 = None
            self.image_label_1.configure(image=None)
            self.image_label_2.configure(image=None)
            self.next_button.configure(state=tk.DISABLED)
            self.delete_button.configure(state=tk.DISABLED)

    def next_pair(self):
        # Move to the next pair of similar images
        self.current_index += 1
        self.load_pair()

    def delete_pair(self):
        if self.current_index < len(self.pairs):
            base_filename, compare_filename = self.pairs[self.current_index]
            base_image_path = os.path.join(self.image_folder, base_filename)
            compare_image_path = os.path.join(self.image_folder, compare_filename)

            response = messagebox.askyesno("Delete Images", f"Do you want to delete the images:\n{base_filename}\n{compare_filename}?")
            if response:
                os.remove(base_image_path)
                os.remove(compare_image_path)
                messagebox.showinfo("Deleted", f"The images {base_filename} and {compare_filename} have been deleted.")
                self.load_pair()
        else:
            # All pairs have been processed
            self.tk_image_1 = None
            self.tk_image_2 = None
            self.image_label_1.configure(image=None)
            self.image_label_2.configure(image=None)
            self.next_button.configure(state=tk.DISABLED)
            self.delete_button.configure(state=tk.DISABLED)

def load_pair(self):
    if self.current_index < len(self.pairs):
        base_filename, compare_filename = self.pairs[self.current_index]
        base_image_path = os.path.join(self.image_folder, base_filename)
        compare_image_path = os.path.join(self.image_folder, compare_filename)

        print(f"Loading pair: {base_image_path}, {compare_image_path}")

        pil_image_1 = Image.open(base_image_path)
        pil_image_2 = Image.open(compare_image_path)

        self.tk_image_1 = ImageTk.PhotoImage(pil_image_1)
        self.tk_image_2 = ImageTk.PhotoImage(pil_image_2)

        self.image_label_1.configure(image=self.tk_image_1)
        self.image_label_2.configure(image=self.tk_image_2)
    else:
        # All pairs have been processed
        self.tk_image_1 = None
        self.tk_image_2 = None
        self.image_label_1.configure(image=None)
        self.image_label_2.configure(image=None)
        self.next_button.configure(state=tk.DISABLED)
        self.delete_button.configure(state=tk.DISABLED)


    def next_pair(self):
        # Move to the next pair of similar images
        self.current_index += 1
        self.load_pair()

    def delete_pair(self):
        if self.current_index < len(self.pairs):
            base_filename, compare_filename = self.pairs[self.current_index]
            base_image_path = os.path.join(self.image_folder, base_filename)
            compare_image_path = os.path.join(self.image_folder, compare_filename)

            response = messagebox.askyesno("Delete Images", f"Do you want to delete the images:\n{base_filename}\n{compare_filename}?")
            if response:
                os.remove(base_image_path)
                os.remove(compare_image_path)
                messagebox.showinfo("Deleted", f"The images {base_filename} and {compare_filename} have been deleted.")
                self.load_pair()



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


image_folder = r"C:\Users\PlicEduard\AI2\classes\none\znovu\Hsx"
similar_pairs = find_similar_images(image_folder)

if similar_pairs:
    root = tk.Tk()
    app = ImageComparisonApp(root, image_folder, similar_pairs)
    root.mainloop()
else:
    print("No similar pairs found.")