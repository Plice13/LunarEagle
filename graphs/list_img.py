import os    


image_folder_path_1 = r'C:\Users\PlicEduard\ondrejov_base'
image_folder_path_2 = r'C:\Users\PlicEduard\ondrejov'
list_of_shapes = list()

list_of_file_names_1 = os.listdir(image_folder_path_1)
list_of_file_names_2 = os.listdir(image_folder_path_2)

for element in list_of_file_names_1:
    if element not in list_of_file_names_2:
        print(element)