import os

folder_path = r'C:\Users\PlicEduard\AI\more\classes\Ekc'

# Check if the folder exists
if not os.path.exists(folder_path):
    print(f"The folder '{folder_path}' does not exist.")
else:
    # List all files in the folder
    files = os.listdir(folder_path)

    # Iterate through each file and rename it
    for file_name in files:
        if file_name.startswith('1') or file_name.startswith('2'):
            # Create the new file name
            new_file_name = f'Ekc-{file_name}'

            # Construct the full paths for the old and new names
            old_path = os.path.join(folder_path, file_name)
            new_path = os.path.join(folder_path, new_file_name)

            # Rename the file
            os.rename(old_path, new_path)

            print(f'Renamed: {file_name} -> {new_file_name}')

print('Renaming complete.')
