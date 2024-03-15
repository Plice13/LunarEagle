import os

def folder_summary(folder_path):
    subfolders = [name for name in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, name))]
    subfolder_names = ', '.join(subfolders)

    item_counts = []
    total_items = 0
    for subfolder in subfolders:
        subfolder_path = os.path.join(folder_path, subfolder)
        items = len(os.listdir(subfolder_path))
        item_counts.append(str(items))
        total_items += items

    item_counts_str = ', '.join(item_counts)
    total_items_str = str(total_items)

    return subfolder_names, item_counts_str, total_items_str

folder_path = r'C:\Users\PlicEduard\AI4_SOC\classes\ciox'

subfolder_names, item_counts_str, total_items_str = folder_summary(folder_path)
print("Subfolder Names:", subfolder_names)
print("Item Counts:", item_counts_str)
print("Total Items:", total_items_str)
