import os

def count_items_in_subfolders(folder):
    subfolders = [f.path for f in os.scandir(folder) if f.is_dir()]
    subfolder_item_counts = {}
    
    for subfolder in subfolders:
        count = len([name for name in os.listdir(subfolder) if os.path.isfile(os.path.join(subfolder, name))])
        subfolder_item_counts[subfolder] = count
    
    return subfolder_item_counts

def get_subfolder_with_lowest_item_count(subfolder_item_counts):
    min_count = min(subfolder_item_counts.values())
    min_subfolders = [subfolder for subfolder, count in subfolder_item_counts.items() if count == min_count]
    return min_subfolders[0]

def delete_extra_items(subfolder_item_counts, target_subfolder):
    for subfolder, count in subfolder_item_counts.items():
        if subfolder != target_subfolder:
            extra_items = count - subfolder_item_counts[target_subfolder]
            if extra_items > 0:
                files_to_delete = sorted(os.listdir(subfolder))[:extra_items]
                for file in files_to_delete:
                    os.remove(os.path.join(subfolder, file))

def main(folder):
    subfolder_item_counts = count_items_in_subfolders(folder)
    target_subfolder = get_subfolder_with_lowest_item_count(subfolder_item_counts)
    delete_extra_items(subfolder_item_counts, target_subfolder)
    print("Items adjusted successfully.")

if __name__ == "__main__":
    folder = r'C:\Users\PlicEduard\AI3_full_circle\a_h_k_r_s_x_0_0\test2'
    main(folder)