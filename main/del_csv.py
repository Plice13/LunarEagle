import csv

# Sample lists
list1 = [1, 1, 1, 1, 1]
list2 = [2]*5
list3 = [1, 2, 0, 1, 2]
list4 = [0]*5

# Combine lists into a list of tuples
combined_lists = zip(list1, list2, list3, list4)

# Define the file name
file_name = "data.csv"

# Write data to CSV file
with open(file_name, mode='w', newline='') as file:
    writer = csv.writer(file)
    # Write header
    writer.writerow(['Column1', 'Column2', 'Column3', 'Column4'])
    # Write data
    writer.writerows(combined_lists)

print("Data has been written to", file_name)
