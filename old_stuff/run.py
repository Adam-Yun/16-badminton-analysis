# Import the os module to work with folders and files
import os

# Path to the folder you want to count files in
folder_path = "label_frames"

# Get a list of all items (files and folders) inside the folder
items = os.listdir(folder_path)

# Count how many items are in the list
file_count = len(items)

# Print the number of items found in the folder
print("Number of files:", file_count)