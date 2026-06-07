# Import os to work with folders and file paths
import os

# Import shutil to copy files
import shutil

# Folder containing all extracted frames
source_folder = "frames"

# Folder where sampled frames will be saved
output_folder = "label_frames"

# Create the output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Get a list of files inside the frames folder
frames = os.listdir(source_folder)

# Print how many files we found
print("Total files found:", len(frames))

# Sort frames based on the number inside the filename
frames = sorted(
    frames,
    key=lambda x: int(x.split("_")[1].split(".")[0])
)

# Number of frames to skip
step = 50

# Counter for how many frames we copy
copied_count = 0

# Loop through frames using the step value
for i in range(0, step*200, step):

    # Get the selected frame filename
    frame_name = frames[i]

    # Build the full source path
    src_path = os.path.join(source_folder, frame_name)

    # Build the destination path
    dst_path = os.path.join(output_folder, frame_name)

    # Copy the file
    shutil.copy(src_path, dst_path)

    # Increase copied counter
    copied_count += 1

# Print how many frames were copied
print("Frames copied:", copied_count)