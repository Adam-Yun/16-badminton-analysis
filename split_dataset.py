# Import os to work with files and directories
import os

from dotenv import load_dotenv

# Import shutil to move files
import shutil

# Import random to shuffle the dataset
import random

# Load variables from the .env file
load_dotenv()

# Folder containing your labeled images
image_folder = os.getenv("LABEL_IMAGES") 

# Folder containing label files
label_folder = os.getenv("LABELS") 

# Dataset folder
dataset_folder = os.getenv("DATASET")

# Create dataset directory structure

os.makedirs("dataset/images/train", exist_ok=True)
os.makedirs("dataset/images/val", exist_ok=True)
os.makedirs("dataset/labels/train", exist_ok=True)
os.makedirs("dataset/labels/val", exist_ok=True)


# Get all image filenames
images = [f for f in os.listdir(image_folder) if f.endswith(".jpg")]

# Shuffle images randomly
random.shuffle(images)

# Compute split index (80% train)
split_index = int(len(images) * 0.8)

# Split dataset
train_images = images[:split_index]
val_images = images[split_index:]


# Function to copy image + label

def copy_files(image_list, image_dest, label_dest):

    for img in image_list:

        # Corresponding label filename
        label_name = img.replace(".jpg", ".txt")

        # Full path to the label file
        label_src = os.path.join(label_folder, label_name)

        # Skip this image if it has no label
        if not os.path.exists(label_src):
            print("Skipping image without label:", img)
            continue

        # Source image path
        img_src = os.path.join(image_folder, img)

        # Destination image path
        img_dst = os.path.join(image_dest, img)

        # Copy the image
        shutil.copy(img_src, img_dst)

        # Destination label path
        label_dst = os.path.join(label_dest, label_name)

        # Copy the label file
        shutil.copy(label_src, label_dst)

# Copy training data
copy_files(train_images, "dataset/images/train", "dataset/labels/train")

# Copy validation data
copy_files(val_images, "dataset/images/val", "dataset/labels/val")


print("Dataset prepared successfully")