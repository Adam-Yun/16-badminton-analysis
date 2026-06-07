# Import os so we can get the current working directory
import os

# Import YOLO model from the ultralytics library
from ultralytics import YOLO

# Load the pretrained YOLOv8 nano model
model = YOLO("yolov8n.pt")

# Get the current working directory where the script is running
current_dir = os.getcwd()

# Train the model
model.train(
    data="dataset.yaml",        # Path to dataset configuration
    epochs=50,                  # Number of training epochs
    imgsz=640,                  # Image size used for training
    project=current_dir,        # Save results in the current working directory
    name="badminton_detector"   # Name of the training run folder
)