# Enable CPU fallback for operations not supported on Apple MPS (like NMS)
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Import OpenCV for reading video frames
import cv2

# Import YOLO model from Ultralytics
from ultralytics import YOLO

# Import dotenv to load environment variables
from dotenv import load_dotenv

# Load variables from the .env file
load_dotenv()

# Get the path to the badminton video from the environment
video_path = os.getenv("VIDEO_PATH")

# Load the trained shuttlecock detection model
model = YOLO("models/weights/best.pt")

# Open the video file
cap = cv2.VideoCapture(video_path)

# Frame counter
frame_id = 0

# Skip frames to speed up processing (process every 3rd frame)
frame_skip = 3

# Number of frames to process together (GPU batching)
batch_size = 8

# Temporary list to hold frames for batch processing
frame_batch = []

# Get the current working directory so outputs save here
current_dir = os.getcwd()

while True:

    # Read the next frame from the video
    ret, frame = cap.read()

    # Stop if video has ended
    if not ret:
        break

    # Skip frames for speed
    if frame_id % frame_skip != 0:
        frame_id += 1
        continue

    # Resize frame to reduce computation
    frame = cv2.resize(frame, (640, 360))

    # Add frame to the batch
    frame_batch.append(frame)

    # If batch is full → run detection
    if len(frame_batch) == batch_size:

        # Run detection on all frames in the batch
        results = model(
            frame_batch,
            device="mps",     # Use Apple GPU
            conf=0.25         # Detection confidence threshold
        )

        # Process results frame by frame
        for r in results:

            # Print detected boxes (for debugging)
            print(r.boxes)

        # Clear batch after processing
        frame_batch = []

    # Move to next frame
    frame_id += 1

# Release the video file
cap.release()

print("Detection complete.")