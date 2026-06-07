# Import the OpenCV library for video and image processing
import cv2
from dotenv import load_dotenv
import os
# Import tqdm to display a progress bar in the terminal
from tqdm import tqdm

# Load variables from the .env file
load_dotenv()

# Path to the badminton video file
video_path = os.getenv("VIDEO_PATH")

# Folder where extracted frames will be saved
output_folder = "frames"

# Create the folder if it does not already exist
os.makedirs(output_folder, exist_ok=True)

# Open the video file so we can read frames from it
cap = cv2.VideoCapture(video_path)

# Get the total number of frames in the video
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Counter to keep track of the frame number
frame_id = 0

# Create a progress bar using the total number of frames
with tqdm(total=total_frames, desc="Extracting Frames") as pbar:

    # Loop through the video frame by frame
    while True:

        # Read the next frame from the video
        ret, frame = cap.read()

        # Stop the loop if no more frames are available
        if not ret:
            break

        # Save the frame as an image file
        cv2.imwrite(f"frames/frame_{frame_id}.jpg", frame)

        # Increase the frame counter
        frame_id += 1

        # Update the progress bar by 1 step
        pbar.update(1)

# Release the video object to free system resources
cap.release()