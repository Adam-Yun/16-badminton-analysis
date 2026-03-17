# Import dotenv to load environment variables
from dotenv import load_dotenv

# Import os for environment variables and working directory
import os

# Enable fallback to CPU for operations unsupported on Apple MPSDoe
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Import YOLO model (needs to be below mps fallback)
from ultralytics import YOLO

# Load variables from the .env file
load_dotenv()

# Get the path to the badminton video from the environment
video_path = os.getenv("VIDEO_PATH")

# Load the trained shuttlecock detection model
model = YOLO("models/weights/best.pt")

# Get the current working directory
current_dir = os.getcwd()

# Run detection on the match video
results = model.predict(
    source=video_path,      # video file to process
    save=True,              # save output video
    conf=0.25,              # detection confidence threshold
    stream=True,            # process frames sequentially
    vid_stride=1,           # process every 3rd frame
    device="mps",           # use Apple GPU
    project=current_dir,    # save results in the current directory
    name="detections"       # folder name for this run
)

# Iterate through results (keeps streaming active)
for r in results:
    pass