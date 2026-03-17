# Import the YOLO model class
from ultralytics import YOLO

# Load your trained model weights
model = YOLO("models/weights/best.pt")

# Run detection on a sample image
# results = model("dataset/images/val/frame_200.jpg")
results = model("frames/frame_100.jpg")

# Show the detection result in a window
results[0].show()

# Loop through each detection result
for result in results:

    # Get bounding boxes detected in this image
    boxes = result.boxes

    # Loop through each detected object
    for box in boxes:

        # Get bounding box coordinates (x1, y1, x2, y2)
        x1, y1, x2, y2 = box.xyxy[0]

        # Get confidence score of the detection
        confidence = box.conf[0]

        # Get class ID
        class_id = int(box.cls[0])

        # Get class name using the model's class dictionary
        class_name = model.names[class_id]

        # Print all the information
        print("Detected object:")
        print("  Class:", class_name)
        print("  Confidence:", float(confidence))
        print("  Bounding box:", float(x1), float(y1), float(x2), float(y2))
        print()