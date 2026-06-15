# Evaluates the trained rally classifier on a folder of test images, printing
# the per-image RALLY / NON-RALLY prediction and confidence, plus totals at the
# end. Uses buffer_size=1 so each image is judged independently.

import cv2
import os
import torch
from dotenv import load_dotenv
from classify_rally.classify_rally import RallyClassifier

load_dotenv()

# 1. Setup - Point to your best model (from .env)
MODEL_PATH = os.getenv("CLASSIFY_RALLY")
# Point to a folder of images you want to test
TEST_FOLDER = "test_rally_scene"

def test_on_images(folder_path):
    # Initialize the classifier
    # We set buffer_size=1 so it judges every image individually without 'memory'
    classifier = RallyClassifier(MODEL_PATH, buffer_size=1)
    
    if not os.path.exists(folder_path):
        print(f"Error: Folder '{folder_path}' not found.")
        return

    # Get a list of images in the folder
    images = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
    images.sort() # Sort them so they are in order

    if not images:
        print(f"No images found in {folder_path}")
        return

    print(f"{'Image Name':<30} | {'AI Prediction':<15} | {'Confidence':<10}")
    print("-" * 65)

    rally_count = 0
    non_rally_count = 0

    for img_name in images:
        img_path = os.path.join(folder_path, img_name)
        frame = cv2.imread(img_path)
        
        if frame is None:
            continue
            
        is_rally, confidence = classifier.predict(frame)
        
        if is_rally:
            rally_count += 1
            prediction = "🎾 RALLY"
        else:
            non_rally_count += 1
            prediction = "☕ NON-RALLY"
            
        print(f"{img_name:<30} | {prediction:<15} | {confidence*100:>8.2f}%")

    print("-" * 50)
    print(f"TEST COMPLETE")
    print(f"Total Images: {len(images)}")
    print(f"Total Rally: {rally_count}")
    print(f"Total Non-Rally: {non_rally_count}")

if __name__ == "__main__":
    if os.path.exists(MODEL_PATH):
        test_on_images(TEST_FOLDER)
    else:
        print(f"Error: {MODEL_PATH} not found. Please train the model first.")
