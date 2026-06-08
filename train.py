# Trains the rally classifier on the images under DATASET
# (expects <DATASET>/train and <DATASET>/val subfolders), saves the best weights
# to TRAIN_MODEL_PATH, and exports an ONNX copy to TRAIN_ONNX_PATH.

import os
import torch
from dotenv import load_dotenv
from classify import train_classifier, export_to_onnx

load_dotenv()

DATASET_PATH = os.getenv("DATASET")
MODEL_PATH = os.getenv("TRAIN_MODEL_PATH", "gatekeeper_best.pth")
ONNX_PATH = os.getenv("TRAIN_ONNX_PATH", "gatekeeper.onnx")
NUM_EPOCHS = int(os.getenv("TRAIN_EPOCHS", "15"))

if __name__ == "__main__":
    if not DATASET_PATH:
        raise SystemExit("DATASET not set in .env")
    if not os.path.exists(DATASET_PATH):
        print(f"Error: Folder '{DATASET_PATH}' not found. Please make sure your images are in '{DATASET_PATH}/train' and '{DATASET_PATH}/val'.")
    else:
        if torch.cuda.is_available():
            print(f"CUDA available — using GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("CUDA not available — training on CPU (this will be slow)")
        print("--- Starting Classifier Training ---")
        try:
            model = train_classifier(DATASET_PATH, num_epochs=NUM_EPOCHS)

            print("\n--- Training Complete! ---")
            print(f"Your model is saved as: {MODEL_PATH}")

            if os.path.exists(MODEL_PATH):
                print("Exporting to ONNX...")
                export_to_onnx(MODEL_PATH, ONNX_PATH)
        except Exception as e:
            print(f"An error occurred during training: {e}")
