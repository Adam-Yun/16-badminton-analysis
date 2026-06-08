# Trains the Gatekeeper rally classifier on the images under DATASET_PATH
# (expects dataset/train and dataset/val subfolders), saves the best weights as
# gatekeeper_best.pth, and then exports an ONNX copy for fast inference.

from gatekeeper import train_gatekeeper, export_to_onnx
import os

# Point this to your dataset folder
DATASET_PATH = "dataset" 

if __name__ == "__main__":
    if not os.path.exists(DATASET_PATH):
        print(f"Error: Folder '{DATASET_PATH}' not found. Please make sure your images are in 'dataset/train' and 'dataset/val'.")
    else:
        print("--- Starting Gatekeeper Training ---")
        
        # This runs the "Schooling" process
        # num_epochs=15 means it will look at your data 15 times
        try:
            model = train_gatekeeper(DATASET_PATH, num_epochs=15)
            
            print("\n--- Training Complete! ---")
            print("Your model is saved as: gatekeeper_best.pth")

            # Create the 'Fast' version (ONNX) immediately
            if os.path.exists("gatekeeper_best.pth"):
                print("Exporting to ONNX...")
                export_to_onnx("gatekeeper_best.pth", "gatekeeper.onnx")
        except Exception as e:
            print(f"An error occurred during training: {e}")
