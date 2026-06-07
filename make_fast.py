from gatekeeper import export_to_onnx
import os

if __name__ == "__main__":
    model_path = "gatekeeper_best.pth"
    
    if os.path.exists(model_path):
        print(f"Found {model_path}. Converting to fast version (ONNX)...")
        try:
            export_to_onnx(model_path, "gatekeeper.onnx")
            print("Done! You now have 'gatekeeper.onnx'.")
        except Exception as e:
            print(f"Export failed: {e}")
    else:
        print(f"Error: {model_path} not found. Did you finish training?")
