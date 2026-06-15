# Converts the trained PyTorch rally classifier (CLASSIFY_RALLY) into an
# ONNX file (TRAIN_ONNX_PATH) for faster inference at runtime.

import os

from dotenv import load_dotenv

from classify_rally.classify_rally import export_to_onnx

load_dotenv()

if __name__ == "__main__":
    model_path = os.getenv("CLASSIFY_RALLY")
    onnx_path = os.getenv("TRAIN_ONNX_PATH", "gatekeeper.onnx")

    if not model_path:
        raise SystemExit("CLASSIFY_RALLY not set in .env")

    if os.path.exists(model_path):
        print(f"Found {model_path}. Converting to fast version (ONNX)...")
        try:
            export_to_onnx(model_path, onnx_path)
            print(f"Done! You now have '{onnx_path}'.")
        except Exception as e:
            print(f"Export failed: {e}")
    else:
        print(f"Error: {model_path} not found. Did you finish training?")
