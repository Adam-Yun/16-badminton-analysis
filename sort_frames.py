"""
Sort Frames Script
-----------------
This script uses the 'Gatekeeper' model to automatically classify video frames
into 'rally' and 'non_rally' categories. It's designed to help filter out
irrelevant footage (like players walking between points) from actual play.

Usage:
    python sort_frames.py <input_folder> [options]

Examples:
    1. Basic usage (copies files to 'sorted_frames'):
       python sort_frames.py ./my_video_frames

    2. Specify output folder and model path:
       python sort_frames.py ./input_frames -o ./classified -m gatekeeper_best.pth

    3. Move files instead of copying (faster, but modifies source):
       python sort_frames.py ./input_frames --move

    4. Adjust sensitivity (threshold 0.7 for more strict 'rally' classification):
       python sort_frames.py ./input_frames -t 0.7

Dependencies:
    - torch, torchvision, PIL, gatekeeper.py
"""

import argparse
import shutil
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

# Import the model architecture from the local gatekeeper module
from gatekeeper import get_model

# Supported image formats
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_model(model_path, device):
    """Loads the trained model weights into the architecture."""
    model = get_model(num_classes=2, pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def build_transform():
    """Standard ImageNet transforms for the model."""
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def iter_images(folder):
    """Generator to yield valid image paths from a folder."""
    for path in sorted(Path(folder).iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            yield path


def main():
    # --- Argument Parsing ---
    parser = argparse.ArgumentParser(description="Sort frames into rally / non_rally folders using the gatekeeper model.")
    parser.add_argument("input", help="Folder containing frames to classify.")
    parser.add_argument("-o", "--output", default="sorted_frames", help="Output folder (default: sorted_frames).")
    parser.add_argument("-m", "--model", default="gatekeeper_best.pth", help="Path to model weights.")
    parser.add_argument("-b", "--batch-size", type=int, default=32, help="Batch size for inference.")
    parser.add_argument("-t", "--threshold", type=float, default=0.5, help="Min rally probability to label as rally.")
    parser.add_argument("--move", action="store_true", help="Move files instead of copying.")
    args = parser.parse_args()

    # --- Setup ---
    input_dir = Path(args.input)
    if not input_dir.is_dir():
        raise SystemExit(f"Input folder not found: {input_dir}")

    # Prepare output directories
    rally_dir = Path(args.output) / "rally"
    non_rally_dir = Path(args.output) / "non_rally"
    rally_dir.mkdir(parents=True, exist_ok=True)
    non_rally_dir.mkdir(parents=True, exist_ok=True)

    # Device selection (CUDA/MPS/CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Using device: {device}")

    # Initialize model and transforms
    model = load_model(args.model, device)
    transform = build_transform()

    # Collect images
    image_paths = list(iter_images(input_dir))
    if not image_paths:
        raise SystemExit(f"No images found in {input_dir}")
    print(f"Found {len(image_paths)} images. Classifying...")

    # Choose between copy and move
    transfer = shutil.move if args.move else shutil.copy2

    # Stats tracking
    rally_count = 0
    non_rally_count = 0
    skipped = 0

    # Batch processing state
    batch_tensors = []
    batch_paths = []

    def flush():
        """Processes the current batch through the model and moves/copies files."""
        nonlocal rally_count, non_rally_count
        if not batch_tensors:
            return
        
        # Stack images into a single tensor and move to device
        batch = torch.stack(batch_tensors).to(device)
        
        # Inference
        with torch.no_grad():
            outputs = model(batch)
            probs = torch.softmax(outputs, dim=1)
        
        # Class 1 is usually 'rally' in our 2-class setup
        rally_probs = probs[:, 1].cpu().tolist()
        
        # Action based on probability
        for path, p in zip(batch_paths, rally_probs):
            if p >= args.threshold:
                transfer(str(path), str(rally_dir / path.name))
                rally_count += 1
            else:
                transfer(str(path), str(non_rally_dir / path.name))
                non_rally_count += 1
        
        batch_tensors.clear()
        batch_paths.clear()

    # --- Main Loop ---
    for i, path in enumerate(image_paths, start=1):
        try:
            # Load and preprocess image
            img = Image.open(path).convert("RGB")
        except Exception as e:
            print(f"  skip {path.name}: {e}")
            skipped += 1
            continue
            
        batch_tensors.append(transform(img))
        batch_paths.append(path)
        
        # Flush batch when it reaches capacity
        if len(batch_tensors) >= args.batch_size:
            flush()
            print(f"  processed {i}/{len(image_paths)}")
    
    # Final flush for remaining images
    flush()

    # --- Summary ---
    print("\nDone.")
    print(f"  rally:     {rally_count}  -> {rally_dir}")
    print(f"  non_rally: {non_rally_count}  -> {non_rally_dir}")
    if skipped:
        print(f"  skipped:   {skipped}")


if __name__ == "__main__":
    main()
