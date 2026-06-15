"""
Sort Frames Script
-----------------
For each subfolder in SORT_INPUT, classifies its frames into rally / non_rally
using the rally classifier model. Output mirrors the input structure under SORT_OUTPUT,
and a combined train/val split is also written to
DATASET/{train,val}/{rally,non_rally} ready for use by train.py.

The train/val split is done at the VIDEO level, not the frame level — whole
subfolders go entirely to train or entirely to val. This prevents near-duplicate
neighboring frames from leaking between train and val.

Set SORT_INPUT, SORT_OUTPUT, DATASET, and CLASSIFY_RALLY in .env, then run:
    python sort.py
"""

import os
import shutil
from pathlib import Path

import torch
from dotenv import load_dotenv
from PIL import Image
from torchvision import transforms, models

load_dotenv()

MODEL_PATH = os.getenv("CLASSIFY_RALLY")
BATCH_SIZE = 32
THRESHOLD = 0.5
VAL_FRACTION = 0.2  # fraction of subfolders (videos) reserved for val
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def get_model(num_classes=2):
    return models.mobilenet_v3_small(weights=None, num_classes=num_classes)


def load_model(model_path, device):
    model = get_model(num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def build_transform():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def iter_images(folder):
    for path in sorted(Path(folder).iterdir()):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
            yield path


def process_folder(folder, split, output_root, dataset_root, model, transform, device):
    rally_dir = output_root / folder.name / "rally"
    non_rally_dir = output_root / folder.name / "non_rally"
    rally_dir.mkdir(parents=True, exist_ok=True)
    non_rally_dir.mkdir(parents=True, exist_ok=True)

    dataset_rally_dir = dataset_root / split / "rally"
    dataset_non_rally_dir = dataset_root / split / "non_rally"

    image_paths = list(iter_images(folder))
    if not image_paths:
        print(f"  [{folder.name}] ({split}) no images, skipping")
        return 0, 0, 0

    print(f"\n[{folder.name}] ({split}) {len(image_paths)} images")

    scored = []
    skipped = 0
    batch_tensors = []
    batch_paths = []

    def flush():
        if not batch_tensors:
            return
        batch = torch.stack(batch_tensors).to(device)
        with torch.no_grad():
            outputs = model(batch)
            probs = torch.softmax(outputs, dim=1)
        rally_probs = probs[:, 1].cpu().tolist()
        scored.extend(zip(batch_paths, rally_probs))
        batch_tensors.clear()
        batch_paths.clear()

    for i, path in enumerate(image_paths, start=1):
        try:
            img = Image.open(path).convert("RGB")
        except Exception as e:
            print(f"  skip {path.name}: {e}")
            skipped += 1
            continue

        batch_tensors.append(transform(img))
        batch_paths.append(path)

        if len(batch_tensors) >= BATCH_SIZE:
            flush()
            print(f"  processed {i}/{len(image_paths)}")

    flush()

    # Split by threshold, sort by confidence (most-confident first), balance counts.
    rally = [(p, prob) for p, prob in scored if prob >= THRESHOLD]
    non_rally = [(p, prob) for p, prob in scored if prob < THRESHOLD]
    rally.sort(key=lambda x: x[1], reverse=True)
    non_rally.sort(key=lambda x: x[1])

    keep = min(len(rally), len(non_rally))
    # Prefix combined-dataset filenames with the source folder to avoid collisions.
    # All of this folder's frames go to a single split (train OR val).
    for p, _ in rally[:keep]:
        shutil.copy2(str(p), str(rally_dir / p.name))
        shutil.copy2(str(p), str(dataset_rally_dir / f"{folder.name}_{p.name}"))
    for p, _ in non_rally[:keep]:
        shutil.copy2(str(p), str(non_rally_dir / p.name))
        shutil.copy2(str(p), str(dataset_non_rally_dir / f"{folder.name}_{p.name}"))

    dropped = (len(rally) - keep) + (len(non_rally) - keep)
    print(
        f"  rally: {keep}  non_rally: {keep}  -> {split}  "
        f"dropped (to balance): {dropped}" + (f"  skipped: {skipped}" if skipped else "")
    )
    return keep, keep, skipped


def main():
    input_path = os.getenv("SORT_INPUT")
    output_path = os.getenv("SORT_OUTPUT")
    dataset_path = os.getenv("DATASET")
    if not input_path:
        raise SystemExit("SORT_INPUT not set in .env")
    if not output_path:
        raise SystemExit("SORT_OUTPUT not set in .env")
    if not dataset_path:
        raise SystemExit("DATASET not set in .env")

    input_dir = Path(input_path)
    if not input_dir.is_dir():
        raise SystemExit(f"Input folder not found: {input_dir}")

    output_root = Path(output_path)
    output_root.mkdir(parents=True, exist_ok=True)

    dataset_root = Path(dataset_path)
    for split in ("train", "val"):
        for cls in ("rally", "non_rally"):
            (dataset_root / split / cls).mkdir(parents=True, exist_ok=True)

    subfolders = sorted([p for p in input_dir.iterdir() if p.is_dir()])
    if not subfolders:
        raise SystemExit(f"No subfolders found in {input_dir}")

    # Video-disjoint split: whole subfolders go to train OR val, never both.
    # This prevents near-duplicate adjacent frames from leaking across splits.
    n_val = max(1, round(len(subfolders) * VAL_FRACTION))
    if len(subfolders) - n_val < 1:
        raise SystemExit(
            f"Need at least 2 subfolders to make a video-disjoint train/val split "
            f"(found {len(subfolders)})."
        )
    val_set = set(subfolders[:n_val])  # first N (alphabetically) → val
    splits = {f: ("val" if f in val_set else "train") for f in subfolders}

    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"Using device: {device}")
    print(f"Found {len(subfolders)} subfolders. Train: {len(subfolders) - n_val}, Val: {n_val}")
    print(f"  val folders: {[f.name for f in subfolders[:n_val]]}")

    model = load_model(MODEL_PATH, device)
    transform = build_transform()

    totals = {
        "train": {"rally": 0, "non_rally": 0},
        "val": {"rally": 0, "non_rally": 0},
    }
    total_skipped = 0
    for folder in subfolders:
        split = splits[folder]
        r, n, s = process_folder(folder, split, output_root, dataset_root, model, transform, device)
        totals[split]["rally"] += r
        totals[split]["non_rally"] += n
        total_skipped += s

    print("\nDone.")
    print(f"  train: rally={totals['train']['rally']}  non_rally={totals['train']['non_rally']}")
    print(f"  val:   rally={totals['val']['rally']}  non_rally={totals['val']['non_rally']}")
    if total_skipped:
        print(f"  total skipped:   {total_skipped}")
    print(f"  per-folder output -> {output_root}")
    print(f"  combined dataset -> {dataset_root}")


if __name__ == "__main__":
    main()
