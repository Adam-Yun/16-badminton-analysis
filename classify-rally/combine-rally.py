import os
import shutil
from pathlib import Path

SORTED_FRAMES = Path("/Users/adam/Desktop/Personal/Personal (Adam)/Github/16-badminton-analysis/sorted_frames")
OUTPUT_DIR = Path("combined_rally")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

copied = 0
skipped = 0

for video_folder in sorted(SORTED_FRAMES.iterdir()):
    rally_dir = video_folder / "rally"
    if not rally_dir.is_dir():
        continue
    for frame in sorted(rally_dir.iterdir()):
        if not frame.is_file():
            continue
        dest = OUTPUT_DIR / f"{video_folder.name}_{frame.name}"
        if dest.exists():
            skipped += 1
            continue
        shutil.copy2(frame, dest)
        copied += 1

print(f"Done. Copied: {copied}  Skipped (already exists): {skipped}")
print(f"Output: {OUTPUT_DIR}")
