"""Extract individual frames from a badminton match video using OpenCV.

Reads VIDEO_PATH and OUTPUT_DIR from a .env file in the project root.
Saves frames as JPEG files and writes a metadata JSON with video properties.
"""

import json
import logging
import os
from pathlib import Path

import cv2
from dotenv import load_dotenv
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

FRAME_FILENAME_PATTERN = "frame_{:05d}.jpg"
JPEG_QUALITY = 95
PROGRESS_INTERVAL = 100

# .env is looked up from the project root (two levels above this file)
_ENV_PATH = Path(__file__).resolve().parents[1] / ".env"


def extract_frames(video_path: Path, output_dir: Path) -> dict:
    """Extract all frames from a video file and save them as JPEGs.

    Args:
        video_path: Path to the source video file.
        output_dir: Directory where extracted frames will be saved.

    Returns:
        Metadata dict with fps, total_frames, width, height, video_path.

    Raises:
        FileNotFoundError: If the video file does not exist.
        RuntimeError: If OpenCV cannot open the video.
    """
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    logger.info(
        "Video: %s | %.2f fps | %d frames | %dx%d",
        video_path.name, fps, total_frames, width, height,
    )

    saved = 0
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]

    with tqdm(total=total_frames, desc="Extracting frames", unit="frame") as pbar:
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            try:
                out_path = output_dir / FRAME_FILENAME_PATTERN.format(frame_idx)
                cv2.imwrite(str(out_path), frame, encode_params)
                saved += 1
            except Exception as exc:
                logger.warning("Failed to save frame %d: %s", frame_idx, exc)

            if frame_idx % PROGRESS_INTERVAL == 0 and frame_idx > 0:
                logger.info("Saved %d / %d frames", frame_idx, total_frames)

            frame_idx += 1
            pbar.update(1)

    cap.release()
    logger.info("Done. Saved %d frames to %s", saved, output_dir)

    metadata = {
        "fps": fps,
        "total_frames": total_frames,
        "width": width,
        "height": height,
        "video_path": str(video_path.resolve()),
        "frames_saved": saved,
        "output_dir": str(output_dir.resolve()),
    }

    meta_path = output_dir / "metadata.json"
    with meta_path.open("w") as f:
        json.dump(metadata, f, indent=2)
    logger.info("Metadata saved to %s", meta_path)

    return metadata


def load_config() -> tuple[Path, Path]:
    """Load VIDEO_PATH and OUTPUT_DIR from the .env file.

    Returns:
        Tuple of (video_path, output_dir) as Path objects.

    Raises:
        FileNotFoundError: If the .env file does not exist.
        ValueError: If required variables are missing from .env.
    """
    if not _ENV_PATH.exists():
        raise FileNotFoundError(
            f".env file not found at {_ENV_PATH}\n"
            "Copy .env.example to .env and fill in your paths."
        )

    load_dotenv(_ENV_PATH)

    raw_video = os.getenv("VIDEO_PATH")
    raw_output = os.getenv("OUTPUT_DIR")

    missing = [name for name, val in [("VIDEO_PATH", raw_video), ("OUTPUT_DIR", raw_output)] if not val]
    if missing:
        raise ValueError(f"Missing required variable(s) in .env: {', '.join(missing)}")

    return Path(raw_video), Path(raw_output)  # type: ignore[arg-type]


if __name__ == "__main__":
    video_path, output_dir = load_config()
    extract_frames(video_path, output_dir)
