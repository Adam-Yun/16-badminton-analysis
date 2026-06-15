"""Full inference pipeline: run TrackNetV2 on a new video and write annotated output.

Slides a window of N frames through the video, runs the model on each window,
applies a Kalman tracker to smooth detections, and saves the result to disk.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy import ndimage
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.tracknet import build_tracknet
from pipeline.augment import get_val_transforms
from pipeline.tracker import KalmanShuttleTracker, TrajectoryBuffer
from utils.visualize import draw_detections

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
RESIZE_H = 360
RESIZE_W = 640
DEFAULT_CONF_THRESHOLD = 0.5
DEFAULT_N_FRAMES = 3


def _preprocess_frame(frame: np.ndarray, transform) -> torch.Tensor:
    """Resize and normalise a single BGR frame for model input.

    Args:
        frame: BGR uint8 frame from OpenCV.
        transform: Albumentations val transform pipeline.

    Returns:
        Float tensor of shape [3, RESIZE_H, RESIZE_W].
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (RESIZE_W, RESIZE_H))
    result = transform(image=rgb, keypoints=[])
    return result["image"].float()


def _find_peak(
    heatmap: np.ndarray,
    conf_threshold: float,
) -> tuple[tuple[int, int], float] | None:
    """Locate the highest-confidence detection in a heatmap.

    Args:
        heatmap: 2-D float array (H, W) with values in [0, 1].
        conf_threshold: Minimum peak value to consider a valid detection.

    Returns:
        ((x, y), confidence) if a peak above threshold is found, else None.
    """
    peak_val = heatmap.max()
    if peak_val < conf_threshold:
        return None
    peak_idx = np.argmax(heatmap)
    py, px = divmod(int(peak_idx), heatmap.shape[1])
    return (px, py), float(peak_val)


def run_inference(
    video_path: Path,
    weights_path: Path,
    output_path: Path,
    n_frames: int = DEFAULT_N_FRAMES,
    conf_threshold: float = DEFAULT_CONF_THRESHOLD,
) -> None:
    """Run TrackNetV2 inference on a video file and write annotated output.

    Args:
        video_path: Path to the source video.
        weights_path: Path to the saved model checkpoint (.pt).
        output_path: Path for the annotated output video.
        n_frames: Number of frames per model input window.
        conf_threshold: Minimum heatmap peak value to count as a detection.

    Raises:
        FileNotFoundError: If the video or weights file cannot be found.
        RuntimeError: If the video cannot be opened.
    """
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # ── Model ────────────────────────────────────────────────────────────────
    model = build_tracknet(n_frames=n_frames).to(device)
    ckpt = torch.load(weights_path, map_location=device)
    state = ckpt["model"] if "model" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    logger.info("Loaded weights from %s", weights_path)

    transform = get_val_transforms()
    tracker = KalmanShuttleTracker()
    traj = TrajectoryBuffer(max_len=30)

    # ── Video I/O ────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (orig_w, orig_h))

    # Sliding window buffer: keep last N raw frames
    window: deque[np.ndarray] = deque(maxlen=n_frames)
    # Buffer of original-resolution frames aligned with the window
    orig_window: deque[np.ndarray] = deque(maxlen=n_frames)

    scale_x = orig_w / RESIZE_W
    scale_y = orig_h / RESIZE_H

    with torch.inference_mode():
        for frame_idx in tqdm(range(total), desc="Inferring", unit="frame"):
            ret, raw = cap.read()
            if not ret:
                break

            try:
                tensor = _preprocess_frame(raw, transform)
            except Exception as exc:
                logger.warning("Frame %d preprocessing failed: %s", frame_idx, exc)
                tensor = torch.zeros(3, RESIZE_H, RESIZE_W)

            window.append(tensor)
            orig_window.append(raw.copy())

            if len(window) < n_frames:
                # Not enough frames yet; write the raw frame unchanged
                writer.write(raw)
                continue

            # ── Model forward pass ───────────────────────────────────────────
            inp = torch.cat(list(window), dim=0).unsqueeze(0).to(device)  # [1, 3N, H, W]
            heatmap = model(inp)[0, 0].cpu().numpy()  # [H, W]

            detection = _find_peak(heatmap, conf_threshold)

            if detection is not None:
                (px, py), conf = detection
                # Scale back to original resolution
                ox = px * scale_x
                oy = py * scale_y
                tracked = tracker.update((ox, oy))
            else:
                conf = 0.0
                tracked = tracker.update(None) if tracker.is_initialized else None

            if tracked is not None:
                traj.append(tracked)

            # ── Annotate the last frame in the window ────────────────────────
            annotated = draw_detections(
                orig_window[-1].copy(),
                tracked_pos=tracked,
                trajectory_buffer=traj,
                conf=conf,
            )
            writer.write(annotated)

    cap.release()
    writer.release()
    logger.info("Annotated video saved to %s", output_path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse inference CLI arguments.

    Args:
        argv: Argument list (defaults to sys.argv).

    Returns:
        Parsed namespace.
    """
    p = argparse.ArgumentParser(description="Run TrackNetV2 inference on a video.")
    p.add_argument("--video", type=Path, required=True, help="Input video path.")
    p.add_argument("--weights", type=Path, required=True, help="Model checkpoint (.pt).")
    p.add_argument("--output", type=Path, required=True, help="Annotated output video path.")
    p.add_argument("--n_frames", type=int, default=DEFAULT_N_FRAMES)
    p.add_argument("--conf_threshold", type=float, default=DEFAULT_CONF_THRESHOLD)
    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    run_inference(
        video_path=args.video,
        weights_path=args.weights,
        output_path=args.output,
        n_frames=args.n_frames,
        conf_threshold=args.conf_threshold,
    )
