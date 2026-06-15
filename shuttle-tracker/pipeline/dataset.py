"""PyTorch Dataset for multi-frame shuttlecock detection with heatmap targets.

Loads consecutive frame windows, generates 2-D Gaussian ground-truth heatmaps,
and applies augmentation transforms from augment.py.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
RESIZE_H = 360
RESIZE_W = 640
DEFAULT_N_FRAMES = 3
GAUSSIAN_SIGMA = 5
REQUIRED_CSV_COLS = {"frame_path", "x", "y", "visible"}


def _make_gaussian_heatmap(
    cx: float,
    cy: float,
    height: int = RESIZE_H,
    width: int = RESIZE_W,
    sigma: float = GAUSSIAN_SIGMA,
) -> np.ndarray:
    """Create a 2-D Gaussian heatmap with a single blob.

    Args:
        cx: Horizontal centre in pixels (0 … width-1).
        cy: Vertical centre in pixels (0 … height-1).
        height: Heatmap height in pixels.
        width: Heatmap width in pixels.
        sigma: Standard deviation of the Gaussian blob.

    Returns:
        Float32 array of shape (height, width) with values in [0, 1].
    """
    xs = np.arange(width, dtype=np.float32)
    ys = np.arange(height, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)
    heatmap = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
    return heatmap.astype(np.float32)


def _load_frame(path: Path, height: int = RESIZE_H, width: int = RESIZE_W) -> np.ndarray:
    """Load a single frame from disk and resize it.

    Args:
        path: Path to the JPEG/PNG frame file.
        height: Target height after resize.
        width: Target width after resize.

    Returns:
        RGB uint8 array of shape (height, width, 3).

    Raises:
        FileNotFoundError: If the image file does not exist.
        RuntimeError: If OpenCV cannot decode the image.
    """
    if not path.exists():
        raise FileNotFoundError(f"Frame not found: {path}")
    img = cv2.imread(str(path))
    if img is None:
        raise RuntimeError(f"OpenCV could not decode: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
    return img


class ShuttleDataset(Dataset):
    """Multi-frame shuttlecock detection dataset with Gaussian heatmap targets.

    Each sample is a window of N consecutive frames stacked along the channel
    axis, paired with a ground-truth heatmap for the *last* frame in the window.

    Args:
        annotations_csv: Path to CSV with columns: frame_path, x, y, visible.
        n_frames: Number of consecutive frames per sample (default 3).
        transform: Augmentation callable from augment.py (optional).
        resize_h: Target height for all frames.
        resize_w: Target width for all frames.
        sigma: Gaussian blob standard deviation in pixels.
    """

    def __init__(
        self,
        annotations_csv: Path,
        n_frames: int = DEFAULT_N_FRAMES,
        transform: Callable | None = None,
        resize_h: int = RESIZE_H,
        resize_w: int = RESIZE_W,
        sigma: float = GAUSSIAN_SIGMA,
    ) -> None:
        super().__init__()
        self.n_frames = n_frames
        self.transform = transform
        self.resize_h = resize_h
        self.resize_w = resize_w
        self.sigma = sigma

        df = pd.read_csv(annotations_csv)
        missing = REQUIRED_CSV_COLS - set(df.columns)
        if missing:
            raise ValueError(f"CSV missing columns: {missing}")

        self.df = df.reset_index(drop=True)

    def __len__(self) -> int:
        """Return the number of valid samples."""
        return max(0, len(self.df) - self.n_frames + 1)

    def __getitem__(self, idx: int) -> dict:
        """Load a window of N frames and generate the ground-truth heatmap.

        Args:
            idx: Index of the sample (anchored at the last frame of the window).

        Returns:
            Dict with keys:
                frames  – float tensor [3*N, H, W]
                heatmap – float tensor [1, H, W]
                visible – int (0 or 1)
        """
        end = idx + self.n_frames
        rows = self.df.iloc[idx:end]
        target_row = rows.iloc[-1]

        # ── Load N frames ────────────────────────────────────────────────────
        frame_list: list[np.ndarray] = []
        for _, row in rows.iterrows():
            try:
                frame = _load_frame(Path(row["frame_path"]), self.resize_h, self.resize_w)
            except Exception as exc:
                warnings.warn(f"Corrupt/missing frame, substituting blank: {exc}")
                frame = np.zeros((self.resize_h, self.resize_w, 3), dtype=np.uint8)
            frame_list.append(frame)

        # ── Ground-truth heatmap ─────────────────────────────────────────────
        visible = int(target_row["visible"])
        if visible:
            cx = float(target_row["x"]) * self.resize_w
            cy = float(target_row["y"]) * self.resize_h
            heatmap = _make_gaussian_heatmap(cx, cy, self.resize_h, self.resize_w, self.sigma)
        else:
            heatmap = np.zeros((self.resize_h, self.resize_w), dtype=np.float32)

        # ── Augmentation (applied to each frame independently) ───────────────
        if self.transform is not None:
            augmented_frames = []
            for frame in frame_list:
                result = self.transform(image=frame, keypoints=[])
                augmented_frames.append(result["image"])  # [3, H, W] tensor
            stacked = torch.cat(augmented_frames, dim=0).float()
        else:
            # Manual normalise to [0,1] float if no transform provided
            tensors = [
                torch.from_numpy(f.transpose(2, 0, 1)).float() / 255.0
                for f in frame_list
            ]
            stacked = torch.cat(tensors, dim=0)

        heatmap_tensor = torch.from_numpy(heatmap).unsqueeze(0)  # [1, H, W]

        return {"frames": stacked, "heatmap": heatmap_tensor, "visible": visible}


def collate_fn(batch: list[dict]) -> dict:
    """Custom collate that stacks samples with variable visibility flags.

    Args:
        batch: List of sample dicts from ShuttleDataset.__getitem__.

    Returns:
        Batched dict with tensors on CPU and a list of visible flags.
    """
    frames = torch.stack([s["frames"] for s in batch])
    heatmaps = torch.stack([s["heatmap"] for s in batch])
    visible = torch.tensor([s["visible"] for s in batch], dtype=torch.long)
    return {"frames": frames, "heatmap": heatmaps, "visible": visible}
