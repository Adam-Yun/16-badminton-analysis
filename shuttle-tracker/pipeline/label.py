"""Interactive frame labeller for shuttlecock position annotation.

Click on the shuttle to record its position. Press keys to mark it as hidden,
undo the last label, or skip frames. Progress is saved after every frame so
you can quit and resume at any time.

Controls
--------
  Left-click       : mark shuttle position (visible=1)
  H                : shuttle hidden / not visible (visible=0)
  U                : undo the last saved label and go back one frame
  S                : skip this frame without saving a label
  Q / Esc          : quit and save progress

Usage
-----
  python pipeline/label.py \\
      --frames  data/frames/match1/ \\
      --output  data/annotations/labels.csv \\
      --pattern "frame_*.jpg"
"""

from __future__ import annotations

import csv
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ── Display constants ─────────────────────────────────────────────────────────
WINDOW_NAME = "Shuttle Labeller  |  click=label  H=hidden  U=undo  S=skip  Q=quit"
CROSSHAIR_COLOR = (0, 255, 0)       # green
CROSSHAIR_RADIUS = 10
CROSSHAIR_THICKNESS = 2
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.55
FONT_COLOR = (255, 255, 255)
FONT_THICKNESS = 1
DISPLAY_MAX_WIDTH = 1280
DISPLAY_MAX_HEIGHT = 720


def _resize_for_display(img: np.ndarray) -> tuple[np.ndarray, float]:
    """Downscale an image so it fits the display window, preserving aspect ratio.

    Args:
        img: BGR image array.

    Returns:
        Tuple of (resized image, scale factor applied).
    """
    h, w = img.shape[:2]
    scale = min(DISPLAY_MAX_WIDTH / w, DISPLAY_MAX_HEIGHT / h, 1.0)
    if scale < 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return img, scale


def _draw_overlay(
    img: np.ndarray,
    frame_idx: int,
    total: int,
    labeled: int,
    last_action: str,
) -> np.ndarray:
    """Draw HUD text (progress + last action) on the display image.

    Args:
        img: Display image to annotate (modified in-place).
        frame_idx: Current 0-based frame index.
        total: Total number of frames.
        labeled: Number of frames labeled so far.
        last_action: Short string describing the last user action.

    Returns:
        Annotated image.
    """
    lines = [
        f"Frame {frame_idx + 1} / {total}",
        f"Labeled: {labeled}",
        f"Last: {last_action}",
    ]
    y = 22
    for line in lines:
        # Shadow for readability on any background
        cv2.putText(img, line, (9, y + 1), FONT, FONT_SCALE, (0, 0, 0), FONT_THICKNESS + 1, cv2.LINE_AA)
        cv2.putText(img, line, (9, y), FONT, FONT_SCALE, FONT_COLOR, FONT_THICKNESS, cv2.LINE_AA)
        y += 20
    return img


def _draw_crosshair(img: np.ndarray, cx: int, cy: int) -> np.ndarray:
    """Draw a crosshair circle at the clicked position.

    Args:
        img: Display image (modified in-place).
        cx: X coordinate of the click.
        cy: Y coordinate of the click.

    Returns:
        Annotated image.
    """
    cv2.circle(img, (cx, cy), CROSSHAIR_RADIUS, CROSSHAIR_COLOR, CROSSHAIR_THICKNESS)
    cv2.line(img, (cx - CROSSHAIR_RADIUS, cy), (cx + CROSSHAIR_RADIUS, cy), CROSSHAIR_COLOR, 1)
    cv2.line(img, (cx, cy - CROSSHAIR_RADIUS), (cx, cy + CROSSHAIR_RADIUS), CROSSHAIR_COLOR, 1)
    return img


def _load_existing(csv_path: Path) -> list[dict]:
    """Load existing CSV rows so labelling can resume.

    Args:
        csv_path: Path to the annotations CSV (may not exist yet).

    Returns:
        List of row dicts with keys: frame_path, x, y, visible.
    """
    if not csv_path.exists():
        return []
    with csv_path.open(newline="") as f:
        return list(csv.DictReader(f))


def _save_rows(csv_path: Path, rows: list[dict]) -> None:
    """Write all label rows to the CSV, overwriting if it exists.

    Args:
        csv_path: Destination path.
        rows: List of dicts with keys: frame_path, x, y, visible.
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["frame_path", "x", "y", "visible"])
        writer.writeheader()
        writer.writerows(rows)


def run_labeller(
    frames_dir: Path,
    output_csv: Path,
    pattern: str = "frame_*.jpg",
) -> None:
    """Launch the interactive labelling UI.

    Args:
        frames_dir: Directory containing frame images.
        output_csv: Path to write (or append to) the annotations CSV.
        pattern: Glob pattern used to find frame files.

    Raises:
        FileNotFoundError: If no frames are found matching the pattern.
    """
    frame_paths = sorted(frames_dir.glob(pattern))
    if not frame_paths:
        raise FileNotFoundError(f"No frames found in {frames_dir} matching '{pattern}'")

    # Resume from existing labels
    existing_rows = _load_existing(output_csv)
    already_labeled: set[str] = {r["frame_path"] for r in existing_rows}
    rows: list[dict] = list(existing_rows)

    # Find the first unlabeled frame
    start_idx = 0
    for i, fp in enumerate(frame_paths):
        if str(fp) not in already_labeled:
            start_idx = i
            break
    else:
        logger.info("All frames already labeled. Nothing to do.")
        return

    logger.info(
        "%d frames found. %d already labeled. Starting from frame %d.",
        len(frame_paths), len(existing_rows), start_idx + 1,
    )

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    click_pos: list[tuple[int, int] | None] = [None]

    def _on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            click_pos[0] = (x, y)

    cv2.setMouseCallback(WINDOW_NAME, _on_mouse)

    idx = start_idx
    last_action = "—"

    while idx < len(frame_paths):
        fp = frame_paths[idx]
        raw = cv2.imread(str(fp))
        if raw is None:
            logger.warning("Cannot read %s, skipping.", fp)
            idx += 1
            continue

        h_orig, w_orig = raw.shape[:2]
        click_pos[0] = None
        display_click: tuple[int, int] | None = None

        while True:
            display, scale = _resize_for_display(raw.copy())
            if display_click:
                display = _draw_crosshair(display, *display_click)
            display = _draw_overlay(display, idx, len(frame_paths), len(rows), last_action)
            cv2.imshow(WINDOW_NAME, display)

            key = cv2.waitKey(30) & 0xFF

            # ── Mouse click: shuttle visible ──────────────────────────────
            if click_pos[0] is not None:
                dx, dy = click_pos[0]
                display_click = (dx, dy)
                # Convert display coords back to original image coords
                real_x = dx / scale
                real_y = dy / scale
                norm_x = real_x / w_orig
                norm_y = real_y / h_orig
                rows.append({
                    "frame_path": str(fp),
                    "x": f"{norm_x:.6f}",
                    "y": f"{norm_y:.6f}",
                    "visible": "1",
                })
                _save_rows(output_csv, rows)
                last_action = f"visible @ ({real_x:.0f}, {real_y:.0f})"
                click_pos[0] = None
                idx += 1
                break

            # ── H: hidden ────────────────────────────────────────────────
            elif key in (ord("h"), ord("H")):
                rows.append({
                    "frame_path": str(fp),
                    "x": "0.0",
                    "y": "0.0",
                    "visible": "0",
                })
                _save_rows(output_csv, rows)
                last_action = "hidden"
                display_click = None
                idx += 1
                break

            # ── U: undo ──────────────────────────────────────────────────
            elif key in (ord("u"), ord("U")):
                if rows:
                    removed = rows.pop()
                    _save_rows(output_csv, rows)
                    last_action = f"undo ({Path(removed['frame_path']).name})"
                    idx = max(0, idx - 1)
                else:
                    last_action = "nothing to undo"
                display_click = None
                break  # re-render the previous frame

            # ── S: skip ──────────────────────────────────────────────────
            elif key in (ord("s"), ord("S")):
                last_action = "skipped"
                display_click = None
                idx += 1
                break

            # ── Q / Esc: quit ─────────────────────────────────────────────
            elif key in (ord("q"), ord("Q"), 27):
                logger.info("Quit. Saved %d labels to %s", len(rows), output_csv)
                cv2.destroyAllWindows()
                return

    cv2.destroyAllWindows()
    logger.info("Done. %d labels saved to %s", len(rows), output_csv)


_ENV_PATH = Path(__file__).resolve().parents[1] / ".env"


def load_config() -> tuple[Path, Path, str]:
    """Load FRAMES_DIR, ANNOTATIONS_CSV, and optionally FRAME_PATTERN from .env.

    Returns:
        Tuple of (frames_dir, output_csv, frame_pattern).

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

    raw_frames = os.getenv("FRAMES_DIR")
    raw_output = os.getenv("ANNOTATIONS_CSV")

    missing = [name for name, val in [("FRAMES_DIR", raw_frames), ("ANNOTATIONS_CSV", raw_output)] if not val]
    if missing:
        raise ValueError(f"Missing required variable(s) in .env: {', '.join(missing)}")

    pattern = os.getenv("FRAME_PATTERN", "frame_*.jpg")
    return Path(raw_frames), Path(raw_output), pattern  # type: ignore[arg-type]


if __name__ == "__main__":
    frames_dir, output_csv, pattern = load_config()
    run_labeller(frames_dir=frames_dir, output_csv=output_csv, pattern=pattern)
