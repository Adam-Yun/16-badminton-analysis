"""Overlay shuttlecock detections and trajectory trails on video frames.

Draws a filled circle at the tracked position and a fading colour trail
for the last 20 trajectory points.
"""

from __future__ import annotations

import cv2
import numpy as np

# ── Drawing constants ─────────────────────────────────────────────────────────
CIRCLE_RADIUS = 6
CIRCLE_COLOR_BGR = (0, 0, 255)      # red
TRAIL_LENGTH = 20
TRAIL_START_BGR = (0, 255, 255)     # yellow (oldest visible)
TRAIL_END_BGR = (0, 0, 255)         # red (newest)
TRAIL_THICKNESS = 2
CONF_FONT = cv2.FONT_HERSHEY_SIMPLEX
CONF_FONT_SCALE = 0.6
CONF_COLOR_BGR = (255, 255, 255)    # white
CONF_THICKNESS = 2
CONF_POSITION = (10, 28)


def _lerp_color(
    c1: tuple[int, int, int],
    c2: tuple[int, int, int],
    t: float,
) -> tuple[int, int, int]:
    """Linearly interpolate between two BGR colours.

    Args:
        c1: Starting colour (B, G, R).
        c2: Ending colour (B, G, R).
        t: Blend factor in [0, 1]; 0 → c1, 1 → c2.

    Returns:
        Interpolated (B, G, R) colour tuple.
    """
    return tuple(int(a + (b - a) * t) for a, b in zip(c1, c2))  # type: ignore[return-value]


def draw_detections(
    frame: np.ndarray,
    tracked_pos: tuple[float, float] | None,
    trajectory_buffer,
    conf: float,
) -> np.ndarray:
    """Draw the tracked position, fading trail, and confidence score on a frame.

    Args:
        frame: BGR uint8 frame to annotate (modified in-place and returned).
        tracked_pos: (x, y) tracked shuttlecock position in pixel coordinates,
                     or None if no active detection.
        trajectory_buffer: TrajectoryBuffer instance holding past positions.
        conf: Confidence score in [0, 1] to display in the top-left corner.

    Returns:
        The annotated frame as a numpy array (same object as input).
    """
    # ── Fading trajectory trail ───────────────────────────────────────────────
    positions = trajectory_buffer.get_array()  # shape (N, 2)
    if len(positions) >= 2:
        trail = positions[-TRAIL_LENGTH:]
        n = len(trail)
        for i in range(1, n):
            t = i / (n - 1)                         # 0 = oldest, 1 = newest
            color = _lerp_color(TRAIL_START_BGR, TRAIL_END_BGR, t)
            alpha = 0.3 + 0.7 * t                   # older points are more transparent
            pt1 = (int(trail[i - 1][0]), int(trail[i - 1][1]))
            pt2 = (int(trail[i][0]), int(trail[i][1]))
            overlay = frame.copy()
            cv2.line(overlay, pt1, pt2, color, TRAIL_THICKNESS)
            cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    # ── Tracked position circle ───────────────────────────────────────────────
    if tracked_pos is not None:
        cx, cy = int(round(tracked_pos[0])), int(round(tracked_pos[1]))
        cv2.circle(frame, (cx, cy), CIRCLE_RADIUS, CIRCLE_COLOR_BGR, thickness=-1)

    # ── Confidence text ───────────────────────────────────────────────────────
    conf_text = f"conf: {conf:.2f}"
    cv2.putText(
        frame,
        conf_text,
        CONF_POSITION,
        CONF_FONT,
        CONF_FONT_SCALE,
        CONF_COLOR_BGR,
        CONF_THICKNESS,
        cv2.LINE_AA,
    )

    return frame
