"""Evaluation metrics for shuttlecock detection quality.

Computes detection rate, false positive rate, and prints a summary table.
"""

from __future__ import annotations

import numpy as np


def detection_rate(
    preds: list[tuple[float, float] | None],
    gts: list[tuple[float, float] | None],
    threshold_px: float = 10.0,
) -> float:
    """Percentage of visible frames where the prediction is within threshold_px of GT.

    Args:
        preds: Predicted (x, y) positions, or None when the model gives no detection.
        gts: Ground-truth (x, y) positions, or None when the shuttle is not visible.
        threshold_px: Distance threshold in pixels for a correct detection.

    Returns:
        Detection rate in [0, 1]. Returns 0.0 if there are no visible GT frames.

    Raises:
        ValueError: If preds and gts have different lengths.
    """
    if len(preds) != len(gts):
        raise ValueError(
            f"Length mismatch: preds={len(preds)}, gts={len(gts)}"
        )

    correct = 0
    visible_count = 0

    for pred, gt in zip(preds, gts):
        if gt is None:
            continue
        visible_count += 1
        if pred is None:
            continue
        dist = np.hypot(pred[0] - gt[0], pred[1] - gt[1])
        if dist <= threshold_px:
            correct += 1

    return correct / visible_count if visible_count > 0 else 0.0


def false_positive_rate(
    preds: list[tuple[float, float] | None],
    gts: list[tuple[float, float] | None],
) -> float:
    """Percentage of frames where the model predicts a shuttle but GT is not visible.

    Args:
        preds: Predicted (x, y) positions, or None when the model gives no detection.
        gts: Ground-truth (x, y) positions, or None when the shuttle is not visible.

    Returns:
        False positive rate in [0, 1]. Returns 0.0 if there are no invisible GT frames.

    Raises:
        ValueError: If preds and gts have different lengths.
    """
    if len(preds) != len(gts):
        raise ValueError(
            f"Length mismatch: preds={len(preds)}, gts={len(gts)}"
        )

    fp = 0
    invisible_count = 0

    for pred, gt in zip(preds, gts):
        if gt is not None:
            continue
        invisible_count += 1
        if pred is not None:
            fp += 1

    return fp / invisible_count if invisible_count > 0 else 0.0


def print_summary(
    preds: list[tuple[float, float] | None],
    gts: list[tuple[float, float] | None],
    threshold_px: float = 10.0,
) -> None:
    """Compute and print a formatted summary of detection metrics.

    Args:
        preds: Predicted (x, y) positions, or None.
        gts: Ground-truth (x, y) positions, or None.
        threshold_px: Pixel distance threshold for a correct detection.
    """
    dr = detection_rate(preds, gts, threshold_px)
    fpr = false_positive_rate(preds, gts)

    n_total = len(preds)
    n_visible = sum(1 for g in gts if g is not None)
    n_invisible = n_total - n_visible
    n_pred = sum(1 for p in preds if p is not None)

    line = "-" * 46
    print(line)
    print(f"{'Shuttlecock Detection Metrics':^46}")
    print(line)
    print(f"  Total frames      : {n_total:>8}")
    print(f"  GT visible frames : {n_visible:>8}")
    print(f"  GT hidden frames  : {n_invisible:>8}")
    print(f"  Frames with pred  : {n_pred:>8}")
    print(line)
    print(f"  Detection rate    : {dr * 100:>7.2f}%  (threshold {threshold_px:.0f} px)")
    print(f"  False positive rt : {fpr * 100:>7.2f}%")
    print(line)
