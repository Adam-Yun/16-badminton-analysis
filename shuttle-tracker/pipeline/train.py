"""Training loop for TrackNetV2 shuttlecock detection.

Trains on multi-frame windows with a weighted BCE heatmap loss, logs metrics
every epoch, saves the best checkpoint and a loss curve plot.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

# Project imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.tracknet import build_tracknet
from pipeline.augment import get_train_transforms, get_val_transforms
from pipeline.dataset import ShuttleDataset, collate_fn

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Training config — edit these ─────────────────────────────────────────────
ANNOTATIONS_CSV = Path("shuttle-tracker/data/annotations/labels.csv")
EPOCHS      = 100
BATCH       = 8
LR          = 1e-4
N_FRAMES    = 3
NUM_WORKERS = 4
RESUME      = None   # set to a Path like Path("shuttle-tracker/models/weights/best.pt") to resume

# ── Internal constants ────────────────────────────────────────────────────────
POSITIVE_WEIGHT        = 10.0
DETECTION_THRESHOLD_PX = 10
VAL_SPLIT              = 0.1
WEIGHTS_DIR            = Path(__file__).resolve().parents[1] / "models" / "weights"


def weighted_bce_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    pos_weight: float = POSITIVE_WEIGHT,
) -> torch.Tensor:
    """Weighted binary cross-entropy over heatmap pixels.

    Args:
        pred: Predicted heatmap [B, 1, H, W], values in [0, 1].
        target: Ground-truth heatmap [B, 1, H, W], values in [0, 1].
        pos_weight: Multiplier applied to positive (shuttle) pixel losses.

    Returns:
        Scalar loss tensor.
    """
    weight = torch.ones_like(target)
    weight[target > 0.5] = pos_weight
    return nn.functional.binary_cross_entropy(pred, target, weight=weight)


def detection_rate(
    preds: torch.Tensor,
    targets: torch.Tensor,
    visible: torch.Tensor,
    threshold_px: int = DETECTION_THRESHOLD_PX,
) -> float:
    """Fraction of visible frames where the predicted peak is ≤ threshold_px from GT.

    Args:
        preds: Predicted heatmaps [B, 1, H, W].
        targets: GT heatmaps [B, 1, H, W].
        visible: Binary visibility flags [B].
        threshold_px: Distance threshold in pixels.

    Returns:
        Detection rate in [0, 1]. Returns 0.0 if no visible frames in batch.
    """
    B, _, H, W = preds.shape
    correct = 0
    total = 0
    for i in range(B):
        if visible[i] == 0:
            continue
        total += 1
        # GT peak
        gt_flat = targets[i, 0].argmax()
        gt_y, gt_x = gt_flat // W, gt_flat % W
        # Predicted peak
        pred_flat = preds[i, 0].argmax()
        p_y, p_x = pred_flat // W, pred_flat % W
        dist = ((gt_x - p_x).float() ** 2 + (gt_y - p_y).float() ** 2).sqrt()
        if dist <= threshold_px:
            correct += 1
    return correct / total if total > 0 else 0.0


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Run one training epoch and return mean loss.

    Args:
        model: TrackNetV2 model.
        loader: Training DataLoader.
        optimizer: AdamW optimizer.
        device: Compute device.

    Returns:
        Mean training loss over all batches.
    """
    model.train()
    total_loss = 0.0
    for batch in tqdm(loader, desc="Train", leave=False):
        frames = batch["frames"].to(device)
        heatmaps = batch["heatmap"].to(device)

        optimizer.zero_grad()
        pred = model(frames)
        loss = weighted_bce_loss(pred, heatmaps)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    """Run validation and return (val_loss, detection_rate).

    Args:
        model: TrackNetV2 model.
        loader: Validation DataLoader.
        device: Compute device.

    Returns:
        Tuple of (mean_val_loss, mean_detection_rate).
    """
    model.eval()
    total_loss = 0.0
    total_dr = 0.0
    for batch in tqdm(loader, desc="Val  ", leave=False):
        frames = batch["frames"].to(device)
        heatmaps = batch["heatmap"].to(device)
        visible = batch["visible"].to(device)

        pred = model(frames)
        loss = weighted_bce_loss(pred, heatmaps)
        total_loss += loss.item()
        total_dr += detection_rate(pred, heatmaps, visible)

    n = len(loader)
    return total_loss / n, total_dr / n


def save_loss_curve(train_losses: list[float], val_losses: list[float], out_path: Path) -> None:
    """Plot and save training / validation loss curves.

    Args:
        train_losses: Per-epoch training losses.
        val_losses: Per-epoch validation losses.
        out_path: Destination PNG path.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, label="Train loss")
    ax.plot(epochs, val_losses, label="Val loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Weighted BCE loss")
    ax.set_title("TrackNetV2 training curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    logger.info("Loss curve saved to %s", out_path)


def main() -> None:
    """Entry point for training."""
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    logger.info("Using device: %s", device)
    logger.info(
        "Config — data=%s | epochs=%d | batch=%d | lr=%s | n_frames=%d",
        ANNOTATIONS_CSV, EPOCHS, BATCH, LR, N_FRAMES,
    )

    # ── Dataset split ────────────────────────────────────────────────────────
    full_ds = ShuttleDataset(
        annotations_csv=ANNOTATIONS_CSV,
        n_frames=N_FRAMES,
        transform=None,   # transforms set per-split below
    )
    n_val = max(1, int(len(full_ds) * VAL_SPLIT))
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(full_ds, [n_train, n_val])

    train_ds.dataset.transform = get_train_transforms()   # type: ignore[attr-defined]
    val_ds.dataset.transform = get_val_transforms()       # type: ignore[attr-defined]

    train_loader = DataLoader(
        train_ds, batch_size=BATCH, shuffle=True,
        num_workers=NUM_WORKERS, collate_fn=collate_fn, pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH, shuffle=False,
        num_workers=NUM_WORKERS, collate_fn=collate_fn, pin_memory=False,
    )

    # ── Model ────────────────────────────────────────────────────────────────
    model = build_tracknet(n_frames=N_FRAMES).to(device)
    optimizer = AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    start_epoch = 0
    if RESUME:
        ckpt = torch.load(RESUME, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0) + 1
        logger.info("Resumed from epoch %d", start_epoch)

    # ── Training loop ────────────────────────────────────────────────────────
    best_val_loss = float("inf")
    patience_counter = 0
    early_stop_patience = 15
    train_losses: list[float] = []
    val_losses: list[float] = []

    for epoch in range(start_epoch, EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_loss, dr = evaluate(model, val_loader, device)
        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        logger.info(
            "Epoch %03d/%03d | train=%.4f | val=%.4f | det_rate=%.3f | lr=%.2e",
            epoch + 1, EPOCHS, train_loss, val_loss, dr,
            optimizer.param_groups[0]["lr"],
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            ckpt_path = WEIGHTS_DIR / "best.pt"
            torch.save(
                {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict()},
                ckpt_path,
            )
            logger.info("  ✓ New best checkpoint saved (val=%.4f)", best_val_loss)
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                logger.info("Early stopping triggered after %d epochs without improvement.", epoch + 1)
                break

    save_loss_curve(train_losses, val_losses, WEIGHTS_DIR / "loss_curve.png")
    logger.info("Training complete. Best val loss: %.4f", best_val_loss)


if __name__ == "__main__":
    main()
