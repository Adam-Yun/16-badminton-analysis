"""Albumentations-based augmentation pipelines for shuttlecock detection training.

Heavy motion blur simulates the fast shuttle; compression artifacts simulate
typical broadcast video quality.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2

# ── Augmentation hyperparameters ─────────────────────────────────────────────
MOTION_BLUR_LIMIT = (10, 40)
MOTION_BLUR_P = 0.7
GAUSSIAN_BLUR_P = 0.3
BRIGHTNESS_CONTRAST_P = 0.4
HUE_SAT_VALUE_P = 0.3
COMPRESSION_QUALITY_LOWER = 50
COMPRESSION_P = 0.4
HORIZONTAL_FLIP_P = 0.5
SHADOW_P = 0.2

RESIZE_H = 360
RESIZE_W = 640

# ImageNet normalisation (applied after ToTensor)
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def get_train_transforms() -> A.Compose:
    """Build the training augmentation pipeline.

    Applies heavy motion blur, photometric jitter, compression artifacts,
    horizontal flip, and random shadows to simulate real broadcast footage.

    Returns:
        An albumentations Compose pipeline that accepts an image (HWC, uint8)
        and returns a normalised float tensor of shape [C, H, W].
    """
    return A.Compose(
        [
            A.Resize(RESIZE_H, RESIZE_W),
            A.MotionBlur(blur_limit=MOTION_BLUR_LIMIT, p=MOTION_BLUR_P),
            A.GaussianBlur(p=GAUSSIAN_BLUR_P),
            A.RandomBrightnessContrast(p=BRIGHTNESS_CONTRAST_P),
            A.HueSaturationValue(p=HUE_SAT_VALUE_P),
            A.ImageCompression(quality_range=(COMPRESSION_QUALITY_LOWER, 100), p=COMPRESSION_P),
            A.HorizontalFlip(p=HORIZONTAL_FLIP_P),
            A.RandomShadow(p=SHADOW_P),
            A.Normalize(mean=MEAN, std=STD),
            ToTensorV2(),
        ],
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
    )


def get_val_transforms() -> A.Compose:
    """Build the validation/inference augmentation pipeline (resize + normalise only).

    Returns:
        An albumentations Compose pipeline that resizes and normalises an image
        without any stochastic augmentations.
    """
    return A.Compose(
        [
            A.Resize(RESIZE_H, RESIZE_W),
            A.Normalize(mean=MEAN, std=STD),
            ToTensorV2(),
        ],
        keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
    )
