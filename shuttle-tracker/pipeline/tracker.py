"""Kalman filter tracker and trajectory buffer for shuttlecock position.

Uses a constant-velocity motion model to smooth predictions and fill gaps
when the model produces no detection above the confidence threshold.
"""

from __future__ import annotations

import numpy as np
from filterpy.kalman import KalmanFilter

# ── Kalman filter noise parameters ───────────────────────────────────────────
MEASUREMENT_NOISE_R = 5.0    # measurement noise (pixels²)
PROCESS_NOISE_Q = 0.1        # process noise (acceleration uncertainty)

# ── Trajectory buffer ─────────────────────────────────────────────────────────
TRAJECTORY_MAX_LEN = 30


class KalmanShuttleTracker:
    """Constant-velocity Kalman filter tracker for a single shuttlecock.

    State vector: [x, y, vx, vy]
    Measurement vector: [x, y]

    Args:
        r: Measurement noise variance (scalar, applied to both x and y).
        q: Process noise variance (scalar, applied to all state dimensions).
    """

    def __init__(
        self,
        r: float = MEASUREMENT_NOISE_R,
        q: float = PROCESS_NOISE_Q,
    ) -> None:
        self._r = r
        self._q = q
        self._initialized = False
        self._kf: KalmanFilter = self._build_filter()

    # ── Private helpers ──────────────────────────────────────────────────────

    def _build_filter(self) -> KalmanFilter:
        """Construct and configure a new KalmanFilter instance.

        Returns:
            Configured but uninitialised KalmanFilter.
        """
        kf = KalmanFilter(dim_x=4, dim_z=2)

        # Transition matrix: constant velocity model
        kf.F = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float32)

        # Measurement matrix: observe x, y only
        kf.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float32)

        kf.R = np.eye(2, dtype=np.float32) * self._r
        kf.Q = np.eye(4, dtype=np.float32) * self._q
        kf.P = np.eye(4, dtype=np.float32) * 10.0  # initial uncertainty

        return kf

    # ── Public API ───────────────────────────────────────────────────────────

    @property
    def is_initialized(self) -> bool:
        """True once the filter has received at least one measurement."""
        return self._initialized

    def update(self, detection: tuple[float, float] | None) -> tuple[float, float]:
        """Update the tracker with an optional new detection.

        If detection is None the filter predicts forward without a measurement
        correction (useful when the shuttle is occluded or below threshold).

        Args:
            detection: (x, y) pixel coordinates, or None for predict-only.

        Returns:
            Smoothed (x, y) position estimate after this step.
        """
        if detection is not None:
            z = np.array([[detection[0]], [detection[1]]], dtype=np.float32)
            if not self._initialized:
                self._kf.x = np.array(
                    [detection[0], detection[1], 0.0, 0.0], dtype=np.float32
                ).reshape(4, 1)
                self._initialized = True

            self._kf.predict()
            self._kf.update(z)
        else:
            if self._initialized:
                self._kf.predict()

        x_est = float(self._kf.x[0, 0])
        y_est = float(self._kf.x[1, 0])
        return x_est, y_est

    def reset(self) -> None:
        """Reinitialise the Kalman filter and clear the initialised flag."""
        self._kf = self._build_filter()
        self._initialized = False


class TrajectoryBuffer:
    """Ring buffer that stores the last N shuttlecock positions.

    Args:
        max_len: Maximum number of positions to retain (default 30).
    """

    def __init__(self, max_len: int = TRAJECTORY_MAX_LEN) -> None:
        self._max_len = max_len
        self._positions: list[tuple[float, float]] = []

    def append(self, position: tuple[float, float]) -> None:
        """Add a new position, discarding the oldest if at capacity.

        Args:
            position: (x, y) pixel coordinates to add.
        """
        self._positions.append(position)
        if len(self._positions) > self._max_len:
            self._positions.pop(0)

    def get_array(self) -> np.ndarray:
        """Return stored positions as a numpy array.

        Returns:
            Float32 array of shape (N, 2) with columns [x, y].
            Returns an empty array of shape (0, 2) if the buffer is empty.
        """
        if not self._positions:
            return np.empty((0, 2), dtype=np.float32)
        return np.array(self._positions, dtype=np.float32)

    def clear(self) -> None:
        """Remove all stored positions."""
        self._positions.clear()

    def __len__(self) -> int:
        """Return the number of positions currently stored."""
        return len(self._positions)
