"""Shared helpers for reading the recorder's five tracker streams."""

from __future__ import annotations

import numpy as np

# Role names in the raw *.tracker.npz, in the pipeline's anchor order
# (head, left wrist, right wrist, left ankle, right ankle).
RAW_ROLES = ("camera", "left_wrist", "right_wrist", "left_ankle", "right_ankle")


def rot6d(matrix: np.ndarray) -> np.ndarray:
    """Continuous 6-D rotation representation (Zhou et al.): the first two matrix columns."""
    return np.asarray(matrix, dtype=np.float32)[..., :, :2].reshape(*matrix.shape[:-2], 6)


def wxyz_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    """Vectorised conversion of the recorder's wxyz quaternions to rotation matrices."""
    q = np.asarray(quaternion, dtype=np.float64)
    q = q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12)
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.stack((
        1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
    ), axis=-1).reshape(q.shape[:-1] + (3, 3)).astype(np.float32)
