"""Small dependency-free SO(3) helpers used by parsing and calibration."""

from __future__ import annotations

import numpy as np


def project_to_rotation(matrix: np.ndarray) -> np.ndarray:
    """Return the closest proper rotation matrix in Frobenius norm."""
    u, _, vt = np.linalg.svd(np.asarray(matrix, dtype=np.float64).reshape(3, 3))
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1.0
        rotation = u @ vt
    return rotation


def quaternion_xyzw_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    x, y, z, w = np.asarray(quaternion, dtype=np.float64).reshape(4)
    norm = np.linalg.norm([x, y, z, w])
    if norm < 1e-10:
        raise ValueError("quaternion must have non-zero norm")
    x, y, z, w = np.array([x, y, z, w]) / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def matrix_to_quaternion_xyzw(rotation: np.ndarray) -> np.ndarray:
    r = project_to_rotation(rotation)
    trace = np.trace(r)
    if trace > 0:
        scale = 2.0 * np.sqrt(trace + 1.0)
        return np.array([(r[2, 1] - r[1, 2]) / scale, (r[0, 2] - r[2, 0]) / scale,
                         (r[1, 0] - r[0, 1]) / scale, 0.25 * scale])
    index = int(np.argmax(np.diag(r)))
    if index == 0:
        scale = 2.0 * np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2])
        result = np.array([0.25 * scale, (r[0, 1] + r[1, 0]) / scale,
                           (r[0, 2] + r[2, 0]) / scale, (r[2, 1] - r[1, 2]) / scale])
    elif index == 1:
        scale = 2.0 * np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2])
        result = np.array([(r[0, 1] + r[1, 0]) / scale, 0.25 * scale,
                           (r[1, 2] + r[2, 1]) / scale, (r[0, 2] - r[2, 0]) / scale])
    else:
        scale = 2.0 * np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1])
        result = np.array([(r[0, 2] + r[2, 0]) / scale, (r[1, 2] + r[2, 1]) / scale,
                           0.25 * scale, (r[1, 0] - r[0, 1]) / scale])
    return result / np.linalg.norm(result)


def rotation_geodesic_angle(first: np.ndarray, second: np.ndarray) -> float:
    relative = project_to_rotation(first).T @ project_to_rotation(second)
    cosine = np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.arccos(cosine))


def average_rotations(rotations: list[np.ndarray]) -> np.ndarray:
    if not rotations:
        raise ValueError("cannot average an empty rotation list")
    accumulator = sum(project_to_rotation(rotation) for rotation in rotations)
    return project_to_rotation(accumulator)
