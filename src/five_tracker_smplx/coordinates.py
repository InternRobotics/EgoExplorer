"""One authoritative coordinate contract for recorder, SMPL-X and viewers.

The current ``*.tracker.npz`` recorder stores a right-handed, Y-up world frame
without an axis reflection.  During the 01--17 calibration the floor marks
align its ``+X`` with the subject's initial left and ``+Z`` with the initial
forward direction.  An arbitrary raw recording may start at any body yaw;
anatomical directions then come from the fitted root rotation, not directly
from the fixed world axes. The capture writer stores tracker quaternions in
``wxyz`` order. After reordering to ``xyzw``, the matrix is the usual active
local-to-world rotation: its **columns** are the physical Tracker axes in the
recorder global frame.

For a fixed mount, the canonical column-vector relation is::

    tracker_world_rotation = joint_world_rotation @ tracker_in_joint_rotation

Keeping this convention explicit avoids mixing a passive/world-to-tracker
matrix with SMPL-X's active local-to-world rotations.
"""

from __future__ import annotations

import numpy as np


POSITION_AXES = "right_handed_y_up; calibration_reference_x_left_z_forward"
TRACKER_ROTATION_ACTION = "column: tracker_world = joint_world @ tracker_in_joint"

# Map the SMPL-X/calibration reference (left, up, forward) into a conventional
# right-handed Z-up viewer frame (reference-left, reference-back, up). This is
# a proper rotation (det=+1). Raw body-facing direction still depends on root
# yaw and viewers must position a "front" camera from that fitted direction.
# Swapping Y/Z without the minus sign is a reflection and mirrors anatomy.
DISPLAY_ROTATION = np.asarray(
    ((1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0)),
    dtype=np.float32,
)


def estimate_root_yaw_from_correspondences(
    canonical_positions: np.ndarray,
    observed_positions: np.ndarray,
    valid: np.ndarray,
    *,
    maximum_frames: int = 30,
) -> float:
    """Estimate world yaw from labelled endpoints without using mount heading.

    A per-frame centroid removes translation.  The remaining horizontal 2-D
    Procrustes solve uses every valid labelled endpoint, so the 180-degree
    solution that exchanges left/right limbs receives a large penalty.
    """
    canonical = np.asarray(canonical_positions, dtype=np.float64)
    observed = np.asarray(observed_positions, dtype=np.float64)
    mask = np.asarray(valid, dtype=bool)
    if canonical.shape != observed.shape or canonical.ndim != 3 or canonical.shape[-1] != 3:
        raise ValueError("canonical and observed positions must both have shape [T,N,3]")
    if mask.shape != canonical.shape[:2]:
        raise ValueError("valid must have shape [T,N]")
    count = min(len(canonical), max(1, int(maximum_frames)))
    dot = 0.0
    cross = 0.0
    usable_frames = 0
    for source, target, frame_valid in zip(
        canonical[:count], observed[:count], mask[:count], strict=True,
    ):
        frame_valid = frame_valid & np.isfinite(source).all(axis=1) & np.isfinite(target).all(axis=1)
        if int(frame_valid.sum()) < 2:
            continue
        source_xz = source[frame_valid][:, (0, 2)]
        target_xz = target[frame_valid][:, (0, 2)]
        source_xz -= source_xz.mean(axis=0, keepdims=True)
        target_xz -= target_xz.mean(axis=0, keepdims=True)
        dot += float(np.sum(source_xz * target_xz))
        cross += float(np.sum(source_xz[:, 0] * target_xz[:, 1] - source_xz[:, 1] * target_xz[:, 0]))
        usable_frames += 1
    if usable_frames == 0 or np.hypot(dot, cross) < 1e-8:
        raise ValueError("cannot estimate root yaw from degenerate labelled endpoints")
    # The horizontal block of SMPL-X R_y(yaw) is the standard 2-D rotation by
    # -yaw in (X,Z), hence the sign reversal below.
    return float(-np.arctan2(cross, dot))


def recorded_tracker_axes_in_world(
    rotation: np.ndarray, *, convention: str = "rows",
) -> np.ndarray:
    """Return tracker axes in the requested viewer convention.

    The corrected source convention is ``columns``.  The historical default is
    retained for API compatibility; new visualizers pass ``convention="columns"``
    explicitly so they cannot silently reintroduce the old interpretation.
    """
    value = np.asarray(rotation)
    if convention == "columns":
        return value
    if convention == "rows":
        return value.T
    raise ValueError("convention must be 'columns' or 'rows'")
