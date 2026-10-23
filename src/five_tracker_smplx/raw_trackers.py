"""Reader for the recorder's pre-normalization global tracker NPZ stream."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .geometry import project_to_rotation, quaternion_xyzw_to_matrix
from .types import Pose, TrackerFrame


_ROLE_TO_TRACKER = {
    "camera": "head",
    "left_wrist": "left_hand",
    "right_wrist": "right_hand",
    "left_ankle": "left_foot",
    "right_ankle": "right_foot",
}


def read_raw_tracker_npz(
    path: str | Path,
    world_rotation: np.ndarray | None = None,
    *,
    quaternion_order: str = "wxyz",
) -> list[TrackerFrame]:
    """Read global 6-DoF poses before any head-first normalization.

    The capture recorder stores quaternions as ``wxyz`` and positions in
    metres. They are reordered to ``xyzw`` and converted to the standard
    active local-to-world matrix. ``quaternion_order='xyzw'`` remains available
    for explicitly supplied legacy fixtures. The raw file does not contain a
    gravity stream, so no gravity arrow is added by this adapter.
    """
    if quaternion_order not in {"wxyz", "xyzw"}:
        raise ValueError("quaternion_order must be 'wxyz' or 'xyzw'")
    values = np.load(Path(path), allow_pickle=False)
    rotation = np.eye(3) if world_rotation is None else project_to_rotation(np.asarray(world_rotation, dtype=np.float64).reshape(3, 3))
    prefix = "tracker_0__pose"
    times = np.asarray(values[f"{prefix}__t_host_us"], dtype=np.int64)
    if len(times) == 0:
        raise ValueError("raw tracker NPZ is empty")
    arrays = {}
    missing_roles: list[str] = []
    for role, name in _ROLE_TO_TRACKER.items():
        position_key = f"{prefix}.{role}.position"
        orientation_key = f"{prefix}.{role}.orientation"
        valid_key = f"{prefix}.{role}.valid"
        present = tuple(key in values for key in (position_key, orientation_key, valid_key))
        if not any(present):
            # Older/incomplete captures can omit one tracker stream entirely.
            # Preserve the five-tracker schema and mark that stream unusable so
            # downstream fitting can continue in a four-tracker degraded mode.
            position = np.zeros((len(times), 3), dtype=np.float64)
            quaternion = np.zeros((len(times), 4), dtype=np.float64)
            # Source recorder order is wxyz, so the identity is (1, 0, 0, 0).
            quaternion[:, 0] = 1.0
            valid = np.zeros(len(times), dtype=bool)
            missing_roles.append(role)
        elif not all(present):
            missing = [
                key for key, exists in zip((position_key, orientation_key, valid_key), present)
                if not exists
            ]
            raise KeyError(f"incomplete raw stream for {role}; missing {missing}")
        else:
            position = np.asarray(values[position_key], dtype=np.float64)
            quaternion = np.asarray(values[orientation_key], dtype=np.float64)
            valid = np.asarray(values[valid_key], dtype=bool)
        if position.shape != (len(times), 3) or quaternion.shape != (len(times), 4) or valid.shape != (len(times),):
            raise ValueError(f"invalid raw stream shape for {role}")
        # Some SDKs write NaN transforms on frames explicitly marked invalid.
        # Pose enforces finite storage, so keep the validity bit and replace
        # only the ignored payload with neutral finite placeholders.
        usable = valid & np.isfinite(position).all(axis=1) & np.isfinite(quaternion).all(axis=1)
        position = position.copy(); quaternion = quaternion.copy()
        position[~usable] = 0.0
        quaternion[~usable] = (1.0, 0.0, 0.0, 0.0) if quaternion_order == "wxyz" else (0.0, 0.0, 0.0, 1.0)
        valid = usable
        # q and -q encode the same SO(3) rotation, but a raw stream may jump
        # between those representatives.  That jump is harmless when each
        # quaternion is converted independently, yet it would look like a
        # 360-degree interpolation to any downstream temporal filter.  Keep a
        # continuous representative for every valid sample without changing
        # the represented rotation or the original positions.
        previous = None
        for index in range(len(quaternion)):
            if not valid[index]:
                # Do not infer a sign relation across an invalid/gapped span.
                previous = None
                continue
            if previous is not None and float(np.dot(previous, quaternion[index])) < 0.0:
                quaternion[index] *= -1.0
            previous = quaternion[index].copy()
        arrays[name] = (position, quaternion, valid)
    values.close()
    start = int(times[0])
    return [
        TrackerFrame(
            timestamp=(int(timestamp) - start) * 1e-6,
            trackers={
                name: Pose(
                    rotation @ position[index],
                    rotation @ quaternion_xyzw_to_matrix(
                        quaternion[index][[1, 2, 3, 0]]
                        if quaternion_order == "wxyz" else quaternion[index]
                    ),
                    bool(valid[index]),
                )
                for name, (position, quaternion, valid) in arrays.items()
            },
            auxiliary={
                "source_pose_frame": "raw_global",
                "missing_tracker_roles": tuple(missing_roles),
            },
        )
        for index, timestamp in enumerate(times)
    ]
