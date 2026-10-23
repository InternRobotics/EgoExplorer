"""Neutral-pose checks and serializable tracker attachment calibration."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .geometry import average_rotations, rotation_geodesic_angle
from .types import FIVE_TRACKERS, Attachment, Calibration, Pose, TrackerFrame


def average_neutral_frame(frames: Sequence[TrackerFrame]) -> TrackerFrame:
    if not frames:
        raise ValueError("calibration needs at least one frame")
    trackers: dict[str, Pose] = {}
    for name in FIVE_TRACKERS:
        valid = [frame[name] for frame in frames if frame[name].valid]
        if not valid:
            raise ValueError(f"tracker {name!r} is invalid throughout the calibration window")
        trackers[name] = Pose(
            position=np.mean([pose.position for pose in valid], axis=0),
            rotation=average_rotations([pose.rotation for pose in valid]),
            valid=True,
        )
    return TrackerFrame(timestamp=frames[-1].timestamp, trackers=trackers)


def validate_stationary_window(
    frames: Sequence[TrackerFrame], max_position_spread_m: float = 0.025, max_rotation_spread_rad: float = 0.12
) -> None:
    """Reject a neutral window with meaningful motion instead of silently biasing offsets."""
    neutral = average_neutral_frame(frames)
    for name in FIVE_TRACKERS:
        positions = np.stack([frame[name].position for frame in frames if frame[name].valid])
        distances = np.linalg.norm(positions - neutral[name].position, axis=1)
        rotations = [rotation_geodesic_angle(neutral[name].rotation, frame[name].rotation) for frame in frames if frame[name].valid]
        if float(np.max(distances)) > max_position_spread_m:
            raise ValueError(f"{name} moved {np.max(distances):.3f} m during neutral calibration")
        if float(np.max(rotations)) > max_rotation_spread_rad:
            raise ValueError(f"{name} rotated {np.rad2deg(np.max(rotations)):.1f}° during neutral calibration")


def build_placeholder_calibration(frames: Sequence[TrackerFrame], attachments: dict[str, dict[str, str]], position_scale: float = 1.0) -> Calibration:
    """Create a calibration shell; fitting later replaces identity segment transforms."""
    validate_stationary_window(frames)
    neutral = average_neutral_frame(frames)
    result = {
        name: Attachment(name, spec["body_joint"], spec.get("segment", name), Pose(np.zeros(3), np.eye(3)))
        for name, spec in attachments.items()
    }
    if set(result) != set(FIVE_TRACKERS):
        raise ValueError("attachments must configure exactly the five tracker names")
    return Calibration(neutral=neutral, attachments=result, position_scale=float(position_scale))
