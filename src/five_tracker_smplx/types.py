"""Data contracts shared by input adapters, calibration and fitting."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

from .geometry import matrix_to_quaternion_xyzw, project_to_rotation

FIVE_TRACKERS = ("head", "left_hand", "right_hand", "left_foot", "right_foot")


@dataclass(frozen=True)
class Pose:
    position: np.ndarray
    rotation: np.ndarray = field(default_factory=lambda: np.eye(3))
    valid: bool = True

    def __post_init__(self) -> None:
        position = np.asarray(self.position, dtype=np.float64).reshape(3)
        rotation = project_to_rotation(np.asarray(self.rotation, dtype=np.float64).reshape(3, 3))
        if not np.isfinite(position).all() or not np.isfinite(rotation).all():
            raise ValueError("pose contains non-finite values")
        object.__setattr__(self, "position", position)
        object.__setattr__(self, "rotation", rotation)

    @property
    def quaternion_xyzw(self) -> np.ndarray:
        return matrix_to_quaternion_xyzw(self.rotation)

    def as_matrix(self) -> np.ndarray:
        result = np.eye(4)
        result[:3, :3] = self.rotation
        result[:3, 3] = self.position
        return result


@dataclass(frozen=True)
class TrackerFrame:
    timestamp: float
    trackers: Mapping[str, Pose]
    # Raw data are deliberately retained until the data provider's hand skeleton
    # convention is known. Do not assume it contains world-space positions.
    fingers: Mapping[str, Any] = field(default_factory=dict)
    # Recorder-specific signals such as gravity are retained without making the
    # core tracker contract depend on a particular SDK.
    auxiliary: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        names = set(self.trackers)
        missing = set(FIVE_TRACKERS) - names
        extra = names - set(FIVE_TRACKERS)
        if missing or extra:
            raise ValueError(f"tracker names must be exactly {FIVE_TRACKERS}; missing={sorted(missing)}, extra={sorted(extra)}")
        if not np.isfinite(self.timestamp):
            raise ValueError("timestamp must be finite")

    def __getitem__(self, name: str) -> Pose:
        return self.trackers[name]


@dataclass(frozen=True)
class Attachment:
    """Rigid transform from a configured SMPL segment to its Vive Tracker."""

    tracker: str
    body_joint: str
    segment: str
    segment_to_tracker: Pose


@dataclass(frozen=True)
class Calibration:
    neutral: TrackerFrame
    attachments: Mapping[str, Attachment]
    position_scale: float


@dataclass(frozen=True)
class SmplTrajectory:
    timestamps: np.ndarray
    betas: np.ndarray
    global_orient: np.ndarray
    transl: np.ndarray
    body_pose: np.ndarray
    left_hand_pose: np.ndarray | None = None
    right_hand_pose: np.ndarray | None = None
    # One rigid transform per tracker, calibrated jointly over the complete
    # sequence.  They are deliberately trajectory-level values, never
    # per-frame parameters.
    attachment_position: np.ndarray | None = None
    attachment_rotation: np.ndarray | None = None
    # Metric similarity scale applied to the shaped SMPL-X body before the
    # world translation.  Kept separate from betas because shape coefficients
    # are not a reliable subject-height parameter.
    scale: float = 1.0
