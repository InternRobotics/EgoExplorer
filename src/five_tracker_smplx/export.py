"""Portable SMPL-family trajectory export."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .types import SmplTrajectory


def write_smplx_npz(path: str | Path, trajectory: SmplTrajectory, *, metadata: dict | None = None) -> None:
    """Write standard axis-angle SMPL-X parameters plus a small JSON sidecar."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamps": trajectory.timestamps,
        "betas": trajectory.betas,
        "global_orient": trajectory.global_orient,
        "transl": trajectory.transl,
        "body_pose": trajectory.body_pose,
        "scale": np.asarray(trajectory.scale, dtype=np.float32),
    }
    if trajectory.left_hand_pose is not None:
        payload["left_hand_pose"] = trajectory.left_hand_pose
    if trajectory.right_hand_pose is not None:
        payload["right_hand_pose"] = trajectory.right_hand_pose
    if trajectory.attachment_position is not None:
        payload["attachment_position"] = trajectory.attachment_position
    if trajectory.attachment_rotation is not None:
        payload["attachment_rotation"] = trajectory.attachment_rotation
    np.savez_compressed(destination, **payload)
    sidecar = destination.with_suffix(".json")
    sidecar.write_text(json.dumps(metadata or {}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
