"""MANUS 25-node raw skeleton to SMPL-X/MANO hand-pose fitting.

The old implementation called this layout ``Wuji``.  The recorder metadata and
the MANUS Core raw-skeleton specification both identify it as MANUS; aliases at
the bottom keep previously generated command lines import-compatible.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Sequence

import numpy as np

from .device import select_device
from .model import ModelPaths, load_body_model
from .optimizer import _global_body_rotations
from .types import SmplTrajectory, TrackerFrame

# MANUS Core raw-skeleton layout. Point 0 is the wrist. Each non-thumb group
# starts at a metacarpal/palm anchor that MANO does not articulate separately.
MANUS_POINT_IDS = {
    "thumb": (1, 2, 3, 4),
    "index": (6, 7, 8, 9),
    "middle": (11, 12, 13, 14),
    "ring": (16, 17, 18, 19),
    "pinky": (21, 22, 23, 24),
}
# The Wuji glove publishes 21 MediaPipe hand landmarks.  That set is exactly
# MANUS's 25 nodes minus the four non-thumb metacarpals — which this fitter
# already excludes from MANO articulation — so the two layouts differ only by
# an index shift, not by which anatomy is measured.
WUJI_POINT_IDS = {
    "thumb": (1, 2, 3, 4),
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}
GLOVE_POINT_IDS = {"manus25": MANUS_POINT_IDS, "wuji21": WUJI_POINT_IDS}
GLOVE_NODE_COUNTS = {"manus25": 25, "wuji21": 21}
# Exact SMPL-X flattened hand-pose order.
MANO_FINGER_ORDER = ("index", "middle", "pinky", "ring", "thumb")
SMPLX_HAND_JOINTS = {
    "left": {"index": (25, 26, 27, 67), "middle": (28, 29, 30, 68), "pinky": (31, 32, 33, 70), "ring": (34, 35, 36, 69), "thumb": (37, 38, 39, 66)},
    "right": {"index": (40, 41, 42, 72), "middle": (43, 44, 45, 73), "pinky": (46, 47, 48, 75), "ring": (49, 50, 51, 74), "thumb": (52, 53, 54, 71)},
}
SMPLX_WRIST_JOINT = {"left": 20, "right": 21}


def glove_target_indices(side: str, layout: str = "manus25") -> tuple[np.ndarray, np.ndarray]:
    """Return matching raw-point and SMPL-X output-joint indices (20 each).

    Only the raw-point half depends on the glove layout; the SMPL-X targets are
    the same joints whichever device measured them.
    """
    if side not in SMPLX_HAND_JOINTS:
        raise ValueError(f"invalid hand side: {side!r}")
    if layout not in GLOVE_POINT_IDS:
        raise ValueError(f"unknown glove layout: {layout!r}")
    point_ids = GLOVE_POINT_IDS[layout]
    raw = np.asarray([index for finger in MANO_FINGER_ORDER for index in point_ids[finger]], dtype=np.int64)
    smplx = np.asarray([index for finger in MANO_FINGER_ORDER for index in SMPLX_HAND_JOINTS[side][finger]], dtype=np.int64)
    return raw, smplx


def manus_target_indices(side: str) -> tuple[np.ndarray, np.ndarray]:
    """MANUS-layout target indices (kept for existing callers)."""
    return glove_target_indices(side, "manus25")


def _all_hand_targets(frames: Sequence[TrackerFrame], side: str, world_rotation: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw_ids, _ = manus_target_indices(side)
    positions, tracker_positions, tracker_rotations = [], [], []
    for frame in frames:
        values = np.asarray(frame.fingers[side]["joint_positions"], dtype=np.float32)
        if values.shape != (25, 3):
            raise ValueError(f"expected 25x3 MANUS positions for {side}, got {values.shape}")
        positions.append((world_rotation @ values[raw_ids].T).T)
        tracker = frame[f"{side}_hand"]
        tracker_positions.append(tracker.position)
        tracker_rotations.append(tracker.rotation)
    return np.asarray(positions), np.asarray(tracker_positions), np.asarray(tracker_rotations)


def _hand_local_targets(frames: Sequence[TrackerFrame], side: str, world_rotation: np.ndarray) -> np.ndarray:
    positions, tracker_positions, tracker_rotations = _all_hand_targets(frames, side, world_rotation)
    # Tracker quaternion matrices use the recorder's row/right-action
    # convention. World-column deltas enter the Tracker frame through R, not
    # R.T (the latter is only correct for conventional column-axis matrices).
    return np.einsum("tij,tkj->tki", tracker_rotations, positions - tracker_positions[:, None]).astype(np.float32)


def fit_manus_hands(
    trajectory: SmplTrajectory,
    frames: Sequence[TrackerFrame],
    *,
    world_rotation: np.ndarray,
    body_model_path: str | Path,
    model_type: str = "smplx",
    gender: str = "neutral",
    device: str = "auto",
    window_size: int = 60,
    iterations: int = 40,
    verbose: bool = False,
) -> SmplTrajectory:
    """Infer both 45-D SMPL-X MANO poses from MANUS's 25-node chain.

    Fitting occurs in each tracker frame, so it is unaffected by residual global
    root error in the sparse body reconstruction.  The tracker-to-SMPL wrist
    orientation is taken from the first trajectory frame, which is the same
    reference convention used by the body fitting stage.
    """
    if len(trajectory.timestamps) != len(frames):
        raise ValueError("trajectory and hand frames must have identical lengths")
    import torch

    selected_device = select_device(device).selected
    body = load_body_model(ModelPaths(Path(body_model_path), None, model_type, gender), selected_device).eval()
    for parameter in body.parameters():
        parameter.requires_grad_(False)
    rotation = np.asarray(world_rotation, dtype=np.float32).reshape(3, 3)
    targets = {side: torch.as_tensor(_hand_local_targets(frames, side, rotation), device=selected_device) for side in ("left", "right")}
    dtype = torch.float32
    with torch.no_grad():
        zero3 = torch.zeros((1, 3), dtype=dtype, device=selected_device)
        zero_body = torch.zeros((1, 63), dtype=dtype, device=selected_device)
        zero_hand = torch.zeros((1, 45), dtype=dtype, device=selected_device)
        zero_exp = torch.zeros((1, 10), dtype=dtype, device=selected_device)
        reference = body(
            global_orient=zero3, body_pose=zero_body, transl=zero3,
            left_hand_pose=zero_hand, right_hand_pose=zero_hand,
            jaw_pose=zero3, leye_pose=zero3, reye_pose=zero3, expression=zero_exp, return_verts=False,
        )
        wrist_rotations = _global_body_rotations(zero3, zero_body.reshape(1, 21, 3), body.parents.to(selected_device))[0]
        # C maps SMPL wrist-local vectors into the corresponding tracker frame.
        tracker_to_wrist = {
            side: torch.as_tensor(frames[0][f"{side}_hand"].rotation, dtype=dtype, device=selected_device) @ wrist_rotations[SMPLX_WRIST_JOINT[side]]
            for side in ("left", "right")
        }
        wrist_positions = {side: reference.joints[0, SMPLX_WRIST_JOINT[side]] for side in ("left", "right")}
        raw_joint_ids = {side: torch.as_tensor(manus_target_indices(side)[1], device=selected_device) for side in ("left", "right")}

    result = {side: [] for side in ("left", "right")}
    previous = {side: None for side in ("left", "right")}
    for start in range(0, len(frames), window_size):
        end = min(len(frames), start + window_size)
        count = end - start
        parameters = {}
        for side in ("left", "right"):
            initial = torch.zeros((count, 45), dtype=dtype, device=selected_device) if previous[side] is None else previous[side][-1:].expand(count, -1).clone()
            parameters[side] = torch.nn.Parameter(initial)
        optimizer = torch.optim.Adam(tuple(parameters.values()), lr=0.045)
        for _ in range(iterations):
            optimizer.zero_grad(set_to_none=True)
            zero3 = torch.zeros((count, 3), dtype=dtype, device=selected_device)
            zero_body = torch.zeros((count, 63), dtype=dtype, device=selected_device)
            zero_exp = torch.zeros((count, 10), dtype=dtype, device=selected_device)
            output = body(
                global_orient=zero3, body_pose=zero_body, transl=zero3,
                left_hand_pose=parameters["left"], right_hand_pose=parameters["right"],
                jaw_pose=zero3, leye_pose=zero3, reye_pose=zero3, expression=zero_exp, return_verts=False,
            )
            loss = torch.zeros((), dtype=dtype, device=selected_device)
            for side in ("left", "right"):
                local = output.joints[:, raw_joint_ids[side]] - wrist_positions[side]
                tracker_local = torch.einsum("ij,tkj->tki", tracker_to_wrist[side], local)
                position_loss = (tracker_local - targets[side][start:end]).square().mean()
                temporal_loss = torch.zeros((), dtype=dtype, device=selected_device)
                if count > 1:
                    temporal_loss = (parameters[side][1:] - parameters[side][:-1]).square().mean()
                if previous[side] is not None:
                    temporal_loss = temporal_loss + 4.0 * (parameters[side][0] - previous[side][-1]).square().mean()
                loss = loss + 100.0 * position_loss + 0.08 * temporal_loss + 0.002 * parameters[side].square().mean()
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            for side in ("left", "right"):
                previous[side] = parameters[side].detach()
                result[side].append(previous[side].cpu().numpy())
            if verbose:
                print(f"hand window {start}:{end}")
    return replace(
        trajectory,
        left_hand_pose=np.concatenate(result["left"], axis=0).astype(np.float32),
        right_hand_pose=np.concatenate(result["right"], axis=0).astype(np.float32),
    )


# Backward-compatible names for outputs made before the glove vendor was
# identified from the recorder metadata. New code must use the MANUS names.
wuji_target_indices = manus_target_indices
fit_wuji_hands = fit_manus_hands
