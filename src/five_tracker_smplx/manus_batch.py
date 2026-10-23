"""Fuse native MANUS hand fits into profile-driven full-body SMPL-X tracks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .export import write_smplx_npz
from .manus_hands import ManusHandFitResult, _capture_origin_us, fit_manus_npz, read_manus_fit, read_manus_npz, write_manus_fit
from .profile_batch import _session_files, _session_id, write_multi_sequence_html
from .profile_fitting import ProfileFitResult, SubjectProfile, _resample_rotvec
from .protocol import align_protocol
from .behavior_protocol import refine_alignment_by_behavior
from .raw_trackers import read_raw_tracker_npz
from .types import SmplTrajectory


def read_smplx_npz(path: str | Path) -> SmplTrajectory:
    """Load the portable trajectory written by :func:`write_smplx_npz`."""
    with np.load(path, allow_pickle=False) as values:
        count = len(values["timestamps"])
        return SmplTrajectory(
            timestamps=values["timestamps"].copy(), betas=values["betas"].copy(),
            global_orient=values["global_orient"].copy(), transl=values["transl"].copy(),
            body_pose=values["body_pose"].copy(),
            left_hand_pose=values["left_hand_pose"].copy() if "left_hand_pose" in values else np.zeros((count, 45), np.float32),
            right_hand_pose=values["right_hand_pose"].copy() if "right_hand_pose" in values else np.zeros((count, 45), np.float32),
            attachment_position=values["attachment_position"].copy() if "attachment_position" in values else None,
            attachment_rotation=values["attachment_rotation"].copy() if "attachment_rotation" in values else None,
            scale=float(values["scale"]) if "scale" in values else 1.0,
        )


def _tracker_origin_us(path: Path) -> int:
    with np.load(path, allow_pickle=False) as values:
        keys = [key for key in values.files if key.endswith("__pose__t_host_us")]
        if len(keys) != 1:
            raise ValueError(f"expected one tracker host-time stream in {path}, got {keys}")
        return int(values[keys[0]][0])


def fuse_manus_result(
    trajectory: SmplTrajectory,
    hand_result: ManusHandFitResult,
    *,
    tracker_origin_us: int,
    glove_origin_us: int,
) -> tuple[SmplTrajectory, dict[str, Any]]:
    """Synchronize sampled MANUS poses onto every tracker/body timestamp."""
    offset_s = (tracker_origin_us - glove_origin_us) / 1e6
    hand_time_on_body = hand_result.timestamps - offset_s
    target_time = np.asarray(trajectory.timestamps, dtype=np.float64)
    left = _resample_rotvec(hand_result.left_hand_pose.reshape(-1, 15, 3), hand_time_on_body, target_time).reshape(-1, 45)
    right = _resample_rotvec(hand_result.right_hand_pose.reshape(-1, 15, 3), hand_time_on_body, target_time).reshape(-1, 45)
    inside = (target_time >= hand_time_on_body[0]) & (target_time <= hand_time_on_body[-1])
    combined = SmplTrajectory(
        timestamps=trajectory.timestamps.copy(), betas=trajectory.betas.copy(),
        global_orient=trajectory.global_orient.copy(), transl=trajectory.transl.copy(),
        body_pose=trajectory.body_pose.copy(), left_hand_pose=left, right_hand_pose=right,
        attachment_position=None if trajectory.attachment_position is None else trajectory.attachment_position.copy(),
        attachment_rotation=None if trajectory.attachment_rotation is None else trajectory.attachment_rotation.copy(),
        scale=trajectory.scale,
    )
    sync = {
        "clock_offset_tracker_minus_video_s": float(offset_s),
        "hand_time_on_body_range_s": [float(hand_time_on_body[0]), float(hand_time_on_body[-1])],
        "body_time_range_s": [float(target_time[0]), float(target_time[-1])],
        "body_frames_covered_by_hands": int(inside.sum()),
        "body_frames_total": int(len(target_time)),
        "coverage_fraction": float(inside.mean()),
        "outside_policy": "nearest MANUS pose (no rotational extrapolation)",
    }
    return combined, sync


def fit_and_fuse_manus_batch(
    data_roots: Sequence[str | Path],
    body_output_dir: str | Path,
    subject_profile: str | Path,
    *,
    body_model_path: str | Path = "models/smplx",
    device: str = "cpu",
    max_hand_frames: int = 900,
    hand_window_size: int = 90,
    hand_iterations: int = 70,
    viewer_max_frames: int = 450,
    reuse_hand_fits: bool = False,
    use_protocol: bool = True,
    write_viewer: bool = True,
    verbose: bool = False,
) -> dict[str, Any]:
    """Fit both hands for all sessions, fuse them, and regenerate one viewer."""
    output = Path(body_output_dir)
    profile = SubjectProfile.load(subject_profile)
    viewer_results: list[tuple[str, Path, ProfileFitResult]] = []
    reports: list[dict[str, Any]] = []
    for tracker_path in _session_files(data_roots):
        session_id = _session_id(tracker_path)
        session_dir = tracker_path.parent
        glove_path = session_dir / f"{session_id}.npz"
        body_dir = output / session_id
        body_path = body_dir / "smplx_profile_driven.npz"
        if not glove_path.exists():
            raise FileNotFoundError(f"missing native glove file for {session_id}: {glove_path}")
        if not body_path.exists():
            raise FileNotFoundError(f"run reconstruct-profile first: {body_path}")
        hand_path = body_dir / "manus_to_smplx_hands.npz"
        if reuse_hand_fits and hand_path.exists() and hand_path.with_suffix(".json").exists():
            hand_result = read_manus_fit(hand_path)
        else:
            hand_result = fit_manus_npz(
                glove_path, subject_profile, body_model_path=body_model_path, device=device,
                max_frames=max_hand_frames, window_size=hand_window_size,
                iterations=hand_iterations, verbose=verbose,
            )
            write_manus_fit(hand_path, hand_result)
        trajectory = read_smplx_npz(body_path)
        streams = read_manus_npz(glove_path)
        glove_origin_us = _capture_origin_us(glove_path, streams)
        combined, sync = fuse_manus_result(
            trajectory, hand_result, tracker_origin_us=_tracker_origin_us(tracker_path),
            glove_origin_us=glove_origin_us,
        )
        combined_path = body_dir / "smplx_with_manus_hands.npz"
        body_summary = json.loads(body_path.with_suffix(".json").read_text(encoding="utf-8"))
        body_dt = np.diff(trajectory.timestamps)
        body_summary.update({
            "duration_s": float(trajectory.timestamps[-1] - trajectory.timestamps[0]),
            "median_rate_hz": float(1.0 / np.median(body_dt)) if len(body_dt) else 0.0,
            "gaps_over_0_5s": int(np.sum(body_dt > .5)),
            "maximum_gap_s": float(np.max(body_dt)) if len(body_dt) else 0.0,
        })
        combined_summary = {
            **body_summary,
            "hand_pose": "native MANUS 25-node position IK; SLERP to every tracker timestamp",
            "manus": hand_result.summary,
            "hand_sync": sync,
        }
        write_smplx_npz(combined_path, combined, metadata=combined_summary)
        diagnostics_path = body_dir / "tracker_diagnostics.npz"
        with np.load(diagnostics_path, allow_pickle=False) as diagnostics:
            valid = diagnostics["valid"].copy()
            predicted = diagnostics["predicted_tracker"].copy()
            residual = diagnostics["position_residual_m"].copy()
        frames = tuple(read_raw_tracker_npz(tracker_path))
        protocol = align_protocol(frames, glove_path=glove_path) if use_protocol else None
        if protocol is not None:
            protocol = refine_alignment_by_behavior(frames, protocol)
        fit_result = ProfileFitResult(
            combined, frames, valid, predicted, residual,
            float(body_summary["ground_y_m"]), combined_summary, protocol,
        )
        viewer_results.append((session_id, tracker_path, fit_result))
        reports.append({
            "session": session_id, "source_frames": len(frames),
            "smplx_frames": len(combined.timestamps), "hand_fit_frames": len(hand_result.timestamps),
            "left_hand_joint_rms_m": hand_result.summary["joint_rms_m"]["left"],
            "right_hand_joint_rms_m": hand_result.summary["joint_rms_m"]["right"],
            "hand_sync": sync, "output": str(combined_path.resolve()),
        })
    # Sharded per-session runs all target one body-output root, so they would
    # race on this single combined file; the batch driver disables it and the
    # viewer is regenerated once at the end instead.
    viewer = output / "all_sequences_with_hands.html"
    if write_viewer:
        write_multi_sequence_html(
            viewer, viewer_results, profile, body_model_path=body_model_path,
            device=device, max_frames=viewer_max_frames,
        )
    report = {
        "subject_profile": str(Path(subject_profile).resolve()),
        "viewer": str(viewer.resolve()) if write_viewer else None,
        "protocol_mode": "calibration" if use_protocol else "unlabelled_inference",
        "sequences": reports,
    }
    (output / "manus_fusion_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )
    return report
