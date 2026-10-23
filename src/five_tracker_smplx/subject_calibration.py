"""Joint subject shape/scale and rigid tracker-mount calibration.

The calibration protocol supplies several *known* body poses.  This module
uses their stationary holds from multiple recordings to separate three kinds
of unknowns which a single first-frame fit cannot separate reliably:

* one subject-wide SMPL-X shape and metric scale;
* one rigid transform from every tracked body segment to its tracker;
* one world yaw/translation for each recording.

Measured body height and arm span anchor the otherwise ambiguous global scale.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .calibration_motion import CalibrationMotionLibrary
from .coordinates import DISPLAY_ROTATION, POSITION_AXES, TRACKER_ROTATION_ACTION
from .geometry import average_rotations, matrix_to_quaternion_xyzw, quaternion_xyzw_to_matrix, rotation_geodesic_angle
from .optimizer import _global_body_rotations, _huber
from .protocol import PROTOCOL, align_protocol
from .raw_trackers import read_raw_tracker_npz


TRACKER_NAMES = ("head", "left_hand", "right_hand", "left_foot", "right_foot")
RAW_ROLES = ("camera", "left_wrist", "right_wrist", "left_ankle", "right_ankle")
ROLE_TO_TRACKER = dict(zip(RAW_ROLES, TRACKER_NAMES, strict=True))
TRACKED_JOINTS = {"head": 15, "left_hand": 20, "right_hand": 21, "left_foot": 7, "right_foot": 8}

# Protocol time is measured from the first video timestamp.  The selected
# intervals lie inside the stable part of each 7-second pose segment.
STATIC_WINDOWS = {
    "neutral": (1.5, 5.5),
    "a_pose": (17.5, 21.5),
    "t_pose": (24.5, 28.5),
    "forward": (31.5, 35.5),
}

_DYNAMIC_PHASE_LABELS = {
    ("head", .125): "左看", ("head", .375): "右看",
    ("head", .625): "低头", ("head", .875): "抬头",
    ("elbows", .125): "右屈肘 1", ("elbows", .375): "左屈肘 1",
    ("elbows", .625): "右屈肘 2", ("elbows", .875): "左屈肘 2",
    ("wrists", .2): "旋前", ("wrists", .4): "旋后",
    ("wrists", .6): "屈腕", ("wrists", .8): "伸腕",
    ("squat", .25): "下蹲 1", ("squat", .75): "下蹲 2",
}


def _longest_hold(
    mask: np.ndarray, timestamps: np.ndarray, *, trim_s: float = .65,
    prefer_first: bool = False,
) -> tuple[np.ndarray, list[float]] | None:
    """Return the stable interior of a qualifying contiguous candidate hold."""
    indices = np.flatnonzero(mask)
    if not len(indices):
        return None
    runs = np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1)
    runs = [run for run in runs if timestamps[run[-1]] - timestamps[run[0]] >= 1.5]
    if not runs:
        return None
    run = runs[0] if prefer_first else max(runs, key=lambda value: timestamps[value[-1]] - timestamps[value[0]])
    begin = float(timestamps[run[0]] + trim_s)
    end = float(timestamps[run[-1]] - trim_s)
    if end <= begin:
        begin, end = float(timestamps[run[0]]), float(timestamps[run[-1]])
    selected = mask & (timestamps >= begin) & (timestamps <= end)
    return selected, [begin, end]


def _static_pose_masks(
    positions: dict[str, np.ndarray], valid: dict[str, np.ndarray], relative_time: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, list[float]], str]:
    """Locate the four calibration holds when subjects follow the protocol slowly.

    The reference animation gives pose order, but capture durations vary from
    roughly 110 to 250 seconds. Wrist separation and wrist height distinguish
    A, T and forward holds without depending on a hard-coded playback speed.
    """
    count = len(relative_time)
    head = positions["camera"]
    left, right = positions["left_wrist"], positions["right_wrist"]
    wrist_midpoint = .5 * (left + right)
    separation = np.linalg.norm(left - right, axis=1)
    wrist_below_head = wrist_midpoint[:, 1] - head[:, 1]
    stacked = np.stack([positions[role] for role in RAW_ROLES], axis=1)
    speed = np.zeros(count, dtype=np.float64)
    if count > 1:
        dt = np.maximum(np.diff(relative_time), 1e-5)
        speed[1:] = np.max(np.linalg.norm(np.diff(stacked, axis=0), axis=2) / dt[:, None], axis=1)
    required = valid["camera"] & valid["left_wrist"] & valid["right_wrist"]
    stable = required & np.isfinite(stacked).all(axis=(1, 2)) & (speed < .12)
    candidates = {
        "a_pose": stable & (relative_time >= 15) & (relative_time <= 70)
        & (separation >= .80) & (separation <= 1.16)
        & (wrist_below_head >= -.62) & (wrist_below_head <= -.30),
        "t_pose": stable & (relative_time >= 20) & (relative_time <= 90)
        & (separation >= 1.05) & (wrist_below_head >= -.36),
        "forward": stable & (relative_time >= 30) & (relative_time <= 110)
        & (separation <= .36) & (wrist_below_head >= -.38),
    }
    masks: dict[str, np.ndarray] = {}
    windows: dict[str, list[float]] = {}
    neutral_begin, neutral_end = STATIC_WINDOWS["neutral"]
    masks["neutral"] = (relative_time >= neutral_begin) & (relative_time <= neutral_end)
    windows["neutral"] = [neutral_begin, neutral_end]
    automatic = True
    for pose_name in ("a_pose", "t_pose", "forward"):
        # Later wrist and finger stages can hold both hands forward longer
        # than stage 05, so protocol order is more reliable than duration.
        hold = _longest_hold(
            candidates[pose_name], relative_time,
            prefer_first=pose_name == "forward",
        )
        if hold is None:
            automatic = False
            begin, end = STATIC_WINDOWS[pose_name]
            masks[pose_name] = (relative_time >= begin) & (relative_time <= end)
            windows[pose_name] = [begin, end]
        else:
            masks[pose_name], windows[pose_name] = hold
    return masks, windows, "geometry_auto" if automatic else "mixed_with_fixed_fallback"


@dataclass(frozen=True)
class StaticObservation:
    session_id: str
    pose_name: str
    motion: str
    phase: float
    weight: float
    position: np.ndarray
    rotation: np.ndarray
    valid: np.ndarray
    sample_count: np.ndarray
    position_mad_m: np.ndarray


@dataclass(frozen=True)
class CalibrationDataset:
    observations: tuple[StaticObservation, ...]
    sessions: tuple[str, ...]
    excluded_streams: tuple[dict[str, Any], ...]
    session_quality: tuple[dict[str, Any], ...]
    locomotion_samples: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class SubjectCalibrationOptions:
    height_m: float
    armspan_m: float
    smplx_prior_path: Path | None = None
    smplx_prior_weight: float = .18
    body_model_path: Path = Path("models/smplx")
    vposer_path: Path = Path("models/vposer/V02_05")
    motion_cache_path: Path = Path("assets/calibration_motion_vposer_mano.npz")
    output_dir: Path = Path("outputs/subject_calibration")
    device: str = "cpu"
    iterations: int = 1100
    optimized_betas: int = 5
    verbose: bool = False


def load_smplx_shape_prior(path: str | Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Load a portable SMPL-X NPZ as a subject-shape prior.

    A single beta vector is preferred.  Per-frame beta estimates are reduced
    with a median so the calibration never inherits frame-to-frame shape
    jitter from a monocular estimator.
    """
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    with np.load(source, allow_pickle=False) as values:
        if "betas" not in values.files:
            raise ValueError(f"SMPL-X prior does not contain betas: {source}")
        raw = np.asarray(values["betas"], dtype=np.float64)
        if raw.shape == (10,):
            betas = raw
            reduction = "single_vector"
        elif raw.ndim == 2 and raw.shape[1] == 10 and len(raw):
            betas = np.median(raw, axis=0)
            reduction = "per_frame_median"
        else:
            raise ValueError(f"SMPL-X prior betas must have shape (10,) or (T,10), got {raw.shape}")
        if not np.isfinite(betas).all():
            raise ValueError(f"SMPL-X prior betas contain NaN/Inf: {source}")
        model_type = str(np.asarray(values["model_type"]).item()) if "model_type" in values.files else "smplx"
        if model_type.lower().replace("-", "") != "smplx":
            raise ValueError(f"shape prior must be SMPL-X, got {model_type!r}")
        metadata = {
            "source": str(source),
            "model_type": model_type,
            "gender": str(np.asarray(values["gender"]).item()) if "gender" in values.files else "unknown",
            "coordinate_frame": (
                str(np.asarray(values["coordinate_frame"]).item())
                if "coordinate_frame" in values.files else "unspecified"
            ),
            "reduction": reduction,
            "source_frame_count": int(raw.shape[0]) if raw.ndim == 2 else (
                int(np.asarray(values["timestamps"]).shape[0]) if "timestamps" in values.files else None
            ),
        }
    return betas.astype(np.float32), metadata


def _npz_prefix(values: np.lib.npyio.NpzFile) -> str:
    candidates = [name[: -len("__t_host_us")] for name in values.files if name.endswith("__t_host_us") and "pose" in name]
    if len(candidates) != 1:
        raise ValueError(f"expected one tracker pose stream, found {candidates}")
    return candidates[0]


def _video_start_us(session_dir: Path, session: dict[str, Any]) -> int:
    video_name = session.get("video", {}).get("timestamps_npz")
    if video_name:
        path = session_dir / video_name
        if path.exists():
            with np.load(path, allow_pickle=False) as values:
                return int(np.asarray(values["t_host_us"])[0])
    time_range = session.get("video", {}).get("t_host_us_range")
    if time_range:
        return int(time_range[0])
    raise ValueError(f"cannot find video start time in {session_dir}")


def _robust_rotation(quaternions: np.ndarray) -> np.ndarray:
    # Raw recorder orientation is wxyz; all internal/output matrices use the
    # standard active local-to-world convention.
    rotations = [quaternion_xyzw_to_matrix(np.asarray(value)[[1, 2, 3, 0]]) for value in quaternions]
    return average_rotations(rotations)


def _relative_yaw_from_quaternions_xyzw(
    quaternions: np.ndarray, baseline_mask: np.ndarray, *, quaternion_order: str = "xyzw",
) -> np.ndarray:
    """Return continuous anatomical yaw from a quaternion stream.

    Absolute Euler yaw is mount-dependent and does not commute with an
    arbitrary Tracker mounting rotation.  Forming ``R0.T @ Rt`` first cancels
    the fixed pre-mount, after which the world-Y yaw is well defined.
    """
    if quaternion_order not in {"wxyz", "xyzw"}:
        raise ValueError("quaternion_order must be 'wxyz' or 'xyzw'")
    rotations = np.stack([
        quaternion_xyzw_to_matrix(
            np.asarray(value)[[1, 2, 3, 0]] if quaternion_order == "wxyz" else np.asarray(value)
        )
        for value in np.asarray(quaternions, dtype=np.float64)
    ])
    mask = np.asarray(baseline_mask, dtype=bool)
    if not mask.any():
        mask = np.ones(len(rotations), dtype=bool)
    baseline = average_rotations(list(rotations[mask]))
    relative = np.einsum("ij,tjk->tik", baseline.T, rotations)
    # Active local-to-world rotation: use the local +X axis (column 0) for
    # heading.  Projecting row 0 is only valid for pure yaw and turns head
    # pitch/roll into a false yaw excursion.
    yaw = np.arctan2(-relative[:, 2, 0], relative[:, 0, 0])
    origin = float(np.arctan2(np.mean(np.sin(yaw[mask])), np.mean(np.cos(yaw[mask]))))
    return np.unwrap(np.arctan2(np.sin(yaw - origin), np.cos(yaw - origin)))


def _horizontal_speed(points: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
    if len(timestamps) < 2:
        return np.zeros(len(timestamps), dtype=np.float64)
    velocity = np.gradient(np.asarray(points)[:, (0, 2)], timestamps, axis=0)
    speed = np.linalg.norm(velocity, axis=1)
    # A three-sample median suppresses isolated optical/tracker spikes without
    # smearing the stance-to-swing transition used for contact calibration.
    if len(speed) >= 3:
        padded = np.pad(speed, (1, 1), mode="edge")
        speed = np.median(np.lib.stride_tricks.sliding_window_view(padded, 3), axis=1)
    return speed


def _boolean_runs(mask: np.ndarray, timestamps: np.ndarray, minimum_s: float) -> list[np.ndarray]:
    indices = np.flatnonzero(mask)
    if not len(indices):
        return []
    return [
        run for run in np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1)
        if timestamps[run[-1]] - timestamps[run[0]] >= minimum_s
    ]


def _session_locomotion_reference(
    session_id: str, alignment, timestamps: np.ndarray,
    positions: dict[str, np.ndarray], quaternions: dict[str, np.ndarray],
    valid: dict[str, np.ndarray],
) -> dict[str, Any] | None:
    """Measure subject-specific turn/gait timing and contact characteristics."""
    segments = {item.key: item for item in alignment.segments}
    if "turns" not in segments or "walk" not in segments:
        return None
    walk = segments["walk"]
    walk_mask = (timestamps >= walk.active_start_s) & (timestamps <= walk.active_end_s)
    if walk_mask.sum() < 8:
        return None

    lift_values: list[float] = []
    contact_heights: list[float] = []
    stance_speeds: list[float] = []
    step_peaks: list[tuple[float, str]] = []
    for side, role in (("left", "left_ankle"), ("right", "right_ankle")):
        usable = walk_mask & valid[role] & np.isfinite(positions[role]).all(axis=1)
        if usable.sum() < 5:
            lift_values.append(float("nan")); contact_heights.append(float("nan")); stance_speeds.append(float("nan"))
            continue
        height = positions[role][:, 1]
        baseline = float(np.percentile(height[usable], 10))
        lift = max(0.0, float(np.percentile(height[usable], 95) - baseline))
        contact_height = float(np.clip(.30 * lift, .018, .045))
        speed = _horizontal_speed(positions[role], timestamps)
        low = usable & (height <= baseline + contact_height)
        stance_speed = float(np.percentile(speed[low], 90)) if low.sum() >= 3 else float("nan")
        lift_values.append(lift); contact_heights.append(contact_height); stance_speeds.append(stance_speed)
        swing = usable & (height >= baseline + max(.025, .35 * lift))
        for run in _boolean_runs(swing, timestamps, .15):
            peak_index = int(run[np.argmax(height[run])])
            step_peaks.append((float(timestamps[peak_index]), side))

    step_peaks.sort()
    intervals = np.diff([item[0] for item in step_peaks])
    # The forward-to-backward reversal contains a deliberate pause and is not
    # a step interval. Keep only ordinary alternating-step timing.
    intervals = intervals[(intervals >= .35) & (intervals <= 1.80)]
    step_interval = float(np.median(intervals)) if len(intervals) else float("nan")

    root_proxy = (
        positions["camera"] + positions["left_ankle"] + positions["right_ankle"]
    ) / 3.0
    root_speed = _horizontal_speed(root_proxy, timestamps)
    walk_root_speed = float(np.percentile(root_speed[walk_mask], 75))
    ankle_separation = np.linalg.norm(
        positions["left_ankle"] - positions["right_ankle"], axis=1,
    )

    turn = segments["turns"]
    turn_mask = (timestamps >= turn.active_start_s) & (timestamps <= turn.active_end_s)
    yaw_roles = ("camera", "left_ankle", "right_ankle")
    yaw_delta = []
    for role in yaw_roles:
        baseline_mask = (
            (timestamps >= turn.start_s)
            & (timestamps <= min(turn.active_start_s, turn.start_s + 1.0))
            & valid[role]
        )
        if not baseline_mask.any():
            baseline_mask = (timestamps < turn.active_start_s) & valid[role]
        yaw_delta.append(_relative_yaw_from_quaternions_xyzw(
            quaternions[role], baseline_mask, quaternion_order="wxyz",
        ))
    yaw_delta_np = np.stack(yaw_delta, axis=1)
    coherent_yaw = np.median(yaw_delta_np, axis=1)
    angular_speed = np.abs(np.gradient(coherent_yaw, timestamps)) if len(timestamps) > 1 else np.zeros(1)
    turn_errors = np.median(
        np.abs(yaw_delta_np[turn_mask] - coherent_yaw[turn_mask, None]), axis=0,
    )
    inverse_error = 1.0 / np.maximum(turn_errors, np.deg2rad(3.0))
    yaw_weights = inverse_error / inverse_error.sum()

    return {
        "session": session_id,
        "gait": {
            "step_interval_s": step_interval,
            "cadence_steps_per_min": 60.0 / step_interval if np.isfinite(step_interval) else float("nan"),
            "ankle_lift_m": lift_values,
            "contact_height_threshold_m": contact_heights,
            "stance_speed_p90_m_s": stance_speeds,
            "root_speed_p75_m_s": walk_root_speed,
            "ankle_separation_m": float(np.median(ankle_separation[walk_mask])),
            "detected_step_peaks": len(step_peaks),
        },
        "turn": {
            "peak_yaw_deg": float(np.rad2deg(np.percentile(np.abs(coherent_yaw[turn_mask]), 95))),
            "angular_speed_p75_deg_s": float(np.rad2deg(np.percentile(angular_speed[turn_mask], 75))),
            "yaw_tracker_weights": yaw_weights.tolist(),
            "head_to_feet_yaw_rms_deg": float(np.rad2deg(np.sqrt(np.mean(np.square(
                yaw_delta_np[turn_mask, 0] - yaw_delta_np[turn_mask, 1:].mean(axis=1)
            ))))),
        },
    }


def _finite_median(values: list[float], fallback: float) -> float:
    usable = np.asarray(values, dtype=np.float64)
    usable = usable[np.isfinite(usable)]
    return float(np.median(usable)) if len(usable) else float(fallback)


def _aggregate_locomotion_reference(samples: tuple[dict[str, Any], ...] | list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate repeated 16/17 captures into one deployable personal prior."""
    if not samples:
        return {}
    step_interval = _finite_median([item["gait"]["step_interval_s"] for item in samples], 1.0)
    ankle_lift = [
        _finite_median([item["gait"]["ankle_lift_m"][side] for item in samples], .10)
        for side in range(2)
    ]
    height_threshold = _finite_median([
        value for item in samples for value in item["gait"]["contact_height_threshold_m"]
    ], .030)
    stance_speed_p90 = _finite_median([
        value for item in samples for value in item["gait"]["stance_speed_p90_m_s"]
    ], .14)
    speed_threshold = float(np.clip(1.15 * stance_speed_p90, .10, .25))
    raw_yaw_weights = np.median(np.asarray([
        item["turn"]["yaw_tracker_weights"] for item in samples
    ], dtype=np.float64), axis=0)
    # The head is cleanest but can deliberately lead the torso. Never let it
    # overwhelm both planted feet when this prior initializes a future turn;
    # retain the learned left/right foot reliability inside the remaining
    # mass.
    head_weight = float(np.clip(raw_yaw_weights[0], .30, .50))
    foot_ratio = raw_yaw_weights[1:] / raw_yaw_weights[1:].sum()
    left_weight = float(np.clip((1.0 - head_weight) * foot_ratio[0], .20, 1.0 - head_weight - .20))
    yaw_weights = [head_weight, left_weight, 1.0 - head_weight - left_weight]
    return {
        "version": 1,
        "learned_from_protocol_stages": ["16 turns", "17 walk"],
        "source_sessions": [item["session"] for item in samples],
        "gait": {
            "step_interval_s": step_interval,
            "cadence_steps_per_min": 60.0 / step_interval,
            "ankle_lift_m": ankle_lift,
            "root_speed_p75_m_s": _finite_median([
                item["gait"]["root_speed_p75_m_s"] for item in samples
            ], .15),
            "ankle_separation_m": _finite_median([
                item["gait"]["ankle_separation_m"] for item in samples
            ], .25),
        },
        "turn": {
            "peak_yaw_deg": _finite_median([item["turn"]["peak_yaw_deg"] for item in samples], 150.0),
            "angular_speed_p75_deg_s": _finite_median([
                item["turn"]["angular_speed_p75_deg_s"] for item in samples
            ], 70.0),
            "yaw_tracker_order": ["head", "left_foot", "right_foot"],
            "yaw_tracker_weights": yaw_weights,
            "head_to_feet_yaw_rms_deg": _finite_median([
                item["turn"]["head_to_feet_yaw_rms_deg"] for item in samples
            ], 12.0),
        },
        "contact": {
            "height_threshold_m": height_threshold,
            "speed_threshold_m_s": speed_threshold,
            "stance_speed_p90_m_s": stance_speed_p90,
            "recommended_foot_lock_weight": 100.0,
        },
        "per_session": list(samples),
    }


def _geometry_exclusions(
    positions: dict[str, np.ndarray], valid: dict[str, np.ndarray], relative_time: np.ndarray,
) -> tuple[set[str], list[dict[str, Any]], dict[str, float]]:
    neutral = (relative_time >= STATIC_WINDOWS["neutral"][0]) & (relative_time <= STATIC_WINDOWS["neutral"][1])
    medians: dict[str, np.ndarray] = {}
    for role in RAW_ROLES:
        mask = neutral & valid[role] & np.isfinite(positions[role]).all(axis=1)
        if mask.any():
            medians[role] = np.median(positions[role][mask], axis=0)
    excluded: set[str] = set()
    reasons: list[dict[str, Any]] = []
    metrics: dict[str, float] = {}
    if "left_ankle" in medians and "right_ankle" in medians:
        separation = float(np.linalg.norm(medians["left_ankle"] - medians["right_ankle"]))
        metrics["neutral_ankle_separation_m"] = separation
        if not 0.08 <= separation <= 0.80:
            # Decide which side is bad from consistency with head/wrists.  A
            # body-mounted tracker more than 2.3 m from the head cannot belong
            # to this 1.6 m subject.  If both are plausible, discard the more
            # distant one and record the ambiguity.
            head = medians.get("camera")
            if head is not None:
                distances = {role: float(np.linalg.norm(medians[role] - head)) for role in ("left_ankle", "right_ankle")}
                bad = max(distances, key=distances.get)
            else:
                bad = "right_ankle"
            excluded.add(bad)
            reasons.append({"role": bad, "reason": "implausible bilateral ankle separation", "measured_m": separation})
    head = medians.get("camera")
    if head is not None:
        for role, point in medians.items():
            if role == "camera":
                continue
            distance = float(np.linalg.norm(point - head))
            metrics[f"neutral_head_to_{role}_m"] = distance
            if distance > 2.30 and role not in excluded:
                excluded.add(role)
                reasons.append({"role": role, "reason": "tracker farther than 2.30 m from head", "measured_m": distance})
    return excluded, reasons, metrics


def load_calibration_dataset(data_dir: str | Path, *, minimum_samples: int = 3) -> CalibrationDataset:
    """Extract robust static-pose observations from recorder session folders."""
    root = Path(data_dir)
    session_jsons = sorted(root.glob("*/*.session.json"))
    if not session_jsons:
        raise FileNotFoundError(f"no *.session.json files below {root}")
    observations: list[StaticObservation] = []
    excluded_streams: list[dict[str, Any]] = []
    quality: list[dict[str, Any]] = []
    locomotion_samples: list[dict[str, Any]] = []
    sessions: list[str] = []
    for session_path in session_jsons:
        session_dir = session_path.parent
        session = json.loads(session_path.read_text(encoding="utf-8"))
        session_id = str(session.get("base", session_dir.name))
        tracker_name = session.get("tracker_npz")
        tracker_path = session_dir / tracker_name if tracker_name else next(session_dir.glob("*.tracker.npz"))
        with np.load(tracker_path, allow_pickle=False) as values:
            prefix = _npz_prefix(values)
            times = np.asarray(values[f"{prefix}__t_host_us"], dtype=np.int64)
            start = _video_start_us(session_dir, session)
            relative_time = (times - start).astype(np.float64) * 1e-6
            positions = {role: np.asarray(values[f"{prefix}.{role}.position"], dtype=np.float64) for role in RAW_ROLES}
            quaternions = {role: np.asarray(values[f"{prefix}.{role}.orientation"], dtype=np.float64) for role in RAW_ROLES}
            valid = {role: np.asarray(values[f"{prefix}.{role}.valid"], dtype=bool) for role in RAW_ROLES}
        excluded, reasons, metrics = _geometry_exclusions(positions, valid, relative_time)
        pose_masks, selected_windows, window_mode = _static_pose_masks(positions, valid, relative_time)
        for reason in reasons:
            excluded_streams.append({"session": session_id, **reason})
        session_pose_counts: dict[str, dict[str, int]] = {}
        for pose_name in STATIC_WINDOWS:
            pose_positions = np.full((len(TRACKER_NAMES), 3), np.nan, dtype=np.float64)
            pose_rotations = np.tile(np.eye(3, dtype=np.float64), (len(TRACKER_NAMES), 1, 1))
            pose_valid = np.zeros(len(TRACKER_NAMES), dtype=bool)
            sample_counts = np.zeros(len(TRACKER_NAMES), dtype=np.int32)
            position_mad = np.full(len(TRACKER_NAMES), np.nan, dtype=np.float64)
            for role_index, role in enumerate(RAW_ROLES):
                mask = (
                    pose_masks[pose_name] & valid[role]
                    & np.isfinite(positions[role]).all(axis=1) & np.isfinite(quaternions[role]).all(axis=1)
                )
                count = int(mask.sum())
                sample_counts[role_index] = count
                if count < minimum_samples or role in excluded:
                    continue
                samples = positions[role][mask]
                median = np.median(samples, axis=0)
                pose_positions[role_index] = median
                position_mad[role_index] = float(np.median(np.linalg.norm(samples - median, axis=1)))
                pose_rotations[role_index] = _robust_rotation(quaternions[role][mask])
                pose_valid[role_index] = True
            observations.append(StaticObservation(
                session_id, pose_name, pose_name, .5, 1.0,
                pose_positions, pose_rotations, pose_valid, sample_counts, position_mad,
            ))
            session_pose_counts[pose_name] = {TRACKER_NAMES[index]: int(value) for index, value in enumerate(sample_counts)}
        # The reference HTML is now a real calibration contract: align the
        # complete recording and sample the distinctive dynamic phases of
        # 02/06--11, rather than estimating mounts from four static holds only.
        tracker_frames = read_raw_tracker_npz(tracker_path)
        video_offset_s = float((times[0] - start) * 1e-6)
        tracker_holds = {
            key: [float(value[0] - video_offset_s), float(value[1] - video_offset_s)]
            for key, value in selected_windows.items()
        }
        glove_name = session.get("gloves_npz")
        glove_path = session_dir / glove_name if glove_name else None
        alignment = align_protocol(tracker_frames, glove_path=glove_path, static_holds=tracker_holds)
        tracker_time = (times - times[0]).astype(np.float64) * 1e-6
        locomotion_sample = _session_locomotion_reference(
            session_id, alignment, tracker_time, positions, quaternions, valid,
        )
        if locomotion_sample is not None:
            locomotion_samples.append(locomotion_sample)
        dynamic_phases = {
            "head": (.125, .375, .625, .875),
            "elbows": (.125, .375, .625, .875),
            "wrists": (.20, .40, .60, .80),
            "squat": (.25, .75),
            "left_knee": (.25, .75),
            "right_knee": (.25, .75),
            "ankles": (.25, .75),
        }
        alignment_lookup = {item.key: item for item in alignment.segments}
        for motion, phases in dynamic_phases.items():
            segment = alignment_lookup[motion]
            for phase in phases:
                sample_time = segment.active_start_s + phase * (segment.active_end_s - segment.active_start_s)
                local = np.abs(tracker_time - sample_time) <= .11
                if local.sum() < 2:
                    nearest = np.argsort(np.abs(tracker_time - sample_time))[:3]
                    local = np.zeros(len(tracker_time), dtype=bool); local[nearest] = True
                pose_positions = np.full((len(TRACKER_NAMES), 3), np.nan, dtype=np.float64)
                pose_rotations = np.tile(np.eye(3, dtype=np.float64), (len(TRACKER_NAMES), 1, 1))
                pose_valid = np.zeros(len(TRACKER_NAMES), dtype=bool)
                sample_counts = np.zeros(len(TRACKER_NAMES), dtype=np.int32)
                position_mad = np.full(len(TRACKER_NAMES), np.nan, dtype=np.float64)
                for role_index, role in enumerate(RAW_ROLES):
                    mask = local & valid[role] & np.isfinite(positions[role]).all(axis=1) & np.isfinite(quaternions[role]).all(axis=1)
                    sample_counts[role_index] = int(mask.sum())
                    if mask.sum() < 2 or role in excluded:
                        continue
                    samples = positions[role][mask]
                    median = np.median(samples, axis=0)
                    pose_positions[role_index] = median
                    position_mad[role_index] = float(np.median(np.linalg.norm(samples - median, axis=1)))
                    pose_rotations[role_index] = _robust_rotation(quaternions[role][mask])
                    pose_valid[role_index] = True
                pose_name = f"{motion}@{phase:.3f}"
                observations.append(StaticObservation(
                    session_id, pose_name, motion, float(phase), .10,
                    pose_positions, pose_rotations, pose_valid, sample_counts, position_mad,
                ))
                session_pose_counts[pose_name] = {TRACKER_NAMES[index]: int(value) for index, value in enumerate(sample_counts)}
        quality.append({
            "session": session_id,
            "tracker_frames": int(len(relative_time)),
            "protocol_time_range_s": [float(relative_time.min()), float(relative_time.max())],
            "excluded_roles": [ROLE_TO_TRACKER[role] for role in sorted(excluded)],
            "geometry": metrics,
            "static_window_mode": window_mode,
            "static_windows_s": selected_windows,
            "static_sample_counts": session_pose_counts,
            "protocol_alignment": alignment.to_mapping(),
            "locomotion_reference": locomotion_sample,
        })
        sessions.append(session_id)
    return CalibrationDataset(
        tuple(observations), tuple(sessions), tuple(excluded_streams), tuple(quality),
        tuple(locomotion_samples),
    )


def _yaw_matrix(yaw):
    import torch

    c, s = torch.cos(yaw), torch.sin(yaw)
    zero, one = torch.zeros_like(c), torch.ones_like(c)
    return torch.stack((c, zero, s, zero, one, zero, -s, zero, c), dim=-1).reshape(-1, 3, 3)


def _rotation_angle_degrees(predicted: np.ndarray, target: np.ndarray) -> float:
    return math.degrees(rotation_geodesic_angle(predicted, target))


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _body_inputs(library: CalibrationMotionLibrary, pose_specs: tuple[tuple[str, float], ...]):
    frames = [library.frame(motion, phase) for motion, phase in pose_specs]
    return (
        np.stack([frame.body_pose for frame in frames]).astype(np.float32),
        np.stack([frame.left_hand_pose for frame in frames]).astype(np.float32),
        np.stack([frame.right_hand_pose for frame in frames]).astype(np.float32),
    )


def _metric_lengths(joints: np.ndarray, scale: float) -> dict[str, float]:
    def length(first: int, second: int) -> float:
        return float(scale * np.linalg.norm(joints[first] - joints[second]))

    return {
        "shoulder_width_m": length(16, 17),
        "left_upper_arm_m": length(16, 18), "right_upper_arm_m": length(17, 19),
        "left_forearm_m": length(18, 20), "right_forearm_m": length(19, 21),
        "left_thigh_m": length(1, 4), "right_thigh_m": length(2, 5),
        "left_shin_m": length(4, 7), "right_shin_m": length(5, 8),
        "pelvis_to_head_joint_m": length(0, 15),
    }


def calibrate_subject(data_dir: str | Path, options: SubjectCalibrationOptions) -> dict[str, Any]:
    """Fit and export a reusable subject calibration profile."""
    import torch

    if options.height_m <= 0 or options.armspan_m <= 0:
        raise ValueError("height and arm span must be positive metres")
    if options.smplx_prior_weight < 0:
        raise ValueError("SMPL-X prior weight must be non-negative")
    if not 1 <= options.optimized_betas <= 10:
        raise ValueError("optimized betas must be between 1 and 10")
    shape_prior_metadata: dict[str, Any] | None = None
    source_betas_np = np.zeros(10, dtype=np.float32)
    if options.smplx_prior_path is not None:
        source_betas_np, shape_prior_metadata = load_smplx_shape_prior(options.smplx_prior_path)
    dataset = load_calibration_dataset(data_dir)
    output_dir = Path(options.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    library = CalibrationMotionLibrary(
        options.body_model_path, options.vposer_path, cache_path=options.motion_cache_path,
        device=options.device, verbose=options.verbose,
    )
    body = library.body
    device = library.device
    dtype = torch.float32
    motion_order = {item.motion: item.index for item in PROTOCOL}
    pose_spec_by_name = {item.pose_name: (item.motion, item.phase) for item in dataset.observations}
    pose_names = tuple(sorted(
        pose_spec_by_name,
        key=lambda name: (motion_order[pose_spec_by_name[name][0]], pose_spec_by_name[name][1]),
    ))
    pose_specs = tuple(pose_spec_by_name[name] for name in pose_names)
    pose_lookup = {name: index for index, name in enumerate(pose_names)}
    body_pose_np, left_hand_np, right_hand_np = _body_inputs(library, pose_specs)
    body_pose = torch.as_tensor(body_pose_np, dtype=dtype, device=device)
    left_hand = torch.as_tensor(left_hand_np, dtype=dtype, device=device)
    right_hand = torch.as_tensor(right_hand_np, dtype=dtype, device=device)
    count_poses = len(pose_names)
    zero3 = torch.zeros((count_poses, 3), dtype=dtype, device=device)
    zero10 = torch.zeros((count_poses, 10), dtype=dtype, device=device)
    source_betas = torch.as_tensor(source_betas_np, dtype=dtype, device=device)
    joint_ids = torch.as_tensor([TRACKED_JOINTS[name] for name in TRACKER_NAMES], device=device)
    parents = body.parents.to(device)

    target_position = torch.as_tensor(np.stack([item.position for item in dataset.observations]), dtype=dtype, device=device)
    target_rotation_np = np.stack([item.rotation for item in dataset.observations])
    target_valid = torch.as_tensor(np.stack([item.valid for item in dataset.observations]), dtype=torch.bool, device=device)
    observation_weight = torch.as_tensor([item.weight for item in dataset.observations], dtype=dtype, device=device)
    # Squatting changes the pelvis/root height while both feet remain planted.
    # Treat that common world translation as a nuisance variable anchored by
    # the two ankle trackers, rather than forcing one session-wide root offset
    # to explain both standing and crouched poses.
    root_anchor_np = np.zeros((len(dataset.observations), len(TRACKER_NAMES)), dtype=bool)
    for observation_index, observation in enumerate(dataset.observations):
        if observation.motion == "squat":
            root_anchor_np[observation_index, 3:5] = observation.valid[3:5]
    root_anchor = torch.as_tensor(root_anchor_np, dtype=torch.bool, device=device)
    obs_pose_index = torch.as_tensor([pose_lookup[item.pose_name] for item in dataset.observations], device=device)
    session_lookup = {name: index for index, name in enumerate(dataset.sessions)}
    obs_session_index = torch.as_tensor([session_lookup[item.session_id] for item in dataset.observations], device=device)
    session_count = len(dataset.sessions)

    with torch.no_grad():
        baseline = body(
            betas=source_betas[None].expand(count_poses, -1),
            global_orient=zero3, transl=zero3, body_pose=body_pose,
            left_hand_pose=left_hand, right_hand_pose=right_hand, jaw_pose=zero3,
            leye_pose=zero3, reye_pose=zero3, expression=zero10, return_verts=True,
        )
        neutral_index = pose_lookup["neutral"]
        t_pose_index = pose_lookup["t_pose"]
        baseline_height = float((baseline.vertices[neutral_index, :, 1].max() - baseline.vertices[neutral_index, :, 1].min()).cpu())
        baseline_armspan = float((baseline.vertices[t_pose_index, :, 0].max() - baseline.vertices[t_pose_index, :, 0].min()).cpu())
    scale_initial = options.height_m / baseline_height
    log_scale = torch.nn.Parameter(torch.tensor(math.log(scale_initial), dtype=dtype, device=device))
    betas_small = torch.nn.Parameter(source_betas[: options.optimized_betas].clone())
    # Attachment vectors are in the corresponding SMPL-X joint frame, metres.
    attachment_position = torch.nn.Parameter(torch.zeros((len(TRACKER_NAMES), 3), dtype=dtype, device=device))

    # T-pose left-to-right direction gives a stable world yaw initialization.
    yaws = np.zeros(session_count, dtype=np.float32)
    for session_id, session_index in session_lookup.items():
        candidates = [item for item in dataset.observations if item.session_id == session_id and item.pose_name == "t_pose"]
        if candidates and candidates[0].valid[1] and candidates[0].valid[2]:
            direction = candidates[0].position[1] - candidates[0].position[2]
            yaws[session_index] = np.arctan2(-direction[2], direction[0])
    root_yaw = torch.nn.Parameter(torch.as_tensor(yaws, dtype=dtype, device=device))
    root_translation = torch.nn.Parameter(torch.zeros((session_count, 3), dtype=dtype, device=device))

    with torch.no_grad():
        model_joints = baseline.joints[:, :22]
        pelvis = model_joints[:, 0]
        canonical = scale_initial * (model_joints[:, joint_ids] - pelvis[:, None])
        roots = _yaw_matrix(root_yaw)
        for session_index in range(session_count):
            values = []
            for observation_index, observation in enumerate(dataset.observations):
                if session_lookup[observation.session_id] != session_index:
                    continue
                pose_index = pose_lookup[observation.pose_name]
                for tracker_index in np.flatnonzero(observation.valid):
                    predicted = roots[session_index] @ canonical[pose_index, tracker_index]
                    values.append(observation.position[tracker_index] - predicted.cpu().numpy())
            if values:
                root_translation[session_index].copy_(torch.as_tensor(np.median(values, axis=0), dtype=dtype, device=device))

    optimizer = torch.optim.Adam([
        {"params": [log_scale], "lr": .008}, {"params": [betas_small], "lr": .018},
        {"params": [attachment_position], "lr": .012}, {"params": [root_yaw], "lr": .008},
        {"params": [root_translation], "lr": .018},
    ])
    mount_limits = torch.as_tensor((.18, .15, .15, .18, .18), dtype=dtype, device=device)
    losses: list[float] = []
    for iteration in range(options.iterations):
        optimizer.zero_grad(set_to_none=True)
        betas = torch.cat((betas_small, source_betas[options.optimized_betas :]))
        output = body(
            betas=betas[None].expand(count_poses, -1), global_orient=zero3, transl=zero3,
            body_pose=body_pose, left_hand_pose=left_hand, right_hand_pose=right_hand,
            jaw_pose=zero3, leye_pose=zero3, reye_pose=zero3, expression=zero10, return_verts=True,
        )
        scale = torch.exp(log_scale)
        joints = output.joints[:, :22]
        pelvis = joints[:, 0]
        joint_rotation = _global_body_rotations(zero3, body_pose.reshape(count_poses, 21, 3), parents)[:, joint_ids]
        canonical = scale * (joints[:, joint_ids] - pelvis[:, None])
        canonical = canonical + (joint_rotation @ attachment_position[None, :, :, None]).squeeze(-1)
        root_rotation = _yaw_matrix(root_yaw)
        selected = canonical[obs_pose_index]
        base_predicted = root_translation[obs_session_index, None] + (
            root_rotation[obs_session_index, None] @ selected[:, :, :, None]
        ).squeeze(-1)
        anchor_error = torch.where(
            root_anchor[:, :, None], target_position - base_predicted,
            torch.zeros_like(base_predicted),
        )
        anchor_count = root_anchor.sum(dim=1, keepdim=True).to(dtype).clamp_min(1.0)
        observation_root_correction = anchor_error.sum(dim=1) / anchor_count
        observation_root_correction = torch.where(
            root_anchor.any(dim=1, keepdim=True), observation_root_correction,
            torch.zeros_like(observation_root_correction),
        )
        predicted = base_predicted + observation_root_correction[:, None]
        mask = target_valid[:, :, None]
        # Invalid streams are represented by NaN in the extracted table.
        # Multiplication by a false mask would still leave NaN in IEEE
        # arithmetic, so remove them explicitly before the robust loss.
        position_error = torch.where(mask, predicted - target_position, torch.zeros_like(predicted))
        weighted = mask * observation_weight[:, None, None]
        position_loss = (_huber(position_error, .025) * weighted).sum() / (weighted.sum().clamp_min(1) * 3.0)
        height = scale * (output.vertices[neutral_index, :, 1].max() - output.vertices[neutral_index, :, 1].min())
        armspan = scale * (output.vertices[t_pose_index, :, 0].max() - output.vertices[t_pose_index, :, 0].min())
        measurement_loss = (height - options.height_m).square() + (armspan - options.armspan_m).square()
        mount_norm = torch.linalg.vector_norm(attachment_position, dim=1)
        mount_bound = torch.relu(mount_norm - mount_limits).square().mean()
        # Bilateral mounts should have comparable radii even though their
        # joint-local axes may be mirrored.
        mount_symmetry = (mount_norm[1] - mount_norm[2]).square() + (mount_norm[3] - mount_norm[4]).square()
        # The ankle units are strapped to the lateral lower leg, not the
        # instep or the anterior shin. Express this prior in the canonical
        # neutral world's anatomical axes; the raw SMPL joint-local axes are
        # not directly interpretable as left/up/forward.
        neutral_ankle_offset = (
            joint_rotation[neutral_index, 3:5] @ attachment_position[3:5, :, None]
        ).squeeze(-1)
        desired_ankle_offset = torch.as_tensor(
            ((.075, .070, 0.0), (-.075, .070, 0.0)), dtype=dtype, device=device,
        )
        ankle_anatomy = (neutral_ankle_offset - desired_ankle_offset).square().mean()
        generic_shape_prior = betas.square().mean()
        video_shape_prior = (betas_small - source_betas[: options.optimized_betas]).square().mean()
        scale_prior = (log_scale - math.log(scale_initial)).square()
        yaw_prior = root_yaw.square().mean()
        loss = (
            420.0 * position_loss + 1800.0 * measurement_loss + 280.0 * mount_bound
            + 3.0 * mount_symmetry + 12.0 * ankle_anatomy
            + (.012 * generic_shape_prior + options.smplx_prior_weight * video_shape_prior
               if shape_prior_metadata is not None else .045 * generic_shape_prior)
            + .18 * scale_prior + .015 * yaw_prior
        )
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            log_scale.clamp_(math.log(.72), math.log(1.15))
            betas_small.clamp_(-3.0, 3.0)
            root_yaw.clamp_(-math.pi, math.pi)
        losses.append(float(loss.detach().cpu()))
        if options.verbose and (iteration == 0 or (iteration + 1) % 100 == 0):
            print(
                f"subject calibration {iteration + 1:4d}/{options.iterations}: "
                f"loss={losses[-1]:.5f}, scale={float(scale.detach()):.4f}, "
                f"height={float(height.detach()):.4f}, armspan={float(armspan.detach()):.4f}"
            )

    with torch.no_grad():
        betas_tensor = torch.cat((betas_small, source_betas[options.optimized_betas :]))
        output = body(
            betas=betas_tensor[None].expand(count_poses, -1), global_orient=zero3, transl=zero3,
            body_pose=body_pose, left_hand_pose=left_hand, right_hand_pose=right_hand,
            jaw_pose=zero3, leye_pose=zero3, reye_pose=zero3, expression=zero10, return_verts=True,
        )
        scale_value = float(torch.exp(log_scale).cpu())
        joints = output.joints[:, :22]
        pelvis = joints[:, 0]
        joint_rotation = _global_body_rotations(zero3, body_pose.reshape(count_poses, 21, 3), parents)[:, joint_ids]
        canonical = scale_value * (joints[:, joint_ids] - pelvis[:, None])
        canonical = canonical + (joint_rotation @ attachment_position[None, :, :, None]).squeeze(-1)
        roots = _yaw_matrix(root_yaw)
        selected = canonical[obs_pose_index]
        base_predicted = root_translation[obs_session_index, None] + (
            roots[obs_session_index, None] @ selected[:, :, :, None]
        ).squeeze(-1)
        anchor_error = torch.where(
            root_anchor[:, :, None], target_position - base_predicted,
            torch.zeros_like(base_predicted),
        )
        anchor_count = root_anchor.sum(dim=1, keepdim=True).to(dtype).clamp_min(1.0)
        observation_root_correction = anchor_error.sum(dim=1) / anchor_count
        observation_root_correction = torch.where(
            root_anchor.any(dim=1, keepdim=True), observation_root_correction,
            torch.zeros_like(observation_root_correction),
        )
        predicted = base_predicted + observation_root_correction[:, None]
        vertices_np = output.vertices.cpu().numpy()
        joints_np = output.joints.cpu().numpy()
        canonical_np = canonical.cpu().numpy()
        predicted_np = predicted.cpu().numpy()
        root_rotations_np = roots.cpu().numpy()
        root_translations_np = root_translation.cpu().numpy()
        betas_np = betas_tensor.cpu().numpy()
        attachment_positions_np = attachment_position.cpu().numpy()
        joint_rotations_np = joint_rotation.cpu().numpy()
        observation_root_corrections_np = observation_root_correction.cpu().numpy()
        estimated_height = scale_value * float(vertices_np[neutral_index, :, 1].max() - vertices_np[neutral_index, :, 1].min())
        estimated_armspan = scale_value * float(vertices_np[t_pose_index, :, 0].max() - vertices_np[t_pose_index, :, 0].min())

    # Source recorder rotations are standard active local-to-world matrices.
    # A fixed mount therefore satisfies tracker_world = joint_world @ mount.
    # Solve the shared tracker-in-joint rotation by SO(3) averaging after the
    # independent position/shape fit.
    attachment_rotations = np.zeros((len(TRACKER_NAMES), 3, 3), dtype=np.float64)
    rotation_residual_samples: list[list[float]] = [[] for _ in TRACKER_NAMES]
    for tracker_index in range(len(TRACKER_NAMES)):
        candidates = []
        candidate_metadata = []
        for observation_index, observation in enumerate(dataset.observations):
            if not observation.valid[tracker_index]:
                continue
            # Absolute hand-mount orientation remains underconstrained by a
            # roughly time-aligned dynamic phase.  Keep its robust static
            # solution; the reconstruction uses mount-independent relative
            # tracker rotations throughout 06/07. Head and ankle dynamics are
            # sufficiently distinctive to refine their absolute mounts.
            if tracker_index in (1, 2) and observation.weight < .9:
                continue
            session_index = session_lookup[observation.session_id]
            pose_index = pose_lookup[observation.pose_name]
            joint_world = root_rotations_np[session_index] @ joint_rotations_np[pose_index, tracker_index]
            mount_joint = joint_world.T @ target_rotation_np[observation_index, tracker_index]
            candidates.append(mount_joint)
            candidate_metadata.append((observation_index, session_index, pose_index))
        if candidates:
            retained = list(candidates)
            for _ in range(3):
                estimate = average_rotations(retained)
                angles = np.asarray([_rotation_angle_degrees(estimate, item) for item in retained])
                median = float(np.median(angles)); mad = float(np.median(np.abs(angles - median)))
                threshold = max(12.0, median + 2.5 * max(mad, 1.0))
                filtered = [item for item, angle in zip(retained, angles, strict=True) if angle <= threshold]
                if len(filtered) < 4 or len(filtered) == len(retained):
                    break
                retained = filtered
            attachment_rotations[tracker_index] = average_rotations(retained)
        else:
            attachment_rotations[tracker_index] = np.eye(3)
        for observation_index, session_index, pose_index in candidate_metadata:
            joint_world = root_rotations_np[session_index] @ joint_rotations_np[pose_index, tracker_index]
            predicted_rotation = joint_world @ attachment_rotations[tracker_index]
            rotation_residual_samples[tracker_index].append(
                _rotation_angle_degrees(predicted_rotation, target_rotation_np[observation_index, tracker_index])
            )

    residual_rows: list[dict[str, Any]] = []
    residual_values: list[float] = []
    static_residual_values: list[float] = []
    per_tracker_values: dict[str, list[float]] = {name: [] for name in TRACKER_NAMES}
    per_tracker_protocol_values: dict[str, list[float]] = {name: [] for name in TRACKER_NAMES}
    for observation_index, observation in enumerate(dataset.observations):
        for tracker_index, tracker_name in enumerate(TRACKER_NAMES):
            if not observation.valid[tracker_index]:
                continue
            residual = float(np.linalg.norm(predicted_np[observation_index, tracker_index] - observation.position[tracker_index]))
            residual_values.append(residual); per_tracker_protocol_values[tracker_name].append(residual)
            if observation.weight >= .9:
                static_residual_values.append(residual); per_tracker_values[tracker_name].append(residual)
            residual_rows.append({
                "session": observation.session_id, "pose": observation.pose_name, "tracker": tracker_name,
                "motion": observation.motion, "phase": observation.phase, "weight": observation.weight,
                "position_residual_m": residual, "samples": int(observation.sample_count[tracker_index]),
                "position_mad_m": float(observation.position_mad_m[tracker_index]),
            })

    tracker_summary = {}
    for tracker_index, tracker_name in enumerate(TRACKER_NAMES):
        values = per_tracker_values[tracker_name]
        angles = rotation_residual_samples[tracker_index]
        tracker_summary[tracker_name] = {
            "position_rms_m": float(np.sqrt(np.mean(np.square(values)))) if values else None,
            "position_median_m": float(np.median(values)) if values else None,
            "full_protocol_position_rms_m": (
                float(np.sqrt(np.mean(np.square(per_tracker_protocol_values[tracker_name]))))
                if per_tracker_protocol_values[tracker_name] else None
            ),
            "rotation_median_deg": float(np.median(angles)) if angles else None,
            "mount_offset_norm_m": float(np.linalg.norm(attachment_positions_np[tracker_index])),
            "rotation_confidence": (
                "high" if angles and float(np.median(angles)) < 10.0
                else "medium" if angles and float(np.median(angles)) < 25.0 else "low"
            ),
        }

    warnings = []
    if dataset.excluded_streams:
        warnings.append(
            f"{len(dataset.excluded_streams)} tracker stream(s) failed geometry validation and were excluded; see excluded_streams"
        )
    low_rotation = [name for name, values in tracker_summary.items() if values["rotation_confidence"] == "low"]
    if low_rotation:
        warnings.append(
            "mount rotations are low-confidence for " + ", ".join(low_rotation)
            + "; fit the time-aligned elbow/wrist dynamic segments before using these rotations downstream"
        )
    forward_counts = [
        int(item.sample_count[index]) for item in dataset.observations if item.pose_name == "forward"
        for index in range(len(TRACKER_NAMES)) if item.valid[index]
    ]
    if forward_counts and float(np.median(forward_counts)) < 10:
        warnings.append("forward-pose holds have fewer than 10 tracker samples per stream and carry lower confidence")

    profile: dict[str, Any] = {
        "format": "five_tracker_smplx_subject_profile_v5_wxyz_source_columns",
        "coordinate_convention": {
            "position_axes": POSITION_AXES,
            "quaternion_order": "xyzw",
            "tracker_rotation_action": TRACKER_ROTATION_ACTION,
            "left_right_semantics": "anatomical subject side; never camera/image side",
        },
        "source_data": str(Path(data_dir).resolve()),
        "model": {"type": "smplx", "gender": "neutral", "num_betas": 10},
        "measurements": {"height_m": options.height_m, "armspan_m": options.armspan_m},
        "shape_prior": {
            "enabled": shape_prior_metadata is not None,
            "source": shape_prior_metadata,
            "source_betas": source_betas_np,
            "weight": options.smplx_prior_weight if shape_prior_metadata is not None else 0.0,
            "optimized_beta_indices": list(range(options.optimized_betas)),
            "locked_to_video_beta_indices": (
                list(range(options.optimized_betas, 10)) if shape_prior_metadata is not None else []
            ),
            "source_mesh_height_unscaled": baseline_height,
            "source_mesh_armspan_unscaled": baseline_armspan,
            "height_initialized_scale": scale_initial,
            "height_scaled_source_armspan_m": scale_initial * baseline_armspan,
        },
        "fit": {
            "scale": scale_value, "betas": betas_np,
            "estimated_mesh_height_m": estimated_height, "estimated_mesh_armspan_m": estimated_armspan,
            "position_rms_m": float(np.sqrt(np.mean(np.square(static_residual_values)))) if static_residual_values else None,
            "full_protocol_position_rms_m": float(np.sqrt(np.mean(np.square(residual_values)))) if residual_values else None,
            "final_loss": losses[-1], "iterations": options.iterations,
        },
        "body_dimensions": _metric_lengths(joints_np[neutral_index], scale_value),
        "tracker_mounts": {
            name: {
                "joint_index": TRACKED_JOINTS[name],
                "position_in_joint_frame_m": attachment_positions_np[index],
                "neutral_world_offset_m": (
                    joint_rotations_np[neutral_index, index] @ attachment_positions_np[index]
                ),
                "rotation_joint_tracker_xyzw": matrix_to_quaternion_xyzw(attachment_rotations[index]),
                "rotation_convention": TRACKER_ROTATION_ACTION,
            }
            for index, name in enumerate(TRACKER_NAMES)
        },
        "sessions": {
            name: {
                "root_yaw_rad": float(root_yaw.detach().cpu().numpy()[index]),
                "root_rotation": root_rotations_np[index], "root_translation_m": root_translations_np[index],
            }
            for index, name in enumerate(dataset.sessions)
        },
        "locomotion_reference": _aggregate_locomotion_reference(dataset.locomotion_samples),
        "quality": {
            "tracker_summary": tracker_summary, "excluded_streams": dataset.excluded_streams,
            "warnings": warnings, "sessions": dataset.session_quality, "protocol_residuals": residual_rows,
        },
    }
    json_path = output_dir / "subject_profile.json"
    json_path.write_text(json.dumps(profile, ensure_ascii=False, indent=2, default=_jsonable), encoding="utf-8")
    npz_path = output_dir / "subject_profile.npz"
    np.savez_compressed(
        npz_path, scale=np.asarray(scale_value, np.float32), betas=betas_np.astype(np.float32),
        source_shape_betas=source_betas_np.astype(np.float32),
        attachment_position=attachment_positions_np.astype(np.float32),
        attachment_rotation=attachment_rotations.astype(np.float32),
        session_names=np.asarray(dataset.sessions), root_rotation=root_rotations_np.astype(np.float32),
        root_translation=root_translations_np.astype(np.float32), pose_names=np.asarray(pose_names),
    )
    html_path = output_dir / "subject_calibration.html"
    _write_calibration_html(
        html_path, dataset, pose_names, vertices_np, np.asarray(body.faces, dtype=np.int32),
        scale_value, joints_np[:, 0], canonical_np, root_rotations_np, root_translations_np,
        observation_root_corrections_np,
    )
    profile["outputs"] = {"json": str(json_path), "npz": str(npz_path), "html": str(html_path)}
    return profile


def _write_calibration_html(
    path: Path, dataset: CalibrationDataset, pose_names: tuple[str, ...], vertices: np.ndarray,
    faces: np.ndarray, scale: float, pelvis: np.ndarray, predicted_canonical: np.ndarray,
    root_rotations: np.ndarray, root_translations: np.ndarray,
    observation_root_corrections: np.ndarray,
) -> None:
    """Write a draggable four-pose mesh/observation diagnostic viewer."""
    session_lookup = {name: index for index, name in enumerate(dataset.sessions)}
    pose_lookup = {name: index for index, name in enumerate(pose_names)}
    frames = []
    tracker_colors = ("#fb7185", "#22d3ee", "#818cf8", "#fbbf24", "#a3e635")
    for pose_index, pose_name in enumerate(pose_names):
        mesh = scale * (vertices[pose_index] - pelvis[pose_index])
        # Every diagnostic pose is expressed pelvis-relative. Anchor its own
        # sole minimum to the ground so squat/knee poses move the head and
        # pelvis down instead of making both feet and the whole body float.
        ground_shift = -float(mesh[:, 1].min())
        mesh[:, 1] += ground_shift
        predicted = predicted_canonical[pose_index].copy(); predicted[:, 1] += ground_shift
        mesh_display = mesh @ DISPLAY_ROTATION.T
        predicted_display = predicted @ DISPLAY_ROTATION.T
        observed_points, observed_text, observed_colors = [], [], []
        residual_x, residual_y, residual_z = [], [], []
        for observation_index, observation in enumerate(dataset.observations):
            if observation.pose_name != pose_name:
                continue
            session_index = session_lookup[observation.session_id]
            for tracker_index in np.flatnonzero(observation.valid):
                point = root_rotations[session_index].T @ (
                    observation.position[tracker_index]
                    - root_translations[session_index]
                    - observation_root_corrections[observation_index]
                )
                point[1] += ground_shift
                point_display = DISPLAY_ROTATION @ point
                observed_points.append(point_display)
                observed_text.append(f"{observation.session_id}<br>{TRACKER_NAMES[tracker_index]}")
                observed_colors.append(tracker_colors[tracker_index])
                residual_x.extend((float(predicted_display[tracker_index, 0]), float(point_display[0]), None))
                residual_y.extend((float(predicted_display[tracker_index, 1]), float(point_display[1]), None))
                residual_z.extend((float(predicted_display[tracker_index, 2]), float(point_display[2]), None))
        observed_array = np.asarray(observed_points).reshape(-1, 3)
        motion, phase = next((item.motion, item.phase) for item in dataset.observations if item.pose_name == pose_name)
        definition = next(item for item in PROTOCOL if item.motion == motion)
        phase_label = _DYNAMIC_PHASE_LABELS.get((motion, round(phase, 3)))
        display_title = (
            definition.title if pose_name == motion else
            f"{definition.title} · {phase_label or f'phase {phase:.3f}'}"
        )
        frames.append({
            "name": pose_name, "title": display_title, "vertices": mesh_display.round(5).tolist(),
            "predicted": predicted_display.round(5).tolist(),
            "observed": observed_array.round(5).tolist() if len(observed_array) else [],
            "observed_text": observed_text, "observed_colors": observed_colors,
            "residual": {"x": residual_x, "y": residual_y, "z": residual_z},
        })
    payload = json.dumps({
        "frames": frames, "faces": faces.tolist(), "tracker_names": TRACKER_NAMES,
        "tracker_colors": tracker_colors, "sessions": dataset.sessions,
    }, ensure_ascii=False, separators=(",", ":"))
    html = """<!doctype html><html lang=\"zh-CN\"><head><meta charset=\"utf-8\">
<title>受试者 SMPL-X 联合标定</title><script src=\"https://cdn.plot.ly/plotly-2.35.2.min.js\"></script>
<style>body{margin:0;background:#0b1020;color:#e5e7eb;font-family:-apple-system,BlinkMacSystemFont,\"PingFang SC\",sans-serif}#bar{padding:12px 18px;background:#111827;display:flex;gap:18px;align-items:center}button{background:#334155;color:white;border:0;border-radius:6px;padding:8px 13px;cursor:pointer}.active{background:#2563eb}#plot{height:calc(100vh - 61px)}.hint{color:#94a3b8;font-size:13px}</style></head>
<body><div id=\"bar\"><span>受试者尺度 / 体型 / 安装偏置联合标定</span><span id=\"buttons\"></span><span class=\"hint\">无镜像：正面时受试者左=画面右。实心点=模型预测，叉号=各次采集观测；线段=残差。</span></div><div id=\"plot\"></div>
<script>const data=__PAYLOAD__;
function traces(k){const f=data.frames[k],v=f.vertices,p=f.predicted,o=f.observed,r=f.residual,F=data.faces;return [
{type:'mesh3d',x:v.map(q=>q[0]),y:v.map(q=>q[1]),z:v.map(q=>q[2]),i:F.map(q=>q[0]),j:F.map(q=>q[1]),k:F.map(q=>q[2]),color:'#94a3b8',opacity:.68,flatshading:false,name:'SMPL-X'},
{type:'scatter3d',mode:'markers+text',x:p.map(q=>q[0]),y:p.map(q=>q[1]),z:p.map(q=>q[2]),text:data.tracker_names,textposition:'top center',marker:{size:6,color:data.tracker_colors},name:'拟合 tracker'},
{type:'scatter3d',mode:'markers',x:o.map(q=>q[0]),y:o.map(q=>q[1]),z:o.map(q=>q[2]),text:f.observed_text,hovertemplate:'%{text}<extra></extra>',marker:{size:5,symbol:'x',color:f.observed_colors,line:{width:2}},name:'采集观测'},
{type:'scatter3d',mode:'lines',x:r.x,y:r.y,z:r.z,line:{color:'#f97316',width:4},name:'位置残差'},
{type:'mesh3d',x:[-1,1,1,-1],y:[-.8,-.8,.8,.8],z:[0,0,0,0],i:[0,0],j:[1,2],k:[2,3],color:'#1e3a5f',opacity:.35,name:'地面'}];}
const layout={margin:{l:0,r:0,t:34,b:0},paper_bgcolor:'#0b1020',plot_bgcolor:'#0b1020',font:{color:'#e5e7eb'},title:{text:'',font:{size:16}},legend:{x:.01,y:.99},scene:{xaxis:{title:'受试者左 +X / 右 -X',range:[-.9,.9],backgroundcolor:'#0b1020',gridcolor:'#334155'},yaxis:{title:'身体后方 +Y / 前方 -Y',range:[-.75,.65],backgroundcolor:'#0b1020',gridcolor:'#334155'},zaxis:{title:'向上 +Z',range:[0,1.75],backgroundcolor:'#0b1020',gridcolor:'#334155'},aspectmode:'manual',aspectratio:{x:1.15,y:.9,z:1.75},camera:{eye:{x:1.25,y:-2.25,z:1.25},up:{x:0,y:0,z:1}}}};
function show(k){layout.title.text=data.frames[k].title;Plotly.react('plot',traces(k),layout,{responsive:true});document.querySelectorAll('button').forEach((b,i)=>b.classList.toggle('active',i===k));}
data.frames.forEach((f,k)=>{const b=document.createElement('button');b.textContent=f.title;b.onclick=()=>show(k);document.getElementById('buttons').appendChild(b)});show(0);</script></body></html>"""
    path.write_text(html.replace("__PAYLOAD__", payload), encoding="utf-8")
