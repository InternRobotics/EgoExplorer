"""External-data WebGL viewer for the complete SMPL-X mesh topology."""

from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from .manus_batch import read_smplx_npz
from .model import ModelPaths, load_body_model
from .coordinates import DISPLAY_ROTATION, recorded_tracker_axes_in_world
from .profile_batch import (
    _BODY_EDGES, _TRACKED_IDS, _session_files, _session_id,
    _unlabelled_segment, _viewer_selection,
)
from .profile_fitting import SubjectProfile
from .protocol import align_protocol
from .behavior_protocol import refine_alignment_by_behavior
from .raw_trackers import read_raw_tracker_npz
from .subject_calibration import TRACKER_NAMES


# Backward-compatible private name used by earlier tests/imports.
_DISPLAY_ROTATION = DISPLAY_ROTATION


_CONSTRAINT_VERTEX_JOINT_GROUPS = (
    # neck, head, jaw and eyes
    (12, 15, 22, 23, 24),
    # collars, shoulders, elbows, wrists and all articulated hand joints
    tuple(range(13, 15)) + tuple(range(16, 22)) + tuple(range(25, 55)),
    # hips, knees, ankles and feet
    (1, 2, 4, 5, 7, 8, 10, 11),
)


def _constraint_vertex_weights(lbs_weights: np.ndarray) -> np.ndarray:
    """Return head/wrist/foot endpoint influence for local viewer blending."""
    weights = np.asarray(lbs_weights, dtype=np.float32)
    if weights.ndim != 2 or weights.shape[1] < 55:
        raise ValueError("SMPL-X LBS weights must have shape [vertices, >=55]")
    grouped = np.stack([
        weights[:, np.asarray(joints, dtype=np.int64)].sum(axis=1)
        for joints in _CONSTRAINT_VERTEX_JOINT_GROUPS
    ], axis=1)
    # The joint groups are disjoint, but guard against malformed weights so
    # the ungrouped/core contribution can never become negative in WebGL.
    total = grouped.sum(axis=1, keepdims=True)
    grouped /= np.maximum(total, 1.0)
    return np.clip(np.rint(grouped * 255.0), 0, 255).astype(np.uint8)


def _default_camera_azimuth(joints: np.ndarray) -> float:
    """Return a front-view camera azimuth from initial anatomical sides."""
    values = np.asarray(joints, dtype=np.float64)
    if values.ndim != 3 or values.shape[1] < 18 or values.shape[2] != 3:
        raise ValueError("joints must have shape [T,>=18,3]")
    initial_count = min(20, len(values))
    lateral = np.mean(
        .5 * (
            (values[:initial_count, 1, :2] - values[:initial_count, 2, :2])
            + (values[:initial_count, 16, :2] - values[:initial_count, 17, :2])
        ),
        axis=0,
    )
    lateral /= max(float(np.linalg.norm(lateral)), 1e-8)
    subject_forward = np.asarray((lateral[1], -lateral[0]))
    return float(np.arctan2(subject_forward[0], subject_forward[1]))


def _quantize_temporal(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Quantize vertices and delta-code time while retaining full topology."""
    center = .5 * (vertices.min(axis=(0, 1)) + vertices.max(axis=(0, 1)))
    centered = vertices - center
    extent = float(np.max(np.abs(centered)))
    frame_delta = float(np.max(np.abs(np.diff(centered, axis=0)))) if len(vertices) > 1 else 0.0
    scale = max(1e-5, extent / 30000.0, frame_delta / 30000.0)
    quantized = np.rint(centered / scale).astype(np.int32)
    delta = quantized.copy()
    if len(delta) > 1:
        delta[1:] -= quantized[:-1]
    maximum = int(np.max(np.abs(delta)))
    if maximum > np.iinfo(np.int16).max:
        raise ValueError(f"temporal mesh delta exceeds int16: {maximum}")
    return delta.astype("<i2"), center.astype(np.float32), scale


def _write_gzip(path: Path, values: np.ndarray | bytes) -> None:
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, compresslevel=6, mtime=0) as packed:
            packed.write(values if isinstance(values, bytes) else values.tobytes(order="C"))


def _normalise_body_roots(
    value: str | Path | Sequence[str | Path],
) -> tuple[Path, ...]:
    """Accept one result root or a per-dataset collection of result roots."""
    raw = (value,) if isinstance(value, (str, Path)) else tuple(value)
    roots = tuple(Path(item) for item in raw)
    if not roots:
        raise ValueError("at least one body output root is required")
    return roots


def _session_output_dir(roots: Sequence[Path], session_id: str) -> Path | None:
    """Find the result directory for one session across combined datasets."""
    for root in roots:
        candidate = root / session_id
        if (
            (candidate / "smplx_with_manus_hands.npz").exists()
            or (candidate / "smplx_profile_driven.npz").exists()
        ):
            return candidate
    return None


_DATASET_COMMON_KEYS = (
    "dataset_id", "source", "frame_count", "source_frame_count",
    "source_indices", "timestamps", "action_indices", "protocol_segments",
    "protocol_mode", "tracker", "tracker_rotation", "valid", "range",
    "default_camera_azimuth_rad", "maximum_gap_s",
)


def _compact_manifest_datasets(sequences: Sequence[dict]) -> list[dict]:
    """Deduplicate observations shared by all fit variants of a dataset."""
    grouped: dict[str, list[dict]] = {}
    for sequence in sequences:
        grouped.setdefault(str(sequence["dataset_id"]), []).append(sequence)
    datasets = []
    for dataset_id, variants in grouped.items():
        reference = next(
            (item for item in variants if item["variant_role"] == "natural"),
            variants[0],
        )
        for item in variants[1:]:
            if (
                item["frame_count"] != reference["frame_count"]
                or item["source_indices"] != reference["source_indices"]
            ):
                raise ValueError(f"{dataset_id}: fit variants do not share Viewer frames")
        common = {key: reference[key] for key in _DATASET_COMMON_KEYS}
        ranges = np.asarray([item["range"] for item in variants], dtype=np.float32)
        common["range"] = np.stack((
            ranges[:, :, 0].min(axis=0), ranges[:, :, 1].max(axis=0),
        ), axis=1).round(4).tolist()
        common["variants"] = [
            {
                key: value for key, value in item.items()
                if key not in _DATASET_COMMON_KEYS and key != "residual"
            }
            for item in variants
        ]
        datasets.append(common)
    return datasets


def write_external_full_mesh_viewer(
    data_roots: Sequence[str | Path],
    body_output_dir: str | Path | Sequence[str | Path],
    subject_profile: str | Path,
    output_dir: str | Path,
    *,
    body_model_path: str | Path = "models/smplx",
    device: str = "cpu",
    max_frames: int = 600,
    protocol_mode: str = "auto",
    hard_body_output_dir: str | Path | Sequence[str | Path] | None = None,
    sole_body_output_dir: str | Path | Sequence[str | Path] | None = None,
    hard_sole_body_output_dir: str | Path | Sequence[str | Path] | None = None,
    floor_reference_output_dir: str | Path | Sequence[str | Path] | None = None,
    skip_missing_natural: bool = False,
    reuse_existing_meshes: bool = False,
) -> dict:
    """Write a small HTML player plus external full-resolution mesh streams."""
    import torch

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if protocol_mode not in {"auto", "calibration", "none"}:
        raise ValueError(f"unsupported protocol mode: {protocol_mode}")
    body_roots = [("natural", _normalise_body_roots(body_output_dir))]
    if hard_body_output_dir is not None:
        body_roots.append(("hard", _normalise_body_roots(hard_body_output_dir)))
    if sole_body_output_dir is not None:
        body_roots.append(("natural_sole", _normalise_body_roots(sole_body_output_dir)))
    if hard_sole_body_output_dir is not None:
        if sole_body_output_dir is None:
            raise ValueError("hard sole-plane results require natural sole-plane results")
        body_roots.append(("hard_sole", _normalise_body_roots(hard_sole_body_output_dir)))
    floor_reference_roots = (
        _normalise_body_roots(floor_reference_output_dir)
        if floor_reference_output_dir is not None else body_roots[0][1]
    )
    profile = SubjectProfile.load(subject_profile)
    body = load_body_model(ModelPaths(Path(body_model_path), None, "smplx", "neutral"), device).eval()
    for parameter in body.parameters():
        parameter.requires_grad_(False)

    faces = np.asarray(body.faces, dtype="<u4")
    faces_path = output / "smplx_faces.u32"
    faces_path.write_bytes(faces.tobytes(order="C"))
    constraint_weights = _constraint_vertex_weights(
        body.lbs_weights.detach().cpu().numpy()
    )
    constraint_weights_path = output / "constraint_vertex_weights.u8"
    constraint_weights_path.write_bytes(constraint_weights.tobytes(order="C"))
    sequences = []
    for tracker_path in _session_files(data_roots):
        session_id = _session_id(tracker_path)
        natural_session_dir = _session_output_dir(body_roots[0][1], session_id)
        if natural_session_dir is None:
            if skip_missing_natural:
                continue
            raise FileNotFoundError(
                f"no natural body result for {session_id} below {body_roots[0][1]}"
            )
        frames = tuple(read_raw_tracker_npz(tracker_path))
        glove_path = tracker_path.with_name(f"{session_id}.npz")
        natural_trajectory_path = natural_session_dir / "smplx_with_manus_hands.npz"
        if not natural_trajectory_path.exists():
            natural_trajectory_path = natural_session_dir / "smplx_profile_driven.npz"
        natural_trajectory = read_smplx_npz(natural_trajectory_path)
        has_protocol_marker = (natural_session_dir / "protocol_alignment.json").exists()
        should_align_protocol = (
            protocol_mode == "calibration" or
            (protocol_mode == "auto" and has_protocol_marker)
        )
        protocol = (
            align_protocol(frames, glove_path=glove_path if glove_path.exists() else None)
            if should_align_protocol else None
        )
        if protocol is not None:
            protocol = refine_alignment_by_behavior(frames, protocol)
        floor_session_dir = _session_output_dir(floor_reference_roots, session_id)
        if floor_session_dir is None:
            raise FileNotFoundError(
                f"no floor-reference body result for {session_id} below {floor_reference_roots}"
            )
        base_summary_path = floor_session_dir / "smplx_profile_driven.json"
        if not base_summary_path.exists():
            base_summary_path = floor_session_dir / "smplx_with_manus_hands.json"
        common_floor = float(json.loads(base_summary_path.read_text(encoding="utf-8"))["ground_y_m"])
        for variant_role, variant_roots in body_roots:
            session_dir = _session_output_dir(variant_roots, session_id)
            # A comparison root may intentionally cover only some datasets.
            # Keep those sequences usable and disable the slider only for the
            # individual session that lacks a second real inference.
            if session_dir is None and variant_role != "natural":
                continue
            if session_dir is None:
                raise FileNotFoundError(
                    f"no {variant_role} body result for {session_id} below {variant_roots}"
                )
            variant_id = session_id if len(body_roots) == 1 else f"{session_id}__{variant_role}"
            mesh_name = f"{variant_id}.mesh.i16.gz"
            sequence_metadata_path = output / f"{variant_id}.sequence.json"
            trajectory_path = session_dir / "smplx_with_manus_hands.npz"
            if not trajectory_path.exists():
                trajectory_path = session_dir / "smplx_profile_driven.npz"
            if not trajectory_path.exists():
                raise FileNotFoundError(trajectory_path)
            cache_source_mtime_ns = max(
                trajectory_path.stat().st_mtime_ns,
                natural_trajectory_path.stat().st_mtime_ns,
                (session_dir / "tracker_diagnostics.npz").stat().st_mtime_ns,
                base_summary_path.stat().st_mtime_ns,
            )
            if reuse_existing_meshes and (output / mesh_name).exists() and sequence_metadata_path.exists():
                cached = json.loads(sequence_metadata_path.read_text(encoding="utf-8"))
                if int(cached.get("_cache_source_mtime_ns", -1)) >= cache_source_mtime_ns:
                    sequences.append(cached)
                    continue
            trajectory = read_smplx_npz(trajectory_path)
            if (
                len(trajectory.timestamps) != len(natural_trajectory.timestamps)
                or not np.allclose(
                    trajectory.timestamps, natural_trajectory.timestamps,
                    rtol=0.0, atol=1e-5,
                )
            ):
                raise ValueError(
                    f"{session_id}: natural/hard trajectories do not share source timestamps"
                )
            selection = (
                np.arange(len(trajectory.timestamps), dtype=np.int64)
                if max_frames <= 0 else
                _viewer_selection(len(trajectory.timestamps), max_frames, trajectory.timestamps, protocol)
            )
            summary = json.loads(trajectory_path.with_suffix(".json").read_text(encoding="utf-8"))
            tracker = np.stack([[frames[index][name].position for name in TRACKER_NAMES] for index in selection]).astype(np.float32)
            tracker_rotation = np.stack([[frames[index][name].rotation for name in TRACKER_NAMES] for index in selection]).astype(np.float32)
            with np.load(session_dir / "tracker_diagnostics.npz", allow_pickle=False) as diagnostics:
                valid = diagnostics["valid"][selection].copy()
                predicted = diagnostics["predicted_tracker"][selection].copy()
                residual = diagnostics["position_residual_m"][selection].copy()
            for values in (tracker, predicted):
                values[..., 1] -= common_floor
            tracker = tracker @ _DISPLAY_ROTATION.T
            predicted = predicted @ _DISPLAY_ROTATION.T
            # Recorder quaternion matrices act on row vectors; their rows are
            # the physical Tracker axes. Convert to a conventional column-axis
            # matrix before applying the world-to-view rotation.
            tracker_rotation = np.einsum(
                "ij,ftjk->ftik", _DISPLAY_ROTATION,
                recorded_tracker_axes_in_world(tracker_rotation, convention="columns"),
            )

            anchors = np.concatenate((tracker, predicted), axis=1)
            anchor_lower = np.nanmin(anchors, axis=(0, 1)) - .55
            anchor_upper = np.nanmax(anchors, axis=(0, 1)) + .55
            center = .5 * (anchor_lower + anchor_upper)
            # Centered absolute coordinates need roughly half the complete
            # range, while a temporal delta can span the complete range when
            # a sparse Viewer selection jumps between distant poses.  Use the
            # latter bound so int16 remains valid even for very small
            # viewer_max_frames smoke exports.
            scale = max(1e-5, float(np.max(anchor_upper - anchor_lower)) / 30000.0)
            joints = np.empty((len(selection), 22, 3), dtype=np.float32)
            betas = torch.as_tensor(trajectory.betas, dtype=torch.float32, device=device)[None]
            previous_quantized = None
            with (output / mesh_name).open("wb") as raw:
                with gzip.GzipFile(filename="", mode="wb", fileobj=raw, compresslevel=6, mtime=0) as packed:
                    for start in range(0, len(selection), 64):
                        selected = selection[start:start + 64]
                        count = len(selected)
                        zero3 = torch.zeros((count, 3), device=device)
                        zero10 = torch.zeros((count, 10), device=device)
                        with torch.no_grad():
                            result = body(
                                betas=betas.expand(count, -1),
                                global_orient=torch.as_tensor(trajectory.global_orient[selected], dtype=torch.float32, device=device),
                                transl=zero3,
                                body_pose=torch.as_tensor(trajectory.body_pose[selected], dtype=torch.float32, device=device),
                                # Tracker hardness changes only the sparse-body
                                # IK endpoint. Keep the independently fitted
                                # MANUS fingers identical across both endpoints
                                # so the Viewer slider cannot close the hands.
                                left_hand_pose=torch.as_tensor(natural_trajectory.left_hand_pose[selected], dtype=torch.float32, device=device),
                                right_hand_pose=torch.as_tensor(natural_trajectory.right_hand_pose[selected], dtype=torch.float32, device=device),
                                jaw_pose=zero3, leye_pose=zero3, reye_pose=zero3, expression=zero10,
                                return_verts=True,
                            )
                        translation = np.asarray(trajectory.transl[selected], dtype=np.float32)
                        vertices = trajectory.scale * result.vertices.cpu().numpy() + translation[:, None]
                        chunk_joints = trajectory.scale * result.joints[:, :22].cpu().numpy() + translation[:, None]
                        vertices[..., 1] -= common_floor; chunk_joints[..., 1] -= common_floor
                        vertices = vertices @ _DISPLAY_ROTATION.T
                        chunk_joints = chunk_joints @ _DISPLAY_ROTATION.T
                        joints[start:start + count] = chunk_joints
                        quantized = np.rint((vertices - center) / scale).astype(np.int32)
                        if np.max(np.abs(quantized)) > 32767:
                            raise ValueError(f"{variant_id}: full-mesh quantizer range exceeded")
                        delta = quantized.copy()
                        if previous_quantized is not None:
                            delta[0] -= previous_quantized
                        if count > 1:
                            delta[1:] -= quantized[:-1]
                        if np.max(np.abs(delta)) > 32767:
                            raise ValueError(f"{variant_id}: full-mesh temporal delta exceeded")
                        packed.write(delta.astype("<i2").tobytes(order="C"))
                        previous_quantized = quantized[-1].copy()
            visible = np.concatenate((joints, np.where(valid[:, :, None], tracker, np.nan)), axis=1)
            lower = np.nanmin(visible, axis=(0, 1)); upper = np.nanmax(visible, axis=(0, 1))
            padding = np.maximum(.12, .08 * (upper - lower)); lower -= padding; upper += padding
            lower[2] = min(-.03, lower[2])
            # Place the default camera in front of the actual first-frame
            # body, not in front of the fixed calibration/world +Z axis. Raw
            # recordings may begin at an arbitrary root yaw.
            default_camera_azimuth = _default_camera_azimuth(joints)
            sequence_metadata = {
                "id": variant_id, "dataset_id": session_id, "variant_role": variant_role,
                "tracker_hardness": float(summary.get("tracker_hardness", .5)),
                "sole_floor_constraint": bool(
                    summary.get("sole_floor_constraint", {}).get("enabled", False)
                    if isinstance(summary.get("sole_floor_constraint"), dict) else
                    summary.get("walking_fit_parameters", {}).get(
                        "sole_floor_constraint", False,
                    )
                ),
                "tracker_constraint_multipliers": summary.get(
                    "tracker_constraint_multipliers",
                    {"head": 1.0, "wrists": 1.0, "feet": 1.0},
                ),
                "_cache_source_mtime_ns": cache_source_mtime_ns,
                "hand_pose_source": str(natural_trajectory_path.resolve()),
                "source": str(tracker_path), "mesh_file": mesh_name,
                "frame_count": int(len(selection)), "source_frame_count": int(len(trajectory.timestamps)),
                "source_indices": selection.tolist(), "timestamps": trajectory.timestamps[selection].round(6).tolist(),
                "action_indices": (
                    protocol.labels(trajectory.timestamps[selection]).tolist()
                    if protocol is not None else [0] * len(selection)
                ),
                "protocol_segments": (
                    protocol.to_mapping()["segments"] if protocol is not None else
                    [_unlabelled_segment(trajectory.timestamps)]
                ),
                "protocol_mode": "calibration" if protocol is not None else "unlabelled_inference",
                "mesh_origin": center.round(7).tolist(), "mesh_scale": float(scale),
                "joints": joints.round(5).tolist(), "tracker": tracker.round(5).tolist(),
                "tracker_rotation": tracker_rotation.round(6).tolist(), "predicted": predicted.round(5).tolist(),
                "valid": valid.tolist(), "residual": np.where(np.isfinite(residual), residual, -1).round(5).tolist(),
                "range": np.stack((lower, upper), axis=1).round(4).tolist(),
                "default_camera_azimuth_rad": default_camera_azimuth,
                "position_rms_m": float(summary["position_rms_m"]),
                "maximum_gap_s": float(summary["maximum_gap_s"]),
            }
            sequences.append(sequence_metadata)
            sequence_metadata_path.write_text(
                json.dumps(sequence_metadata, ensure_ascii=False, separators=(",", ":")),
                encoding="utf-8",
            )

    manifest = {
        "format": "five_tracker_smplx_full_mesh_v2",
        "vertex_count": int(body.v_template.shape[0]),
        "face_count": int(len(faces)),
        "faces_file": faces_path.name,
        "constraint_vertex_weights_file": constraint_weights_path.name,
        "tracker_names": list(TRACKER_NAMES),
        "tracker_joint_ids": _TRACKED_IDS.tolist(),
        "edges": [list(edge) for edge in _BODY_EDGES],
        "datasets": _compact_manifest_datasets(sequences),
    }
    manifest_path = output / "manifest.json.gz"
    _write_gzip(manifest_path, json.dumps(manifest, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
    (output / "index.html").write_text(_HTML, encoding="utf-8")
    (output / "README.txt").write_text(
        "完整 SMPL-X WebGL 播放器\n\n"
        "请在本目录启动 HTTP 服务：\n"
        "  python3 -m http.server 8000\n\n"
        "然后访问：http://localhost:8000/\n"
        "浏览器出于安全策略不能在 file:// 页面中读取旁边的压缩数据。\n",
        encoding="utf-8",
    )
    report = {
        "viewer": str((output / "index.html").resolve()),
        "manifest": str(manifest_path.resolve()),
        "vertex_count": manifest["vertex_count"],
        "face_count": manifest["face_count"],
        "display_frames": {item["id"]: item["frame_count"] for item in sequences},
        "source_frames": {item["id"]: item["source_frame_count"] for item in sequences},
        "protocol_mode": protocol_mode,
        "hardness_variants": len(body_roots),
        "sole_plane_variants": int(sole_body_output_dir is not None) + int(
            hard_sole_body_output_dir is not None
        ),
        "floor_reference_output_dir": ";".join(
            str(path.resolve()) for path in floor_reference_roots
        ),
    }
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


_HTML = r'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>完整 SMPL-X Mesh</title>
<style>*{box-sizing:border-box}body{margin:0;background:#09101d;color:#e5e7eb;font:14px -apple-system,BlinkMacSystemFont,"PingFang SC",sans-serif;overflow:hidden}#bar{min-height:82px;display:flex;gap:9px;align-items:center;align-content:center;flex-wrap:wrap;padding:7px 13px;background:#111827;border-bottom:1px solid #334155}select,button{color:#e5e7eb;background:#243146;border:1px solid #475569;border-radius:5px;padding:6px 9px}#frame{width:min(440px,30vw)}#hardness{width:170px;vertical-align:middle}.groupConstraint{width:92px;vertical-align:middle}.constraintValue{display:inline-block;min-width:38px;color:#fbbf24}input[type=range]{accent-color:#3b82f6}input:disabled{opacity:.42}#status{height:34px;padding:7px 14px;color:#cbd5e1;background:#0f172a;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.tag{color:#93c5fd}canvas{display:block;width:100vw;height:calc(100vh - 116px)}label{white-space:nowrap}#orientation{position:fixed;right:12px;bottom:10px;z-index:2;padding:7px 10px;border-radius:6px;background:rgba(15,23,42,.82);color:#bfdbfe;border:1px solid #3b82f6;pointer-events:none}#error{display:none;position:fixed;inset:120px 20px auto;padding:18px;background:#7f1d1d;border-radius:8px;z-index:3}</style></head><body>
<div id="bar"><span>序列</span><select id="sequence"></select><span>动作</span><select id="action"></select><button id="play">播放</button><span>速度</span><select id="speed"><option value=".25">0.25×</option><option value=".5">0.5×</option><option value="1" selected>1× 原速</option><option value="2">2×</option></select><input id="frame" type="range" min="0" step="1"><span id="time"></span><label>Tracker 总约束（平移+旋转） <input id="hardness" type="range" min="0" max="100" value="0" step="1"><span id="hardnessValue" class="constraintValue">0%</span></label><label>头 <input id="headConstraint" class="groupConstraint" type="range" min="0" max="100" value="0" step="1"><span id="headConstraintValue" class="constraintValue">0%</span></label><label>腕 <input id="wristConstraint" class="groupConstraint" type="range" min="0" max="100" value="0" step="1"><span id="wristConstraintValue" class="constraintValue">0%</span></label><label>脚 <input id="footConstraint" class="groupConstraint" type="range" min="0" max="100" value="0" step="1"><span id="footConstraintValue" class="constraintValue">0%</span></label><label><input id="soleFloor" type="checkbox">支撑期脚底平面贴地</label><label><input id="mesh" type="checkbox" checked>完整 Mesh</label><label><input id="axes" type="checkbox">Tracker 三轴</label><label><input id="residual" type="checkbox" checked>残差线</label><label><input id="follow" type="checkbox" checked>相机跟随</label><button id="reset">重置视角</button></div><div id="status">正在读取 manifest…</div><div id="orientation">相机按当前序列首帧人体朝向置于正面：受试者左 = 画面右（无镜像）</div><div id="error"></div><canvas id="view"></canvas>
<script>
const $=id=>document.getElementById(id),canvas=$('view'),status=$('status'),err=$('error'),seqSel=$('sequence'),actionSel=$('action'),speedSel=$('speed'),slider=$('frame'),hardness=$('hardness'),hardnessValue=$('hardnessValue'),soleFloor=$('soleFloor'),time=$('time');let D,S,S0,S1,currentDataset,faces,constraintWeights,meshDelta0,meshDelta1,meshPos,mix=0,groupMix={head:0,wrists:0,feet:0},frame=0,playing=false,playWall=0,playTime=0,az=Math.PI,el=.20,dist=2.6,target=[0,0,.8],drag=false,px=0,py=0,manifestSignature='';const meshCache=new Map(),groupControls={head:[$('headConstraint'),$('headConstraintValue')],wrists:[$('wristConstraint'),$('wristConstraintValue')],feet:[$('footConstraint'),$('footConstraintValue')]};
const gl=canvas.getContext('webgl2',{antialias:true,alpha:false});if(!gl)throw Error('浏览器不支持 WebGL2');
const VS=`#version 300 es\nin vec3 p;uniform mat4 mvp;uniform float pointSize;out vec3 worldPosition;void main(){worldPosition=p;gl_Position=mvp*vec4(p,1.0);gl_PointSize=pointSize;}`;
const FS=`#version 300 es\nprecision highp float;in vec3 worldPosition;uniform vec4 color;uniform vec3 eye;uniform float lit;out vec4 outColor;void main(){if(lit<.5){outColor=color;return;}vec3 N=normalize(cross(dFdx(worldPosition),dFdy(worldPosition)));if(!gl_FrontFacing)N=-N;vec3 V=normalize(eye-worldPosition);vec3 key=normalize(vec3(-.45,-.55,.85));vec3 fill=normalize(vec3(.65,.25,.45));float kd=max(dot(N,key),0.0),fd=max(dot(N,fill),0.0);float hemi=.5+.5*N.z;float rim=pow(1.0-max(dot(N,V),0.0),2.2);vec3 H=normalize(key+V);float spec=pow(max(dot(N,H),0.0),32.0);float shade=.25+.23*hemi+.55*kd+.18*fd;vec3 rgb=color.rgb*shade+vec3(.30,.24,.20)*spec+vec3(.20,.32,.48)*rim*.35;float bands=.985+sin(worldPosition.z*28.0)*.015;outColor=vec4(rgb*bands,color.a);}`;
function shader(type,src){const s=gl.createShader(type);gl.shaderSource(s,src);gl.compileShader(s);if(!gl.getShaderParameter(s,gl.COMPILE_STATUS))throw Error(gl.getShaderInfoLog(s));return s}const prog=gl.createProgram();gl.attachShader(prog,shader(gl.VERTEX_SHADER,VS));gl.attachShader(prog,shader(gl.FRAGMENT_SHADER,FS));gl.linkProgram(prog);gl.useProgram(prog);const locP=gl.getAttribLocation(prog,'p'),locM=gl.getUniformLocation(prog,'mvp'),locC=gl.getUniformLocation(prog,'color'),locPS=gl.getUniformLocation(prog,'pointSize');
const meshVAO=gl.createVertexArray(),meshVBO=gl.createBuffer(),meshEBO=gl.createBuffer(),lineVAO=gl.createVertexArray(),lineVBO=gl.createBuffer();gl.bindVertexArray(meshVAO);gl.bindBuffer(gl.ARRAY_BUFFER,meshVBO);gl.enableVertexAttribArray(locP);gl.vertexAttribPointer(locP,3,gl.FLOAT,false,0,0);gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,meshEBO);gl.bindVertexArray(lineVAO);gl.bindBuffer(gl.ARRAY_BUFFER,lineVBO);gl.enableVertexAttribArray(locP);gl.vertexAttribPointer(locP,3,gl.FLOAT,false,0,0);
function perspective(fovy,a,n,f){const t=1/Math.tan(fovy/2),o=new Float32Array(16);o[0]=t/a;o[5]=t;o[10]=(f+n)/(n-f);o[11]=-1;o[14]=2*f*n/(n-f);return o}function norm(a){const l=Math.hypot(...a)||1;return a.map(x=>x/l)}function cross(a,b){return[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]]}function dot(a,b){return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]}function look(eye,c,up){const z=norm(eye.map((x,i)=>x-c[i])),x=norm(cross(up,z)),y=cross(z,x),o=new Float32Array(16);o.set([x[0],y[0],z[0],0,x[1],y[1],z[1],0,x[2],y[2],z[2],0,-dot(x,eye),-dot(y,eye),-dot(z,eye),1]);return o}function mul(a,b){const o=new Float32Array(16);for(let c=0;c<4;c++)for(let r=0;r<4;r++)for(let k=0;k<4;k++)o[c*4+r]+=a[k*4+r]*b[c*4+k];return o}
function camera(){const ce=Math.cos(el),eye=[target[0]+dist*ce*Math.sin(az),target[1]+dist*ce*Math.cos(az),target[2]+dist*Math.sin(el)];return{mvp:mul(perspective(.72,canvas.width/canvas.height,.02,50),look(eye,target,[0,0,1])),eye}}
const locLit=gl.getUniformLocation(prog,'lit'),locEye=gl.getUniformLocation(prog,'eye');function color(hex,a=1){return[(hex>>16&255)/255,(hex>>8&255)/255,(hex&255)/255,a]}function drawBuffer(mode,values,c,size=1){if(!values.length)return;gl.bindVertexArray(lineVAO);gl.bindBuffer(gl.ARRAY_BUFFER,lineVBO);gl.bufferData(gl.ARRAY_BUFFER,new Float32Array(values),gl.DYNAMIC_DRAW);gl.uniform1f(locLit,0);gl.uniform4fv(locC,color(c));gl.uniform1f(locPS,size);gl.drawArrays(mode,0,values.length/3)}function segment(out,a,b){out.push(...a,...b)}
function blend3(a,b,f){return[a[0]+f*(b[0]-a[0]),a[1]+f*(b[1]-a[1]),a[2]+f*(b[2]-a[2])]}
function jointConstraint(i){if(i===12||i===15)return groupMix.head;if([13,14,16,17,18,19,20,21].includes(i))return groupMix.wrists;if([1,2,4,5,7,8,10,11].includes(i))return groupMix.feet;return mix}
function trackerConstraint(i){return i===0?groupMix.head:(i<3?groupMix.wrists:groupMix.feet)}
function currentPoints(key){return S0[key][frame].map((p,i)=>blend3(p,S1[key][frame][i],key==='predicted'?trackerConstraint(i):jointConstraint(i)))}
function unpackFrame(f){const n=D.vertex_count,off=f*n*3,o0=S0.mesh_origin,q0=S0.mesh_scale,o1=S1.mesh_origin,q1=S1.mesh_scale;for(let v=0;v<n;v++){const wh=constraintWeights[v*3]/255,ww=constraintWeights[v*3+1]/255,wf=constraintWeights[v*3+2]/255,bf=mix*(1-wh-ww-wf)+groupMix.head*wh+groupMix.wrists*ww+groupMix.feet*wf;for(let c=0;c<3;c++){const i=v*3+c,a=o0[c]+q0*meshDelta0[off+i],b=o1[c]+q1*meshDelta1[off+i];meshPos[i]=a+bf*(b-a)}}gl.bindBuffer(gl.ARRAY_BUFFER,meshVBO);gl.bufferSubData(gl.ARRAY_BUFFER,0,meshPos)}
function render(){resize();gl.clearColor(.035,.063,.114,1);gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);gl.enable(gl.DEPTH_TEST);gl.useProgram(prog);const cam=camera();gl.uniformMatrix4fv(locM,false,cam.mvp);gl.uniform3fv(locEye,cam.eye);if($('mesh').checked){gl.bindVertexArray(meshVAO);gl.uniform1f(locLit,1);gl.uniform4fv(locC,color(0xd3a08b,1));gl.uniform1f(locPS,1);gl.disable(gl.CULL_FACE);gl.drawElements(gl.TRIANGLES,D.face_count*3,gl.UNSIGNED_INT,0)}const j=currentPoints('joints'),t=S.tracker[frame],pred=currentPoints('predicted'),valid=S.valid[frame],sk=[],mount=[],res=[],grid=[];for(const e of D.edges)segment(sk,j[e[0]],j[e[1]]);const gx=Math.round(target[0]/.2)*.2,gy=Math.round(target[1]/.2)*.2;for(let k=-10;k<=10;k++){segment(grid,[gx+k*.2,gy-2,0],[gx+k*.2,gy+2,0]);segment(grid,[gx-2,gy+k*.2,0],[gx+2,gy+k*.2,0])}for(let n=0;n<5;n++)if(valid[n]){segment(mount,j[D.tracker_joint_ids[n]],pred[n]);if($('residual').checked)segment(res,pred[n],t[n])}gl.lineWidth(1);drawBuffer(gl.LINES,grid,0x27415f);drawBuffer(gl.LINES,sk,0x172033);drawBuffer(gl.LINES,mount,0x2563eb);drawBuffer(gl.LINES,res,0xf97316);for(let n=0;n<5;n++)if(valid[n]){drawBuffer(gl.POINTS,t[n], [0xf43f5e,0x22d3ee,0x818cf8,0xfbbf24,0xa3e635][n],9);drawBuffer(gl.POINTS,pred[n],0xf8fafc,5)}if($('axes').checked)for(let a=0;a<3;a++){const ax=[];for(let n=0;n<5;n++)if(valid[n]){const p=t[n],r=S.tracker_rotation[frame][n],b=[p[0]+.11*r[0][a],p[1]+.11*r[1][a],p[2]+.11*r[2][a]];segment(ax,p,b)}drawBuffer(gl.LINES,ax,[0xef4444,0x22c55e,0x3b82f6][a])}}
function update(){frame=Math.max(0,Math.min(+slider.value,S.frame_count-1));slider.value=frame;unpackFrame(frame);const a=S.protocol_segments[S.action_indices[frame]],pred=currentPoints('predicted'),vals=[];for(let n=0;n<5;n++)if(S.valid[frame][n])vals.push(Math.hypot(pred[n][0]-S.tracker[frame][n][0],pred[n][1]-S.tracker[frame][n][1],pred[n][2]-S.tracker[frame][n][2]));const rms=Math.sqrt(vals.reduce((x,y)=>x+y*y,0)/Math.max(vals.length,1)),endpointMix=(groupMix.head+2*groupMix.wrists+2*groupMix.feet)/5,sequenceRms=(1-endpointMix)*S0.position_rms_m+endpointMix*S1.position_rms_m,c=S0.tracker_constraint_multipliers||{head:1,wrists:1,feet:1},allEndpoint=[mix,groupMix.head,groupMix.wrists,groupMix.feet].every(x=>x===0)||[mix,groupMix.head,groupMix.wrists,groupMix.feet].every(x=>x===1);actionSel.value=S.action_indices[frame];time.textContent=`源帧 ${S.source_indices[frame]}/${S.source_frame_count-1} · ${S.timestamps[frame].toFixed(2)}s`;hardnessValue.textContent=`${Math.round(mix*100)}%`;for(const[k,[input,value]]of Object.entries(groupControls))value.textContent=`${Math.round(groupMix[k]*100)}%`;status.innerHTML=`<span class=tag>${a.title}</span>　完整拓扑 ${D.vertex_count} 顶点 / ${D.face_count} 三角面　当前残差 ${(rms*100).toFixed(2)} cm　端点序列 RMS 约 ${(sequenceRms*100).toFixed(2)} cm　脚底平面 <b>${soleFloor.checked?'开':'关'}</b>　拟合倍率 头 ${(+c.head).toFixed(2)}× / 腕 ${(+c.wrists).toFixed(2)}× / 脚 ${(+c.feet).toFixed(2)}×　<span class=tag>${allEndpoint?'真实推理解':'局部/中间档为两次真实推理解的交互预览'}</span>`;if($('follow').checked){const h=S.tracker[frame][0];target=[h[0],h[1],.85]}render()}
async function gunzip(url){const r=await fetch(url);if(!r.ok)throw Error(`${url}: HTTP ${r.status}`);if(!('DecompressionStream'in window))throw Error('浏览器不支持 gzip 流解压，请升级 Safari/Chrome');return await new Response(r.body.pipeThrough(new DecompressionStream('gzip'))).arrayBuffer()}
async function loadMesh(v){if(!meshCache.has(v.id))meshCache.set(v.id,(async()=>{const buf=await gunzip(v.mesh_file),d=new Int16Array(buf),stride=D.vertex_count*3;for(let f=1;f<v.frame_count;f++){const o=f*stride,p=o-stride;for(let k=0;k<stride;k++)d[o+k]+=d[p+k]}return d})());return meshCache.get(v.id)}
async function activateVariants(resetFrame=false){const variants=currentDataset.variants,useSole=soleFloor.checked,baseRole=useSole?'natural_sole':'natural',hardRole=useSole?'hard_sole':'hard',raw0=variants.find(v=>v.variant_role===baseRole)||variants.find(v=>v.variant_role==='natural')||variants[0],raw1=variants.find(v=>v.variant_role===hardRole)||raw0;S0={...currentDataset,...raw0};S1=raw1===raw0?S0:{...currentDataset,...raw1};if(S0.frame_count!==S1.frame_count||S0.source_indices.some((v,k)=>v!==S1.source_indices[k]))throw Error(`${S0.dataset_id}: 两个硬度端点的帧选择不一致`);S=S0;status.textContent=`正在加载${useSole?'脚底严格贴地':'基础'}自然档与 Tracker 硬约束档完整 mesh…`;[meshDelta0,meshDelta1]=await Promise.all([loadMesh(S0),S1===S0?Promise.resolve(null):loadMesh(S1)]);if(S1===S0)meshDelta1=meshDelta0;hardness.disabled=S1===S0;Object.values(groupControls).forEach(([input])=>input.disabled=S1===S0);const stride=D.vertex_count*3;meshPos=new Float32Array(stride);gl.bindBuffer(gl.ARRAY_BUFFER,meshVBO);gl.bufferData(gl.ARRAY_BUFFER,meshPos.byteLength,gl.DYNAMIC_DRAW);slider.max=S.frame_count-1;if(resetFrame){slider.value=0;frame=0}else slider.value=Math.min(frame,S.frame_count-1);update();syncClock()}
async function loadSequence(i){playing=false;$('play').textContent='播放';currentDataset=D.datasets[i];const variants=currentDataset.variants,rawBase=variants.find(v=>v.variant_role==='natural')||variants[0],base={...currentDataset,...rawBase},hasSole=!!variants.find(v=>v.variant_role==='natural_sole');soleFloor.disabled=!hasSole;soleFloor.checked=hasSole;hardness.value=0;mix=0;for(const[k,[input]]of Object.entries(groupControls)){input.value=0;groupMix[k]=0}S=base;slider.max=S.frame_count-1;slider.value=0;actionSel.innerHTML='';S.protocol_segments.forEach((a,k)=>{const o=document.createElement('option');o.value=k;o.textContent=a.title;actionSel.appendChild(o)});const r=S.range;target=[(r[0][0]+r[0][1])/2,(r[1][0]+r[1][1])/2,.85];dist=Math.max(2.9,(r[2][1]-r[2][0])*1.45);az=S.default_camera_azimuth_rad??Math.PI;await activateVariants(true)}
function normalizeDatasets(value){if(value.datasets)return value.datasets;const groups=new Map;value.sequences.forEach(s=>{if(!groups.has(s.dataset_id))groups.set(s.dataset_id,{dataset_id:s.dataset_id,protocol_mode:s.protocol_mode,variants:[]});groups.get(s.dataset_id).variants.push(s)});return[...groups.values()]}
function datasetSignature(datasets){return datasets.map(s=>`${s.dataset_id}:${s.variants.map(v=>v.variant_role).sort().join(',')}`).join('|')}
function populateSequences(selectedId){seqSel.innerHTML='';D.datasets.forEach((s,i)=>{const o=document.createElement('option');o.value=i;o.textContent=`[${s.protocol_mode==='calibration'?'标定':'测试'}] ${s.dataset_id}`;seqSel.appendChild(o)});const found=D.datasets.findIndex(s=>s.dataset_id===selectedId);return found>=0?found:Math.max(0,D.datasets.length-1)}
async function refreshManifest(){const next=JSON.parse(new TextDecoder().decode(await gunzip(`manifest.json.gz?refresh=${Date.now()}`)));next.datasets=normalizeDatasets(next);const signature=datasetSignature(next.datasets);if(!next.datasets.length||signature===manifestSignature)return;const selectedId=currentDataset?.dataset_id;D.datasets=next.datasets;manifestSignature=signature;const index=populateSequences(selectedId);seqSel.value=index;await loadSequence(index)}
async function init(){try{D=JSON.parse(new TextDecoder().decode(await gunzip('manifest.json.gz')));const[f,w]=await Promise.all([fetch(D.faces_file),fetch(D.constraint_vertex_weights_file)]);if(!f.ok||!w.ok)throw Error('完整 Mesh 索引或分组蒙皮权重读取失败');faces=new Uint32Array(await f.arrayBuffer());constraintWeights=new Uint8Array(await w.arrayBuffer());if(constraintWeights.length!==D.vertex_count*3)throw Error('分组蒙皮权重维度错误');gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,meshEBO);gl.bufferData(gl.ELEMENT_ARRAY_BUFFER,faces,gl.STATIC_DRAW);D.datasets=normalizeDatasets(D);manifestSignature=datasetSignature(D.datasets);populateSequences(D.datasets[0]?.dataset_id);await loadSequence(0);setInterval(()=>refreshManifest().catch(()=>{}),5000)}catch(e){err.style.display='block';err.innerHTML=`<b>数据读取失败：</b>${e.message}<br><br>请不要直接双击 file:// 页面。在此目录运行：<code>python3 -m http.server 8000</code>，然后访问 <code>http://localhost:8000/</code>。`;status.textContent='加载失败'}}
function resize(){const d=devicePixelRatio||1,w=Math.floor(canvas.clientWidth*d),h=Math.floor(canvas.clientHeight*d);if(canvas.width!==w||canvas.height!==h){canvas.width=w;canvas.height=h;gl.viewport(0,0,w,h)}}function syncClock(now=performance.now()){if(!S)return;playWall=now;playTime=S.timestamps[frame]}canvas.onpointerdown=e=>{drag=true;px=e.clientX;py=e.clientY;canvas.setPointerCapture(e.pointerId)};canvas.onpointermove=e=>{if(!drag)return;az-=(e.clientX-px)*.008;el=Math.max(-.25,Math.min(1.25,el+(e.clientY-py)*.008));px=e.clientX;py=e.clientY;render()};canvas.onpointerup=()=>drag=false;canvas.onwheel=e=>{e.preventDefault();dist=Math.max(.5,Math.min(10,dist*Math.exp(e.deltaY*.001)));render()};slider.oninput=()=>{update();syncClock()};hardness.oninput=()=>{mix=+hardness.value/100;for(const[k,[input]]of Object.entries(groupControls)){input.value=hardness.value;groupMix[k]=mix}update();syncClock()};for(const[k,[input]]of Object.entries(groupControls))input.oninput=()=>{groupMix[k]=+input.value/100;update();syncClock()};soleFloor.onchange=async()=>{try{soleFloor.disabled=true;await activateVariants(false)}catch(e){err.style.display='block';err.textContent=e.message}finally{soleFloor.disabled=!currentDataset.variants.find(v=>v.variant_role==='natural_sole')}};seqSel.onchange=()=>loadSequence(+seqSel.value);actionSel.onchange=()=>{const i=+actionSel.value,a=S.protocol_segments[i],mid=(a.active_start_s+a.active_end_s)/2;let b=0,d=1e9;S.timestamps.forEach((t,k)=>{if(S.action_indices[k]===i&&Math.abs(t-mid)<d){b=k;d=Math.abs(t-mid)}});slider.value=b;update();syncClock()};speedSel.onchange=()=>syncClock();for(const id of['mesh','axes','residual'])$(id).onchange=render;$('follow').onchange=update;$('reset').onclick=()=>{az=S.default_camera_azimuth_rad??Math.PI;el=.20;dist=Math.max(2.9,(S.range[2][1]-S.range[2][0])*1.45);$('follow').checked=true;update()};$('play').onclick=()=>{playing=!playing;$('play').textContent=playing?'暂停':'播放';syncClock()};function tick(now){if(playing&&S){const wanted=playTime+(now-playWall)/1000*(+speedSel.value);if(frame+1<S.frame_count&&wanted>=S.timestamps[frame+1]){slider.value=frame+1;update()}else if(frame+1>=S.frame_count&&wanted>S.timestamps[S.frame_count-1]){frame=0;slider.value=0;update();syncClock(now)}}requestAnimationFrame(tick)}requestAnimationFrame(tick);window.onresize=render;init();
</script></body></html>'''
