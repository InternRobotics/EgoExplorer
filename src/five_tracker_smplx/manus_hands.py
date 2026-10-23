"""Offline MANUS raw-skeleton to SMPL-X/MANO hand fitting and inspection.

This adapter consumes the recorder's native ``*.npz`` glove product.  MANUS
returns global node transforms with the wrist at the origin; the recorder stores
the quaternion as xyzw.  We first remove the wrist orientation, then solve the
15 MANO joint rotations from the 20 matching nodes/tips.  The four non-thumb
metacarpal nodes remain useful visualization and palm-shape anchors, but are not
MANO articulation targets.
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .coordinates import DISPLAY_ROTATION
from .device import select_device
from .hands import (
    GLOVE_NODE_COUNTS,
    GLOVE_POINT_IDS,
    MANO_FINGER_ORDER,
    MANUS_POINT_IDS,
    SMPLX_HAND_JOINTS,
    SMPLX_WRIST_JOINT,
    glove_target_indices,
    manus_target_indices,
)
from .model import ModelPaths, load_body_model


MANUS_HIERARCHY = np.asarray(
    (-1, 0, 1, 2, 3, 0, 5, 6, 7, 8, 0, 10, 11, 12, 13, 0, 15, 16, 17, 18, 0, 20, 21, 22, 23),
    dtype=np.int64,
)
MANUS_NODE_NAMES = (
    "wrist",
    "thumb_metacarpal", "thumb_proximal", "thumb_distal", "thumb_tip",
    "index_metacarpal", "index_proximal", "index_intermediate", "index_distal", "index_tip",
    "middle_metacarpal", "middle_proximal", "middle_intermediate", "middle_distal", "middle_tip",
    "ring_metacarpal", "ring_proximal", "ring_intermediate", "ring_distal", "ring_tip",
    "pinky_metacarpal", "pinky_proximal", "pinky_intermediate", "pinky_distal", "pinky_tip",
)
# Wuji publishes MediaPipe's 21 landmarks: the same chain structure with the
# four non-thumb metacarpals dropped.  Verified against the recordings — bone
# lengths come out anatomically correct under this parenting (palm bones
# 8.2-9.7 cm decreasing index to pinky, phalanges 1.2-5.2 cm), while forcing
# the MANUS parenting onto them yields impossible 12 cm bones.
WUJI_HIERARCHY = np.asarray(
    (-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 0, 13, 14, 15, 0, 17, 18, 19),
    dtype=np.int64,
)
WUJI_NODE_NAMES = (
    "wrist",
    "thumb_cmc", "thumb_mcp", "thumb_ip", "thumb_tip",
    "index_mcp", "index_pip", "index_dip", "index_tip",
    "middle_mcp", "middle_pip", "middle_dip", "middle_tip",
    "ring_mcp", "ring_pip", "ring_dip", "ring_tip",
    "pinky_mcp", "pinky_pip", "pinky_dip", "pinky_tip",
)
GLOVE_LAYOUTS = {
    "manus25": {"hierarchy": MANUS_HIERARCHY, "node_names": MANUS_NODE_NAMES,
                "device_type": "manus_glove"},
    "wuji21": {"hierarchy": WUJI_HIERARCHY, "node_names": WUJI_NODE_NAMES,
               "device_type": "wuji_glove"},
}
LAYOUT_BY_NODE_COUNT = {count: name for name, count in GLOVE_NODE_COUNTS.items()}


MANO_TARGET_NAMES = tuple(
    f"{finger}_{joint}"
    for finger in MANO_FINGER_ORDER
    for joint in ("1", "2", "3", "tip")
)


@dataclass(frozen=True)
class ManusHandStream:
    side: str
    device_name: str
    timestamps_us: np.ndarray
    positions: np.ndarray
    quaternions_xyzw: np.ndarray
    layout: str = "manus25"


@dataclass(frozen=True)
class ManusHandFitResult:
    source: Path
    timestamps: np.ndarray
    source_indices: np.ndarray
    betas: np.ndarray
    body_scale: float
    left_hand_pose: np.ndarray
    right_hand_pose: np.ndarray
    source_points_local: np.ndarray
    source_points_aligned: np.ndarray
    model_points: np.ndarray
    alignment_rotation: np.ndarray
    alignment_scale: np.ndarray
    joint_rms_m: np.ndarray
    summary: dict[str, Any]


def _quaternion_xyzw_to_matrix(values: np.ndarray) -> np.ndarray:
    q = np.asarray(values, dtype=np.float64)
    if q.shape[-1] != 4:
        raise ValueError(f"expected xyzw quaternions, got {q.shape}")
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    if np.any(norm < 1e-8):
        raise ValueError("MANUS skeleton contains a zero quaternion")
    x, y, z, w = np.moveaxis(q / norm, -1, 0)
    return np.stack((
        1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
        2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
        2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
    ), axis=-1).reshape(q.shape[:-1] + (3, 3))


def read_manus_npz(path: str | Path) -> dict[str, ManusHandStream]:
    """Read both native MANUS raw-skeleton streams and resolve their sides."""
    source = Path(path)
    values = np.load(source, allow_pickle=False)
    if "meta" not in values:
        raise ValueError(f"MANUS NPZ has no metadata: {source}")
    metadata = json.loads(str(values["meta"].item()))
    device_type = metadata.get("device_type")
    supported = {spec["device_type"]: name for name, spec in GLOVE_LAYOUTS.items()}
    if device_type not in supported:
        raise ValueError(f"unsupported glove device_type: {device_type!r}; expected one of {sorted(supported)}")
    result: dict[str, ManusHandStream] = {}
    for glove in metadata.get("gloves", []):
        side = glove.get("slot")
        device_name = glove.get("device_name")
        if side not in {"left", "right"} or not device_name:
            continue
        key = f"{device_name}__hand_skeleton"
        time_key = f"{key}__t_host_us"
        if key not in values or time_key not in values:
            raise ValueError(f"missing MANUS stream {key!r}")
        skeleton = np.asarray(values[key], dtype=np.float32)
        if skeleton.ndim != 3 or skeleton.shape[2] != 7:
            raise ValueError(f"expected {key} shape [T,N,7], got {skeleton.shape}")
        layout = LAYOUT_BY_NODE_COUNT.get(int(skeleton.shape[1]))
        if layout is None:
            raise ValueError(f"{key}: unsupported node count {skeleton.shape[1]}")
        # The 2026-09 MANUS recorder exports the MediaPipe-style 21-node skeleton
        # (metacarpals folded into the wrist) under device_type=manus_glove, so the
        # node count, not the device type, decides the layout.  Each stream still
        # carries a global wrist orientation like the 25-node export.
        if layout != supported[device_type] and not (device_type == "manus_glove" and layout == "wuji21"):
            raise ValueError(f"{key}: {skeleton.shape[1]} nodes contradicts device_type={device_type!r}")
        # Only MANUS records a hierarchy; Wuji's layout is fixed by the SDK.
        hierarchy = np.asarray(glove.get("hand_hierarchy", []), dtype=np.int64)
        expected = GLOVE_LAYOUTS[layout]["hierarchy"]
        if hierarchy.size and not np.array_equal(hierarchy, expected):
            raise ValueError(f"{key}: hierarchy differs from the supported {layout} layout")
        result[side] = ManusHandStream(
            side=side,
            device_name=device_name,
            timestamps_us=np.asarray(values[time_key], dtype=np.int64),
            positions=skeleton[..., :3],
            quaternions_xyzw=skeleton[..., 3:7],
            layout=layout,
        )
    if set(result) != {"left", "right"}:
        raise ValueError(f"expected left and right MANUS streams, got {sorted(result)}")
    return result


def wrist_local_points(stream: ManusHandStream) -> np.ndarray:
    """Remove the MANUS wrist transform from global raw-skeleton node points."""
    rotation = _quaternion_xyzw_to_matrix(stream.quaternions_xyzw[:, 0])
    centered = stream.positions - stream.positions[:, :1]
    return np.einsum("tji,tkj->tki", rotation, centered).astype(np.float32)


def _capture_origin_us(path: Path, streams: dict[str, ManusHandStream]) -> int:
    session_path = path.with_suffix(".session.json")
    if session_path.exists():
        metadata = json.loads(session_path.read_text(encoding="utf-8"))
        video_range = metadata.get("video", {}).get("t_host_us_range")
        if isinstance(video_range, list) and len(video_range) == 2:
            return int(video_range[0])
    return min(int(stream.timestamps_us[0]) for stream in streams.values())


def _nearest_indices(source: np.ndarray, targets: np.ndarray) -> np.ndarray:
    right = np.searchsorted(source, targets).clip(0, len(source) - 1)
    left = np.maximum(0, right - 1)
    choose_left = np.abs(source[left] - targets) <= np.abs(source[right] - targets)
    return np.where(choose_left, left, right).astype(np.int64)


def _shared_selection(streams: dict[str, ManusHandStream], maximum: int) -> tuple[np.ndarray, np.ndarray]:
    start = max(int(stream.timestamps_us[0]) for stream in streams.values())
    end = min(int(stream.timestamps_us[-1]) for stream in streams.values())
    if end <= start:
        raise ValueError("left and right MANUS streams do not overlap")
    available = min(np.count_nonzero((stream.timestamps_us >= start) & (stream.timestamps_us <= end)) for stream in streams.values())
    count = available if maximum <= 0 else min(available, maximum)
    targets = np.linspace(start, end, count, dtype=np.float64)
    indices = np.stack([_nearest_indices(streams[side].timestamps_us, targets) for side in ("left", "right")], axis=1)
    actual = np.stack([streams[side].timestamps_us[indices[:, index]] for index, side in enumerate(("left", "right"))], axis=1)
    return indices, np.rint(actual.mean(axis=1)).astype(np.int64)


def _straightness(points: np.ndarray, layout: str = "manus25") -> np.ndarray:
    # Wrist plus each finger's four MANO points; the node ids differ per glove
    # layout, so derive the chains instead of hard-coding MANUS indices.
    chains = tuple((0, *GLOVE_POINT_IDS[layout][finger]) for finger in MANO_FINGER_ORDER)
    scores = []
    for chain in chains:
        length = sum(np.linalg.norm(points[:, second] - points[:, first], axis=1) for first, second in zip(chain[:-1], chain[1:]))
        scores.append(np.linalg.norm(points[:, chain[-1]] - points[:, 0], axis=1) / np.maximum(length, 1e-8))
    return np.mean(scores, axis=0)


def _similarity_alignment(source: np.ndarray, target: np.ndarray) -> tuple[float, np.ndarray]:
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    scale = float(np.sqrt(np.sum(target * target) / np.maximum(np.sum(source * source), 1e-12)))
    u, _, vt = np.linalg.svd((scale * source).T @ target)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T
    return scale, rotation.astype(np.float32)


def _body_output(body, betas, body_scale: float, left_pose, right_pose):
    import torch

    count = len(left_pose)
    device = left_pose.device
    zero3 = torch.zeros((count, 3), dtype=torch.float32, device=device)
    return body(
        betas=betas[None].expand(count, -1), global_orient=zero3, transl=zero3,
        body_pose=torch.zeros((count, 63), dtype=torch.float32, device=device),
        left_hand_pose=left_pose, right_hand_pose=right_pose,
        jaw_pose=zero3, leye_pose=zero3, reye_pose=zero3,
        expression=torch.zeros((count, 10), dtype=torch.float32, device=device),
        return_verts=False,
    ).joints * body_scale


def fit_manus_npz(
    input_path: str | Path,
    subject_profile: str | Path,
    *,
    body_model_path: str | Path = "models/smplx",
    device: str = "cpu",
    max_frames: int = 360,
    window_size: int = 90,
    iterations: int = 120,
    verbose: bool = False,
) -> ManusHandFitResult:
    """Fit sampled native MANUS frames to both 45-D SMPL-X hand poses."""
    import torch

    source = Path(input_path)
    streams = read_manus_npz(source)
    all_local = {side: wrist_local_points(stream) for side, stream in streams.items()}
    indices, timestamps_us = _shared_selection(streams, max_frames)
    origin_us = _capture_origin_us(source, streams)
    timestamps = ((timestamps_us - origin_us) / 1e6).astype(np.float64)
    selected_local = np.stack([all_local[side][indices[:, index]] for index, side in enumerate(("left", "right"))], axis=1)

    profile = json.loads(Path(subject_profile).read_text(encoding="utf-8"))
    betas_np = np.asarray(profile["fit"]["betas"], dtype=np.float32)
    body_scale = float(profile["fit"]["scale"])
    selected_device = select_device(device).selected
    body = load_body_model(ModelPaths(Path(body_model_path), None, "smplx", "neutral"), selected_device).eval()
    for parameter in body.parameters():
        parameter.requires_grad_(False)
    betas = torch.as_tensor(betas_np, dtype=torch.float32, device=selected_device)
    flat = {
        "left": -body.left_hand_mean.detach().to(selected_device),
        "right": -body.right_hand_mean.detach().to(selected_device),
    }
    with torch.no_grad():
        reference = _body_output(body, betas, body_scale, flat["left"][None], flat["right"][None])[0].cpu().numpy()

    # Wuji hands arrive already wrist-local (their root quaternion is identity),
    # which wrist_local_points handles without a special case; only the point
    # indices differ between the two layouts.
    layout = LAYOUT_BY_NODE_COUNT.get(int(all_local["left"].shape[1]), "manus25")
    raw_ids, _ = glove_target_indices("left", layout)
    alignment_scale, alignment_rotation = [], []
    aligned = np.empty_like(selected_local)
    targets: dict[str, torch.Tensor] = {}
    smpl_ids: dict[str, torch.Tensor] = {}
    for side_index, side in enumerate(("left", "right")):
        _, output_ids = manus_target_indices(side)
        best = np.argsort(_straightness(all_local[side], layout))[-min(12, len(all_local[side])):]
        source_reference = np.median(all_local[side][best][:, raw_ids], axis=0)
        target_reference = reference[output_ids] - reference[SMPLX_WRIST_JOINT[side]]
        source_scale, source_rotation = _similarity_alignment(source_reference, target_reference)
        alignment_scale.append(source_scale)
        alignment_rotation.append(source_rotation)
        aligned[:, side_index] = source_scale * (selected_local[:, side_index] @ source_rotation.T)
        targets[side] = torch.as_tensor(aligned[:, side_index, raw_ids], dtype=torch.float32, device=selected_device)
        smpl_ids[side] = torch.as_tensor(output_ids, dtype=torch.long, device=selected_device)

    poses: dict[str, list[np.ndarray]] = {"left": [], "right": []}
    previous: dict[str, torch.Tensor | None] = {"left": None, "right": None}
    count_total = len(timestamps)
    for start in range(0, count_total, max(1, window_size)):
        end = min(count_total, start + max(1, window_size))
        count = end - start
        parameters = {
            side: torch.nn.Parameter(flat[side][None].expand(count, -1).clone())
            for side in ("left", "right")
        }
        optimizer = torch.optim.Adam(tuple(parameters.values()), lr=0.04)
        for _ in range(iterations):
            optimizer.zero_grad(set_to_none=True)
            joints = _body_output(body, betas, body_scale, parameters["left"], parameters["right"])
            loss = torch.zeros((), dtype=torch.float32, device=selected_device)
            for side in ("left", "right"):
                wrist = SMPLX_WRIST_JOINT[side]
                predicted = joints[:, smpl_ids[side]] - joints[:, wrist, None]
                target = targets[side][start:end]
                position_loss = (predicted - target).square().mean()
                predicted_chain = predicted.reshape(count, 5, 4, 3)
                target_chain = target.reshape(count, 5, 4, 3)
                predicted_edges = torch.cat((predicted_chain[:, :, :1], predicted_chain[:, :, 1:] - predicted_chain[:, :, :-1]), dim=2)
                target_edges = torch.cat((target_chain[:, :, :1], target_chain[:, :, 1:] - target_chain[:, :, :-1]), dim=2)
                direction_loss = (1 - torch.nn.functional.cosine_similarity(predicted_edges, target_edges, dim=-1)).mean()
                prior_loss = (parameters[side] - flat[side]).square().mean()
                temporal_loss = torch.zeros((), dtype=torch.float32, device=selected_device)
                if count > 1:
                    temporal_loss = (parameters[side][1:] - parameters[side][:-1]).square().mean()
                if previous[side] is not None:
                    temporal_loss = temporal_loss + 2.0 * (parameters[side][0] - previous[side]).square().mean()
                joint_norm = torch.linalg.vector_norm(parameters[side].reshape(count, 15, 3), dim=-1)
                limit_loss = torch.relu(joint_norm - 2.7).square().mean()
                loss = loss + 1000.0 * position_loss + 0.015 * direction_loss + 0.003 * prior_loss + 0.004 * temporal_loss + 0.02 * limit_loss
            loss.backward()
            optimizer.step()
        for side in ("left", "right"):
            value = parameters[side].detach()
            poses[side].append(value.cpu().numpy().astype(np.float32))
            previous[side] = value[-1].clone()
        if verbose:
            print(f"MANUS hand window {start}:{end} loss={float(loss.detach().cpu()):.6f}")

    left_pose = np.concatenate(poses["left"])
    right_pose = np.concatenate(poses["right"])
    model_points = np.empty((count_total, 2, 21, 3), dtype=np.float32)
    for start in range(0, count_total, 120):
        end = min(count_total, start + 120)
        with torch.no_grad():
            joints = _body_output(
                body, betas, body_scale,
                torch.as_tensor(left_pose[start:end], device=selected_device),
                torch.as_tensor(right_pose[start:end], device=selected_device),
            ).cpu().numpy()
        for side_index, side in enumerate(("left", "right")):
            _, output_ids = manus_target_indices(side)
            wrist = joints[:, SMPLX_WRIST_JOINT[side], None]
            model_points[start:end, side_index, 0] = 0
            model_points[start:end, side_index, 1:] = joints[:, output_ids] - wrist
    difference = model_points[:, :, 1:] - aligned[:, :, raw_ids]
    joint_rms = np.sqrt(np.mean(np.sum(difference * difference, axis=-1), axis=-1)).astype(np.float32)
    anchor_drift = {}
    for side in ("left", "right"):
        anchors = all_local[side][:, (1, 5, 10, 15, 20)]
        anchor_drift[side] = float(np.sqrt(np.mean((anchors - np.median(anchors, axis=0)[None]) ** 2)))
    summary = {
        "source": str(source.resolve()),
        "source_frames": {side: len(streams[side].timestamps_us) for side in ("left", "right")},
        "fitted_frames": count_total,
        "duration_s": float(timestamps[-1] - timestamps[0]),
        "joint_rms_m": {side: float(np.sqrt(np.mean(joint_rms[:, index] ** 2))) for index, side in enumerate(("left", "right"))},
        "joint_p95_m": {side: float(np.percentile(joint_rms[:, index], 95)) for index, side in enumerate(("left", "right"))},
        "manus_to_smpl_scale": {side: float(alignment_scale[index]) for index, side in enumerate(("left", "right"))},
        "palm_anchor_drift_m": anchor_drift,
        "quaternion_order": "xyzw (recorder output)",
        "mano_pose_order": list(MANO_FINGER_ORDER),
    }
    return ManusHandFitResult(
        source=source, timestamps=timestamps, source_indices=indices, betas=betas_np, body_scale=body_scale,
        left_hand_pose=left_pose, right_hand_pose=right_pose,
        source_points_local=selected_local, source_points_aligned=aligned, model_points=model_points,
        alignment_rotation=np.stack(alignment_rotation), alignment_scale=np.asarray(alignment_scale, dtype=np.float32),
        joint_rms_m=joint_rms, summary=summary,
    )


def write_manus_fit(path: str | Path, result: ManusHandFitResult) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        timestamps=result.timestamps, source_indices=result.source_indices,
        betas=result.betas, body_scale=np.asarray(result.body_scale, dtype=np.float32),
        left_hand_pose=result.left_hand_pose, right_hand_pose=result.right_hand_pose,
        source_points_local=result.source_points_local, source_points_aligned=result.source_points_aligned,
        model_points=result.model_points, alignment_rotation=result.alignment_rotation,
        alignment_scale=result.alignment_scale, joint_rms_m=result.joint_rms_m,
    )
    destination.with_suffix(".json").write_text(json.dumps(result.summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_manus_fit(path: str | Path) -> ManusHandFitResult:
    """Load a cached native MANUS fit for deterministic re-fusion/viewing."""
    source = Path(path)
    summary = json.loads(source.with_suffix(".json").read_text(encoding="utf-8"))
    with np.load(source, allow_pickle=False) as values:
        return ManusHandFitResult(
            source=Path(summary["source"]), timestamps=values["timestamps"].copy(),
            source_indices=values["source_indices"].copy(), betas=values["betas"].copy(),
            body_scale=float(values["body_scale"]), left_hand_pose=values["left_hand_pose"].copy(),
            right_hand_pose=values["right_hand_pose"].copy(),
            source_points_local=values["source_points_local"].copy(),
            source_points_aligned=values["source_points_aligned"].copy(),
            model_points=values["model_points"].copy(),
            alignment_rotation=values["alignment_rotation"].copy(),
            alignment_scale=values["alignment_scale"].copy(),
            joint_rms_m=values["joint_rms_m"].copy(), summary=summary,
        )


def _hand_topology(body, side: str) -> tuple[np.ndarray, np.ndarray]:
    joint_ids = [SMPLX_WRIST_JOINT[side], *[joint for finger in MANO_FINGER_ORDER for joint in SMPLX_HAND_JOINTS[side][finger][:3]]]
    weights = body.lbs_weights.detach().cpu().numpy()[:, joint_ids].sum(axis=1)
    mask = weights > 0.4
    faces = np.asarray(body.faces, dtype=np.int64)
    faces = faces[np.all(mask[faces], axis=1)]
    vertices = np.unique(faces)
    remap = np.full(len(mask), -1, dtype=np.int64)
    remap[vertices] = np.arange(len(vertices))
    return vertices, remap[faces].astype(np.int32)


def write_manus_html(
    path: str | Path,
    result: ManusHandFitResult,
    *,
    body_model_path: str | Path = "models/smplx",
    device: str = "cpu",
) -> None:
    # The fit result carries the raw points it was built from, so the glove
    # layout is recoverable without threading it through every caller.
    viewer_layout = LAYOUT_BY_NODE_COUNT.get(
        int(np.asarray(result.source_points_local).shape[-2]), "manus25")
    """Write an interactive two-hand MANUS points vs SMPL-X/MANO mesh viewer."""
    import torch

    selected_device = select_device(device).selected
    body = load_body_model(ModelPaths(Path(body_model_path), None, "smplx", "neutral"), selected_device).eval()
    for parameter in body.parameters():
        parameter.requires_grad_(False)
    betas = torch.as_tensor(result.betas, dtype=torch.float32, device=selected_device)
    topology = {side: _hand_topology(body, side) for side in ("left", "right")}
    offsets = np.asarray(((0.045, 0.0, 0.0), (-0.045, 0.0, 0.0)), dtype=np.float32)
    mesh_frames = []
    for start in range(0, len(result.timestamps), 90):
        end = min(len(result.timestamps), start + 90)
        count = end - start
        zero3 = torch.zeros((count, 3), dtype=torch.float32, device=selected_device)
        with torch.no_grad():
            output = body(
                betas=betas[None].expand(count, -1), global_orient=zero3, transl=zero3,
                body_pose=torch.zeros((count, 63), device=selected_device),
                left_hand_pose=torch.as_tensor(result.left_hand_pose[start:end], device=selected_device),
                right_hand_pose=torch.as_tensor(result.right_hand_pose[start:end], device=selected_device),
                jaw_pose=zero3, leye_pose=zero3, reye_pose=zero3,
                expression=torch.zeros((count, 10), device=selected_device), return_verts=True,
            )
        vertices = output.vertices.cpu().numpy() * result.body_scale
        joints = output.joints.cpu().numpy() * result.body_scale
        combined = []
        for side_index, side in enumerate(("left", "right")):
            vertex_ids, _ = topology[side]
            wrist = joints[:, SMPLX_WRIST_JOINT[side], None]
            combined.append(vertices[:, vertex_ids] - wrist + offsets[side_index, None])
        mesh_frames.append(np.concatenate(combined, axis=1).astype(np.float32))
    mesh = np.concatenate(mesh_frames)
    raw = result.source_points_aligned + offsets[None, :, None]
    model = result.model_points + offsets[None, :, None]
    # Use the same proper left/back/up display rotation as every body viewer;
    # a bare Y/Z swap would reflect the two hands.
    mesh = mesh @ DISPLAY_ROTATION.T
    raw = raw @ DISPLAY_ROTATION.T
    model = model @ DISPLAY_ROTATION.T
    lower = np.min(np.concatenate((mesh.reshape(-1, 3), raw.reshape(-1, 3), model.reshape(-1, 3))), axis=0)
    upper = np.max(np.concatenate((mesh.reshape(-1, 3), raw.reshape(-1, 3), model.reshape(-1, 3))), axis=0)
    margin = np.maximum(0.025, 0.08 * (upper - lower))
    lower -= margin
    upper += margin
    center = 0.5 * (mesh.min(axis=(0, 1)) + mesh.max(axis=(0, 1)))
    quantization = max(1e-6, float(np.max(np.abs(mesh - center))) / 32000.0)
    packed = np.rint((mesh - center) / quantization).astype("<i2")
    packed_base64 = base64.b64encode(packed.tobytes()).decode("ascii")
    vertex_counts = [len(topology[side][0]) for side in ("left", "right")]
    payload = json.dumps({
        "timestamps": result.timestamps.round(6).tolist(),
        "source_indices": result.source_indices.tolist(),
        "mesh": packed_base64, "mesh_origin": center.round(7).tolist(), "mesh_scale": quantization,
        "vertex_counts": vertex_counts,
        "faces": [topology[side][1].tolist() for side in ("left", "right")],
        "raw": raw.round(5).tolist(), "model": model.round(5).tolist(),
        "errors": result.joint_rms_m.round(6).tolist(),
        "range": np.stack((lower, upper), axis=1).round(5).tolist(),
        "hierarchy": GLOVE_LAYOUTS[viewer_layout]["hierarchy"].tolist(),
        "raw_target_ids": glove_target_indices("left", viewer_layout)[0].tolist(),
        "node_names": GLOVE_LAYOUTS[viewer_layout]["node_names"],
        "mano_names": ("wrist", *MANO_TARGET_NAMES),
        "glove_layout": viewer_layout,
        "glove_label": {"manus25": "MANUS 25点", "wuji21": "Wuji 21点"}[viewer_layout],
        "alignment_scale": result.alignment_scale.round(6).tolist(),
        "summary": result.summary,
    }, ensure_ascii=False, separators=(",", ":"))
    document = r'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MANUS → SMPL-X / MANO</title><script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>body{margin:0;background:#07101f;color:#e5e7eb;font:14px -apple-system,BlinkMacSystemFont,"PingFang SC",sans-serif}#bar{display:flex;gap:10px;align-items:center;flex-wrap:wrap;padding:9px 14px;background:#111827;border-bottom:1px solid #334155}button,select{color:#e5e7eb;background:#243146;border:1px solid #475569;border-radius:5px;padding:6px 10px}#frame{width:min(560px,42vw)}#status{padding:6px 14px;color:#cbd5e1;background:#0f172a}#plot{height:calc(100vh - 98px)}label{white-space:nowrap}.good{color:#86efac}.info{color:#93c5fd}</style></head><body>
<div id="bar"><button id="play">播放</button><input id="frame" type="range" min="0" step="1"><span id="time"></span><select id="side"><option value="both">双手</option><option value="left">左手</option><option value="right">右手</option></select><label><input id="mesh" type="checkbox" checked> Mesh</label><label><input id="raw" type="checkbox" checked> <span id="rawLabel">原始点</span></label><label><input id="corr" type="checkbox" checked> 对应残差</label></div><div id="status"></div><div id="plot"></div>
<script>const D=__PAYLOAD__;document.getElementById('rawLabel').textContent=D.glove_label;
const slider=document.getElementById('frame'),play=document.getElementById('play'),time=document.getElementById('time'),sideSel=document.getElementById('side'),meshToggle=document.getElementById('mesh'),rawToggle=document.getElementById('raw'),corrToggle=document.getElementById('corr'),status=document.getElementById('status');slider.max=D.timestamps.length-1;let timer=null,decoded=null;const sideNames=['左手','右手'],rawColors=['#22d3ee','#f59e0b'],modelColors=['#60a5fa','#fb7185'];
function unpack(){if(decoded)return decoded;const s=atob(D.mesh),b=new Uint8Array(s.length);for(let i=0;i<s.length;i++)b[i]=s.charCodeAt(i);return decoded=new Int16Array(b.buffer)}
function vertices(frame,side){const q=unpack(),total=D.vertex_counts[0]+D.vertex_counts[1],start=side?D.vertex_counts[0]:0,count=D.vertex_counts[side],base=frame*total*3+start*3,x=[],y=[],z=[];for(let i=0;i<count;i++){x.push(D.mesh_origin[0]+D.mesh_scale*q[base+3*i]);y.push(D.mesh_origin[1]+D.mesh_scale*q[base+3*i+1]);z.push(D.mesh_origin[2]+D.mesh_scale*q[base+3*i+2])}return{x,y,z}}
function line(points,edges){const x=[],y=[],z=[];for(const [a,b] of edges){x.push(points[a][0],points[b][0],null);y.push(points[a][1],points[b][1],null);z.push(points[a][2],points[b][2],null)}return{x,y,z}}
const rawEdges=[];for(let i=1;i<D.hierarchy.length;i++)rawEdges.push([D.hierarchy[i],i]);const modelEdges=[];for(let f=0;f<5;f++){const a=1+4*f;modelEdges.push([0,a],[a,a+1],[a+1,a+2],[a+2,a+3])}
function active(){return sideSel.value==='both'?[0,1]:[sideSel.value==='left'?0:1]}
function action(t){const a=[[0,7,'01 站立'],[7,16,'02 头部'],[16,23,'03 A字'],[23,30,'04 T字'],[30,37,'05 前平举'],[37,45,'06 屈肘'],[45,53,'07 手腕'],[53,61,'08 浅蹲'],[61,68,'09 左抬腿'],[68,75,'10 右抬腿'],[75,82,'11 踝关节'],[82,89,'12 张开/并拢'],[89,96,'13 握拳'],[96,105,'14 对指'],[105,114,'15 单指屈伸']];const s=a.find(x=>t>=x[0]&&t<x[1]);return s?s[2]:'结束姿态'}
function traces(frame){const out=[];for(const s of active()){if(meshToggle.checked){const v=vertices(frame,s),F=D.faces[s];out.push({type:'mesh3d',x:v.x,y:v.y,z:v.z,i:F.map(x=>x[0]),j:F.map(x=>x[1]),k:F.map(x=>x[2]),color:s?'#d6b0a5':'#b8c9d8',opacity:.72,flatshading:false,lighting:{ambient:.62,diffuse:.7,specular:.15,roughness:.8},name:`${sideNames[s]} SMPL-X mesh`})}const rp=D.raw[frame][s],mp=D.model[frame][s],ml=line(mp,modelEdges);out.push({type:'scatter3d',mode:'lines+markers',x:ml.x,y:ml.y,z:ml.z,line:{color:modelColors[s],width:7},marker:{size:3,color:modelColors[s]},name:`${sideNames[s]} MANO骨架`,hoverinfo:'skip'});if(rawToggle.checked){const rl=line(rp,rawEdges);out.push({type:'scatter3d',mode:'lines',x:rl.x,y:rl.y,z:rl.z,line:{color:rawColors[s],width:4},name:`${sideNames[s]} ${D.glove_label}链`},{type:'scatter3d',mode:'markers',x:rp.map(p=>p[0]),y:rp.map(p=>p[1]),z:rp.map(p=>p[2]),text:D.node_names,hovertemplate:'${D.glove_label} %{text}<extra></extra>',marker:{size:4,color:rawColors[s]},name:`${sideNames[s]} ${D.glove_label}`})}if(corrToggle.checked){const x=[],y=[],z=[];for(let k=0;k<20;k++){const a=rp[D.raw_target_ids[k]],b=mp[k+1];x.push(a[0],b[0],null);y.push(a[1],b[1],null);z.push(a[2],b[2],null)}out.push({type:'scatter3d',mode:'lines',x,y,z,line:{color:'#86efac',width:3,dash:'dot'},name:`${sideNames[s]} 对应残差`})}}return out}
function draw(){const f=+slider.value,t=D.timestamps[f],view=sideSel.value,r=D.range;time.textContent=`frame ${f}/${D.timestamps.length-1} · t=${t.toFixed(2)} s`;status.innerHTML=`<span class="info">${action(t)}</span>　左手 RMS <span class="good">${(D.errors[f][0]*1000).toFixed(1)} mm</span>　右手 RMS <span class="good">${(D.errors[f][1]*1000).toFixed(1)} mm</span>　${D.glove_label}→SMPL 尺度 ${D.alignment_scale[0].toFixed(3)} / ${D.alignment_scale[1].toFixed(3)}　源帧 ${D.source_indices[f][0]} / ${D.source_indices[f][1]}`;const range=view==='both'?r:(view==='left'?[[0.0,r[0][1]],r[1],r[2]]:[[r[0][0],0.0],r[1],r[2]]);Plotly.react('plot',traces(f),{margin:{l:0,r:0,b:0,t:34},paper_bgcolor:'#07101f',font:{color:'#e5e7eb'},title:{text:''+D.glove_label+' → SMPL-X/MANO 21关键点（腕局部坐标）',font:{size:15}},legend:{x:.01,y:.99,bgcolor:'rgba(7,16,31,.55)'},uirevision:'manus-hands-'+view,scene:{xaxis:{title:'受试者左 +X / 右 -X',range:range[0],gridcolor:'#334155'},yaxis:{title:'身体后方 +Y / 前方 -Y',range:range[1],gridcolor:'#334155'},zaxis:{title:'向上 +Z',range:range[2],gridcolor:'#334155'},aspectmode:'data',camera:{eye:{x:0,y:-1.35,z:.65},up:{x:0,y:0,z:1}}}},{responsive:true,scrollZoom:true})}
slider.oninput=draw;sideSel.onchange=draw;meshToggle.onchange=draw;rawToggle.onchange=draw;corrToggle.onchange=draw;play.onclick=()=>{if(timer){clearInterval(timer);timer=null;play.textContent='播放';return}play.textContent='暂停';timer=setInterval(()=>{slider.value=(+slider.value+1)%D.timestamps.length;draw()},1000/12)};draw();</script></body></html>'''.replace("__PAYLOAD__", payload)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(document, encoding="utf-8")


def fit_and_visualize_manus(
    input_path: str | Path,
    subject_profile: str | Path,
    output_dir: str | Path,
    **kwargs,
) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    body_model_path = kwargs.get("body_model_path", "models/smplx")
    device = kwargs.get("device", "cpu")
    result = fit_manus_npz(input_path, subject_profile, **kwargs)
    npz_path = output / "manus_to_smplx_hands.npz"
    html_path = output / "manus_to_smplx_hands.html"
    write_manus_fit(npz_path, result)
    write_manus_html(html_path, result, body_model_path=body_model_path, device=device)
    report = {**result.summary, "npz": str(npz_path.resolve()), "html": str(html_path.resolve())}
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report
