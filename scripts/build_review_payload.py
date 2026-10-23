#!/usr/bin/env python
"""Build the data payload for the three-pane review viewer.

Everything comes from one unified 30 Hz npz, so the video frame, the mesh pose
and the tactile reading shown for a given slider position are the same instant
by construction rather than by the viewer guessing an alignment.

The mesh is the bulky part (10475 vertices per frame), so frames are uniformly
subsampled to a reviewable window and the vertex stream is delta-coded against
the previous frame and quantised to int16, the same scheme the full-mesh viewer
uses.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
from pathlib import Path

import numpy as np


KEYFRAME_EVERY = 30


def _encode_mesh(vertices: np.ndarray) -> tuple[bytes, float, int, int]:
    """Delta-code to int8 steps with periodic absolute int32 keyframes.

    The delta is taken against the *reconstructed* previous frame, not the
    original one.  Coding original-to-original looks equivalent but is not:
    each step carries its own rounding error, and those errors random-walk
    independently per vertex over the run between keyframes, which shreds the
    surface.  Predicting from the reconstruction bounds the error at half a
    quantiser step for every frame no matter how long the run.

    Keyframes are written as one contiguous block ahead of the deltas: mixing
    them would leave the int32 reads at offsets that are not 4-byte aligned,
    which typed-array views reject.
    """
    count = len(vertices)
    steps = np.diff(vertices, axis=0)
    span = float(np.abs(steps).max()) if len(steps) else 1.0
    scale = max(span, 1e-4) / 120.0
    keys: list[bytes] = []
    deltas: list[bytes] = []
    recon = np.zeros_like(vertices[0], dtype=np.float64)
    for i in range(count):
        if i % KEYFRAME_EVERY == 0:
            quant = np.round(vertices[i] / scale).astype("<i4")
            keys.append(quant.tobytes())
            recon = quant.astype(np.float64) * scale
        else:
            step = np.clip(np.round((vertices[i] - recon) / scale), -127, 127)
            deltas.append(step.astype("<i1").tobytes())
            recon = recon + step * scale
    return b"".join(keys) + b"".join(deltas), scale, KEYFRAME_EVERY, len(keys)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unified", type=Path, required=True, help="unified_30hz/<session>.npz")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--body-model-path", type=Path, default=Path("models/smplx"))
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    import torch
    from five_tracker_smplx.model import ModelPaths, load_body_model

    data = dict(np.load(args.unified, allow_pickle=False))
    total = len(data["timestamps"])
    picks = (np.linspace(0, total - 1, min(args.max_frames, total))).astype(int)
    picks = np.unique(picks)
    times = data["timestamps"][picks]

    # A recording whose projection failed still has video and tactile worth
    # reviewing; it is exported without a body and gets the same page minus the
    # mesh, with the failure reason in its place.
    has_body = bool(data.get("has_body", np.array(True)))

    faces = None
    vertices = None
    quant = keyframe = keyframe_count = 0
    if not has_body:
        body = None
    else:
        body = load_body_model(ModelPaths(args.body_model_path, None, "smplx", "neutral"), args.device).eval()
        for parameter in body.parameters():
            parameter.requires_grad_(False)
        faces = np.asarray(body.faces, dtype="<u4")

    vertices = []
    scale = float(data["scale"]) if has_body else 1.0
    with torch.no_grad():
        for start in (range(0, len(picks), 64) if has_body else ()):
            index = picks[start:start + 64]
            count = len(index)
            zero3 = torch.zeros((count, 3), dtype=torch.float32)
            out = body(
                betas=torch.tensor(data["betas"])[None].expand(count, -1),
                global_orient=torch.tensor(data["global_orient"][index]),
                body_pose=torch.tensor(data["body_pose"][index]).reshape(count, -1),
                transl=zero3,
                left_hand_pose=torch.tensor(data["left_hand_pose"][index]),
                right_hand_pose=torch.tensor(data["right_hand_pose"][index]),
                jaw_pose=zero3, leye_pose=zero3, reye_pose=zero3,
                expression=torch.zeros((count, 10)), return_verts=True,
            )
            vertices.append(scale * out.vertices.cpu().numpy() + data["transl"][index][:, None])
    vertices = (np.concatenate(vertices).astype(np.float32) if has_body else None)

    # The capture frame is Y-up with the floor at ground_y_m; the viewer draws a
    # Z-up world with the floor at 0, so convert once here rather than per frame.
    ground = float(data.get("ground_y_m", 0.0))
    def to_display(points: np.ndarray) -> np.ndarray:
        out = np.empty_like(points)
        out[..., 0] = points[..., 0]
        out[..., 1] = -points[..., 2]
        out[..., 2] = points[..., 1] - ground
        return out
    if has_body:
        vertices = to_display(vertices)
    tracker_display = to_display(data["tracker_position"][picks])

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    name = args.unified.stem
    # One shared topology for every session; several workers write into the
    # same directory, so only the first one creates it rather than having them
    # all rewrite the file a reader may be part-way through.
    if has_body:
        faces_path = output / "smplx_faces.u32"
        if not faces_path.exists():
            faces_path.write_bytes(faces.tobytes(order="C"))
        payload_bytes, quant, keyframe, keyframe_count = _encode_mesh(vertices)
        with gzip.open(output / f"{name}.mesh.i8c.gz", "wb", compresslevel=9) as handle:
            handle.write(payload_bytes)

    # Video time is elapsed wall clock, which is the right seek target even for
    # the variable-frame-rate captures.
    video_frame_count = video_wall_span = 0
    video_npz = Path(str(data["source_tracker_npz"])).with_name(
        Path(str(data["source_tracker_npz"])).name.replace(".tracker.npz", ".video.npz"))
    if video_npz.exists():
        with np.load(video_npz, allow_pickle=False) as handle:
            stamps = np.asarray(handle["t_host_us"], dtype=np.float64) * 1e-6
        video_frame_count, video_wall_span = len(stamps), float(stamps[-1] - stamps[0])
    payload = {
        "session": name,
        "rate_hz": float(data["rate_hz"]),
        "frames": int(len(picks)),
        "source_frames": int(total),
        "has_body": has_body,
        "failure": ({"kind": str(data.get("failure_kind", "")),
                     "why": str(data.get("failure_why", ""))} if not has_body else None),
        "vertex_count": int(vertices.shape[1]) if has_body else 0,
        "face_count": int(len(faces)) if has_body else 0,
        "quant_scale": quant,
        "keyframe_every": keyframe,
        "keyframe_count": keyframe_count,
        "mesh_encoding": "int8 deltas with int32 absolute keyframes every N frames",
        "glove_layout": str(data.get("glove_layout", "?")),
        "hand_pose_source": str(data.get("hand_pose_source", "?")),
        "tracker_native_hz": float(data.get("tracker_native_hz", 0)),
        "upsample_ratio": float(data.get("upsample_ratio", 0)),
        "t0": float(times[0]),
        "time": [round(float(t - times[0]), 4) for t in times],
        "video": None,
        "tracker": np.round(tracker_display, 4).tolist(),
        "coordinate_frame": "display_z_up_floor_zero",
        "ground_y_m": ground,
        "tracker_roles": [str(x) for x in data["tracker_roles"]],
        "range": ([[float(vertices[..., a].min()), float(vertices[..., a].max())]
                   for a in range(3)] if has_body else [[0, 1], [0, 1], [0, 1]]),
    }
    if "video_frame_index" in data:
        # Seek by frame index, not wall-clock seconds.  The MANUS-era recorder
        # writes its mp4 at a constant nominal 60 fps while actually capturing
        # ~52 Hz, so the file's internal clock runs 15% fast and seeking by
        # elapsed wall time drifts several seconds by the end.  Frame index
        # scaled onto the file's own duration is correct for both that case and
        # the variable-frame-rate Wuji captures.
        payload["video"] = {
            "frame_index": data["video_frame_index"][picks].astype(int).tolist(),
            "frame_count": int(video_frame_count),
            "wall_span_s": round(float(video_wall_span), 4),
            "offset_ms": np.round(data["video_time_offset_s"][picks] * 1000, 2).tolist(),
            "native_hz": float(data.get("video_native_hz", 0)),
        }
    tactile = {}
    for side in ("L", "R"):
        if f"tactile_{side}_L4_contact" not in data:
            continue
        tactile[side] = {
            # L4 is the reviewable product: per-region binary contact.
            "contact": data[f"tactile_{side}_L4_contact"][picks].astype(np.uint8).tolist(),
            "score": np.round(data[f"tactile_{side}_L2_score"][picks], 3).tolist(),
            "coarse": data[f"tactile_{side}_coarse_contact"][picks].astype(np.uint8).tolist(),
            "coverage": np.round(data[f"tactile_{side}_coverage"], 3).tolist(),
        }
    if tactile:
        payload["tactile"] = tactile
        payload["region_names"] = [str(x) for x in data["tactile_region_names"]]
        payload["coarse_names"] = [str(x) for x in data["tactile_coarse_names"]]
        payload["tactile_levels"] = str(data.get("tactile_levels", ""))
        spec_path = Path(os.environ.get("EGOEXPLORER_TACTILE_DIR", Path(__file__).resolve().parents[1] / "tactile")) / "wuji_channel_map.json"
        if spec_path.exists():
            spec = json.loads(spec_path.read_text(encoding="utf-8"))
            acc: dict[str, list] = {}
            for entry in spec["channels"]:
                if entry["finger"] == "invalid":
                    continue
                acc.setdefault(entry["region"], []).append(entry["uv"])
            payload["region_uv"] = {
                name: [round(float(np.mean([p[0] for p in pts])), 4),
                       round(float(np.mean([p[1] for p in pts])), 4), len(pts)]
                for name, pts in acc.items()
            }
    (output / f"{name}.payload.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    mesh_note = (f"mesh {(output / f'{name}.mesh.i8c.gz').stat().st_size / 1e6:.1f} MB"
                 if has_body else f"无 mesh（{(payload['failure'] or {}).get('kind', '?')}）")
    print(f"{name}: {len(picks)}/{total} 帧, {mesh_note}, "
          f"触觉={'有' if tactile else '无'} -> {output}")


if __name__ == "__main__":
    main()
