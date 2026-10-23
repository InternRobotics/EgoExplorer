#!/usr/bin/env python
"""Build a browsable gallery page for a five-tracker -> SMPL-X batch output.

The gallery is one self-contained HTML file: every session becomes a card with
a projected pose strip, the tracker markers, the root path and the per-frame
tracker residual, so a bad reconstruction can be spotted without loading the
heavy full-mesh viewer.  Sessions that failed in the pipeline are listed too,
with the reason, so the page covers the whole batch and not only its survivors.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import re
import math
import os
import sys
from pathlib import Path

import numpy as np

# SMPL-X body kinematic tree (first 22 joints) drawn as the thumbnail skeleton.
BODY_JOINTS = 22
TRACKER_LABELS = ("头", "左腕", "右腕", "左踝", "右踝")
POSE_SAMPLES = 3
U_HALF = .75
V_LO = -.2
V_HI = 2.05
SPARK_POINTS = 72
PATH_POINTS = 48


def _dataset_of(source: str) -> str:
    parts = Path(source).parts
    for index, part in enumerate(parts):
        # Corrected recordings live in a parallel tree with the same layout.
        if part in ("ego_data_sync", "ego_data_lr_fixed") and index + 1 < len(parts):
            return parts[index + 1]
    return "?"


def _camera_basis(joints: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (right, forward) unit vectors in the ground plane.

    The subject's left maps to image right, matching the full-mesh viewer's
    "受试者左 = 画面右（无镜像）" convention.
    """
    head = min(len(joints), 20)
    lateral = np.mean(
        .5 * ((joints[:head, 1] - joints[:head, 2]) + (joints[:head, 16] - joints[:head, 17])),
        axis=0,
    )[[0, 2]]
    norm = float(np.linalg.norm(lateral))
    lateral = lateral / norm if norm > 1e-8 else np.asarray((1.0, 0.0))
    forward = np.asarray((-lateral[1], lateral[0]))
    return lateral, forward


def _quantize(values: np.ndarray, lo: float, hi: float, scale: int) -> list[int]:
    span = max(hi - lo, 1e-6)
    return np.clip((values - lo) / span * scale, 0, scale).round().astype(int).tolist()


def _pose_strip(joints: np.ndarray, trackers: np.ndarray, ground_y: float) -> dict:
    """Project sampled poses into a fixed metric window shared by every card.

    A fixed window (1.6 m wide, -0.4..2.1 m above the estimated floor) is what
    makes the gallery scannable: every figure is drawn at the same scale with
    the ground line always in the same place, so a body sunk through the floor
    or blown up by a bad scale is visible at a glance instead of being hidden
    by a per-session autoscale.
    """
    right, _ = _camera_basis(joints)
    xz = joints[:, :BODY_JOINTS][:, :, [0, 2]]
    u = xz @ right
    v = joints[:, :BODY_JOINTS, 1] - ground_y
    tu = trackers[:, :, [0, 2]] @ right
    tv = trackers[:, :, 1] - ground_y
    # Centre each pose on its own pelvis so a walking subject stays framed.
    centre = u[:, 0:1]
    u, tu = u - centre, tu - centre
    clipped = bool(
        np.abs(u).max() > U_HALF or v.min() < V_LO or v.max() > V_HI
    )
    return {
        "j": [
            _quantize(u[i], -U_HALF, U_HALF, 1000) + _quantize(v[i], V_LO, V_HI, 1000)
            for i in range(len(u))
        ],
        "t": [
            _quantize(tu[i], -U_HALF, U_HALF, 1000) + _quantize(tv[i], V_LO, V_HI, 1000)
            for i in range(len(tu))
        ],
        "g": int(round((0.0 - V_LO) / (V_HI - V_LO) * 1000)),
        "clip": clipped,
    }



MESH_THUMB_W = 120
MESH_THUMB_H = 180
# Same metric window as the skeleton strip so the two read as one picture.
# Light is defined in camera space, not world space: the subject faces a
# different direction in every recording, and a fixed world light would make
# half the thumbnails dark for no reason the reader can act on.
_LIGHT_CAMERA = np.asarray((.30, .45, -.84), dtype=np.float64)  # right, up, toward camera
_SKIN = np.asarray((211, 160, 139), dtype=np.float64)
_BACKDROP = (11, 13, 18)


def _vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    normals = np.zeros_like(vertices)
    a, b, c = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    face_normals = np.cross(b - a, c - a)
    for column in range(3):
        np.add.at(normals, faces[:, column], face_normals)
    lengths = np.linalg.norm(normals, axis=-1, keepdims=True)
    return normals / np.maximum(lengths, 1e-9)


def _render_mesh_png(vertices: np.ndarray, faces: np.ndarray, right: np.ndarray,
                     ground_y: float) -> str:
    """Rasterise one frame of the mesh into a small PNG data URI.

    Splatting the 10k vertices with a depth sort is enough at thumbnail size and
    needs no GL context on a headless machine; filling 21k triangles in Python would
    not finish in reasonable time across 800 sessions.
    """
    from PIL import Image

    forward = np.asarray((-right[1], right[0]))
    u = vertices[:, [0, 2]] @ right
    v = vertices[:, 1] - ground_y
    depth = vertices[:, [0, 2]] @ forward
    u = u - np.median(u)

    x = (u + U_HALF) / (2 * U_HALF) * MESH_THUMB_W
    y = MESH_THUMB_H - (v - V_LO) / (V_HI - V_LO) * MESH_THUMB_H
    normals = _vertex_normals(vertices, faces)
    camera_normals = np.stack((
        normals[:, [0, 2]] @ right,
        normals[:, 1],
        normals[:, [0, 2]] @ forward,
    ), axis=-1)
    direction = _LIGHT_CAMERA / np.linalg.norm(_LIGHT_CAMERA)
    shade = np.clip(camera_normals @ direction, 0, 1) * .72 + .28

    # One flat list of (pixel, depth, shade) over a small disc per vertex, so a
    # single depth-ordered assignment fills the silhouette without holes.
    offsets = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if dx * dx + dy * dy <= 2]
    px = np.concatenate([np.round(x).astype(np.int32) + dx for dx, _ in offsets])
    py = np.concatenate([np.round(y).astype(np.int32) + dy for _, dy in offsets])
    pd = np.tile(depth, len(offsets))
    ps = np.tile(shade, len(offsets))
    inside = (px >= 0) & (px < MESH_THUMB_W) & (py >= 0) & (py < MESH_THUMB_H)
    px, py, pd, ps = px[inside], py[inside], pd[inside], ps[inside]
    order = np.argsort(-pd)  # far first; later writes win
    canvas = np.zeros((MESH_THUMB_H, MESH_THUMB_W, 3), dtype=np.uint8)
    canvas[:, :] = _BACKDROP
    floor = int(round(MESH_THUMB_H - (0.0 - V_LO) / (V_HI - V_LO) * MESH_THUMB_H))
    if 0 <= floor < MESH_THUMB_H:
        canvas[floor, :] = (44, 51, 66)
    flat = (py[order] * MESH_THUMB_W + px[order])
    colours = (_SKIN[None, :] * ps[order][:, None]).astype(np.uint8)
    canvas.reshape(-1, 3)[flat] = colours
    buffer = io.BytesIO()
    Image.fromarray(canvas).save(buffer, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _resample(values: np.ndarray, count: int) -> np.ndarray:
    if len(values) <= count:
        return values
    edges = np.linspace(0, len(values), count + 1).astype(int)
    return np.asarray([values[a:max(b, a + 1)].max() for a, b in zip(edges[:-1], edges[1:])])


def build_session(session_dir: Path, body, device: str, faces: np.ndarray) -> dict:
    import torch

    # Sessions re-projected on their own carry only the projection's summary;
    # the batch pipeline's richer metadata exists just for whole-pipeline runs.
    meta_path = session_dir / "pipeline_metadata.json"
    if not meta_path.exists():
        meta_path = session_dir / "smplx_profile_driven.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    with np.load(session_dir / "smplx_profile_driven.npz", allow_pickle=False) as npz:
        timestamps = npz["timestamps"]
        betas = npz["betas"]
        scale = float(npz["scale"])
        global_orient = npz["global_orient"]
        body_pose = npz["body_pose"]
        transl = npz["transl"]
        left_hand = npz["left_hand_pose"]
        right_hand = npz["right_hand_pose"]
    with np.load(session_dir / "tracker_diagnostics.npz", allow_pickle=False) as npz:
        residual = np.linalg.norm(npz["position_residual_m"], axis=-1)
        rotation_error = npz["rotation_error_rad"]
        tracker_position = npz["tracker_position"]
        valid = npz["valid"]

    frames = len(timestamps)
    picks = np.clip(
        (np.linspace(.18, .82, POSE_SAMPLES) * frames).astype(int), 0, max(frames - 1, 0)
    )
    # Frame 0 rides along with the sampled poses so the mesh render costs one
    # extra row in the same forward pass.
    picks = np.concatenate(([0], picks))
    with torch.no_grad():
        count = len(picks)
        zero3 = torch.zeros((count, 3), dtype=torch.float32, device=device)
        output = body(
            betas=torch.tensor(betas, device=device)[None].expand(count, -1),
            global_orient=torch.tensor(global_orient[picks], device=device),
            transl=zero3,
            body_pose=torch.tensor(body_pose[picks], device=device).reshape(count, -1),
            left_hand_pose=torch.tensor(left_hand[picks], device=device),
            right_hand_pose=torch.tensor(right_hand[picks], device=device),
            jaw_pose=zero3, leye_pose=zero3, reye_pose=zero3,
            expression=zero3.new_zeros((count, 10)), return_verts=True,
        )
        joints = scale * output.joints.cpu().numpy() + transl[picks][:, None]
        first_vertices = scale * output.vertices[0].cpu().numpy() + transl[picks[0]]
    picks = picks[1:]
    joints = joints[1:]

    residual_mm = 1000.0 * residual.mean(axis=1)
    # Recover the stats the batch metadata would have carried.
    if "position_rms_m" not in meta:
        meta["position_rms_m"] = float(np.sqrt(np.mean(residual ** 2)))
    if "maximum_gap_s" not in meta and frames > 1:
        meta["maximum_gap_s"] = float(np.max(np.diff(timestamps)))
    # A recording whose trackers never move is a dead capture, not a still pose.
    travel = float((tracker_position.max(axis=0) - tracker_position.min(axis=0)).max())
    duration = float(timestamps[-1] - timestamps[0]) if frames > 1 else 0.0
    spark = _resample(residual_mm, SPARK_POINTS)
    spark_max = float(max(spark.max(), 1e-3))
    path = transl[:, [0, 2]]
    step = max(1, len(path) // PATH_POINTS)
    path = path[::step]
    lo = path.min(axis=0)
    span = float(max((path.max(axis=0) - lo).max(), 1e-3))
    return {
        "id": session_dir.name,
        "ds": _dataset_of(meta.get("source", "")),
        "lrfix": "ego_data_lr_fixed" in str(meta.get("source", "")),
        "st": "ok",
        "fr": frames,
        "dur": round(duration, 1),
        "rms": round(1000.0 * float(meta.get("position_rms_m", 0.0)), 2),
        "p50": round(float(np.percentile(residual_mm, 50)), 2),
        "p95": round(float(np.percentile(residual_mm, 95)), 2),
        "rot": round(math.degrees(float(np.mean(rotation_error))), 2),
        "gap": round(float(meta.get("maximum_gap_s", 0.0)), 3),
        "val": round(float(valid.mean()), 4),
        "mov": round(travel, 2),
        "pose": _pose_strip(joints, tracker_position[picks], float(meta.get("ground_y_m", 0.0))),
        "mesh": _render_mesh_png(first_vertices, faces, _camera_basis(joints)[0],
                                 float(meta.get("ground_y_m", 0.0))),
        "sp": _quantize(spark, 0.0, spark_max, 100),
        "spm": round(spark_max, 1),
        "pa": (
            _quantize(path[:, 0], float(lo[0]), float(lo[0]) + span, 100)
            + _quantize(path[:, 1], float(lo[1]), float(lo[1]) + span, 100)
        ),
        "pam": round(span, 2),
    }


TRACKER_ROLE_KEYS = ("camera", "left_wrist", "right_wrist", "left_ankle", "right_ankle")


def _missing_tracker_roles(tracker_path: Path) -> list:
    """Roles the recording never captured, which the proposal stage rejects on."""
    try:
        with np.load(tracker_path, allow_pickle=False) as npz:
            keys = set(npz.files)
    except Exception:
        return []
    if "tracker_0__pose__t_host_us" not in keys:
        return ["整条 tracker 流"]
    return [r for r in TRACKER_ROLE_KEYS if f"tracker_0__pose.{r}.position" not in keys]


def failure_reason(session: str, proposals_root: Path, frames_hint: dict) -> tuple[str, str]:
    """Classify why a session has no projection, from what it left behind."""
    batch = proposals_root.parent
    # Anything the pipeline has not actually attempted is queued, not broken.
    # Newly synced recordings leave the same empty directory as a proposal that
    # failed, so the attempt marker is what separates them.
    attempted = ((batch / "projected" / f".failed_{session}").exists()
                 or (batch / "projected" / session).is_dir())
    if not attempted:
        return "pending", "尚未处理（已同步，排队中）"

    proposal = proposals_root / session / "smplx_profile_driven.npz"
    contact = proposals_root / session / "contact_probability.npy"
    if not proposal.exists():
        return "proposal", "提案阶段失败（tracker 有效样本过少）"
    if contact.exists():
        with np.load(proposal, allow_pickle=False) as npz:
            trajectory_frames = int(npz["transl"].shape[0])
        contact_frames = int(np.load(contact, mmap_mode="r").shape[0])
        if contact_frames < trajectory_frames:
            # This used to crash the last chunk outright.  The projector now
            # resamples contact onto the trajectory clock, so a short stream is
            # only worth recording, not blaming for the missing result.
            return "diverged", (
                f"投影未产出（contact_probability {contact_frames} 帧 / 轨迹 "
                f"{trajectory_frames} 帧，已按时间戳重采样）"
            )
    return "diverged", "投影发散（non-finite SMPL-X 参数）"


def render(output: Path, payload: dict) -> None:
    """Inline the payload into the template so the page opens without a server."""
    template = (Path(__file__).resolve().parent / "gallery_template.html").read_text(encoding="utf-8")
    (output / "index.html").write_text(
        template.replace("/*__GALLERY_DATA__*/null", json.dumps(payload, ensure_ascii=False)),
        encoding="utf-8",
    )


SLUG_OVERRIDES: dict = {}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-root", type=Path, required=True, help="pipeline --output-root")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--body-model-path", type=Path, default=Path("models/smplx"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--viewer-dir", default="sessions", help="relative path holding per-session mesh viewers")
    parser.add_argument("--sync-root", type=Path, default=None,
                        help="input tree this batch was built from; defaults to the main one")
    parser.add_argument("--review-dir", default="../review",
                        help="relative path holding the three-pane review pages")
    parser.add_argument("--limit", type=int, default=0, help="debug: only process the first N sessions")
    parser.add_argument("--title", default="ego_data 批处理结果")
    parser.add_argument(
        "--tracker-drift", type=Path, default=None,
        help="check_tracker_drift.py output; defaults to <output-dir>/tracker_drift.json",
    )
    parser.add_argument(
        "--tracker-lr", type=Path, default=None,
        help="check_tracker_lr.py output; defaults to <output-dir>/tracker_lr.json",
    )
    parser.add_argument(
        "--glove-sync", type=Path, default=None,
        help="check_glove_sync.py output; defaults to <output-dir>/glove_sync.json",
    )
    parser.add_argument(
        "--from-json", action="store_true",
        help="re-render index.html from an existing gallery.json (template edits, no recomputation)",
    )
    args = parser.parse_args()
    global SLUG_OVERRIDES
    _ovr = (args.output_dir / args.review_dir / "slug_overrides.json")
    SLUG_OVERRIDES = json.loads(_ovr.read_text(encoding="utf-8")) if _ovr.exists() else {}

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from five_tracker_smplx.model import ModelPaths, load_body_model

    if args.from_json:
        payload = json.loads((args.output_dir / "gallery.json").read_text(encoding="utf-8"))
        render(args.output_dir, payload)
        print(f"[gallery] re-rendered {len(payload['sessions'])} sessions -> {args.output_dir / 'index.html'}")
        return

    projected = args.batch_root / "projected"
    proposals = args.batch_root / "proposals"
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)

    body = load_body_model(ModelPaths(args.body_model_path, None, "smplx", "neutral"), args.device).eval()
    for parameter in body.parameters():
        parameter.requires_grad_(False)
    faces = np.asarray(body.faces, dtype=np.int64)

    sessions = sorted(
        d for d in projected.iterdir()
        if d.is_dir() and (d / "smplx_profile_driven.npz").exists()
    )
    if args.limit:
        sessions = sessions[:args.limit]

    cards, failures = [], []
    for index, session_dir in enumerate(sessions, 1):
        try:
            card = build_session(session_dir, body, args.device, faces)
        except Exception as error:  # a single unreadable session must not sink the page
            print(f"[gallery {index}/{len(sessions)}] {session_dir.name}: skipped ({error!r})", flush=True)
            continue
        card["vw"] = (output / args.viewer_dir / card["id"] / "index.html").exists()
        # The review pages are named by the ASCII slug, not the full id:
        # Chrome will not load media whose filename carries the task name.
        slug = SLUG_OVERRIDES.get(card["id"]) or re.sub(r"^(session_\d+_\d+).*", r"\1", card["id"])
        card["rvslug"] = slug
        card["rv"] = (output / args.review_dir / f"{slug}.review.html").exists()
        cards.append(card)
        if index % 25 == 0 or index == len(sessions):
            print(f"[gallery {index}/{len(sessions)}] {session_dir.name}", flush=True)

    # Only filled in when the matching quality-check JSON exists; a batch that
    # has not had the checks run reached the failure loop with them unbound.
    by_tracker: dict = {}
    by_drift: dict = {}
    by_session: dict = {}
    tracker_path = args.tracker_lr or (output / "tracker_lr.json")
    if tracker_path.exists():
        by_tracker = {row["session"]: row for row in json.loads(tracker_path.read_text(encoding="utf-8"))}
        for card in cards:
            row = by_tracker.get(card["id"])
            if not row:
                continue
            card["trk"] = row["status"]
            if row.get("swapped_pairs"):
                card["trkpairs"] = row["swapped_pairs"]
            if row.get("frozen_trackers"):
                card["frozen"] = row["frozen_trackers"]

    drift_path = args.tracker_drift or (output / "tracker_drift.json")
    if drift_path.exists():
        by_drift = {row["session"]: row for row in json.loads(drift_path.read_text(encoding="utf-8"))}
        for card in cards:
            row = by_drift.get(card["id"])
            if row and row.get("status") not in (None, "ok"):
                card["drift"] = row["status"]
                card["driftat"] = row.get("flagged") or []

    sync_path = args.glove_sync or (output / "glove_sync.json")
    if sync_path.exists():
        by_session = {row["session"]: row for row in json.loads(sync_path.read_text(encoding="utf-8"))}
        for card in cards:
            row = by_session.get(card["id"])
            if row:
                card["hand"] = row["status"]
                card["lag"] = row.get("lag_s")
                card["nodes"] = row.get("glove_nodes")
        payload_sync = {"sessions": len(by_session)}
    else:
        payload_sync = None

    # pipeline_report.json is rewritten by any single-session pipeline run, so
    # it cannot be trusted to list what is still outstanding.  Derive the
    # failures from the artefacts on disk instead.
    # Must follow the batch being built, not a fixed path: the L1 tree built
    # with the main batch's sync root listed all 1083 main sessions as L1
    # failures.
    sync_root = args.sync_root
    if sync_root is None:
        raise SystemExit("--sync-root is required (the input tree containing ego_data_sync/<batch>/<session>)")
    done = {card["id"] for card in cards}
    all_sessions = {}
    # rglob, not a fixed two-level glob: wuji/test/<session>/ sits one level
    # deeper and "*/session_*/" silently dropped those recordings.
    for path in sync_root.rglob("*.tracker.npz"):
        rel = path.parent.relative_to(sync_root)
        all_sessions[path.name[: -len(".tracker.npz")]] = (str(rel.parent), path)
    for session, (dataset, tracker_path) in sorted(all_sessions.items()):
        if session in done:
            continue
        kind, reason = failure_reason(session, proposals, {})
        # A never-recorded tracker role is a property of the capture, not a
        # retry that has not happened yet.
        absent = _missing_tracker_roles(tracker_path)
        if absent:
            kind = "no_tracker"
            reason = "缺少 tracker 数据流：" + "、".join(absent) + "（该录制无法投影）"
        marker = projected.parent / "projected" / f".failed_{session}"
        attempts = 0
        if marker.exists():
            try:
                attempts = int(marker.read_text().strip() or 0)
            except (OSError, ValueError):
                attempts = 0
        if attempts >= 99:
            reason = f"{reason}（已停止重试）"
        elif attempts:
            reason = f"{reason}（已重试 {attempts} 轮学习率阶梯仍失败）"
        failure = {"id": session, "st": "failed", "kind": kind, "why": reason, "ds": dataset, "fr": 0}
        for source, key in ((by_tracker if tracker_path.exists() else {}, "trk"),
                            (by_session if sync_path.exists() else {}, "hand")):
            row = source.get(session)
            if row:
                failure[key] = row["status"]
        if drift_path.exists():
            row = by_drift.get(session)
            if row and row.get("status") not in (None, "ok"):
                failure["drift"] = row["status"]
                failure["driftat"] = row.get("flagged") or []
        if tracker_path.exists():
            row = by_tracker.get(session)
            if row and row.get("frozen_trackers"):
                failure["frozen"] = row["frozen_trackers"]
        failures.append(failure)

    # Must stay below the loop that fills `failures`: tagging earlier iterated
    # an empty list and shipped a gallery whose failure cards all claimed to
    # have no review page.
    for row in failures:
        slug = SLUG_OVERRIDES.get(row["id"]) or re.sub(r"^(session_\d+_\d+).*", r"\1", row["id"])
        row["rvslug"] = slug
        row["rv"] = (output / args.review_dir / f"{slug}.review.html").exists()

    payload = {
        "title": args.title,
        "batch_root": str(args.batch_root),
        "viewer_dir": args.viewer_dir,
        "review_dir": args.review_dir,
        "tracker_labels": list(TRACKER_LABELS),
        "parents": [int(p) for p in body.parents[:BODY_JOINTS].cpu().numpy()],
        "sessions": cards,
        "failures": failures,
        "glove_sync": payload_sync,
    }
    (output / "gallery.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    render(output, payload)
    print(f"[gallery] {len(cards)} sessions, {len(failures)} failures -> {output / 'index.html'}")


if __name__ == "__main__":
    main()
