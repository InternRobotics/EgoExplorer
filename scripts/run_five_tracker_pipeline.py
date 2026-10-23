#!/usr/bin/env python3
"""Batch five-tracker -> AvatarPoser -> projected SMPL-X reconstruction.

The command deliberately keeps the learned proposal and differentiable
projection as separate, inspectable stages.  It is safe to run on a folder of
raw ``*.tracker.npz`` sessions: sessions with too few valid tracker samples
are reported and skipped instead of aborting the whole batch.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

RAW_ROLES = ("camera", "left_wrist", "right_wrist", "left_ankle", "right_ankle")
PROFILE_ROLES = ("head", "left_hand", "right_hand", "left_foot", "right_foot")


def session_id(path: Path) -> str:
    suffix = ".tracker.npz"
    return path.name[:-len(suffix)] if path.name.endswith(suffix) else path.stem


def _timestamp_seconds(raw: np.lib.npyio.NpzFile) -> np.ndarray:
    for key in ("tracker_0__pose__t_host_us", "tracker_0__pose__t_us", "tracker_0__pose__t_dev_us"):
        if key in raw:
            t = np.asarray(raw[key], dtype=np.float64)
            if len(t) >= 2 and np.isfinite(t).all() and np.nanmax(np.abs(np.diff(t))) > 10:
                t = (t - t[0]) * 1e-6
            else:
                t = t - t[0]
            # A few recordings carry non-monotonic host timestamps (wuji 0918:
            # two frames 7 ms early).  Slerp downstream needs strictly increasing
            # times, and dropping frames would misalign the index-sliced raw
            # streams, so nudge offenders onto the running maximum instead.
            if len(t) > 1 and np.any(np.diff(t) <= 0):
                t = np.maximum.accumulate(t)
                ties = np.concatenate(([False], np.diff(t) <= 0))
                t = t + np.cumsum(ties) * 1e-4
            return t
    raise KeyError("raw session has no tracker timestamp field")


def _raw_arrays(path: Path):
    raw = np.load(path, allow_pickle=False)
    t = _timestamp_seconds(raw)
    count = len(t)
    positions, rotations, valid = [], [], []
    for role in RAW_ROLES:
        pkey = f"tracker_0__pose.{role}.position"
        rkey = f"tracker_0__pose.{role}.orientation"
        if pkey not in raw or rkey not in raw:
            raw.close()
            raise ValueError(f"missing {role} position/orientation")
        positions.append(np.asarray(raw[pkey][:count], dtype=np.float32))
        # The recorder stores quaternions as wxyz; scipy expects xyzw.
        q = np.asarray(raw[rkey][:count], dtype=np.float64)
        q_ok = np.isfinite(q).all(axis=-1) & (np.linalg.norm(q, axis=-1) > 1e-8)
        q_safe = np.where(q_ok[:, None], q, np.array([1., 0., 0., 0.]))
        rotations.append(Rotation.from_quat(q_safe[:, [1, 2, 3, 0]]).as_matrix().astype(np.float32))
        vkey = f"tracker_0__pose.{role}.valid"
        valid.append((np.asarray(raw[vkey][:count], dtype=bool) if vkey in raw else np.ones(count, dtype=bool)) & q_ok)
    raw.close()
    p = np.stack(positions, axis=1)
    r = np.stack(rotations, axis=1)
    v = np.stack(valid, axis=1) & np.isfinite(p).all(axis=-1) & np.isfinite(r).all(axis=(-2, -1))
    return t, p, r, v


def estimate_ground_y(path: Path) -> float:
    """Estimate the raw-coordinate sole plane from the lower ankle percentile."""
    _, p, _, valid = _raw_arrays(path)
    ankle = p[:, 3:5, 1][valid[:, 3:5]]
    if ankle.size == 0:
        raise ValueError("cannot estimate floor: no valid ankle samples")
    return float(np.percentile(ankle, 5.0) - 0.12)


def write_reference(raw_path: Path, profile_path: Path, out_dir: Path, ground_y: float | None) -> tuple[Path, dict]:
    t, _, _, valid = _raw_arrays(raw_path)
    if len(t) < 2 or any(int(valid[:, i].sum()) < 2 for i in range(5)):
        raise ValueError("fewer than two valid samples for one or more trackers")
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    fit = profile.get("fit", {})
    mounts = profile["tracker_mounts"]
    mount_p, mount_r = [], []
    for role in PROFILE_ROLES:
        item = mounts[role]
        mount_p.append(item.get("position_in_joint_frame_m", item.get("neutral_world_offset_m", [0, 0, 0])))
        q = item.get("rotation_joint_tracker_xyzw", item.get("rotation_in_joint_frame_xyzw"))
        if q is None:
            q = item.get("rotation_pre_joint_xyzw", [0, 0, 0, 1])
        mount_r.append(Rotation.from_quat(np.asarray(q, dtype=np.float64)).as_matrix())
    n = len(t)
    out_dir.mkdir(parents=True, exist_ok=True)
    ref_path = out_dir / "smplx_profile_driven.npz"
    np.savez_compressed(
        ref_path, timestamps=t.astype(np.float64), betas=np.asarray(fit.get("betas", np.zeros(10)), dtype=np.float32),
        scale=np.asarray(float(fit.get("scale", 1.0)), dtype=np.float32),
        attachment_position=np.asarray(mount_p, dtype=np.float32), attachment_rotation=np.asarray(mount_r, dtype=np.float32),
        global_orient=np.zeros((n, 3), np.float32), body_pose=np.zeros((n, 63), np.float32), transl=np.zeros((n, 3), np.float32),
        left_hand_pose=np.zeros((n, 45), np.float32), right_hand_pose=np.zeros((n, 45), np.float32),
    )
    floor = estimate_ground_y(raw_path) if ground_y is None else float(ground_y)
    meta = {"source": str(raw_path.resolve()), "session": session_id(raw_path), "ground_y_m": floor,
            "optimization_ground_y_m": floor, "frames": n, "valid_counts": valid.sum(axis=0).tolist(),
            "timestamp_source": "host_us/us/dev_us fallback"}
    ref_path.with_suffix(".json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return ref_path, meta


def _run(cmd: list[str], root: Path, env: dict[str, str]) -> None:
    subprocess.run(cmd, cwd=root, env=env, check=True)


def process_one(raw_path: Path, args, root: Path) -> dict:
    sid = session_id(raw_path)
    ref_dir = args.output_root / "references" / sid
    proposal_dir = args.output_root / "proposals" / sid
    projected_dir = args.output_root / "projected" / sid
    if args.skip_existing and (projected_dir / "smplx_profile_driven.npz").exists():
        return {"session": sid, "status": "skipped", "frames": int(np.load(projected_dir / "smplx_profile_driven.npz")["timestamps"].shape[0])}
    ref, meta = write_reference(raw_path, args.subject_profile, ref_dir, args.ground_y)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "src") + os.pathsep + str(root / "scripts") + os.pathsep + env.get("PYTHONPATH", "")
    if args.workers > 1:
        env["OMP_NUM_THREADS"] = "1"; env["MKL_NUM_THREADS"] = "1"
    _run([sys.executable, "scripts/export_avatarposer_five_finetuned.py", "--reference", str(ref), "--checkpoint", str(args.checkpoint), "--output", str(proposal_dir), "--device", args.device, "--batch-size", "128", "--window", "40"], root, env)
    projector_cmd = [sys.executable, str(args.projector_script), "--trajectory", str(proposal_dir / "smplx_profile_driven.npz"), "--output", str(projected_dir), "--iterations", str(args.iterations), "--learning-rate", str(args.learning_rate), "--chunk-size", str(args.chunk_size), "--contact-weight", "0", "--mesh-collision-weight", str(args.mesh_collision_weight), "--mesh-collision-radius", str(args.mesh_collision_radius), "--mesh-sample-count", str(args.mesh_sample_count), "--global-floor-penetration-weight", "50", "--joint-limit-weight", "100", "--max-knee-flexion-deg", "125", "--max-elbow-flexion-deg", "140", "--torso-lean-limit-deg", str(args.torso_lean_limit_deg), "--knee-hyperextension-weight", str(args.knee_hyperextension_weight), "--pelvis-roll-weight", str(args.pelvis_roll_weight), "--hull-collision-weight", "0", "--device", args.device, "--body-model-path", str(args.body_model_path)]
    # Only the merged projector (project_sparse_tracker_baseline_l2.py) knows --surface-mode; the production script keeps its legacy term.
    if args.surface_mode:
        projector_cmd += ["--surface-mode", args.surface_mode]
    _run(projector_cmd, root, env)
    diag_path = projected_dir / "projection_diagnostics.json"
    if diag_path.exists():
        meta["projection_diagnostics"] = json.loads(diag_path.read_text(encoding="utf-8"))
    tracker_diag = projected_dir / "tracker_diagnostics.npz"
    if tracker_diag.exists():
        with np.load(tracker_diag) as d:
            residual = np.asarray(d["position_residual_m"], dtype=np.float64)
            finite = np.isfinite(residual)
            meta["position_rms_m"] = float(np.sqrt(np.mean(residual[finite] ** 2))) if finite.any() else float("nan")
    with np.load(ref) as reference_values:
        reference_timestamps = np.asarray(reference_values["timestamps"], dtype=np.float64)
    meta["maximum_gap_s"] = float(np.max(np.diff(reference_timestamps))) if len(reference_timestamps) > 1 else 0.0
    result_meta_path = projected_dir / "smplx_profile_driven.json"
    if result_meta_path.exists():
        result_meta = json.loads(result_meta_path.read_text(encoding="utf-8"))
        result_meta.update({"position_rms_m": meta.get("position_rms_m", float("nan")), "maximum_gap_s": meta["maximum_gap_s"]})
        result_meta_path.write_text(json.dumps(result_meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    result_path = projected_dir / "smplx_profile_driven.npz"
    if result_path.exists():
        with np.load(result_path) as z: meta["frames"] = int(len(z["timestamps"]))
    (projected_dir / "pipeline_metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"session": sid, "status": "ok", "frames": meta["frames"], "output": str(projected_dir)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-root", type=Path, required=True)
    p.add_argument("--subject-profile", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--body-model-path", type=Path, default=Path("models/smplx"))
    p.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    p.add_argument("--workers", type=int, default=1, help="parallel sessions; use 1 on CPU to avoid PyTorch oversubscription")
    # Defaults are the production configuration (60 Adam steps at lr 0.004; L1 limits with a 90-degree
    # torso lean so bends and squats are not pinned; L2 signed_top8 surface term; L3 floor).
    p.add_argument("--iterations", type=int, default=60)
    p.add_argument("--learning-rate", type=float, default=.004, help="0.008 diverges on 20-35%% of real sessions")
    p.add_argument("--chunk-size", type=int, default=256)
    p.add_argument("--ground-y", type=float, default=None, help="raw-coordinate floor; default is ankle 5th percentile minus 0.12 m")
    p.add_argument("--visualize", action="store_true"); p.add_argument("--viewer-max-frames", type=int, default=0)
    p.add_argument("--skip-existing", action="store_true"); p.add_argument("--session", action="append", default=[])
    p.add_argument("--estimate-seconds-per-frame", type=float, default=.16, help="ETA prior in seconds per frame")
    p.add_argument("--pelvis-roll-weight", type=float, default=100.0, help="lateral pelvis tilt penalty")
    p.add_argument("--knee-hyperextension-weight", type=float, default=100.0, help="signed backward-knee penalty")
    p.add_argument("--torso-lean-limit-deg", type=float, default=90.0, help="Layer-1 torso lean soft limit; 15 pins bends, lunges and sit-downs")
    p.add_argument("--projector-script", type=Path, default=Path("scripts/project_sparse_tracker_baseline_l2.py"), help="projection script")
    p.add_argument("--surface-mode", default="signed_top8", help="Layer-2 surface term mode")
    p.add_argument("--mesh-collision-weight", type=float, default=3000.0, help="Layer-2 weight")
    p.add_argument("--mesh-collision-radius", type=float, default=0.018, help="Layer-2 clearance radius in metres")
    p.add_argument("--mesh-sample-count", type=int, default=256, help="Layer-2 surface samples")
    args = p.parse_args(); args.workers = max(1, int(args.workers))
    root = Path(__file__).resolve().parents[1]
    raws = sorted(args.input_root.rglob("*.tracker.npz"))
    if args.session: raws = [x for x in raws if session_id(x) in set(args.session)]
    if not raws: raise SystemExit("no *.tracker.npz sessions found")
    args.output_root.mkdir(parents=True, exist_ok=True)
    report = {"input_root": str(args.input_root.resolve()), "output_root": str(args.output_root.resolve()), "sessions": [], "started_at": time.time(), "estimate_seconds_per_frame": args.estimate_seconds_per_frame, "workers": args.workers}
    started = time.perf_counter()
    total_frames = 0
    for raw_path in raws:
        try:
            with np.load(raw_path, allow_pickle=False) as raw_values:
                total_frames += max(1, len(_timestamp_seconds(raw_values)))
        except Exception:
            # The session will receive a useful per-file failure below; ETA can
            # still be printed without making discovery itself fatal.
            total_frames += 1
    print(f"[pipeline] {len(raws)} sessions, ~{total_frames} frames; workers={args.workers}; ETA prior {total_frames * args.estimate_seconds_per_frame / 60:.1f} min", flush=True)
    def wrapped(x):
        t0 = time.perf_counter()
        try:
            r = process_one(x, args, root); r["elapsed_s"] = time.perf_counter() - t0; return r
        except Exception as exc:
            return {"session": session_id(x), "status": "failed", "error": repr(exc), "elapsed_s": time.perf_counter() - t0}
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(wrapped, x) for x in raws]
        for i, f in enumerate(concurrent.futures.as_completed(futures), 1):
            r = f.result(); report["sessions"].append(r)
            elapsed = time.perf_counter() - started; eta = elapsed / i * (len(raws) - i)
            print(f"[pipeline {i}/{len(raws)}] {r['session']}: {r['status']} ({r.get('elapsed_s', 0):.1f}s), ETA {eta/60:.1f} min", flush=True)
    if args.visualize:
        env = os.environ.copy(); env["PYTHONPATH"] = str(root / "src") + os.pathsep + env.get("PYTHONPATH", "")
        good = [x for x in report["sessions"] if x["status"] in ("ok", "skipped")]
        if good:
            cmd = [sys.executable, "-m", "five_tracker_smplx.cli", "full-mesh-viewer", "--input-root", str(args.input_root), "--body-output-dir", str(args.output_root / "projected"), "--subject-profile", str(args.subject_profile), "--output-dir", str(args.output_root / "viewer"), "--body-model-path", str(args.body_model_path), "--device", args.device, "--viewer-max-frames", str(args.viewer_max_frames), "--protocol-mode", "none", "--skip-missing-natural"]
            try: _run(cmd, root, env); report["viewer"] = str(args.output_root / "viewer" / "index.html")
            except Exception as exc: report["viewer_error"] = repr(exc)
    report["elapsed_s"] = time.perf_counter() - started; report["finished_at"] = time.time()
    (args.output_root / "pipeline_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"ok": sum(x["status"] == "ok" for x in report["sessions"]), "skipped": sum(x["status"] == "skipped" for x in report["sessions"]), "failed": sum(x["status"] == "failed" for x in report["sessions"]), "elapsed_s": report["elapsed_s"], "report": str(args.output_root / "pipeline_report.json")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
