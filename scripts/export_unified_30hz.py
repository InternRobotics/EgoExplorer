#!/usr/bin/env python
"""Resample every stream of one recording onto a single fixed-rate timeline.

Nothing in these recordings shares a clock rate: the tracker runs at 15 Hz on
the MANUS rigs and 65 Hz on the Wuji rig, gloves at 21.8 Hz or 120 Hz, video at
30-60 Hz, and the SMPL-X result is written on the tracker's own timestamps
(relative to its first sample, while every other stream is absolute host wall
clock).  Consumers that zip these arrays by index are silently wrong.

This exports one npz per session where index i of every array means the same
instant.  The grid is anchored to absolute time so the same instant gets the
same index in any session exported at the same rate.

Resampling respects what each signal is:
  rotations          slerp (never lerp on axis-angle or matrices)
  positions          linear
  validity           logical AND over the window it came from
  tactile            max over the window — contact events are short, and point
                     sampling a 120 Hz signal at 30 Hz would drop most of them
  video              nearest frame index, plus how far away it actually was

Every stream also gets an `*_interpolated` mask flagging grid samples that sit
further than one native period from any real sample, so a consumer can tell a
measurement from a bridge across a dropout.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

TRACKER_ROLES = ("camera", "left_wrist", "right_wrist", "left_ankle", "right_ankle")
DEFAULT_RATE = 30.0
# A grid sample further than this many native periods from any real sample was
# bridged across a dropout rather than measured.
GAP_PERIODS = 1.5


def _slerp_axis_angle(times: np.ndarray, values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Slerp a [T, J*3] axis-angle stack onto `grid`, joint by joint."""
    flat = values.reshape(len(values), -1, 3)
    out = np.empty((len(grid), flat.shape[1], 3), dtype=np.float32)
    for joint in range(flat.shape[1]):
        rotations = Rotation.from_rotvec(flat[:, joint].astype(np.float64))
        out[:, joint] = Slerp(times, rotations)(grid).as_rotvec().astype(np.float32)
    return out.reshape(len(grid), -1)


def _slerp_matrices(times: np.ndarray, matrices: np.ndarray, grid: np.ndarray) -> np.ndarray:
    shape = matrices.shape
    flat = matrices.reshape(shape[0], -1, 3, 3)
    out = np.empty((len(grid), flat.shape[1], 3, 3), dtype=np.float32)
    for index in range(flat.shape[1]):
        rotations = Rotation.from_matrix(flat[:, index].astype(np.float64))
        out[:, index] = Slerp(times, rotations)(grid).as_matrix().astype(np.float32)
    return out.reshape((len(grid),) + shape[1:])


def _lerp(times: np.ndarray, values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    flat = values.reshape(len(values), -1)
    out = np.stack([np.interp(grid, times, flat[:, c]) for c in range(flat.shape[1])], axis=-1)
    return out.reshape((len(grid),) + values.shape[1:]).astype(np.float32)




# Wuji touch pipeline (wuji_correct / wuji_layout / health table) ships in <toolkit>/tactile;
# EGOEXPLORER_TACTILE_DIR points elsewhere if needed.
TACTILE_WORK = Path(os.environ.get("EGOEXPLORER_TACTILE_DIR", Path(__file__).resolve().parents[1] / "tactile"))
COARSE_PARTS = ("thumb", "index", "middle", "ring", "pinky", "palm")


def _load_tactile_pipeline():
    """Import the project's own L0->L4 tactile pipeline, or None if absent.

    The levels are defined once in wuji_correct.py (L0 raw, L1 baseline-removed
    residual, L2 noise-normalised z, L3 gain-corrected, L4 per-region contact
    with dual-criterion hysteresis).  Re-deriving them here would guarantee the
    exported contact disagrees with the reference viewer, so import that module
    rather than reimplementing its thresholds.
    """
    if not (TACTILE_WORK / "wuji_correct.py").exists():
        return None
    if str(TACTILE_WORK) not in sys.path:
        sys.path.insert(0, str(TACTILE_WORK))
    try:
        import wuji_correct
        import wuji_layout
        return wuji_correct, wuji_layout
    except Exception:
        return None


def _load_health(session: str) -> dict | None:
    """Per-taxel health coefficients: session-specific when known, else the day's."""
    path = TACTILE_WORK / "health" / "health_db.npz"
    if not path.exists():
        return None
    with np.load(path, allow_pickle=True) as db:
        names = [str(x) for x in db["sessions"]]
        days = [str(x) for x in db["uday"]]
        if session in names:
            index = names.index(session)
            return {side: np.asarray(db[f"{side}_health_session"][index]) for side in ("left", "right")}
        # Fall back to the health measured on the session's own capture day.
        match = re.search(r"(\d{8})", session)
        day = match.group(1) if match else ""
        index = days.index(day) if day in days else len(days) - 1
        return {side: np.asarray(db[f"{side}_health_day"][index]) for side in ("left", "right")}


def _window_or(times: np.ndarray, values: np.ndarray, grid: np.ndarray, step: float) -> np.ndarray:
    """Any-contact-in-the-window; a 30 Hz probe would miss brief touches."""
    return _window_reduce(times, values.astype(np.uint8), grid, step, np.max).astype(bool)


def _finite_rows(values: np.ndarray) -> np.ndarray:
    flat = values.reshape(len(values), -1)
    return np.isfinite(flat).all(axis=1)


def _resample_tracker(times: np.ndarray, positions: np.ndarray, quaternions: np.ndarray,
                      valid: np.ndarray, grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resample each tracker independently over only its usable samples.

    Recordings contain frames where a tracker reports a zero-norm quaternion or
    a non-finite position while still flagged valid.  Feeding those to Slerp
    aborts the whole session, and interpolating through them would invent pose;
    each tracker is therefore resampled over its own good samples and marked
    unusable wherever the grid leaves their span.
    """
    count = positions.shape[1]
    out_pos = np.zeros((len(grid), count, 3), dtype=np.float32)
    out_rot = np.tile(np.eye(3, dtype=np.float32), (len(grid), count, 1, 1))
    out_ok = np.zeros((len(grid), count), dtype=bool)
    for index in range(count):
        quaternion = quaternions[:, index]
        norm = np.linalg.norm(quaternion, axis=-1)
        usable = (valid[:, index] & _finite_rows(positions[:, index])
                  & np.isfinite(quaternion).all(axis=-1) & (norm > 1e-8))
        if usable.sum() < 2:
            continue
        good_times = times[usable]
        inside = (grid >= good_times[0]) & (grid <= good_times[-1])
        if not inside.any():
            continue
        unit = quaternion[usable] / norm[usable][:, None]
        rotations = Rotation.from_quat(np.stack([unit[:, 1], unit[:, 2], unit[:, 3], unit[:, 0]], axis=-1))
        out_rot[inside, index] = Slerp(good_times, rotations)(grid[inside]).as_matrix().astype(np.float32)
        for axis in range(3):
            out_pos[inside, index, axis] = np.interp(grid[inside], good_times, positions[usable, index, axis])
        out_ok[inside, index] = True
    return out_pos, out_rot, out_ok


def _gap_mask(times: np.ndarray, grid: np.ndarray, native_period: float) -> np.ndarray:
    index = np.clip(np.searchsorted(times, grid), 1, len(times) - 1)
    nearest = np.minimum(np.abs(grid - times[index - 1]), np.abs(times[index] - grid))
    return nearest > GAP_PERIODS * native_period


def _monotonic(times: np.ndarray) -> np.ndarray:
    """Force a host-clock stream to be strictly increasing.

    One wuji 0918 recording carries two frames stamped ~7 ms before their
    predecessor.  Slerp refuses non-increasing times, and dropping the offending
    samples would misalign every array that is indexed alongside them, so the
    offenders are nudged onto the running maximum instead.
    """
    t = np.asarray(times, dtype=np.float64)
    if len(t) < 2 or np.all(np.diff(t) > 0):
        return t
    t = np.maximum.accumulate(t)
    ties = np.concatenate(([False], np.diff(t) <= 0))
    return t + np.cumsum(ties) * 1e-4


def _window_reduce(times: np.ndarray, values: np.ndarray, grid: np.ndarray, step: float,
                   reducer=np.max) -> np.ndarray:
    """Reduce every native sample inside each grid interval (not point sampling)."""
    edges = np.concatenate((grid - step / 2, [grid[-1] + step / 2]))
    slot = np.clip(np.searchsorted(edges, times) - 1, 0, len(grid) - 1)
    out = np.zeros((len(grid),) + values.shape[1:], dtype=values.dtype)
    order = np.argsort(slot, kind="stable")
    slot_sorted, values_sorted = slot[order], values[order]
    bounds = np.searchsorted(slot_sorted, np.arange(len(grid) + 1))
    for i in range(len(grid)):
        lo, hi = bounds[i], bounds[i + 1]
        if hi > lo:
            out[i] = reducer(values_sorted[lo:hi], axis=0)
        elif i:
            out[i] = out[i - 1]   # hold across an empty slot rather than invent a zero
    return out


def export_session(session_dir: Path, output_path: Path, rate: float = DEFAULT_RATE,
                   keep_raw_tactile: bool = False, source_override: Path | None = None,
                   session_name: str | None = None, failure: dict | None = None) -> dict:
    """Resample one recording onto the shared 30 Hz grid.

    With `source_override` the recording has no SMPL-X result -- the projection
    diverged or has not run yet -- so the grid is built from the tracker clock
    alone and only the measured streams (tracker, glove, tactile, video) are
    written.  Everything downstream of here is shared with the normal path, so
    a failed session's tactile is computed by exactly the same L0->L4 code as a
    successful one rather than by a parallel implementation that could drift.
    """
    no_body = source_override is not None
    if no_body:
        source = Path(source_override)
        name = session_name or source.name.replace(".tracker.npz", "")
        meta: dict = {}
        body = None
        hand_source = "none_no_projection"
    else:
        name = session_dir.name
        meta = json.loads((session_dir / "smplx_profile_driven.json").read_text(encoding="utf-8"))
        source = Path(meta["source"])
        with np.load(session_dir / "smplx_profile_driven.npz", allow_pickle=False) as z:
            body = {k: np.asarray(z[k]) for k in z.files}
    # A glove fusion, when one has been run, supplies real hand pose on the same
    # body frames; prefer it so the unified export carries measured fingers
    # instead of the neutral placeholder.
    fused_path = session_dir / "smplx_with_manus_hands.npz"
    if not no_body:
        hand_source = str(meta.get("hand_pose_source", "unknown"))
    if body is not None and fused_path.exists():
        with np.load(fused_path, allow_pickle=False) as z:
            fused = {k: np.asarray(z[k]) for k in z.files}
        if len(fused["timestamps"]) == len(body["timestamps"]):
            body["left_hand_pose"] = fused["left_hand_pose"]
            body["right_hand_pose"] = fused["right_hand_pose"]
            hand_source = "glove_fused"
    with np.load(source, allow_pickle=False) as z:
        has_pose = "tracker_0__pose__t_host_us" in z
        if not has_pose and not no_body:
            raise KeyError("tracker_0__pose__t_host_us is not a file in the archive")
        if has_pose:
            tracker_time = _monotonic(np.asarray(z["tracker_0__pose__t_host_us"], dtype=np.float64) * 1e-6)
        else:
            # Some recovered recordings carry no tracker stream at all.  With no
            # body to export either, the video clock is the only timebase left,
            # and it is enough to show the operator what was recorded.
            video_only = source.with_name(source.name.replace(".tracker.npz", ".video.npz"))
            with np.load(video_only, allow_pickle=False) as vz:
                tracker_time = _monotonic(np.asarray(vz["t_host_us"], dtype=np.float64) * 1e-6)

        def _role(role: str, suffix: str, width: int, fill: float) -> np.ndarray:
            key = f"tracker_0__pose.{role}.{suffix}"
            if key in z:
                return np.asarray(z[key], dtype=np.float64)
            # A role the rig never recorded is reported as unusable rather than
            # aborting the export; the mask below is what consumers read.
            if not no_body:
                raise KeyError(f"{key} is not a file in the archive")
            out = np.zeros((len(tracker_time), width), dtype=np.float64)
            if fill:
                out[:, -1] = fill
            return out

        tracker_pos = np.stack([_role(r, "position", 3, 0.0) for r in TRACKER_ROLES], axis=1)
        tracker_quat = np.stack([_role(r, "orientation", 4, 1.0) for r in TRACKER_ROLES], axis=1)
        tracker_valid = np.stack([
            np.asarray(z[f"tracker_0__pose.{r}.valid"], dtype=bool)
            if f"tracker_0__pose.{r}.valid" in z
            else np.full(len(tracker_time), f"tracker_0__pose.{r}.position" in z, dtype=bool)
            for r in TRACKER_ROLES], axis=1)

    # The SMPL-X result is stamped relative to the tracker's first sample while
    # every other stream is absolute; lift it onto the shared clock.
    if no_body:
        body_time = None
        start, end = tracker_time[0], tracker_time[-1]
    else:
        body_time = np.asarray(body["timestamps"], dtype=np.float64)
        if body_time[0] < 1e6:
            body_time = body_time + tracker_time[0]
        if len(body_time) != len(tracker_time):
            raise ValueError(f"body has {len(body_time)} frames, tracker has {len(tracker_time)}")
        start = max(body_time[0], tracker_time[0])
        end = min(body_time[-1], tracker_time[-1])
    step = 1.0 / rate
    # Anchor to absolute time so the same instant lands on the same index in
    # every session exported at this rate.
    grid = np.arange(np.ceil(start * rate) / rate, end, step)
    if len(grid) < 2:
        raise ValueError("recording too short for the requested rate")

    tracker_period = float(np.median(np.diff(tracker_time)))
    out: dict[str, np.ndarray] = {
        "timestamps": grid.astype(np.float64),
        "rate_hz": np.array(rate),
        "hand_pose_source": np.array(hand_source),
        # Raw capture frame is Y-up with the floor at this height; consumers
        # that want Z-up with the floor at 0 apply (x, y, z) -> (x, -z, y - ground).
        "ground_y_m": np.array(float(meta.get("ground_y_m", 0.0))),
        "coordinate_frame": np.array("raw_capture_y_up"),
        "tracker_interpolated": _gap_mask(tracker_time, grid, tracker_period),
        "tracker_native_hz": np.array(round(1.0 / tracker_period, 3)),
        "tracker_roles": np.array(list(TRACKER_ROLES)),
        "has_body": np.array(not no_body),
    }
    if not no_body:
        out.update({
            "betas": body["betas"],
            "scale": body["scale"],
            "attachment_position": body["attachment_position"],
            "attachment_rotation": body["attachment_rotation"],
            "global_orient": _slerp_axis_angle(body_time, body["global_orient"], grid),
            "body_pose": _slerp_axis_angle(body_time, body["body_pose"], grid),
            "transl": _lerp(body_time, body["transl"], grid),
            "left_hand_pose": _slerp_axis_angle(body_time, body["left_hand_pose"], grid),
            "right_hand_pose": _slerp_axis_angle(body_time, body["right_hand_pose"], grid),
            "body_interpolated": _gap_mask(body_time, grid, tracker_period),
        })
    if failure:
        out["failure_kind"] = np.array(str(failure.get("kind", "")))
        out["failure_why"] = np.array(str(failure.get("why", "")))

    position, rotation, usable = _resample_tracker(
        tracker_time, tracker_pos, tracker_quat, tracker_valid, grid)
    out["tracker_position"] = position
    out["tracker_rotation"] = rotation
    out["tracker_valid"] = usable

    report = {"session": name, "rate_hz": rate, "frames": int(len(grid)),
              "tracker_native_hz": float(out["tracker_native_hz"]),
              # >1 means the grid runs faster than the tracker ever sampled, so
              # that fraction of every output frame is interpolated by
              # construction — distinct from the dropout mask below.
              "upsample_ratio": round(rate / float(out["tracker_native_hz"]), 3),
              "body_interpolated_frac": (round(float(out["body_interpolated"].mean()), 4)
                                         if not no_body else None),
              "has_body": not no_body,
              "tracker_valid_frac": round(float(out["tracker_valid"].mean()), 4),
              "hand_pose_source": hand_source,
              "streams": (["tracker"] if no_body else ["body", "tracker"])}
    if failure:
        report["failure"] = failure

    glove_path = source.with_name(source.name.replace(".tracker.npz", ".npz"))
    if glove_path.exists():
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
        try:
            from five_tracker_smplx.manus_hands import read_manus_npz, wrist_local_points
            streams = read_manus_npz(glove_path)
            for side, stream in streams.items():
                times = stream.timestamps_us.astype(np.float64) * 1e-6
                period = float(np.median(np.diff(times)))
                local = wrist_local_points(stream)
                good = _finite_rows(local) & np.isfinite(times)
                if good.sum() < 2:
                    continue
                out[f"glove_{side}_points"] = _lerp(times[good], local[good], grid)
                out[f"glove_{side}_interpolated"] = _gap_mask(times[good], grid, period)
                out[f"glove_{side}_native_hz"] = np.array(round(1.0 / period, 3))
            out["glove_layout"] = np.array(next(iter(streams.values())).layout)
            report["streams"].append("glove")
            report["glove_layout"] = str(out["glove_layout"])
            report["glove_native_hz"] = float(out["glove_left_native_hz"])
        except Exception as error:
            report["glove_error"] = repr(error)[:200]

        pipeline = _load_tactile_pipeline()
        health = _load_health(name)
        with np.load(glove_path, allow_pickle=False) as z:
            tactile_keys = [k for k in z.files if k.endswith("__tactile")]
            for key in tactile_keys:
                device = key.split("__")[0]
                side_name = "left" if device == "glove_0" else "right"
                side = "L" if device == "glove_0" else "R"
                # Prefer the tactile stream's own clock: the skeleton stream can be one
                # sample shorter (seen on wuji 0929), which used to drop the whole hand.
                time_key = f"{device}__tactile__t_host_us"
                if time_key not in z.files:
                    time_key = f"{device}__hand_skeleton__t_host_us"
                if time_key not in z.files:
                    continue
                times = _monotonic(np.asarray(z[time_key], dtype=np.float64) * 1e-6)
                values = np.asarray(z[key], dtype=np.float32)
                if len(times) != len(values):
                    continue
                out[f"tactile_{side}_native_hz"] = np.array(
                    round(1.0 / float(np.median(np.diff(times))), 3))
                if pipeline is None or health is None:
                    # Without the health table only L0 is meaningful.
                    out[f"tactile_{side}_L0_raw"] = _window_reduce(times, values, grid, step, np.max)
                    report.setdefault("tactile", []).append(f"{device}->L0 only (无 health 表)")
                    continue
                wuji_correct, _ = pipeline
                native_fps = float(len(times) / max(times[-1] - times[0], 1e-3))
                # Run the levels at the native rate: the hysteresis minimum
                # durations are specified in seconds, and decimating first would
                # drop brief contacts before they can be detected.
                levels = wuji_correct.process(values, native_fps, health[side_name])
                out[f"tactile_{side}_L4_contact"] = _window_or(
                    times, levels["region_contact"], grid, step)
                out[f"tactile_{side}_L2_score"] = _window_reduce(
                    times, np.nan_to_num(levels["region_score"]).astype(np.float32),
                    grid, step, np.max)
                out[f"tactile_{side}_L1_residual"] = _window_reduce(
                    times, np.nan_to_num(levels["region_res"]).astype(np.float32),
                    grid, step, np.max)
                out[f"tactile_{side}_coarse_contact"] = _window_or(
                    times, levels["coarse_contact"], grid, step)
                out[f"tactile_{side}_coarse_score"] = _window_reduce(
                    times, np.nan_to_num(levels["coarse_score"]).astype(np.float32),
                    grid, step, np.max)
                out[f"tactile_{side}_coverage"] = levels["coverage"].astype(np.float32)
                out[f"tactile_{side}_sigma"] = levels["sigma"].astype(np.float32)
                if keep_raw_tactile:
                    out[f"tactile_{side}_L0_raw"] = _window_reduce(times, values, grid, step, np.max)
                report.setdefault("tactile", []).append(
                    f"{device}->{side} L1/L2/L4 (17区) + coarse(6部位)")
                report[f"tactile_{side}_contact_frac"] = round(
                    float(out[f"tactile_{side}_L4_contact"].mean()), 4)
            if tactile_keys and pipeline is not None and health is not None:
                _, wuji_layout = pipeline
                out["tactile_region_names"] = np.array(list(wuji_layout.REGION_NAMES))
                out["tactile_coarse_names"] = np.array(list(COARSE_PARTS))
                out["tactile_levels"] = np.array(
                    "L0 raw / L1 residual(减滚动p10基线) / L2 z(除稳健MAD) / "
                    "L3 gain-corrected / L4 contact(区域80分位+双判据迟滞)；定义见 wuji_correct.py")

    video_path = source.with_name(source.name.replace(".tracker.npz", ".video.npz"))
    if video_path.exists():
        with np.load(video_path, allow_pickle=False) as z:
            video_time = _monotonic(np.asarray(z["t_host_us"], dtype=np.float64) * 1e-6)
        index = np.clip(np.searchsorted(video_time, grid), 1, len(video_time) - 1)
        left, right = index - 1, index
        pick = np.where(np.abs(grid - video_time[left]) <= np.abs(video_time[right] - grid), left, right)
        out["video_frame_index"] = pick.astype(np.int32)
        out["video_time_offset_s"] = (video_time[pick] - grid).astype(np.float32)
        out["video_native_hz"] = np.array(round(1.0 / float(np.median(np.diff(video_time))), 3))
        report["streams"].append("video")
        report["video_native_hz"] = float(out["video_native_hz"])
        report["video_max_offset_ms"] = round(float(np.abs(out["video_time_offset_s"]).max() * 1000), 2)

    out["upsample_ratio"] = np.array(report["upsample_ratio"])
    out["source_tracker_npz"] = np.array(str(source))
    out["export_note"] = np.array(
        "所有数组按索引对齐到同一条 %.6g Hz 绝对时间栅格；旋转用 Slerp，位置线性插值，"
        "触觉在窗口内取峰值，视频取最近帧号。*_interpolated 为 True 的采样点是跨越丢帧空洞插出来的，不是测量值。" % rate)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **out)
    output_path.with_suffix(".json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--projected-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--rate", type=float, default=DEFAULT_RATE)
    parser.add_argument("--session", action="append", default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--keep-raw-tactile", action="store_true",
                        help="also store the full 744-channel grid (about 17 MB per hand per session)")
    args = parser.parse_args()

    names = args.session or sorted(
        d.name for d in args.projected_root.iterdir()
        if d.is_dir() and (d / "smplx_profile_driven.npz").exists())
    if args.limit:
        names = names[: args.limit]
    ok = failed = 0
    for index, name in enumerate(names, 1):
        target = args.output_root / f"{name}.npz"
        if args.skip_existing and target.exists():
            ok += 1
            continue
        try:
            report = export_session(args.projected_root / name, args.output_root / f"{name}.npz",
                                    args.rate, args.keep_raw_tactile)
            ok += 1
            if index % 25 == 0 or index == len(names) or len(names) <= 5:
                print(f"[{index}/{len(names)}] {name}: {report['frames']} 帧 "
                      f"tracker {report['tracker_native_hz']}Hz -> {args.rate}Hz "
                      f"插值率 {report['body_interpolated_frac']:.1%}", flush=True)
        except Exception as error:
            failed += 1
            print(f"[{index}/{len(names)}] {name}: FAILED {error!r}"[:200], flush=True)
    print(f"\n完成 {ok} 条，失败 {failed} 条 -> {args.output_root}")


if __name__ == "__main__":
    main()
