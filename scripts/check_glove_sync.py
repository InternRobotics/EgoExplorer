#!/usr/bin/env python
"""Check glove/tracker pairing and timing alignment for ego_data recordings.

The glove skeleton's root node carries a world-frame orientation on manus
recordings, so its angular speed can be cross-correlated against the tracker
wrist's angular speed.  The lag at peak correlation is the real timing offset,
and comparing the labelled pairing (glove_0 -> left) against the crossed one
reveals recordings where the two gloves were worn or labelled the wrong way
round.

Wuji recordings carry a different glove: 21 nodes per hand instead of manus's
25, no world-frame root orientation (the root quaternion is a constant
identity) and an extra tactile channel.  The correlation test cannot say
anything about those, so they are reported as `unsupported` rather than being
lumped in with genuine failures.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

ROLE_OF_SLOT = {0: "left", 1: "right"}
MAX_ANGULAR_SPEED = 20.0  # rad/s; anything above this is a corrupt quaternion
MIN_SAMPLES = 200
CONFIDENT = .5



def _dataset_of(tracker_npz: Path) -> str:
    """Dataset name = the first directory under the sync root, whatever the depth."""
    parts = tracker_npz.parts
    for index, part in enumerate(parts):
        if part in ("ego_data_sync", "ego_data_lr_fixed") and index + 1 < len(parts):
            return parts[index + 1]
    return tracker_npz.parent.parent.name


RESAMPLE_HZ = 30.0  # angular speed is measured on this grid, whatever the stream rate


def _angular_speed(quaternions: np.ndarray, times: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Angular speed (rad/s) on a common RESAMPLE_HZ grid.

    The 2026-09 MANUS recorder streams the 21-node skeleton at ~220 Hz; differencing
    consecutive raw quaternions there is dominated by jitter (the "speed" hits the
    20 rad/s clip on 5-10 % of frames) and the wrist correlation collapses to ~0.01,
    which read as "weak"/unsynchronised for 279 of 338 manus_new sessions.  Resampling
    the (normalised) quaternion components to 30 Hz first recovers 0.87 against the
    matching wrist.  The corrupt-frame fraction is still measured on the raw stream.
    """
    q = np.asarray(quaternions, dtype=np.float64)
    t = np.asarray(times, dtype=np.float64)
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    keep = np.isfinite(norm).ravel() & (norm.ravel() > 1e-6) & np.isfinite(t)
    if keep.sum() < 20:
        return None, None, 1.0
    q, t = q[keep] / norm[keep], t[keep]
    order = np.argsort(t, kind="stable")
    q, t = q[order], t[order]
    t, first = np.unique(t, return_index=True)
    q = q[first]
    if len(t) < 20 or t[-1] - t[0] < 1.0:
        return None, None, 1.0
    raw_dot = np.abs(np.sum(q[:-1] * q[1:], axis=-1)).clip(0, 1)
    raw_speed = 2 * np.arccos(raw_dot) / np.maximum(np.diff(t), 1e-6)
    corrupt = float((raw_speed > MAX_ANGULAR_SPEED).mean())
    # Keep the sign continuous before interpolating (q and -q are the same rotation).
    flip = np.cumsum(np.sum(q[:-1] * q[1:], axis=-1) < 0) % 2
    q[1:] *= np.where(flip, -1.0, 1.0)[:, None]
    grid = np.arange(t[0], t[-1], 1.0 / RESAMPLE_HZ)
    qi = np.stack([np.interp(grid, t, q[:, k]) for k in range(q.shape[1])], axis=-1)
    qi /= np.maximum(np.linalg.norm(qi, axis=-1, keepdims=True), 1e-9)
    dot = np.abs(np.sum(qi[:-1] * qi[1:], axis=-1)).clip(0, 1)
    speed = 2 * np.arccos(dot) / np.maximum(np.diff(grid), 1e-6)
    return np.clip(speed, 0, MAX_ANGULAR_SPEED), .5 * (grid[:-1] + grid[1:]), corrupt


def _best_lag(ta, va, tb, vb, span=2.0, step=1 / 60) -> tuple[float, float]:
    lo, hi = max(ta[0], tb[0]), min(ta[-1], tb[-1])
    grid = np.arange(lo, hi, step)
    if len(grid) < MIN_SAMPLES:
        return 0.0, -1.0
    a = np.interp(grid, ta, va)
    b = np.interp(grid, tb, vb)
    a = (a - a.mean()) / (a.std() + 1e-9)
    b = (b - b.mean()) / (b.std() + 1e-9)
    best, best_lag = -1.0, 0.0
    for k in range(-int(span / step), int(span / step) + 1):
        x, y = (a[k:], b[:len(b) - k]) if k >= 0 else (a[:len(a) + k], b[-k:])
        if len(x) < MIN_SAMPLES:
            continue
        value = float(np.mean(x * y))
        if value > best:
            best, best_lag = value, k * step
    return best_lag, best


def check_session(tracker_npz: Path) -> dict:
    session = tracker_npz.name[: -len(".tracker.npz")]
    result = {"session": session, "dataset": _dataset_of(tracker_npz)}
    glove_npz = tracker_npz.with_name(f"{session}.npz")
    with np.load(tracker_npz, allow_pickle=False) as z:
        times = np.asarray(z["tracker_0__pose__t_host_us"], dtype=np.float64) * 1e-6
        wrists = {
            side: np.asarray(z[f"tracker_0__pose.{side}_wrist.orientation"], dtype=np.float64)
            for side in ("left", "right")
        }
    result["tracker_hz"] = round(float(1 / np.median(np.diff(times))), 2)
    if not glove_npz.exists():
        return {**result, "status": "no_glove"}
    with np.load(glove_npz, allow_pickle=False) as z:
        gloves = {}
        for index in (0, 1):
            key = f"glove_{index}__hand_skeleton"
            if key in z.files:
                gloves[index] = (
                    np.asarray(z[key + "__t_host_us"], dtype=np.float64) * 1e-6,
                    np.asarray(z[key]),
                )
        result["tactile"] = any(k.endswith("__tactile") for k in z.files)
    if len(gloves) < 2:
        return {**result, "status": "no_glove"}
    result["glove_nodes"] = int(gloves[0][1].shape[1])
    result["glove_hz"] = round(float(1 / np.median(np.diff(gloves[0][0]))), 2)

    roots = {i: gloves[i][1][:, 0, 3:].astype(np.float64) for i in gloves}
    # Either glove being constant makes the pairing decision meaningless, so
    # test each one rather than requiring both to be flat.
    if any(r.std() < 1e-6 for r in roots.values()):
        # Wuji gloves store finger pose only; there is no world-frame root
        # rotation to correlate against the tracker.
        return {**result, "status": "unsupported",
                "reason": "手套根节点无全局朝向（wuji 格式），无法用运动互相关判定"}

    tracker_speed = {side: _angular_speed(wrists[side], times) for side in wrists}
    glove_speed = {i: _angular_speed(roots[i], gloves[i][0]) for i in gloves}
    if any(v[0] is None for v in list(tracker_speed.values()) + list(glove_speed.values())):
        return {**result, "status": "stream_broken"}
    if any(v[0].std() < 1e-9 for v in glove_speed.values()):
        return {**result, "status": "unsupported",
                "reason": "手套根节点角速度恒为零，无法用运动互相关判定"}
    result["glove_corrupt_frac"] = round(max(v[2] for v in glove_speed.values()), 4)
    result["tracker_corrupt_frac"] = round(max(v[2] for v in tracker_speed.values()), 4)

    pairs = {}
    for index in gloves:
        for side in ("left", "right"):
            lag, corr = _best_lag(
                tracker_speed[side][1], tracker_speed[side][0],
                glove_speed[index][1], glove_speed[index][0],
            )
            pairs[f"{index}:{side}"] = (round(lag, 4), round(corr, 3))
    result["pairs"] = pairs
    labelled = min(pairs["0:left"][1], pairs["1:right"][1])
    crossed = min(pairs["0:right"][1], pairs["1:left"][1])
    result["corr_labelled"] = round(labelled, 3)
    result["corr_crossed"] = round(crossed, 3)
    if max(labelled, crossed) < CONFIDENT:
        result["status"] = "weak"
    elif crossed > labelled:
        result["status"] = "swapped"
        result["lag_s"] = pairs["0:right"][0]
    else:
        result["status"] = "ok"
        result["lag_s"] = pairs["0:left"][0]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sync-root", type=Path,
                        required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    # Recordings are not all two levels deep (wuji/test/<session>/ exists),
    # and a fixed-depth glob silently skipped those sessions entirely.
    files = sorted(args.sync_root.rglob("*.tracker.npz"))
    if args.datasets:
        wanted = set(args.datasets)
        files = [f for f in files if f.parent.parent.name in wanted]
    if args.limit:
        files = files[: args.limit]

    rows = []
    for index, path in enumerate(files, 1):
        try:
            rows.append(check_session(path))
        except Exception as error:
            rows.append({"session": path.name[: -len(".tracker.npz")],
                         "dataset": path.parent.parent.name,
                         "status": "error", "reason": repr(error)})
        if index % 25 == 0 or index == len(files):
            print(f"[glove-sync {index}/{len(files)}] {path.parent.name}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    tally = {}
    for row in rows:
        tally.setdefault(row["dataset"], {}).setdefault(row["status"], 0)
        tally[row["dataset"]][row["status"]] += 1
    order = ["ok", "swapped", "weak", "unsupported", "no_glove", "stream_broken", "error"]
    print("\n%-16s %s" % ("dataset", "  ".join("%-12s" % s for s in order)))
    for dataset in sorted(tally):
        print("%-16s %s" % (dataset, "  ".join("%-12d" % tally[dataset].get(s, 0) for s in order)))
    print(f"\n-> {args.output}")


if __name__ == "__main__":
    main()
