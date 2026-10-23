#!/usr/bin/env python3
"""Decide, per session, which physical hand each glove stream belongs to — and
from that, whether the wrist trackers are swapped.

The glove skeleton is *chiral*: for a hand, the scalar triple product
``(index_mcp - wrist) x (pinky_mcp - wrist) . (thumb_tip - wrist)`` has one sign
for a left hand and the other for a right hand, in every pose.  Measured on this
corpus it is 100 % consistent within a stream (frac>0 is exactly 0.000 or 1.000),
so it identifies the hand *absolutely* — no motion correlation, no assumption
about the labels being right.

That gives two independent facts per session:

* ``slots_swapped`` — the ``slot`` labels in session.json contradict the
  chirality of the streams they name.
* ``wrist_trackers_swapped`` — the glove stream of a given physical hand moves
  with the *other* wrist tracker (angular-speed cross-correlation, resampled to
  30 Hz first: the 2026-09 MANUS recorder streams at ~220 Hz and raw differencing
  there is pure jitter).

`check_glove_sync.py` can only report "the labelled pairing correlates worse than
the crossed one", which conflates these two; the fix applied on 2026-09-19 guessed
between them with an XOR against the tracker check and got 3 of 70 backwards.
A mirrored glove stream fed to the wrong MANO side is exactly the "fingers bend
the wrong way" artefact.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

# wrist, thumb tip, index MCP, pinky MCP for each supported layout
LAYOUT = {21: {"thumb": 4, "index_mcp": 5, "pinky_mcp": 17},
          25: {"thumb": 4, "index_mcp": 6, "pinky_mcp": 21}}
RESAMPLE_HZ = 30.0
MAX_ANGULAR_SPEED = 20.0
MIN_CHIRALITY_MARGIN = 0.95   # fraction of frames agreeing with the majority sign


def _dataset_of(path: Path) -> str:
    parts = path.parts
    for i, p in enumerate(parts):
        if p in ("ego_data_sync", "ego_data_lr_fixed") and i + 1 < len(parts):
            return parts[i + 1]
    return path.parent.parent.name


def chirality(points: np.ndarray) -> tuple[str, float] | None:
    """(side, agreement) from the skeleton's handedness, or None if unsupported."""
    layout = LAYOUT.get(points.shape[1])
    if layout is None:
        return None
    p = np.asarray(points[:, :, :3], dtype=np.float64)
    good = np.isfinite(p).all(axis=(1, 2))
    if good.sum() < 20:
        return None
    p = p[good]
    wrist = p[:, 0]
    triple = np.sum(np.cross(p[:, layout["index_mcp"]] - wrist,
                             p[:, layout["pinky_mcp"]] - wrist) * (p[:, layout["thumb"]] - wrist), axis=1)
    frac = float((triple > 0).mean())
    side = "right" if frac > 0.5 else "left"
    return side, max(frac, 1 - frac)


def _angular_speed(quat: np.ndarray, times: np.ndarray):
    q = np.asarray(quat, dtype=np.float64)
    t = np.asarray(times, dtype=np.float64)
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    keep = np.isfinite(norm).ravel() & (norm.ravel() > 1e-6) & np.isfinite(t)
    if keep.sum() < 20:
        return None, None
    q, t = q[keep] / norm[keep], t[keep]
    order = np.argsort(t, kind="stable")
    q, t = q[order], t[order]
    t, first = np.unique(t, return_index=True)
    q = q[first]
    if len(t) < 20 or t[-1] - t[0] < 1.0:
        return None, None
    flip = np.cumsum(np.sum(q[:-1] * q[1:], axis=-1) < 0) % 2
    q[1:] *= np.where(flip, -1.0, 1.0)[:, None]
    grid = np.arange(t[0], t[-1], 1.0 / RESAMPLE_HZ)
    qi = np.stack([np.interp(grid, t, q[:, k]) for k in range(q.shape[1])], axis=-1)
    qi /= np.maximum(np.linalg.norm(qi, axis=-1, keepdims=True), 1e-9)
    dot = np.abs(np.sum(qi[:-1] * qi[1:], axis=-1)).clip(0, 1)
    speed = 2 * np.arccos(dot) / np.maximum(np.diff(grid), 1e-6)
    return np.clip(speed, 0, MAX_ANGULAR_SPEED), .5 * (grid[:-1] + grid[1:])


def _best_corr(ta, va, tb, vb, span=2.0, step=1 / 30) -> tuple[float, float]:
    lo, hi = max(ta[0], tb[0]), min(ta[-1], tb[-1])
    grid = np.arange(lo, hi, step)
    if len(grid) < 200:
        return 0.0, -1.0
    a = np.interp(grid, ta, va); b = np.interp(grid, tb, vb)
    a = (a - a.mean()) / (a.std() + 1e-9); b = (b - b.mean()) / (b.std() + 1e-9)
    best, lag = -1.0, 0.0
    for k in range(-int(span / step), int(span / step) + 1):
        x, y = (a[k:], b[:len(b) - k]) if k >= 0 else (a[:len(a) + k], b[-k:])
        if len(x) < 200:
            continue
        v = float(np.mean(x * y))
        if v > best:
            best, lag = v, k * step
    return lag, best


def check_session(tracker_npz: Path) -> dict:
    sid = tracker_npz.name[: -len(".tracker.npz")]
    out = {"session": sid, "dataset": _dataset_of(tracker_npz)}
    glove_npz = tracker_npz.with_name(f"{sid}.npz")
    meta_path = tracker_npz.with_name(f"{sid}.session.json")
    if not glove_npz.exists():
        return {**out, "status": "no_glove"}
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
    slots = {g.get("device_name"): g.get("slot") for g in meta.get("gloves", [])}
    out["slots"] = slots
    out["slot_corrected"] = bool(meta.get("glove_lr_correction"))

    with np.load(glove_npz, allow_pickle=False) as z:
        streams = {}
        for dev in ("glove_0", "glove_1"):
            key = f"{dev}__hand_skeleton"
            if key not in z.files:
                continue
            times_key = f"{dev}__hand_skeleton__t_host_us"
            streams[dev] = (np.asarray(z[key]),
                            np.asarray(z[times_key], dtype=np.float64) * 1e-6 if times_key in z.files else None)
    if len(streams) < 2:
        return {**out, "status": "no_glove", "streams": sorted(streams)}

    phys, agree = {}, {}
    for dev, (pts, _) in streams.items():
        c = chirality(pts)
        if c is None:
            return {**out, "status": "unsupported_layout", "nodes": int(pts.shape[1])}
        phys[dev], agree[dev] = c
    out["physical_side"] = phys
    out["chirality_agreement"] = {k: round(v, 3) for k, v in agree.items()}
    out["nodes"] = int(streams["glove_0"][0].shape[1])
    if min(agree.values()) < MIN_CHIRALITY_MARGIN or len(set(phys.values())) != 2:
        return {**out, "status": "chirality_unclear"}
    out["slots_swapped"] = bool(any(slots.get(d) is not None and slots[d] != phys[d] for d in phys))

    # Which wrist tracker moves with which physical hand?
    with np.load(tracker_npz, allow_pickle=False) as z:
        tkey = "tracker_0__pose__t_host_us"
        if tkey not in z.files:
            return {**out, "status": "no_tracker_clock"}
        tt = np.asarray(z[tkey], dtype=np.float64) * 1e-6
        wrists = {}
        for side in ("left", "right"):
            key = f"tracker_0__pose.{side}_wrist.orientation"
            if key in z.files:
                wrists[side] = np.asarray(z[key], dtype=np.float64)[:, [1, 2, 3, 0]]
    if len(wrists) < 2:
        return {**out, "status": "missing_wrist_tracker", "have": sorted(wrists)}

    tracker_speed = {s: _angular_speed(q, tt) for s, q in wrists.items()}
    glove_speed = {}
    for dev, (pts, times) in streams.items():
        if times is None:
            return {**out, "status": "no_glove_clock"}
        q = np.asarray(pts[:, 0, 3:], dtype=np.float64)
        if np.std(q) < 1e-6:
            return {**out, "status": "no_glove_orientation",
                    "note": "wuji 手套根节点无全局朝向，无法用运动相关定位腕部 tracker"}
        glove_speed[dev] = _angular_speed(q, times)
    if any(v[0] is None for v in list(tracker_speed.values()) + list(glove_speed.values())):
        return {**out, "status": "stream_broken"}

    corr = {}
    for dev, (gs, gt) in glove_speed.items():
        for side, (ts_, tt_) in tracker_speed.items():
            lag, value = _best_corr(tt_, ts_, gt, gs)
            corr[f"{dev}->{side}"] = (round(lag, 3), round(value, 3))
    out["corr"] = corr
    # For each physical hand, the tracker it actually follows.
    follows, strength = {}, {}
    for dev in glove_speed:
        pairs = {s: corr[f"{dev}->{s}"][1] for s in tracker_speed}
        best = max(pairs, key=pairs.get)
        follows[phys[dev]] = best
        strength[phys[dev]] = round(pairs[best] - min(pairs.values()), 3)
    out["hand_follows_tracker"] = follows
    out["follow_margin"] = strength
    if min(strength.values()) < 0.2 or len(set(follows.values())) != 2:
        return {**out, "status": "correlation_weak"}
    out["wrist_trackers_swapped"] = bool(any(hand != tracker for hand, tracker in follows.items()))
    problems = [k for k in ("slots_swapped", "wrist_trackers_swapped") if out.get(k)]
    return {**out, "status": "+".join(problems) if problems else "ok"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sync-root", type=Path, required=True)
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    files = sorted(args.sync_root.rglob("*.tracker.npz"))
    if args.datasets:
        wanted = set(args.datasets)
        files = [f for f in files if _dataset_of(f) in wanted]
    if args.limit:
        files = files[: args.limit]

    rows = []
    for i, path in enumerate(files, 1):
        try:
            rows.append(check_session(path))
        except Exception as exc:
            rows.append({"session": path.name[: -len(".tracker.npz")], "dataset": _dataset_of(path),
                         "status": "error", "reason": repr(exc)})
        if i % 50 == 0 or i == len(files):
            print(f"[hand-identity {i}/{len(files)}]", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")

    tally: dict[str, dict[str, int]] = {}
    for row in rows:
        tally.setdefault(row["dataset"], {}).setdefault(row["status"], 0)
        tally[row["dataset"]][row["status"]] += 1
    cols = sorted({k for v in tally.values() for k in v})
    print("\n%-12s %s" % ("dataset", "  ".join("%-22s" % c for c in cols)))
    for ds in sorted(tally):
        print("%-12s %s" % (ds, "  ".join("%-22d" % tally[ds].get(c, 0) for c in cols)))
    print(f"\n-> {args.output}")


if __name__ == "__main__":
    main()
