#!/usr/bin/env python3
"""Judge tracker left/right from the recording's initial standing stance.

Every recording starts the same way: the subject stands still, legs uncrossed,
arms hanging naturally forward.  That pose pins both pairs down far better than
statistics over the whole recording, where limbs cross the midline constantly —
`check_tracker_lr.py` therefore falls back to "the widest 10 % of frames", which
is noisy for the ankles because feet stay close together all recording long.

For each session this reports, over the still window at the start:

* `wrist` / `ankle` verdict — sign of (left − right) projected on the subject's
  anatomical left axis (head tracker rotated into the head joint frame by the
  subject profile, exactly as check_tracker_lr does).
* `hands_forward` — are both wrists ahead of the head along the head's forward
  axis, the sanity check that the start pose really is the expected one.
* per-tracker `travel_m` over the whole recording and a `dead` list: trackers
  whose horizontal travel never exceeds a few centimetres, i.e. not worn or
  failed, whose constant pose otherwise produces a confident-looking verdict.

Output is one JSON row per session, same shape as check_tracker_lr.py so the
two can be compared directly.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ROLES = ("camera", "left_wrist", "right_wrist", "left_ankle", "right_ankle")
PAIRS = {"wrist": ("left_wrist", "right_wrist"), "ankle": ("left_ankle", "right_ankle")}
# A pair this far apart along the left axis is unambiguous; anything smaller is
# reported as "ambiguous" rather than guessed at.
MARGIN_M = 0.04
# Horizontal travel below this over the whole recording means the tracker was
# not actually moving with a limb.
DEAD_TRAVEL_M = 0.05
# Plausible human geometry in the stance window.  A tracker that has drifted or
# flown off produces separations of metres, which otherwise reads as a very
# confident "swapped" -- the sign is meaningless once the position is garbage.
WRIST_GAP_M = (0.05, 1.10)
ANKLE_GAP_M = (0.02, 0.70)
# Measured on this corpus: head tracker sits 1.29-1.51 m above the ankle trackers
# (p1-p99 across all three datasets), so the window is set around that, not around
# an anatomical standing height -- both mounts carry their own offset.
HEAD_ANKLE_M = (0.95, 1.70)
HANDS_FORWARD_M = (-0.40, 1.00)
STILL_SPEED = 0.15       # m/s, counts as "standing still"
WINDOW_S = 1.5           # length of the stance window
SEARCH_S = 20.0          # look for the window inside this much of the start


def _dataset_of(path: Path) -> str:
    parts = path.parts
    for i, p in enumerate(parts):
        if p in ("ego_data_sync", "ego_data_lr_fixed") and i + 1 < len(parts):
            return parts[i + 1]
    return path.parent.parent.name


def _wxyz_to_matrix(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    return Rotation.from_quat(np.stack([q[:, 1], q[:, 2], q[:, 3], q[:, 0]], axis=-1)).as_matrix()


def check_session(tracker_npz: Path, head_rotation: np.ndarray) -> dict:
    sid = tracker_npz.name[: -len(".tracker.npz")]
    out = {"session": sid, "dataset": _dataset_of(tracker_npz)}
    with np.load(tracker_npz, allow_pickle=False) as z:
        missing = [r for r in ROLES if f"tracker_0__pose.{r}.position" not in z]
        if missing:
            return {**out, "status": "missing_roles", "missing": missing}
        pos = {r: np.asarray(z[f"tracker_0__pose.{r}.position"], dtype=np.float64) for r in ROLES}
        cam = np.asarray(z["tracker_0__pose.camera.orientation"], dtype=np.float64)
        for key in ("tracker_0__pose__t_host_us", "tracker_0__pose__t_us"):
            if key in z:
                t = np.asarray(z[key], dtype=np.float64) * 1e-6
                break
        else:
            return {**out, "status": "no_timestamps"}
    ok = np.isfinite(cam).all(-1) & (np.linalg.norm(cam, axis=-1) > 1e-8) & np.isfinite(t)
    for r in ROLES:
        ok &= np.isfinite(pos[r]).all(-1)
    if ok.sum() < 50:
        return {**out, "status": "too_few_frames", "frames": int(ok.sum())}
    idx = np.flatnonzero(ok)
    t = t[idx] - t[idx][0]
    pos = {r: v[idx] for r, v in pos.items()}
    cam = cam[idx]

    travel = {r: float(max(np.ptp(v[:, 0]), np.ptp(v[:, 2]))) for r, v in pos.items()}
    out["travel_m"] = {r: round(v, 3) for r, v in travel.items()}
    out["dead"] = sorted(r for r, v in travel.items() if v < DEAD_TRAVEL_M)

    # Stance window: the calmest WINDOW_S inside the first SEARCH_S, measured on
    # the sum of all five tracker speeds.
    speed = np.zeros(len(t))
    for r in ROLES:
        d = np.linalg.norm(np.gradient(pos[r], axis=0), axis=1) / np.maximum(np.gradient(t), 1e-6)
        speed += np.nan_to_num(d)
    search = t <= min(SEARCH_S, t[-1])
    best, best_start = None, 0.0
    starts = np.arange(0.0, max(0.1, min(SEARCH_S, t[-1]) - WINDOW_S), 0.25)
    for s in starts:
        w = (t >= s) & (t < s + WINDOW_S)
        if w.sum() < 5:
            continue
        m = float(np.mean(speed[w]))
        if best is None or m < best:
            best, best_start = m, float(s)
    if best is None:
        w = search
        out["stance"] = {"start_s": 0.0, "mean_speed": None, "frames": int(w.sum()), "fallback": True}
    else:
        w = (t >= best_start) & (t < best_start + WINDOW_S)
        out["stance"] = {"start_s": round(best_start, 2), "mean_speed": round(best / 5, 3),
                         "frames": int(w.sum()), "still": bool(best / 5 < STILL_SPEED)}

    head = _wxyz_to_matrix(cam[w]) @ head_rotation.T
    left_axis = head[:, :, 0][:, [0, 2]]
    left_axis /= np.maximum(np.linalg.norm(left_axis, axis=-1, keepdims=True), 1e-9)
    fwd_axis = head[:, :, 2][:, [0, 2]]
    fwd_axis /= np.maximum(np.linalg.norm(fwd_axis, axis=-1, keepdims=True), 1e-9)

    gap = {p: float(np.median(np.linalg.norm((pos[l][w] - pos[r][w])[:, [0, 2]], axis=-1)))
           for p, (l, r) in PAIRS.items()}
    head_ankle = float(np.median(pos["camera"][w][:, 1] - .5 * (pos["left_ankle"][w][:, 1] + pos["right_ankle"][w][:, 1])))
    plausible = {
        "wrist_gap": WRIST_GAP_M[0] <= gap["wrist"] <= WRIST_GAP_M[1],
        "ankle_gap": ANKLE_GAP_M[0] <= gap["ankle"] <= ANKLE_GAP_M[1],
        "head_above_ankles": HEAD_ANKLE_M[0] <= head_ankle <= HEAD_ANKLE_M[1],
    }
    out["geometry"] = {"wrist_gap_m": round(gap["wrist"], 3), "ankle_gap_m": round(gap["ankle"], 3),
                       "head_above_ankles_m": round(head_ankle, 3), **plausible}

    verdicts = {}
    for pair, (l, r) in PAIRS.items():
        if travel[l] < DEAD_TRAVEL_M or travel[r] < DEAD_TRAVEL_M:
            verdicts[pair] = {"verdict": "unrecorded"}
            continue
        delta = (pos[l][w] - pos[r][w])[:, [0, 2]]
        signed = np.sum(delta * left_axis, axis=-1)
        med = float(np.median(signed))
        # Only judge the side when the pose is anatomically possible.
        if not (plausible[f"{pair}_gap"] and plausible["head_above_ankles"]):
            verdict = "implausible"
        elif med > MARGIN_M:
            verdict = "ok"
        elif med < -MARGIN_M:
            verdict = "swapped"
        else:
            verdict = "ambiguous"
        verdicts[pair] = {
            "verdict": verdict,
            "separation_m": round(med, 4),
            "separation_abs_m": round(float(np.median(np.linalg.norm(delta, axis=-1))), 4),
            "agreement": round(float((np.sign(signed) == np.sign(med)).mean()), 3),
        }
    out["pairs"] = verdicts

    # Start pose sanity: both hands ahead of the head, hands below head height.
    fwd = {}
    for r in ("left_wrist", "right_wrist"):
        d = (pos[r][w] - pos["camera"][w])[:, [0, 2]]
        fwd[r] = round(float(np.median(np.sum(d * fwd_axis, axis=-1))), 3)
    out["hands_forward_m"] = fwd
    out["hands_ahead"] = bool(HANDS_FORWARD_M[0] < min(fwd.values()) and max(fwd.values()) < HANDS_FORWARD_M[1])
    out["hands_below_head_m"] = {r: round(float(np.median(pos["camera"][w][:, 1] - pos[r][w][:, 1])), 3)
                                 for r in ("left_wrist", "right_wrist")}
    out["ankle_gap_m"] = round(float(np.median(np.linalg.norm(
        (pos["left_ankle"][w] - pos["right_ankle"][w])[:, [0, 2]], axis=-1))), 3)

    swapped = [p for p, v in verdicts.items() if v["verdict"] == "swapped"]
    implausible = [p for p, v in verdicts.items() if v["verdict"] == "implausible"]
    out["implausible_pairs"] = implausible
    if out["dead"]:
        status = "dead_tracker"
    elif implausible:
        status = "implausible_geometry"
    elif swapped:
        status = "swapped"
    elif any(v["verdict"] == "ambiguous" for v in verdicts.values()):
        status = "ambiguous"
    else:
        status = "ok"
    return {**out, "status": status, "swapped_pairs": swapped}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sync-root", type=Path, required=True)
    ap.add_argument("--profile", type=Path, required=True)
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    profile = json.loads(args.profile.read_text(encoding="utf-8"))
    head_rotation = Rotation.from_quat(
        profile["tracker_mounts"]["head"]["rotation_joint_tracker_xyzw"]).as_matrix()

    files = sorted(args.sync_root.rglob("*.tracker.npz"))
    if args.datasets:
        wanted = set(args.datasets)
        files = [f for f in files if _dataset_of(f) in wanted]
    if args.limit:
        files = files[: args.limit]

    rows = []
    for i, path in enumerate(files, 1):
        try:
            rows.append(check_session(path, head_rotation))
        except Exception as exc:  # a broken recording must not stop the sweep
            rows.append({"session": path.name[: -len(".tracker.npz")], "dataset": _dataset_of(path),
                         "status": "error", "reason": repr(exc)})
        if i % 100 == 0 or i == len(files):
            print(f"[stance {i}/{len(files)}]", flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")

    tally: dict[str, dict[str, int]] = {}
    for row in rows:
        key = row["status"]
        if key == "swapped":
            key = "swapped:" + "+".join(row["swapped_pairs"])
        tally.setdefault(row["dataset"], {}).setdefault(key, 0)
        tally[row["dataset"]][key] += 1
    cols = sorted({k for v in tally.values() for k in v})
    print("\n%-12s %s" % ("dataset", "  ".join("%-16s" % c for c in cols)))
    for ds in sorted(tally):
        print("%-12s %s" % (ds, "  ".join("%-16d" % tally[ds].get(c, 0) for c in cols)))
    print(f"\n-> {args.output}")


if __name__ == "__main__":
    main()
