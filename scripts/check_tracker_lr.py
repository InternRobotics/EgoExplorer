#!/usr/bin/env python
"""Detect left/right swapped trackers in ego_data recordings.

A tracker physically strapped to the wrong limb is invisible to the pipeline —
it fits a body that holds its hands and feet on the wrong sides and reports a
large but not obviously fatal residual.  The subject profile pins the geometry
down well enough to test it directly: the head mount maps the camera tracker's
world rotation into the head joint frame, whose +x column is the subject's own
left (profile: ``calibration_reference_x_left_z_forward``).  Projecting
``left - right`` for each tracker pair onto that axis gives a signed lateral
separation that stays positive for a whole recording unless the pair is
reversed.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

PAIRS = {"wrist": ("left_wrist", "right_wrist"), "ankle": ("left_ankle", "right_ankle")}
ROLES = ("camera", "left_wrist", "right_wrist", "left_ankle", "right_ankle")
# A limb crosses the midline often enough that a clean verdict needs a wide
# margin; anything between these fractions is reported rather than guessed at.
SURE_OK = .80
SURE_SWAPPED = .20
MIN_SEPARATION_M = .05
# A tracker that never moves horizontally was not actually worn for this
# recording (some resumed sessions carry head and ankles only).  Its constant
# pose still yields a constant left-minus-right vector, which reads as a clean
# verdict, so the pair must be reported as unrecorded instead of swapped.
MIN_TRAVEL_M = .05



def _dataset_of(tracker_npz: Path) -> str:
    """Dataset name = the first directory under the sync root, whatever the depth."""
    parts = tracker_npz.parts
    for index, part in enumerate(parts):
        if part in ("ego_data_sync", "ego_data_lr_fixed") and index + 1 < len(parts):
            return parts[index + 1]
    return tracker_npz.parent.parent.name


def _matrices_from_wxyz(quaternions: np.ndarray) -> np.ndarray:
    q = np.asarray(quaternions, dtype=np.float64)
    return Rotation.from_quat(np.stack([q[:, 1], q[:, 2], q[:, 3], q[:, 0]], axis=-1)).as_matrix()


def check_session(tracker_npz: Path, head_rotation: np.ndarray) -> dict:
    session = tracker_npz.name[: -len(".tracker.npz")]
    result = {"session": session, "dataset": _dataset_of(tracker_npz)}
    with np.load(tracker_npz, allow_pickle=False) as archive:
        missing = [r for r in ROLES if f"tracker_0__pose.{r}.position" not in archive]
        if missing:
            return {**result, "status": "missing_roles", "missing": missing}
        positions = {r: np.asarray(archive[f"tracker_0__pose.{r}.position"], dtype=np.float64) for r in ROLES}
        camera = np.asarray(archive["tracker_0__pose.camera.orientation"], dtype=np.float64)
    usable = np.isfinite(camera).all(-1) & (np.linalg.norm(camera, axis=-1) > 1e-8)
    for role in ROLES:
        usable &= np.isfinite(positions[role]).all(-1)
    if usable.sum() < 50:
        return {**result, "status": "too_few_frames", "frames": int(usable.sum())}

    head_joint = _matrices_from_wxyz(camera[usable]) @ head_rotation.T
    left_axis = head_joint[:, :, 0][:, [0, 2]]
    left_axis /= np.maximum(np.linalg.norm(left_axis, axis=-1, keepdims=True), 1e-9)

    travel = {r: float(max(np.ptp(positions[r][usable][:, 0]), np.ptp(positions[r][usable][:, 2])))
              for r in ROLES}
    verdicts, swapped, frozen = {}, [], [r for r, t in travel.items() if t < MIN_TRAVEL_M]
    for pair, (left, right) in PAIRS.items():
        if travel[left] < MIN_TRAVEL_M or travel[right] < MIN_TRAVEL_M:
            verdicts[pair] = {"verdict": "unrecorded",
                              "travel_m": {left: round(travel[left], 4), right: round(travel[right], 4)},
                              "frames": int(usable.sum())}
            continue
        delta = (positions[left][usable] - positions[right][usable])[:, [0, 2]]
        signed = np.sum(delta * left_axis, axis=-1)
        separation = np.linalg.norm(delta, axis=-1)
        share = float((signed > 0).mean())
        median = float(np.median(signed))
        # The limbs cross the midline constantly during work, so the frames
        # where they are furthest apart carry the clearest evidence — that is
        # also where the subject's calibration pose lands.
        widest = separation >= np.percentile(separation, 90)
        widest_median = float(np.median(signed[widest]))
        if abs(median) < MIN_SEPARATION_M and abs(widest_median) < MIN_SEPARATION_M:
            verdict = "ambiguous"
        elif share >= SURE_OK or (share >= .5 and widest_median > MIN_SEPARATION_M):
            verdict = "ok"
        elif share <= SURE_SWAPPED or (share < .5 and widest_median < -MIN_SEPARATION_M):
            verdict = "swapped"
        else:
            verdict = "ambiguous"
        verdicts[pair] = {"verdict": verdict, "median_m": round(median, 4),
                          "widest_median_m": round(widest_median, 4),
                          "max_separation_m": round(float(separation.max()), 4),
                          "positive_share": round(share, 4), "frames": int(usable.sum())}
        if verdict == "swapped":
            swapped.append(pair)
    status = ("swapped" if swapped else
              "unrecorded" if any(v["verdict"] == "unrecorded" for v in verdicts.values()) else
              "ambiguous" if any(v["verdict"] == "ambiguous" for v in verdicts.values()) else "ok")
    return {**result, "status": status, "swapped_pairs": swapped,
            "frozen_trackers": frozen, "pairs": verdicts}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sync-root", type=Path,
                        required=True)
    parser.add_argument("--profile", type=Path,
                        required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    profile = json.loads(args.profile.read_text(encoding="utf-8"))
    head_rotation = Rotation.from_quat(
        profile["tracker_mounts"]["head"]["rotation_joint_tracker_xyzw"]
    ).as_matrix()

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
            rows.append(check_session(path, head_rotation))
        except Exception as error:
            rows.append({"session": path.name[: -len(".tracker.npz")],
                         "dataset": path.parent.parent.name,
                         "status": "error", "reason": repr(error)})
        if index % 50 == 0 or index == len(files):
            print(f"[tracker-lr {index}/{len(files)}]", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")

    tally = {}
    for row in rows:
        key = row["status"]
        if key == "swapped":
            key = "swapped:" + "+".join(row["swapped_pairs"])
        tally.setdefault(row["dataset"], {}).setdefault(key, 0)
        tally[row["dataset"]][key] += 1
    columns = sorted({k for v in tally.values() for k in v})
    print("\n%-16s %s" % ("dataset", "  ".join("%-18s" % c for c in columns)))
    for dataset in sorted(tally):
        print("%-16s %s" % (dataset, "  ".join("%-18d" % tally[dataset].get(c, 0) for c in columns)))
    print(f"\n-> {args.output}")


if __name__ == "__main__":
    main()
