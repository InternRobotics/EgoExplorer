#!/usr/bin/env python
"""Detect trackers whose pose drifts away over the course of a recording.

Drift is invisible frame-to-frame: every sample looks plausible, the tracker
reports valid, and the optimiser quietly absorbs part of the error into body
pose.  What gives it away is that the body is rigid over minutes — the distance
from the head tracker to a wrist or ankle oscillates with motion but has no
long-run trend.  A tracker that slides in its mount, or whose pose estimate
accumulates error, shows a steady trend in that distance instead.

Two independent signals are reported per tracker:

``trend_m``      slope of a least-squares line through the head-to-tracker
                 distance, times the session duration, i.e. how far the
                 distance moved end-to-end.
``residual_growth_mm``
                 how much the projection's own residual for that tracker grew
                 from the first fifth of the recording to the last, which
                 catches drift the geometry misses (e.g. both endpoints of the
                 pair drifting together).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

ROLES = ("camera", "left_wrist", "right_wrist", "left_ankle", "right_ankle")
LIMBS = ROLES[1:]
# Thresholds come from the population of 840 recordings rather than guesswork.
# Head-to-ankle distance is a tight invariant (p1 1.21 m, p99 1.43 m) because
# the segment is rigid and the foot stays on the floor; head-to-wrist is far
# looser (p50 0.43 m, p95 1.12 m) since it follows whatever the hands do, so
# only gross violations are called on the wrists.
ANATOMICAL_MAX_M = {"left_ankle": 1.60, "right_ankle": 1.60,
                    "left_wrist": 1.30, "right_wrist": 1.30}
# First-to-last quintile change; ankle p99 is 0.44 m, wrist p99 is 1.72 m.
CHANGE_BAD_M = {"left_ankle": .45, "right_ankle": .45,
                "left_wrist": .85, "right_wrist": .85}
CHANGE_WARN_M = {"left_ankle": .25, "right_ankle": .25,
                 "left_wrist": .55, "right_wrist": .55}
MIN_DURATION_S = 60.0
MIN_TRAVEL_M = .05
# The projection's own per-tracker residual is an independent witness: a
# tracker can sit at a perfectly plausible distance from the head and still
# disagree with any consistent body (a shifted mount, a rotated calibration, a
# slow positional bias).  Population medians are 1.5-7.5 mm with p75 under
# 12 mm, so these thresholds sit well clear of ordinary fitting error.
RESIDUAL_HIGH_MM = 15.0
RESIDUAL_BAD_MM = 60.0



def _dataset_of(tracker_npz: Path) -> str:
    """Dataset name = the first directory under the sync root, whatever the depth."""
    parts = tracker_npz.parts
    for index, part in enumerate(parts):
        if part in ("ego_data_sync", "ego_data_lr_fixed") and index + 1 < len(parts):
            return parts[index + 1]
    return tracker_npz.parent.parent.name


def _quintile_medians(values: np.ndarray) -> list[float]:
    step = max(len(values) // 5, 1)
    return [float(np.median(values[i * step:(i + 1) * step])) for i in range(5)]


def _monotone(values: list[float]) -> bool:
    ups = sum(b > a for a, b in zip(values, values[1:]))
    downs = sum(b < a for a, b in zip(values, values[1:]))
    return ups == 4 or downs == 4


def check_session(tracker_npz: Path, projected_dir: Path) -> dict:
    session = tracker_npz.name[: -len(".tracker.npz")]
    result = {"session": session, "dataset": _dataset_of(tracker_npz)}
    with np.load(tracker_npz, allow_pickle=False) as archive:
        if any(f"tracker_0__pose.{r}.position" not in archive for r in ROLES):
            return {**result, "status": "missing_roles"}
        positions = {r: np.asarray(archive[f"tracker_0__pose.{r}.position"], dtype=np.float64) for r in ROLES}
        times = np.asarray(archive["tracker_0__pose__t_host_us"], dtype=np.float64) * 1e-6
    usable = np.isfinite(times)
    for role in ROLES:
        usable &= np.isfinite(positions[role]).all(-1)
    if usable.sum() < 100:
        return {**result, "status": "too_short"}
    times = times[usable]
    duration = float(times[-1] - times[0])
    head = positions["camera"][usable]

    trackers, flagged = {}, []
    for role in LIMBS:
        offsets = positions[role][usable] - head
        distance = np.linalg.norm(offsets, axis=-1)
        quintiles = _quintile_medians(distance)
        change = quintiles[-1] - quintiles[0]
        travel = float(max(np.ptp(positions[role][usable][:, 0]), np.ptp(positions[role][usable][:, 2])))
        median = float(np.median(distance))
        entry = {"median_m": round(median, 3), "max_quintile_m": round(max(quintiles), 3),
                 "quintiles_m": [round(q, 3) for q in quintiles],
                 "change_m": round(change, 3), "travel_m": round(travel, 3),
                 "monotone": _monotone(quintiles)}
        worst = max(quintiles)
        if travel < MIN_TRAVEL_M:
            entry["verdict"] = "unrecorded"
        elif median > ANATOMICAL_MAX_M[role] or worst > ANATOMICAL_MAX_M[role] + .3:
            # No posture puts the tracker this far from the head.  Testing the
            # worst fifth as well as the median catches a tracker that only
            # comes off part-way through, which a whole-session median hides.
            entry["verdict"] = "off_body"
        elif abs(change) >= CHANGE_BAD_M[role]:
            # A gross change is a gross change however short the clip; only the
            # subtle drift threshold below needs a long enough recording to
            # separate it from ordinary posture change.
            entry["verdict"] = "drift" if entry["monotone"] else "jump"
        elif duration >= MIN_DURATION_S and abs(change) >= CHANGE_WARN_M[role] and entry["monotone"]:
            entry["verdict"] = "slight_drift"
        else:
            entry["verdict"] = "ok"
        trackers[role] = entry
        if entry["verdict"] in ("off_body", "drift", "jump", "slight_drift"):
            flagged.append(role)

    diagnostics = projected_dir / session / "tracker_diagnostics.npz"
    if diagnostics.exists():
        try:
            with np.load(diagnostics, allow_pickle=False) as archive:
                residual = np.linalg.norm(archive["position_residual_m"], axis=-1) * 1000
            fifth = max(len(residual) // 5, 1)
            growth = residual[-fifth:].mean(axis=0) - residual[:fifth].mean(axis=0)
            medians = np.median(residual, axis=0)
            for index, role in enumerate(ROLES):
                if role not in trackers:
                    continue
                entry = trackers[role]
                entry["residual_growth_mm"] = round(float(growth[index]), 2)
                entry["residual_median_mm"] = round(float(medians[index]), 2)
                if entry["verdict"] == "ok":
                    if medians[index] >= RESIDUAL_BAD_MM:
                        entry["verdict"] = "fit_bad"
                        flagged.append(role)
                    elif medians[index] >= RESIDUAL_HIGH_MM:
                        entry["verdict"] = "fit_high"
                        flagged.append(role)
        except Exception:
            pass

    severity = {t: trackers[t]["verdict"] for t in flagged}
    order = ("off_body", "drift", "jump", "fit_bad", "slight_drift", "fit_high")
    status = next((name for name in order if name in severity.values()), "ok")
    return {**result, "status": status, "flagged": sorted(flagged),
            "duration_s": round(duration, 1), "trackers": trackers}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sync-root", type=Path,
                        required=True)
    parser.add_argument("--projected", type=Path,
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
            rows.append(check_session(path, args.projected))
        except Exception as error:
            rows.append({"session": path.name[: -len(".tracker.npz")],
                         "dataset": path.parent.parent.name,
                         "status": "error", "reason": repr(error)})
        if index % 50 == 0 or index == len(files):
            print(f"[drift {index}/{len(files)}]", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    tally = {}
    for row in rows:
        tally.setdefault(row["dataset"], {}).setdefault(row["status"], 0)
        tally[row["dataset"]][row["status"]] += 1
    columns = sorted({k for v in tally.values() for k in v})
    print("\n%-16s %s" % ("dataset", "  ".join("%-14s" % c for c in columns)))
    for dataset in sorted(tally):
        print("%-16s %s" % (dataset, "  ".join("%-14d" % tally[dataset].get(c, 0) for c in columns)))
    worst = sorted((r for r in rows if r.get("flagged")),
                   key=lambda r: -max(abs(r["trackers"][t]["change_m"]) for t in r["flagged"]))[:12]
    print("\n最严重的 12 条:")
    for row in worst:
        detail = ", ".join(
            f"{t}={row['trackers'][t]['verdict']} 中位{row['trackers'][t]['median_m']:.2f}m "
            f"Δ{row['trackers'][t]['change_m']:+.2f}m" for t in row["flagged"])
        print("  %-40s %s" % (row["session"][:38], detail))
    print(f"\n-> {args.output}")


if __name__ == "__main__":
    main()
