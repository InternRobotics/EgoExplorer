#!/usr/bin/env python
"""Write left/right-corrected copies of recordings flagged by check_tracker_lr.py.

Originals are never touched.  Each corrected recording lands in a parallel tree
with the same layout, carries a ``tracker_lr_correction`` key inside the NPZ and
a sidecar JSON beside it, so anything downstream can tell at a glance that the
file was rewritten and on what evidence.  The session's other files (glove,
video, session.json) are symlinked through so the corrected tree stays a drop-in
replacement for the original input root.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

PAIR_ROLES = {"wrist": ("left_wrist", "right_wrist"), "ankle": ("left_ankle", "right_ankle")}
STREAMS = ("position", "orientation", "valid")
TOOL = "check_tracker_lr.py + fix_tracker_lr.py"


def correct_session(source: Path, destination: Path, row: dict, dry_run: bool) -> dict:
    pairs = row["swapped_pairs"]
    with np.load(source, allow_pickle=True) as archive:
        data = {key: archive[key] for key in archive.files}

    swapped_keys = []
    for pair in pairs:
        left, right = PAIR_ROLES[pair]
        for stream in STREAMS:
            left_key = f"tracker_0__pose.{left}.{stream}"
            right_key = f"tracker_0__pose.{right}.{stream}"
            if left_key not in data or right_key not in data:
                continue
            data[left_key], data[right_key] = data[right_key].copy(), data[left_key].copy()
            swapped_keys.extend([left_key, right_key])

    note = {
        "corrected": True,
        "tool": TOOL,
        "corrected_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "swapped_pairs": pairs,
        "swapped_keys": sorted(swapped_keys),
        "original_file": str(source),
        "evidence": {p: row["pairs"][p] for p in pairs},
        "note": ("左右 tracker 佩戴/标注颠倒，已对调该对的 position/orientation/valid。"
                 "判据：以头部 tracker 经 subject profile 反解出的解剖左轴，"
                 "投影 left-right 的横向分离量在全程为负。"),
    }
    data["tracker_lr_correction"] = np.array(json.dumps(note, ensure_ascii=False))

    if dry_run:
        return note
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **data)
    destination.with_name(destination.name.replace(".tracker.npz", ".tracker_lr_correction.json")
                          ).write_text(json.dumps(note, ensure_ascii=False, indent=2), encoding="utf-8")
    # Keep the rest of the session reachable so the corrected tree can stand in
    # for the original input root.
    for sibling in source.parent.iterdir():
        if sibling.name == source.name:
            continue
        link = destination.parent / sibling.name
        if not link.exists():
            try:
                link.symlink_to(sibling)
            except OSError:
                pass
    return note


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True, help="check_tracker_lr.py output")
    parser.add_argument("--sync-root", type=Path,
                        required=True)
    parser.add_argument("--output-root", type=Path,
                        required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    rows = [r for r in json.loads(args.report.read_text(encoding="utf-8")) if r["status"] == "swapped"]
    print(f"{len(rows)} sessions flagged as swapped")
    # Recordings sit at varying depths under the sync root, so locate each one
    # and mirror its relative path rather than assuming dataset/session/file.
    index = {path.name[: -len(".tracker.npz")]: path
             for path in args.sync_root.rglob("*.tracker.npz")}
    written, sessions = 0, []
    for row in rows:
        source = index.get(row["session"])
        if source is None or not source.exists():
            print(f"  !! missing source for {row['session']}")
            continue
        destination = args.output_root / source.parent.relative_to(args.sync_root) / source.name
        note = correct_session(source, destination, row, args.dry_run)
        written += 1
        sessions.append({"session": row["session"], "dataset": row["dataset"],
                         "swapped_pairs": row["swapped_pairs"]})
        print(f"  {row['session']:<44} swapped={'+'.join(row['swapped_pairs'])}")
    if not args.dry_run:
        args.output_root.mkdir(parents=True, exist_ok=True)
        (args.output_root / "corrections.json").write_text(
            json.dumps({"tool": TOOL, "written_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                        "sessions": sessions}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{'would write' if args.dry_run else 'wrote'} {written} corrected recordings -> {args.output_root}")


if __name__ == "__main__":
    main()
