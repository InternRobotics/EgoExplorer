#!/usr/bin/env python3
"""Write glove-side-corrected copies of recordings flagged by check_glove_sync.py.

Companion of fix_tracker_lr.py, same contract: originals untouched, corrected
session lands in a parallel tree with the same relative layout, other files are
symlinked through so the tree is a drop-in input root.  Only ``session.json``
is rewritten: the two gloves' ``slot`` labels are exchanged and a
``glove_lr_correction`` note records the evidence.  Every downstream consumer
(fuse-manus-batch, export_unified_30hz, viewers) takes the glove side from that
file, so re-projecting the session from the corrected tree fixes them all.

The glove test is relative to the *tracker* labels (glove root angular speed vs the
wrist tracker of the same side), so when check_tracker_lr.py says the wrist trackers
themselves are swapped the glove verdict inverts: a glove that "disagrees" with a
swapped tracker is on the correct hand, and a glove that "agrees" with it is swapped
too.  Pass --tracker-report to apply that XOR; without it the raw verdict is used.

If fix_tracker_lr.py already wrote a corrected tracker into the same tree it is
kept; a session.json symlink left by that tool is replaced by the rewritten file.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--report", type=Path, required=True, help="check_glove_sync.py output")
    ap.add_argument("--sync-root", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--tracker-report", type=Path, default=None, help="check_tracker_lr.py output; wrist-swapped sessions invert the glove verdict")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    wrist_swapped = set()
    if args.tracker_report is not None:
        for r in json.loads(args.tracker_report.read_text(encoding="utf-8")):
            if r.get("status") == "swapped" and "wrist" in r.get("swapped_pairs", []):
                wrist_swapped.add(r["session"])
    all_rows = json.loads(args.report.read_text(encoding="utf-8"))
    rows = []
    for r in all_rows:
        if r.get("status") not in ("swapped", "ok"):
            continue
        flagged = r["status"] == "swapped"
        if r["session"] in wrist_swapped:
            flagged = not flagged
            r = {**r, "tracker_wrist_swapped": True}
        if flagged:
            rows.append(r)
    print(f"{len(rows)} sessions need a glove slot swap "
          f"({sum(1 for r in all_rows if r.get('status') == 'swapped')} raw glove-swapped, "
          f"{len(wrist_swapped)} wrist-tracker-swapped inverting the verdict)")
    index = {p.name[: -len(".tracker.npz")]: p for p in args.sync_root.rglob("*.tracker.npz")}
    written, sessions = 0, []
    for row in rows:
        sid = row["session"]
        tracker = index.get(sid)
        if tracker is None:
            print(f"  !! missing source for {sid}")
            continue
        src_dir = tracker.parent
        session_json = src_dir / f"{sid}.session.json"
        if not session_json.exists():
            print(f"  !! {sid}: no session.json")
            continue
        meta = json.loads(session_json.read_text(encoding="utf-8"))
        gloves = meta.get("gloves") or []
        if len(gloves) != 2 or {g.get("slot") for g in gloves} != {"left", "right"}:
            print(f"  !! {sid}: unexpected glove slots {[g.get('slot') for g in gloves]}")
            continue
        for g in gloves:
            g["slot"] = "right" if g["slot"] == "left" else "left"
        note = {
            "corrected": True, "tool": "check_glove_sync.py + fix_glove_lr.py",
            "corrected_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "original_file": str(session_json),
            "evidence": {"pairs": row.get("pairs"), "corr_labelled": row.get("corr_labelled"),
                         "corr_crossed": row.get("corr_crossed"), "glove_check_status": row.get("status"),
                         "tracker_wrist_swapped": bool(row.get("tracker_wrist_swapped"))},
            "note": ("手套 slot 左右标反，已对调 session.json 里两只手套的 slot，数据文件本身未改动。判据：手套根节点角速度与"
                     "手腕 tracker 的互相关；若该 session 的腕部 tracker 本身被判左右反，则手套判定取反（手套与被换错的 tracker "
                     "同步 = 手套也戴反；与之相反 = 手套是对的）。"),
        }
        meta["glove_lr_correction"] = note
        dest_dir = args.output_root / src_dir.relative_to(args.sync_root)
        sessions.append({"session": sid, "dataset": row.get("dataset"), "corr_labelled": row.get("corr_labelled"),
                         "corr_crossed": row.get("corr_crossed"), "glove_check_status": row.get("status"),
                         "tracker_wrist_swapped": bool(row.get("tracker_wrist_swapped"))})
        print(f"  {sid:<44} glove-check {row.get('status'):7s} wrist-tracker-swapped {bool(row.get('tracker_wrist_swapped'))!s:5s} "
              f"labelled {row.get('corr_labelled')} crossed {row.get('corr_crossed')}")
        if args.dry_run:
            written += 1
            continue
        dest_dir.mkdir(parents=True, exist_ok=True)
        target = dest_dir / session_json.name
        if target.is_symlink() or target.exists():
            target.unlink()
        target.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        (dest_dir / f"{sid}.glove_lr_correction.json").write_text(json.dumps(note, ensure_ascii=False, indent=2), encoding="utf-8")
        for sibling in src_dir.iterdir():
            if sibling.name == session_json.name:
                continue
            link = dest_dir / sibling.name
            if link.exists() or link.is_symlink():
                continue  # e.g. a corrected tracker written by fix_tracker_lr.py
            link.symlink_to(os.path.realpath(sibling))
        written += 1
    if not args.dry_run:
        args.output_root.mkdir(parents=True, exist_ok=True)
        (args.output_root / "glove_corrections.json").write_text(
            json.dumps({"tool": "check_glove_sync.py + fix_glove_lr.py", "written_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                        "sessions": sessions}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{'would write' if args.dry_run else 'wrote'} {written} corrected sessions -> {args.output_root}")


if __name__ == "__main__":
    main()
