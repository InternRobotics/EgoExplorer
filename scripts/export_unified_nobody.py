#!/usr/bin/env python
"""Export the 30 Hz streams for recordings that have no SMPL-X result.

A session whose projection diverged still has a usable video and, on the Wuji
rig, usable tactile.  Those are exactly what you need to see in order to judge
*why* it failed, so they get the same unified export and the same review page
as a successful one -- just without the mesh.

The failure reason is carried into the export so the page can state it rather
than leaving an empty pane.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_ego_gallery import failure_reason          # noqa: E402
from export_unified_30hz import DEFAULT_RATE, export_session  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sync-root", type=Path, required=True)
    parser.add_argument("--batch-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--rate", type=float, default=DEFAULT_RATE)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--of", type=int, default=1)
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args()

    projected = args.batch_root / "projected"
    proposals = args.batch_root / "proposals"
    todo = []
    stale = 0
    for tracker in sorted(args.sync_root.rglob("*.tracker.npz")):
        name = tracker.name[: -len(".tracker.npz")]
        if (projected / name / "smplx_profile_driven.npz").exists():
            # A session projected after its no-body export was written must not
            # keep the mesh-less page: the review builder skips anything that
            # already has one, so the stale artefacts are removed here.
            # Every shard walks the whole list before sharding, so all of them
            # reach this cleanup for the same name.  exists()-then-unlink is not
            # atomic and the losing shard died with FileNotFoundError, taking its
            # whole slice of the work with it.
            for path in (args.output_root / f"{name}.npz",
                         args.output_root / f"{name}.json"):
                try:
                    path.unlink()
                    stale += 1
                except FileNotFoundError:
                    pass
            continue
        todo.append((name, tracker))
    if stale:
        print(f"removed {stale} stale no-body exports (now projected)", flush=True)
    todo = [row for i, row in enumerate(todo) if i % args.of == args.shard]

    print(f"[nobody shard {args.shard}/{args.of}] {len(todo)} sessions", flush=True)
    ok = failed = skipped = 0
    for index, (name, tracker) in enumerate(todo, 1):
        target = args.output_root / f"{name}.npz"
        if args.skip_existing and target.exists():
            skipped += 1
            continue
        kind, why = failure_reason(name, proposals, {})
        try:
            report = export_session(
                projected / name, target, args.rate,
                source_override=tracker, session_name=name,
                failure={"kind": kind, "why": why})
            ok += 1
            if index % 25 == 0 or index == len(todo) or len(todo) <= 5:
                print(f"[{index}/{len(todo)}] {name}: {report['frames']} 帧 "
                      f"streams={'+'.join(report['streams'])} ({kind})", flush=True)
        except Exception as error:
            failed += 1
            print(f"[{index}/{len(todo)}] {name}: FAILED {error!r}"[:200], flush=True)
    print(f"\n完成 {ok} 条，失败 {failed} 条，跳过 {skipped} 条 -> {args.output_root}")


if __name__ == "__main__":
    main()
