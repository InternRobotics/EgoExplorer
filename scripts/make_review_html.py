#!/usr/bin/env python3
"""Fold a review payload into the viewer template to make one standalone page.

The template keeps four `/*__NAME__*/` slots so it stays valid JavaScript on its
own and can be opened directly while editing.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("payload", type=Path)
    ap.add_argument("--template", type=Path, default=HERE / "review_viewer_template.html")
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--slug", default=None, help="basename of the mesh/video assets")
    ap.add_argument("--faces", default="smplx_faces.u32")
    ap.add_argument("--video", action="store_true",
                    help="a <slug>.360.mp4 sits beside the page; without it the "
                         "video pane says so instead of showing a black box")
    args = ap.parse_args()

    payload = json.loads(args.payload.read_text())
    slug = args.slug or re.sub(r"^(session_\d+_\d+).*", r"\1", args.payload.name)
    html = args.template.read_text()
    for slot, value in (("__PAYLOAD__", json.dumps(payload, separators=(",", ":"))),
                        ("__MESH__", json.dumps(f"{slug}.mesh.i8c.gz")),
                        ("__FACES__", json.dumps(args.faces)),
                        ("__VIDEO__", json.dumps(
                            f"{slug}.360.mp4" if args.video else ""))):
        pattern = rf"/\*{slot}\*/[^,;]*"
        html, n = re.subn(pattern, lambda _m, v=value: v, html, count=1)
        assert n == 1, f"slot {slot} not found in template"

    out = args.output or args.payload.with_name(f"{slug}.review.html")
    out.write_text(html)
    print(f"{out} ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
