#!/usr/bin/env python3
"""Calibrate Wuji gloves and move the calibration between machines.

    python scripts/wuji_calibration_tool.py users
    python scripts/wuji_calibration_tool.py create-user "<name>"
    python scripts/wuji_calibration_tool.py calibrate --user <user_id>
    python scripts/wuji_calibration_tool.py export --user <user_id> [--out wuji_calibration/calibration_bundle.zip]
    python scripts/wuji_calibration_tool.py preview [bundle.zip]
    python scripts/wuji_calibration_tool.py import  [bundle.zip]

Run from capture/ on the Linux machine the gloves are connected to (after
scripts/setup_wuji_gloves.sh), with Wuji Studio closed.  `calibrate` runs the SDK's
hand-model calibration for every connected glove under the given SDK user; follow the
poses the SDK asks for.  `export` writes that user's calibration as the bundle the
recorder loads at startup.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wuji_calibration import DEFAULT_BUNDLE, export_bundle, import_and_select_bundle, preview_bundle  # noqa: E402


def _manager():
    try:
        from wuji_sdk import SdkManager
    except ImportError as ex:
        raise SystemExit(f"wuji_sdk is not installed ({ex}); pip install -r requirements.txt") from ex
    return SdkManager.instance()


def _print(obj):
    print(json.dumps(obj, ensure_ascii=False, indent=2, default=str))


def cmd_users(args):
    manager = _manager()
    current = manager.current_user()["user_id"]
    for user in manager.list_users():
        mark = "*" if user["user_id"] == current else " "
        print(f"{mark} {user['user_id'] or '(default)':36s}  {user['display_name']}")


def cmd_create_user(args):
    _print(_manager().create_user(args.name, description=args.description))


def _feedback_printer():
    """Print SDK calibration feedback once per step/state change, with its hints."""
    last = [None]

    def on_feedback(fb):
        key = (fb.get("step_index"), fb.get("state"))
        if key == last[0]:
            return
        last[0] = key
        step = f"{fb.get('step_index', 0) + 1}/{fb.get('step_total', '?')}"
        hints = "; ".join(fb.get("hints") or [])
        print(f"  step {step} {fb.get('step_name', '')} [{fb.get('state', '')}] {hints}", flush=True)

    return on_feedback


def cmd_calibrate(args):
    manager = _manager()
    manager.switch_user(args.user)
    devices = manager.scan()
    if not devices:
        raise SystemExit("no Wuji glove found; check scripts/setup_wuji_gloves.sh --check")
    for i, device in enumerate(devices):
        glove = manager.connect(sn=device.sn, device_name=f"calib_{i}")
        print(f"calibrating glove {device.sn}: follow the poses requested by the SDK", flush=True)
        try:
            result = glove.calibrate_blocking(timeout_s=args.timeout, on_feedback=_feedback_printer())
            _print({"sn": device.sn, "handedness": result.get("handedness"),
                    "poses_collected": result.get("poses_collected")})
        finally:
            manager.disconnect(f"calib_{i}")


def cmd_export(args):
    _print(export_bundle(args.user, args.out, _manager()))


def cmd_preview(args):
    _print(preview_bundle(args.bundle, _manager()))


def cmd_import(args):
    _print(import_and_select_bundle(args.bundle, _manager()))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("users", help="list SDK users (* = current)").set_defaults(func=cmd_users)
    p = sub.add_parser("create-user", help="create an SDK user to hold one set of glove calibrations")
    p.add_argument("name")
    p.add_argument("--description", default=None)
    p.set_defaults(func=cmd_create_user)
    p = sub.add_parser("calibrate", help="run hand-model calibration for every connected glove")
    p.add_argument("--user", required=True, help="SDK user id the calibration is stored under")
    p.add_argument("--timeout", type=float, default=900.0)
    p.set_defaults(func=cmd_calibrate)
    p = sub.add_parser("export", help="write one user's calibration as a bundle")
    p.add_argument("--user", required=True)
    p.add_argument("--out", type=Path, default=DEFAULT_BUNDLE)
    p.set_defaults(func=cmd_export)
    for name, func, text in (("preview", cmd_preview, "validate a bundle without importing"),
                             ("import", cmd_import, "import a bundle and select its owner user")):
        p = sub.add_parser(name, help=text)
        p.add_argument("bundle", type=Path, nargs="?", default=DEFAULT_BUNDLE)
        p.set_defaults(func=func)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
