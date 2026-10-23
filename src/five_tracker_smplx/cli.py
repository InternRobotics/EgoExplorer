"""Command line entry point: subject calibration, hand fusion and mesh viewers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .device import select_device
from .full_mesh_viewer import write_external_full_mesh_viewer
from .manus_batch import fit_and_fuse_manus_batch
from .manus_hands import fit_and_visualize_manus
from .subject_calibration import SubjectCalibrationOptions, calibrate_subject


def main() -> None:
    parser = argparse.ArgumentParser(description="EgoExplorer five-tracker -> SMPL-X tools")
    subparsers = parser.add_subparsers(dest="command", required=True)
    device_info = subparsers.add_parser("device-info", help="report the selected PyTorch compute device")
    device_info.add_argument("--device", default="auto", choices=("auto", "cpu", "mps", "cuda"))
    subject = subparsers.add_parser("calibrate-subject", help="jointly fit subject scale, SMPL-X shape and fixed tracker mounts")
    subject.add_argument("--data", required=True, help="directory containing recorder session folders")
    subject.add_argument("--height", type=float, required=True, help="measured standing height in metres")
    subject.add_argument("--armspan", type=float, required=True, help="measured fingertip arm span in metres")
    subject.add_argument("--smplx-prior", default=None, help="portable SMPL-X NPZ whose betas seed and regularize subject shape")
    subject.add_argument("--smplx-prior-weight", type=float, default=.18)
    subject.add_argument("--output-dir", default="outputs/subject_calibration")
    subject.add_argument("--body-model-path", default="models/smplx")
    subject.add_argument("--vposer-path", default="models/vposer/V02_05")
    subject.add_argument("--motion-cache", default="assets/calibration_motion_vposer_mano.npz")
    subject.add_argument("--device", default="cpu", choices=("cpu", "mps", "cuda"))
    subject.add_argument("--iterations", type=int, default=1100)
    subject.add_argument("--optimized-betas", type=int, default=5, choices=range(1, 11))
    subject.add_argument("--verbose", action="store_true")
    native_hands = subparsers.add_parser("fit-manus-hands", help="fit native MANUS glove NPZ and write an interactive hand viewer")
    native_hands.add_argument("--input", required=True, help="recorder native glove .npz")
    native_hands.add_argument("--subject-profile", required=True)
    native_hands.add_argument("--output-dir", default="outputs/manus_hand_fit")
    native_hands.add_argument("--body-model-path", default="models/smplx")
    native_hands.add_argument("--device", default="cpu", choices=("cpu", "mps", "cuda"))
    native_hands.add_argument("--max-frames", type=int, default=360, help="uniform native frames to fit; <=0 keeps all")
    native_hands.add_argument("--window-size", type=int, default=90)
    native_hands.add_argument("--iterations", type=int, default=120)
    native_hands.add_argument("--verbose", action="store_true")
    fuse_hands = subparsers.add_parser("fuse-manus-batch", help="fit native MANUS hands and fuse them into profile-driven full-body sequences")
    fuse_hands.add_argument("--input-root", action="append", required=True, help="session root to scan; repeatable")
    fuse_hands.add_argument("--body-output-dir", required=True, help="existing reconstruct-profile output root")
    fuse_hands.add_argument("--subject-profile", required=True)
    fuse_hands.add_argument("--body-model-path", default="models/smplx")
    fuse_hands.add_argument("--device", default="cpu", choices=("cpu", "mps", "cuda"))
    fuse_hands.add_argument("--max-hand-frames", type=int, default=900)
    fuse_hands.add_argument("--window-size", type=int, default=90)
    fuse_hands.add_argument("--iterations", type=int, default=70)
    fuse_hands.add_argument("--viewer-max-frames", type=int, default=450)
    fuse_hands.add_argument("--reuse-hand-fits", action="store_true", help="reuse existing per-session MANUS fits and only re-fuse/regenerate viewer")
    fuse_hands.add_argument("--no-protocol", action="store_true", help="treat inputs as ordinary trajectories, not the 01--17 calibration protocol")
    fuse_hands.add_argument("--no-viewer", action="store_true", help="skip the combined HTML; required when sharding per session")
    fuse_hands.add_argument("--verbose", action="store_true")
    full_mesh = subparsers.add_parser("full-mesh-viewer", help="write an external-data WebGL viewer with complete SMPL-X topology")
    full_mesh.add_argument("--input-root", action="append", required=True, help="session root to scan; repeatable")
    full_mesh.add_argument("--body-output-dir", action="append", required=True, help="existing reconstructed/fused body output root; repeat for combined datasets")
    full_mesh.add_argument(
        "--hard-body-output-dir", action="append", default=None,
        help="optional 100%% tracker-hard reconstruction root; enables the interactive hardness slider",
    )
    full_mesh.add_argument(
        "--sole-body-output-dir", action="append", default=None,
        help="optional natural reconstruction with strict contact-only sole-plane constraint",
    )
    full_mesh.add_argument(
        "--hard-sole-body-output-dir", action="append", default=None,
        help="optional tracker-hard reconstruction with strict sole-plane constraint",
    )
    full_mesh.add_argument("--subject-profile", required=True)
    full_mesh.add_argument("--output-dir", required=True)
    full_mesh.add_argument("--body-model-path", default="models/smplx")
    full_mesh.add_argument("--device", default="cpu", choices=("cpu", "mps", "cuda"))
    full_mesh.add_argument("--viewer-max-frames", type=int, default=0, help="maximum display frames; <=0 exports every source frame")
    full_mesh.add_argument("--protocol-mode", choices=("auto", "calibration", "none"), default="auto", help="auto uses protocol labels only for reconstructed sessions carrying a calibration marker")
    full_mesh.add_argument("--floor-reference-output-dir", action="append", default=None, help="optional output root supplying the shared viewer ground plane; repeat for combined datasets")
    full_mesh.add_argument("--skip-missing-natural", action="store_true", help="build an incremental viewer from only natural results currently available")
    full_mesh.add_argument("--reuse-existing-meshes", action="store_true", help="reuse already rendered per-variant mesh streams and metadata")
    args = parser.parse_args()

    if args.command == "device-info":
        print(json.dumps(select_device(args.device).to_mapping(), ensure_ascii=False, indent=2))
        return
    if args.command == "calibrate-subject":
        profile = calibrate_subject(args.data, SubjectCalibrationOptions(
            height_m=args.height, armspan_m=args.armspan,
            smplx_prior_path=Path(args.smplx_prior) if args.smplx_prior else None,
            smplx_prior_weight=args.smplx_prior_weight,
            body_model_path=Path(args.body_model_path), vposer_path=Path(args.vposer_path),
            motion_cache_path=Path(args.motion_cache), output_dir=Path(args.output_dir),
            device=args.device, iterations=args.iterations, optimized_betas=args.optimized_betas,
            verbose=args.verbose,
        ))
        print(json.dumps({"fit": profile["fit"], "quality": {"tracker_summary": profile["quality"]["tracker_summary"], "excluded_streams": profile["quality"]["excluded_streams"]}, "outputs": profile["outputs"]}, ensure_ascii=False, indent=2, default=lambda value: value.tolist() if isinstance(value, np.ndarray) else str(value)))
        return
    if args.command == "fit-manus-hands":
        report = fit_and_visualize_manus(
            args.input, args.subject_profile, args.output_dir,
            body_model_path=args.body_model_path, device=args.device,
            max_frames=args.max_frames, window_size=args.window_size,
            iterations=args.iterations, verbose=args.verbose,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    if args.command == "fuse-manus-batch":
        report = fit_and_fuse_manus_batch(
            args.input_root, args.body_output_dir, args.subject_profile,
            body_model_path=args.body_model_path, device=args.device,
            max_hand_frames=args.max_hand_frames, hand_window_size=args.window_size,
            hand_iterations=args.iterations, viewer_max_frames=args.viewer_max_frames,
            reuse_hand_fits=args.reuse_hand_fits, use_protocol=not args.no_protocol,
            write_viewer=not args.no_viewer, verbose=args.verbose,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    if args.command == "full-mesh-viewer":
        report = write_external_full_mesh_viewer(
            args.input_root, args.body_output_dir, args.subject_profile, args.output_dir,
            body_model_path=args.body_model_path, device=args.device,
            max_frames=args.viewer_max_frames, protocol_mode=args.protocol_mode,
            hard_body_output_dir=args.hard_body_output_dir,
            sole_body_output_dir=args.sole_body_output_dir,
            hard_sole_body_output_dir=args.hard_sole_body_output_dir,
            floor_reference_output_dir=args.floor_reference_output_dir,
            skip_missing_natural=args.skip_missing_natural,
            reuse_existing_meshes=args.reuse_existing_meshes,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return



if __name__ == "__main__":
    main()
