#!/usr/bin/env python3
"""Convert AMASS (SMPL+H "G" release) clips into five-tracker synthetic training sequences in the same
format as generate_sparse_tracker_synthetic.py, posed with a calibrated subject profile (shape, scale,
mounts).  Only root + 21 body joints are taken from AMASS (identical layout in SMPL-X); hands are zero.

Clips are resampled to 60 Hz and cut into fixed-length sequences; the recording's own translation is
kept (the trainer normalizes positions to the first head position anyway)."""
from __future__ import annotations
import argparse, json, sys, tarfile, io
from pathlib import Path
import numpy as np, torch
from scipy.spatial.transform import Rotation, Slerp
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from five_tracker_smplx.profile_fitting import SubjectProfile
from five_tracker_smplx.optimizer import _global_body_rotations
from five_tracker_smplx.model import ModelPaths, load_body_model
IDS = [15, 20, 21, 7, 8]; FPS = 60

def resample(poses66, trans, src_fps):
    T = len(poses66); ts = np.arange(T) / src_fps; ut = np.arange(0, ts[-1], 1 / FPS)
    out = np.zeros((len(ut), 66), np.float32)
    for j in range(22):
        out[:, 3*j:3*j+3] = Slerp(ts, Rotation.from_rotvec(poses66[:, 3*j:3*j+3]))(ut).as_rotvec()
    tr = np.stack([np.interp(ut, ts, trans[:, a]) for a in range(3)], -1).astype(np.float32)
    return out, tr

def rot6d(m): return np.concatenate((m[..., :, 0], m[..., :, 1]), axis=-1).astype(np.float32)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--amass", type=Path, nargs="+", required=True, help="AMASS subset .tar.bz2 archives or extracted directories")
    p.add_argument("--profile", type=Path, required=True); p.add_argument("--output", type=Path, required=True)
    p.add_argument("--frames", type=int, default=240, help="frames per training sequence at 60 Hz")
    p.add_argument("--stride", type=int, default=240, help="stride between sequence starts (frames)")
    p.add_argument("--max-sequences", type=int, default=0); p.add_argument("--start-index", type=int, default=1000)
    p.add_argument("--source-up", choices=("z","y"), default="z", help="up axis of the source clips (AMASS SMPL+H is z-up)")
    p.add_argument("--min-clip-s", type=float, default=2.0); p.add_argument("--seed", type=int, default=0); p.add_argument("--device", default="cpu")
    a = p.parse_args(); a.output.mkdir(parents=True, exist_ok=True); rng = np.random.default_rng(a.seed)
    profile = SubjectProfile.load(a.profile); body = load_body_model(ModelPaths(Path("models/smplx"), Path("models/vposer/V02_05"), "smplx", "neutral"), a.device)
    mount_p = torch.tensor(profile.attachment_position, dtype=torch.float32, device=a.device); mount_r = torch.tensor(profile.attachment_rotation, dtype=torch.float32, device=a.device)
    betas = torch.tensor(profile.betas, dtype=torch.float32, device=a.device); scale = float(profile.scale)
    def clips():
        for src in a.amass:
            if src.is_dir():
                for f in sorted(src.rglob("*_poses.npz")): yield str(f), np.load(f)
            else:
                with tarfile.open(src, "r:bz2") as tf:
                    for m in tf:
                        if m.isfile() and m.name.endswith("_poses.npz"):
                            yield m.name, np.load(io.BytesIO(tf.extractfile(m).read()))
    n_seq = 0; manifest = []
    for name, z in clips():
        if "poses" not in z.files: continue
        fps = float(z["mocap_framerate"]) if "mocap_framerate" in z.files else 120.0
        poses = np.asarray(z["poses"], np.float64); trans = np.asarray(z["trans"], np.float64)
        if len(poses) / fps < a.min_clip_s: continue
        # AMASS is z-up; the project world is y-up (+Y up, as in the recorder and the other synthetic sets).
        p66_raw, tr_raw = resample(poses[:, :66], trans, fps)
        if a.source_up == "z":
            Rfix = Rotation.from_euler("x", -90, degrees=True)
            p66 = p66_raw.copy(); p66[:, :3] = (Rfix * Rotation.from_rotvec(p66_raw[:, :3])).as_rotvec().astype(np.float32); tr = Rfix.apply(tr_raw).astype(np.float32)
        else: p66, tr = p66_raw, tr_raw
        for s in range(0, len(p66) - a.frames + 1, a.stride):
            seg = p66[s:s + a.frames]; tseg = tr[s:s + a.frames]; count = len(seg)
            go = torch.tensor(seg[:, :3], device=a.device); bp = torch.tensor(seg[:, 3:], device=a.device).reshape(count, 21, 3); trt = torch.tensor(tseg, device=a.device)
            with torch.no_grad():
                out = body(betas=betas[None].expand(count, -1), global_orient=go, transl=torch.zeros(count, 3, device=a.device), body_pose=bp.reshape(count, -1),
                           left_hand_pose=torch.zeros(count, 45, device=a.device), right_hand_pose=torch.zeros(count, 45, device=a.device), jaw_pose=torch.zeros(count, 3, device=a.device),
                           leye_pose=torch.zeros(count, 3, device=a.device), reye_pose=torch.zeros(count, 3, device=a.device), expression=torch.zeros(count, 10, device=a.device), return_verts=False)
                rotations = _global_body_rotations(go, bp, body.parents)
                tp = scale * out.joints[:, IDS] + trt[:, None]; tp = tp + (rotations[:, IDS] @ mount_p[None, :, :, None]).squeeze(-1); trr = rotations[:, IDS] @ mount_r[None]
                ankle_y = out.joints[:, (7, 8), 1].cpu().numpy() + tseg[:, None, 1]
            tpn = tp.cpu().numpy(); trn = trr.cpu().numpy()
            speed = np.linalg.norm(np.diff(tpn[:, (3, 4)], axis=0, prepend=tpn[:1, (3, 4)]), axis=-1) * FPS
            base = np.percentile(ankle_y, 20, axis=0); contact = ((ankle_y <= base[None] + .025) & (speed <= .18)).astype(np.float32)
            head_rot = trn[:, 0]; rel_pos = np.einsum("tij,tnj->tni", head_rot.transpose(0, 2, 1), tpn - tpn[:, :1]); rel_rot = np.einsum("tij,tnjk->tnik", head_rot.transpose(0, 2, 1), trn)
            inputs = np.concatenate((rot6d(rel_rot), rel_pos, np.ones((count, 5, 1), np.float32)), -1).reshape(count, -1)
            root_rel = Rotation.from_matrix(head_rot.transpose(0, 2, 1) @ Rotation.from_rotvec(seg[:, :3]).as_matrix()).as_rotvec().astype(np.float32)
            targets = np.concatenate((root_rel, np.einsum("tij,tj->ti", head_rot.transpose(0, 2, 1), tseg - tpn[:, 0]), seg[:, 3:]), -1).astype(np.float32)
            idx = a.start_index + n_seq
            np.savez_compressed(a.output / f"sequence_{idx:04d}.npz", inputs=inputs, targets=targets, contact=contact, global_orient=seg[:, :3].astype(np.float32), transl=tseg,
                                body_pose=seg[:, 3:].astype(np.float32), tracker_positions=tpn.astype(np.float32), tracker_rotations=trn.astype(np.float32), source=name, start_frame_60hz=s)
            manifest.append({"sequence": f"sequence_{idx:04d}", "source": name, "start_60hz": int(s)}); n_seq += 1
            if a.max_sequences and n_seq >= a.max_sequences: break
        if a.max_sequences and n_seq >= a.max_sequences: break
    json.dump({"profile": str(a.profile), "amass": [str(x) for x in a.amass], "frames": a.frames, "stride": a.stride, "fps": FPS, "sequences": manifest}, open(a.output / "manifest.json", "w"), indent=1)
    print(f"wrote {n_seq} sequences to {a.output}")

if __name__ == "__main__": main()
