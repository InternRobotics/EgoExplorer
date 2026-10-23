#!/usr/bin/env python3
"""Generate self-consistent five-tracker training sequences from VPoser/SMPL-X.

The generated labels are exact for the chosen SMPL-X body and attachment
calibration.  They are not intended to replace mocap data; they provide a
controlled way to train and test the temporal architecture without teaching
it the branch errors of the old IK solver.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from five_tracker_smplx.model import ModelPaths, load_body_model, load_vposer
from five_tracker_smplx.optimizer import _global_body_rotations
from tracker_streams import rot6d as _rot6d


IDS = (15, 20, 21, 7, 8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", type=Path, default=Path("profiles/collector_a_subject_profile.json"))
    parser.add_argument("--sequences", type=int, default=24)
    parser.add_argument("--frames", type=int, default=240)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    import torch
    from five_tracker_smplx.profile_fitting import SubjectProfile

    rng = np.random.default_rng(args.seed)
    profile = SubjectProfile.load(args.profile)
    body = load_body_model(ModelPaths(Path("models/smplx"), None, "smplx", "neutral"), args.device).eval()
    vposer = load_vposer(ModelPaths(Path("models/smplx"), Path("models/vposer/V02_05"), "smplx", "neutral"), args.device).eval()
    for parameter in body.parameters():
        parameter.requires_grad_(False)
    for parameter in vposer.parameters():
        parameter.requires_grad_(False)
    parents = body.parents.to(args.device)
    ids = torch.tensor(IDS, dtype=torch.long, device=args.device)
    mount_position = torch.tensor(profile.attachment_position, dtype=torch.float32, device=args.device)
    mount_rotation = torch.tensor(profile.attachment_rotation, dtype=torch.float32, device=args.device)
    betas = torch.tensor(profile.betas, dtype=torch.float32, device=args.device)
    args.output.mkdir(parents=True, exist_ok=True)
    for sequence_id in range(args.sequences):
        count = args.frames
        latent_np = np.zeros((count, 32), dtype=np.float32)
        latent_np[0] = rng.normal(0.0, 0.45, 32)
        for frame in range(1, count):
            latent_np[frame] = .985 * latent_np[frame - 1] + rng.normal(0.0, .08, 32)
        # Add a smooth turn and small root translation so the model sees the
        # same global/local ambiguity as a real sequence.
        phase = np.linspace(0.0, 1.0, count, dtype=np.float32)
        yaw = (rng.uniform(-.4, .4) + rng.uniform(1.0, 2.5) * np.sin(np.pi * phase)).astype(np.float32)
        root_np = np.stack((.04 * np.sin(2 * np.pi * phase), yaw, .03 * np.sin(np.pi * phase)), axis=1).astype(np.float32)
        translation_np = np.stack((.12 * np.sin(2 * np.pi * phase + rng.uniform()), .02 * np.sin(4 * np.pi * phase), .18 * phase), axis=1).astype(np.float32)
        with torch.no_grad():
            latent = torch.tensor(latent_np, dtype=torch.float32, device=args.device)
            body_pose = vposer.decode(latent)["pose_body"]
            global_orient = torch.tensor(root_np, dtype=torch.float32, device=args.device)
            translation = torch.tensor(translation_np, dtype=torch.float32, device=args.device)
            output = body(
                betas=betas[None].expand(count, -1), global_orient=global_orient,
                transl=torch.zeros((count, 3), device=args.device), body_pose=body_pose.reshape(count, -1),
                left_hand_pose=torch.zeros((count, 45), device=args.device), right_hand_pose=torch.zeros((count, 45), device=args.device),
                jaw_pose=torch.zeros((count, 3), device=args.device), leye_pose=torch.zeros((count, 3), device=args.device), reye_pose=torch.zeros((count, 3), device=args.device), expression=torch.zeros((count, 10), device=args.device), return_verts=False,
            )
            rotations = _global_body_rotations(global_orient, body_pose, parents)
            tracker_positions = float(profile.scale) * output.joints[:, ids] + translation[:, None]
            tracker_positions = tracker_positions + (rotations[:, ids] @ mount_position[None, :, :, None]).squeeze(-1)
            tracker_rotations = rotations[:, ids] @ mount_rotation[None]
            head_pos = tracker_positions[:, :1]
            head_rot = tracker_rotations[:, 0]
            rel_pos = torch.einsum("tij,tnj->tni", head_rot.transpose(-1, -2), tracker_positions - head_pos)
            rel_rot = torch.einsum("tij,tnjk->tnik", head_rot.transpose(-1, -2), tracker_rotations)
            inputs = np.concatenate((_rot6d(rel_rot.cpu().numpy()), rel_pos.cpu().numpy()), axis=-1)
            inputs = np.concatenate((inputs, np.ones((count, 5, 1), dtype=np.float32)), axis=-1).reshape(count, -1)
            root_matrix = torch.tensor(Rotation.from_rotvec(root_np).as_matrix(), dtype=torch.float32, device=args.device)
            root_rel = head_rot.transpose(-1, -2) @ root_matrix
            root_rel_vec = Rotation.from_matrix(root_rel.cpu().numpy()).as_rotvec().astype(np.float32)
            translation_rel = torch.einsum("tij,tj->ti", head_rot.transpose(-1, -2), translation - tracker_positions[:, 0]).cpu().numpy()
            targets = np.concatenate((root_rel_vec, translation_rel, body_pose.cpu().numpy().reshape(count, -1)), axis=-1).astype(np.float32)
            ankle_y = output.joints[:, (7, 8), 1].cpu().numpy() + translation_np[:, None, 1]
            ankle_speed = np.linalg.norm(np.diff(tracker_positions[:, (3, 4)].cpu().numpy(), axis=0, prepend=tracker_positions[:1, (3, 4)].cpu().numpy()), axis=-1) * 60.0
            baseline = np.percentile(ankle_y, 20, axis=0)
            contact = (ankle_y <= baseline[None] + .025) & (ankle_speed <= .18)
        # Small observation noise/dropout prevents the model from memorizing
        # exact synthetic geometry and exercises the validity channel.
        noisy_inputs = inputs.copy()
        noisy_inputs[:, :50:10] += rng.normal(0.0, .002, noisy_inputs[:, :50:10].shape).astype(np.float32)
        valid = np.ones((count, 5), dtype=bool)
        drop = rng.random((count, 5)) < .015
        valid[drop] = False
        noisy_inputs.reshape(count, 5, 10)[..., -1] = valid.astype(np.float32)
        noisy_inputs.reshape(count, 5, 10)[~valid] = 0.0
        np.savez_compressed(args.output / f"sequence_{sequence_id:03d}.npz", inputs=noisy_inputs, targets=targets, contact=contact.astype(np.float32), global_orient=root_np, transl=translation_np, body_pose=body_pose.cpu().numpy().reshape(count, -1), tracker_positions=tracker_positions.cpu().numpy(), tracker_rotations=tracker_rotations.cpu().numpy())
    print(f"generated {args.sequences} sequences x {args.frames} frames at {args.output}")


if __name__ == "__main__":
    main()
