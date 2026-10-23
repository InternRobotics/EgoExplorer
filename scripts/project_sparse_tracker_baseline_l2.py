#!/usr/bin/env python3
"""Project a learned sparse-tracker pose back onto exact tracker observations.

The Constrain stage of C3: the network proposal is refined by differentiable SMPL-X
FK against the trackers (L1 skeleton + joint limits), the body's own surface (L2) and
the floor (L3).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation, Slerp
from scipy.spatial import ConvexHull

from five_tracker_smplx.model import ModelPaths, load_body_model
from five_tracker_smplx.optimizer import _global_body_rotations
from tracker_streams import RAW_ROLES, wxyz_to_matrix as _wxyz_to_matrix
from body_part_surface import BodyPartSurface
from directional_surface import ExploratorySurface
from joint_acceleration import joint_acceleration_loss
from stable_joint_angles import flexion, inclination


IDS = (15, 20, 21, 7, 8)


def _winding_numbers(points: torch.Tensor, triangles: torch.Tensor) -> torch.Tensor:
    """Generalized winding number for a small batch of query points."""
    centered = triangles[:, None, :, :, :] - points[:, :, None, None, :]
    norms = torch.linalg.vector_norm(centered, dim=-1).clamp_min(1e-8)
    numerator = (centered[..., 0, :] * torch.cross(centered[..., 1, :], centered[..., 2, :], dim=-1)).sum(-1)
    denominator = norms.prod(-1)
    denominator = denominator + (centered[..., 0, :] * centered[..., 1, :]).sum(-1) * norms[..., 2]
    denominator = denominator + (centered[..., 0, :] * centered[..., 2, :]).sum(-1) * norms[..., 1]
    denominator = denominator + (centered[..., 1, :] * centered[..., 2, :]).sum(-1) * norms[..., 0]
    return torch.atan2(numerator, denominator).sum(-1) / (2.0 * np.pi)



def _load_contact_probability(path: Path, timestamps: np.ndarray) -> np.ndarray:
    """Read the proposal's foot-contact probabilities onto the trajectory clock.

    The proposal exporter runs the network on a uniform 60 Hz grid but writes
    the trajectory back on the source capture timestamps, so the two streams
    only ever share a length when the capture happens to run at 60 Hz.  Slicing
    the longer one silently misaligned the signal for sub-60 Hz captures and
    crashed outright for the faster ones, so resample onto the trajectory clock
    instead.
    """
    values = np.asarray(np.load(path), dtype=np.float32)
    count = len(timestamps)
    if len(values) == count:
        return values
    relative = np.asarray(timestamps, dtype=np.float64) - float(timestamps[0])
    # Rebuild the exporter's grid exactly; fall back to a uniform span if the
    # producer ever changes its sampling.
    grid = np.unique(np.append(np.arange(0.0, relative[-1], 1.0 / 60.0), relative[-1]))
    if len(grid) != len(values):
        grid = np.linspace(0.0, float(relative[-1]), len(values))
    flat = values.reshape(len(values), -1)
    resampled = np.stack(
        [np.interp(relative, grid, flat[:, column]) for column in range(flat.shape[1])],
        axis=1,
    )
    return resampled.reshape((count,) + values.shape[1:]).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory", type=Path, required=True, help="proposal smplx_profile_driven.npz")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--chunk-overlap", type=int, default=0, help="overlap between optimization chunks; only the center step is committed")
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--learning-rate", type=float, default=.015)
    parser.add_argument("--max-step-pose-deg", type=float, default=0.0, help="optional per-iteration axis-angle step limit; zero disables")
    parser.add_argument("--max-step-translation", type=float, default=0.0, help="optional per-iteration translation step limit in metres; zero disables")
    parser.add_argument("--position-weight", type=float, default=3000.0)
    parser.add_argument("--rotation-weight", type=float, default=20.0)
    parser.add_argument("--pose-prior-weight", type=float, default=2.0)
    parser.add_argument("--contact-weight", type=float, default=80.0)
    parser.add_argument("--contact-height-offset", type=float, default=0.0, help="offset applied to the estimated sole plane")
    parser.add_argument("--sole-radius", type=float, default=.14)
    parser.add_argument("--self-collision-weight", type=float, default=0.0, help="geometric arm/torso capsule barrier; zero disables")
    parser.add_argument("--self-collision-radius", type=float, default=.075)
    parser.add_argument("--mesh-collision-weight", type=float, default=0.0, help="sampled SMPL-X arm/torso surface distance barrier; zero disables")
    parser.add_argument("--mesh-collision-radius", type=float, default=.018)
    parser.add_argument("--mesh-sample-count", type=int, default=96, help="number of torso and arm surface vertices used by the CPU mesh barrier")
    parser.add_argument("--mesh-sample-mode", choices=("topk", "uniform"), default="topk", help="topk: highest LBS-weight vertices; uniform: seeded random subset of part-dominated vertices")
    parser.add_argument("--mesh-sample-exclude-radius", type=float, default=0.0, help="drop sampled arm/torso vertices within this distance (m) of the shoulder joints on the neutral mesh; zero disables")
    parser.add_argument("--mesh-include-hands", action="store_true", help="include finger-dominated vertices in the arm sample pool")
    parser.add_argument("--mesh-torso-include-thighs", action="store_true", help="include thigh-dominated vertices in the torso sample pool")
    parser.add_argument("--hull-collision-mode", choices=("hinge", "push"), default="hinge", help="push: (d_hull + radius)^2 on inside points, gradient at any depth")
    parser.add_argument("--mesh-collision-legs-weight", type=float, default=0.0, help="surface hinge between the two lower legs (shank+foot); zero disables")
    parser.add_argument("--hull-collision-weight", type=float, default=0.0, help="inside/outside convex-hull arm/torso barrier; zero disables")
    parser.add_argument("--hull-collision-radius", type=float, default=.030)
    parser.add_argument("--hull-recompute-every", type=int, default=0, help="recompute the detached inside mask every N iterations (0 keeps the iteration-0 mask)")
    parser.add_argument("--floor-penetration-weight", type=float, default=0.0, help="penalize any sampled sole vertex below ground; zero disables")
    parser.add_argument("--floor-clearance", type=float, default=0.0, help="required sole clearance above ground in metres")
    parser.add_argument("--global-floor-penetration-weight", type=float, default=0.0, help="penalize every SMPL-X vertex below ground; zero disables")
    parser.add_argument("--sole-flatness-weight", type=float, default=0.0, help="penalize height variance across the lowest sole vertices while a foot is in contact")
    parser.add_argument("--joint-limit-weight", type=float, default=0.0, help="penalize implausible knee/elbow flexion and excessive torso lean")
    parser.add_argument("--max-knee-flexion-deg", type=float, default=130.0)
    parser.add_argument("--max-elbow-flexion-deg", type=float, default=150.0)
    parser.add_argument("--torso-lean-limit-deg", type=float, default=20.0)
    parser.add_argument("--knee-hyperextension-weight", type=float, default=0.0, help="signed penalty on knee local flexion below -knee-hyperextension-deg (backward-bent knee); zero disables")
    parser.add_argument("--knee-hyperextension-deg", type=float, default=5.0)
    parser.add_argument("--pelvis-roll-weight", type=float, default=0.0, help="penalty on lateral pelvis tilt (pelvis left axis leaving the horizontal) beyond --pelvis-roll-limit-deg; zero disables")
    parser.add_argument("--pelvis-roll-limit-deg", type=float, default=10.0)
    parser.add_argument("--pose-reference", type=Path, default=None, help="npz from build_pose_reference.py (body_pose_ref, mask, joint_mask); branch-selection prior for teacher labels")
    parser.add_argument("--pose-reference-weight", type=float, default=0.0, help="weight of the masked squared axis-angle distance to the reference pose; zero disables")
    parser.add_argument("--body-model-path", type=Path, default=Path("models/smplx"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--surface-mode", choices=("legacy", "general", "unsigned_dense", "signed_mean", "signed_top8"), default="legacy")
    parser.add_argument("--joint-acceleration-weight", type=float, default=0.)
    parser.add_argument("--attachment-lr", type=float, default=0.02, help="learning rate of the solved attachment offsets (their scale is centimetres, not radians per frame)")
    parser.add_argument("--attachment-warmup-iterations", type=int, default=240, help="extra iterations on the first chunk while the attachments are still free")
    parser.add_argument("--freeze-attachments-after-first-chunk", action="store_true", help="solve the attachments once on the first chunk, then keep them fixed for the session")
    parser.add_argument("--solve-attachments", action="store_true", help="self-calibration ablation: treat the five attachment offsets (translation + rotation delta) as free variables shared across the session, optimized jointly with the pose from the given initial attachments")
    args = parser.parse_args()
    import torch

    with np.load(args.trajectory) as values:
        timestamps = np.asarray(values["timestamps"], dtype=np.float64)
        betas = np.asarray(values["betas"], dtype=np.float32)
        global_orient = np.asarray(values["global_orient"], dtype=np.float32).copy()
        transl = np.asarray(values["transl"], dtype=np.float32).copy()
        body_pose = np.asarray(values["body_pose"], dtype=np.float32).copy().reshape(-1, 21, 3)
        scale = float(np.asarray(values["scale"]))
        left_hand = np.asarray(values["left_hand_pose"], dtype=np.float32).copy()
        right_hand = np.asarray(values["right_hand_pose"], dtype=np.float32).copy()
        mount_position = np.asarray(values["attachment_position"], dtype=np.float32)
        mount_rotation = np.asarray(values["attachment_rotation"], dtype=np.float32)
    meta_path = args.trajectory.with_suffix(".json")
    meta = json.loads(meta_path.read_text())
    raw = np.load(meta["source"], allow_pickle=False)
    count = len(timestamps)
    pose_ref = None
    if args.pose_reference is not None and args.pose_reference_weight > 0:
        with np.load(args.pose_reference) as pr:
            pose_ref = (np.asarray(pr["body_pose_ref"], np.float32)[:count], np.asarray(pr["mask"], np.float32)[:count], np.asarray(pr["joint_mask"], np.float32))
        if len(pose_ref[0]) != count: raise SystemExit(f"pose reference has {len(pose_ref[0])} frames, trajectory has {count}")
    target_pos = np.stack([np.asarray(raw[f"tracker_0__pose.{role}.position"][:count], dtype=np.float32) for role in RAW_ROLES], axis=1)
    target_rot = np.stack([_wxyz_to_matrix(np.asarray(raw[f"tracker_0__pose.{role}.orientation"][:count], dtype=np.float64)) for role in RAW_ROLES], axis=1)
    valid = np.stack([
        np.asarray(raw[f"tracker_0__pose.{role}.valid"][:count], dtype=bool)
        if f"tracker_0__pose.{role}.valid" in raw else
        np.ones(count, dtype=bool)
        for role in RAW_ROLES
    ], axis=1)
    raw.close()
    valid &= np.isfinite(target_pos).all(axis=-1) & np.isfinite(target_rot).all(axis=(-2, -1))
    for tracker_index, role in enumerate(RAW_ROLES):
        good = np.flatnonzero(valid[:, tracker_index])
        if len(good) < 2:
            raise ValueError(f"too few valid tracker samples for {role}")
        clipped = np.clip(timestamps, timestamps[good[0]], timestamps[good[-1]])
        target_rot[:, tracker_index] = Slerp(
            timestamps[good], Rotation.from_matrix(target_rot[good, tracker_index])
        )(clipped).as_matrix()
        for axis in range(3):
            target_pos[:, tracker_index, axis] = np.interp(
                timestamps, timestamps[good], target_pos[good, tracker_index, axis]
            )
    contact_path = args.trajectory.parent / "contact_probability.npy"
    if contact_path.exists():
        contact_probability = _load_contact_probability(contact_path, timestamps)
    else:
        foot = target_pos[:, 3:5]
        dt = float(np.median(np.diff(timestamps))) if count > 1 else 1.0 / 60.0
        speed = np.linalg.norm(np.diff(foot[:, :, (0, 2)], axis=0, prepend=foot[:1, :, (0, 2)]), axis=-1) / max(dt, 1e-4)
        baseline = np.percentile(foot[:, :, 1], 20, axis=0)
        contact_probability = (((foot[:, :, 1] <= baseline[None] + .03) & (speed <= .18)).astype(np.float32))
    start, end = max(0, args.start), min(count, count if args.end is None else args.end)
    if end <= start:
        raise SystemExit("empty projection range")
    device = args.device
    body = load_body_model(ModelPaths(args.body_model_path, None, "smplx", "neutral"), device).eval()
    for parameter in body.parameters():
        parameter.requires_grad_(False)
    mount_p = torch.tensor(mount_position, dtype=torch.float32, device=device)
    mount_r = torch.tensor(mount_rotation, dtype=torch.float32, device=device)
    mount_dp = torch.nn.Parameter(torch.zeros((5, 3), dtype=torch.float32, device=device))
    mount_dr = torch.nn.Parameter(torch.zeros((5, 3), dtype=torch.float32, device=device))
    from five_tracker_smplx.optimizer import _axis_angle_to_matrix as _aa2m
    def effective_mounts():
        if not args.solve_attachments: return mount_p, mount_r
        return mount_p + mount_dp, mount_r @ _aa2m(mount_dr[None])[0]
    parents = body.parents.to(device)
    ids = torch.tensor(IDS, dtype=torch.long, device=device)
    beta_t = torch.tensor(betas, device=device)
    # Select stable left/right sole vertex sets from the neutral shaped mesh.
    # A soft-min over these vertices provides a differentiable approximation
    # to the lowest heel/toe point without relying on a single ankle joint.
    with torch.no_grad():
        zero = torch.zeros((1, 3), dtype=torch.float32, device=device)
        neutral = body(
            betas=beta_t[None], global_orient=zero, transl=zero,
            body_pose=torch.zeros((1, 63), dtype=torch.float32, device=device),
            left_hand_pose=torch.zeros((1, 45), dtype=torch.float32, device=device),
            right_hand_pose=torch.zeros((1, 45), dtype=torch.float32, device=device),
            jaw_pose=zero, leye_pose=zero, reye_pose=zero,
            expression=torch.zeros((1, 10), dtype=torch.float32, device=device), return_verts=True,
        )
        neutral_vertices = neutral.vertices[0]
        neutral_joints = neutral.joints[0]
        sole_ids = []
        for joint_id in (7, 8):
            distance = torch.linalg.vector_norm(neutral_vertices - neutral_joints[joint_id], dim=-1)
            candidate = torch.where((distance < args.sole_radius) & (neutral_vertices[:, 1] < neutral_joints[joint_id, 1] + .06))[0]
            if len(candidate) < 32:
                candidate = torch.topk(distance, k=32, largest=False).indices
            else:
                # Prefer the lowest patch of the foot, rather than the points
                # merely closest to the ankle.  This gives the floor and
                # contact terms a chance to see heel/toe geometry.
                candidate = candidate[torch.argsort(neutral_vertices[candidate, 1])[:32]]
            sole_ids.append(candidate)
        # Skinning-weight regions provide a stable, subject-specific surface
        # partition without pretending that a handful of joint centers is a
        # mesh collision detector.  This CPU-friendly probe samples only the
        # arm and torso regions; the CUDA torch-mesh-isect implementation is
        # documented separately and remains optional.
        torso_score = neutral.body_model.lbs_weights[0, :, (0, 3, 6, 9, 12)].sum(dim=-1) if hasattr(neutral, "body_model") else body.lbs_weights[:, (0, 3, 6, 9, 12)].sum(dim=-1)
        arm_score = body.lbs_weights[:, (13, 16, 18, 20, 14, 17, 19, 21)].sum(dim=-1)
        sample_count = max(8, int(args.mesh_sample_count))
        if args.mesh_include_hands:
            arm_score = arm_score + body.lbs_weights[:, 25:55].sum(dim=-1)
        if args.mesh_torso_include_thighs:
            torso_score = torso_score + body.lbs_weights[:, (1, 2)].sum(dim=-1)
        if args.mesh_sample_mode == "uniform":
            # Coverage rather than weight: any vertex the part dominates is a candidate, and a fixed-seed subset is kept.
            gen = torch.Generator(device="cpu").manual_seed(0)
            def uniform_ids(score):
                cand = torch.where(score > 0.5)[0]
                perm = torch.randperm(len(cand), generator=gen)[:min(sample_count, len(cand))]
                return cand[perm.to(cand.device)]
            torso_ids = uniform_ids(torso_score)
            arm_ids = uniform_ids(arm_score)
            lleg_ids = uniform_ids(body.lbs_weights[:, (4, 7, 10)].sum(dim=-1))
            rleg_ids = uniform_ids(body.lbs_weights[:, (5, 8, 11)].sum(dim=-1))
        else:
            torso_ids = torch.topk(torso_score, k=min(sample_count, torso_score.numel())).indices
            arm_ids = torch.topk(arm_score, k=min(sample_count, arm_score.numel())).indices
            lleg_ids = torch.topk(body.lbs_weights[:, (4, 7, 10)].sum(dim=-1), k=sample_count).indices
            rleg_ids = torch.topk(body.lbs_weights[:, (5, 8, 11)].sum(dim=-1), k=sample_count).indices
        if args.mesh_sample_exclude_radius > 0:
            # the armpit is a legitimate contact: drop samples near the shoulder joints of the neutral mesh
            shoulders = neutral_joints[(16, 17), :]
            def away(ids):
                d = torch.cdist(neutral_vertices[ids], shoulders).amin(dim=-1)
                return ids[d > args.mesh_sample_exclude_radius]
            torso_ids, arm_ids = away(torso_ids), away(arm_ids)
        if args.hull_collision_weight > 0:
            torso_dom = torch.isin(body.lbs_weights.argmax(dim=1), torch.tensor((0, 3, 6, 9, 12), device=device))
            torso_dom_ids = torch.where(torso_dom)[0]
            hull = ConvexHull(neutral_vertices[torso_dom_ids].cpu().numpy())
            hull_local_ids = torch.as_tensor(np.unique(hull.simplices), dtype=torch.long, device=device)
            hull_vertex_ids = torso_dom_ids[hull_local_ids]
            hull_faces = torch.as_tensor(hull.simplices, dtype=torch.long, device=device)
            hull_faces = torch.searchsorted(hull_local_ids, hull_faces)
        else:
            hull_vertex_ids = torch.empty((0,), dtype=torch.long, device=device)
            hull_faces = torch.empty((0, 3), dtype=torch.long, device=device)
    general_surface = BodyPartSurface(neutral_vertices, body.lbs_weights, body.faces, args.mesh_sample_count) if args.surface_mode == "general" and args.mesh_collision_weight > 0 else None
    if args.surface_mode in ("unsigned_dense", "signed_mean", "signed_top8") and args.mesh_collision_weight > 0:
        general_surface = ExploratorySurface(neutral_vertices, body.lbs_weights, body.faces, args.surface_mode, args.mesh_sample_count)
    chunk_size = max(1, args.chunk_size)
    overlap = min(max(0, int(args.chunk_overlap)), max(0, chunk_size - 1))
    step = max(1, chunk_size - overlap)
    for chunk_index, write_start in enumerate(range(start, end, step)):
        write_end = min(end, write_start + step)
        # Optimize a wider temporal context, then commit only the center
        # region.  The overlap gives the optimizer access to the preceding
        # solution and prevents independent chunks from choosing opposite IK
        # branches at a turn or squat.
        chunk_start = max(start, write_start - overlap)
        chunk_end = min(end, write_end + overlap)
        sl = slice(chunk_start, chunk_end)
        chunk_count = chunk_end - chunk_start
        go0 = torch.tensor(global_orient[sl], dtype=torch.float32, device=device)
        bp0 = torch.tensor(body_pose[sl], dtype=torch.float32, device=device)
        tr0 = torch.tensor(transl[sl], dtype=torch.float32, device=device)
        go = torch.nn.Parameter(go0.clone())
        bp = torch.nn.Parameter(bp0.clone())
        tr = torch.nn.Parameter(tr0.clone())
        solve_now = args.solve_attachments and (chunk_index == 0 or not args.freeze_attachments_after_first_chunk)
        groups = [{"params": [go, bp, tr], "lr": args.learning_rate}]
        if solve_now: groups.append({"params": [mount_dp, mount_dr], "lr": args.attachment_lr})
        optimizer = torch.optim.Adam(groups)
        chunk_iterations = args.iterations + (args.attachment_warmup_iterations if (solve_now and chunk_index == 0) else 0)
        target_p = torch.tensor(target_pos[sl], dtype=torch.float32, device=device)
        target_r = torch.tensor(target_rot[sl], dtype=torch.float32, device=device)
        valid_t = torch.tensor(valid[sl], dtype=torch.bool, device=device)
        times_t = torch.as_tensor(timestamps[sl], dtype=torch.float64, device=device)
        inside_mask = None
        for iteration in range(chunk_iterations):
            # Keep a copy for the optional trust-region clamp below.  Adam can
            # otherwise take a large barrier gradient step and switch the
            # elbow/knee branch, especially at chunk boundaries.
            go_before = go.detach().clone()
            bp_before = bp.detach().clone()
            tr_before = tr.detach().clone()
            zero3 = torch.zeros((chunk_count, 3), dtype=torch.float32, device=device)
            output = body(
                betas=beta_t[None].expand(chunk_count, -1),
                global_orient=go, transl=zero3, body_pose=bp.reshape(chunk_count, -1),
                left_hand_pose=torch.tensor(left_hand[sl], device=device),
                right_hand_pose=torch.tensor(right_hand[sl], device=device),
                jaw_pose=zero3, leye_pose=zero3, reye_pose=zero3,
                expression=torch.zeros((chunk_count, 10), dtype=torch.float32, device=device), return_verts=True,
            )
            rotations = _global_body_rotations(go, bp, parents)
            predicted = scale * output.joints[:, ids] + tr[:, None]
            eff_p, eff_r = effective_mounts()
            predicted = predicted + (rotations[:, ids] @ eff_p[None, :, :, None]).squeeze(-1)
            predicted_r = rotations[:, ids] @ eff_r[None]
            mask = valid_t[..., None]
            valid_count = mask.sum().clamp_min(1.0)
            pos_loss = ((predicted - target_p).square() * mask).sum() / (valid_count * 3.0)
            rot_loss = ((predicted_r - target_r).square() * mask[..., None]).sum() / (valid_count * 9.0)
            prior = (bp - bp0).square().mean() + .2 * (go - go0).square().mean() + .2 * (tr - tr0).square().mean()
            temporal = (bp[1:] - bp[:-1]).square().mean() if chunk_count > 1 else torch.zeros((), device=device)
            world_joints = scale * output.joints + tr[:, None]
            acceleration_loss = joint_acceleration_loss(world_joints, times_t, valid_t.all(1)) if args.joint_acceleration_weight > 0 else world_joints.sum()*0
            contact = torch.tensor(contact_probability[sl], dtype=torch.float32, device=device).clamp(0.0, 1.0)
            ground_y = float(meta.get("ground_y_m", 0.0))
            world_vertices = scale * output.vertices + tr[:, None]
            sole_points = torch.stack((world_vertices[:, sole_ids[0]], world_vertices[:, sole_ids[1]]), dim=1)
            # Soft-min keeps gradients distributed over heel/toe points.
            sole_height = -.012 * torch.logsumexp(-sole_points[:, :, :, 1] / .012, dim=-1)
            contact_height_loss = ((sole_height - (ground_y + args.contact_height_offset)).square() * contact).sum() / contact.sum().clamp_min(1.0)
            if chunk_count > 1:
                sole_center = sole_points[:, :, :, (0, 2)].mean(dim=2)
                ankle_xz_delta = sole_center[1:] - sole_center[:-1]
                adjacent_contact = contact[1:] * contact[:-1]
                contact_velocity_loss = (ankle_xz_delta.square().sum(dim=-1) * adjacent_contact).sum() / adjacent_contact.sum().clamp_min(1.0)
            else:
                contact_velocity_loss = torch.zeros((), device=device)
            contact_loss = contact_height_loss + 0.25 * contact_velocity_loss
            self_collision_loss = torch.zeros((), dtype=torch.float32, device=device)
            if args.self_collision_weight > 0:
                # Approximate the torso by pelvis->spine3->neck capsules and
                # keep elbow/wrist centers outside it.  This smooth geometric
                # barrier is an inexpensive stand-in for the mesh
                # self-intersection penalty used by SMPLify-X.
                torso_points = world_joints[:, (0, 9, 12)]
                arms = world_joints[:, (18, 20, 19, 21)]
                barriers = []
                for arm_index, radius_scale in ((0, 1.2), (1, .8), (2, 1.2), (3, .8)):
                    point = arms[:, arm_index]
                    for segment in range(2):
                        a, b = torso_points[:, segment], torso_points[:, segment + 1]
                        direction = b - a
                        fraction = ((point - a) * direction).sum(dim=-1) / direction.square().sum(dim=-1).clamp_min(1e-6)
                        closest = a + fraction.clamp(0.0, 1.0)[:, None] * direction
                        clearance = torch.as_tensor(args.self_collision_radius * radius_scale, dtype=torch.float32, device=device)
                        barriers.append(torch.relu(clearance - torch.linalg.vector_norm(point - closest, dim=-1)).square())
                self_collision_loss = torch.stack(barriers, dim=1).mean()
            mesh_collision_loss = torch.zeros((), dtype=torch.float32, device=device)
            if args.mesh_collision_weight > 0:
                torso_surface = world_vertices[:, torso_ids]
                arm_surface = world_vertices[:, arm_ids]
                # For each sampled arm point, only its nearest torso point is
                # relevant.  The hinge is zero outside the clearance radius.
                nearest = torch.cdist(arm_surface, torso_surface).amin(dim=-1)
                mesh_collision_loss = torch.relu(args.mesh_collision_radius - nearest).square().mean()
                if general_surface is not None:
                    mesh_collision_loss = general_surface(world_vertices, args.mesh_collision_radius)
            legs_collision_loss = torch.zeros((), dtype=torch.float32, device=device)
            if args.mesh_collision_legs_weight > 0:
                nearest_leg = torch.cdist(world_vertices[:, lleg_ids], world_vertices[:, rleg_ids]).amin(dim=-1)
                legs_collision_loss = torch.relu(args.mesh_collision_radius - nearest_leg).square().mean()
            hull_collision_loss = torch.zeros((), dtype=torch.float32, device=device)
            if args.hull_collision_weight > 0 and hull_faces.numel() > 0:
                hull_points = world_vertices[:, arm_ids]
                hull_triangles = world_vertices[:, hull_vertex_ids[hull_faces]]
                if iteration == 0 or (args.hull_recompute_every > 0 and iteration % args.hull_recompute_every == 0):
                    with torch.no_grad():
                        inside_mask = _winding_numbers(hull_points.detach(), hull_triangles.detach()).abs() > 0.5
                if inside_mask is not None and bool(inside_mask.any()):
                    nearest_hull = torch.cdist(hull_points, world_vertices[:, hull_vertex_ids]).amin(dim=-1)
                    if args.hull_collision_mode == "push":
                        hull_collision_loss = ((nearest_hull + args.hull_collision_radius).square() * inside_mask).sum() / inside_mask.sum().clamp_min(1.0)
                    else:
                        hull_collision_loss = (torch.relu(args.hull_collision_radius - nearest_hull).square() * inside_mask).sum() / inside_mask.sum().clamp_min(1.0)
            # Apply the floor inequality to every sampled sole vertex.  A sum
            # (rather than a mean over the 64 vertices) keeps a few deeply
            # penetrating vertices from being numerically drowned out by the
            # many already-valid sole samples.
            floor_penetration_loss = torch.relu((ground_y + args.floor_clearance) - sole_points[..., :, 1]).square().sum(dim=-1).mean()
            global_floor_loss = torch.relu((ground_y + args.floor_clearance) - world_vertices[..., 1]).square().sum(dim=-1).mean()
            sole_flatness_loss = torch.zeros((), dtype=torch.float32, device=device)
            if args.sole_flatness_weight > 0:
                sole_y = sole_points[..., :, 1]
                sole_var = sole_y.var(dim=-1, unbiased=False)
                sole_flatness_loss = (sole_var * contact).sum() / contact.sum().clamp_min(1.0)
            joint_limit_loss = torch.zeros((), dtype=torch.float32, device=device)
            if args.joint_limit_weight > 0:

                knee = torch.stack((flexion(world_joints[:, 1], world_joints[:, 4], world_joints[:, 7]), flexion(world_joints[:, 2], world_joints[:, 5], world_joints[:, 8])), dim=-1)
                elbow = torch.stack((flexion(world_joints[:, 16], world_joints[:, 18], world_joints[:, 20]), flexion(world_joints[:, 17], world_joints[:, 19], world_joints[:, 21])), dim=-1)
                torso_vec = world_joints[:, 12] - world_joints[:, 0]
                torso_lean = inclination(torso_vec)
                knee_barrier = torch.relu(knee - np.deg2rad(args.max_knee_flexion_deg)).square().mean()
                elbow_barrier = torch.relu(elbow - np.deg2rad(args.max_elbow_flexion_deg)).square().mean()
                torso_barrier = torch.relu(torso_lean - np.deg2rad(args.torso_lean_limit_deg)).square().mean()
                joint_limit_loss = knee_barrier + elbow_barrier + torso_barrier
            knee_hyper_loss = torch.zeros((), dtype=torch.float32, device=device)
            if args.knee_hyperextension_weight > 0:
                # SMPL-X knee flexion is a positive rotation about the local x axis (body_pose joints 4,5 ->
                # indices 3,4).  The unsigned flexion hinge above cannot tell a backward-bent knee from a
                # bent one, so penalize negative flexion beyond a small anatomical tolerance directly.
                knee_x = bp.reshape(chunk_count, 21, 3)[:, 3:5, 0]
                knee_hyper_loss = torch.relu(-knee_x - np.deg2rad(args.knee_hyperextension_deg)).square().mean()
            pelvis_roll_loss = torch.zeros((), dtype=torch.float32, device=device)
            if args.pelvis_roll_weight > 0:
                # Lateral pelvis tilt: the vertical component of the pelvis' own left axis (column 0 of its
                # global rotation).  A forward bend rotates about that axis and leaves it horizontal, so this
                # term does not fight bends; it only discourages the sideways-tilted trunk that an IK branch
                # error produces during single-leg actions.
                pelvis_left_up = rotations[:, 0, 1, 0].abs()
                pelvis_roll_loss = torch.relu(pelvis_left_up - float(np.sin(np.deg2rad(args.pelvis_roll_limit_deg)))).square().mean()
            pose_ref_loss = torch.zeros((), dtype=torch.float32, device=device)
            if pose_ref is not None:
                ref_bp = torch.tensor(pose_ref[0][sl], device=device).reshape(chunk_count, 21, 3)
                ref_m = torch.tensor(pose_ref[1][sl], device=device)[:, None, None] * torch.tensor(pose_ref[2], device=device)[None, :, None]
                pose_ref_loss = ((bp.reshape(chunk_count, 21, 3) - ref_bp).square() * ref_m).sum() / (ref_m.sum() * 3.0).clamp_min(1.0)
            loss = args.joint_acceleration_weight * acceleration_loss + args.position_weight * pos_loss + args.rotation_weight * rot_loss + args.pose_prior_weight * prior + 5.0 * temporal + args.contact_weight * contact_loss + args.self_collision_weight * self_collision_loss + args.mesh_collision_weight * mesh_collision_loss + args.mesh_collision_legs_weight * legs_collision_loss + args.hull_collision_weight * hull_collision_loss + args.floor_penetration_weight * floor_penetration_loss + args.global_floor_penetration_weight * global_floor_loss + args.sole_flatness_weight * sole_flatness_loss + args.joint_limit_weight * joint_limit_loss + args.knee_hyperextension_weight * knee_hyper_loss + args.pelvis_roll_weight * pelvis_roll_loss + args.pose_reference_weight * pose_ref_loss
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"non-finite projection loss at chunk {chunk_start}:{chunk_end}, "
                    f"iteration {iteration + 1}; reduce barrier weights or learning rate"
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_((go, bp, tr), 2.0)
            optimizer.step()
            if args.max_step_pose_deg > 0 or args.max_step_translation > 0:
                with torch.no_grad():
                    if args.max_step_pose_deg > 0:
                        limit = float(np.deg2rad(args.max_step_pose_deg))
                        go.copy_(go_before + (go - go_before).clamp(-limit, limit))
                        bp.copy_(bp_before + (bp - bp_before).clamp(-limit, limit))
                    if args.max_step_translation > 0:
                        limit = float(args.max_step_translation)
                        tr.copy_(tr_before + (tr - tr_before).clamp(-limit, limit))
            if not (torch.isfinite(go).all() and torch.isfinite(bp).all() and torch.isfinite(tr).all()):
                raise FloatingPointError(
                    f"non-finite SMPL-X parameters at chunk {chunk_start}:{chunk_end}, "
                    f"iteration {iteration + 1}; reduce barrier weights or learning rate"
                )
            if iteration in (0, args.iterations - 1):
                print(f"frames {chunk_start}:{chunk_end} iteration {iteration + 1}: position={float(torch.sqrt(pos_loss).detach()) * 1000:.2f} mm rotation={float(torch.sqrt(rot_loss).detach()) * 180 / np.pi:.3f} deg")
        with torch.no_grad():
            local_start = write_start - chunk_start
            local_end = local_start + (write_end - write_start)
            core = slice(local_start, local_end)
            global_orient[write_start:write_end] = go[core].cpu().numpy()
            transl[write_start:write_end] = tr[core].cpu().numpy()
            body_pose[write_start:write_end] = bp[core].cpu().numpy()
    args.output.mkdir(parents=True, exist_ok=True)
    # Recompute the observed endpoint residuals at the final full-rate result.
    # Keeping this beside the trajectory makes the audit independent of the
    # optimizer's last minibatch and lets a hard-binding run be checked later.
    position_residual = np.full((count, 5, 3), np.nan, dtype=np.float32)
    predicted_tracker = np.full((count, 5, 3), np.nan, dtype=np.float32)
    rotation_error_rad = np.full((count, 5), np.nan, dtype=np.float32)
    body.eval()
    with torch.no_grad():
        for check_start in range(start, end, max(1, args.chunk_size)):
            check_end = min(end, check_start + max(1, args.chunk_size))
            check_slice = slice(check_start, check_end)
            check_count = check_end - check_start
            go_check = torch.as_tensor(global_orient[check_slice], dtype=torch.float32, device=device)
            bp_check = torch.as_tensor(body_pose[check_slice], dtype=torch.float32, device=device)
            tr_check = torch.as_tensor(transl[check_slice], dtype=torch.float32, device=device)
            zero3 = torch.zeros((check_count, 3), dtype=torch.float32, device=device)
            output = body(
                betas=beta_t[None].expand(check_count, -1), global_orient=go_check,
                transl=zero3, body_pose=bp_check.reshape(check_count, -1),
                left_hand_pose=torch.as_tensor(left_hand[check_slice], device=device),
                right_hand_pose=torch.as_tensor(right_hand[check_slice], device=device),
                jaw_pose=zero3, leye_pose=zero3, reye_pose=zero3,
                expression=torch.zeros((check_count, 10), device=device), return_verts=False,
            )
            rotations = _global_body_rotations(go_check, bp_check, parents)
            predicted = scale * output.joints[:, ids] + tr_check[:, None]
            eff_p, eff_r = effective_mounts()
            predicted = predicted + (rotations[:, ids] @ eff_p[None, :, :, None]).squeeze(-1)
            predicted_r = rotations[:, ids] @ eff_r[None]
            target_p_check = torch.as_tensor(target_pos[check_slice], dtype=torch.float32, device=device)
            target_r_check = torch.as_tensor(target_rot[check_slice], dtype=torch.float32, device=device)
            position_residual[check_slice] = (predicted - target_p_check).cpu().numpy()
            predicted_tracker[check_slice] = predicted.cpu().numpy()
            relative = torch.matmul(predicted_r.transpose(-1, -2), target_r_check)
            trace = relative.diagonal(dim1=-2, dim2=-1).sum(-1)
            rotation_error_rad[check_slice] = torch.acos(((trace - 1.0) * .5).clamp(-1.0, 1.0)).cpu().numpy()
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output / "tracker_diagnostics.npz",
        position_residual_m=position_residual,
        predicted_tracker=predicted_tracker,
        tracker_position=target_pos,
        tracker_rotation=target_rot,
        rotation_error_rad=rotation_error_rad,
        valid=valid,
    )
    # The five-tracker pipeline never reads a glove: hand pose is whatever the
    # proposal carried, which is a neutral all-zero MANO pose.  Keeping the
    # arrays (removing them would break existing readers) while stating their
    # provenance stops a consumer from mistaking zeros for a measurement.
    hand_measured = bool(np.any(left_hand) or np.any(right_hand))
    hand_pose_source = "measured" if hand_measured else "none_neutral_zero"
    np.savez_compressed(
        args.output / "smplx_profile_driven.npz", timestamps=timestamps, betas=betas,
        global_orient=global_orient, transl=transl, body_pose=body_pose.reshape(count, -1),
        scale=scale, left_hand_pose=left_hand, right_hand_pose=right_hand,
        attachment_position=(effective_mounts()[0].detach().cpu().numpy() if args.solve_attachments else mount_position), attachment_rotation=(effective_mounts()[1].detach().cpu().numpy() if args.solve_attachments else mount_rotation),
        hand_pose_source=np.array(hand_pose_source),
    )
    result_meta = dict(meta)
    result_meta["surface_mode"] = args.surface_mode
    result_meta["surface_sampling"] = general_surface.manifest if general_surface is not None else {"mode": "legacy arm-torso", "samples_per_region": args.mesh_sample_count}
    result_meta["joint_acceleration_weight"] = args.joint_acceleration_weight
    result_meta["solve_attachments"] = bool(args.solve_attachments)
    if args.solve_attachments:
        result_meta["solved_attachment_delta_position_m"] = mount_dp.detach().cpu().numpy().tolist(); result_meta["solved_attachment_delta_rotvec"] = mount_dr.detach().cpu().numpy().tolist()
    result_meta["joint_acceleration_definition"] = "world first 22 body joints; irregular timestamps; valid triplets; gaps <= 100 ms; mean square m/s^2"
    result_meta.update({"method": "learned sparse-tracker proposal + differentiable FK/IK projection + contact + arm/torso collision barrier", "projection_range": [start, end], "projection_chunk_size": args.chunk_size, "projection_chunk_overlap": int(overlap), "projection_iterations": args.iterations, "projection_position_weight": args.position_weight, "projection_rotation_weight": args.rotation_weight, "projection_pose_prior_weight": args.pose_prior_weight, "projection_contact_weight": args.contact_weight, "contact_height_offset_m": args.contact_height_offset, "self_collision_weight": args.self_collision_weight, "self_collision_radius_m": args.self_collision_radius, "mesh_collision_weight": args.mesh_collision_weight, "mesh_collision_radius_m": args.mesh_collision_radius, "mesh_sample_count": int(args.mesh_sample_count), "hull_collision_weight": args.hull_collision_weight, "hull_collision_radius_m": args.hull_collision_radius, "hull_recompute_every": args.hull_recompute_every, "floor_penetration_weight": args.floor_penetration_weight, "global_floor_penetration_weight": args.global_floor_penetration_weight, "sole_flatness_weight": args.sole_flatness_weight, "joint_limit_weight": args.joint_limit_weight, "max_knee_flexion_deg": args.max_knee_flexion_deg, "max_elbow_flexion_deg": args.max_elbow_flexion_deg, "torso_lean_limit_deg": args.torso_lean_limit_deg, "knee_hyperextension_weight": args.knee_hyperextension_weight, "knee_hyperextension_deg": args.knee_hyperextension_deg, "pelvis_roll_weight": args.pelvis_roll_weight, "pelvis_roll_limit_deg": args.pelvis_roll_limit_deg, "pose_reference": str(args.pose_reference) if args.pose_reference else None, "pose_reference_weight": args.pose_reference_weight, "diagnostics": "tracker_diagnostics.npz", "hand_pose_source": hand_pose_source, "hand_pose_warning": (None if hand_measured else "left_hand_pose/right_hand_pose 为 SMPL-X 中性零姿态，不是测量值；本管线只消费五个 tracker，未读取任何手套数据")})
    (args.output / "smplx_profile_driven.json").write_text(json.dumps(result_meta, ensure_ascii=False, indent=2) + "\n")
    print(f"wrote {args.output / 'smplx_profile_driven.npz'}")


if __name__ == "__main__":
    main()
