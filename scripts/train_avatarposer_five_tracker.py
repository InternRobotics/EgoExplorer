#!/usr/bin/env python3
"""Fine-tune the official AvatarPoser Transformer for five SMPL-X trackers.

Keeps AvatarPoser's 3-layer Transformer and decoder, copies its first 54 input
channels / output weights and adds the two ankle trackers (90 channels).  Trains on
exact SMPL-X synthetic sequences (--synthetic-root) mixed with teacher-labelled natural
recordings (--natural-root); the released checkpoint uses both (guidelines, section 7.7).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from tracker_streams import RAW_ROLES, wxyz_to_matrix as _wxyz_to_matrix


def sixd(m):
    return np.concatenate((m[..., :, 0], m[..., :, 1]), axis=-1)


def pack_five_features(rotation, rotation_velocity, position, position_velocity):
    """Pack five trackers while preserving AvatarPoser's official 54 channels.

    The pretrained model expects head/left-hand/right-hand blocks first, in
    the order rotation, rotation velocity, position, position velocity.  The
    two foot blocks are appended afterwards (36 channels).
    """
    upper = np.concatenate((
        sixd(rotation[:, :3]).reshape(len(rotation), -1),
        sixd(rotation_velocity[:, :3]).reshape(len(rotation), -1),
        position[:, :3].reshape(len(rotation), -1),
        position_velocity[:, :3].reshape(len(rotation), -1),
    ), axis=-1)
    feet = np.concatenate((
        sixd(rotation[:, 3:]).reshape(len(rotation), -1),
        sixd(rotation_velocity[:, 3:]).reshape(len(rotation), -1),
        position[:, 3:].reshape(len(rotation), -1),
        position_velocity[:, 3:].reshape(len(rotation), -1),
    ), axis=-1)
    return np.concatenate((upper, feet), axis=-1)


def _sixd_to_matrix(v):
    """Decode the continuous 6D rotation representation used by AvatarPoser."""
    a = np.asarray(v, dtype=np.float32)[..., :3]
    b = np.asarray(v, dtype=np.float32)[..., 3:]
    a = a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-8)
    b = b - np.sum(a * b, axis=-1, keepdims=True) * a
    b = b / np.maximum(np.linalg.norm(b, axis=-1, keepdims=True), 1e-8)
    return np.stack((a, b, np.cross(a, b)), axis=-1)


def distal_isolation_augmentation(x, y, *, side: int, angle_deg: float):
    """Perturb one wrist's orientation while leaving torso/limb centers fixed.

    This is the synthetic hard-negative used by SparseTrack's ``locked elbow,
    rotating wrist`` idea.  The wrist local rotation target is changed with
    the tracker orientation, while positions and every proximal joint remain
    unchanged.  It teaches the network not to explain a distal twist by
    rotating the shoulder or torso.
    """
    x_aug = np.asarray(x, dtype=np.float32).copy()
    y_aug = np.asarray(y, dtype=np.float32).copy()
    # The feature vector is packed as official upper-body blocks followed by
    # foot blocks; unpack the rotation groups before applying the perturbation.
    tracker_rot = np.concatenate((
        _sixd_to_matrix(x_aug[:, :18].reshape(len(x_aug), 3, 6)),
        _sixd_to_matrix(x_aug[:, 54:66].reshape(len(x_aug), 2, 6)),
    ), axis=1)
    pose_rot = _sixd_to_matrix(y_aug[:, :132].reshape(len(y_aug), 22, 6))
    axis = np.asarray((0.0, 1.0, 0.0), dtype=np.float32)
    angle = np.deg2rad(float(angle_deg))
    c, s = np.cos(angle), np.sin(angle)
    q = np.asarray(((c, 0.0, s), (0.0, 1.0, 0.0), (-s, 0.0, c)), dtype=np.float32)
    tracker_rot[:, side] = tracker_rot[:, side] @ q
    # Tracker index 1/2 corresponds to SMPL-X wrist joint 20/21.  Applying
    # the same local rotation keeps the synthetic observation and target
    # kinematics consistent up to the fixed mount transform.
    pose_rot[:, 19 + side] = pose_rot[:, 19 + side] @ q
    dr = np.concatenate(
        (np.broadcast_to(np.eye(3, dtype=np.float32), tracker_rot[:1].shape),
         tracker_rot[:-1].transpose(0, 1, 3, 2) @ tracker_rot[1:]), axis=0,
    )
    rotation_velocity = np.concatenate((
        _sixd_to_matrix(x_aug[:, 18:36].reshape(len(x_aug), 3, 6)),
        _sixd_to_matrix(x_aug[:, 66:78].reshape(len(x_aug), 2, 6)),
    ), axis=1)
    position = np.concatenate((x_aug[:, 36:45].reshape(len(x_aug), 3, 3), x_aug[:, 78:84].reshape(len(x_aug), 2, 3)), axis=1)
    position_velocity = np.concatenate((x_aug[:, 45:54].reshape(len(x_aug), 3, 3), x_aug[:, 84:90].reshape(len(x_aug), 2, 3)), axis=1)
    x_aug[:] = pack_five_features(tracker_rot, rotation_velocity, position, position_velocity)
    y_aug[:, :132] = sixd(pose_rot).reshape(len(y_aug), -1)
    return x_aug, y_aug


def tracker_mount_noise_augmentation(x, y, rng, *, position_std=0.018, angle_std_deg=8.0):
    """Apply fixed per-tracker mount and small measurement perturbations."""
    x_aug = np.asarray(x, dtype=np.float32).copy()
    rot = np.concatenate((_sixd_to_matrix(x_aug[:, :18].reshape(len(x_aug), 3, 6)),
                          _sixd_to_matrix(x_aug[:, 54:66].reshape(len(x_aug), 2, 6))), axis=1)
    pos = np.concatenate((x_aug[:, 36:45].reshape(len(x_aug), 3, 3),
                          x_aug[:, 78:84].reshape(len(x_aug), 2, 3)), axis=1)
    offset = rng.normal(0.0, float(position_std), size=(5, 3)).astype(np.float32)
    axes = rng.normal(size=(5, 3)); axes /= np.maximum(np.linalg.norm(axes, axis=-1, keepdims=True), 1e-8)
    angles = rng.normal(0.0, np.deg2rad(float(angle_std_deg)), size=5)
    mount_rot = Rotation.from_rotvec(axes * angles[:, None]).as_matrix().astype(np.float32)
    rot = np.einsum('nij,tnjk->tnik', mount_rot, rot)
    pos = pos + offset[None] + rng.normal(0.0, float(position_std) * .15, size=pos.shape).astype(np.float32)
    dr = np.concatenate((np.broadcast_to(np.eye(3, dtype=np.float32), rot[:1].shape), rot[:-1].transpose(0, 1, 3, 2) @ rot[1:]), axis=0)
    dp = np.diff(pos, axis=0, prepend=pos[:1])
    return pack_five_features(rot, dr, pos, dp), np.asarray(y, dtype=np.float32).copy()


def features_and_target(path: Path, *, root_local: bool = False):
    with np.load(path) as z:
        p = np.asarray(z["tracker_positions"], np.float32)
        r = np.asarray(z["tracker_rotations"], np.float32)
        root = np.asarray(z["global_orient"], np.float32)
        body = np.asarray(z["body_pose"], np.float32).reshape(-1, 21, 3)
        contact = np.asarray(z["contact"], np.float32)
    q = Rotation.from_euler("x", 90, degrees=True).as_matrix().astype(np.float32)
    p = np.einsum("ij,tnj->tni", q, p - p[0, 0])
    r = q @ r
    original_r = r.copy()
    if root_local:
        head_r = r[:, :1]
        r = np.einsum("tnij,tnjk->tnik", head_r.transpose(0, 1, 3, 2), r)
        p = np.einsum("tnij,tnj->tni", head_r.transpose(0, 1, 3, 2), p - p[:, :1])
    dr = np.concatenate((np.broadcast_to(np.eye(3, dtype=np.float32), r[:1].shape), r[:-1].transpose(0, 1, 3, 2) @ r[1:]), axis=0)
    dp = np.diff(p, axis=0, prepend=p[:1])
    x = pack_five_features(r, dr, p, dp)
    root_r = q @ Rotation.from_rotvec(root).as_matrix().astype(np.float32)
    if root_local:
        # ``r`` now contains head-relative rotations.  Express the target
        # root in that same head frame.
        # ``original_r`` is already in the q-transformed world frame.
        head_q = original_r[:, 0]
        root_r = np.einsum("tij,tjk->tik", head_q.transpose(0, 2, 1), root_r)
    y = np.concatenate((sixd(np.concatenate((root_r[:, None], Rotation.from_rotvec(body.reshape(-1, 3)).as_matrix().reshape(-1, 21, 3, 3)), axis=1)).reshape(len(p), -1), contact), axis=-1)
    return x.astype(np.float32), y.astype(np.float32)


def natural_features_and_target(path: Path, mount_reference: tuple[np.ndarray, np.ndarray] | None = None, *, root_local: bool = False):
    """Convert one recorder/output pair to the same 60 Hz AMASS-like tensor."""
    meta = json.loads(path.with_suffix(".json").read_text())
    with np.load(path, allow_pickle=False) as z:
        ts = np.asarray(z["timestamps"], np.float64)
        root = np.asarray(z["global_orient"], np.float32)
        body = np.asarray(z["body_pose"], np.float32).reshape(-1, 21, 3)
        betas = np.asarray(z["betas"], np.float32)
        if mount_reference is None:
            mount_p = np.asarray(z["attachment_position"], np.float32)
            mount_r = np.asarray(z["attachment_rotation"], np.float32)
        else:
            mount_p, mount_r = mount_reference
    with np.load(meta["source"], allow_pickle=False) as raw:
        p = np.stack([raw[f"tracker_0__pose.{role}.position"][:len(ts)] for role in RAW_ROLES], axis=1).astype(np.float32)
        r = np.stack([_wxyz_to_matrix(raw[f"tracker_0__pose.{role}.orientation"][:len(ts)]) for role in RAW_ROLES], axis=1)
    good = np.isfinite(p).all(axis=-1) & np.isfinite(r).all(axis=(-1, -2))
    for j in range(5):
        valid = good[:, j]
        if valid.sum() < 2:
            return None
        p[:, j] = np.stack([np.interp(ts, ts[valid], p[valid, j, a]) for a in range(3)], axis=-1)
        r[:, j] = Slerp(ts[valid], Rotation.from_matrix(r[valid, j]))(np.clip(ts, ts[valid][0], ts[valid][-1])).as_matrix()
    joint_r = r @ mount_r.transpose(0, 2, 1)[None]
    joint_p = p - (joint_r @ mount_p[None, :, :, None]).squeeze(-1)
    q = Rotation.from_euler("x", 90, degrees=True).as_matrix().astype(np.float32)
    ts = ts - ts[0]
    # Keep the original pose labels until the tracker stream has been
    # resampled.  Natural recordings are often ~15 Hz while the network
    # operates at 60 Hz; root-local conversion must use the same timestamps.
    root_ts = Rotation.from_rotvec(root).as_matrix().astype(np.float32)
    body_ts = Rotation.from_rotvec(body.reshape(-1, 3)).as_matrix().reshape(-1, 21, 3, 3).astype(np.float32)
    # The recorded rate is irregular and often ~15 Hz.  Reconstructing the
    # 60 Hz representation makes the pretrained temporal receptive field
    # (40 frames) have the same physical duration as in AvatarPoser.
    ut = np.unique(np.append(np.arange(0.0, ts[-1], 1.0 / 60.0), ts[-1]))
    pp = np.einsum("ij,tnj->tni", q, joint_p - np.array((joint_p[0, 0, 0], meta.get("ground_y_m", joint_p[0, 0, 1]), joint_p[0, 0, 2]), np.float32))
    rr = q @ joint_r
    up = np.stack([np.stack([np.interp(ut, ts, pp[:, j, a]) for a in range(3)], -1) for j in range(5)], 1)
    ur = np.stack([Slerp(ts, Rotation.from_matrix(rr[:, j]))(ut).as_matrix() for j in range(5)], 1)
    head_ur = ur[:, :1].copy()
    if root_local:
        ur = np.einsum("tnij,tnjk->tnik", head_ur.transpose(0, 1, 3, 2), ur)
        up = np.einsum("tnij,tnj->tni", head_ur.transpose(0, 1, 3, 2), up - up[:, :1])
    dr = np.concatenate((np.broadcast_to(np.eye(3, dtype=np.float32), ur[:1].shape), ur[:-1].transpose(0, 1, 3, 2) @ ur[1:]), 0)
    dp = np.diff(up, axis=0, prepend=up[:1])
    x = pack_five_features(ur, dr, up, dp)
    root_r = q @ Slerp(ts, Rotation.from_matrix(root_ts))(ut).as_matrix().astype(np.float32)
    if root_local:
        root_r = np.einsum("tnij,tnjk->tnik", head_ur.transpose(0, 1, 3, 2), root_r[:, None])[:, 0]
    body_r = np.stack([Slerp(ts, Rotation.from_matrix(body_ts[:, j]))(ut).as_matrix() for j in range(21)], axis=1).astype(np.float32)
    # Derive conservative contact labels from resampled ankle tracker joints.
    ankle_y = up[:, 3:, 1]
    ankle_speed = np.linalg.norm(np.diff(up[:, 3:, :], axis=0, prepend=up[:1, 3:, :]), axis=-1) * 60.0
    baseline = np.percentile(ankle_y, 12.0, axis=0, keepdims=True)
    contact = ((ankle_y <= baseline + 0.035) & (ankle_speed <= 0.20)).astype(np.float32)
    yy = np.concatenate((sixd(np.concatenate((root_r[:, None], body_r), 1)).reshape(len(ut), -1), contact), -1)
    y = yy
    return x.astype(np.float32), y.astype(np.float32), betas


def make_model(checkpoint: Path | None, device: str):
    import torch
    from torch import nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear_embedding = nn.Linear(90, 256)
            self.transformer_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(256, nhead=8), num_layers=3)
            self.stabilizer = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 6))
            self.joint_rotation_decoder = nn.Sequential(nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 126))
            self.contact = nn.Linear(256, 2)
        def forward(self, x):
            h = self.transformer_encoder(self.linear_embedding(x).permute(1, 0, 2)).permute(1, 0, 2)[:, -1]
            return torch.cat((self.stabilizer(h), self.joint_rotation_decoder(h), self.contact(h)), dim=-1)

    model = Net().to(device)
    if checkpoint is not None:
        state = torch.load(checkpoint, map_location=device, weights_only=True)
        # Official AvatarPoser checkpoints are bare state dicts, while this
        # trainer stores them under ``model``.  Accept both so public-data
        # pretraining can be followed by local-domain adaptation.
        if isinstance(state, dict) and "model" in state and isinstance(state["model"], dict):
            state = state["model"]
        own = model.state_dict()
        for name in ("transformer_encoder", "stabilizer", "joint_rotation_decoder"):
            for key, value in state.items():
                if key.startswith(name + ".") and key in own and own[key].shape == value.shape:
                    own[key] = value
        # Official channels are head/left-hand/right-hand in exactly the
        # first 54 slots of our 90-channel representation.  When continuing
        # from one of our own five-tracker checkpoints, retain all 90 slots.
        source_embedding = state["linear_embedding.weight"]
        if source_embedding.shape[1] == 90:
            own["linear_embedding.weight"] = source_embedding
        else:
            own["linear_embedding.weight"][:, :54] = source_embedding
            # The added foot channels must not perturb the official
            # three-tracker function at initialization.
            own["linear_embedding.weight"][:, 54:] = 0.0
        own["linear_embedding.bias"] = state["linear_embedding.bias"]
        model.load_state_dict(own)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic-root", type=Path, required=True)
    parser.add_argument("--natural-root", type=Path, default=None, help="optional natural SMPL-X outputs with source tracker streams")
    parser.add_argument("--mount-reference", type=Path, default=None, help="SMPL-X NPZ whose attachment fields standardize every natural teacher")
    parser.add_argument("--holdout-token", default="20260817_104936")
    parser.add_argument("--max-natural-windows", type=int, default=6000)
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/external_models/avatarposer.pth"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window", type=int, default=40)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--hard-negative-ratio", type=float, default=0.0, help="fraction of synthetic sequences duplicated with isolated wrist twists")
    parser.add_argument("--hard-negative-angle-deg", type=float, default=25.0, help="isolated wrist twist angle used for synthetic hard negatives")
    parser.add_argument("--mount-noise-ratio", type=float, default=0.0)
    parser.add_argument("--mount-position-std", type=float, default=0.018)
    parser.add_argument("--mount-angle-std-deg", type=float, default=8.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--root-local", action="store_true", help="express tracker features and root target in the head-local frame")
    parser.add_argument("--three-anchor", action="store_true", help="baseline: zero the two foot-tracker blocks (channels 54:90) so the official 3-anchor AvatarPoser is fine-tuned on the same mount-corrected data")
    args = parser.parse_args()
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    all_x, all_y = [], []
    hard_negative_sequences = 0
    rng = np.random.default_rng(20260905)
    for path in sorted(args.synthetic_root.glob("sequence_*.npz")):
        x, y = features_and_target(path, root_local=args.root_local)
        all_x.extend(x[i - args.window + 1:i + 1] for i in range(args.window - 1, len(x)))
        all_y.extend(y[i] for i in range(args.window - 1, len(y)))
        if args.hard_negative_ratio > 0 and rng.random() < args.hard_negative_ratio:
            side = int(rng.integers(1, 3))
            angle = float(args.hard_negative_angle_deg) * (1.0 if rng.random() < .5 else -1.0)
            xn, yn = distal_isolation_augmentation(
                x, y, side=side, angle_deg=angle,
            )
            all_x.extend(xn[i - args.window + 1:i + 1] for i in range(args.window - 1, len(xn)))
            all_y.extend(yn[i] for i in range(args.window - 1, len(yn)))
            hard_negative_sequences += 1
        if args.mount_noise_ratio > 0 and rng.random() < args.mount_noise_ratio:
            xn, yn = tracker_mount_noise_augmentation(
                x, y, rng, position_std=args.mount_position_std,
                angle_std_deg=args.mount_angle_std_deg,
            )
            all_x.extend(xn[i - args.window + 1:i + 1] for i in range(args.window - 1, len(xn)))
            all_y.extend(yn[i] for i in range(args.window - 1, len(yn)))
    natural_count = 0
    mount_reference = None
    if args.mount_reference is not None:
        with np.load(args.mount_reference, allow_pickle=False) as z:
            mount_reference = (
                np.asarray(z["attachment_position"], np.float32).copy(),
                np.asarray(z["attachment_rotation"], np.float32).copy(),
            )
    if args.natural_root is not None:
        natural_paths = sorted(args.natural_root.glob("*/smplx_profile_driven.npz"))
        for path in natural_paths:
            if args.holdout_token and args.holdout_token in path.parent.name:
                continue
            converted = natural_features_and_target(path, mount_reference=mount_reference, root_local=args.root_local)
            if converted is None:
                continue
            x_n, y_n, _ = converted
            starts = np.linspace(args.window - 1, len(x_n) - 1, min(args.max_natural_windows // max(1, len(natural_paths)), len(x_n) - args.window + 1), dtype=np.int64)
            all_x.extend(x_n[i - args.window + 1:i + 1] for i in starts)
            all_y.extend(y_n[i] for i in starts)
            natural_count += 1
    if not all_x:
        raise SystemExit("no synthetic sequences")
    x = torch.from_numpy(np.stack(all_x)); y = torch.from_numpy(np.stack(all_y))
    loader = DataLoader(TensorDataset(x, y), batch_size=args.batch_size, shuffle=True)
    if args.three_anchor:
        x[..., 54:] = 0.0
        print("three-anchor mode: foot tracker channels zeroed", flush=True)
    model = make_model(args.checkpoint, args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    history = []
    for epoch in range(args.epochs):
        model.train(); losses = []
        for xb, yb in loader:
            pred = model(xb.to(args.device)); pose = torch.nn.functional.smooth_l1_loss(pred[:, :132], yb[:, :132].to(args.device))
            contact = torch.nn.functional.binary_cross_entropy_with_logits(pred[:, 132:], yb[:, 132:].to(args.device))
            loss = pose + .25 * contact
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); losses.append(float(loss.detach()))
        history.append(float(np.mean(losses))); print(f"epoch {epoch + 1}/{args.epochs}: loss={history[-1]:.6f}", flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "input_dim": 90, "output_dim": 134, "three_anchor": bool(args.three_anchor), "window": args.window, "source_checkpoint": str(args.checkpoint), "source_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()}, args.output / "model.pt")
    (args.output / "report.json").write_text(json.dumps({"method": "AvatarPoser Transformer fine-tuned for five tracker SMPL-X", "synthetic_sequences": len(list(args.synthetic_root.glob("sequence_*.npz"))), "hard_negative_sequences": hard_negative_sequences, "hard_negative_ratio": float(args.hard_negative_ratio), "hard_negative_angle_deg": float(args.hard_negative_angle_deg), "mount_noise_ratio": float(args.mount_noise_ratio), "mount_position_std": float(args.mount_position_std), "mount_angle_std_deg": float(args.mount_angle_std_deg), "natural_sequences": natural_count, "windows": len(x), "history": history, "source_checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(), "mount_reference": str(args.mount_reference) if args.mount_reference else None, "warning": "natural labels are provisional; requires mocap/video validation"}, indent=2) + "\n")


if __name__ == "__main__":
    main()
