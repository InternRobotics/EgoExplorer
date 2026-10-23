"""SMPL-X + VPoser fitting for five 6-DoF Vive trackers.

The initial stationary recording is used as a *reference attachment* calibration.
This is important for this hardware layout: there is no pelvis tracker and the
exact location/orientation of each tracker on a glove, shoe or headset is not
known beforehand.  The method therefore preserves the measured rigid transform
of each tracker relative to its assigned SMPL-X joint at the reference pose and
fits the subsequent relative motion.  It cannot repair tracker streams that
were expressed in different world frames before this calibration.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .calibration import average_neutral_frame
from .device import select_device
from .model import ModelPaths, load_body_model, load_vposer
from .types import Calibration, SmplTrajectory, TrackerFrame


# SMPL/SMPL-X body-joint indices.  We deliberately track ankles, rather than
# feet: the recorded trackers are mounted on the lateral ankle.
_TRACKED_JOINTS = {
    "head": 15,
    "left_hand": 20,
    "right_hand": 21,
    "left_foot": 7,
    "right_foot": 8,
}
_TRACKER_NAMES = tuple(_TRACKED_JOINTS)


@dataclass(frozen=True)
class FitOptions:
    device: str = "auto"
    window_size: int = 60
    max_iterations: int = 80
    sample_stride: int = 6
    tracker_position_weight: float = 30.0
    tracker_rotation_weight: float = 0.3
    temporal_weight: float = 2.0
    boundary_weight: float = 30.0
    root_temporal_weight: float = 8.0
    root_boundary_weight: float = 10.0
    smoothing_window: int = 5
    vposer_weight: float = 0.02
    calibration_iterations: int = 700
    shared_refinement_iterations: int = 140
    shared_refinement_batch_size: int = 90
    torso_upright_weight: float = 8.0
    model_type: str = "smplx"
    gender: str = "neutral"
    body_model_path: str = "models/smplx"
    vposer_path: str = "models/vposer/V02_05"
    verbose: bool = False


def _axis_angle_to_matrix(axis_angle):
    """Rodrigues map for a tensor shaped ``[..., 3]``."""
    import torch

    angle = torch.linalg.vector_norm(axis_angle, dim=-1, keepdim=True)
    x, y, z = axis_angle.unbind(dim=-1)
    zero = torch.zeros_like(x)
    skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), dim=-1).reshape(*axis_angle.shape[:-1], 3, 3)
    eye = torch.eye(3, dtype=axis_angle.dtype, device=axis_angle.device).expand_as(skew)
    angle2 = angle.square()
    # Clamp denominators before ``where``: PyTorch evaluates both branches, and
    # a literal 0/0 would otherwise poison gradients at the identity rotation.
    a = torch.where(angle > 1e-5, torch.sin(angle) / angle.clamp_min(1e-8), 1.0 - angle2 / 6.0)
    b = torch.where(angle > 1e-5, (1.0 - torch.cos(angle)) / angle2.clamp_min(1e-8), 0.5 - angle2 / 24.0)
    return eye + a.unsqueeze(-1) * skew + b.unsqueeze(-1) * (skew @ skew)


def _global_body_rotations(global_orient, body_pose, parents):
    """Forward-kinematics rotations for SMPL-X joints 0..21."""
    import torch

    local = _axis_angle_to_matrix(torch.cat((global_orient[:, None], body_pose), dim=1))
    rotations = [local[:, 0]]
    for index in range(1, 22):
        rotations.append(rotations[int(parents[index])] @ local[:, index])
    return torch.stack(rotations, dim=1)


def _rotation_distance_squared(predicted, target):
    import torch

    relative = predicted.transpose(-1, -2) @ target
    trace = relative.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    cosine = ((trace - 1.0) * 0.5).clamp(-1.0 + 1e-6, 1.0 - 1e-6)
    return torch.acos(cosine).square()


def _huber(values, delta: float):
    """Robust coordinate loss that does not let one bad tracker frame win."""
    import torch

    magnitude = values.abs()
    threshold = torch.as_tensor(delta, dtype=values.dtype, device=values.device)
    return torch.where(magnitude <= threshold, 0.5 * values.square(), threshold * (magnitude - 0.5 * threshold))


def _smooth_trajectory_values(values: np.ndarray, window: int) -> np.ndarray:
    """Offline triangular smoothing for sampled SMPL-X trajectory parameters."""
    if window < 3 or len(values) < window:
        return values
    if window % 2 == 0:
        window += 1
    half = window // 2
    weights = np.arange(1, half + 2, dtype=np.float32)
    weights = np.concatenate((weights, weights[-2::-1]))
    weights /= weights.sum()
    padded = np.pad(values, ((half, half), (0, 0)), mode="edge")
    return np.stack([
        np.convolve(padded[:, column], weights, mode="valid")
        for column in range(values.shape[1])
    ], axis=1).astype(np.float32)


class SmplxVPoserFitter:
    """Fit a downsampled SMPL-X trajectory, preserving tracker attachments.

    ``sample_stride=6`` turns the supplied approximately 60 Hz recording into
    a 10 Hz initial reconstruction.  It keeps optimization practical on macOS
    CPU, while retaining timestamps; later resampling/export can be added
    without changing the fitted coordinate system.
    """

    def __init__(self, *, calibration: Calibration, options: FitOptions | None = None) -> None:
        self.calibration = calibration
        self.options = options or FitOptions()

    def _reference(self, body, vposer, vposer_zero, pose_template, device):
        """Jointly fit the static standing frame and bounded mount offsets.

        The raw recorder stream gives global tracker poses.  A standing,
        arms-forward calibration frame therefore lets us solve the initial
        body pose and fixed segment-to-tracker translations together, rather
        than incorrectly absorbing an arm pose error into a large attachment.
        """
        import torch

        neutral = self.calibration.neutral
        dtype = torch.float32
        joint_ids = torch.as_tensor([_TRACKED_JOINTS[name] for name in _TRACKER_NAMES], device=device)
        ref_positions = torch.as_tensor(np.stack([neutral[name].position for name in _TRACKER_NAMES]), dtype=dtype, device=device)
        latent = torch.nn.Parameter(torch.zeros((1, 32), dtype=dtype, device=device))
        global_orient = torch.nn.Parameter(torch.zeros((1, 3), dtype=dtype, device=device))
        # Initialise the root beneath the tracked head; the optimizer is free
        # to correct this from all five targets.
        translation = torch.nn.Parameter((ref_positions[0] - torch.tensor((0.0, 1.55, 0.0), dtype=dtype, device=device))[None])
        # SMPL-X rest axes use +X toward anatomical left. The ankle trackers
        # are tied to the outer sides of the lower legs, so seed and softly
        # retain that lateral placement instead of letting IK interpret them
        # as a toe/instep mount.
        attachment_centers = torch.as_tensor(
            ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.12, 0.0, 0.0), (-0.12, 0.0, 0.0)),
            dtype=dtype, device=device,
        )
        attachment_position = torch.nn.Parameter(attachment_centers.clone())
        limits = torch.as_tensor((0.15, 0.14, 0.14, 0.18, 0.18), dtype=dtype, device=device)
        optimizer = torch.optim.Adam((latent, global_orient, translation, attachment_position), lr=0.03)
        parents = body.parents.to(device)
        hands = torch.zeros((1, 45), dtype=dtype, device=device)
        for _ in range(self.options.calibration_iterations):
            optimizer.zero_grad(set_to_none=True)
            body_pose = vposer.decode(latent)["pose_body"] - vposer_zero + pose_template
            output = body(
                global_orient=global_orient, body_pose=body_pose.reshape(1, -1), transl=translation,
                left_hand_pose=hands, right_hand_pose=hands, return_verts=False,
            )
            rotations = _global_body_rotations(global_orient, body_pose, parents)[0, joint_ids]
            predicted = output.joints[0, joint_ids] + (rotations @ attachment_position[:, :, None]).squeeze(-1)
            position_loss = (predicted - ref_positions).square().mean()
            # Physical mounts are centimetre-scale.  A soft bound retains
            # enough freedom for the headband/side-ankle cases while refusing
            # the previous 0.4--1.0 m fake attachments.
            attachment_bound = torch.relu(torch.linalg.vector_norm(attachment_position, dim=1) - limits).square().mean()
            ankle_side_prior = (attachment_position[3:] - attachment_centers[3:]).square().mean()
            # The initial calibration pose has both palms down.  The tracker
            # cases are mounted on the backs of the hands, hence their centres
            # should sit a few centimetres above the anatomical wrists rather
            # than on the palm/forearm side.  This is a soft physical prior;
            # the measured positions retain the final say.
            hand_world_offset = (rotations[1:3] @ attachment_position[1:3, :, None]).squeeze(-1)
            hand_back_prior = (hand_world_offset - torch.tensor((0.0, 0.085, 0.0), dtype=dtype, device=device)).square().mean()
            upright_loss = global_orient[:, (0, 2)].square().mean()
            loss = 100.0 * position_loss + 300.0 * attachment_bound + 25.0 * ankle_side_prior + 30.0 * hand_back_prior + 0.20 * latent.square().mean() + 10.0 * upright_loss
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            body_pose = vposer.decode(latent)["pose_body"] - vposer_zero + pose_template
            output = body(
                global_orient=global_orient, body_pose=body_pose.reshape(1, -1), transl=translation,
                left_hand_pose=hands, right_hand_pose=hands, return_verts=False,
            )
            rotations = _global_body_rotations(global_orient, body_pose, parents)[0, joint_ids]
            predicted = output.joints[0, joint_ids] + (rotations @ attachment_position[:, :, None]).squeeze(-1)
            rms = torch.sqrt((predicted - ref_positions).square().mean()).item()
        ref_tracker_rotations = torch.as_tensor(np.stack([neutral[name].rotation for name in _TRACKER_NAMES]), dtype=dtype, device=device)
        # Source convention: tracker_world = joint_world @ tracker_in_joint.
        # Therefore tracker_in_joint = joint_world.T @ tracker_world.
        local_rotation = rotations.transpose(-1, -2) @ ref_tracker_rotations
        if self.options.verbose:
            mount_norms = attachment_position.detach().norm(dim=1).cpu().numpy()
            print(f"static calibration: tracker RMS={rms:.4f} m; attachment norms={np.round(mount_norms, 3).tolist()}")
        return (
            latent.detach(), global_orient.detach(), translation.detach(),
        ), attachment_position.detach(), local_rotation.detach()

    def _fit_window(self, body, vposer, vposer_zero, pose_template, targets_pos, targets_rot, state, attachment_position, attachment_rotation, parents):
        import torch

        count = targets_pos.shape[0]
        device = targets_pos.device
        dtype = targets_pos.dtype
        latent = torch.nn.Parameter(state[0].clone())
        global_orient = torch.nn.Parameter(state[1].clone())
        translation = torch.nn.Parameter(state[2].clone())
        optimizer = torch.optim.Adam((latent, global_orient, translation), lr=0.035)
        joint_ids = torch.as_tensor([_TRACKED_JOINTS[name] for name in _TRACKER_NAMES], device=device)
        for _ in range(self.options.max_iterations):
            optimizer.zero_grad(set_to_none=True)
            body_pose = vposer.decode(latent)["pose_body"] - vposer_zero + pose_template
            hands = torch.zeros((count, 45), dtype=dtype, device=device)
            output = body(
                global_orient=global_orient, body_pose=body_pose.reshape(count, -1), transl=translation,
                left_hand_pose=hands, right_hand_pose=hands,
                jaw_pose=torch.zeros((count, 3), dtype=dtype, device=device),
                leye_pose=torch.zeros((count, 3), dtype=dtype, device=device),
                reye_pose=torch.zeros((count, 3), dtype=dtype, device=device),
                expression=torch.zeros((count, 10), dtype=dtype, device=device), return_verts=False,
            )
            joint_rotations = _global_body_rotations(global_orient, body_pose, parents)[:, joint_ids]
            predicted_pos = output.joints[:, joint_ids] + (joint_rotations @ attachment_position[None, :, :, None]).squeeze(-1)
            predicted_rot = joint_rotations @ attachment_rotation[None]
            position_loss = (predicted_pos - targets_pos).square().mean()
            rotation_loss = _rotation_distance_squared(predicted_rot, targets_rot).mean()
            # Latent continuity and root continuity suppress jitter caused by
            # the underconstrained (five-tracker/no-pelvis) geometry.
            temporal_loss = torch.zeros((), dtype=dtype, device=device)
            if count > 1:
                temporal_loss = (latent[1:] - latent[:-1]).square().mean()
                temporal_loss = temporal_loss + 0.2 * (global_orient[1:] - global_orient[:-1]).square().mean()
                # With the raw global head stream, whole-body translation is
                # observed. Keep it smooth, but do not force walking/reaching
                # displacement into artificial hip and knee articulation.
                temporal_loss = temporal_loss + self.options.root_temporal_weight * (translation[1:] - translation[:-1]).square().mean()
            # The first state is initialized from the preceding window.  Keep
            # it anchored there so independent window minima cannot introduce
            # a pelvis/root jump at exactly every window boundary.
            boundary_loss = (latent[0] - state[0][0]).square().mean()
            boundary_loss = boundary_loss + 0.2 * (global_orient[0] - state[1][0]).square().mean()
            boundary_loss = boundary_loss + self.options.root_boundary_weight * (translation[0] - state[2][0]).square().mean()
            loss = (
                self.options.tracker_position_weight * position_loss
                + self.options.tracker_rotation_weight * rotation_loss
                + self.options.temporal_weight * temporal_loss
                + self.options.boundary_weight * boundary_loss
                + self.options.vposer_weight * latent.square().mean()
            )
            loss.backward()
            optimizer.step()
        with torch.no_grad():
            body_pose = vposer.decode(latent)["pose_body"] - vposer_zero + pose_template
            hands = torch.zeros((count, 45), dtype=dtype, device=device)
            output = body(
                global_orient=global_orient, body_pose=body_pose.reshape(count, -1), transl=translation,
                left_hand_pose=hands, right_hand_pose=hands,
                jaw_pose=torch.zeros((count, 3), dtype=dtype, device=device),
                leye_pose=torch.zeros((count, 3), dtype=dtype, device=device),
                reye_pose=torch.zeros((count, 3), dtype=dtype, device=device),
                expression=torch.zeros((count, 10), dtype=dtype, device=device), return_verts=False,
            )
            joint_rotations = _global_body_rotations(global_orient, body_pose, parents)[:, joint_ids]
            predicted_pos = output.joints[:, joint_ids] + (joint_rotations @ attachment_position[None, :, :, None]).squeeze(-1)
            rms = torch.sqrt((predicted_pos - targets_pos).square().mean()).item()
        return (latent.detach(), global_orient.detach(), translation.detach()), body_pose.detach(), rms

    def _refine_shared_mounts(
        self, body, vposer, vposer_zero, pose_template, targets_pos, targets_rot,
        initial_global, initial_translation, initial_pose, attachment_position, attachment_rotation, parents,
    ):
        """Jointly refine all frames and one fixed transform for every tracker.

        This is intentionally a *sequence-level* calibration.  The five
        mounts are shared parameters, so an apparent offset cannot disappear
        by changing an attachment on an individual frame.  A robust loss lets
        occasional optical tracking glitches be absorbed as residuals instead
        of corrupting the common physical mount.
        """
        import torch

        count = targets_pos.shape[0]
        dtype, device = targets_pos.dtype, targets_pos.device
        joint_ids = torch.as_tensor([_TRACKED_JOINTS[name] for name in _TRACKER_NAMES], device=device)
        # Re-encode the first-pass VPoser body pose.  The decoder's latent
        # parameters keep the refined sequence on the learned human manifold.
        with torch.no_grad():
            encoded_pose = (initial_pose.reshape(count, 21, 3) - pose_template + vposer_zero).reshape(count, -1)
            latent_start = vposer.encode(encoded_pose).mean
        latent = torch.nn.Parameter(latent_start.clone())
        global_orient = torch.nn.Parameter(initial_global.clone())
        translation = torch.nn.Parameter(initial_translation.clone())
        mount_position = torch.nn.Parameter(attachment_position.clone())
        # A small local rotation correction is composed with the standing-pose
        # estimate.  It remains one parameter vector per physical tracker.
        mount_rotation_delta = torch.nn.Parameter(torch.zeros((len(_TRACKER_NAMES), 3), dtype=dtype, device=device))
        optimizer = torch.optim.Adam((latent, global_orient, translation, mount_position, mount_rotation_delta), lr=0.012)
        limits = torch.as_tensor((0.15, 0.14, 0.14, 0.18, 0.18), dtype=dtype, device=device)
        ankle_centers = torch.as_tensor(((0.12, 0.0, 0.0), (-0.12, 0.0, 0.0)), dtype=dtype, device=device)
        base_mount_position = attachment_position.detach().clone()
        batch_size = min(count, max(8, self.options.shared_refinement_batch_size))
        batch_count = int(np.ceil(count / batch_size))
        for _ in range(self.options.shared_refinement_iterations):
            # The mounts remain the same Parameter objects in every chunk.
            # Processing chunks only bounds the autograd memory on macOS. We
            # accumulate every chunk's gradient before stepping, which is the
            # same shared-parameter objective as a full-sequence pass.
            optimizer.zero_grad(set_to_none=True)
            for start in range(0, count, batch_size):
                end = min(count, start + batch_size)
                size = end - start
                body_pose = vposer.decode(latent[start:end])["pose_body"] - vposer_zero + pose_template
                output = body(
                    global_orient=global_orient[start:end], body_pose=body_pose.reshape(size, -1), transl=translation[start:end],
                    left_hand_pose=torch.zeros((size, 45), dtype=dtype, device=device), right_hand_pose=torch.zeros((size, 45), dtype=dtype, device=device),
                    jaw_pose=torch.zeros((size, 3), dtype=dtype, device=device), leye_pose=torch.zeros((size, 3), dtype=dtype, device=device),
                    reye_pose=torch.zeros((size, 3), dtype=dtype, device=device), expression=torch.zeros((size, 10), dtype=dtype, device=device), return_verts=False,
                )
                joint_rotations = _global_body_rotations(global_orient[start:end], body_pose, parents)[:, joint_ids]
                predicted_position = output.joints[:, joint_ids] + (joint_rotations @ mount_position[None, :, :, None]).squeeze(-1)
                # tracker_world = joint_world @ tracker_in_joint.  The small
                # correction is expressed in the joint frame and composed on
                # the right of the existing mount.
                refined_mount_rotation = attachment_rotation @ _axis_angle_to_matrix(mount_rotation_delta)
                predicted_rotation = joint_rotations @ refined_mount_rotation[None]
                position_loss = _huber(predicted_position - targets_pos[start:end], delta=0.035).mean()
                rotation_loss = _huber(torch.sqrt(_rotation_distance_squared(predicted_rotation, targets_rot[start:end]) + 1e-8), delta=0.35).mean()
                if size > 1:
                    temporal_loss = (latent[start + 1:end] - latent[start:end - 1]).square().mean()
                    temporal_loss = temporal_loss + 0.2 * (global_orient[start + 1:end] - global_orient[start:end - 1]).square().mean()
                    temporal_loss = temporal_loss + self.options.root_temporal_weight * (translation[start + 1:end] - translation[start:end - 1]).square().mean()
                else:
                    temporal_loss = torch.zeros((), dtype=dtype, device=device)
                if start:
                    temporal_loss = temporal_loss + (latent[start] - latent[start - 1]).square().mean()
                    temporal_loss = temporal_loss + self.options.root_temporal_weight * (translation[start] - translation[start - 1]).square().mean()
                torso = output.joints[:, 12] - output.joints[:, 0]
                torso_up = torso[:, 1] / torch.linalg.vector_norm(torso, dim=1).clamp_min(1e-6)
                upright_loss = torch.relu(np.cos(np.deg2rad(30.0)) - torso_up).square().mean()
                # Priors apply once per outer pass, not once per data chunk.
                mount_prior = (
                    300.0 * torch.relu(torch.linalg.vector_norm(mount_position, dim=1) - limits).square().mean()
                    + 25.0 * (mount_position[3:] - ankle_centers).square().mean()
                    + 2.0 * (mount_position - base_mount_position).square().mean()
                    + 0.05 * mount_rotation_delta.square().mean()
                ) / batch_count
                data_loss = (
                    self.options.tracker_position_weight * position_loss
                    + 1.5 * self.options.tracker_rotation_weight * rotation_loss
                    + self.options.temporal_weight * temporal_loss
                    + self.options.vposer_weight * latent[start:end].square().mean()
                    + self.options.torso_upright_weight * upright_loss
                )
                ((data_loss + mount_prior) / batch_count).backward()
            optimizer.step()
        with torch.no_grad():
            body_pose = vposer.decode(latent)["pose_body"] - vposer_zero + pose_template
            refined_mount_rotation = attachment_rotation @ _axis_angle_to_matrix(mount_rotation_delta)
            output = body(
                global_orient=global_orient, body_pose=body_pose.reshape(count, -1), transl=translation,
                left_hand_pose=torch.zeros((count, 45), dtype=dtype, device=device), right_hand_pose=torch.zeros((count, 45), dtype=dtype, device=device),
                jaw_pose=torch.zeros((count, 3), dtype=dtype, device=device), leye_pose=torch.zeros((count, 3), dtype=dtype, device=device), reye_pose=torch.zeros((count, 3), dtype=dtype, device=device),
                expression=torch.zeros((count, 10), dtype=dtype, device=device), return_verts=False,
            )
            joint_rotations = _global_body_rotations(global_orient, body_pose, parents)[:, joint_ids]
            predicted_position = output.joints[:, joint_ids] + (joint_rotations @ mount_position[None, :, :, None]).squeeze(-1)
            rms = torch.sqrt((predicted_position - targets_pos).square().mean()).item()
        if self.options.verbose:
            print(f"shared-mount refinement: tracker position RMS={rms:.4f} m; mount norms={np.round(mount_position.detach().norm(dim=1).cpu().numpy(), 3).tolist()}")
        return (
            global_orient.detach(), translation.detach(), body_pose.detach(),
            mount_position.detach(), refined_mount_rotation.detach(),
        )

    def fit(self, frames: Sequence[TrackerFrame]) -> SmplTrajectory:
        if not frames:
            raise ValueError("cannot fit an empty frame sequence")
        import torch

        selection = np.arange(0, len(frames), max(1, self.options.sample_stride), dtype=np.int64)
        sampled = [frames[index] for index in selection]
        if any(not frame[name].valid for frame in sampled for name in _TRACKER_NAMES):
            raise ValueError("all five trackers must be valid in fitted frames")
        selected_device = select_device(self.options.device).selected
        paths = ModelPaths(Path(self.options.body_model_path), Path(self.options.vposer_path), self.options.model_type, self.options.gender)
        body = load_body_model(paths, selected_device).eval()
        vposer = load_vposer(paths, selected_device)
        if vposer is None:
            raise ValueError("VPoser path is required for this fitter")
        for parameter in body.parameters():
            parameter.requires_grad_(False)
        for parameter in vposer.parameters():
            parameter.requires_grad_(False)
        target_pos = torch.as_tensor(np.stack([[frame[name].position for name in _TRACKER_NAMES] for frame in sampled]), dtype=torch.float32, device=selected_device)
        target_rot = torch.as_tensor(np.stack([[frame[name].rotation for name in _TRACKER_NAMES] for frame in sampled]), dtype=torch.float32, device=selected_device)
        pose_template = torch.zeros((1, 21, 3), dtype=torch.float32, device=selected_device)
        # SMPL-X rest pose stands upright; rotate both shoulders from lateral
        # rest into the known forward-extended calibration pose.
        pose_template[0, 15, 1] = -np.pi / 2.0
        pose_template[0, 16, 1] = np.pi / 2.0
        with torch.no_grad():
            vposer_zero = vposer.decode(torch.zeros((1, 32), dtype=torch.float32, device=selected_device))["pose_body"]
        reference_state, attachment_position, attachment_rotation = self._reference(body, vposer, vposer_zero, pose_template, selected_device)
        parents = body.parents.to(selected_device)
        all_global, all_trans, all_pose = [], [], []
        previous = None
        for start in range(0, len(sampled), self.options.window_size):
            end = min(len(sampled), start + self.options.window_size)
            count = end - start
            if previous is None:
                state = (
                    reference_state[0].expand(count, -1).clone(),
                    reference_state[1].expand(count, -1).clone(),
                    reference_state[2].expand(count, -1).clone(),
                )
            else:
                state = tuple(value[-1:].expand(count, -1).clone() for value in previous)
            previous, pose, rms = self._fit_window(body, vposer, vposer_zero, pose_template, target_pos[start:end], target_rot[start:end], state, attachment_position, attachment_rotation, parents)
            if self.options.verbose:
                print(f"window {start}:{end}; tracker position RMS={rms:.4f} m")
            all_global.append(previous[1].cpu().numpy())
            all_trans.append(previous[2].cpu().numpy())
            all_pose.append(pose.cpu().numpy())
        count = len(sampled)
        initial_global = torch.as_tensor(np.concatenate(all_global, axis=0).astype(np.float32), device=selected_device)
        initial_translation = torch.as_tensor(np.concatenate(all_trans, axis=0).astype(np.float32), device=selected_device)
        initial_pose = torch.as_tensor(np.concatenate(all_pose, axis=0).astype(np.float32), device=selected_device)
        refined_global, refined_translation, refined_pose, refined_attachment_position, refined_attachment_rotation = self._refine_shared_mounts(
            body, vposer, vposer_zero, pose_template, target_pos, target_rot,
            initial_global, initial_translation, initial_pose, attachment_position, attachment_rotation, parents,
        )
        # Do not apply a post-hoc parameter smoother here: it changes the
        # forward kinematics after the shared rigid mounts have been solved.
        return SmplTrajectory(
            timestamps=np.asarray([frame.timestamp for frame in sampled], dtype=np.float64),
            betas=np.zeros((10,), dtype=np.float32),
            global_orient=refined_global.cpu().numpy().astype(np.float32),
            transl=refined_translation.cpu().numpy().astype(np.float32),
            body_pose=refined_pose.cpu().numpy().reshape(count, 63).astype(np.float32),
            left_hand_pose=np.zeros((count, 45), dtype=np.float32),
            right_hand_pose=np.zeros((count, 45), dtype=np.float32),
            attachment_position=refined_attachment_position.cpu().numpy().astype(np.float32),
            attachment_rotation=refined_attachment_rotation.cpu().numpy().astype(np.float32),
        )
