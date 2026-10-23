"""Profile-driven temporal SMPL-X fitting for raw five-tracker recordings."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from .calibration_motion import _protocol_body_value, protocol_body_correction, protocol_body_latent
from .coordinates import estimate_root_yaw_from_correspondences
from .device import select_device
from .geometry import quaternion_xyzw_to_matrix
from .model import ModelPaths, load_body_model, load_vposer
from .optimizer import (
    _axis_angle_to_matrix, _global_body_rotations, _huber,
    _rotation_distance_squared, _smooth_trajectory_values,
)
from .subject_calibration import TRACKED_JOINTS, TRACKER_NAMES
from .protocol import ProtocolAlignment
from .types import SmplTrajectory, TrackerFrame


def _matrix_to_axis_angle_torch(rotations):
    """Differentiable SO(3) logarithm for strict tracker projection.

    In strict mode the parent joints remain optimization variables while the
    tracked joint's local rotation is written from its measured global
    rotation.  This keeps the rigid tracker orientation an equality
    constraint throughout the position solve instead of a large-but-soft loss.
    """
    import torch

    # The usual ``acos(trace) / sin(angle)`` logarithm is well behaved away
    # from pi, but becomes an unstable 0/0 at a half-turn.  A tracker mount
    # can legitimately make a wrist's desired local frame close to 180 deg;
    # returning an arbitrary axis there is exactly the source of the old
    # one-sided hand flip.  Build a quaternion with the largest-component
    # branch instead.  At pi this recovers the axis from the symmetric part of
    # R, while for small angles we retain the first-order skew expression.
    trace = rotations.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    vee = torch.stack((
        rotations[..., 2, 1] - rotations[..., 1, 2],
        rotations[..., 0, 2] - rotations[..., 2, 0],
        rotations[..., 1, 0] - rotations[..., 0, 1],
    ), dim=-1)
    diagonal = torch.stack((
        1.0 + trace,
        1.0 + rotations[..., 0, 0] - rotations[..., 1, 1] - rotations[..., 2, 2],
        1.0 - rotations[..., 0, 0] + rotations[..., 1, 1] - rotations[..., 2, 2],
        1.0 - rotations[..., 0, 0] - rotations[..., 1, 1] + rotations[..., 2, 2],
    ), dim=-1).clamp_min(0.0)
    branch = torch.argmax(diagonal, dim=-1)
    q = torch.zeros((*rotations.shape[:-2], 4), dtype=rotations.dtype, device=rotations.device)
    # The four candidates are deliberately formed with clamped denominators;
    # only the candidate selected by ``branch`` is used in the forward pass.
    eps = torch.as_tensor(1e-6, dtype=rotations.dtype, device=rotations.device)
    # Add epsilon under the square root as well as to the quotient
    # denominators.  ``torch.gather`` selects one branch in the forward pass,
    # but autograd still evaluates the other three candidates; sqrt(0) there
    # would otherwise create an inf*zero NaN at exact half-turns.
    qw = .5 * torch.sqrt(diagonal[..., 0] + eps)
    qx = .5 * torch.sqrt(diagonal[..., 1] + eps)
    qy = .5 * torch.sqrt(diagonal[..., 2] + eps)
    qz = .5 * torch.sqrt(diagonal[..., 3] + eps)
    cand0 = torch.stack((qw, vee[..., 0] / (4.0 * qw).clamp_min(eps),
                         vee[..., 1] / (4.0 * qw).clamp_min(eps),
                         vee[..., 2] / (4.0 * qw).clamp_min(eps)), dim=-1)
    cand1 = torch.stack((vee[..., 0] / (4.0 * qx).clamp_min(eps), qx,
                         (rotations[..., 0, 1] + rotations[..., 1, 0]) / (4.0 * qx).clamp_min(eps),
                         (rotations[..., 0, 2] + rotations[..., 2, 0]) / (4.0 * qx).clamp_min(eps)), dim=-1)
    cand2 = torch.stack((vee[..., 1] / (4.0 * qy).clamp_min(eps),
                         (rotations[..., 0, 1] + rotations[..., 1, 0]) / (4.0 * qy).clamp_min(eps), qy,
                         (rotations[..., 1, 2] + rotations[..., 2, 1]) / (4.0 * qy).clamp_min(eps)), dim=-1)
    cand3 = torch.stack((vee[..., 2] / (4.0 * qz).clamp_min(eps),
                         (rotations[..., 0, 2] + rotations[..., 2, 0]) / (4.0 * qz).clamp_min(eps),
                         (rotations[..., 1, 2] + rotations[..., 2, 1]) / (4.0 * qz).clamp_min(eps), qz), dim=-1)
    candidates = torch.stack((cand0, cand1, cand2, cand3), dim=-2)
    q = torch.gather(candidates, -2, branch[..., None, None].expand(*branch.shape, 1, 4)).squeeze(-2)
    q = q / torch.linalg.vector_norm(q, dim=-1, keepdim=True).clamp_min(eps)
    # q and -q represent the same rotation.  Use the non-negative scalar
    # branch so the returned angle is in [0, pi] and does not jump at a sign
    # change in the raw quaternion stream.
    q = torch.where((q[..., :1] < 0.0), -q, q)
    sine = torch.linalg.vector_norm(q[..., 1:], dim=-1)
    angle = 2.0 * torch.atan2(sine, q[..., 0].clamp_min(0.0))
    axis = q[..., 1:] / sine[..., None].clamp_min(eps)
    regular = axis * angle[..., None]
    small = .5 * vee
    return torch.where((sine < 1e-4).unsqueeze(-1), small, regular)


@dataclass(frozen=True)
class SubjectProfile:
    scale: float
    betas: np.ndarray
    attachment_position: np.ndarray
    attachment_rotation: np.ndarray
    rotation_weight: np.ndarray
    session_yaw: dict[str, float]
    excluded_trackers: dict[str, tuple[str, ...]]
    locomotion_reference: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path) -> "SubjectProfile":
        values = json.loads(Path(path).read_text(encoding="utf-8"))
        names = TRACKER_NAMES
        mounts = values["tracker_mounts"]
        positions = np.stack([mounts[name]["position_in_joint_frame_m"] for name in names]).astype(np.float32)
        legacy = [name for name in names if "rotation_joint_tracker_xyzw" not in mounts[name]]
        if legacy:
            raise ValueError(
                "legacy subject profile does not contain the standard tracker-in-joint rotation convention; "
                "it cannot be converted safely and must be rebuilt with calibrate-subject "
                f"(legacy mounts: {', '.join(legacy)})"
            )
        rotations = np.stack([
            quaternion_xyzw_to_matrix(np.asarray(
                mounts[name]["rotation_joint_tracker_xyzw"],
                dtype=np.float64,
            ))
            for name in names
        ]).astype(np.float32)
        summary = values.get("quality", {}).get("tracker_summary", {})
        confidence_weight = {"high": 1.0, "medium": .25, "low": 0.0}
        rotation_weight = np.asarray([
            confidence_weight.get(summary.get(name, {}).get("rotation_confidence", "low"), 0.0)
            for name in names
        ], dtype=np.float32)
        session_yaw = {
            name: float(session["root_yaw_rad"])
            for name, session in values.get("sessions", {}).items()
        }
        role_map = {"camera": "head", "left_wrist": "left_hand", "right_wrist": "right_hand", "left_ankle": "left_foot", "right_ankle": "right_foot"}
        excluded: dict[str, list[str]] = {}
        for stream in values.get("quality", {}).get("excluded_streams", []):
            tracker = role_map.get(stream.get("role"), stream.get("role"))
            excluded.setdefault(str(stream["session"]), []).append(str(tracker))
        return cls(
            scale=float(values["fit"]["scale"]), betas=np.asarray(values["fit"]["betas"], dtype=np.float32),
            attachment_position=positions, attachment_rotation=rotations,
            rotation_weight=rotation_weight, session_yaw=session_yaw,
            excluded_trackers={key: tuple(item) for key, item in excluded.items()},
            locomotion_reference=dict(values.get("locomotion_reference") or {}),
        )


@dataclass(frozen=True)
class ProfileFitOptions:
    body_model_path: Path = Path("models/smplx")
    vposer_path: Path = Path("models/vposer/V02_05")
    motion_cache_path: Path = Path("assets/calibration_motion_vposer_mano.npz")
    device: str = "cpu"
    sample_stride: int = 3
    window_size: int = 60
    iterations: int = 55
    lateral_ankle_mount: bool = True
    ankle_lateral_m: float = .075
    ankle_vertical_m: float = .070
    root_smoothing_window: int = 11
    initial_transition_s: float = 3.0
    output_full_rate: bool = True
    natural_pose_only: bool = True
    post_articulation_refinement: bool = False
    tracker_position_weight: float = 3000.0
    # 0 keeps a strongly regularized VPoser reconstruction; 1 makes tracker
    # endpoints dominate every optimized source frame.  The legacy/default
    # behaviour is 0.5, for which ``tracker_position_weight`` is unchanged.
    tracker_hardness: float = .5
    # Relative within-frame weights. The global hardness controls the overall
    # tracker-vs-prior tradeoff; these three multipliers redistribute that
    # tracker evidence without changing the established all-ones result.
    head_tracker_constraint: float = 1.0
    wrist_tracker_constraint: float = 1.0
    foot_tracker_constraint: float = 1.0
    optimizer_learning_rate: float = .028
    tracker_rotation_weight: float = .10
    temporal_weight: float = 1.8
    pose_prior_weight: float = .12
    # Strength of the personal turn/gait seed learned from protocol 16/17.
    # Zero disables transfer to ordinary recordings.
    locomotion_prior_weight: float = 1.0
    upright_weight: float = 30.0
    # Contact-only virtual sole plane: heel/forefoot height plus foot-normal
    # alignment. Penetration and horizontal foot locking remain independent.
    sole_floor_constraint: bool = True
    foot_contact_weight: float = 80.0
    foot_penetration_weight: float = 180.0
    # Horizontal zero-velocity constraint for a planted ankle/toe pair.
    # This is a true m/s velocity loss, so its strength is frame-rate stable.
    foot_lock_weight: float = 100.0
    contact_height_threshold_m: float = .030
    # Zero disables speed gating and preserves the established contact rule.
    contact_speed_threshold_m_s: float = 0.0
    root_tracking_weight: float = 20.0
    root_bound_weight: float = 180.0
    root_bound_m: float = .060
    second_order_weight: float = 400.0
    leg_correction_weight: float = .035
    joint_continuity_weight: float = 12.0
    joint_jump_weight: float = 5000.0
    maximum_joint_speed_rad_s: float = 9.0
    # Production E2 safeguards.  The behavior-driven protocol splitter is
    # applied by profile_batch only when protocol fitting is enabled; these
    # joint barriers remain useful for both labelled and ordinary sequences.
    knee_hinge_weight: float = 220.0
    joint_range_weight: float = 80.0
    behavior_driven_protocol: bool = True
    # Experimental branch/torso safeguards.  Kept off by default so the
    # exploratory solve can be compared with the established E2 pipeline.
    biomechanical_branch_repair: bool = False
    torso_bend_weight: float = 35.0
    # Experimental temporal branch selector.  These terms are deliberately
    # evaluated only when ``biomechanical_branch_repair`` is enabled: they do
    # not replace tracker matching, but choose the least-jittery SO(3) branch
    # among solutions with similar five-tracker residuals.
    branch_joint_smooth_weight: float = 260.0
    branch_joint_acceleration_weight: float = 100.0
    branch_leg_smooth_weight: float = 420.0
    branch_root_acceleration_weight: float = 80.0
    # Weight of the robust common-yaw target used only for the experimental
    # ordinary-recording branch.  Production defaults retain the historical
    # weak branch selector; a strict-binding experiment can raise this to
    # make sustained body turns observable without treating retreat motion as
    # yaw.
    root_yaw_target_weight: float = 90.0
    branch_max_joint_speed_rad_s: float = 5.5
    branch_max_joint_step_deg: float = 28.0
    leg_crossing_weight: float = 20000.0
    leg_crossing_margin_m: float = .060
    hard_root_projection: bool = True
    # Isolated debugging mode.  The normal fitter treats endpoint matching as
    # a very strong soft objective so that it can remain robust to bad frames.
    # This switch adds a final chain-only IK projection whose sole purpose is
    # to test the rigid-mount hypothesis without changing production runs.
    hard_tracker_binding: bool = False
    hard_binding_iterations: int = 100
    # Optional memory-bounded strict solve. Zero keeps the historical
    # sequence-global graph; a positive value is used only by isolated
    # experiments and retains explicit cross-window branch anchors.
    hard_binding_batch_size: int = 0
    # Optional local equality refinement after the continuation/Adam pass.
    # It is disabled by default because production fitting is unchanged; the
    # isolated hard-binding experiment can use a few small Adam steps to
    # reduce the remaining millimetre-scale endpoint residual without changing
    # the selected anatomical branch.
    hard_binding_refinement_iterations: int = 0
    hard_binding_refinement_trust_weight: float = 2.0e3
    hard_binding_refinement_global_step_weight: float = 0.0
    hard_binding_refinement_neck_weight: float = 0.0
    hard_binding_learning_rate: float = .012
    hard_binding_position_weight: float = 1.0e6
    hard_binding_rotation_weight: float = 20.0
    # Isolated strict-solve probe for the least compatible endpoint.  The
    # default preserves the historical tracker weighting.
    hard_binding_head_position_multiplier: float = 1.25
    hard_binding_hand_position_multiplier: float = 1.0
    hard_binding_foot_position_multiplier: float = 1.0
    hard_binding_dynamic_foot_weight: float = 0.0
    # Debug-only equality mode.  The measured raw tracker rotations are
    # converted through the frozen mount into exact SMPL-X global rotations;
    # only their parent chains are then optimized for endpoint position.
    hard_binding_rotation_exact: bool = False
    # Independent experiment branch: release every SMPL-X body joint to the
    # temporal IK solve at the hard tracker endpoint.  The established
    # VPoser-plus-chain path remains the default and is therefore unaffected.
    full_body_ik: bool = False
    full_body_joint_weight: float = .018
    full_body_joint_temporal_weight: float = .20
    # Equality-binding diagnostics.  These are intentionally opt-in: a
    # strict endpoint solve may need the torso/root to move as well as the
    # five endpoint chains when a frozen mount and the shaped skeleton are
    # not exactly consistent.
    hard_binding_release_all_body: bool = False
    hard_binding_release_root: bool = False
    hard_binding_pure_equality: bool = False
    # In strict all-body mode the root translation is the least-squares
    # centroid that makes the five measured endpoint positions closest for
    # the current pose. Keeping this as an analytic projection removes a
    # spurious translation degree of freedom from the non-convex Adam solve.
    hard_binding_project_translation: bool = False
    # Keep root translation fixed to a smoothed ordinary-fit trajectory in the
    # strict diagnostic. A five-point endpoint objective otherwise lets the
    # root translate 10--20 cm to avoid an incorrect limb branch.
    hard_binding_optimize_translation: bool = True
    # A hard endpoint solve may otherwise use root translation as a cheap
    # alternative to changing a limb branch. Limit only its deviation from
    # the ordinary full-rate tracker fit; this is not a fixed world anchor.
    hard_binding_translation_bound_m: float = .150
    # In the strict all-body branch, keep the pelvis close to the horizontal
    # projection of the two rigid ankle centers.  Otherwise a squat can be
    # explained by moving the pelvis roughly half a metre sideways and
    # compensating with a folded spine while all five endpoints still match.
    hard_binding_pelvis_ankle_bound_m: float = .220
    # Weakly prefer the pelvis to remain centered over the rigid ankle pair
    # even *inside* the bound.  The bound alone is inactive for a 10--20 cm
    # sideways branch, which can still make the torso look as if it falls
    # sideways while all five endpoints remain nearly exact.
    hard_binding_pelvis_ankle_center_weight: float = .010
    # During a measured low-head-height / stationary-foot squat, softly pull
    # the pelvis toward the rigid ankle midpoint.  This is an observation
    # gated branch selector, not a time-indexed pose clamp: it is inactive for
    # ordinary turns/walks and leaves vertical squat depth free.
    hard_binding_squat_center_weight: float = 0.0
    # Optional observation-gated projection used only in isolated hard
    # experiments.  It blends the root XZ translation toward the rigid ankle
    # support center during a detected squat; unlike global pelvis projection,
    # it is inactive during standing, turning and walking.
    hard_binding_project_squat_center: bool = False
    # Optional squat-gated torso direction cue.  It compares the rigid
    # ankle-midpoint→head observation with the solved pelvis→head direction,
    # but only during the same low-head-height/stationary-foot gate.  This
    # encourages a natural squat bend without imposing that direction on
    # turns, reaches or ordinary walking.
    hard_binding_squat_torso_direction_weight: float = 0.0
    # Limb-plane limits for the strict branch selector.  Endpoint positions
    # alone admit a solution where a knee/elbow moves far sideways from its
    # parent-to-endpoint plane; that looks like a crossed or folded limb even
    # when left/right ordering remains technically valid.
    hard_binding_knee_lateral_limit_m: float = .160
    hard_binding_elbow_lateral_limit_m: float = .140
    hard_binding_limb_plane_weight: float = .80
    # Optional dead-zone-centred selector for the limb plane.  The limit
    # barrier above only supplies a gradient after a branch has already
    # moved sideways; this term can gently pull a large excursion back toward
    # the subject's neutral knee/elbow plane without hard-clamping normal
    # gait variation.  It is off by default and only used by isolated runs.
    hard_binding_limb_center_weight: float = 0.0
    hard_binding_limb_center_deadzone_m: float = .080
    # Optional preconditioning pass that moves only the proximal/middle limb
    # joints toward a feasible plane before the endpoint continuation starts.
    # This is a branch initializer, not an additional tracker constraint.
    hard_binding_limb_precondition_iterations: int = 0
    # Optional A/B branch seed for the right knee.  The sign follows the
    # local SMPL-X knee x-axis used by this project; zero leaves the ordinary
    # fitted branch unchanged.  This is deliberately a diagnostic knob, not
    # a time-indexed motion prior.
    hard_binding_right_knee_kick_deg: float = 0.0
    # Optional analytic two-bone target used only to select a limb branch.
    # It is disabled by default; when enabled, the target is reconstructed
    # from the measured wrist/ankle joint centres and the shaped neutral bone
    # lengths, with the forward sagittal solution selected explicitly.
    hard_binding_analytic_limb_weight: float = 0.0
    hard_binding_analytic_arm_scale: float = 1.0
    hard_binding_torso_side_lean_limit_deg: float = 18.0
    hard_binding_root_tilt_limit_deg: float = 10.0
    # Upper-spine/head hinge limit for the strict branch selector.  This is
    # deliberately broad enough for a nod, but prevents a squat from being
    # explained by folding the neck while the head tracker remains exact.
    hard_binding_neck_bend_limit_deg: float = 34.0
    # Geodesic relative-rotation limit for the neck/head chain.  This catches
    # the squat branch where the spine3-to-neck and neck-to-head frames each
    # spin by roughly 100 degrees even though the head endpoint is matched.
    # It is separate from the axial yaw diagnostic because pitch/roll are
    # equally important for rejecting a physically folded neck.
    hard_binding_neck_relative_angle_limit_deg: float = 75.0
    # Minimum geometric reach of the pelvis-to-neck/head chain in the strict
    # branch selector.  A sparse endpoint solve can otherwise satisfy the
    # head tracker by collapsing the torso into a sharply folded posture
    # during a squat.  The defaults retain the historical broad lower bounds;
    # isolated experiments can raise them without changing production.
    hard_binding_pelvis_head_min_m: float = .36
    hard_binding_pelvis_neck_min_m: float = .27
    # 180 degrees disables this optional diagnostic barrier.  Strict head
    # tracker rotation can legitimately require relative neck yaw when the
    # calibrated forehead mount and shaped SMPL-X neck are not perfectly
    # compatible; hard-locking it would trade away the head endpoint.
    hard_binding_neck_twist_limit_deg: float = 180.0
    # Per-segment upper-spine bend limit.  A free endpoint solve can otherwise
    # satisfy the head and hand positions by making spine1 and spine3 rotate
    # roughly 80 degrees in opposite directions (an S-shaped compensation).
    hard_binding_spine_bend_limit_deg: float = 65.0
    # Keep each upper-spine transverse axis close to the horizontal plane of
    # the pelvis.  Endpoint positions can otherwise be matched by rolling
    # spine1/2/3 so their local x-axis points almost vertically; this leaves
    # the shoulder/hip *positions* apparently ordered while the torso frame
    # itself is physically twisted.  The limit is on the geometric axis,
    # not on an axis-angle component, so normal forward flexion remains free.
    hard_binding_spine_axis_tilt_limit_deg: float = 25.0
    # Optional endpoint-derived torso direction cue.  The vector from the
    # rigid ankle midpoint to the corrected head center is observable even
    # when the pelvis is not tracked.  Comparing its direction with the
    # solved pelvis-to-head vector rejects a folded/face-plant torso while
    # remaining agnostic to its absolute length.
    hard_binding_torso_direction_weight: float = 0.0
    # Optional proximal-chain anchor to the ordinary tracker-informed solve.
    # It is applied only to hip/spine local rotations and acts as a branch
    # selector, not as a VPoser pose clamp; endpoint positions and rotations
    # remain the hard observations.
    hard_binding_proximal_anchor_weight: float = 0.0
    # During a measured common body turn, suppress opposite local axial
    # compensation in the hip/spine/collar/shoulder chain.  Root yaw is
    # observable from the tracker constellation, but endpoint equality alone
    # can leave the pelvis facing the old direction while the hips/spines
    # rotate back against it.  This term is gated by the measured turn and
    # constrains only the local Y (axial) component; sagittal bend and lateral
    # motion remain free.  It is isolated to hard-binding experiments.
    hard_binding_torso_local_twist_weight: float = 0.0
    # Softly discourage opposite-sign bending of adjacent spine segments.
    # This must stay small: tracker endpoint equations remain dominant and a
    # squat/forward bend is allowed to distribute over the spine naturally.
    hard_binding_spine_direction_weight: float = .03
    # Optional narrower selector: penalize only a genuinely opposing pair of
    # adjacent spine bend vectors, rather than all changes in their magnitude.
    # This is useful when a strict endpoint solve has already found the right
    # low-residual branch but one segment folds back against its neighbour.
    hard_binding_spine_opposite_weight: float = 0.0
    # Optional geometric barrier on the actual pelvis-to-neck chain.  Local
    # axis signs are only a proxy because SMPL-X's neutral spine is curved;
    # this hinge catches a sharp segment-to-segment kink while allowing a
    # smooth squat/bend to distribute over several joints.
    hard_binding_spine_curve_limit_deg: float = 70.0
    hard_binding_spine_curve_weight: float = 0.0
    # Keep the global pelvis orientation on the measured common-yaw branch;
    # forward squat/bend is represented by the spine and hip joints instead
    # of pitching the whole 1.15 m leg lever.
    hard_binding_root_yaw_only: bool = False
    # Optional isolated diagnostic projection.  It places the pelvis
    # horizontally at the rigid ankle-midpoint target before evaluating each
    # IK step; unlike centroid projection it preserves the measured mount
    # offsets and leaves the vertical translation free for squats.
    hard_binding_project_pelvis: bool = False
    # In a strict endpoint solve the shoulders can otherwise lag the measured
    # common body turn by using a locally twisted spine to absorb arm
    # position residuals.  This is opt-in to the hard-binding experiment only;
    # the production E2 path is unchanged.
    hard_binding_torso_alignment_weight: float = 2.0e6
    # Adjacent-frame local SO(3) step limit for the strict branch selector.
    # The recorder runs at roughly 15 Hz; this is wide for a real turn while
    # rejecting the 90--180 degree mirrored IK branches.
    hard_binding_joint_step_limit_deg: float = 32.0
    # Foot units in this project are physically described by the lower-shank
    # axes (knee-to-ankle / ankle-to-toe).  Keep the shank frame as the strict
    # default; the ankle-joint frame is available only for an explicit A/B
    # experiment because ankle flexion rotates the foot independently.
    hard_binding_use_ankle_rotation: bool = False
    verbose: bool = False
    progress_callback: Callable[[str, int, int], None] | None = None


@dataclass(frozen=True)
class ProfileFitResult:
    trajectory: SmplTrajectory
    frames: tuple[TrackerFrame, ...]
    valid: np.ndarray
    predicted_tracker: np.ndarray
    position_residual_m: np.ndarray
    ground_y_m: float
    summary: dict
    protocol: ProtocolAlignment | None = None
    predicted_tracker_rotation: np.ndarray | None = None
    rotation_residual_deg: np.ndarray | None = None


def _tracker_constraint_multipliers(options: ProfileFitOptions) -> np.ndarray:
    """Return per-role weights in canonical head/hands/feet order."""
    return np.asarray((
        options.head_tracker_constraint,
        options.wrist_tracker_constraint, options.wrist_tracker_constraint,
        options.foot_tracker_constraint, options.foot_tracker_constraint,
    ), dtype=np.float32)


def _rotvec_to_quaternion(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    angle = np.linalg.norm(values, axis=-1, keepdims=True)
    multiplier = np.where(angle > 1e-9, np.sin(.5 * angle) / np.maximum(angle, 1e-12), .5)
    return np.concatenate((values * multiplier, np.cos(.5 * angle)), axis=-1)


def _quaternion_to_rotvec(values: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(values, dtype=np.float64)
    quaternion /= np.linalg.norm(quaternion, axis=-1, keepdims=True).clip(min=1e-12)
    # q and -q are the same rotation. A non-negative scalar part selects the
    # shortest axis-angle representation and avoids 2π jumps.
    quaternion = np.where(quaternion[..., 3:4] < 0, -quaternion, quaternion)
    vector = quaternion[..., :3]
    length = np.linalg.norm(vector, axis=-1, keepdims=True)
    angle = 2.0 * np.arctan2(length, quaternion[..., 3:4].clip(min=1e-12))
    multiplier = np.where(length > 1e-9, angle / np.maximum(length, 1e-12), 2.0)
    return vector * multiplier


def _resample_rotvec(values: np.ndarray, source_time: np.ndarray, target_time: np.ndarray) -> np.ndarray:
    """SLERP one or more flattened axis-angle tracks onto target timestamps."""
    source = np.asarray(values, dtype=np.float64)
    source_time = np.asarray(source_time, dtype=np.float64)
    target_time = np.asarray(target_time, dtype=np.float64)
    if len(source_time) < 2:
        return np.broadcast_to(source[0], (len(target_time),) + source.shape[1:]).copy().astype(np.float32)
    # Outside the source interval the nearest measured hand/body pose is more
    # defensible than unconstrained rotational extrapolation.
    interpolation_time = np.clip(target_time, source_time[0], source_time[-1])
    original_shape = source.shape[1:]
    rotations = source.reshape(len(source), -1, 3)
    quaternion = _rotvec_to_quaternion(rotations)
    upper = np.searchsorted(source_time, interpolation_time, side="right").clip(1, len(source_time) - 1)
    lower = upper - 1
    denominator = np.maximum(source_time[upper] - source_time[lower], 1e-12)
    alpha = ((interpolation_time - source_time[lower]) / denominator)[:, None, None]
    first, second = quaternion[lower], quaternion[upper]
    dot = np.sum(first * second, axis=-1, keepdims=True)
    second = np.where(dot < 0, -second, second)
    dot = np.abs(dot).clip(0.0, 1.0)
    theta = np.arccos(dot)
    sine = np.sin(theta)
    linear = (1.0 - alpha) * first + alpha * second
    spherical = (
        np.sin((1.0 - alpha) * theta) / np.maximum(sine, 1e-12) * first
        + np.sin(alpha * theta) / np.maximum(sine, 1e-12) * second
    )
    interpolated = np.where(sine < 1e-6, linear, spherical)
    interpolated /= np.linalg.norm(interpolated, axis=-1, keepdims=True).clip(min=1e-12)
    return _quaternion_to_rotvec(interpolated).reshape((len(target_time),) + original_shape).astype(np.float32)


def _resample_linear(values: np.ndarray, source_time: np.ndarray, target_time: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return np.stack([np.interp(target_time, source_time, values[:, index]) for index in range(values.shape[1])], axis=1).astype(np.float32)


def _smooth_continuous(values: np.ndarray, timestamps: np.ndarray, window: int) -> np.ndarray:
    """Smooth only densely sampled runs, never across a recorder gap."""
    result = np.asarray(values, dtype=np.float32).copy()
    boundaries = np.flatnonzero(np.diff(timestamps) >= .45) + 1
    for indices in np.split(np.arange(len(values)), boundaries):
        if len(indices) < 3:
            continue
        local_window = min(int(window), len(indices) if len(indices) % 2 else len(indices) - 1)
        if local_window >= 3:
            result[indices] = _smooth_trajectory_values(result[indices], local_window)
    return result


def _smooth_seconds(values: np.ndarray, timestamps: np.ndarray, duration_s: float) -> np.ndarray:
    """Smooth a scalar/vector signal over a physical duration, respecting gaps."""
    result = np.asarray(values, dtype=np.float32).copy()
    if len(result) < 3 or duration_s <= 0:
        return result
    cadence = float(np.median(np.diff(timestamps)))
    window = max(3, int(round(duration_s / max(cadence, 1e-4))))
    if window % 2 == 0:
        window += 1
    return _smooth_continuous(result, timestamps, window)


def _initial_yaw_window_count(
    timestamps: np.ndarray, *, duration_s: float = .50, maximum_frames: int = 30,
) -> int:
    """Return a sample-rate-independent prefix length for initial heading."""
    values = np.asarray(timestamps, dtype=np.float64)
    if not len(values):
        raise ValueError("timestamps cannot be empty")
    count = int(np.searchsorted(values, values[0] + duration_s, side="right"))
    return min(int(maximum_frames), max(1, count))


def _coherent_yaw_offsets(
    target_rotation: np.ndarray, valid: np.ndarray, timestamps: np.ndarray,
    tracker_weights: np.ndarray, smoothing_s: float,
) -> np.ndarray:
    """Fuse head and ankle orientation changes into a continuous root-yaw seed."""
    tracker_indices = (0, 3, 4)
    principal_deltas = []
    usable_masks = []
    horizontal_confidence = []
    baseline_count = min(20, len(timestamps))
    for tracker_index in tracker_indices:
        rotations = np.asarray(target_rotation[:, tracker_index], dtype=np.float64)
        usable = np.asarray(valid[:, tracker_index], dtype=bool) & np.isfinite(rotations).all(axis=(1, 2))
        baseline = usable.copy(); baseline[baseline_count:] = False
        if not baseline.any():
            baseline = usable
        average = rotations[baseline].mean(axis=0)
        u, _, vt = np.linalg.svd(average); baseline_rotation = u @ vt
        if np.linalg.det(baseline_rotation) < 0:
            u[:, -1] *= -1; baseline_rotation = u @ vt
        # Recorder rotations act on the right: R0.T @ Rt cancels the fixed
        # pre-mount exactly and exposes the anatomical joint/root motion.
        relative = np.einsum("ij,fjk->fik", baseline_rotation.T, rotations)
        # ``rotation`` is an active local-to-world matrix.  The horizontal
        # heading of its local +X axis is therefore obtained from column 0:
        # atan2(-R[z, x], R[x, x]).  The former row-0 projection was only
        # equivalent for a pure Y rotation; head pitch/roll could turn it into
        # a spurious 180-degree yaw while the SO(3) change stayed tiny.
        yaw = np.arctan2(-relative[:, 2, 0], relative[:, 0, 0])
        horizontal_confidence.append(np.hypot(relative[:, 0, 0], relative[:, 2, 0]))
        origin = float(np.arctan2(
            np.mean(np.sin(yaw[baseline])), np.mean(np.cos(yaw[baseline])),
        )) if baseline.any() else 0.0
        principal_deltas.append(np.arctan2(np.sin(yaw - origin), np.cos(yaw - origin)))
        usable_masks.append(usable)
    principal = np.stack(principal_deltas, axis=1)
    # The head supplies the unambiguous revolution count. Individual feet can
    # cross +/-pi while stepping and must be placed on the nearest branch of
    # that head trajectory; independently unwrapping them can invent extra
    # full turns in a long raw sequence.
    head = np.unwrap(principal[:, 0])
    delta = principal.copy(); delta[:, 0] = head
    for side in (1, 2):
        difference = np.arctan2(
            np.sin(principal[:, side] - head), np.cos(principal[:, side] - head),
        )
        delta[:, side] = head + difference
    usable = np.stack(usable_masks, axis=1)
    confidence = np.stack(horizontal_confidence, axis=1) >= .20
    weights = np.asarray(tracker_weights, dtype=np.float64).reshape(1, 3) * usable * confidence
    # A foot's yaw is not a valid body-base cue when it is nearly antipodal to
    # the head.  Around pi the nearest angular branch is undefined; averaging
    # the two representatives can create an artificial jump which a later
    # speed limiter would incorrectly stretch into a slow 180-degree turn.
    # Keep the head, and any genuinely agreeing foot, but reject the ambiguous
    # foot samples.  This is deliberately local to the root-yaw seed and does
    # not discard the physical tracker observations used by IK.
    for side in (1, 2):
        difference = np.arctan2(
            np.sin(principal[:, side] - head),
            np.cos(principal[:, side] - head),
        )
        weights[:, side] *= np.abs(difference) <= np.deg2rad(120.0)
    fused = (delta * weights).sum(axis=1) / weights.sum(axis=1).clip(min=1e-8)
    fused[weights.sum(axis=1) == 0] = 0.0
    return _smooth_seconds(fused[:, None], timestamps, smoothing_s)[:, 0]


def _segment_distance_squared(
    first_start, first_end, second_start, second_end,
):
    """Squared distance between two 3-D line segments, per batch item.

    This is a differentiable closest-point formulation used only as a branch
    barrier.  A point/segment endpoint test is insufficient: two limbs can
    have separated knees and ankles while their interiors pass through one
    another.  The small epsilon keeps nearly parallel segments finite.
    """
    import torch

    first_start = torch.as_tensor(first_start)
    first_end = torch.as_tensor(first_end)
    second_start = torch.as_tensor(second_start)
    second_end = torch.as_tensor(second_end)
    u = first_end - first_start
    v = second_end - second_start
    w = first_start - second_start
    a = (u * u).sum(dim=-1)
    b = (u * v).sum(dim=-1)
    c = (v * v).sum(dim=-1)
    d = (u * w).sum(dim=-1)
    e = (v * w).sum(dim=-1)
    denominator = a * c - b * b
    # For non-parallel segments, the unconstrained closest parameters are
    # s=(be-cd)/D and t=(ae-bd)/D.  Parallel pairs use the first segment
    # endpoint as a stable fallback and are still covered by the endpoint
    # distance terms below.
    safe = denominator.abs() > 1e-8
    safe_denominator = torch.where(safe, denominator, torch.ones_like(denominator))
    s = torch.where(safe, (b * e - c * d) / safe_denominator, torch.zeros_like(denominator))
    t = torch.where(safe, (a * e - b * d) / safe_denominator, torch.zeros_like(denominator))
    s = s.clamp(0.0, 1.0)
    t = t.clamp(0.0, 1.0)
    closest = w + s[..., None] * u - t[..., None] * v
    interior = (closest * closest).sum(dim=-1)
    # Clamping both parameters independently can miss a closest point when
    # one parameter hits an endpoint. Include all four endpoint/segment
    # distances and take the smooth hard minimum with torch.minimum.
    def point_segment_squared(point, start, end):
        direction = end - start
        fraction = ((point - start) * direction).sum(dim=-1) / (
            (direction * direction).sum(dim=-1).clamp_min(1e-8)
        )
        fraction = fraction.clamp(0.0, 1.0)
        return ((point - (start + fraction[..., None] * direction)) ** 2).sum(dim=-1)

    candidates = (
        interior,
        point_segment_squared(first_start, second_start, second_end),
        point_segment_squared(first_end, second_start, second_end),
        point_segment_squared(second_start, first_start, first_end),
        point_segment_squared(second_end, first_start, first_end),
    )
    result = candidates[0]
    for candidate in candidates[1:]:
        result = torch.minimum(result, candidate)
    return result


def _body_yaw_offsets(
    target_rotation: np.ndarray, valid: np.ndarray, timestamps: np.ndarray,
    smoothing_s: float = .16,
) -> np.ndarray:
    """Estimate only the common body-base yaw from the five raw rotations.

    The fixed mount cancels exactly in a frame-to-frame relative rotation,
    ``R_t R_(t-1)^T``.  Estimate the yaw increment independently for all five
    trackers and use a robust median: a head-only motion, a single swinging
    foot, or an individual arm motion is then an outlier, while a genuine body
    turn is present in the majority of the constellation.  The previous
    implementation required the head and both feet to agree at *every* frame;
    recorder axis branches during a long turn caused it to reject an actual
    160-degree body rotation and leave the torso facing the old direction.
    This remains only a branch-selection target; the raw five endpoint
    position/rotation losses are still the binding constraints.
    """
    rotations = np.asarray(target_rotation, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    count = len(rotations)
    if rotations.ndim != 4 or rotations.shape[1:] != (len(TRACKER_NAMES), 3, 3):
        raise ValueError("target_rotation must have shape [frames, 5, 3, 3]")
    if count == 0:
        return np.zeros(0, dtype=np.float32)
    finite = valid & np.isfinite(rotations).all(axis=(2, 3))
    relative = rotations[1:] @ np.swapaxes(rotations[:-1], -1, -2)
    pair_valid = finite[1:] & finite[:-1]
    # For a general SO(3) increment this is the signed heading of local +X
    # projected onto the world horizontal plane.  It is invariant to each
    # tracker's fixed mount and remains well behaved through a 180-degree
    # crossing because it is evaluated frame-to-frame.
    yaw_increment = np.arctan2(-relative[:, :, 2, 0], relative[:, :, 0, 0])
    yaw_increment[~pair_valid] = np.nan
    increments = np.nanmedian(yaw_increment, axis=1)
    increments[~np.isfinite(increments)] = 0.0
    # Median filtering the increments (not the integrated angle) rejects an
    # isolated recorder branch without erasing a sustained body turn.
    filtered = increments.copy()
    for index in range(1, len(filtered) - 1):
        start = max(0, index - 2)
        end = min(len(filtered), index + 3)
        filtered[index] = float(np.median(increments[start:end]))
    # At ~15 Hz a sub-degree increment is within the observed tracker jitter;
    # integrating it for two minutes would manufacture a large body turn.
    filtered[np.abs(filtered) < np.deg2rad(1.2)] = 0.0
    offsets = np.concatenate(([0.0], np.cumsum(filtered)))
    return _smooth_seconds(offsets[:, None].astype(np.float32), timestamps, smoothing_s)[:, 0]


def _position_base_yaw_offsets(
    target_position: np.ndarray, valid: np.ndarray, timestamps: np.ndarray,
    smoothing_s: float = .16,
) -> np.ndarray:
    """Estimate a body-base yaw change from the five endpoint positions.

    This is intentionally a diagnostic/experimental seed, not a replacement
    for tracker orientation.  After removing the frame translation, a weighted
    2-D Procrustes fit of head, hands and ankles gives the common planar turn
    while limb motion appears as a residual rather than as a root rotation.
    The unwrapped result is smoothed in time and therefore cannot introduce a
    one-frame 180-degree root branch.
    """
    positions = np.asarray(target_position, dtype=np.float64)
    usable = np.asarray(valid, dtype=bool)
    if positions.ndim != 3 or positions.shape[1:] != (len(TRACKER_NAMES), 3):
        raise ValueError("target_position must have shape [frames, 5, 3]")
    count = len(positions)
    baseline_count = min(count, max(8, _initial_yaw_window_count(timestamps)))
    weights = np.asarray((.85, .65, .65, 1.0, 1.0), dtype=np.float64)
    base_mask = usable[:baseline_count] & np.isfinite(positions[:baseline_count]).all(axis=2)
    base_values = positions[:baseline_count].copy()
    base_values[~base_mask] = np.nan
    with np.errstate(invalid="ignore"):
        base = np.nanmedian(base_values, axis=0)
    base_valid = np.isfinite(base).all(axis=1)
    if base_valid.sum() < 3:
        return np.zeros(count, dtype=np.float32)
    angles = np.zeros(count, dtype=np.float64)
    for frame_index in range(count):
        frame_valid = usable[frame_index] & np.isfinite(positions[frame_index]).all(axis=1) & base_valid
        if frame_valid.sum() < 3:
            angles[frame_index] = angles[frame_index - 1] if frame_index else 0.0
            continue
        frame_weights = weights * frame_valid
        source = base[frame_valid][:, (0, 2)]
        target = positions[frame_index, frame_valid][:, (0, 2)]
        source = source - np.average(source, axis=0, weights=frame_weights[frame_valid])
        target = target - np.average(target, axis=0, weights=frame_weights[frame_valid])
        cosine = np.sum(frame_weights[frame_valid] * (source[:, 0] * target[:, 0] + source[:, 1] * target[:, 1]))
        sine = np.sum(frame_weights[frame_valid] * (source[:, 0] * target[:, 1] - source[:, 1] * target[:, 0]))
        angles[frame_index] = np.arctan2(sine, cosine)
    angles = np.unwrap(angles)
    if count > 1:
        # Reject impossible position-constellation branch changes before the
        # low-pass filter. This is only a seed for the root and does not alter
        # any measured endpoint.
        maximum_speed = np.deg2rad(180.0)
        for index in range(1, count):
            dt = max(float(timestamps[index] - timestamps[index - 1]), 1e-4)
            angles[index] = angles[index - 1] + np.clip(
                angles[index] - angles[index - 1], -maximum_speed * dt, maximum_speed * dt,
            )
    return _smooth_seconds(angles.astype(np.float32)[:, None], timestamps, smoothing_s)[:, 0]


def _debounce_contact_mask(
    mask: np.ndarray, timestamps: np.ndarray, *, minimum_contact_s: float = .10,
    maximum_gap_s: float = .12,
) -> np.ndarray:
    """Fill tiny stance gaps and reject isolated one-frame contacts."""
    result = np.asarray(mask, dtype=bool).copy()
    if len(result) < 2:
        return result

    def runs(value: bool) -> list[np.ndarray]:
        indices = np.flatnonzero(result == value)
        return [] if not len(indices) else list(
            np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1)
        )

    for gap in runs(False):
        if gap[0] == 0 or gap[-1] == len(result) - 1:
            continue
        duration = float(timestamps[gap[-1]] - timestamps[gap[0]])
        if result[gap[0] - 1] and result[gap[-1] + 1] and duration <= maximum_gap_s:
            result[gap] = True
    for contact in runs(True):
        duration = float(timestamps[contact[-1]] - timestamps[contact[0]])
        if duration < minimum_contact_s:
            result[contact] = False
    return result


def _local_joint_step_summary(
    global_orient: np.ndarray, body_pose: np.ndarray, timestamps: np.ndarray,
) -> dict[str, float | int | None]:
    """Summarize frame-local SO(3) steps without axis-angle branch artefacts."""
    local_pose = np.concatenate((
        np.asarray(global_orient, dtype=np.float64).reshape(-1, 1, 3),
        np.asarray(body_pose, dtype=np.float64).reshape(-1, 21, 3),
    ), axis=1)
    timestamps = np.asarray(timestamps, dtype=np.float64)
    if len(local_pose) < 2:
        return {
            "p95_max_step_deg": 0.0, "p99_max_step_deg": 0.0,
            "maximum_step_deg": 0.0, "maximum_step_frame": None,
            "maximum_step_joint_index": None, "steps_over_42_deg": 0,
            "steps_over_60_deg": 0, "steps_over_90_deg": 0,
        }
    quaternion = _rotvec_to_quaternion(local_pose)
    dot = np.abs(np.sum(quaternion[1:] * quaternion[:-1], axis=-1)).clip(0.0, 1.0)
    angle_deg = np.degrees(2.0 * np.arccos(dot))
    # A recorder gap does not represent one physical frame-to-frame motion.
    adjacent = (np.diff(timestamps) > 0) & (np.diff(timestamps) < .45)
    pair_indices = np.flatnonzero(adjacent)
    if not len(pair_indices):
        return {
            "p95_max_step_deg": 0.0, "p99_max_step_deg": 0.0,
            "maximum_step_deg": 0.0, "maximum_step_frame": None,
            "maximum_step_joint_index": None, "steps_over_42_deg": 0,
            "steps_over_60_deg": 0, "steps_over_90_deg": 0,
        }
    adjacent_angles = angle_deg[pair_indices]
    maximum_by_pair = adjacent_angles.max(axis=1)
    local_pair, joint_index = np.unravel_index(
        int(np.argmax(adjacent_angles)), adjacent_angles.shape,
    )
    return {
        "p95_max_step_deg": float(np.percentile(maximum_by_pair, 95)),
        "p99_max_step_deg": float(np.percentile(maximum_by_pair, 99)),
        "maximum_step_deg": float(adjacent_angles[local_pair, joint_index]),
        "maximum_step_frame": int(pair_indices[local_pair]),
        "maximum_step_joint_index": int(joint_index),
        "steps_over_42_deg": int(np.sum(maximum_by_pair > 42.0)),
        "steps_over_60_deg": int(np.sum(maximum_by_pair > 60.0)),
        "steps_over_90_deg": int(np.sum(maximum_by_pair > 90.0)),
    }


def _stream_validity(frames: Sequence[TrackerFrame], session_id: str, profile: SubjectProfile) -> tuple[np.ndarray, list[str]]:
    valid = np.asarray([[frame[name].valid for name in TRACKER_NAMES] for frame in frames], dtype=bool)
    excluded = set(profile.excluded_trackers.get(session_id, ()))
    positions = np.stack([[frame[name].position for name in TRACKER_NAMES] for frame in frames])
    # A recorder can mark a wildly incorrect pose as valid.  Detect persistent
    # ankle geometry independently so the batch command also works on sessions
    # not present when the profile JSON was created.
    ankle_separation = np.linalg.norm(positions[:, 3] - positions[:, 4], axis=1)
    both_ankles = valid[:, 3] & valid[:, 4] & np.isfinite(ankle_separation)
    median_separation = float(np.median(ankle_separation[both_ankles])) if np.any(both_ankles) else None
    if median_separation is not None and not .08 <= median_separation <= .80:
        head_distance = np.median(np.linalg.norm(positions[:, 3:5] - positions[:, :1], axis=2), axis=0)
        excluded.add(TRACKER_NAMES[3 + int(np.argmax(head_distance))])
    for tracker_index, tracker_name in enumerate(TRACKER_NAMES):
        if tracker_name in excluded:
            valid[:, tracker_index] = False
    # Frame-local impossible points are ignored without rejecting a healthy
    # stream for the entire recording.
    distance_to_head = np.linalg.norm(positions - positions[:, :1], axis=2)
    valid &= distance_to_head < 2.30
    if not np.any(valid.sum(axis=1) >= 3):
        raise ValueError(f"{session_id}: fewer than three usable trackers throughout the recording")
    return valid, sorted(excluded)


class ProfileDrivenFitter:
    """Fit motion while keeping the calibrated subject and mounts fixed."""

    def __init__(self, profile: SubjectProfile, options: ProfileFitOptions | None = None) -> None:
        import torch

        self.profile = profile
        self.options = options or ProfileFitOptions()
        if not 0.0 <= self.options.tracker_hardness <= 1.0:
            raise ValueError("tracker_hardness must be between 0 and 1")
        if self.options.iterations < 1 or self.options.window_size < 2:
            raise ValueError("iterations must be positive and window_size must be at least two")
        nonnegative = (
            "tracker_position_weight", "tracker_rotation_weight", "temporal_weight",
            "head_tracker_constraint", "wrist_tracker_constraint", "foot_tracker_constraint",
            "pose_prior_weight", "locomotion_prior_weight", "upright_weight", "foot_contact_weight",
            "foot_penetration_weight", "foot_lock_weight", "root_tracking_weight",
            "root_bound_weight", "second_order_weight", "leg_correction_weight",
            "joint_continuity_weight", "joint_jump_weight",
            "knee_hinge_weight", "joint_range_weight", "torso_bend_weight",
            "branch_joint_smooth_weight", "branch_joint_acceleration_weight",
            "branch_leg_smooth_weight", "branch_root_acceleration_weight",
            "root_yaw_target_weight",
            "branch_max_joint_speed_rad_s", "branch_max_joint_step_deg",
            "leg_crossing_weight",
            "leg_crossing_margin_m",
            "hard_binding_iterations", "hard_binding_batch_size", "hard_binding_learning_rate",
            "hard_binding_refinement_iterations",
            "hard_binding_refinement_trust_weight",
            "hard_binding_refinement_global_step_weight",
            "hard_binding_refinement_neck_weight",
            "hard_binding_position_weight", "hard_binding_rotation_weight",
            "hard_binding_head_position_multiplier", "hard_binding_hand_position_multiplier",
            "hard_binding_foot_position_multiplier",
            "hard_binding_dynamic_foot_weight",
            "hard_binding_translation_bound_m", "hard_binding_pelvis_ankle_bound_m",
            "hard_binding_pelvis_ankle_center_weight", "hard_binding_squat_center_weight",
            "hard_binding_squat_torso_direction_weight",
            "hard_binding_knee_lateral_limit_m",
            "hard_binding_elbow_lateral_limit_m",
            "hard_binding_limb_plane_weight",
            "hard_binding_limb_center_weight",
            "hard_binding_limb_center_deadzone_m",
            "hard_binding_analytic_limb_weight",
            "hard_binding_analytic_arm_scale",
            "hard_binding_torso_side_lean_limit_deg", "hard_binding_torso_alignment_weight",
                "hard_binding_root_tilt_limit_deg",
            "hard_binding_neck_bend_limit_deg",
            "hard_binding_neck_twist_limit_deg",
            "hard_binding_neck_relative_angle_limit_deg",
            "hard_binding_pelvis_head_min_m",
            "hard_binding_pelvis_neck_min_m",
            "hard_binding_spine_bend_limit_deg",
            "hard_binding_spine_direction_weight",
            "hard_binding_spine_axis_tilt_limit_deg",
            "hard_binding_torso_direction_weight",
            "hard_binding_proximal_anchor_weight",
            "hard_binding_torso_local_twist_weight",
            "hard_binding_spine_opposite_weight",
            "hard_binding_spine_curve_limit_deg",
            "hard_binding_spine_curve_weight",
            "hard_binding_joint_step_limit_deg",
            "full_body_joint_weight", "full_body_joint_temporal_weight",
        )
        for name in nonnegative:
            if float(getattr(self.options, name)) < 0:
                raise ValueError(f"{name} must be non-negative")
        tracker_constraint = _tracker_constraint_multipliers(self.options)
        if not np.isfinite(tracker_constraint).all() or tracker_constraint.sum() <= 0:
            raise ValueError("at least one finite tracker constraint multiplier must be positive")
        if self.options.optimizer_learning_rate <= 0:
            raise ValueError("optimizer_learning_rate must be positive")
        if self.options.hard_tracker_binding and self.options.hard_binding_iterations < 1:
            raise ValueError("hard_binding_iterations must be positive")
        if self.options.hard_binding_batch_size < 0:
            raise ValueError("hard_binding_batch_size must be non-negative")
        if self.options.hard_binding_limb_precondition_iterations < 0:
            raise ValueError("hard_binding_limb_precondition_iterations must be non-negative")
        if not np.isfinite(self.options.hard_binding_right_knee_kick_deg):
            raise ValueError("hard_binding_right_knee_kick_deg must be finite")
        if self.options.hard_tracker_binding and self.options.hard_binding_learning_rate <= 0:
            raise ValueError("hard_binding_learning_rate must be positive")
        if self.options.contact_height_threshold_m <= 0 or self.options.contact_speed_threshold_m_s < 0:
            raise ValueError("contact thresholds must be positive (speed may be zero to disable gating)")
        if self.options.root_bound_m < 0 or self.options.maximum_joint_speed_rad_s <= 0:
            raise ValueError("root bound must be non-negative and maximum joint speed must be positive")
        self.device = select_device(self.options.device).selected
        paths = ModelPaths(self.options.body_model_path, self.options.vposer_path, "smplx", "neutral")
        self.body = load_body_model(paths, self.device).eval()
        self.vposer = load_vposer(paths, self.device).eval()
        if self.vposer is None:
            raise ValueError("VPoser is required")
        for model in (self.body, self.vposer):
            for parameter in model.parameters():
                parameter.requires_grad_(False)
        self.parents = self.body.parents.to(self.device)
        self.position_joint_ids = torch.as_tensor(
            [TRACKED_JOINTS[name] for name in TRACKER_NAMES], device=self.device,
        )
        # Foot tracker positions are anchored at ankle points (7/8), while
        # the calibrated box-axis prior describes X along knee-to-ankle, i.e.
        # the lower-shank frame (4/5).  Preserve that established production
        # convention.  A strict-binding A/B can explicitly request the ankle
        # joint frame, but it is not assumed to be physically correct.
        self.rotation_joint_ids = torch.as_tensor(
            (15, 20, 21, 7, 8) if (
                self.options.hard_tracker_binding
                and self.options.hard_binding_use_ankle_rotation
            )
            else (15, 20, 21, 4, 5), device=self.device,
        )
        self.betas = torch.as_tensor(profile.betas, dtype=torch.float32, device=self.device)
        self.mount_position = torch.as_tensor(profile.attachment_position, dtype=torch.float32, device=self.device).clone()
        self.mount_rotation = torch.as_tensor(
            profile.attachment_rotation, dtype=torch.float32, device=self.device,
        ).clone()
        self.rotation_weight = torch.as_tensor(profile.rotation_weight, dtype=torch.float32, device=self.device)
        self.tracker_constraint_multiplier = torch.as_tensor(
            tracker_constraint, dtype=torch.float32, device=self.device,
        )
        self.initial_latents = self._load_initial_latents()
        contact_reference = profile.locomotion_reference.get("contact", {})
        use_personal_contact = bool(profile.locomotion_reference) and self.options.locomotion_prior_weight > 0
        self.contact_height_threshold_m = float(
            contact_reference.get("height_threshold_m", self.options.contact_height_threshold_m)
            if use_personal_contact and np.isclose(self.options.contact_height_threshold_m, .030)
            else self.options.contact_height_threshold_m
        )
        self.contact_speed_threshold_m_s = float(
            contact_reference.get("speed_threshold_m_s", self.options.contact_speed_threshold_m_s)
            if use_personal_contact and self.options.contact_speed_threshold_m_s == 0.0
            else self.options.contact_speed_threshold_m_s
        )
        if self.options.lateral_ankle_mount and not (
            self.options.hard_tracker_binding
            and self.options.hard_binding_use_ankle_rotation
        ):
            self._set_lateral_ankle_mounts()
        self._neutral_knee_lateral = self._compute_neutral_knee_lateral()

    def _compute_neutral_knee_lateral(self) -> tuple[float, float]:
        """Return neutral signed knee offsets used only as a branch selector."""
        import torch

        with torch.no_grad():
            latent = torch.as_tensor(
                self.initial_latents.get("neutral", np.zeros(32, dtype=np.float32)),
                dtype=torch.float32, device=self.device,
            )[None]
            pose = self.vposer.decode(latent)["pose_body"].reshape(1, 21, 3)
            output = self.body(
                betas=self.betas[None], global_orient=torch.zeros((1, 3), device=self.device),
                transl=torch.zeros((1, 3), device=self.device), body_pose=pose.reshape(1, -1),
                left_hand_pose=torch.zeros((1, 45), device=self.device),
                right_hand_pose=torch.zeros((1, 45), device=self.device),
                jaw_pose=torch.zeros((1, 3), device=self.device),
                leye_pose=torch.zeros((1, 3), device=self.device),
                reye_pose=torch.zeros((1, 3), device=self.device),
                expression=torch.zeros((1, 10), device=self.device), return_verts=False,
            )
            joints = output.joints[0]
            # This is the shaped neutral ankle-midpoint offset in the pelvis
            # frame.  It is used only by the isolated hard-binding branch to
            # reject an implausible lateral pelvis translation; vertical
            # crouching remains unconstrained.
            self._neutral_ankle_mid = (
                .5 * (joints[7] + joints[8])
            ).detach().cpu().numpy().astype(np.float32)
        offsets = []
        limb_lengths = []
        for hip, knee, ankle in ((1, 4, 7), (2, 5, 8)):
            fraction = (
                (joints[hip, 1] - joints[knee, 1])
                / (joints[hip, 1] - joints[ankle, 1]).clamp_min(1e-6)
            )
            line = joints[hip] + fraction * (joints[ankle] - joints[hip])
            offsets.append(float((joints[knee, 0] - line[0]).cpu()))
            limb_lengths.append((
                float(torch.linalg.vector_norm(joints[hip] - joints[knee]).cpu()),
                float(torch.linalg.vector_norm(joints[knee] - joints[ankle]).cpu()),
            ))
        arm_lengths = []
        for shoulder, elbow, wrist in ((16, 18, 20), (17, 19, 21)):
            arm_lengths.append((
                float(torch.linalg.vector_norm(joints[shoulder] - joints[elbow]).cpu()),
                float(torch.linalg.vector_norm(joints[elbow] - joints[wrist]).cpu()),
            ))
        self._neutral_limb_lengths = tuple(limb_lengths)
        self._neutral_arm_lengths = tuple(arm_lengths)
        return tuple(offsets)

    def _joint_constraint_losses(self, body_pose, world_joints, rotations):
        """Soft E2 knee-side and broad anatomical-range barriers."""
        import torch

        dtype = world_joints.dtype
        zero = torch.zeros((), dtype=dtype, device=self.device)
        knee_hinge_loss = zero
        joint_range_loss = zero
        leg_crossing_loss = zero
        if (
            self.options.knee_hinge_weight <= 0
            and self.options.joint_range_weight <= 0
            and not self.options.biomechanical_branch_repair
        ):
            return knee_hinge_loss, joint_range_loss, leg_crossing_loss
        if self.options.knee_hinge_weight > 0 or self.options.biomechanical_branch_repair:
            root_local = (
                rotations[:, 0].transpose(-1, -2)
                @ (world_joints - world_joints[:, :1]).transpose(-1, -2)
            ).transpose(-1, -2)
            for leg_index, (hip, knee, ankle) in enumerate(((1, 4, 7), (2, 5, 8))):
                denominator = (root_local[:, hip, 1] - root_local[:, ankle, 1]).clamp_min(1e-6)
                fraction = ((root_local[:, hip, 1] - root_local[:, knee, 1]) / denominator).clamp(0.0, 1.0)
                line_x = root_local[:, hip, 0] + fraction * (root_local[:, ankle, 0] - root_local[:, hip, 0])
                lateral = root_local[:, knee, 0] - line_x
                neutral_lateral = float(self._neutral_knee_lateral[leg_index])
                side = 1.0 if neutral_lateral >= 0.0 else -1.0
                wrong_side = torch.relu(.005 - side * lateral).square()
                weak_shape = .10 * (lateral - neutral_lateral).square()
                knee_hinge_loss = knee_hinge_loss + (wrong_side + weak_shape).mean()
                if self.options.biomechanical_branch_repair:
                    # In the SMPL-X/root convention used by this project the
                    # normal knee bend is toward local +Z (forward).  A
                    # mirrored IK branch bends the knee behind the hip--ankle
                    # line while preserving the same joint angle, so the
                    # ordinary angle barrier cannot reject it.
                    line_z = root_local[:, hip, 2] + fraction * (
                        root_local[:, ankle, 2] - root_local[:, hip, 2]
                    )
                    sagittal = root_local[:, knee, 2] - line_z
                    knee_hinge_loss = knee_hinge_loss + .80 * torch.relu(
                        -.012 - sagittal
                    ).square().mean()
            if self.options.biomechanical_branch_repair:
                # A valid IK solution must preserve the left/right leg order.
                # Endpoint positions alone do not prevent a mirrored solution
                # in which both ankle trackers are matched while the SMPL-X
                # legs cross through one another. Use the root-local lateral
                # coordinate (subject-left is +X) at hip, knee and ankle.
                margin = float(self.options.leg_crossing_margin_m)
                for left_id, right_id, joint_margin in (
                    (1, 2, margin), (4, 5, .75 * margin), (7, 8, margin),
                ):
                    separation = root_local[:, left_id, 0] - root_local[:, right_id, 0]
                    leg_crossing_loss = leg_crossing_loss + torch.relu(
                        float(joint_margin) - separation
                    ).square().mean()
                # A lateral ordering check alone does not catch a diagonal
                # pass-through: the two knees can remain left/right ordered
                # while a lower-leg or forearm segment crosses the opposite
                # segment in depth. Add a conservative segment-segment
                # clearance barrier for both legs and both arms.
                segment_pairs = (
                    ((1, 4), (2, 5)), ((1, 4), (5, 8)),
                    ((4, 7), (2, 5)), ((4, 7), (5, 8)),
                    ((16, 18), (17, 19)), ((16, 18), (19, 21)),
                    ((18, 20), (17, 19)), ((18, 20), (19, 21)),
                )
                for pair_index, ((first_start, first_end), (second_start, second_end)) in enumerate(segment_pairs):
                    clearance = .035 if pair_index < 4 else .025
                    distance_squared = _segment_distance_squared(
                        root_local[:, first_start], root_local[:, first_end],
                        root_local[:, second_start], root_local[:, second_end],
                    )
                    leg_crossing_loss = leg_crossing_loss + torch.relu(
                        float(clearance * clearance) - distance_squared
                    ).square().mean()
        if self.options.joint_range_weight > 0:
            def joint_angle(first, middle, last):
                first_vector = world_joints[:, first] - world_joints[:, middle]
                second_vector = world_joints[:, last] - world_joints[:, middle]
                denominator = (
                    torch.linalg.vector_norm(first_vector, dim=-1)
                    * torch.linalg.vector_norm(second_vector, dim=-1)
                ).clamp_min(1e-6)
                cosine = (first_vector * second_vector).sum(dim=-1) / denominator
                # Avoid the infinite derivative of acos at a perfectly
                # straight joint, which otherwise can create NaNs in Adam.
                return torch.acos(cosine.clamp(-1.0 + 1e-5, 1.0 - 1e-5))

            knee_min = torch.as_tensor(np.deg2rad(55.0), dtype=dtype, device=self.device)
            elbow_min = torch.as_tensor(np.deg2rad(20.0), dtype=dtype, device=self.device)
            for hip, knee, ankle in ((1, 4, 7), (2, 5, 8)):
                joint_range_loss = joint_range_loss + torch.relu(
                    knee_min - joint_angle(hip, knee, ankle)
                ).square().mean()
            for shoulder, elbow, wrist in ((16, 18, 20), (17, 19, 21)):
                joint_range_loss = joint_range_loss + torch.relu(
                    elbow_min - joint_angle(shoulder, elbow, wrist)
                ).square().mean()
            if self.options.biomechanical_branch_repair:
                # Elbow flexion is toward local -Z.  Penalize only the
                # opposite-side branch; a nearly straight arm remains free.
                for shoulder, elbow, wrist in ((16, 18, 20), (17, 19, 21)):
                    denominator = (
                        root_local[:, shoulder, 1] - root_local[:, wrist, 1]
                    ).clamp_min(1e-6)
                    fraction = (
                        (root_local[:, shoulder, 1] - root_local[:, elbow, 1])
                        / denominator
                    ).clamp(0.0, 1.0)
                    line_z = root_local[:, shoulder, 2] + fraction * (
                        root_local[:, wrist, 2] - root_local[:, shoulder, 2]
                    )
                    sagittal = root_local[:, elbow, 2] - line_z
                    joint_range_loss = joint_range_loss + .80 * torch.relu(
                        sagittal - .012
                    ).square().mean()
            pose_vectors = body_pose.reshape(-1, 21, 3)
            for joint_index in (0, 1, 3, 4, 15, 16, 17, 18):
                local_angle = torch.linalg.vector_norm(pose_vectors[:, joint_index], dim=-1)
                limit = 2.0943951023931953 if joint_index in (3, 4, 17, 18) else 2.356194490192345
                joint_range_loss = joint_range_loss + .10 * torch.relu(local_angle - limit).square().mean()
        return (
            torch.nan_to_num(knee_hinge_loss),
            torch.nan_to_num(joint_range_loss),
            torch.nan_to_num(leg_crossing_loss),
        )

    def _set_lateral_ankle_mounts(self) -> None:
        """Place ankle tracker centres on the lateral lower-leg surfaces.

        The profile stores offsets in opaque SMPL joint-local axes. Convert an
        interpretable anatomical neutral-pose offset into those local axes so
        it remains rigid when the ankle/leg rotates.
        """
        import torch

        with torch.no_grad():
            latent = torch.as_tensor(self.initial_latents["neutral"], dtype=torch.float32, device=self.device)[None]
            pose = self.vposer.decode(latent)["pose_body"].reshape(1, 21, 3)
            rotation = _global_body_rotations(
                torch.zeros((1, 3), device=self.device), pose, self.parents,
            )[0]
            desired = torch.as_tensor((
                (self.options.ankle_lateral_m, self.options.ankle_vertical_m, 0.0),
                (-self.options.ankle_lateral_m, self.options.ankle_vertical_m, 0.0),
            ), dtype=torch.float32, device=self.device)
            for row, (position_joint_id, rotation_joint_id) in enumerate(((7, 4), (8, 5))):
                tracker_index = 3 + row
                # Convert the saved ankle-local orientation into the physical
                # shank-local convention at the neutral reference pose.
                self.mount_rotation[tracker_index] = (
                    rotation[rotation_joint_id].T
                    @ rotation[position_joint_id]
                    @ self.mount_rotation[tracker_index]
                )
                self.mount_position[tracker_index] = rotation[rotation_joint_id].T @ desired[row]

    def _load_initial_latents(self) -> dict[str, np.ndarray]:
        cache_path = self.options.motion_cache_path
        if not cache_path.exists():
            self.initial_corrections = {"neutral": np.zeros(63, dtype=np.float32), "forward": np.zeros(63, dtype=np.float32)}
            return {"neutral": np.zeros(32, dtype=np.float32), "forward": np.zeros(32, dtype=np.float32)}
        with np.load(cache_path, allow_pickle=False) as values:
            keys = values["body_keys"].tolist()
            latents = values["body_latents"].astype(np.float32)
            corrections = values["body_corrections"].astype(np.float32)
        self.initial_corrections = dict(zip(keys, corrections, strict=True))
        return dict(zip(keys, latents, strict=True))

    def _protocol_latents(self, protocol: ProtocolAlignment | None, timestamps: np.ndarray, fallback: str) -> np.ndarray:
        if protocol is None:
            return np.broadcast_to(self.initial_latents[fallback], (len(timestamps), 32)).copy().astype(np.float32)
        result = []
        for timestamp in timestamps:
            segment = protocol.segment_at(float(timestamp))
            # Do not replay the authored step count for stages 16/17. The
            # operator's pivot/walk timing is recovered from the measured
            # ankle heights in ``_observed_protocol_priors`` below.
            if segment.key in {"turns", "walk"}:
                result.append(self.initial_latents["neutral"].copy())
                continue
            # Static holds use their complete labelled stage so the seed also
            # follows the enter/exit transition; dynamic phases use the
            # measured active interval containing the deliberate movement.
            dynamic = segment.key in {"head", "elbows", "wrists", "squat", "left_knee", "right_knee", "ankles"}
            phase = segment.phase(float(timestamp), active=dynamic)
            result.append(
                _protocol_body_value(self.initial_latents, segment.motion, phase)
                if self.options.natural_pose_only else
                protocol_body_latent(self.initial_latents, segment.motion, phase)
            )
        return np.asarray(result, dtype=np.float32)

    def _protocol_corrections(self, protocol: ProtocolAlignment | None, timestamps: np.ndarray, fallback: str) -> np.ndarray:
        if protocol is None:
            return np.zeros((len(timestamps), 63), dtype=np.float32)
        result = []
        for timestamp in timestamps:
            segment = protocol.segment_at(float(timestamp))
            dynamic = segment.key in {"head", "elbows", "wrists", "squat", "left_knee", "right_knee", "ankles"}
            phase = segment.phase(float(timestamp), active=dynamic)
            result.append(protocol_body_correction(self.initial_corrections, segment.motion, phase))
        return np.asarray(result, dtype=np.float32)

    def _personal_locomotion_priors(
        self, timestamps: np.ndarray, target_position: np.ndarray, valid: np.ndarray,
        latent: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
        """Transfer the subject's stage-17 swing scale/timing to ordinary raw data.

        The new recording still determines every actual step.  The learned
        cadence only sets the smoothing time scale, while measured left/right
        ankle lifts select the corresponding personal gait seed frame by frame.
        """
        reference = self.profile.locomotion_reference
        weight = float(self.options.locomotion_prior_weight)
        if not reference or weight <= 0 or not all(
            key in self.initial_latents for key in ("neutral", "walk_left_swing", "walk_right_swing")
        ):
            return latent, np.zeros(len(timestamps), dtype=np.float32), {
                "active_fraction": 0.0, "maximum_swing_amount": 0.0,
            }
        gait = reference.get("gait", {})
        learned_lift = np.asarray(gait.get("ankle_lift_m", (.10, .10)), dtype=np.float64)
        if learned_lift.shape != (2,):
            learned_lift = np.asarray((.10, .10), dtype=np.float64)
        step_interval = float(gait.get("step_interval_s", 1.0))
        amounts = np.zeros((len(timestamps), 2), dtype=np.float32)
        for side_index, tracker_index in enumerate((3, 4)):
            usable = valid[:, tracker_index] & np.isfinite(target_position[:, tracker_index]).all(axis=1)
            if usable.sum() < 3:
                continue
            height = target_position[:, tracker_index, 1]
            baseline = float(np.percentile(height[usable], 10))
            denominator = max(float(learned_lift[side_index]) - self.contact_height_threshold_m, .035)
            amount = np.clip(
                (height - baseline - .75 * self.contact_height_threshold_m) / denominator,
                0.0, 1.0,
            )
            amount[~usable] = 0.0
            amounts[:, side_index] = _smooth_seconds(
                amount[:, None], timestamps, np.clip(.16 * step_interval, .10, .28),
            )[:, 0]
        # Simultaneous high values are possible at tracker glitches. Normalize
        # the pair so the latent never extrapolates beyond one learned swing.
        total = amounts.sum(axis=1)
        excessive = total > 1.0
        amounts[excessive] /= total[excessive, None]
        blend_strength = min(weight, 1.0)
        neutral = self.initial_latents["neutral"]
        gait_latent = (
            neutral[None]
            + blend_strength * amounts[:, :1] * (self.initial_latents["walk_left_swing"] - neutral)[None]
            + blend_strength * amounts[:, 1:] * (self.initial_latents["walk_right_swing"] - neutral)[None]
        )
        swing = amounts.max(axis=1)
        active = swing >= .06
        result = latent.copy(); result[active] = gait_latent[active]
        prior_factor = np.where(active, weight * (.35 + 1.65 * swing), 0.0).astype(np.float32)
        return result.astype(np.float32), prior_factor, {
            "active_fraction": float(active.mean()),
            "maximum_swing_amount": float(swing.max(initial=0.0)),
        }

    def _personal_root_yaw_offsets(
        self, target_rotation: np.ndarray, valid: np.ndarray, timestamps: np.ndarray,
    ) -> np.ndarray:
        reference = self.profile.locomotion_reference
        if not reference or self.options.locomotion_prior_weight <= 0:
            self._root_yaw_seed_diagnostics = {
                "enabled": False,
                "reason": "locomotion_prior_disabled_or_unavailable",
            }
            return np.zeros(len(timestamps), dtype=np.float32)
        turn = reference.get("turn", {})
        profile_disagreement = float(turn.get("head_to_feet_yaw_rms_deg", float("nan")))
        if (
            self.options.biomechanical_branch_repair
            and (not np.isfinite(profile_disagreement) or profile_disagreement > 90.0)
        ):
            # The profile's turn prior is not safe to transfer when its own
            # head/feet yaw agreement is already near-antipodal.  In that
            # situation a prior is more likely to invent a body turn than to
            # resolve one.  Keep the measured tracker observations active and
            # let the temporal IK solve the base orientation instead.
            self._root_yaw_seed_diagnostics = {
                "enabled": False,
                "reason": "profile_head_feet_yaw_disagreement",
                "head_to_feet_yaw_rms_deg": profile_disagreement,
                "threshold_deg": 90.0,
            }
            return np.zeros(len(timestamps), dtype=np.float32)
        weights = np.asarray(turn.get("yaw_tracker_weights", (.40, .30, .30)), dtype=np.float64)
        if weights.shape != (3,) or not np.isfinite(weights).all() or weights.sum() <= 0:
            weights = np.asarray((.40, .30, .30), dtype=np.float64)
        weights *= np.asarray((
            self.options.head_tracker_constraint,
            self.options.foot_tracker_constraint,
            self.options.foot_tracker_constraint,
        ), dtype=np.float64)
        if not np.isfinite(weights).all() or weights.sum() <= 0:
            weights = np.asarray((.40, .30, .30), dtype=np.float64)
        weights /= weights.sum()
        angular_speed = max(float(turn.get("angular_speed_p75_deg_s", 75.0)), 10.0)
        smoothing_s = float(np.clip(12.0 / angular_speed, .10, .30))
        offsets = _coherent_yaw_offsets(
            target_rotation, valid, timestamps, weights, smoothing_s,
        )
        if self.options.biomechanical_branch_repair and len(offsets) > 1:
            # A raw quaternion can switch to the opposite local branch even
            # when the physical tracker turned only a few degrees.  Unwrap
            # the fused seed and limit its frame-to-frame increment before it
            # reaches the non-convex root solve. This does not remove a real
            # multi-turn sequence; it only rejects impossible one-frame spins.
            limited = np.unwrap(np.asarray(offsets, dtype=np.float64))
            max_speed = np.deg2rad(max(150.0, 2.0 * angular_speed))
            for index in range(1, len(limited)):
                dt = max(float(timestamps[index] - timestamps[index - 1]), 1e-4)
                step = np.clip(limited[index] - limited[index - 1], -max_speed * dt, max_speed * dt)
                limited[index] = limited[index - 1] + step
            offsets = _smooth_seconds(
                limited.astype(np.float32)[:, None], timestamps, smoothing_s,
            )[:, 0]
        self._root_yaw_seed_diagnostics = {
            "enabled": True,
            "source": "head_plus_consistent_ankle_so3_heading",
            "profile_head_to_feet_yaw_rms_deg": profile_disagreement,
        }
        return (min(float(self.options.locomotion_prior_weight), 1.0) * offsets).astype(np.float32)

    def _observed_protocol_priors(
        self, protocol: ProtocolAlignment | None, timestamps: np.ndarray,
        target_position: np.ndarray, latent: np.ndarray, correction: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Drive repeated calibration motions from measured endpoint geometry.

        A linear clock is a poor phase estimate when the operator holds one
        repetition longer than another.  In particular, it can command a leg
        back to the floor while its ankle tracker is still at the top of a
        raise.  Use tracker geometry to choose the reference-pose amplitude;
        optimization remains responsible for the exact pose.
        """
        if protocol is None:
            return latent, correction
        result_latent = latent.copy()
        result_correction = correction.copy()

        def blend(key: str, amount: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            amount = np.clip(amount.astype(np.float32), 0.0, 1.0)[:, None]
            amount = amount * amount * (3.0 - 2.0 * amount)
            neutral_latent = self.initial_latents["neutral"]
            neutral_correction = self.initial_corrections["neutral"]
            return (
                neutral_latent[None] + amount * (self.initial_latents[key] - neutral_latent)[None],
                neutral_correction[None] + amount * (self.initial_corrections[key] - neutral_correction)[None],
            )

        neutral_segment = next((item for item in protocol.segments if item.key == "neutral"), None)
        neutral_mask = np.ones(len(timestamps), dtype=bool) if neutral_segment is None else (
            (timestamps >= neutral_segment.active_start_s) & (timestamps <= neutral_segment.active_end_s)
        )
        if not neutral_mask.any():
            neutral_mask[: min(20, len(timestamps))] = True

        # 06: infer the alternating side and amplitude from each wrist's
        # measured height. This is more reliable than assuming that the
        # operator hits four equally spaced peaks, and it prevents a high
        # wrist from being explained by shoulder deformation on the wrong
        # side when the VPoser prior is deliberately strong.
        elbow_segment = next((item for item in protocol.segments if item.key == "elbows"), None)
        if elbow_segment is not None:
            active = (
                (timestamps >= elbow_segment.active_start_s)
                & (timestamps <= elbow_segment.active_end_s)
            )
            if active.any():
                base = self.initial_latents["t_pose"]
                driven = np.broadcast_to(base, (len(timestamps), len(base))).copy()
                for tracker_index, pose_key in ((1, "elbow_left"), (2, "elbow_right")):
                    height = target_position[:, tracker_index, 1]
                    low = float(np.percentile(height[active], 12))
                    high = float(np.percentile(height[active], 96))
                    amount = np.clip((height - low - .012) / max(high - low - .012, .04), 0.0, 1.0)
                    amount = amount * amount * (3.0 - 2.0 * amount)
                    driven += amount[:, None] * (self.initial_latents[pose_key] - base)[None]
                result_latent[active] = driven[active]

        # 07: the arms are raised only for the central wrist-articulation
        # block.  Blend the transition from the directly observed hand height
        # and separation so lowered hands are never seeded half a metre ahead.
        wrist_segment = next((item for item in protocol.segments if item.key == "wrists"), None)
        if wrist_segment is not None:
            stage = (timestamps >= wrist_segment.start_s) & (timestamps <= wrist_segment.end_s)
            if stage.any():
                separation = np.linalg.norm(target_position[:, 1] - target_position[:, 2], axis=1)
                below = .5 * (target_position[:, 1, 1] + target_position[:, 2, 1]) - target_position[:, 0, 1]
                height_amount = np.clip((below + .62) / .30, 0.0, 1.0)
                separation_amount = np.clip((.58 - separation) / .24, 0.0, 1.0)
                arm_amount = np.minimum(height_amount, separation_amount)
                wrist_latent, wrist_correction = blend("forward", arm_amount)
                # In manifold-only mode the active interval already carries
                # pronation/supination/flexion VPoser latents.  Replace only
                # the raise/lower transitions; otherwise we would erase the
                # very wrist motion that VPoser is meant to represent.
                replace_latent = stage
                if self.options.natural_pose_only:
                    active = (
                        (timestamps >= wrist_segment.active_start_s)
                        & (timestamps <= wrist_segment.active_end_s)
                    )
                    replace_latent = stage & ~active
                result_latent[replace_latent] = wrist_latent[replace_latent]
                result_correction[stage] = wrist_correction[stage]

        # 08--10: use head drop and left/right ankle height differences as
        # phase-free pose amplitudes.  This follows both repetitions even if
        # their peaks and holds are not evenly spaced in time.
        head_height = target_position[:, 0, 1] - .5 * (
            target_position[:, 3, 1] + target_position[:, 4, 1]
        )
        base_head_height = float(np.median(head_height[neutral_mask]))
        squat_segment = next((item for item in protocol.segments if item.key == "squat"), None)
        if squat_segment is not None:
            active = (timestamps >= squat_segment.active_start_s) & (timestamps <= squat_segment.active_end_s)
            if active.any():
                drop = base_head_height - head_height
                scale = max(float(np.percentile(drop[active], 95)), .06)
                values = np.clip((drop - .015) / max(scale - .015, .03), 0.0, 1.0)
                pose_latent, pose_correction = blend("squat", values)
                result_latent[active] = pose_latent[active]
                result_correction[active] = pose_correction[active]
        for segment_key, pose_key, lifted, support in (
            ("left_knee", "left_leg", 3, 4),
            ("right_knee", "right_leg", 4, 3),
        ):
            segment = next((item for item in protocol.segments if item.key == segment_key), None)
            if segment is None:
                continue
            active = (timestamps >= segment.active_start_s) & (timestamps <= segment.active_end_s)
            if not active.any():
                continue
            lift = target_position[:, lifted, 1] - target_position[:, support, 1]
            baseline = float(np.median(lift[neutral_mask]))
            height = lift - baseline
            scale = max(float(np.percentile(height[active], 96)), .055)
            values = np.clip((height - .012) / max(scale - .012, .03), 0.0, 1.0)
            pose_latent, pose_correction = blend(pose_key, values)
            result_latent[active] = pose_latent[active]
            result_correction[active] = pose_correction[active]

        # 16--17: use the actual ankle lifts to recover every pivot/walking
        # swing. This preserves the observed number and timing of steps rather
        # than concatenating a fixed reference-animation path into the fit.
        locomotion_mask = np.zeros(len(timestamps), dtype=bool)
        for segment_key in ("turns", "walk"):
            segment = next((item for item in protocol.segments if item.key == segment_key), None)
            if segment is not None:
                locomotion_mask |= (
                    (timestamps >= segment.active_start_s)
                    & (timestamps <= segment.active_end_s)
                )
        if locomotion_mask.any() and all(
            key in self.initial_latents
            for key in ("neutral", "walk_left_swing", "walk_right_swing")
        ):
            amounts = np.zeros((len(timestamps), 2), dtype=np.float32)
            neutral = self.initial_latents["neutral"]
            for side_index, tracker_index in enumerate((3, 4)):
                height = target_position[:, tracker_index, 1]
                baseline = float(np.median(height[neutral_mask]))
                observed_lift = height[locomotion_mask] - baseline
                lift_scale = max(float(np.percentile(observed_lift, 95)), .055)
                amount = np.clip(
                    (height - baseline - .012) / max(lift_scale - .012, .035),
                    0.0, 1.0,
                )
                amounts[:, side_index] = _smooth_seconds(
                    amount[:, None], timestamps, .12,
                )[:, 0]
            total = amounts.sum(axis=1)
            excessive = total > 1.0
            amounts[excessive] /= total[excessive, None]
            observed_gait = (
                neutral[None]
                + amounts[:, :1]
                * (self.initial_latents["walk_left_swing"] - neutral)[None]
                + amounts[:, 1:]
                * (self.initial_latents["walk_right_swing"] - neutral)[None]
            )
            result_latent[locomotion_mask] = observed_gait[locomotion_mask]
            result_correction[locomotion_mask] = 0.0
        return result_latent.astype(np.float32), result_correction.astype(np.float32)

    def _first_pose(self, target_position: np.ndarray, valid: np.ndarray) -> str:
        usable = valid[: min(12, len(valid)), 1] & valid[: min(12, len(valid)), 2]
        if usable.any():
            # This selects the numerically closest VPoser seed, not an action
            # label. Hands held close together benefit from the forward-arm
            # seed even when the wrists start below shoulder height.
            wrist_distance = np.linalg.norm(target_position[: len(usable), 1] - target_position[: len(usable), 2], axis=1)
            return "forward" if float(np.median(wrist_distance[usable])) < .35 else "neutral"
        return "neutral"

    def _initial_state(
        self, target_position, target_rotation, valid, session_id: str,
        initial_latent, pose_correction=None, root_yaw_offset=None,
        yaw_estimation_frames: int | None = None,
    ):
        import torch

        count = len(target_position)
        dtype = torch.float32
        latent = torch.as_tensor(initial_latent, dtype=dtype, device=self.device)
        if latent.ndim == 1:
            latent = latent[None].expand(count, -1).clone()
        elif len(latent) != count:
            raise ValueError(f"initial latent count {len(latent)} != target count {count}")
        with torch.no_grad():
            pose = self.vposer.decode(latent)["pose_body"].reshape(count, 21, 3)
            if pose_correction is not None:
                pose = pose + torch.as_tensor(pose_correction, dtype=dtype, device=self.device).reshape(count, 21, 3)
            zero3 = torch.zeros((count, 3), dtype=dtype, device=self.device)
            zero10 = torch.zeros((count, 10), dtype=dtype, device=self.device)
            zero_output = self.body(
                betas=self.betas[None].expand(count, -1), global_orient=zero3, transl=zero3,
                body_pose=pose.reshape(count, -1), left_hand_pose=torch.zeros((count, 45), device=self.device),
                right_hand_pose=torch.zeros((count, 45), device=self.device), jaw_pose=zero3,
                leye_pose=zero3, reye_pose=zero3, expression=zero10, return_verts=False,
            )
            zero_rotations = _global_body_rotations(zero3, pose, self.parents)[:, self.rotation_joint_ids]
            zero_positions = self.profile.scale * zero_output.joints[:, self.position_joint_ids]
            zero_positions += (zero_rotations @ self.mount_position[None, :, :, None]).squeeze(-1)
        if session_id in self.profile.session_yaw:
            yaw = self.profile.session_yaw[session_id]
            yaw_source = "calibrated_session"
        else:
            # Absolute Tracker heading contains the unknown pre-mount and was
            # previously able to choose the 180-degree left/right-swapped root.
            # Labelled endpoint correspondences determine yaw without relying
            # on Tracker local axes or on reuse of an old headband mounting.
            yaw = estimate_root_yaw_from_correspondences(
                zero_positions.cpu().numpy(), target_position, valid,
                maximum_frames=(
                    30 if yaw_estimation_frames is None else yaw_estimation_frames
                ),
            )
            yaw_source = "labelled_endpoint_procrustes"
        self._initial_yaw_diagnostics = {
            "source": yaw_source,
            "yaw_deg": float(np.rad2deg(yaw)),
            "estimation_frames": (
                None if session_id in self.profile.session_yaw else
                int(30 if yaw_estimation_frames is None else yaw_estimation_frames)
            ),
        }
        global_orient = torch.zeros((count, 3), dtype=dtype, device=self.device)
        if root_yaw_offset is None:
            global_orient[:, 1] = yaw
        else:
            offset = np.asarray(root_yaw_offset, dtype=np.float32)
            if offset.shape != (count,):
                raise ValueError(f"root yaw offset shape {offset.shape} != {(count,)}")
            # Keep the unwrapped scalar. Rodrigues accepts equivalent angles
            # outside [-pi, pi], and continuity across a 180-degree crossing
            # is far more important than choosing the shortest representation
            # independently in every frame.
            global_orient[:, 1] = torch.as_tensor(yaw + offset, dtype=dtype, device=self.device)
        latent_all = latent.clone()
        with torch.no_grad():
            output = self.body(
                betas=self.betas[None].expand(count, -1), global_orient=global_orient, transl=zero3,
                body_pose=pose.reshape(count, -1), left_hand_pose=torch.zeros((count, 45), device=self.device),
                right_hand_pose=torch.zeros((count, 45), device=self.device), jaw_pose=zero3, leye_pose=zero3,
                reye_pose=zero3, expression=zero10, return_verts=False,
            )
            rotations = _global_body_rotations(global_orient, pose, self.parents)[:, self.rotation_joint_ids]
            without_translation = self.profile.scale * output.joints[:, self.position_joint_ids]
            without_translation += (rotations @ self.mount_position[None, :, :, None]).squeeze(-1)
        base = without_translation.cpu().numpy()
        translation = np.full((count, 3), np.nan, dtype=np.float32)
        for frame_index in range(count):
            differences = target_position[frame_index, valid[frame_index]] - base[frame_index, valid[frame_index]]
            if len(differences):
                translation[frame_index] = np.median(differences, axis=0)
        # A recorder-wide dropout is occasionally present for a handful of
        # adjacent frames.  Tracker losses remain disabled there; interpolate
        # only the initial root seed so temporal optimization can bridge the
        # gap without introducing NaNs.
        observed = np.isfinite(translation).all(axis=1)
        if not observed.any():
            raise ValueError(f"{session_id}: no usable tracker observations")
        if not observed.all():
            indices = np.arange(count)
            for axis in range(3):
                translation[:, axis] = np.interp(indices, indices[observed], translation[observed, axis])
        return latent_all, global_orient, torch.as_tensor(translation, device=self.device)

    def _ground_context(
        self, target_position, target_rotation, valid, timestamps, initial, initial_latent,
        protocol_keys=None, pose_correction=None,
    ):
        """Estimate one horizontal floor and flat-foot contacts.

        Tracker height alone is not a floor measurement because the strap is
        above the sole. We use the calibrated rigid ankle transform to infer
        the anatomical ankle joint, then subtract the shaped SMPL-X ankle-to-
        sole distance. For locomotion, optional horizontal-speed gating avoids
        treating a low but still swinging foot as planted.
        """
        import torch

        dtype = torch.float32
        with torch.no_grad():
            latent_value = np.asarray(initial_latent, dtype=np.float32)
            if latent_value.ndim > 1:
                latent_value = latent_value[0]
            latent = torch.as_tensor(latent_value, dtype=dtype, device=self.device)[None]
            pose = self.vposer.decode(latent)["pose_body"].reshape(1, 21, 3)
            if pose_correction is not None:
                correction_value = np.asarray(pose_correction, dtype=np.float32)
                if correction_value.ndim > 1:
                    correction_value = correction_value[0]
                pose = pose + torch.as_tensor(correction_value, dtype=dtype, device=self.device).reshape(1, 21, 3)
            zero3 = torch.zeros((1, 3), device=self.device); zero10 = torch.zeros((1, 10), device=self.device)
            output = self.body(
                betas=self.betas[None], global_orient=initial[1][:1], transl=zero3,
                body_pose=pose.reshape(1, -1), left_hand_pose=torch.zeros((1, 45), device=self.device),
                right_hand_pose=torch.zeros((1, 45), device=self.device), jaw_pose=zero3,
                leye_pose=zero3, reye_pose=zero3, expression=zero10, return_verts=True,
            )
        model_joints = (self.profile.scale * output.joints[0]).cpu().numpy()
        model_floor = float((self.profile.scale * output.vertices[0, :, 1]).min().cpu())
        joint_height = np.asarray([
            (model_joints[7, 1] - model_floor, model_joints[10, 1] - model_floor),
            (model_joints[8, 1] - model_floor, model_joints[11, 1] - model_floor),
        ], dtype=np.float32)
        initial_mask = timestamps <= timestamps[0] + 6.0
        ground_candidates = []
        baseline_rotation = []
        baseline_height = []
        for side_index, tracker_index in enumerate((3, 4)):
            mask = initial_mask & valid[:, tracker_index]
            if not mask.any():
                baseline_rotation.append(np.eye(3)); baseline_height.append(float("nan")); continue
            rotations = target_rotation[mask, tracker_index]
            accumulator = rotations.mean(axis=0); u, _, vt = np.linalg.svd(accumulator); base_rotation = u @ vt
            if np.linalg.det(base_rotation) < 0:
                u[:, -1] *= -1; base_rotation = u @ vt
            baseline_rotation.append(base_rotation)
            baseline_height.append(float(np.median(target_position[mask, tracker_index, 1])))
            if self.options.lateral_ankle_mount:
                # During the initial upright hold, the lateral strap centre is
                # ankle_vertical_m above the anatomical ankle. Estimating that
                # vertical component through a tracker-to-joint rotation is
                # unnecessarily fragile (and previously placed the floor only
                # ~2 cm below the strap). The shaped ankle-to-sole distance is
                # metric, so this direct estimate matches the physical mount.
                tracker_height = joint_height[side_index, 0] + self.options.ankle_vertical_m
                ground_candidates.extend(
                    (target_position[mask, tracker_index, 1] - tracker_height).tolist()
                )
            else:
                observed_joint_rotation = np.einsum(
                    "fij,jk->fik",
                    rotations,
                    self.mount_rotation[tracker_index].cpu().numpy().T,
                )
                observed_joint = target_position[mask, tracker_index] - np.einsum(
                    "fij,j->fi", observed_joint_rotation, self.mount_position[tracker_index].cpu().numpy(),
                )
                ground_candidates.extend((observed_joint[:, 1] - joint_height[side_index, 0]).tolist())
        ground_y = float(np.median(ground_candidates)) if ground_candidates else model_floor + float(initial[2][0, 1].cpu())
        contact = np.zeros((len(timestamps), 2), dtype=bool)
        for side_index, tracker_index in enumerate((3, 4)):
            if not np.isfinite(baseline_height[side_index]):
                continue
            contact[:, side_index] = (
                valid[:, tracker_index]
                & (
                    target_position[:, tracker_index, 1]
                    <= baseline_height[side_index] + self.contact_height_threshold_m
                )
            )
            if self.contact_speed_threshold_m_s > 0 and len(timestamps) > 1:
                horizontal = target_position[:, tracker_index][:, (0, 2)]
                velocity = np.gradient(horizontal, timestamps, axis=0)
                speed = np.linalg.norm(velocity, axis=1)
                if len(speed) >= 3:
                    padded = np.pad(speed, (1, 1), mode="edge")
                    speed = np.median(
                        np.lib.stride_tricks.sliding_window_view(padded, 3), axis=1,
                    )
                contact[:, side_index] &= speed <= self.contact_speed_threshold_m_s
            contact[:, side_index] = _debounce_contact_mask(
                contact[:, side_index], timestamps,
                maximum_gap_s=min(.16, .35 * float(
                    self.profile.locomotion_reference.get("gait", {}).get("step_interval_s", .40)
                )),
            )
        if protocol_keys is not None:
            keys = np.asarray(protocol_keys)
            contact[keys == "squat"] = True
            contact[keys == "left_knee", 0] = False
            contact[keys == "left_knee", 1] = True
            contact[keys == "right_knee", 0] = True
            contact[keys == "right_knee", 1] = False
            contact[keys == "ankles"] = False
        contact_height = ground_y + joint_height
        if protocol_keys is None:
            # Ordinary recordings may contain metres of locomotion. Keep the
            # pelvis close to the robust tracker-derived root trajectory, not
            # to one fixed calibration spot on the floor.
            root_anchor_xz = _smooth_continuous(
                initial[2][:, (0, 2)].cpu().numpy(), timestamps,
                self.options.root_smoothing_window,
            )
        else:
            root_anchor_xz = np.median(
                initial[2][initial_mask][:, (0, 2)].cpu().numpy(), axis=0,
            ).astype(np.float32)
        return ground_y, contact_height, contact, root_anchor_xz

    def _fit_window(
        self, target_position, target_rotation, valid, timestamps, initial,
        contact, ground_y, contact_height, root_anchor_xz,
        latent_prior=None, global_prior=None, observation_factor=None, smooth_factor=None,
        target_rotation_baseline=None, joint_rotation_baseline=None, relative_rotation_weight=None,
        root_yaw_target=None,
        protocol_prior_factor=None,
        head_forward_weight=None,
        head_body_lock_weight=None,
        shoulder_lock_weight=None,
        elbow_lock_weight=None,
        distal_joint_mask=None,
        head_joint_mask=None,
        leg_joint_mask=None,
        torso_joint_mask=None,
        torso_bend_factor=None,
        pose_correction=None,
        previous_pose=None,
        previous_timestamp: float | None = None,
        boundary_factor: float = 1.0,
    ):
        import torch

        count = len(timestamps)
        dtype = torch.float32
        latent = torch.nn.Parameter(initial[0].clone())
        global_orient = torch.nn.Parameter(initial[1].clone())
        translation = torch.nn.Parameter(initial[2].clone())
        distal_residual = torch.nn.Parameter(torch.zeros((count, 4, 3), dtype=dtype, device=self.device))
        head_residual = torch.nn.Parameter(torch.zeros((count, 2, 3), dtype=dtype, device=self.device))
        leg_residual = torch.nn.Parameter(torch.zeros((count, 6, 3), dtype=dtype, device=self.device))
        torso_residual = torch.nn.Parameter(torch.zeros((count, 3, 3), dtype=dtype, device=self.device))
        full_body_factor = float(np.clip(
            (self.options.tracker_hardness - .75) / .25, 0.0, 1.0,
        )) if self.options.full_body_ik else 0.0
        full_body_residual = (
            torch.nn.Parameter(torch.zeros((count, 21, 3), dtype=dtype, device=self.device))
            if full_body_factor > 0 else None
        )
        if torso_joint_mask is None:
            torso_joint_mask = np.zeros((count, 3), dtype=np.float32)
        if torso_bend_factor is None:
            torso_bend_factor = np.zeros(count, dtype=np.float32)
        torso_gate = np.asarray(torso_joint_mask, dtype=np.float32) * np.asarray(
            torso_bend_factor, dtype=np.float32,
        )[:, None]
        torso_gate_t = torch.as_tensor(torso_gate, dtype=dtype, device=self.device)
        optimizer_parameters = [
            latent, global_orient, translation, distal_residual, head_residual,
            leg_residual, torso_residual,
        ]
        if full_body_residual is not None:
            optimizer_parameters.append(full_body_residual)
        optimizer = torch.optim.Adam(optimizer_parameters, lr=self.options.optimizer_learning_rate)
        zero3 = torch.zeros((count, 3), dtype=dtype, device=self.device)
        zero10 = torch.zeros((count, 10), dtype=dtype, device=self.device)
        tracker_weight = torch.as_tensor(
            (1.25, 1.0, 1.0, 1.15, 1.15), dtype=dtype, device=self.device,
        ) * self.tracker_constraint_multiplier
        adjacency = torch.as_tensor(np.diff(timestamps) < .45, dtype=dtype, device=self.device)
        triplet = adjacency[1:] * adjacency[:-1] if count > 2 else torch.zeros((0,), dtype=dtype, device=self.device)
        contact = torch.as_tensor(contact, dtype=dtype, device=self.device)
        contact_height = torch.as_tensor(contact_height, dtype=dtype, device=self.device)
        root_anchor_xz = torch.as_tensor(root_anchor_xz, dtype=dtype, device=self.device)
        latent_prior = initial[0].detach() if latent_prior is None else torch.as_tensor(latent_prior, dtype=dtype, device=self.device)
        global_prior = initial[1].detach() if global_prior is None else torch.as_tensor(global_prior, dtype=dtype, device=self.device)
        observation_factor = torch.ones(count, dtype=dtype, device=self.device) if observation_factor is None else torch.as_tensor(observation_factor, dtype=dtype, device=self.device)
        smooth_factor = torch.ones(count, dtype=dtype, device=self.device) if smooth_factor is None else torch.as_tensor(smooth_factor, dtype=dtype, device=self.device)
        target_rotation_baseline = None if target_rotation_baseline is None else torch.as_tensor(target_rotation_baseline, dtype=dtype, device=self.device)
        joint_rotation_baseline = None if joint_rotation_baseline is None else torch.as_tensor(joint_rotation_baseline, dtype=dtype, device=self.device)
        relative_rotation_weight = torch.zeros((count, len(TRACKER_NAMES)), dtype=dtype, device=self.device) if relative_rotation_weight is None else torch.as_tensor(relative_rotation_weight, dtype=dtype, device=self.device)
        root_yaw_target = None if root_yaw_target is None else torch.as_tensor(root_yaw_target, dtype=dtype, device=self.device)
        protocol_prior_factor = torch.zeros(count, dtype=dtype, device=self.device) if protocol_prior_factor is None else torch.as_tensor(protocol_prior_factor, dtype=dtype, device=self.device)
        head_forward_weight = torch.ones(count, dtype=dtype, device=self.device) if head_forward_weight is None else torch.as_tensor(head_forward_weight, dtype=dtype, device=self.device)
        head_body_lock_weight = torch.zeros(count, dtype=dtype, device=self.device) if head_body_lock_weight is None else torch.as_tensor(head_body_lock_weight, dtype=dtype, device=self.device)
        shoulder_lock_weight = torch.zeros(count, dtype=dtype, device=self.device) if shoulder_lock_weight is None else torch.as_tensor(shoulder_lock_weight, dtype=dtype, device=self.device)
        elbow_lock_weight = torch.zeros(count, dtype=dtype, device=self.device) if elbow_lock_weight is None else torch.as_tensor(elbow_lock_weight, dtype=dtype, device=self.device)
        distal_joint_mask = torch.zeros((count, 4), dtype=dtype, device=self.device) if distal_joint_mask is None else torch.as_tensor(distal_joint_mask, dtype=dtype, device=self.device)
        head_joint_mask = torch.zeros((count, 2), dtype=dtype, device=self.device) if head_joint_mask is None else torch.as_tensor(head_joint_mask, dtype=dtype, device=self.device)
        leg_joint_mask = torch.zeros((count, 6), dtype=dtype, device=self.device) if leg_joint_mask is None else torch.as_tensor(leg_joint_mask, dtype=dtype, device=self.device)
        pose_correction = torch.zeros((count, 21, 3), dtype=dtype, device=self.device) if pose_correction is None else torch.as_tensor(pose_correction, dtype=dtype, device=self.device).reshape(count, 21, 3)
        previous_pose = None if previous_pose is None else torch.as_tensor(
            previous_pose, dtype=dtype, device=self.device,
        ).reshape(21, 3)
        continuity_mask = torch.zeros((count, 21), dtype=dtype, device=self.device)
        continuity_mask[:, 17:21] = torch.maximum(
            continuity_mask[:, 17:21], distal_joint_mask,
        )
        continuity_mask[:, (11, 14)] = torch.maximum(
            continuity_mask[:, (11, 14)], head_joint_mask,
        )
        continuity_mask[:, (0, 1, 3, 4, 6, 7)] = torch.maximum(
            continuity_mask[:, (0, 1, 3, 4, 6, 7)], leg_joint_mask,
        )
        # VPoser is a natural-pose prior, not a guarantee against a rare
        # decoded SO(3) branch switch. Apply the hard speed barrier to all 21
        # joints, while retaining the lighter continuity preference only on
        # explicitly released IK chains above. This avoids over-smoothing the
        # natural body solution merely to catch one bad wrist frame.
        jump_mask = torch.ones((count, 21), dtype=dtype, device=self.device)
        with torch.no_grad():
            prior_pose = self.vposer.decode(latent_prior)["pose_body"].reshape(count, 21, 3) + pose_correction
            prior_rotations = _global_body_rotations(global_prior, prior_pose, self.parents)
        for iteration in range(self.options.iterations):
            optimizer.zero_grad(set_to_none=True)
            body_pose = self.vposer.decode(latent)["pose_body"].reshape(count, 21, 3) + pose_correction
            body_pose = body_pose.clone()
            if full_body_residual is not None:
                # Release every body joint in the hard-endpoint experiment.
                # VPoser remains the initialization and weak shape prior, but
                # no spine/shoulder/knee/ankle joint is structurally frozen.
                body_pose += full_body_factor * full_body_residual
            body_pose[:, 17:21] += distal_residual * distal_joint_mask[:, :, None]
            body_pose[:, (11, 14)] += head_residual * head_joint_mask[:, :, None]
            body_pose[:, (0, 1, 3, 4, 6, 7)] += leg_residual * leg_joint_mask[:, :, None]
            body_pose[:, (2, 5, 8)] += torso_residual * torso_gate_t[:, :, None]
            output = self.body(
                betas=self.betas[None].expand(count, -1), global_orient=global_orient, transl=zero3,
                body_pose=body_pose.reshape(count, -1), left_hand_pose=torch.zeros((count, 45), device=self.device),
                right_hand_pose=torch.zeros((count, 45), device=self.device), jaw_pose=zero3, leye_pose=zero3,
                reye_pose=zero3, expression=zero10, return_verts=False,
            )
            rotations = _global_body_rotations(global_orient, body_pose, self.parents)
            tracked_rotation = rotations[:, self.rotation_joint_ids]
            predicted = self.profile.scale * output.joints[:, self.position_joint_ids] + translation[:, None]
            predicted += (tracked_rotation @ self.mount_position[None, :, :, None]).squeeze(-1)
            error = torch.where(valid[:, :, None], predicted - target_position, torch.zeros_like(predicted))
            weights = valid * tracker_weight[None]
            weighted_tracker = weights * observation_factor[:, None]
            # Tracker mounting points are the primary observations, not a
            # stylistic soft cue. Keep a generous Huber transition only for
            # genuine tracking outliers; all normal centimetre-scale errors
            # remain in the quadratic (hard-constraint-like) region.
            position_loss = (_huber(error, .20).sum(dim=2) * weighted_tracker).sum() / (weighted_tracker.sum().clamp_min(1) * 3.0)
            predicted_rotation = tracked_rotation @ self.mount_rotation[None]
            rotation_weights = (
                valid * self.rotation_weight[None]
                * self.tracker_constraint_multiplier[None]
            )
            # Tracker hardness is a single physical stiffness control: it must
            # scale orientation evidence together with position evidence.
            # Keep a small non-zero floor at 0% so the unconstrained endpoint
            # remains numerically well-posed, while 100% is exactly the
            # configured rotation weight.
            rotation_hardness_factor = .05 + .95 * float(np.clip(self.options.tracker_hardness, 0.0, 1.0))
            rotation_loss = rotation_hardness_factor * (
                (_rotation_distance_squared(predicted_rotation, target_rotation) * rotation_weights).sum()
                / rotation_weights.sum().clamp_min(1)
            )
            relative_rotation_loss = torch.zeros((), dtype=dtype, device=self.device)
            if target_rotation_baseline is not None and joint_rotation_baseline is not None:
                predicted_tracker_baseline = joint_rotation_baseline @ self.mount_rotation[None]
                predicted_tracker = tracked_rotation @ self.mount_rotation[None]
                predicted_delta = predicted_tracker_baseline.transpose(-1, -2) @ predicted_tracker
                target_delta = target_rotation_baseline.transpose(-1, -2)[None] @ target_rotation
                relative_weights = (
                    valid * relative_rotation_weight
                    * self.tracker_constraint_multiplier[None]
                )
                relative_rotation_loss = rotation_hardness_factor * (
                    (_rotation_distance_squared(predicted_delta, target_delta) * relative_weights).sum()
                    / relative_weights.sum().clamp_min(1)
                )
            temporal = torch.zeros((), dtype=dtype, device=self.device)
            if count > 1 and adjacency.sum() > 0:
                pair_smooth = adjacency * torch.minimum(smooth_factor[1:], smooth_factor[:-1])
                denominator = pair_smooth.sum().clamp_min(1)
                temporal += (((latent[1:] - latent[:-1]).square().mean(dim=1)) * pair_smooth).sum() / denominator
                temporal += .25 * (((global_orient[1:] - global_orient[:-1]).square().mean(dim=1)) * pair_smooth).sum() / denominator
                temporal += 4.0 * (((translation[1:] - translation[:-1]).square().mean(dim=1)) * pair_smooth).sum() / denominator
                distal_pair_mask = torch.minimum(distal_joint_mask[1:], distal_joint_mask[:-1])
                distal_denominator = distal_pair_mask.sum().clamp_min(1)
                temporal += .15 * (
                    (distal_residual[1:] - distal_residual[:-1]).square().mean(dim=2)
                    * distal_pair_mask
                ).sum() / distal_denominator
                head_pair_mask = torch.minimum(head_joint_mask[1:], head_joint_mask[:-1])
                temporal += .12 * (
                    (head_residual[1:] - head_residual[:-1]).square().mean(dim=2) * head_pair_mask
                ).sum() / head_pair_mask.sum().clamp_min(1)
                leg_pair_mask = torch.minimum(leg_joint_mask[1:], leg_joint_mask[:-1])
                temporal += .12 * (
                    (leg_residual[1:] - leg_residual[:-1]).square().mean(dim=2) * leg_pair_mask
                ).sum() / leg_pair_mask.sum().clamp_min(1)
                if full_body_residual is not None:
                    temporal += self.options.full_body_joint_temporal_weight * (
                        (full_body_residual[1:] - full_body_residual[:-1]).square().mean(dim=2)
                    ).mean()
            # Sparse endpoint IK has multiple rotational branches.  Compare
            # corrected local joint rotations on SO(3), not their axis-angle
            # coordinates, and strongly reject implausible one-frame branch
            # changes while retaining a light nearest-solution preference for
            # ordinary motion.
            local_pose_rotation = _axis_angle_to_matrix(body_pose)
            joint_continuity = torch.zeros((), dtype=dtype, device=self.device)
            joint_jump = torch.zeros((), dtype=dtype, device=self.device)
            root_jump = torch.zeros((), dtype=dtype, device=self.device)
            # These four terms are an experimental *branch selector*.  They
            # are not tracker observations and are kept separate from the
            # established continuity loss so the production path is byte-for-
            # byte unaffected.  All calculations below are on SO(3), avoiding
            # the axis-angle wrapping that can turn a smooth 179/181 degree
            # sequence into a fake jump.
            branch_joint_smooth = torch.zeros((), dtype=dtype, device=self.device)
            branch_joint_acceleration = torch.zeros((), dtype=dtype, device=self.device)
            branch_leg_smooth = torch.zeros((), dtype=dtype, device=self.device)
            branch_root_acceleration = torch.zeros((), dtype=dtype, device=self.device)
            branch_root_alignment = torch.zeros((), dtype=dtype, device=self.device)
            if count > 1:
                pair_joint_mask = torch.minimum(continuity_mask[1:], continuity_mask[:-1])
                pair_joint_mask *= adjacency[:, None]
                pair_denominator = pair_joint_mask.sum().clamp_min(1)
                pair_jump_mask = torch.minimum(jump_mask[1:], jump_mask[:-1])
                pair_jump_mask *= adjacency[:, None]
                pair_angle_sq = _rotation_distance_squared(
                    local_pose_rotation[:-1], local_pose_rotation[1:],
                )
                pair_angle = torch.sqrt(pair_angle_sq + 1e-8)
                allowed = torch.as_tensor(
                    np.minimum(
                        self.options.maximum_joint_speed_rad_s * np.diff(timestamps),
                        np.deg2rad(42.0),
                    ), dtype=dtype, device=self.device,
                )[:, None]
                joint_continuity += (pair_angle_sq * pair_joint_mask).sum() / pair_denominator
                pair_excess = torch.relu(pair_angle - allowed)
                # Do not dilute one bad branch switch by every normal
                # frame/joint in the whole window. Average only over actual
                # speed-limit violations so the configured barrier retains
                # its intended magnitude for isolated 50--90 degree jumps.
                pair_violation = (
                    (pair_excess.detach() > 0).to(dtype) * pair_jump_mask
                )
                joint_jump += (
                    pair_excess.square() * pair_violation
                ).sum() / pair_violation.sum().clamp_min(1)
                if self.options.biomechanical_branch_repair:
                    # A speed limit of 9 rad/s with a 15 Hz stream permits a
                    # 34 degree frame-to-frame jump.  That is mathematically
                    # legal but visibly large for a standing calf or pelvis.
                    # The experimental branch uses a conservative limit only
                    # for selecting among already tracker-compatible minima;
                    # it does not clamp the final pose.
                    branch_allowed = torch.as_tensor(
                        np.minimum(
                            self.options.branch_max_joint_speed_rad_s * np.diff(timestamps),
                            np.deg2rad(self.options.branch_max_joint_step_deg),
                        ), dtype=dtype, device=self.device,
                    )[:, None]
                    branch_angle = torch.sqrt(pair_angle_sq + 1e-8)
                    branch_pair_mask = adjacency[:, None]
                    branch_pair_joint_mask = branch_pair_mask.expand(-1, branch_angle.shape[1])
                    branch_denominator = branch_pair_joint_mask.sum().clamp_min(1)
                    # Huber on the angle keeps ordinary fast motions possible,
                    # while still making a 20--35 degree branch less attractive
                    # than a nearby continuous solution.
                    branch_joint_smooth = (
                        _huber(branch_angle, .14) * branch_pair_joint_mask
                    ).sum() / branch_denominator
                    leg_ids = torch.as_tensor(
                        (0, 1, 3, 4, 6, 7), dtype=torch.long, device=self.device,
                    )
                    branch_leg_smooth = (
                        _huber(branch_angle[:, leg_ids], .11) * branch_pair_joint_mask[:, leg_ids]
                    ).sum() / branch_pair_joint_mask[:, leg_ids].sum().clamp_min(1)
                    pair_branch_excess = torch.relu(branch_angle - branch_allowed)
                    branch_violation = (
                        (pair_branch_excess.detach() > 0).to(dtype) * branch_pair_joint_mask
                    )
                    # Add a stricter leg/root-independent barrier. The old
                    # 42-degree production barrier remains in ``joint_jump``
                    # for compatibility and this term is only experimental.
                    joint_jump += (
                        pair_branch_excess.square() * branch_violation
                    ).sum() / branch_violation.sum().clamp_min(1)

                    if count > 2 and triplet.sum() > 0:
                        pair_relative = (
                            local_pose_rotation[:-1].transpose(-1, -2)
                            @ local_pose_rotation[1:]
                        )
                        pair_log = _matrix_to_axis_angle_torch(pair_relative)
                        # Work in per-observation increments rather than
                        # dividing by dt. The objective is to reject a branch
                        # switch between adjacent recorded frames; a 15 Hz
                        # stream otherwise amplifies a harmless 2--3 degree
                        # change into a very large rad/s^2 penalty.
                        acceleration = pair_log[1:] - pair_log[:-1]
                        acceleration_norm = torch.linalg.vector_norm(acceleration, dim=-1)
                        triple_mask = triplet[:, None]
                        triple_joint_mask = triple_mask.expand(-1, acceleration_norm.shape[1])
                        branch_joint_acceleration = (
                            _huber(acceleration_norm, .12) * triple_mask
                        ).sum() / triple_joint_mask.sum().clamp_min(1)
                        branch_leg_acceleration = (
                            _huber(acceleration_norm[:, leg_ids], .10)
                            * triple_joint_mask[:, leg_ids]
                        ).sum() / triple_joint_mask[:, leg_ids].sum().clamp_min(1)
                        # Leg chains are the user's dominant failure mode. Add
                        # their acceleration to the leg term without changing
                        # the public option name or the default path.
                        branch_leg_smooth = branch_leg_smooth + .65 * branch_leg_acceleration
                    # The same branch-speed barrier must also cover the root.
                    # Previously the diagnostic summary reported these large
                    # jumps as joint index 0, but the loss constrained only
                    # body_pose joints and allowed the pelvis to spin between
                    # two equally good sparse-tracker solutions.
                    root_rotation = _axis_angle_to_matrix(global_orient)
                    # The observed discontinuity is a yaw branch switch, not
                    # a legitimate whole-root SO(3) jump.  Use the projected
                    # forward-axis yaw so a real bend/pitch remains free.
                    # Standard yaw extraction from the root local +X axis;
                    # row-0 projection couples pitch/roll into yaw.
                    root_yaw = torch.atan2(-root_rotation[:, 2, 0], root_rotation[:, 0, 0])
                    root_yaw_delta = torch.atan2(
                        torch.sin(root_yaw[1:] - root_yaw[:-1]),
                        torch.cos(root_yaw[1:] - root_yaw[:-1]),
                    )
                    root_angle = torch.abs(root_yaw_delta)
                    if root_yaw_target is not None:
                        # Use a robust angle residual rather than cosine-only
                        # loss: cosine has zero gradient at the undesirable
                        # 180-degree branch. The position-derived target is a
                        # soft base-orientation cue; tracker endpoint losses
                        # remain the physical observations.
                        root_error = torch.atan2(
                            torch.sin(root_yaw - root_yaw_target),
                            torch.cos(root_yaw - root_yaw_target),
                        )
                        branch_root_alignment = _huber(root_error, .35).mean()
                    root_allowed = torch.as_tensor(
                        np.minimum(
                            self.options.branch_max_joint_speed_rad_s * np.diff(timestamps),
                            np.deg2rad(self.options.branch_max_joint_step_deg),
                        ), dtype=dtype, device=self.device,
                    )
                    root_excess = torch.relu(root_angle - root_allowed)
                    root_violation = (root_excess.detach() > 0).to(dtype) * adjacency
                    root_jump = (
                        root_excess.square() * root_violation
                    ).sum() / root_violation.sum().clamp_min(1)
                    if count > 2 and triplet.sum() > 0:
                        root_acceleration = torch.abs(root_yaw_delta[1:] - root_yaw_delta[:-1])
                        branch_root_acceleration = (
                            _huber(root_acceleration, .12) * triplet
                        ).sum() / triplet.sum().clamp_min(1)
            boundary_dt = (
                float(timestamps[0] - previous_timestamp)
                if previous_timestamp is not None else float("inf")
            )
            if previous_pose is not None and 0 < boundary_dt < .45:
                first_mask = continuity_mask[0]
                first_denominator = first_mask.sum().clamp_min(1)
                first_jump_mask = jump_mask[0]
                previous_rotation = _axis_angle_to_matrix(previous_pose)
                boundary_angle_sq = _rotation_distance_squared(
                    previous_rotation, local_pose_rotation[0],
                )
                boundary_angle = torch.sqrt(boundary_angle_sq + 1e-8)
                boundary_allowed = min(
                    (
                        self.options.branch_max_joint_speed_rad_s * max(boundary_dt, 1e-4)
                        if self.options.biomechanical_branch_repair
                        else self.options.maximum_joint_speed_rad_s * max(boundary_dt, 1e-4)
                    ),
                    float(np.deg2rad(
                        self.options.branch_max_joint_step_deg
                        if self.options.biomechanical_branch_repair else 42.0
                    )),
                )
                joint_continuity += (boundary_angle_sq * first_mask).sum() / first_denominator
                boundary_excess = torch.relu(boundary_angle - boundary_allowed)
                boundary_violation = (
                    (boundary_excess.detach() > 0).to(dtype) * first_jump_mask
                )
                joint_jump += (
                    boundary_excess.square() * boundary_violation
                ).sum() / boundary_violation.sum().clamp_min(1)
                if self.options.biomechanical_branch_repair:
                    # Window stitching is another common source of a single
                    # backward knee. Keep the first frame on the same SO(3)
                    # branch as the preceding fitted window for all joints,
                    # not only chains released by the current action mask.
                    boundary_all_angle = torch.sqrt(boundary_angle_sq + 1e-8)
                    branch_joint_smooth += _huber(
                        boundary_all_angle, .14,
                    ).mean()
            boundary = (latent[0] - initial[0][0]).square().mean()
            boundary += .25 * (global_orient[0] - initial[1][0]).square().mean()
            boundary += 2.0 * (translation[0] - initial[2][0]).square().mean()
            torso = output.joints[:, 12] - output.joints[:, 0]
            torso_up = torso[:, 1] / torch.linalg.vector_norm(torso, dim=1).clamp_min(1e-6)
            bend_gate = torch.as_tensor(
                np.asarray(torso_bend_factor, dtype=np.float32), dtype=dtype, device=self.device,
            )
            upright_multiplier = 1.0 - .90 * bend_gate
            upright = (
                torch.relu(float(np.cos(np.deg2rad(32.0))) - torso_up).square()
                * upright_multiplier
            ).mean()
            torso_bend_loss = torch.zeros((), dtype=dtype, device=self.device)
            if self.options.biomechanical_branch_repair and float(np.max(torso_bend_factor, initial=0.0)) > 0:
                torso_local = (
                    rotations[:, 0].transpose(-1, -2) @ torso[:, :, None]
                ).squeeze(-1)
                observed_head = target_position[:, 0]
                observed_feet = .5 * (target_position[:, 3] + target_position[:, 4])
                observed_local = (
                    rotations[:, 0].transpose(-1, -2)
                    @ (observed_head - observed_feet)[:, :, None]
                ).squeeze(-1)
                torso_pitch = torch.atan2(torso_local[:, 2], torso_local[:, 1].clamp_min(1e-5))
                observed_pitch = torch.atan2(observed_local[:, 2], observed_local[:, 1].clamp_min(1e-5))
                pitch_error = torch.atan2(
                    torch.sin(torso_pitch - observed_pitch),
                    torch.cos(torso_pitch - observed_pitch),
                )
                torso_bend_loss = (
                    pitch_error.square() * bend_gate
                ).sum() / bend_gate.sum().clamp_min(1.0)
            head_delta = output.joints[:, 15] - output.joints[:, 0]
            head_horizontal = torch.linalg.vector_norm(head_delta[:, (0, 2)], dim=1)
            head_vertical = torch.relu(head_horizontal - .24).square().mean()
            # Normal standing/squat calibration keeps the head close to the
            # sagittal centreline.  The previous 24 cm dead-zone allowed a
            # visibly side-shifted neck while still reporting zero penalty.
            head_delta_local = (
                rotations[:, 0].transpose(-1, -2) @ head_delta[:, :, None]
            ).squeeze(-1)
            head_center = torch.relu(torch.abs(head_delta_local[:, 0]) - .025).square().mean()
            protocol_prior = (
                (latent - latent_prior).square().mean(dim=1) * protocol_prior_factor
            ).mean()
            heading_prior = (global_orient - global_prior).square().mean()
            # The protocol asks the subject to look straight ahead outside
            # the deliberate 02 head stage. Sparse positions alone otherwise
            # permit a small but visually obvious systematic neck yaw.
            root_to_head = rotations[:, 0].transpose(-1, -2) @ rotations[:, 15]
            identity = torch.eye(3, dtype=dtype, device=self.device)[None].expand(count, -1, -1)
            head_forward = (
                _rotation_distance_squared(root_to_head, identity) * head_forward_weight
            ).sum() / head_forward_weight.sum().clamp_min(1)
            # During 02 the feet, pelvis and torso remain facing forward. The
            # head tracker delta must be explained by neck/head articulation,
            # not by rotating the entire sparse body solution.
            lock_denominator = head_body_lock_weight.sum().clamp_min(1)
            head_root_lock = (
                _rotation_distance_squared(rotations[:, 0], prior_rotations[:, 0])
                * head_body_lock_weight
            ).sum() / lock_denominator
            torso_ids = torch.as_tensor((3, 6, 9, 12), dtype=torch.long, device=self.device)
            head_torso_lock = (
                _rotation_distance_squared(rotations[:, torso_ids], prior_rotations[:, torso_ids]).mean(dim=1)
                * head_body_lock_weight
            ).sum() / lock_denominator
            shoulder_ids = torch.as_tensor((16, 17), dtype=torch.long, device=self.device)
            shoulder_denominator = shoulder_lock_weight.sum().clamp_min(1)
            proximal_shoulder_lock = (
                _rotation_distance_squared(rotations[:, shoulder_ids], prior_rotations[:, shoulder_ids]).mean(dim=1)
                * shoulder_lock_weight
            ).sum() / shoulder_denominator
            elbow_ids = torch.as_tensor((18, 19), dtype=torch.long, device=self.device)
            elbow_denominator = elbow_lock_weight.sum().clamp_min(1)
            proximal_elbow_lock = (
                _rotation_distance_squared(rotations[:, elbow_ids], prior_rotations[:, elbow_ids]).mean(dim=1)
                * elbow_lock_weight
            ).sum() / elbow_denominator
            world_joints = self.profile.scale * output.joints + translation[:, None]
            knee_hinge_loss, joint_range_loss, leg_crossing_loss = self._joint_constraint_losses(
                body_pose, world_joints, rotations,
            )
            sole_floor_loss = torch.zeros((), dtype=dtype, device=self.device)
            if self.options.sole_floor_constraint:
                for side_index, (ankle_id, toe_id, foot_rotation_id) in enumerate(
                    ((7, 10, 10), (8, 11, 11))
                ):
                    side_contact = contact[:, side_index]
                    if side_contact.sum() > 0:
                        # The shaped neutral ankle/toe-to-sole offsets are
                        # already included in contact_height. Their residuals
                        # are therefore the virtual heel/forefoot distances to
                        # the one shared floor plane, not raw joint heights.
                        heights = torch.stack((
                            world_joints[:, ankle_id, 1],
                            world_joints[:, toe_id, 1],
                        ), dim=1)
                        height_error = (
                            heights - contact_height[side_index]
                        ).square().mean(dim=1)
                        # Matching two longitudinal points fixes pitch but not
                        # roll. Align the foot frame's local +Y/sole normal to
                        # gravity as a small 10-cm-lever equivalent penalty.
                        foot_up = rotations[:, foot_rotation_id, :, 1]
                        normal_error = 2.0 * torch.relu(1.0 - foot_up[:, 1])
                        sole_floor_loss += (
                            (height_error + .01 * normal_error) * side_contact
                        ).sum() / side_contact.sum()
            # When consecutive observations agree that a foot is planted,
            # penalize horizontal ankle/toe velocity.  This is deliberately a
            # velocity constraint instead of a fixed world anchor: the latter
            # would be brittle at contact transitions and recorder gaps.
            foot_lock = torch.zeros((), dtype=dtype, device=self.device)
            if count > 1:
                pair_dt = torch.as_tensor(
                    np.maximum(np.diff(timestamps), 1e-4), dtype=dtype, device=self.device,
                )
                for side_index, (ankle_id, toe_id) in enumerate(((7, 10), (8, 11))):
                    planted_pair = (
                        contact[1:, side_index] * contact[:-1, side_index] * adjacency
                    )
                    if planted_pair.sum() > 0:
                        horizontal_velocity = (
                            world_joints[1:, (ankle_id, toe_id)][:, :, (0, 2)]
                            - world_joints[:-1, (ankle_id, toe_id)][:, :, (0, 2)]
                        ) / pair_dt[:, None, None]
                        foot_lock += (
                            horizontal_velocity.square().mean(dim=(1, 2)) * planted_pair
                        ).sum() / planted_pair.sum()
            # A flat floor is a one-sided constraint during foot lifts.
            penetration = torch.relu(
                torch.as_tensor(ground_y, dtype=dtype, device=self.device)
                - world_joints[:, torch.as_tensor((10, 11), device=self.device), 1]
            ).square().mean()
            support = (contact.sum(dim=1) > 0).to(dtype)
            root_delta = translation[:, (0, 2)] - root_anchor_xz
            root_distance = torch.linalg.vector_norm(root_delta, dim=1)
            root_stationary = ((root_delta.square().mean(dim=1)) * support).sum() / support.sum().clamp_min(1)
            root_bound = (
                torch.relu(root_distance - self.options.root_bound_m).square() * support
            ).sum() / support.sum().clamp_min(1)
            second_order = torch.zeros((), dtype=dtype, device=self.device)
            if count > 2 and triplet.sum() > 0:
                triple_smooth = triplet * torch.minimum(torch.minimum(smooth_factor[2:], smooth_factor[1:-1]), smooth_factor[:-2])
                denominator = triple_smooth.sum().clamp_min(1)
                second_order += (((translation[2:] - 2 * translation[1:-1] + translation[:-2]).square().mean(dim=1)) * triple_smooth).sum() / denominator
                second_order += .15 * (((global_orient[2:] - 2 * global_orient[1:-1] + global_orient[:-2]).square().mean(dim=1)) * triple_smooth).sum() / denominator
                second_order += .08 * (((latent[2:] - 2 * latent[1:-1] + latent[:-2]).square().mean(dim=1)) * triple_smooth).sum() / denominator
            # A logarithmic control gives the user a useful range without
            # changing the established 50% result: 0/50/100% correspond to
            # roughly 95/3,000/95,000 for the default base weight.  At the
            # hard end, tracker endpoints dominate the pose priors while the
            # latter remain non-zero to select a human-looking solution when
            # the sparse observations are underdetermined.
            hardness = float(self.options.tracker_hardness)
            effective_tracker_weight = self.options.tracker_position_weight * (10.0 ** (3.0 * (hardness - .5)))
            regularizer_scale = 2.0 ** (2.0 * (.5 - hardness))
            # Keep the branch selector relevant at the hard endpoint.  A
            # fixed small continuity weight is otherwise numerically drowned
            # by the ~95k tracker position coefficient at 100% hardness.  The
            # square-root growth is intentionally sub-linear: tracker errors
            # still win when the observations disagree, while a tie between
            # two IK branches is decided by continuity/anatomical plausibility.
            branch_weight_scale = (
                regularizer_scale * (1.0 + .025 * np.sqrt(effective_tracker_weight))
                if self.options.biomechanical_branch_repair else 0.0
            )
            # Let the sparse IK first approach the observed motion, then lock
            # the chosen SO(3) branch. A full-strength barrier from iteration
            # zero can trap an arbitrary raw recording in its coarse VPoser
            # seed even though the final trajectory itself would be smooth.
            jump_barrier_factor = float(np.clip(
                (iteration / max(self.options.iterations - 1, 1) - .25) / .35,
                0.0, 1.0,
            ))
            loss = (
                effective_tracker_weight * position_loss
                + self.options.tracker_rotation_weight * rotation_loss + 1.8 * relative_rotation_loss
                + self.options.temporal_weight * regularizer_scale * temporal + 18.0 * boundary_factor * boundary
                + regularizer_scale * (
                    self.options.pose_prior_weight if self.options.natural_pose_only else .025
                ) * latent.square().mean()
                + regularizer_scale * protocol_prior + .35 * regularizer_scale * heading_prior
                + 12.0 * head_forward
                + 45.0 * head_root_lock + 35.0 * head_torso_lock
                + 30.0 * proximal_shoulder_lock + 32.0 * proximal_elbow_lock
                + self.options.upright_weight * upright + 12.0 * head_vertical + 180.0 * head_center
                + self.options.torso_bend_weight * torso_bend_loss
                + self.options.foot_contact_weight * sole_floor_loss
                + self.options.foot_penetration_weight * penetration
                + self.options.foot_lock_weight * foot_lock
                + self.options.root_tracking_weight * root_stationary
                + self.options.root_bound_weight * root_bound
                + self.options.second_order_weight * second_order
                + self.options.joint_continuity_weight * joint_continuity
                + self.options.joint_jump_weight * jump_barrier_factor * joint_jump
                + .50 * self.options.joint_jump_weight * jump_barrier_factor * root_jump
                + branch_weight_scale * self.options.branch_joint_smooth_weight * branch_joint_smooth
                + branch_weight_scale * self.options.branch_joint_acceleration_weight * branch_joint_acceleration
                + branch_weight_scale * self.options.branch_leg_smooth_weight * branch_leg_smooth
                + branch_weight_scale * self.options.branch_root_acceleration_weight * branch_root_acceleration
                + branch_weight_scale * self.options.root_yaw_target_weight * branch_root_alignment
                + self.options.knee_hinge_weight * knee_hinge_loss
                + self.options.joint_range_weight * joint_range_loss
                + self.options.leg_crossing_weight * leg_crossing_loss
                + .018 * (
                    distal_residual.square().mean(dim=2) * distal_joint_mask
                ).sum() / distal_joint_mask.sum().clamp_min(1)
                + .025 * (
                    head_residual.square().mean(dim=2) * head_joint_mask
                ).sum() / head_joint_mask.sum().clamp_min(1)
                + self.options.leg_correction_weight * (
                    leg_residual.square().mean(dim=2) * leg_joint_mask
                ).sum() / leg_joint_mask.sum().clamp_min(1)
                + .020 * (
                    torso_residual.square().mean(dim=2) * torso_gate_t
                ).sum() / torso_gate_t.sum().clamp_min(1.0)
                + (
                    self.options.full_body_joint_weight
                    * full_body_factor
                    * full_body_residual.square().mean()
                    if full_body_residual is not None else 0.0
                )
            )
            loss.backward(); optimizer.step()
            with torch.no_grad():
                latent.clamp_(-3.5 if self.options.natural_pose_only else -4.5, 3.5 if self.options.natural_pose_only else 4.5)
                distal_residual.clamp_(-2.8, 2.8)
                head_residual.clamp_(-1.7, 1.7)
                leg_residual.clamp_(-1.8, 1.8)
                torso_residual.clamp_(-.45, .45)
                if full_body_residual is not None:
                    # Bound only the correction; the underlying SMPL-X joint
                    # remains fully free to flex within this neighborhood.
                    full_body_residual.clamp_(-1.6, 1.6)
        with torch.no_grad():
            pose = self.vposer.decode(latent)["pose_body"].reshape(count, 21, 3) + pose_correction
            if full_body_residual is not None:
                pose += full_body_factor * full_body_residual
            pose[:, 17:21] += distal_residual * distal_joint_mask[:, :, None]
            pose[:, (11, 14)] += head_residual * head_joint_mask[:, :, None]
            pose[:, (0, 1, 3, 4, 6, 7)] += leg_residual * leg_joint_mask[:, :, None]
            pose[:, (2, 5, 8)] += torso_residual * torso_gate_t[:, :, None]
        return latent.detach(), global_orient.detach(), translation.detach(), pose.detach()

    def _refine_protocol_articulation(
        self, pose: np.ndarray, global_orient: np.ndarray, translation: np.ndarray,
        timestamps: np.ndarray, target_position: np.ndarray, target_rotation: np.ndarray,
        valid: np.ndarray, protocol: ProtocolAlignment | None,
    ) -> np.ndarray:
        """Second-stage IK for head turns, elbow flexion and wrist motion.

        The main VPoser solve establishes a stable pelvis, torso, floor and
        shoulder configuration.  Starting a free elbow solve before that base
        exists can converge to the mirrored bending branch.  This refinement
        holds the base fixed and changes only the anatomical chain named by
        the protocol, so tracker residual cannot be reduced by rotating the
        torso or moving the pelvis.
        """
        if protocol is None:
            return pose
        import torch

        result = np.asarray(pose, dtype=np.float32).reshape(len(timestamps), 21, 3).copy()
        iterations = max(180, min(360, 2 * self.options.iterations + 100))

        def indices(key: str, tracker_indices: tuple[int, ...]) -> np.ndarray:
            segment = next((item for item in protocol.segments if item.key == key), None)
            if segment is None:
                return np.empty(0, dtype=np.int64)
            mask = (timestamps >= segment.active_start_s) & (timestamps <= segment.active_end_s)
            mask &= np.all(valid[:, tracker_indices], axis=1)
            return np.flatnonzero(mask)

        def body_trackers(
            body_pose: torch.Tensor, orient: torch.Tensor, transl: torch.Tensor,
            tracker_indices: tuple[int, ...],
        ) -> tuple[torch.Tensor, torch.Tensor]:
            count = len(body_pose)
            zero3 = torch.zeros((count, 3), dtype=torch.float32, device=self.device)
            zero10 = torch.zeros((count, 10), dtype=torch.float32, device=self.device)
            output = self.body(
                betas=self.betas[None].expand(count, -1), global_orient=orient, transl=zero3,
                body_pose=body_pose.reshape(count, -1),
                left_hand_pose=torch.zeros((count, 45), device=self.device),
                right_hand_pose=torch.zeros((count, 45), device=self.device),
                jaw_pose=zero3, leye_pose=zero3, reye_pose=zero3, expression=zero10,
                return_verts=False,
            )
            selection = torch.as_tensor(tracker_indices, dtype=torch.long, device=self.device)
            position_joint_ids = self.position_joint_ids[selection]
            rotation_joint_ids = self.rotation_joint_ids[selection]
            rotation = _global_body_rotations(orient, body_pose, self.parents)[:, rotation_joint_ids]
            predicted = self.profile.scale * output.joints[:, position_joint_ids] + transl[:, None]
            predicted += (rotation @ self.mount_position[selection, :, None]).squeeze(-1)
            return predicted, rotation

        # 02: permit the neck and head only.  Position is weighted strongly,
        # while tracker orientation prevents a positional match with a visibly
        # inverted or under-rotated face direction.
        selected = indices("head", (0,))
        if len(selected) >= 3:
            base = torch.as_tensor(result[selected], device=self.device)
            orient = torch.as_tensor(global_orient[selected], device=self.device)
            transl = torch.as_tensor(translation[selected], device=self.device)
            position_target = torch.as_tensor(target_position[selected, 0], device=self.device)
            rotation_target = torch.as_tensor(target_rotation[selected, 0], device=self.device)
            delta = torch.nn.Parameter(torch.zeros((len(selected), 2, 3), device=self.device))
            optimizer = torch.optim.Adam((delta,), lr=.018)
            for _ in range(iterations):
                optimizer.zero_grad(set_to_none=True)
                body_pose = base.clone()
                body_pose[:, 11] += delta[:, 0]
                body_pose[:, 14] += delta[:, 1]
                predicted, joint_rotation = body_trackers(body_pose, orient, transl, (0,))
                position_loss = (predicted[:, 0] - position_target).square().mean()
                predicted_rotation = joint_rotation[:, 0] @ self.mount_rotation[0]
                rotation_loss = _rotation_distance_squared(predicted_rotation, rotation_target).mean()
                smooth = (delta[1:] - delta[:-1]).square().mean()
                loss = 1800.0 * position_loss + 12.0 * rotation_loss + 3.0 * smooth + .2 * delta.square().mean()
                loss.backward(); optimizer.step()
                with torch.no_grad():
                    delta.clamp_(-1.5, 1.5)
            with torch.no_grad():
                refined = base.clone()
                refined[:, 11] += delta[:, 0]
                refined[:, 14] += delta[:, 1]
                result[selected] = refined.cpu().numpy()

        # 06: allow modest shoulder motion plus elbow and wrist articulation.
        # The torso and root remain fixed.  Wrist deltas are strongly penalized
        # because most endpoint displacement should come from elbow flexion.
        selected = indices("elbows", (1, 2))
        if len(selected) >= 3:
            base = torch.as_tensor(result[selected], device=self.device)
            orient = torch.as_tensor(global_orient[selected], device=self.device)
            transl = torch.as_tensor(translation[selected], device=self.device)
            position_target = torch.as_tensor(target_position[selected, 1:3], device=self.device)
            delta = torch.nn.Parameter(torch.zeros((len(selected), 6, 3), device=self.device))
            optimizer = torch.optim.Adam((delta,), lr=.022)
            for _ in range(iterations):
                optimizer.zero_grad(set_to_none=True)
                body_pose = base.clone()
                body_pose[:, 15:21] += delta
                predicted, _ = body_trackers(body_pose, orient, transl, (1, 2))
                position_loss = (predicted - position_target).square().mean()
                smooth = (delta[1:] - delta[:-1]).square().mean()
                regularizer = (
                    5.0 * delta[:, :2].square().mean()
                    + delta[:, 2:4].square().mean()
                    + 8.0 * delta[:, 4:].square().mean()
                )
                loss = 1500.0 * position_loss + 3.0 * smooth + .25 * regularizer
                loss.backward(); optimizer.step()
                with torch.no_grad():
                    delta[:, :2].clamp_(-.5, .5)
                    delta[:, 2:4].clamp_(-1.5, 1.5)
                    delta[:, 4:].clamp_(-.7, .7)
            with torch.no_grad():
                refined = base.clone()
                refined[:, 15:21] += delta
                result[selected] = refined.cpu().numpy()

        # 07: refine the same arm chain around the already established forward
        # pose.  Tracker *relative* rotation is used here because absolute hand
        # mount orientation is low-confidence; this preserves pronation,
        # supination and flexion without inventing a new mount frame.
        selected = indices("wrists", (1, 2))
        if len(selected) >= 3:
            base = torch.as_tensor(result[selected], device=self.device)
            orient = torch.as_tensor(global_orient[selected], device=self.device)
            transl = torch.as_tensor(translation[selected], device=self.device)
            position_target = torch.as_tensor(target_position[selected, 1:3], device=self.device)
            rotation_target = torch.as_tensor(target_rotation[selected, 1:3], device=self.device)
            with torch.no_grad():
                _, base_joint_rotation = body_trackers(base, orient, transl, (1, 2))
                joint_rotation_baseline = base_joint_rotation[0].clone()
                tracker_rotation_baseline = rotation_target[0].clone()
            delta = torch.nn.Parameter(torch.zeros((len(selected), 6, 3), device=self.device))
            optimizer = torch.optim.Adam((delta,), lr=.020)
            for _ in range(iterations):
                optimizer.zero_grad(set_to_none=True)
                body_pose = base.clone()
                body_pose[:, 15:21] += delta
                predicted, joint_rotation = body_trackers(body_pose, orient, transl, (1, 2))
                position_loss = (predicted - position_target).square().mean()
                predicted_delta = joint_rotation_baseline.transpose(-1, -2) @ joint_rotation
                target_delta = tracker_rotation_baseline.transpose(-1, -2) @ rotation_target
                rotation_loss = _rotation_distance_squared(predicted_delta, target_delta).mean()
                smooth = (delta[1:] - delta[:-1]).square().mean()
                regularizer = (
                    5.0 * delta[:, :2].square().mean()
                    + 4.0 * delta[:, 2:4].square().mean()
                    + .3 * delta[:, 4:].square().mean()
                )
                loss = 1500.0 * position_loss + 8.0 * rotation_loss + 3.0 * smooth + .25 * regularizer
                loss.backward(); optimizer.step()
                with torch.no_grad():
                    delta[:, :2].clamp_(-.4, .4)
                    delta[:, 2:4].clamp_(-.6, .6)
                    delta[:, 4:].clamp_(-2.8, 2.8)
            with torch.no_grad():
                refined = base.clone()
                refined[:, 15:21] += delta
                result[selected] = refined.cpu().numpy()
        return result.astype(np.float32)

    def _hard_bind_trajectory(
        self, trajectory: SmplTrajectory, target_position: np.ndarray,
        target_rotation: np.ndarray, valid: np.ndarray,
        root_yaw_target: np.ndarray | None = None,
    ) -> tuple[SmplTrajectory, np.ndarray]:
        """Project each fitted frame onto the rigid five-tracker endpoints.

        This is deliberately an opt-in diagnostic stage.  The ordinary fit is
        intentionally soft because real recordings contain outliers.  For
        debugging a suspected mount/IK problem, however, a high loss weight is
        not the same as an equality constraint: VPoser, temporal terms and
        post-fit root/foot projections can still trade a few millimetres of
        endpoint error for a lower prior cost.  Here the calibrated mount is
        frozen and only the anatomical chains that carry the five mounts are
        released.  The output is therefore a direct test of whether the body
        model can be pulled onto the measured tracker constellation without
        changing the SE(3) installation.  With
        ``hard_binding_rotation_exact`` enabled, the five measured tracker
        rotations are additionally imposed as SO(3) equalities inside this
        same solve.
        """
        import torch

        count = len(trajectory.timestamps)
        if count == 0:
            return trajectory, np.empty((0, len(TRACKER_NAMES)), dtype=np.float32)
        position_ids = self.position_joint_ids
        rotation_ids = self.rotation_joint_ids
        # body_pose index i is SMPL-X joint i+1 (pelvis/root is global_orient).
        # The legacy diagnostic releases only endpoint chains and keeps the
        # torso as an anchor.  That is useful for isolating an endpoint-chain
        # error, but it can make an otherwise feasible strict constellation
        # look impossible when the shaped skeleton differs by a few cm.  The
        # independent all-body mode below releases every local joint while
        # preserving the old chain-only default.
        released = np.zeros((21, 1), dtype=np.float32)
        released[[11, 14], 0] = 1.0              # neck, head
        released[15:21, 0] = 1.0                 # shoulders, elbows, wrists
        released[[0, 1, 3, 4, 6, 7], 0] = 1.0   # hips, knees, ankles
        strict_rotation = bool(self.options.hard_binding_rotation_exact)
        release_all_body = bool(self.options.hard_binding_release_all_body)
        if release_all_body:
            released[:, 0] = 1.0
        # The equality mode writes these tracked joints from raw tracker R;
        # their parent chains remain available for the endpoint position solve.
        # The foot body index follows the selected physical frame: 3/4 for
        # the lower-shank knee frames, or 6/7 for the explicit ankle-frame A/B.
        selected_rotation_joint_ids = tuple(int(value) for value in rotation_ids[-2:].detach().cpu().tolist())
        strict_joint_ids = (15, 20, 21, *selected_rotation_joint_ids)
        strict_body_indices = np.asarray(
            (14, 19, 20, *(joint_id - 1 for joint_id in selected_rotation_joint_ids)),
            dtype=np.int64,
        )
        if strict_rotation:
            released[strict_body_indices, 0] = 0.0
        release_mask = torch.as_tensor(
            released.reshape(1, 21, 1), dtype=torch.float32, device=self.device,
        )
        identity = torch.eye(3, dtype=torch.float32, device=self.device)
        target_position = np.asarray(target_position, dtype=np.float32)
        target_rotation = np.asarray(target_rotation, dtype=np.float32)
        valid = np.asarray(valid, dtype=bool)
        # A robust constellation centroid provides the expected root motion
        # for the temporal IK branch selector. It is not used to replace the
        # five endpoint equations; it only says that a stationary tracker
        # constellation should not be explained by a 10 cm root jump.
        centroid_valid = valid & np.isfinite(target_position).all(axis=2)
        centroid_weight_np = centroid_valid.astype(np.float32)
        safe_target_position = np.nan_to_num(
            target_position, nan=0.0, posinf=0.0, neginf=0.0,
        )
        target_centroid = (
            (safe_target_position * centroid_weight_np[:, :, None]).sum(axis=1)
            / centroid_weight_np.sum(axis=1, keepdims=True).clip(min=1.0)
        ).astype(np.float32)
        root_yaw_target = (
            None if root_yaw_target is None else
            np.asarray(root_yaw_target, dtype=np.float32).reshape(-1)
        )
        if root_yaw_target is not None and len(root_yaw_target) != count:
            raise ValueError("hard-binding root_yaw_target must match trajectory length")
        # Detect a squat from the observations themselves rather than from
        # frame numbers.  A large drop of the rigid head above a stationary
        # ankle pair is the characteristic cue; a speed gate prevents a
        # walking stride from being mistaken for a supported squat.  The gate
        # only activates an XZ pelvis-centering preference, leaving vertical
        # translation and all joint flexion unconstrained.
        squat_center_gate_np = np.zeros((count,), dtype=np.float32)
        if (
            self.options.hard_binding_squat_center_weight > 0
            or self.options.hard_binding_project_squat_center
        ) and count:
            safe_rotation = np.nan_to_num(target_rotation, nan=0.0, posinf=0.0, neginf=0.0)
            mount_rotation_np = self.mount_rotation.detach().cpu().numpy()
            target_global_rotation_np = np.einsum(
                "ftij,tjk->ftik", safe_rotation, np.transpose(mount_rotation_np, (0, 2, 1)),
            )
            mount_position_np = self.mount_position.detach().cpu().numpy()
            mount_offset_np = np.einsum(
                "ftij,tj->fti", target_global_rotation_np, mount_position_np,
            )
            target_joint_center_np = np.nan_to_num(target_position, nan=0.0) - mount_offset_np
            ankle_mid_np = .5 * (
                target_joint_center_np[:, 3] + target_joint_center_np[:, 4]
            )
            head_relative_height = target_joint_center_np[:, 0, 1] - ankle_mid_np[:, 1]
            usable_height = (
                valid[:, 0] & valid[:, 3] & valid[:, 4]
                & np.isfinite(head_relative_height)
            )
            if np.any(usable_height):
                upright_reference = float(np.nanpercentile(
                    head_relative_height[usable_height], 90.0,
                ))
                height_drop = upright_reference - head_relative_height
                height_gate = 1.0 / (
                    1.0 + np.exp(-(height_drop - .10) / .035)
                )
                height_gate[~usable_height] = 0.0
                timestamps_np = np.asarray(trajectory.timestamps, dtype=np.float64)
                if count > 1:
                    dt = np.diff(timestamps_np).clip(min=1e-3)
                    ankle_velocity = np.linalg.norm(np.diff(ankle_mid_np, axis=0), axis=1) / dt
                    ankle_velocity = np.concatenate((ankle_velocity, ankle_velocity[-1:]))
                else:
                    ankle_velocity = np.zeros((1,), dtype=np.float64)
                stable_gate = np.exp(-np.square(ankle_velocity / .15))
                squat_center_gate_np = np.clip(
                    height_gate * stable_gate, 0.0, 1.0,
                ).astype(np.float32)
        output_pose = np.asarray(trajectory.body_pose, dtype=np.float32).reshape(count, 21, 3).copy()
        output_transl = np.asarray(trajectory.transl, dtype=np.float32).copy()
        output_global = np.asarray(trajectory.global_orient, dtype=np.float32).copy()
        # In the strict diagnostic, optimize the whole recording jointly. A
        # finite-batch memory boundary is also a non-convex IK boundary: even
        # with a first-frame anchor, Adam can pick a different leg/torso branch
        # several frames later. The normal production fit remains windowed;
        # this opt-in hard solve is deliberately full-sequence so a turn or a
        # squat is never split into independent branch decisions.
        batch_size = count if int(self.options.hard_binding_batch_size) <= 0 else min(
            count, int(self.options.hard_binding_batch_size),
        )
        # This reference is sequence-global, rather than reset at every
        # optimization window.  Resetting it per 96-frame batch leaves a seam
        # where the shoulders/spine can pick a new yaw branch even though the
        # tracker constellation is continuous (the observed 50--120 frame
        # failure crossed exactly such a seam).
        torso_reference_shoulder = None
        torso_reference_spine = None
        torso_reference_hip = None
        torso_reference_shoulder_frame = None
        torso_reference_hip_frame = None
        torso_local_reference_rotation = None
        torso_local_reference_yaw = None
        for start in range(0, count, batch_size):
            end = min(count, start + batch_size)
            size = end - start
            base_pose = torch.as_tensor(output_pose[start:end], dtype=torch.float32, device=self.device)
            base_global = torch.as_tensor(output_global[start:end], dtype=torch.float32, device=self.device)
            root_target_available = bool(
                self.options.hard_binding_release_root and root_yaw_target is not None
            )
            if root_target_available:
                # The measured head+feet common yaw is the observable body-base
                # turn.  Apply it directly to the ordinary-fit root instead of
                # asking Adam to discover the same angle while endpoint losses
                # are simultaneously changing the spine.  This removes the
                # root/spine trade-off that produced 100-degree spine twists
                # and occasional 180-degree global branches.
                target_yaw = torch.as_tensor(
                    root_yaw_target[start:end], dtype=torch.float32, device=self.device,
                )
                if self.options.hard_binding_root_yaw_only:
                    # Freeze the initial pelvis pitch/roll and replace only
                    # its horizontal yaw by the measured common turn.  The
                    # ordinary fit's frame-wise root tilt is precisely the
                    # degree of freedom that produced the 30 cm squat drift.
                    first_matrix = _axis_angle_to_matrix(base_global[:1])
                    first_yaw = torch.atan2(
                        -first_matrix[:, 2, 0], first_matrix[:, 0, 0],
                    )
                    cf = torch.cos(first_yaw)
                    sf = torch.sin(first_yaw)
                    first_yaw_matrix = torch.zeros_like(first_matrix)
                    first_yaw_matrix[:, 0, 0] = cf
                    first_yaw_matrix[:, 0, 2] = sf
                    first_yaw_matrix[:, 1, 1] = 1.0
                    first_yaw_matrix[:, 2, 0] = -sf
                    first_yaw_matrix[:, 2, 2] = cf
                    fixed_tilt = first_yaw_matrix.transpose(-1, -2) @ first_matrix
                    c = torch.cos(target_yaw)
                    s = torch.sin(target_yaw)
                    yaw_matrix = torch.zeros((size, 3, 3), dtype=base_global.dtype, device=self.device)
                    yaw_matrix[:, 0, 0] = c
                    yaw_matrix[:, 0, 2] = s
                    yaw_matrix[:, 1, 1] = 1.0
                    yaw_matrix[:, 2, 0] = -s
                    yaw_matrix[:, 2, 2] = c
                    global_orient = _matrix_to_axis_angle_torch(
                        yaw_matrix @ fixed_tilt.expand(size, -1, -1)
                    ).detach()
                else:
                    base_matrix = _axis_angle_to_matrix(base_global)
                    base_yaw = torch.atan2(
                        -base_matrix[:, 2, 0], base_matrix[:, 0, 0],
                    )
                    yaw_delta = torch.atan2(
                        torch.sin(target_yaw - base_yaw),
                        torch.cos(target_yaw - base_yaw),
                    )
                    c = torch.cos(yaw_delta)
                    s = torch.sin(yaw_delta)
                    yaw_matrix = torch.zeros_like(base_matrix)
                    yaw_matrix[:, 0, 0] = c
                    yaw_matrix[:, 0, 2] = s
                    yaw_matrix[:, 1, 1] = 1.0
                    yaw_matrix[:, 2, 0] = -s
                    yaw_matrix[:, 2, 2] = c
                    # Start at the measured common turn, then allow a small
                    # optimizer correction for endpoint position feasibility.
                    global_orient = torch.nn.Parameter(
                        _matrix_to_axis_angle_torch(yaw_matrix @ base_matrix).detach()
                    )
            else:
                global_orient = (
                    torch.nn.Parameter(base_global.clone())
                    if self.options.hard_binding_release_root else base_global
                )
            base_translation = torch.as_tensor(output_transl[start:end], dtype=torch.float32, device=self.device)
            # The hard solve is evaluated in finite batches to keep the
            # SMPL-X graph bounded. Endpoint equality alone does not prevent
            # Adam from selecting a different translation/IK branch at the
            # first frame of the next batch. Keep explicit boundary anchors so
            # a squat or turn cannot acquire a discontinuity at a chunk seam.
            previous_translation = None
            previous_previous_translation = None
            previous_global = None
            if start > 0:
                previous_translation = torch.as_tensor(
                    output_transl[start - 1], dtype=torch.float32, device=self.device,
                )
                previous_global = torch.as_tensor(
                    output_global[start - 1], dtype=torch.float32, device=self.device,
                )
                if start > 1:
                    previous_previous_translation = torch.as_tensor(
                        output_transl[start - 2], dtype=torch.float32, device=self.device,
                    )
            target_pos = torch.as_tensor(target_position[start:end], dtype=torch.float32, device=self.device)
            target_centroid_t = torch.as_tensor(
                target_centroid[start:end], dtype=torch.float32, device=self.device,
            )
            target_rot_np = np.nan_to_num(
                target_rotation[start:end], nan=0.0, posinf=0.0, neginf=0.0,
            )
            target_rot = torch.as_tensor(target_rot_np, dtype=torch.float32, device=self.device)
            valid_t = torch.as_tensor(valid[start:end], dtype=torch.bool, device=self.device)
            # Convert the two foot tracker origins to ankle-joint centers
            # once per batch.  This is the same rigid SE(3) mount equation
            # used by the endpoint residual, so the pelvis projection below
            # cannot silently introduce a second foot convention.
            neutral_ankle = torch.as_tensor(
                self._neutral_ankle_mid, dtype=torch.float32, device=self.device,
            )
            adjacency = torch.as_tensor(
                np.diff(trajectory.timestamps[start:end]) < .45,
                dtype=torch.float32, device=self.device,
            )
            # tracker_world = joint_world @ mount, therefore the desired
            # global anatomical frame is tracker_world @ mount.T.
            target_global_rot = target_rot @ self.mount_rotation.transpose(-1, -2)[None]
            # Convert the tracker origins to ankle-joint centers in the same
            # anatomical frame.  ``mount_position`` is expressed in the
            # tracked joint frame, so using raw ``target_rot`` here would
            # omit the fitted tracker-to-joint rotation and create a false
            # horizontal pelvis offset during turns/squats.
            foot_mount_offset = (
                target_global_rot[:, 3:5]
                @ self.mount_position[None, 3:5, :, None]
            ).squeeze(-1)
            target_ankle_mid = .5 * (
                target_pos[:, 3] - foot_mount_offset[:, 0]
                + target_pos[:, 4] - foot_mount_offset[:, 1]
            )
            target_joint_center = target_pos - (
                target_global_rot @ self.mount_position[None, :, :, None]
            ).squeeze(-1)

            def enforce_strict_rotations(pose):
                """Keep each valid tracked joint globally equal to raw R."""
                if not strict_rotation:
                    return pose
                # Head, wrists and knees are not ancestors of one another, so
                # each local solution can use the current freely optimized
                # parent orientation without a recursive conflict.
                parent_rotations = _global_body_rotations(global_orient, pose, self.parents)
                constrained = pose.clone()
                for target_index, (joint_id, body_index) in enumerate(
                    zip(strict_joint_ids, strict_body_indices.tolist(), strict=True),
                ):
                    parent_id = int(self.parents[joint_id].item())
                    desired_local = (
                        parent_rotations[:, parent_id].transpose(-1, -2)
                        @ target_global_rot[:, target_index]
                    )
                    desired_axis_angle = _matrix_to_axis_angle_torch(desired_local)
                    constrained[:, body_index] = torch.where(
                        valid_t[:, target_index, None],
                        desired_axis_angle,
                        constrained[:, body_index],
                    )
                return constrained

            optimize_translation = bool(
                self.options.hard_binding_optimize_translation
                and self.options.hard_binding_release_all_body
            )
            project_translation = bool(
                self.options.hard_binding_project_translation
                and optimize_translation
            )
            pose_delta = torch.nn.Parameter(torch.zeros_like(base_pose))
            translation = (
                torch.nn.Parameter(base_translation.clone())
                if optimize_translation else base_translation
            )
            optimizer_parameters = [pose_delta]
            if optimize_translation and not project_translation:
                optimizer_parameters.append(translation)
            if self.options.hard_binding_release_root and not self.options.hard_binding_root_yaw_only:
                optimizer_parameters.append(global_orient)
            optimizer = torch.optim.Adam(
                optimizer_parameters,
                lr=float(self.options.hard_binding_learning_rate),
            )
            zero3 = torch.zeros((size, 3), dtype=torch.float32, device=self.device)
            zero10 = torch.zeros((size, 10), dtype=torch.float32, device=self.device)
            # Optional branch preconditioning.  A two-bone chain has two
            # equally valid elbow/knee solutions for the same endpoints.  If
            # Adam starts on the sideways solution, a later hinge barrier can
            # be too local to cross that basin.  Before tightening any
            # tracker equation, run a short limb-only pass on the released
            # hip/knee and shoulder/elbow joints.  This changes only the
            # initial point of the hard solve; it is not a tracker clamp and
            # is disabled by default.
            precondition_steps = int(
                self.options.hard_binding_limb_precondition_iterations
            )
            if (
                precondition_steps > 0
                and self.options.hard_binding_limb_center_weight > 0
                and not self.options.hard_binding_pure_equality
            ):
                seed_mask = torch.zeros_like(release_mask)
                seed_body_indices = (0, 1, 3, 4, 15, 16, 17, 18)
                seed_mask[:, seed_body_indices, :] = 1.0
                seed_mask = seed_mask * release_mask
                seed_optimizer = torch.optim.Adam((pose_delta,), lr=min(
                    float(self.options.hard_binding_learning_rate), .006,
                ))
                for _ in range(precondition_steps):
                    seed_optimizer.zero_grad(set_to_none=True)
                    seed_pose = enforce_strict_rotations(
                        base_pose + pose_delta * seed_mask
                    )
                    seed_model = self.body(
                        betas=self.betas[None].expand(size, -1),
                        global_orient=global_orient, transl=zero3,
                        body_pose=seed_pose.reshape(size, -1),
                        left_hand_pose=torch.zeros((size, 45), dtype=torch.float32, device=self.device),
                        right_hand_pose=torch.zeros((size, 45), dtype=torch.float32, device=self.device),
                        jaw_pose=zero3, leye_pose=zero3, reye_pose=zero3,
                        expression=zero10, return_verts=False,
                    )
                    seed_rotations = _global_body_rotations(
                        global_orient, seed_pose, self.parents,
                    )
                    seed_world = self.profile.scale * seed_model.joints + translation[:, None]
                    seed_root_local = (
                        seed_rotations[:, 0].transpose(-1, -2)
                        @ (seed_world - seed_world[:, :1]).transpose(-1, -2)
                    ).transpose(-1, -2)
                    seed_loss = torch.zeros((), device=self.device)
                    for leg_index, (hip, knee, ankle) in enumerate(((1, 4, 7), (2, 5, 8))):
                        denominator = (
                            seed_root_local[:, hip, 1] - seed_root_local[:, ankle, 1]
                        ).clamp_min(.05)
                        fraction = (
                            (seed_root_local[:, hip, 1] - seed_root_local[:, knee, 1])
                            / denominator
                        ).clamp(0.0, 1.0)
                        line_x = seed_root_local[:, hip, 0] + fraction * (
                            seed_root_local[:, ankle, 0] - seed_root_local[:, hip, 0]
                        )
                        lateral = seed_root_local[:, knee, 0] - line_x
                        neutral_lateral = float(self._neutral_knee_lateral[leg_index])
                        side = 1.0 if neutral_lateral >= 0.0 else -1.0
                        seed_loss = seed_loss + torch.relu(
                            .005 - side * lateral
                        ).square().mean()
                        center_error = torch.relu(
                            torch.abs(lateral - neutral_lateral)
                            - float(self.options.hard_binding_limb_center_deadzone_m)
                        )
                        seed_loss = seed_loss + float(
                            self.options.hard_binding_limb_center_weight
                        ) * center_error.square().mean()
                    for shoulder, elbow, wrist in ((16, 18, 20), (17, 19, 21)):
                        denominator = (
                            seed_root_local[:, shoulder, 1] - seed_root_local[:, wrist, 1]
                        ).clamp_min(.05)
                        fraction = (
                            (seed_root_local[:, shoulder, 1] - seed_root_local[:, elbow, 1])
                            / denominator
                        ).clamp(0.0, 1.0)
                        line_x = seed_root_local[:, shoulder, 0] + fraction * (
                            seed_root_local[:, wrist, 0] - seed_root_local[:, shoulder, 0]
                        )
                        elbow_lateral = seed_root_local[:, elbow, 0] - line_x
                        elbow_side = 1.0 if shoulder == 16 else -1.0
                        seed_loss = seed_loss + torch.relu(
                            .005 - elbow_side * elbow_lateral
                        ).square().mean()
                        seed_loss = seed_loss + float(
                            self.options.hard_binding_limb_center_weight
                        ) * torch.relu(
                            torch.abs(elbow_lateral)
                            - float(self.options.hard_binding_limb_center_deadzone_m)
                        ).square().mean()
                    if size > 1:
                        seed_loss = seed_loss + .05 * (
                            (pose_delta[1:] - pose_delta[:-1]) * seed_mask
                        ).square().mean()
                    seed_loss = seed_loss + .0005 * (pose_delta * seed_mask).square().mean()
                    seed_loss.backward()
                    seed_optimizer.step()
                    with torch.no_grad():
                        pose_delta.mul_(seed_mask)
                        pose_delta.clamp_(-1.8, 1.8)
            if self.options.hard_binding_right_knee_kick_deg != 0.0:
                # SMPL-X body-pose index 4 is the right knee (joint id 5).
                # Apply this only as an initial condition; every subsequent
                # iteration is still governed by the rigid endpoint
                # equations and the temporal/anatomical losses above.
                with torch.no_grad():
                    pose_delta[:, 4, 0] -= float(np.deg2rad(
                        self.options.hard_binding_right_knee_kick_deg
                    ))
                    pose_delta[:, 4, 0].clamp_(-1.8, 1.8)
            valid_float = valid_t.to(torch.float32)
            dynamic_foot_multiplier = torch.ones((size, 1), dtype=torch.float32, device=self.device)
            if float(self.options.hard_binding_dynamic_foot_weight) > 0:
                timestamps_batch = np.asarray(trajectory.timestamps[start:end], dtype=np.float64)
                if size > 1:
                    dt_batch = np.diff(timestamps_batch).clip(min=1e-3)
                    foot_speed = np.linalg.norm(
                        np.diff(target_position[start:end, 3:5], axis=0), axis=2,
                    ).max(axis=1) / dt_batch
                    foot_speed = np.concatenate((foot_speed, foot_speed[-1:]))
                else:
                    foot_speed = np.zeros((1,), dtype=np.float64)
                dynamic_foot_multiplier = torch.as_tensor(
                    1.0 + float(self.options.hard_binding_dynamic_foot_weight)
                    * np.clip(foot_speed / .50, 0.0, 4.0),
                    dtype=torch.float32, device=self.device,
                )[:, None]
            for iteration in range(int(self.options.hard_binding_iterations)):
                optimizer.zero_grad(set_to_none=True)
                pose = enforce_strict_rotations(base_pose + pose_delta * release_mask)
                model = self.body(
                    betas=self.betas[None].expand(size, -1),
                    global_orient=global_orient, transl=zero3,
                    body_pose=pose.reshape(size, -1),
                    left_hand_pose=torch.zeros((size, 45), dtype=torch.float32, device=self.device),
                    right_hand_pose=torch.zeros((size, 45), dtype=torch.float32, device=self.device),
                    jaw_pose=zero3, leye_pose=zero3, reye_pose=zero3,
                    expression=zero10, return_verts=False,
                )
                rotations = _global_body_rotations(global_orient, pose, self.parents)
                # For pelvis centering, use only the root's horizontal yaw.
                # Applying the full root SO(3) to the 1.15 m ankle lever turns
                # a harmless pelvis pitch/roll into a spurious 30 cm world-X/Z
                # offset.  Root pitch/roll are still constrained separately
                # by the anatomical tilt barrier below.
                root_yaw_center = torch.atan2(
                    -rotations[:, 0, 2, 0], rotations[:, 0, 0, 0],
                )
                c_center = torch.cos(root_yaw_center)
                s_center = torch.sin(root_yaw_center)
                yaw_center = torch.zeros_like(rotations[:, 0])
                yaw_center[:, 0, 0] = c_center
                yaw_center[:, 0, 2] = s_center
                yaw_center[:, 1, 1] = 1.0
                yaw_center[:, 2, 0] = -s_center
                yaw_center[:, 2, 2] = c_center
                predicted_no_translation = self.profile.scale * model.joints[:, position_ids]
                predicted_no_translation = predicted_no_translation + (
                    rotations[:, rotation_ids] @ self.mount_position[None, :, :, None]
                ).squeeze(-1)
                if project_translation:
                    safe_target = torch.nan_to_num(
                        target_pos, nan=0.0, posinf=0.0, neginf=0.0,
                    )
                    centroid_weight = valid_float[:, :, None]
                    translation = (
                        (safe_target - predicted_no_translation) * centroid_weight
                    ).sum(dim=1) / centroid_weight.sum(dim=1).clamp_min(1.0)
                effective_translation = translation
                if (
                    self.options.hard_binding_project_pelvis
                    and optimize_translation
                ):
                    # Place the pelvis at the horizontal target implied by
                    # the rigid ankle pair and shaped neutral ankle offset.
                    # Retain the optimizer's vertical translation: this is a
                    # horizontal root-centering projection, not a squat lock.
                    target_pelvis = target_ankle_mid - (
                        yaw_center @ (self.profile.scale * neutral_ankle)[None, :, None]
                    ).squeeze(-1)
                    effective_translation = translation.clone()
                    effective_translation[:, (0, 2)] = (
                        target_pelvis[:, (0, 2)]
                        - self.profile.scale * model.joints[:, 0, (0, 2)]
                    )
                if (
                    self.options.hard_binding_project_squat_center
                    and optimize_translation
                ):
                    # Apply the same pelvis target only where the measured
                    # head-height/foot-speed gate identifies a supported
                    # squat.  The blend is recomputed from the current body
                    # pose every iteration, so the endpoint equations can
                    # adjust the hips/spine/knees around a centered pelvis;
                    # standing and locomotion retain the ordinary root path.
                    target_pelvis = target_ankle_mid - (
                        yaw_center @ (self.profile.scale * neutral_ankle)[None, :, None]
                    ).squeeze(-1)
                    squat_gate_projection = torch.as_tensor(
                        squat_center_gate_np[start:end],
                        dtype=effective_translation.dtype, device=self.device,
                    ).clamp(0.0, 1.0)
                    projected_translation = translation.clone()
                    projected_translation[:, (0, 2)] = (
                        target_pelvis[:, (0, 2)]
                        - self.profile.scale * model.joints[:, 0, (0, 2)]
                    )
                    # Respect the isolated solve's translation deviation bound
                    # while still allowing the squat-specific center shift.
                    offset = projected_translation - translation
                    bound = float(self.options.hard_binding_translation_bound_m)
                    if bound > 0:
                        offset = offset.clamp(-bound, bound)
                    projected_translation = translation + offset
                    effective_translation = (
                        translation * (1.0 - squat_gate_projection[:, None])
                        + projected_translation * squat_gate_projection[:, None]
                    )
                predicted = predicted_no_translation + effective_translation[:, None]
                position_error = torch.where(
                    valid_t[:, :, None], predicted - target_pos,
                    torch.zeros_like(predicted),
                )
                position_weights = valid_float.clone()
                position_weights[:, 0] *= float(self.options.hard_binding_head_position_multiplier)
                position_weights[:, 1:3] *= float(self.options.hard_binding_hand_position_multiplier)
                position_weights[:, 3:5] *= float(self.options.hard_binding_foot_position_multiplier)
                position_weights[:, 3:5] *= dynamic_foot_multiplier
                position_loss = (
                    position_error.square().sum(dim=2) * position_weights
                ).sum() / (position_weights.sum().clamp_min(1.0) * 3.0)
                predicted_rotation = rotations[:, rotation_ids] @ self.mount_rotation[None]
                safe_target_rotation = torch.where(
                    valid_t[:, :, None, None], target_rot,
                    identity[None, None],
                )
                rotation_loss = (
                    _rotation_distance_squared(predicted_rotation, safe_target_rotation)
                    * valid_float
                ).sum() / valid_float.sum().clamp_min(1.0)
                torso_twist_loss = torch.zeros((), device=self.device)
                torso_alignment_loss = torch.zeros((), device=self.device)
                hard_anatomical_loss = torch.zeros((), device=self.device)
                hard_limb_plane_loss = torch.zeros((), device=self.device)
                hard_limb_center_loss = torch.zeros((), device=self.device)
                hard_analytic_limb_loss = torch.zeros((), device=self.device)
                hard_torso_direction_loss = torch.zeros((), device=self.device)
                hard_squat_torso_direction_loss = torch.zeros((), device=self.device)
                hard_proximal_anchor_loss = torch.zeros((), device=self.device)
                hard_torso_local_twist_loss = torch.zeros((), device=self.device)
                hard_crossing_loss = torch.zeros((), device=self.device)
                pelvis_ankle_center_loss = torch.zeros((), device=self.device)
                hard_squat_center_loss = torch.zeros((), device=self.device)
                if not self.options.hard_binding_pure_equality:
                    # Endpoint rotations alone are underdetermined: with a
                    # fixed pelvis, the optimizer can satisfy a turning head
                    # and hands by folding spine1/2/3 into an axial twist.
                    # Keep the shoulder and hip transverse axes in the same
                    # horizontal frame, and reject excessive local spine
                    # twist while leaving sagittal bending unconstrained.
                    world_joints = self.profile.scale * model.joints + effective_translation[:, None]
                    hip_axis = world_joints[:, 1] - world_joints[:, 2]
                    shoulder_axis = world_joints[:, 16] - world_joints[:, 17]
                    hip_yaw = torch.atan2(hip_axis[:, 2], hip_axis[:, 0])
                    shoulder_yaw = torch.atan2(shoulder_axis[:, 2], shoulder_axis[:, 0])
                    torso_yaw_error = torch.atan2(
                        torch.sin(shoulder_yaw - hip_yaw),
                        torch.cos(shoulder_yaw - hip_yaw),
                    )
                    torso_twist_loss = torch.relu(
                        torch.abs(torso_yaw_error) - float(np.deg2rad(25.0))
                    ).square().mean()
                    # ``pose`` excludes the pelvis/root: body-pose index i is
                    # SMPL-X joint i+1.  Thus spine1/2/3 (joint ids 3/6/9)
                    # live at pose indices 2/5/8, not 3/6/9 (which are the
                    # left knee/ankle/foot).
                    spine_twist = pose[:, (2, 5, 8), 1]
                    torso_twist_loss = torso_twist_loss + .35 * torch.relu(
                        torch.abs(spine_twist) - float(np.deg2rad(35.0))
                    ).square().mean()
                    cumulative_spine_twist = spine_twist.sum(dim=1)
                    torso_twist_loss = torso_twist_loss + .15 * torch.relu(
                        torch.abs(cumulative_spine_twist) - float(np.deg2rad(60.0))
                    ).square().mean()
                    # The shoulder and upper-spine axes must undergo the same
                    # common-mode yaw as the pelvis.  Without this term, a
                    # turning sequence can leave the root/hips facing the new
                    # direction while the shoulder girdle stays in the old
                    # frame (or twists by the opposite amount) to absorb hand
                    # endpoint errors.  ``root_yaw`` uses the SMPL-X global
                    # x-axis convention, hence the plus sign with the
                    # anatomical transverse-axis yaw below.
                    root_matrix_for_torso = rotations[:, 0]
                    root_yaw_for_torso = torch.atan2(
                        -root_matrix_for_torso[:, 2, 0],
                        root_matrix_for_torso[:, 0, 0],
                    )
                    spine_axis = rotations[:, 9, :, 0]  # spine3 global x-axis
                    spine_yaw = torch.atan2(spine_axis[:, 2], spine_axis[:, 0])
                    # A yaw-only constraint is insufficient: during the
                    # previous squat branch the shoulder line remained at
                    # the correct horizontal yaw but rolled almost vertical
                    # (one shoulder was ~18 cm above the other).  That is a
                    # valid way to move the hands toward their endpoints, but
                    # it is not a human torso.  Keep the *3-D transverse
                    # frames* of the hip and shoulder girdles close to their
                    # first-frame anatomical frame.  Forward bending is still
                    # free because this constrains only the left-right axes,
                    # not the pelvis-to-spine vector.
                    root_local_hip_axis = (
                        root_matrix_for_torso.transpose(-1, -2) @ hip_axis[..., None]
                    ).squeeze(-1)
                    root_local_shoulder_axis = (
                        root_matrix_for_torso.transpose(-1, -2) @ shoulder_axis[..., None]
                    ).squeeze(-1)
                    root_local_hip_axis = root_local_hip_axis / torch.linalg.vector_norm(
                        root_local_hip_axis, dim=-1, keepdim=True,
                    ).clamp_min(1e-6)
                    root_local_shoulder_axis = root_local_shoulder_axis / torch.linalg.vector_norm(
                        root_local_shoulder_axis, dim=-1, keepdim=True,
                    ).clamp_min(1e-6)
                    if torso_reference_shoulder is None:
                        torso_reference_shoulder = (
                            shoulder_yaw[0] + root_yaw_for_torso[0]
                        ).detach()
                        torso_reference_spine = (
                            spine_yaw[0] + root_yaw_for_torso[0]
                        ).detach()
                        torso_reference_hip = (
                            hip_yaw[0] + root_yaw_for_torso[0]
                        ).detach()
                        torso_reference_shoulder_frame = root_local_shoulder_axis[0].detach()
                        torso_reference_hip_frame = root_local_hip_axis[0].detach()
                    shoulder_root_error = torch.atan2(
                        torch.sin(
                            shoulder_yaw + root_yaw_for_torso
                            - torso_reference_shoulder
                        ),
                        torch.cos(
                            shoulder_yaw + root_yaw_for_torso
                            - torso_reference_shoulder
                        ),
                    )
                    spine_root_error = torch.atan2(
                        torch.sin(
                            spine_yaw + root_yaw_for_torso
                            - torso_reference_spine
                        ),
                        torch.cos(
                            spine_yaw + root_yaw_for_torso
                            - torso_reference_spine
                        ),
                    )
                    hip_root_error = torch.atan2(
                        torch.sin(
                            hip_yaw + root_yaw_for_torso
                            - torso_reference_hip
                        ),
                        torch.cos(
                            hip_yaw + root_yaw_for_torso
                            - torso_reference_hip
                        ),
                    )
                    # The common turn is observable from the five rigid
                    # tracker rotations, but the endpoint positions alone do
                    # not identify how that turn is distributed through the
                    # pelvis, spine and shoulder girdle.  A penalty only on
                    # the axial frame errors makes the torso follow that
                    # measured turn while leaving sagittal bending and all
                    # endpoint-chain joints free.  Unlike the old 12-degree
                    # barrier this is centred on the initial anatomical
                    # frame, so a 180-degree turn cannot be represented by a
                    # stationary shoulder frame plus compensating arms.
                    torso_alignment_loss = (
                        hip_root_error.square().mean()
                        + shoulder_root_error.square().mean()
                        + spine_root_error.square().mean()
                    ) / 3.0
                    # Use a modest fraction of the common-frame loss so a
                    # real forward squat can still change the torso/root
                    # pitch, while a lateral shoulder roll cannot be traded
                    # against hand endpoint error.  The transverse axis is a
                    # rigid anatomical frame; it should not acquire a large
                    # vertical component merely because an endpoint is hard
                    # to reach.
                    torso_alignment_loss = torso_alignment_loss + .05 * (
                        (root_local_shoulder_axis - torso_reference_shoulder_frame).square().mean()
                        + (root_local_hip_axis - torso_reference_hip_frame).square().mean()
                    ) / 2.0
                    # A small residual is allowed for the shaped model's
                    # neutral S-curve, but a larger allowance lets the
                    # shoulder girdle visibly lag the pelvis during a turn.
                    torso_frame_limit = float(np.deg2rad(12.0))
                    torso_twist_loss = torso_twist_loss + torch.relu(
                        torch.abs(shoulder_root_error) - torso_frame_limit
                    ).square().mean()
                    torso_twist_loss = torso_twist_loss + torch.relu(
                        torch.abs(spine_root_error) - torso_frame_limit
                    ).square().mean()
                    if self.options.biomechanical_branch_repair:
                        # The ordinary fit has these barriers too, but the
                        # strict endpoint solve used to omit them.  That made
                        # the final equality projection free to select a
                        # mirrored IK branch: left/right knees or elbows
                        # could cross while all five tracked endpoints still
                        # matched.  Evaluate the barriers on the *same*
                        # world joints and root-local frame used by the
                        # endpoint equations so they remain active at 100%
                        # tracker binding.
                        root_local = (
                            rotations[:, 0].transpose(-1, -2)
                            @ (world_joints - world_joints[:, :1]).transpose(-1, -2)
                        ).transpose(-1, -2)
                        if self.options.hard_binding_proximal_anchor_weight > 0:
                            # These joints carry the torso/base branch.  Keep
                            # them near the already tracker-informed ordinary
                            # solution while allowing distal chains to move to
                            # the exact rigid endpoints.
                            proximal_ids = torch.as_tensor(
                                (0, 1, 2, 5, 8), dtype=torch.long, device=self.device,
                            )
                            proximal_rotation = _axis_angle_to_matrix(
                                pose[:, proximal_ids]
                            )
                            proximal_base_rotation = _axis_angle_to_matrix(
                                base_pose[:, proximal_ids]
                            )
                            hard_proximal_anchor_loss = _rotation_distance_squared(
                                proximal_rotation, proximal_base_rotation,
                            ).mean()
                        if self.options.hard_binding_torso_local_twist_weight > 0:
                            # Measure *relative local SO(3)* axial change from
                            # the first body frame.  Subtracting axis-angle Y
                            # components directly is not a valid rotation
                            # difference and, more importantly, comparing to
                            # ``base_pose`` at the same frame merely anchors
                            # the already counter-rotated solution.  The log
                            # map of R0.T @ Rt isolates the added local twist
                            # while leaving sagittal squat/bend components
                            # free.  Gate it by measured common yaw so an
                            # ordinary straight walk is unaffected.
                            torso_local_ids = torch.as_tensor(
                                (0, 1, 2, 5, 8, 11, 12, 13, 15, 16),
                                dtype=torch.long, device=self.device,
                            )
                            if root_yaw_target is not None:
                                target_yaw_local = torch.as_tensor(
                                    root_yaw_target[start:end],
                                    dtype=torch.float32, device=self.device,
                                )
                                if torso_local_reference_yaw is None:
                                    torso_local_reference_yaw = target_yaw_local[:1].detach()
                                turn_delta = torch.atan2(
                                    torch.sin(target_yaw_local - torso_local_reference_yaw),
                                    torch.cos(target_yaw_local - torso_local_reference_yaw),
                                )
                                turn_gate = torch.clamp(
                                    torch.abs(turn_delta) / float(np.deg2rad(15.0)),
                                    0.0, 1.0,
                                )
                            else:
                                turn_gate = torch.zeros((size,), dtype=torch.float32, device=self.device)
                            local_torso_rotation = _axis_angle_to_matrix(
                                pose[:, torso_local_ids]
                            )
                            if torso_local_reference_rotation is None:
                                torso_local_reference_rotation = _axis_angle_to_matrix(
                                    base_pose[:1, torso_local_ids]
                                ).detach()
                            local_torso_reference = torso_local_reference_rotation
                            local_torso_relative = (
                                local_torso_reference.transpose(-1, -2)
                                @ local_torso_rotation
                            )
                            local_twist_delta = _matrix_to_axis_angle_torch(
                                local_torso_relative
                            )[..., 1]
                            hard_torso_local_twist_loss = (
                                local_twist_delta.square().mean(dim=1) * turn_gate
                            ).sum() / turn_gate.sum().clamp_min(1.0)
                        if self.options.hard_binding_torso_direction_weight > 0:
                            # Both vectors are expressed in world coordinates:
                            # the measured one uses the same frozen SE(3)
                            # mount inversion as the endpoint equations, and
                            # the solved one uses the pelvis/head joints.  A
                            # cosine residual constrains direction only, so a
                            # genuine squat can still shorten the projection
                            # of the torso without being forced to a neutral
                            # pose or height.
                            observed_torso = target_joint_center[:, 0] - target_ankle_mid
                            solved_torso = world_joints[:, 15] - world_joints[:, 0]
                            observed_torso = observed_torso / torch.linalg.vector_norm(
                                observed_torso, dim=-1, keepdim=True,
                            ).clamp_min(1e-6)
                            solved_torso = solved_torso / torch.linalg.vector_norm(
                                solved_torso, dim=-1, keepdim=True,
                            ).clamp_min(1e-6)
                            direction_cosine = (
                                observed_torso * solved_torso
                            ).sum(dim=-1).clamp(-1.0, 1.0)
                            direction_mask = (
                                valid_t[:, 0] & valid_t[:, 3] & valid_t[:, 4]
                            ).to(world_joints.dtype)
                            hard_torso_direction_loss = (
                                (1.0 - direction_cosine) * direction_mask
                            ).sum() / direction_mask.sum().clamp_min(1.0)
                        if self.options.hard_binding_squat_torso_direction_weight > 0:
                            # During a supported squat, use the observed
                            # ankle-midpoint→head direction as a geometric
                            # cue for the pelvis→head chain.  Unlike the
                            # all-frame torso cue above, this cannot suppress
                            # a genuine turning or reaching motion.
                            solved_torso_direction = world_joints[:, 15] - world_joints[:, 0]
                            solved_torso_direction = solved_torso_direction / torch.linalg.vector_norm(
                                solved_torso_direction, dim=-1, keepdim=True,
                            ).clamp_min(1e-6)
                            observed_head_direction = target_joint_center[:, 0] - target_ankle_mid
                            observed_head_direction = observed_head_direction / torch.linalg.vector_norm(
                                observed_head_direction, dim=-1, keepdim=True,
                            ).clamp_min(1e-6)
                            squat_gate_direction = torch.as_tensor(
                                squat_center_gate_np[start:end],
                                dtype=world_joints.dtype, device=self.device,
                            )
                            squat_direction_mask = (
                                valid_t[:, 0] & valid_t[:, 3] & valid_t[:, 4]
                            ).to(world_joints.dtype) * squat_gate_direction
                            squat_direction_cosine = (
                                solved_torso_direction * observed_head_direction
                            ).sum(dim=-1).clamp(-1.0, 1.0)
                            hard_squat_torso_direction_loss = (
                                (1.0 - squat_direction_cosine) * squat_direction_mask
                            ).sum() / squat_direction_mask.sum().clamp_min(1.0)
                        if self.options.hard_binding_analytic_limb_weight > 0:
                            # Build the two-bone solution in the current root
                            # frame.  For a known parent and measured endpoint
                            # there are two circle intersections; choosing the
                            # +sagittal branch makes the knee/elbow bend
                            # forward instead of allowing a sideways mirror.
                            target_local = (
                                rotations[:, 0].transpose(-1, -2)
                                @ (target_joint_center - world_joints[:, :1]).transpose(-1, -2)
                            ).transpose(-1, -2)

                            def analytic_two_bone(parent, middle, endpoint_index, lengths, valid_index):
                                parent_point = root_local[:, parent]
                                endpoint_point = target_local[:, endpoint_index]
                                delta = endpoint_point - parent_point
                                distance = torch.linalg.vector_norm(delta, dim=-1).clamp_min(.05)
                                direction = delta / distance[:, None]
                                first_length = float(self.profile.scale * lengths[0])
                                second_length = float(self.profile.scale * lengths[1])
                                along = (
                                    first_length * first_length
                                    - second_length * second_length
                                    + distance * distance
                                ) / (2.0 * distance)
                                along = along.clamp(0.0, first_length)
                                height = torch.sqrt(
                                    torch.relu(first_length * first_length - along * along)
                                )
                                # +Z is the forward sagittal half-space in
                                # this SMPL-X/root convention.  The normal is
                                # entirely in the YZ plane, so the target has
                                # no artificial lateral knee/elbow kick.
                                normal = torch.stack(
                                    (
                                        torch.zeros_like(direction[:, 0]),
                                        -direction[:, 2],
                                        direction[:, 1],
                                    ), dim=-1,
                                )
                                normal = normal / torch.linalg.vector_norm(
                                    normal, dim=-1, keepdim=True,
                                ).clamp_min(1e-6)
                                normal = torch.where(
                                    (normal[:, 2:3] < 0.0), -normal, normal,
                                )
                                target_middle = parent_point + along[:, None] * direction + height[:, None] * normal
                                error = (root_local[:, middle] - target_middle).square().sum(dim=-1)
                                mask = valid_t[:, valid_index].to(error.dtype)
                                return (error * mask).sum() / mask.sum().clamp_min(1.0)

                            for leg_index, (hip, knee, ankle) in enumerate(((1, 4, 7), (2, 5, 8))):
                                hard_analytic_limb_loss = hard_analytic_limb_loss + analytic_two_bone(
                                    hip, knee, leg_index + 3, self._neutral_limb_lengths[leg_index], leg_index + 3,
                                )
                            for arm_index, (shoulder, elbow, wrist) in enumerate(((16, 18, 20), (17, 19, 21))):
                                hard_analytic_limb_loss = hard_analytic_limb_loss + float(
                                    self.options.hard_binding_analytic_arm_scale
                                ) * analytic_two_bone(
                                    shoulder, elbow, arm_index + 1, self._neutral_arm_lengths[arm_index], arm_index + 1,
                                )
                        # Preserve anatomical left/right ordering at the
                        # proximal, middle and distal points.  The margins
                        # are smaller than the normal stance widths, so they
                        # do not prevent a narrow stance; they only reject a
                        # crossed-through-body solution.
                        for left_id, right_id, margin in (
                            (1, 2, .055), (4, 5, .040), (7, 8, .045),
                            (16, 17, .045), (18, 19, .030), (20, 21, .020),
                        ):
                            separation = root_local[:, left_id, 0] - root_local[:, right_id, 0]
                            hard_crossing_loss = hard_crossing_loss + torch.relu(
                                float(margin) - separation
                            ).square().mean()
                        # Prevent a true segment pass-through, which can
                        # occur even when the corresponding left/right joint
                        # x coordinates retain their order. Use the same
                        # clearance margins as the ordinary branch selector;
                        # tracker endpoints remain the dominant equality term.
                        segment_pairs = (
                            ((1, 4), (2, 5)), ((1, 4), (5, 8)),
                            ((4, 7), (2, 5)), ((4, 7), (5, 8)),
                            ((16, 18), (17, 19)), ((16, 18), (19, 21)),
                            ((18, 20), (17, 19)), ((18, 20), (19, 21)),
                        )
                        for pair_index, ((first_start, first_end), (second_start, second_end)) in enumerate(segment_pairs):
                            clearance = .035 if pair_index < 4 else .025
                            distance_squared = _segment_distance_squared(
                                world_joints[:, first_start], world_joints[:, first_end],
                                world_joints[:, second_start], world_joints[:, second_end],
                            )
                            hard_crossing_loss = hard_crossing_loss + torch.relu(
                                float(clearance * clearance) - distance_squared
                            ).square().mean()

                        def hard_joint_angle(first, middle, last):
                            first_vector = world_joints[:, first] - world_joints[:, middle]
                            second_vector = world_joints[:, last] - world_joints[:, middle]
                            denominator = (
                                torch.linalg.vector_norm(first_vector, dim=-1)
                                * torch.linalg.vector_norm(second_vector, dim=-1)
                            ).clamp_min(1e-6)
                            cosine = (first_vector * second_vector).sum(dim=-1) / denominator
                            return torch.acos(cosine.clamp(-1.0 + 1e-5, 1.0 - 1e-5))

                        # A knee must stay on its own side of the hip--ankle
                        # line and bend toward the normal forward sagittal
                        # half-space.  The angle floor blocks a folded leg;
                        # it still permits a full squat (the lower bound is
                        # deliberately broad at 55 degrees).
                        knee_min = float(np.deg2rad(55.0))
                        for leg_index, (hip, knee, ankle) in enumerate(((1, 4, 7), (2, 5, 8))):
                            denominator = (root_local[:, hip, 1] - root_local[:, ankle, 1]).clamp_min(.05)
                            fraction = (
                                (root_local[:, hip, 1] - root_local[:, knee, 1]) / denominator
                            ).clamp(0.0, 1.0)
                            line_x = root_local[:, hip, 0] + fraction * (
                                root_local[:, ankle, 0] - root_local[:, hip, 0]
                            )
                            lateral = root_local[:, knee, 0] - line_x
                            neutral_lateral = float(self._neutral_knee_lateral[leg_index])
                            side = 1.0 if neutral_lateral >= 0.0 else -1.0
                            hard_limb_plane_loss = hard_limb_plane_loss + torch.relu(
                                .008 - side * lateral
                            ).square().mean()
                            hard_limb_plane_loss = hard_limb_plane_loss + torch.relu(
                                torch.abs(lateral)
                                - float(self.options.hard_binding_knee_lateral_limit_m)
                            ).square().mean()
                            if self.options.hard_binding_limb_center_weight > 0:
                                center_error = torch.relu(
                                    torch.abs(lateral - neutral_lateral)
                                    - float(self.options.hard_binding_limb_center_deadzone_m)
                                )
                                hard_limb_center_loss = hard_limb_center_loss + float(
                                    self.options.hard_binding_limb_center_weight
                                ) * center_error.square().mean()
                            line_z = root_local[:, hip, 2] + fraction * (
                                root_local[:, ankle, 2] - root_local[:, hip, 2]
                            )
                            sagittal = root_local[:, knee, 2] - line_z
                            hard_anatomical_loss = hard_anatomical_loss + torch.relu(
                                -.015 - sagittal
                            ).square().mean()
                            hard_anatomical_loss = hard_anatomical_loss + .35 * torch.relu(
                                knee_min - hard_joint_angle(hip, knee, ankle)
                            ).square().mean()

                        # Elbows use the opposite sagittal half-space from
                        # knees.  This removes the mirrored elbow branch while
                        # allowing the arm to be straight or flexed normally.
                        elbow_min = float(np.deg2rad(20.0))
                        for shoulder, elbow, wrist in ((16, 18, 20), (17, 19, 21)):
                            denominator = (root_local[:, shoulder, 1] - root_local[:, wrist, 1]).clamp_min(.05)
                            fraction = (
                                (root_local[:, shoulder, 1] - root_local[:, elbow, 1]) / denominator
                            ).clamp(0.0, 1.0)
                            line_z = root_local[:, shoulder, 2] + fraction * (
                                root_local[:, wrist, 2] - root_local[:, shoulder, 2]
                            )
                            sagittal = root_local[:, elbow, 2] - line_z
                            # Elbows should stay on their own shoulder-wrist
                            # side and within a moderate lateral excursion;
                            # otherwise the endpoint equality can be met by
                            # swinging an arm through the torso.
                            line_x = root_local[:, shoulder, 0] + fraction * (
                                root_local[:, wrist, 0] - root_local[:, shoulder, 0]
                            )
                            elbow_lateral = root_local[:, elbow, 0] - line_x
                            elbow_side = 1.0 if shoulder == 16 else -1.0
                            hard_limb_plane_loss = hard_limb_plane_loss + (
                                torch.relu(.005 - elbow_side * elbow_lateral).square()
                                + torch.relu(
                                    torch.abs(elbow_lateral)
                                    - float(self.options.hard_binding_elbow_lateral_limit_m)
                                ).square()
                            ).mean()
                            if self.options.hard_binding_limb_center_weight > 0:
                                center_error = torch.relu(
                                    torch.abs(elbow_lateral)
                                    - float(self.options.hard_binding_limb_center_deadzone_m)
                                )
                                hard_limb_center_loss = hard_limb_center_loss + float(
                                    self.options.hard_binding_limb_center_weight
                                ) * center_error.square().mean()
                            hard_anatomical_loss = hard_anatomical_loss + torch.relu(
                                sagittal - .015
                            ).square().mean()
                            hard_anatomical_loss = hard_anatomical_loss + .25 * torch.relu(
                                elbow_min - hard_joint_angle(shoulder, elbow, wrist)
                            ).square().mean()

                        # A squat may lean forward, but a near-horizontal
                        # pelvis-to-upper-spine vector is a fall/branch, not a
                        # human squat. Keep this broad so normal bending is
                        # unaffected while rejecting the 400--500 frame
                        # face-plant solution seen in the previous output.
                        torso_vector = root_local[:, 9] - root_local[:, 0]
                        torso_forward_lean = torch.abs(torch.atan2(
                            torso_vector[:, 2], torso_vector[:, 1].clamp_min(1e-5),
                        ))
                        torso_side_lean = torch.abs(torch.atan2(
                            torso_vector[:, 0], torso_vector[:, 1].clamp_min(1e-5),
                        ))
                        # The head tracker is a rigid orientation endpoint,
                        # but the neck remains a free joint.  Without an
                        # axial barrier, a squat can satisfy the head pose by
                        # spinning the neck roughly 100 degrees while the
                        # pelvis/spine stay upright.  Allow a broad human
                        # head turn, but reject that compensation branch.
                        spine3_yaw = torch.atan2(
                            -rotations[:, 9, 2, 0], rotations[:, 9, 0, 0],
                        )
                        neck_yaw = torch.atan2(
                            -rotations[:, 12, 2, 0], rotations[:, 12, 0, 0],
                        )
                        neck_twist = torch.atan2(
                            torch.sin(neck_yaw - spine3_yaw),
                            torch.cos(neck_yaw - spine3_yaw),
                        )
                        hard_anatomical_loss = hard_anatomical_loss + .65 * torch.relu(
                            torch.abs(neck_twist) - float(np.deg2rad(
                                self.options.hard_binding_neck_twist_limit_deg
                            ))
                        ).square().mean()
                        neck_relative_limit = float(np.deg2rad(
                            self.options.hard_binding_neck_relative_angle_limit_deg
                        ))
                        neck_relative_spine = torch.sqrt(
                            _rotation_distance_squared(
                                rotations[:, 9], rotations[:, 12],
                            ) + 1e-8
                        )
                        neck_relative_head = torch.sqrt(
                            _rotation_distance_squared(
                                rotations[:, 12], rotations[:, 15],
                            ) + 1e-8
                        )
                        hard_anatomical_loss = hard_anatomical_loss + .80 * (
                            torch.relu(neck_relative_spine - neck_relative_limit).square()
                            + torch.relu(neck_relative_head - neck_relative_limit).square()
                        ).mean()
                        hard_anatomical_loss = hard_anatomical_loss + .50 * torch.relu(
                            torso_forward_lean - float(np.deg2rad(58.0))
                        ).square().mean()
                        # Keep each upper-spine segment within a broad
                        # anatomical bend range, and discourage adjacent
                        # segments from bending in opposite directions.  The
                        # latter is the signature of the observed 400--480
                        # frame S-shaped compensation; it is not needed for a
                        # normal squat, where the bend is shared smoothly by
                        # spine1/2/3.
                        spine_local = pose[:, (2, 5, 8)]
                        spine_angle = torch.linalg.vector_norm(spine_local, dim=-1)
                        spine_limit = float(np.deg2rad(
                            self.options.hard_binding_spine_bend_limit_deg
                        ))
                        hard_anatomical_loss = hard_anatomical_loss + .70 * torch.relu(
                            spine_angle - spine_limit
                        ).square().mean()
                        spine_bend_axes = spine_local[..., (0, 2)]
                        # A forward squat rotates the torso about its
                        # transverse x-axis and therefore keeps that axis in
                        # the root horizontal plane.  A roll/sideways branch
                        # instead gives spine1/2/3 a large vertical x-axis
                        # component (the failure seen around 400--480).
                        spine_frame_local = (
                            rotations[:, 0].transpose(-1, -2)[:, None]
                            @ rotations[:, (3, 6, 9)]
                        )
                        spine_axis_tilt = torch.abs(spine_frame_local[:, :, 1, 0])
                        spine_axis_limit = float(np.sin(np.deg2rad(
                            self.options.hard_binding_spine_axis_tilt_limit_deg
                        )))
                        spine_axis_weights = torch.as_tensor(
                            (0.45, 0.70, 1.0), dtype=world_joints.dtype, device=self.device,
                        )
                        hard_anatomical_loss = hard_anatomical_loss + (
                            torch.relu(spine_axis_tilt - spine_axis_limit).square()
                            * spine_axis_weights[None]
                        ).mean()
                        if spine_bend_axes.shape[1] > 1:
                            spine_direction_change = spine_bend_axes[:, 1:] - spine_bend_axes[:, :-1]
                            hard_anatomical_loss = hard_anatomical_loss + float(
                                self.options.hard_binding_spine_direction_weight
                            ) * (
                                spine_direction_change.square().mean()
                            )
                            if self.options.hard_binding_spine_opposite_weight > 0:
                                # The observed failure is a sign reversal in
                                # the main sagittal (local X) bend: spine1
                                # and spine3 take opposite X rotations while
                                # the middle segment stays moderate.  Use a
                                # component-wise barrier here instead of a
                                # full vector-angle penalty; the latter also
                                # penalizes harmless changes in bend plane and
                                # was found to move the solve to a higher-RMS
                                # branch.
                                bend_x = spine_bend_axes[..., 0]
                                bend_x_squashed = torch.tanh(bend_x)
                                opposite_adjacent = torch.relu(
                                    -bend_x_squashed[:, 1:] * bend_x_squashed[:, :-1]
                                )
                                # The problematic branch can leave spine2
                                # near zero while spine1/spine3 fold in
                                # opposite directions.  Include the outer
                                # pair explicitly; otherwise an adjacent
                                # sign test misses exactly that S pattern.
                                opposite_outer = torch.relu(
                                    -bend_x_squashed[:, :1] * bend_x_squashed[:, -1:]
                                )
                                opposite = torch.cat(
                                    (opposite_adjacent, opposite_outer), dim=1
                                )
                                hard_anatomical_loss = hard_anatomical_loss + float(
                                    self.options.hard_binding_spine_opposite_weight
                                ) * opposite.square().mean()
                        if self.options.hard_binding_spine_curve_weight > 0:
                            spine_points = world_joints[:, (0, 3, 6, 9, 12, 15)]
                            spine_segments = spine_points[:, 1:] - spine_points[:, :-1]
                            spine_segments = spine_segments / torch.linalg.vector_norm(
                                spine_segments, dim=-1, keepdim=True
                            ).clamp_min(1e-6)
                            spine_cos = (
                                spine_segments[:, 1:] * spine_segments[:, :-1]
                            ).sum(dim=-1).clamp(-1.0 + 1e-5, 1.0 - 1e-5)
                            spine_curve = torch.acos(spine_cos)
                            curve_limit = float(np.deg2rad(
                                self.options.hard_binding_spine_curve_limit_deg
                            ))
                            hard_anatomical_loss = hard_anatomical_loss + float(
                                self.options.hard_binding_spine_curve_weight
                            ) * torch.relu(
                                spine_curve - curve_limit
                            ).square().mean()
                        hard_anatomical_loss = hard_anatomical_loss + .30 * torch.relu(
                            torso_side_lean - float(np.deg2rad(
                                self.options.hard_binding_torso_side_lean_limit_deg
                            ))
                        ).square().mean()
                        # A free global pelvis can absorb the endpoint
                        # equations by pitching/rolling the entire base.  At
                        # a 1.15 m leg lever this turns a modest 15-degree
                        # root tilt into a 30 cm horizontal pelvis error.  A
                        # squat should bend at the hips/knees/spine instead;
                        # keep the pelvis up-axis close to world +Y while
                        # retaining free root yaw and a small sagittal tilt.
                        root_up = rotations[:, 0, :, 1]
                        root_tilt = torch.acos(root_up[:, 1].clamp(-1.0 + 1e-5, 1.0 - 1e-5))
                        hard_anatomical_loss = hard_anatomical_loss + .80 * torch.relu(
                            root_tilt - float(np.deg2rad(
                                self.options.hard_binding_root_tilt_limit_deg
                            ))
                        ).square().mean()
                        # Endpoint equality alone does not identify the root
                        # translation.  During the bad squat branch the
                        # pelvis moved ~0.55 m sideways from the ankle pair,
                        # then the spine/head folded back to the measured
                        # head tracker.  Convert the measured tracker origins
                        # back to ankle joint centers using the *same frozen
                        # mount* used by the endpoint equations, and reject
                        # only an excessive horizontal pelvis-to-ankle-midpoint
                        # offset.  The vertical component is deliberately
                        # omitted so a genuine squat remains free to lower the
                        # pelvis and bend both knees.
                        target_ankle_local = (
                            yaw_center.transpose(-1, -2)
                            @ (target_ankle_mid - world_joints[:, 0])[..., None]
                        ).squeeze(-1)
                        pelvis_ankle_error = target_ankle_local[:, (0, 2)] - neutral_ankle[[0, 2]]
                        pelvis_ankle_excess = torch.relu(
                            torch.linalg.vector_norm(pelvis_ankle_error, dim=-1)
                            - float(self.options.hard_binding_pelvis_ankle_bound_m)
                        )
                        foot_pair_valid = valid_t[:, 3] & valid_t[:, 4]
                        pelvis_ankle_center_loss = (
                            (
                                pelvis_ankle_excess.square()
                                + float(self.options.hard_binding_pelvis_ankle_center_weight)
                                * pelvis_ankle_error.square().sum(dim=-1)
                            ) * foot_pair_valid.to(torch.float32)
                        ).sum() / foot_pair_valid.to(torch.float32).sum().clamp_min(1.0)
                        if self.options.hard_binding_squat_center_weight > 0:
                            squat_gate = torch.as_tensor(
                                squat_center_gate_np[start:end],
                                dtype=world_joints.dtype, device=self.device,
                            ) * foot_pair_valid.to(world_joints.dtype)
                            hard_squat_center_loss = (
                                pelvis_ankle_error.square().sum(dim=-1) * squat_gate
                            ).sum() / squat_gate.sum().clamp_min(1.0)
                        # The transverse shoulder/hip axes are allowed a
                        # small anatomical roll, but they must not become a
                        # near-vertical ladder while the solver chases a hand
                        # or foot endpoint.  This is deliberately an
                        # inequality (20 deg), rather than forcing a rigid
                        # neutral torso during a genuine squat or reach.
                        shoulder_tilt = torch.abs(root_local_shoulder_axis[:, 1])
                        hip_tilt = torch.abs(root_local_hip_axis[:, 1])
                        hard_anatomical_loss = hard_anatomical_loss + .65 * torch.relu(
                            shoulder_tilt - float(np.sin(np.deg2rad(20.0)))
                        ).square().mean()
                        hard_anatomical_loss = hard_anatomical_loss + .35 * torch.relu(
                            hip_tilt - float(np.sin(np.deg2rad(20.0)))
                        ).square().mean()
                        # Endpoint equality can otherwise be satisfied by
                        # folding the neck/spine chain: at the bad 400--500
                        # frame branch the head tracker was correct, but the
                        # pelvis-to-head distance collapsed to ~22 cm.  Bone
                        # lengths are fixed by SMPL-X; a human squat may lean
                        # the torso, but it cannot collapse the torso into a
                        # single hinge.  Keep broad lower bounds (well below
                        # the neutral lengths) and reject an upper-neck bend
                        # that points back toward the pelvis.
                        pelvis_to_head = torch.linalg.vector_norm(
                            world_joints[:, 15] - world_joints[:, 0], dim=-1,
                        )
                        pelvis_to_neck = torch.linalg.vector_norm(
                            world_joints[:, 12] - world_joints[:, 0], dim=-1,
                        )
                        hard_anatomical_loss = hard_anatomical_loss + 1.20 * torch.relu(
                            float(self.options.hard_binding_pelvis_head_min_m) - pelvis_to_head
                        ).square().mean()
                        hard_anatomical_loss = hard_anatomical_loss + .65 * torch.relu(
                            float(self.options.hard_binding_pelvis_neck_min_m) - pelvis_to_neck
                        ).square().mean()

                        def chain_bend_angle(first, middle, last):
                            first_vector = world_joints[:, middle] - world_joints[:, first]
                            second_vector = world_joints[:, last] - world_joints[:, middle]
                            denominator = (
                                torch.linalg.vector_norm(first_vector, dim=-1)
                                * torch.linalg.vector_norm(second_vector, dim=-1)
                            ).clamp_min(1e-6)
                            cosine = (first_vector * second_vector).sum(dim=-1) / denominator
                            return torch.acos(cosine.clamp(-1.0 + 1e-5, 1.0 - 1e-5))

                        # The neck/head chain should continue away from the
                        # pelvis.  Keep a broad nod allowance, but do not let
                        # a squat be explained by folding the neck while the
                        # head tracker remains exact.
                        neck_bend = chain_bend_angle(0, 12, 15)
                        hard_anatomical_loss = hard_anatomical_loss + .85 * torch.relu(
                            neck_bend - float(np.deg2rad(
                                self.options.hard_binding_neck_bend_limit_deg
                            ))
                        ).square().mean()
                anchor = pose_delta.square().mean()
                smooth = (
                    (pose_delta[1:] - pose_delta[:-1]).square().mean()
                    if size > 1 else torch.zeros((), device=self.device)
                )
                root_anchor = (
                    (global_orient - base_global).square().mean()
                    if self.options.hard_binding_release_root else
                    torch.zeros((), device=self.device)
                )
                root_smooth = (
                    (global_orient[1:] - global_orient[:-1]).square().mean()
                    if self.options.hard_binding_release_root and size > 1 else
                    torch.zeros((), device=self.device)
                )
                root_yaw_alignment = torch.zeros((), device=self.device)
                if self.options.hard_binding_release_root and root_yaw_target is not None:
                    # Releasing global_orient improves endpoint position
                    # feasibility, but without a common-mode yaw cue it also
                    # re-introduces the 180-degree root branch that this stage is
                    # intended to diagnose.  Keep root yaw on the measured
                    # all-tracker common-turn trajectory while leaving pitch/roll
                    # and every local joint free for the equality solve.
                    root_matrix = _axis_angle_to_matrix(global_orient)
                    root_yaw = torch.atan2(
                        -root_matrix[:, 2, 0], root_matrix[:, 0, 0],
                    )
                    target_yaw = torch.as_tensor(
                        root_yaw_target[start:end], dtype=torch.float32, device=self.device,
                    )
                    root_error = torch.atan2(
                        torch.sin(root_yaw - target_yaw),
                        torch.cos(root_yaw - target_yaw),
                    )
                    root_yaw_alignment = _huber(root_error, .35).mean()
                # The strict endpoint equations are frame-local and can
                # otherwise choose a different IK branch on adjacent frames.
                # Select the nearest SO(3) branch without relaxing the five
                # tracker equalities.  A 42-degree barrier is deliberately
                # broad enough for a real fast motion but makes 90/180-degree
                # elbow/knee flips overwhelmingly expensive.  This branch
                # selector must be active even when the root is anchored.
                strict_pair_smooth = torch.zeros((), device=self.device)
                strict_pair_jump = torch.zeros((), device=self.device)
                strict_global_smooth = torch.zeros((), device=self.device)
                strict_global_jump = torch.zeros((), device=self.device)
                translation_velocity_loss = torch.zeros((), device=self.device)
                translation_acceleration_loss = torch.zeros((), device=self.device)
                if size > 1:
                    local_rotation = _axis_angle_to_matrix(pose)
                    strict_pair_angle = torch.sqrt(
                        _rotation_distance_squared(local_rotation[:-1], local_rotation[1:]) + 1e-8,
                    )
                    strict_pair_smooth = strict_pair_angle.square().mean()
                    strict_pair_jump = torch.relu(
                        strict_pair_angle - float(np.deg2rad(
                            self.options.hard_binding_joint_step_limit_deg
                        ))
                    ).square().mean()
                    # Local rotations can each stay below the threshold while
                    # their parent-chain composition flips a shoulder, hand,
                    # or head by 90--120 degrees in one frame.  The 400--500
                    # squat failure had exactly this signature.  Constrain
                    # global anatomical frames as well, while leaving the
                    # measured root turn itself available to the endpoint
                    # solve.  A wider 42-degree allowance admits real fast
                    # motion but rejects composed branch jumps.
                    global_pair_angle = torch.sqrt(
                        _rotation_distance_squared(rotations[:-1], rotations[1:]) + 1e-8,
                    )
                    global_chain_mask = torch.ones_like(global_pair_angle)
                    global_chain_mask[:, 0] = 0.0
                    strict_global_smooth = (
                        global_pair_angle.square() * global_chain_mask * adjacency[:, None]
                    ).sum() / (global_chain_mask * adjacency[:, None]).sum().clamp_min(1.0)
                    global_allowed = float(np.deg2rad(
                        max(42.0, self.options.hard_binding_joint_step_limit_deg)
                    ))
                    global_excess = torch.relu(global_pair_angle - global_allowed)
                    strict_global_jump = (
                        global_excess.square() * global_chain_mask * adjacency[:, None]
                    ).sum() / (global_chain_mask * adjacency[:, None]).sum().clamp_min(1.0)
                    # Match root motion to the measured five-tracker
                    # constellation. This is a velocity prior, not a fixed
                    # world position: walking and reaching remain free, while
                    # a branch that moves the entire body against stationary
                    # trackers is rejected.
                    observed_velocity = target_centroid_t[1:] - target_centroid_t[:-1]
                    predicted_velocity = effective_translation[1:] - effective_translation[:-1]
                    translation_velocity_loss = (
                        (predicted_velocity - observed_velocity).square().mean(dim=1)
                        * adjacency
                    ).sum() / adjacency.sum().clamp_min(1.0)
                    if size > 2:
                        observed_acceleration = (
                            target_centroid_t[2:] - 2.0 * target_centroid_t[1:-1]
                            + target_centroid_t[:-2]
                        )
                        predicted_acceleration = (
                            effective_translation[2:] - 2.0 * effective_translation[1:-1]
                            + effective_translation[:-2]
                        )
                        triplet_adjacency = adjacency[1:] * adjacency[:-1]
                        translation_acceleration_loss = (
                            (predicted_acceleration - observed_acceleration).square().mean(dim=1)
                            * triplet_adjacency
                        ).sum() / triplet_adjacency.sum().clamp_min(1.0)
                boundary_branch = torch.zeros((), device=self.device)
                boundary_translation = torch.zeros((), device=self.device)
                boundary_global = torch.zeros((), device=self.device)
                boundary_pose_anchor = torch.zeros((), device=self.device)
                boundary_translation_path = torch.zeros((), device=self.device)
                boundary_pose_path = torch.zeros((), device=self.device)
                boundary_velocity = torch.zeros((), device=self.device)
                if start > 0:
                    previous_pose = torch.as_tensor(
                        output_pose[start - 1], dtype=torch.float32, device=self.device,
                    ).reshape(1, 21, 3)
                    previous_rotation = _axis_angle_to_matrix(previous_pose)
                    boundary_angle = torch.sqrt(
                        _rotation_distance_squared(previous_rotation, _axis_angle_to_matrix(pose[:1])) + 1e-8,
                    )
                    boundary_branch = (
                        200.0 * boundary_angle.square().mean()
                        + 10000.0 * torch.relu(boundary_angle - float(np.deg2rad(42.0))).square().mean()
                    )
                    if previous_translation is not None:
                        # Use the previous velocity as a one-step prediction
                        # when available. This keeps ordinary walking motion
                        # continuous while still allowing a real contact
                        # transition to be explained by the endpoint loss.
                        translation_target = previous_translation
                        if previous_previous_translation is not None:
                            translation_target = previous_translation + (
                                previous_translation - previous_previous_translation
                            )
                        boundary_translation = (
                            (effective_translation[0] - translation_target).square().mean()
                        )
                        observed_boundary_velocity = target_centroid_t[0] - torch.as_tensor(
                            target_centroid[start - 1], dtype=torch.float32, device=self.device,
                        )
                        boundary_velocity = (
                            (effective_translation[0] - previous_translation - observed_boundary_velocity)
                            .square().mean()
                        )
                    if (
                        self.options.hard_binding_release_root
                        and previous_global is not None
                    ):
                        boundary_global = _rotation_distance_squared(
                            _axis_angle_to_matrix(global_orient[:1]),
                            _axis_angle_to_matrix(previous_global[None]),
                        ).mean()
                    # The old boundary term only became strong above 42°.
                    # Anchor every local joint weakly too, preventing a
                    # 20–40° shoulder/hip branch from replacing the smooth
                    # solution at a chunk boundary.
                    boundary_pose_anchor = _rotation_distance_squared(
                        previous_rotation, _axis_angle_to_matrix(pose[:1]),
                    ).mean()
                    # Also protect a short bridge after the boundary. The
                    # first frame alone is insufficient: the previous solver
                    # could keep frame ``start`` smooth and then jump on
                    # frame ``start+1`` while still satisfying every endpoint
                    # equation. The ordinary full-rate fit is a smooth,
                    # tracker-informed target for this bridge; it is not a
                    # pose prior for the rest of the hard solve.
                    bridge = min(24, size)
                    boundary_translation_path = (
                        (effective_translation[:bridge] - base_translation[:bridge]).square().mean()
                    )
                    free_body = torch.ones(21, dtype=torch.bool, device=self.device)
                    free_body[torch.as_tensor(strict_body_indices, dtype=torch.long, device=self.device)] = False
                    if bool(free_body.any()):
                        boundary_pose_path = _rotation_distance_squared(
                            _axis_angle_to_matrix(pose[:bridge, free_body]),
                            _axis_angle_to_matrix(base_pose[:bridge, free_body]),
                        ).mean()
                # A single Adam solve started with a 1e8 endpoint coefficient
                # is badly conditioned: the first update can choose a
                # mirrored two-bone branch before the anatomical barriers have
                # had a chance to shape the solution.  Conversely, solving
                # only with the historical 1e6 coefficient leaves a few cm of
                # endpoint error.  Use a continuation schedule so the same
                # trajectory first enters the anatomically valid basin and
                # only then tightens the endpoint equations to the requested
                # hard-binding value.  This does not alter the configured
                # final weight; it only controls the path taken by the
                # non-convex optimizer.
                progress = iteration / max(
                    int(self.options.hard_binding_iterations) - 1, 1,
                )
                continuation = .05 + .95 * (
                    .5 - .5 * np.cos(np.pi * progress)
                )
                scheduled_position_weight = (
                    float(self.options.hard_binding_position_weight)
                    * continuation
                )
                scheduled_rotation_weight = (
                    float(self.options.hard_binding_rotation_weight)
                    * continuation
                )
                equality_loss = (
                    scheduled_position_weight * position_loss
                    + scheduled_rotation_weight * rotation_loss
                )
                # Keep the anatomical branch barriers commensurate with the
                # endpoint equality.  The diagnostic is intentionally allowed
                # to raise the position coefficient (for example 1e8 when
                # checking millimetre-level binding); leaving the crossing
                # selector at its historical 1e6 then makes a crossed leg a
                # cheaper solution than a 1--2 cm endpoint correction.  Scale
                # only the hard branch selectors, never the production fit.
                endpoint_scale = max(
                    1.0, scheduled_position_weight / 1.0e6,
                )
                scaled_torso_alignment_weight = float(
                    self.options.hard_binding_torso_alignment_weight
                ) * np.sqrt(endpoint_scale)
                scaled_anatomical_weight = 30000.0 * endpoint_scale
                # Keep the limb-plane selector active from the first
                # continuation step.  If it follows ``endpoint_scale`` from
                # zero, Adam can enter a sideways knee/elbow basin before the
                # anatomical term becomes visible, and later gradients are
                # insufficient to cross that non-convex barrier.  The final
                # scale is still identical to the ordinary hard-anatomical
                # scale, so this changes only the path into the solution.
                scaled_limb_plane_weight = (
                    30000.0 * float(self.options.hard_binding_limb_plane_weight)
                    * max(1.0, endpoint_scale)
                )
                # Use the centred selector most strongly while the endpoint
                # continuation is entering a branch, then let the measured
                # tracker equations recover precision.  A constant centre
                # pull improves the branch but can keep paying for a small
                # neutral-pose mismatch at the final equality stage.
                scaled_limb_center_weight = (
                    30000.0 * float(self.options.hard_binding_limb_center_weight)
                    * max(1.0, endpoint_scale)
                    * (2.0 - 1.5 * progress)
                )
                scaled_analytic_limb_weight = (
                    30000.0 * float(self.options.hard_binding_analytic_limb_weight)
                    * max(1.0, endpoint_scale)
                )
                scaled_crossing_weight = 1000000.0 * endpoint_scale
                scaled_strict_smooth_weight = 800.0 * endpoint_scale
                scaled_strict_jump_weight = 100000.0 * endpoint_scale
                regularization_loss = (
                    .002 * anchor + .015 * smooth
                    + .010 * root_anchor + .025 * root_smooth
                    + (
                        100000.0 * root_yaw_alignment
                        if strict_rotation and root_yaw_target is not None else
                        float(self.options.root_yaw_target_weight) * root_yaw_alignment
                    )
                    + scaled_strict_smooth_weight * strict_pair_smooth
                    + scaled_strict_jump_weight * strict_pair_jump
                    + .35 * scaled_strict_smooth_weight * strict_global_smooth
                    + 1.50 * scaled_strict_jump_weight * strict_global_jump
                    # This is a branch selector, not a soft tracker loss.  It
                    # must be strong enough to reject the 100-degree spine
                    # solution that otherwise satisfies endpoint equalities by
                    # twisting spine1 instead of turning the pelvis.
                    + 50000.0 * torso_twist_loss
                    + scaled_torso_alignment_weight * torso_alignment_loss
                    # Hard endpoint equalities must not re-introduce a
                    # crossed-leg/folded-elbow branch after the ordinary
                    # biomechanical repair.  This term is active only in the
                    # opt-in hard-binding experiment.
                    + scaled_anatomical_weight * hard_anatomical_loss
                    + scaled_limb_plane_weight * hard_limb_plane_loss
                    + scaled_limb_center_weight * hard_limb_center_loss
                    + scaled_analytic_limb_weight * hard_analytic_limb_loss
                    + scaled_anatomical_weight
                    * float(self.options.hard_binding_torso_direction_weight)
                    * hard_torso_direction_loss
                    + scaled_anatomical_weight
                    * float(self.options.hard_binding_proximal_anchor_weight)
                    * hard_proximal_anchor_loss
                    + scaled_anatomical_weight
                    * float(self.options.hard_binding_torso_local_twist_weight)
                    * hard_torso_local_twist_loss
                    + scaled_anatomical_weight * 2.0 * pelvis_ankle_center_loss
                    + scaled_anatomical_weight
                    * float(self.options.hard_binding_squat_center_weight)
                    * hard_squat_center_loss
                    + scaled_anatomical_weight
                    * float(self.options.hard_binding_squat_torso_direction_weight)
                    * hard_squat_torso_direction_loss
                    # Side-order violations are the highest-risk failure:
                    # unlike a small joint-angle mismatch, a crossed knee or
                    # wrist is an unmistakable wrong IK branch. Give only
                    # these barriers the stronger selector weight; all other
                    # anatomical terms remain weak enough not to pull the
                    # endpoints away from their rigid tracker targets.
                    + scaled_crossing_weight * hard_crossing_loss
                    + 20000.0 * boundary_translation
                    + 15000.0 * boundary_global
                    + 10000.0 * boundary_pose_anchor
                    + 100000.0 * boundary_translation_path
                    + 20000.0 * boundary_pose_path
                    + 1000000.0 * translation_velocity_loss
                    + 2000000.0 * translation_acceleration_loss
                    + 1000000.0 * boundary_velocity
                    + boundary_branch
                )
                # Pure mode is an intentionally pathological diagnostic: it
                # leaves only the endpoint equations so the user can see
                # which SO(3) branches are inherently ambiguous.  The normal
                # strict experiment adds only branch-selection terms, never a
                # VPoser style pose attraction.
                loss = equality_loss + (
                    torch.zeros((), device=self.device)
                    if self.options.hard_binding_pure_equality else regularization_loss
                )
                loss.backward()
                # The ordinary diagnostic uses a modest global clip.  With a
                # 1e6 positional equality weight that clip would reduce a
                # centimetre translation step to micrometres, effectively
                # freezing the endpoint before it can meet the raw tracker.
                # Equality mode therefore permits a much larger norm; Adam
                # still normalizes each parameter update.
                torch.nn.utils.clip_grad_norm_(
                    optimizer_parameters, 1000.0 if strict_rotation else 10.0,
                )
                optimizer.step()
                with torch.no_grad():
                    if not project_translation:
                        # Keep the root on the already smooth ordinary-fit
                        # trajectory. Without this bound, the sparse endpoint
                        # equations can move the whole body 10--20 cm in a
                        # few frames to avoid a local knee/shoulder branch.
                        translation_offset = (
                            translation - base_translation
                        ).clamp(
                            -float(self.options.hard_binding_translation_bound_m),
                            float(self.options.hard_binding_translation_bound_m),
                        )
                        translation.copy_(base_translation + translation_offset)
                    pose_delta.clamp_(-1.5, 1.5)
                    if not self.options.hard_binding_pure_equality:
                        # A differentiable penalty selects the nearest branch,
                        # but a non-convex endpoint solve can still remain on a
                        # high-twist local minimum.  Project only the axial
                        # component of spine1/2/3 after each step.  The x/z
                        # components remain free, so bending and crouching are
                        # not frozen; this is specifically an anti-torsion
                        # bound for the torso, not a VPoser pose clamp.
                        spine_indices = (2, 5, 8)
                        limit = float(np.deg2rad(35.0))
                        lower = -limit - base_pose[:, spine_indices, 1]
                        upper = limit - base_pose[:, spine_indices, 1]
                        pose_delta[:, spine_indices, 1].clamp_(lower, upper)
            if int(self.options.hard_binding_refinement_iterations) > 0:
                # Adam's continuation path selects the anatomical branch, but
                # a weighted penalty generally leaves a few millimetres of
                # endpoint residual.  Refine from that already-valid branch
                # with small Adam equality steps.  Unlike a line-search
                # method, this does not cross the non-smooth branch barriers
                # in one trial step.
                refine_pose_anchor = pose_delta.detach().clone()
                refine_root_anchor = (
                    global_orient.detach().clone()
                    if self.options.hard_binding_release_root
                    and not self.options.hard_binding_root_yaw_only else None
                )
                refine_translation_anchor = (
                    translation.detach().clone()
                    if optimize_translation and not project_translation else None
                )
                refine_parameters = list(optimizer_parameters)
                refine_optimizer = torch.optim.Adam(
                    refine_parameters,
                    lr=min(.002, .25 * float(self.options.hard_binding_learning_rate)),
                )

                def refine_closure():
                    refine_optimizer.zero_grad(set_to_none=True)
                    refine_pose = enforce_strict_rotations(
                        base_pose + pose_delta * release_mask,
                    )
                    refine_model = self.body(
                        betas=self.betas[None].expand(size, -1),
                        global_orient=global_orient, transl=zero3,
                        body_pose=refine_pose.reshape(size, -1),
                        left_hand_pose=torch.zeros((size, 45), dtype=torch.float32, device=self.device),
                        right_hand_pose=torch.zeros((size, 45), dtype=torch.float32, device=self.device),
                        jaw_pose=zero3, leye_pose=zero3, reye_pose=zero3,
                        expression=zero10, return_verts=False,
                    )
                    refine_rotations = _global_body_rotations(
                        global_orient, refine_pose, self.parents,
                    )
                    refine_world = (
                        self.profile.scale * refine_model.joints
                        + translation[:, None]
                    )
                    refine_predicted = self.profile.scale * refine_model.joints[:, position_ids]
                    refine_predicted = refine_predicted + (
                        refine_rotations[:, rotation_ids]
                        @ self.mount_position[None, :, :, None]
                    ).squeeze(-1) + translation[:, None]
                    refine_error = torch.where(
                        valid_t[:, :, None], refine_predicted - target_pos,
                        torch.zeros_like(target_pos),
                    )
                    refine_position_weights = valid_float.clone()
                    refine_position_weights[:, 0] *= float(self.options.hard_binding_head_position_multiplier)
                    refine_position_weights[:, 1:3] *= float(self.options.hard_binding_hand_position_multiplier)
                    refine_position_weights[:, 3:5] *= float(self.options.hard_binding_foot_position_multiplier)
                    refine_position_weights[:, 3:5] *= dynamic_foot_multiplier
                    refine_position_loss = (
                        refine_error.square().sum(dim=2) * refine_position_weights
                    ).sum() / (refine_position_weights.sum().clamp_min(1.0) * 3.0)
                    refine_global_step_loss = torch.zeros((), device=self.device)
                    if (
                        float(self.options.hard_binding_refinement_global_step_weight) > 0
                        and size > 1
                    ):
                        refine_global_angle = torch.sqrt(
                            _rotation_distance_squared(
                                refine_rotations[:-1], refine_rotations[1:],
                            ) + 1e-8,
                        )
                        refine_global_mask = torch.ones_like(refine_global_angle)
                        refine_global_mask[:, 0] = 0.0
                        refine_global_excess = torch.relu(
                            refine_global_angle - float(np.deg2rad(42.0))
                        )
                        refine_global_step_loss = (
                            refine_global_excess.square() * refine_global_mask
                        ).sum() / refine_global_mask.sum().clamp_min(1.0)
                    refine_neck_loss = torch.zeros((), device=self.device)
                    if float(self.options.hard_binding_refinement_neck_weight) > 0:
                        neck_first = refine_world[:, 0] - refine_world[:, 12]
                        neck_second = refine_world[:, 15] - refine_world[:, 12]
                        neck_denominator = (
                            torch.linalg.vector_norm(neck_first, dim=-1)
                            * torch.linalg.vector_norm(neck_second, dim=-1)
                        ).clamp_min(1e-6)
                        neck_cosine = (
                            neck_first * neck_second
                        ).sum(dim=-1) / neck_denominator
                        neck_angle = torch.acos(
                            neck_cosine.clamp(-1.0 + 1e-5, 1.0 - 1e-5)
                        )
                        refine_neck_loss = torch.relu(
                            neck_angle - float(np.deg2rad(34.0))
                        ).square().mean()
                    _, refine_range_loss, refine_crossing_loss = self._joint_constraint_losses(
                        refine_pose, refine_world, refine_rotations,
                    )
                    trust = float(self.options.hard_binding_refinement_trust_weight) * (
                        pose_delta - refine_pose_anchor
                    ).square().mean()
                    if refine_root_anchor is not None:
                        trust = trust + float(self.options.hard_binding_refinement_trust_weight) * (
                            global_orient - refine_root_anchor
                        ).square().mean()
                    if refine_translation_anchor is not None:
                        trust = trust + float(self.options.hard_binding_refinement_trust_weight) * (
                            translation - refine_translation_anchor
                        ).square().mean()
                    refine_loss = (
                        float(self.options.hard_binding_position_weight) * refine_position_loss
                        + 3.0e6 * refine_range_loss
                        + 1.0e8 * refine_crossing_loss
                        + float(self.options.hard_binding_refinement_global_step_weight)
                        * refine_global_step_loss
                        + float(self.options.hard_binding_refinement_neck_weight)
                        * refine_neck_loss
                        + trust
                    )
                    refine_loss.backward()
                    return refine_loss

                for _ in range(int(self.options.hard_binding_refinement_iterations)):
                    refine_optimizer.step(refine_closure)
                    with torch.no_grad():
                        pose_delta.clamp_(-1.5, 1.5)
                        if refine_translation_anchor is not None:
                            translation_offset = (
                                translation - base_translation
                            ).clamp(
                                -float(self.options.hard_binding_translation_bound_m),
                                float(self.options.hard_binding_translation_bound_m),
                            )
                            translation.copy_(base_translation + translation_offset)
                with torch.no_grad():
                    pose_delta.clamp_(-1.5, 1.5)
            with torch.no_grad():
                pose = enforce_strict_rotations(base_pose + pose_delta * release_mask)
                model = self.body(
                    betas=self.betas[None].expand(size, -1),
                    global_orient=global_orient, transl=zero3,
                    body_pose=pose.reshape(size, -1),
                    left_hand_pose=torch.zeros((size, 45), dtype=torch.float32, device=self.device),
                    right_hand_pose=torch.zeros((size, 45), dtype=torch.float32, device=self.device),
                    jaw_pose=zero3, leye_pose=zero3, reye_pose=zero3,
                    expression=zero10, return_verts=False,
                )
                rotations = _global_body_rotations(global_orient, pose, self.parents)
                predicted_no_translation = self.profile.scale * model.joints[:, position_ids]
                predicted_no_translation = predicted_no_translation + (
                    rotations[:, rotation_ids] @ self.mount_position[None, :, :, None]
                ).squeeze(-1)
                if project_translation:
                    safe_target = torch.nan_to_num(
                        target_pos, nan=0.0, posinf=0.0, neginf=0.0,
                    )
                    centroid_weight = valid_t[:, :, None].to(torch.float32)
                    translation = (
                        (safe_target - predicted_no_translation) * centroid_weight
                    ).sum(dim=1) / centroid_weight.sum(dim=1).clamp_min(1.0)
                effective_translation = translation
                if (
                    self.options.hard_binding_project_pelvis
                    and optimize_translation
                ):
                    target_pelvis = target_ankle_mid - (
                        yaw_center @ (self.profile.scale * neutral_ankle)[None, :, None]
                    ).squeeze(-1)
                    effective_translation = translation.clone()
                    effective_translation[:, (0, 2)] = (
                        target_pelvis[:, (0, 2)]
                        - self.profile.scale * model.joints[:, 0, (0, 2)]
                    )
                if (
                    self.options.hard_binding_project_squat_center
                    and optimize_translation
                ):
                    # Mirror the per-iteration squat projection for the
                    # final exported state.  Omitting this final projection
                    # would optimize against a centered pelvis but write the
                    # pre-projection translation, making the saved trajectory
                    # appear to ignore the constraint and reintroducing the
                    # apparent face-plant in the viewer.
                    target_pelvis = target_ankle_mid - (
                        yaw_center @ (self.profile.scale * neutral_ankle)[None, :, None]
                    ).squeeze(-1)
                    squat_gate_projection = torch.as_tensor(
                        squat_center_gate_np[start:end],
                        dtype=effective_translation.dtype, device=self.device,
                    ).clamp(0.0, 1.0)
                    projected_translation = translation.clone()
                    projected_translation[:, (0, 2)] = (
                        target_pelvis[:, (0, 2)]
                        - self.profile.scale * model.joints[:, 0, (0, 2)]
                    )
                    bound = float(self.options.hard_binding_translation_bound_m)
                    offset = projected_translation - translation
                    if bound > 0:
                        offset = offset.clamp(-bound, bound)
                    projected_translation = translation + offset
                    effective_translation = (
                        translation * (1.0 - squat_gate_projection[:, None])
                        + projected_translation * squat_gate_projection[:, None]
                    )
                predicted = predicted_no_translation + effective_translation[:, None]
                output_pose[start:end] = pose.cpu().numpy()
                output_transl[start:end] = effective_translation.cpu().numpy()
                if self.options.hard_binding_release_root:
                    output_global[start:end] = global_orient.cpu().numpy()
                residual = torch.linalg.vector_norm(
                    predicted - target_pos, dim=2,
                ).cpu().numpy()
                residual[~valid[start:end]] = np.nan
        result = SmplTrajectory(
            timestamps=np.asarray(trajectory.timestamps).copy(),
            betas=np.asarray(trajectory.betas).copy(),
            global_orient=output_global,
            transl=output_transl,
            body_pose=output_pose.reshape(count, 63),
            left_hand_pose=np.asarray(trajectory.left_hand_pose).copy(),
            right_hand_pose=np.asarray(trajectory.right_hand_pose).copy(),
            attachment_position=np.asarray(trajectory.attachment_position).copy()
            if trajectory.attachment_position is not None else None,
            attachment_rotation=np.asarray(trajectory.attachment_rotation).copy()
            if trajectory.attachment_rotation is not None else None,
            scale=float(trajectory.scale),
        )
        # Recompute one final residual array from the public prediction path so
        # the diagnostics and viewer use exactly the same convention.
        residual = np.linalg.norm(self.predict_trackers(result) - target_position, axis=2).astype(np.float32)
        residual[~valid] = np.nan
        return result, residual

    def fit(
        self, frames: Sequence[TrackerFrame], *, session_id: str,
        protocol: ProtocolAlignment | None = None,
    ) -> ProfileFitResult:
        import torch

        if not frames:
            raise ValueError("cannot fit an empty recording")
        source_timestamps = np.asarray([frame.timestamp for frame in frames], dtype=np.float64)
        # The maximum setting is genuinely frame-local: it optimizes every
        # measured source frame instead of fitting a sparse subset and then
        # interpolating the pose between them.
        effective_stride = 1 if self.options.tracker_hardness >= .999 else max(1, self.options.sample_stride)
        selection = np.arange(0, len(frames), effective_stride, dtype=np.int64)
        if protocol is not None:
            dynamic_indices = []
            for segment in protocol.segments:
                if segment.key not in {"head", "elbows", "wrists", "squat", "left_knee", "right_knee", "ankles"}:
                    continue
                dynamic_indices.extend(np.flatnonzero(
                    (source_timestamps >= segment.active_start_s - .35)
                    & (source_timestamps <= segment.active_end_s + .35)
                ).tolist())
            selection = np.unique(np.r_[selection, np.asarray(dynamic_indices, dtype=np.int64)])
        if selection[-1] != len(frames) - 1:
            selection = np.append(selection, len(frames) - 1)
        sampled = tuple(frames[index] for index in selection)
        target_position_np = np.stack([[frame[name].position for name in TRACKER_NAMES] for frame in sampled]).astype(np.float32)
        target_rotation_np = np.stack([[frame[name].rotation for name in TRACKER_NAMES] for frame in sampled]).astype(np.float32)
        valid_np, excluded = _stream_validity(sampled, session_id, self.profile)
        first_pose = self._first_pose(target_position_np, valid_np)
        timestamps = np.asarray([frame.timestamp for frame in sampled], dtype=np.float64)
        protocol_latent_np = self._protocol_latents(protocol, timestamps, first_pose)
        protocol_correction_np = self._protocol_corrections(protocol, timestamps, first_pose)
        protocol_latent_np, protocol_correction_np = self._observed_protocol_priors(
            protocol, timestamps, target_position_np, protocol_latent_np, protocol_correction_np,
        )
        locomotion_prior_factor = np.zeros(len(sampled), dtype=np.float32)
        locomotion_diagnostics = {"active_fraction": 0.0, "maximum_swing_amount": 0.0}
        root_yaw_offset_np = np.zeros(len(sampled), dtype=np.float32)
        body_yaw_offset_np = np.zeros(len(sampled), dtype=np.float32)
        if protocol is None:
            protocol_latent_np, locomotion_prior_factor, locomotion_diagnostics = (
                self._personal_locomotion_priors(
                    timestamps, target_position_np, valid_np, protocol_latent_np,
                )
            )
            root_yaw_offset_np = self._personal_root_yaw_offsets(
                target_rotation_np, valid_np, timestamps,
            )
            if self.options.biomechanical_branch_repair:
                # This is a measured body-base cue, distinct from the
                # optional personal locomotion prior above. It is only used
                # to select a stable root branch; tracker endpoint losses
                # still determine the actual pose.
                body_yaw_offset_np = _body_yaw_offsets(
                    target_rotation_np, valid_np, timestamps,
                )
        else:
            # Seed only the labelled stage-16 root turn from observed head and
            # ankle rotations. Stage 02 remains head-only and cannot leak into
            # the pelvis/root orientation.
            turn_segment = next(
                (item for item in protocol.segments if item.key == "turns"), None,
            )
            if turn_segment is not None:
                turn_reference = self.profile.locomotion_reference.get("turn", {})
                turn_weights = np.asarray(
                    turn_reference.get("yaw_tracker_weights", (.40, .30, .30)),
                    dtype=np.float64,
                )
                if (
                    turn_weights.shape != (3,)
                    or not np.isfinite(turn_weights).all()
                    or turn_weights.sum() <= 0
                ):
                    turn_weights = np.asarray((.40, .30, .30), dtype=np.float64)
                turn_weights /= turn_weights.sum()
                observed_turn = _coherent_yaw_offsets(
                    target_rotation_np, valid_np, timestamps, turn_weights, .16,
                )
                turn_mask = (
                    (timestamps >= turn_segment.start_s)
                    & (timestamps <= turn_segment.end_s)
                )
                origin_mask = (
                    (timestamps >= turn_segment.start_s - .45)
                    & (timestamps <= turn_segment.start_s + .20)
                )
                turn_origin = float(np.median(observed_turn[origin_mask]))
                root_yaw_offset_np[turn_mask] = (
                    observed_turn[turn_mask] - turn_origin
                )
        # The deployable reconstruction must remain on VPoser's learned human
        # pose manifold.  The calibration-reference corrections are useful as
        # authoring diagnostics, but adding arbitrary joint-local axis angles
        # after VPoser can produce inverted necks and broken-looking limbs.
        if self.options.natural_pose_only:
            protocol_correction_np.fill(0.0)
        protocol_keys = []
        observation_factor = np.ones(len(sampled), dtype=np.float32)
        smooth_factor = np.ones(len(sampled), dtype=np.float32)
        relative_rotation_weight = np.zeros((len(sampled), len(TRACKER_NAMES)), dtype=np.float32)
        protocol_prior_factor = locomotion_prior_factor.copy()
        head_forward_weight = np.ones(len(sampled), dtype=np.float32)
        head_body_lock_weight = np.zeros(len(sampled), dtype=np.float32)
        shoulder_lock_weight = np.zeros(len(sampled), dtype=np.float32)
        elbow_lock_weight = np.zeros(len(sampled), dtype=np.float32)
        distal_joint_mask = np.zeros((len(sampled), 4), dtype=np.float32)
        head_joint_mask = np.zeros((len(sampled), 2), dtype=np.float32)
        leg_joint_mask = np.zeros((len(sampled), 6), dtype=np.float32)
        torso_joint_mask = np.zeros((len(sampled), 3), dtype=np.float32)
        torso_bend_factor = np.zeros(len(sampled), dtype=np.float32)
        # Above 75% the control deliberately transitions from a pure VPoser
        # prior into bounded anatomical IK on the chains that actually carry
        # the five rigid mounts.  This gives the 100% endpoint a way to obey
        # measurements that are outside the exact VPoser latent manifold,
        # while lower settings retain the natural learned-pose representation.
        hard_articulation = float(np.clip(
            (self.options.tracker_hardness - .75) / .25, 0.0, 1.0,
        ))
        if hard_articulation > 0:
            distal_joint_mask[:] = hard_articulation
            head_joint_mask[:] = hard_articulation
            leg_joint_mask[:] = hard_articulation
            if self.options.biomechanical_branch_repair and protocol is None:
                torso_joint_mask[:] = hard_articulation
                # A head tracker alone cannot identify a unique full-spine
                # pose.  Use only relative head-vs-foot motion as a gate, so
                # ordinary walking does not get interpreted as a bow or squat.
                usable = (
                    valid_np[:, 0] & valid_np[:, 3] & valid_np[:, 4]
                    & np.isfinite(target_position_np[:, 0]).all(axis=1)
                    & np.isfinite(target_position_np[:, 3:5]).all(axis=(1, 2))
                )
                if usable.any():
                    foot_mid = .5 * (target_position_np[:, 3] + target_position_np[:, 4])
                    relative = target_position_np[:, 0] - foot_mid
                    # The first seconds of an ordinary recording are not
                    # guaranteed to be neutral (the operator may already be
                    # reaching into a cabinet).  Estimate the upright
                    # head-vs-feet reference from the high end of the whole
                    # sequence instead of treating the recording start as a
                    # calibration pose.
                    high = float(np.percentile(relative[usable, 1], 90.0))
                    baseline_mask = usable & (relative[:, 1] >= high - .035)
                    if baseline_mask.sum() < 3:
                        baseline_mask = usable
                    baseline = np.median(relative[baseline_mask], axis=0)
                    delta = relative - baseline[None]
                    vertical_drop = np.maximum(-delta[:, 1] - .025, 0.0) / .16
                    horizontal_shift = np.linalg.norm(delta[:, (0, 2)], axis=1)
                    horizontal_shift = np.maximum(horizontal_shift - .08, 0.0) / .22
                    torso_bend_factor = np.clip(
                        np.maximum(vertical_drop, horizontal_shift), 0.0, 1.0,
                    ).astype(np.float32)
                    torso_bend_factor[~usable] = 0.0
                    torso_bend_factor = _smooth_seconds(
                        torso_bend_factor[:, None], timestamps, .12,
                    )[:, 0].astype(np.float32)
        factor_by_key = {
            "head": (1.8, .35), "elbows": (3.0, .18), "wrists": (2.2, .30),
            "squat": (3.5, .16), "left_knee": (5.0, .12), "right_knee": (5.0, .12),
            "ankles": (2.5, .22),
        }
        if protocol is not None:
            for index, timestamp in enumerate(timestamps):
                segment = protocol.segment_at(float(timestamp)); protocol_keys.append(segment.key)
                protocol_prior_factor[index] = .06
                if segment.active_start_s - .35 <= timestamp <= segment.active_end_s + .35 and segment.key in factor_by_key:
                    observation_factor[index], smooth_factor[index] = factor_by_key[segment.key]
                    protocol_prior_factor[index] = (
                        {
                            "head": 2.0, "elbows": 50.0, "wrists": 1.5, "squat": 1.5,
                            "left_knee": 1.5, "right_knee": 1.5, "ankles": .5,
                        } if self.options.natural_pose_only else {
                            "head": 8.0, "elbows": 12.0, "wrists": 12.0, "squat": 2.0,
                            "left_knee": 3.0, "right_knee": 3.0, "ankles": 1.5,
                        }
                    )[segment.key]
                    if segment.key == "head":
                        head_forward_weight[index] = 0.0
                        head_body_lock_weight[index] = 1.0
                        relative_rotation_weight[index, 0] = 8.0
                    elif segment.key == "elbows":
                        head_body_lock_weight[index] = 1.0
                        shoulder_lock_weight[index] = 1.0
                        relative_rotation_weight[index, 1:3] = 6.0
                    elif segment.key == "wrists":
                        head_body_lock_weight[index] = 1.0
                        shoulder_lock_weight[index] = 1.0
                        elbow_lock_weight[index] = 1.0
                        if not self.options.natural_pose_only:
                            distal_joint_mask[index, 2:] = 1.0
                        relative_rotation_weight[index, 1:3] = 12.0
                    elif segment.key == "squat":
                        relative_rotation_weight[index, 3:5] = 1.5
                    elif segment.key == "left_knee":
                        if not self.options.natural_pose_only:
                            leg_joint_mask[index, (0, 2, 4)] = 1.0
                        relative_rotation_weight[index, 3] = 4.0
                    elif segment.key == "right_knee":
                        if not self.options.natural_pose_only:
                            leg_joint_mask[index, (1, 3, 5)] = 1.0
                        relative_rotation_weight[index, 4] = 4.0
                    elif segment.key == "ankles":
                        if not self.options.natural_pose_only:
                            leg_joint_mask[index, 4:6] = 1.0
                        relative_rotation_weight[index, 3:5] = 4.0
        else:
            protocol_keys = [first_pose] * len(sampled)
        # Use a fixed physical start interval. Previously this was always 30
        # sampled frames: about 2 s at stride 1 but 6 s at stride 3, allowing
        # motion after recording start to corrupt the initial body heading.
        yaw_estimation_frames = _initial_yaw_window_count(timestamps)
        initial = self._initial_state(
            target_position_np, target_rotation_np, valid_np, session_id,
            protocol_latent_np, protocol_correction_np,
            root_yaw_offset=root_yaw_offset_np,
            yaw_estimation_frames=yaw_estimation_frames,
        )
        root_yaw_target_np = None
        if self.options.biomechanical_branch_repair and protocol is None:
            root_yaw_target_np = (
                np.deg2rad(float(self._initial_yaw_diagnostics["yaw_deg"]))
                + body_yaw_offset_np
            ).astype(np.float32)
        baseline_index = int(np.argmin(np.abs(timestamps - min(3.0, timestamps[-1]))))
        with torch.no_grad():
            baseline_pose = self.vposer.decode(initial[0][baseline_index:baseline_index + 1])["pose_body"].reshape(1, 21, 3)
            baseline_pose = baseline_pose + torch.as_tensor(
                protocol_correction_np[baseline_index:baseline_index + 1], device=self.device,
            ).reshape(1, 21, 3)
            joint_rotation_baseline = _global_body_rotations(
                initial[1][baseline_index:baseline_index + 1], baseline_pose, self.parents,
            )[0, self.rotation_joint_ids].cpu().numpy()
        target_rotation_baseline = target_rotation_np[baseline_index]
        ground_y, contact_height, contact_np, root_anchor_xz = self._ground_context(
            target_position_np, target_rotation_np, valid_np, timestamps=timestamps,
            initial=initial, initial_latent=protocol_latent_np,
            protocol_keys=protocol_keys if protocol is not None else None,
            pose_correction=protocol_correction_np,
        )
        target_position = torch.as_tensor(target_position_np, device=self.device)
        target_rotation = torch.as_tensor(target_rotation_np, device=self.device)
        valid = torch.as_tensor(valid_np, dtype=torch.bool, device=self.device)
        all_latent, all_global, all_translation, all_pose = [], [], [], []
        previous = None
        for start in range(0, len(sampled), self.options.window_size):
            end = min(len(sampled), start + self.options.window_size)
            count = end - start
            if previous is None:
                state = tuple(value[start:end].clone() for value in initial)
            else:
                latent_state = initial[0][start:end].clone()
                global_state = initial[1][start:end].clone()
                bridge = min(3, count)
                for offset in range(bridge):
                    amount = offset / max(bridge - 1, 1)
                    latent_state[offset] = (1.0 - amount) * previous[0][-1] + amount * latent_state[offset]
                    global_state[offset] = (1.0 - amount) * previous[1][-1] + amount * global_state[offset]
                # Follow gross measured translation while retaining exact
                # continuity with the previous optimized window.
                translation_state = initial[2][start:end].clone()
                translation_state += previous[2][-1] - translation_state[0]
                state = (latent_state, global_state, translation_state)
            window_positional = (
                target_position[start:end], target_rotation[start:end], valid[start:end], timestamps[start:end], state,
                contact_np[start:end], ground_y, contact_height,
                root_anchor_xz if np.ndim(root_anchor_xz) == 1 else root_anchor_xz[start:end],
            )
            window_keyword = dict(
                latent_prior=protocol_latent_np[start:end], global_prior=initial[1][start:end].detach(),
                observation_factor=observation_factor[start:end], smooth_factor=smooth_factor[start:end],
                target_rotation_baseline=target_rotation_baseline,
                joint_rotation_baseline=joint_rotation_baseline,
                relative_rotation_weight=relative_rotation_weight[start:end],
                protocol_prior_factor=protocol_prior_factor[start:end],
                root_yaw_target=(
                    None if root_yaw_target_np is None else root_yaw_target_np[start:end]
                ),
                head_forward_weight=head_forward_weight[start:end],
                head_body_lock_weight=head_body_lock_weight[start:end],
                shoulder_lock_weight=shoulder_lock_weight[start:end],
                elbow_lock_weight=elbow_lock_weight[start:end],
                distal_joint_mask=distal_joint_mask[start:end],
                head_joint_mask=head_joint_mask[start:end],
                leg_joint_mask=leg_joint_mask[start:end],
                torso_joint_mask=torso_joint_mask[start:end],
                torso_bend_factor=torso_bend_factor[start:end],
                pose_correction=protocol_correction_np[start:end],
                previous_pose=None if previous is None else previous[3][-1],
                previous_timestamp=None if start == 0 else float(timestamps[start - 1]),
            )
            window_result = self._fit_window(
                *window_positional, **window_keyword, boundary_factor=1.0,
            )
            output_result = window_result
            if start == 0:
                # Solve the first window a second time without pinning frame 0
                # to the coarse VPoser seed.  Only its initial prefix is used
                # for output; the stable bounded solution remains the warm
                # start for all later non-convex windows.  This removes the
                # recording-start transient without changing the solution
                # basin of the rest of the sequence.
                free_start = self._fit_window(
                    *window_positional, **window_keyword, boundary_factor=0.0,
                )
                blend = np.clip(
                    (timestamps[start:end] - timestamps[start])
                    / max(float(self.options.initial_transition_s), 1e-6),
                    0.0, 1.0,
                ).astype(np.float32)
                output_values = []
                for free_value, stable_value in zip(free_start, window_result, strict=True):
                    amount = torch.as_tensor(
                        blend.reshape((count,) + (1,) * (stable_value.ndim - 1)),
                        dtype=stable_value.dtype, device=self.device,
                    )
                    output_values.append((1.0 - amount) * free_value + amount * stable_value)
                output_result = tuple(output_values)
            previous = window_result
            all_latent.append(output_result[0].cpu().numpy())
            all_global.append(output_result[1].cpu().numpy())
            all_translation.append(output_result[2].cpu().numpy())
            all_pose.append(output_result[3].cpu().numpy())
            if self.options.verbose:
                print(f"{session_id}: fitted {start}:{end}/{len(sampled)}")
            if self.options.progress_callback is not None:
                self.options.progress_callback(session_id, end, len(sampled))
        latent_np = np.concatenate(all_latent).astype(np.float32)
        global_np = np.concatenate(all_global).astype(np.float32)
        translation_np = np.concatenate(all_translation).astype(np.float32)
        pose_np = np.concatenate(all_pose).reshape(len(sampled), 21, 3).astype(np.float32)
        if self.options.post_articulation_refinement and not self.options.natural_pose_only:
            pose_np = self._refine_protocol_articulation(
                pose_np, global_np, translation_np, timestamps,
                target_position_np, target_rotation_np, valid_np, protocol,
            )
        pose_np = pose_np.reshape(len(sampled), 63)
        smoothing_window = 1 if self.options.tracker_hardness >= .999 else self.options.root_smoothing_window
        global_np = _smooth_continuous(global_np, timestamps, smoothing_window)
        translation_np = _smooth_continuous(translation_np, timestamps, smoothing_window)
        optimized_trajectory = SmplTrajectory(
            timestamps=timestamps, betas=self.profile.betas.copy(), global_orient=global_np, transl=translation_np,
            body_pose=pose_np, left_hand_pose=np.zeros((len(sampled), 45), np.float32),
            right_hand_pose=np.zeros((len(sampled), 45), np.float32),
            attachment_position=self.mount_position.cpu().numpy().copy(), attachment_rotation=self.mount_rotation.cpu().numpy().copy(),
            scale=self.profile.scale,
        )
        if self.options.output_full_rate and len(sampled) != len(frames):
            output_timestamps = np.asarray([frame.timestamp for frame in frames], dtype=np.float64)
            if self.options.natural_pose_only and hard_articulation == 0:
                # Interpolating decoded joint axis angles can leave VPoser's
                # learned manifold. Interpolate latent values and decode them
                # again so every exported source-rate frame is a VPoser pose.
                latent_full = _resample_linear(latent_np, timestamps, output_timestamps)
                decoded = []
                with torch.no_grad():
                    for start in range(0, len(latent_full), 256):
                        value = torch.as_tensor(
                            latent_full[start:start + 256], dtype=torch.float32, device=self.device,
                        )
                        decoded.append(self.vposer.decode(value)["pose_body"].cpu().numpy())
                body_pose_full = np.concatenate(decoded).reshape(len(frames), 63).astype(np.float32)
            else:
                body_pose_full = _resample_rotvec(
                    pose_np.reshape(len(sampled), 21, 3), timestamps, output_timestamps,
                ).reshape(len(frames), 63)
            trajectory = SmplTrajectory(
                timestamps=output_timestamps, betas=self.profile.betas.copy(),
                global_orient=_resample_rotvec(global_np, timestamps, output_timestamps),
                transl=_resample_linear(translation_np, timestamps, output_timestamps),
                body_pose=body_pose_full,
                left_hand_pose=np.zeros((len(frames), 45), np.float32), right_hand_pose=np.zeros((len(frames), 45), np.float32),
                attachment_position=self.mount_position.cpu().numpy().copy(), attachment_rotation=self.mount_rotation.cpu().numpy().copy(),
                scale=self.profile.scale,
            )
            output_frames = tuple(frames)
            output_valid, _ = _stream_validity(output_frames, session_id, self.profile)
            output_target = np.stack([[frame[name].position for name in TRACKER_NAMES] for frame in output_frames]).astype(np.float32)
        else:
            trajectory = optimized_trajectory
            output_frames = sampled
            output_valid = valid_np
            output_target = target_position_np
        output_target_rotation = np.stack(
            [[frame[name].rotation for name in TRACKER_NAMES] for frame in output_frames],
        ).astype(np.float32)
        hard_root_yaw_target = None
        if root_yaw_target_np is not None:
            if len(root_yaw_target_np) == len(trajectory.timestamps) and np.allclose(
                timestamps, trajectory.timestamps, rtol=0.0, atol=1e-5,
            ):
                hard_root_yaw_target = root_yaw_target_np
            elif len(root_yaw_target_np) == len(sampled):
                hard_root_yaw_target = _resample_linear(
                    root_yaw_target_np, timestamps, trajectory.timestamps,
                ).astype(np.float32)
        if self.options.hard_tracker_binding:
            # This opt-in diagnostic projection is the final operation that
            # changes pose/translation.  Do not subsequently apply the normal
            # common-mode root or planted-foot corrections: either would move
            # the endpoints away from the exact rigid-mount test solution.
            trajectory, hard_binding_residual = self._hard_bind_trajectory(
                trajectory, output_target, output_target_rotation, output_valid,
                hard_root_yaw_target,
            )
            translation_correction = np.zeros_like(trajectory.transl)
        elif self.options.hard_root_projection:
            translation_correction = self.project_root_translation(
                trajectory, output_target, output_valid,
            )
            trajectory.transl[...] += translation_correction
            hard_binding_residual = None
        else:
            translation_correction = np.zeros_like(trajectory.transl)
            hard_binding_residual = None
        if len(contact_np) == len(trajectory.timestamps) and np.allclose(timestamps, trajectory.timestamps):
            output_contact = contact_np
        else:
            output_contact = np.stack([
                np.interp(trajectory.timestamps, timestamps, contact_np[:, side].astype(np.float32)) >= .5
                for side in range(2)
            ], axis=1)
        if self.options.hard_tracker_binding:
            foot_lock_projection = np.zeros_like(trajectory.transl)
        else:
            foot_lock_projection = self.project_planted_feet(trajectory, output_contact)
            trajectory.transl[...] += foot_lock_projection
        optimization_ground_y = ground_y
        mesh_floor = self.mesh_floor_height(trajectory)
        # No independent floor sensor exists.  Keep one flat plane, but place
        # it at a robust low percentile of the reconstructed soles so the
        # viewer does not cut through the feet when sparse tracker fitting
        # changes foot pitch by a few centimetres.
        ground_y = min(ground_y, float(np.percentile(mesh_floor, 5)))
        predicted = self.predict_trackers(trajectory)
        residual = np.linalg.norm(predicted - output_target, axis=2)
        residual[~output_valid] = np.nan
        predicted_rotation = self.predict_tracker_rotations(trajectory)
        relative_rotation = np.einsum(
            "ftji,ftjk->ftik", predicted_rotation, output_target_rotation,
        )
        rotation_cosine = np.clip(
            (np.trace(relative_rotation, axis1=2, axis2=3) - 1.0) * .5,
            -1.0, 1.0,
        )
        rotation_residual_deg = np.degrees(np.arccos(rotation_cosine)).astype(np.float32)
        rotation_residual_deg[~output_valid] = np.nan
        per_tracker = {
            name: float(np.sqrt(np.nanmean(np.square(residual[:, index])))) if np.isfinite(residual[:, index]).any() else None
            for index, name in enumerate(TRACKER_NAMES)
        }
        foot_sliding = self.foot_sliding_summary(trajectory, output_contact)
        released_body_names = (
            (
                "left_hip", "right_hip", "spine1", "left_knee", "right_knee",
                "spine2", "left_ankle", "right_ankle", "spine3", "left_foot",
                "right_foot", "neck", "left_collar", "right_collar", "head",
                "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                "left_wrist", "right_wrist",
            ) if self.options.hard_binding_release_all_body else (
                "neck", "head", "left_shoulder", "right_shoulder",
                "left_elbow", "right_elbow", "left_wrist", "right_wrist",
                "left_hip", "right_hip", "left_knee", "right_knee",
                "left_ankle", "right_ankle",
            )
        )
        if self.options.hard_tracker_binding:
            hard_root_mode = (
                "root yaw initialized from and constrained to measured common turn"
                if self.options.hard_binding_release_root and hard_root_yaw_target is not None
                else "root optimized"
                if self.options.hard_binding_release_root
                else "root anchored"
            )
            if self.options.hard_binding_rotation_exact:
                hard_binding_description = (
                    "fixed rigid mounts; raw tracker rotation equality + all-body endpoint IK; "
                    + hard_root_mode + "; pure equality"
                    if self.options.hard_binding_pure_equality else
                    "fixed rigid mounts; raw tracker rotation equality + all-body endpoint IK; "
                    + hard_root_mode
                    + "; robust common-yaw + torso-frame/side-order/hinge barriers; post projections disabled"
                )
            else:
                hard_binding_description = (
                    "fixed rigid mounts; final all-body hard endpoint IK; "
                    + hard_root_mode + "; post projections disabled"
                    if self.options.hard_binding_release_all_body else
                    "fixed rigid mounts; final chain-only hard endpoint IK; "
                    + hard_root_mode + "; post projections disabled"
                )
        else:
            hard_binding_description = "fixed rigid mounts; strong endpoint loss; exact weighted root-translation projection"
        summary = {
            "session": session_id, "source_frames": len(frames), "optimized_frames": len(sampled),
            "fitted_frames": len(trajectory.timestamps), "output_full_rate": self.options.output_full_rate,
            "sample_stride": effective_stride, "requested_sample_stride": self.options.sample_stride,
            "iterations_per_window": self.options.iterations,
            "initial_pose": first_pose,
            "initial_root_yaw": dict(getattr(self, "_initial_yaw_diagnostics", {})),
            "root_yaw_seed_diagnostics": dict(getattr(self, "_root_yaw_seed_diagnostics", {})),
            "body_yaw_target_diagnostics": {
                "enabled": bool(root_yaw_target_np is not None),
                "source": "all_five_tracker_frame_increment_median_deadband_1p2deg" if root_yaw_target_np is not None else None,
                "range_deg": float(np.rad2deg(np.ptp(np.unwrap(body_yaw_offset_np.astype(np.float64)))))
                if len(body_yaw_offset_np) else 0.0,
            },
            "duration_s": float(trajectory.timestamps[-1] - trajectory.timestamps[0]),
            "median_rate_hz": float(1.0 / np.median(np.diff(trajectory.timestamps))) if len(trajectory.timestamps) > 1 else 0.0,
            "gaps_over_0_5s": int(np.sum(np.diff(trajectory.timestamps) > .5)),
            "maximum_gap_s": float(np.max(np.diff(trajectory.timestamps))) if len(trajectory.timestamps) > 1 else 0.0,
            "excluded_trackers": excluded, "position_rms_m": float(np.sqrt(np.nanmean(np.square(residual)))),
            "ground_y_m": ground_y,
            "optimization_ground_y_m": optimization_ground_y,
            "ground_visualization_policy": "lower of physical ankle estimate and fifth percentile reconstructed sole height",
            "contact_fraction": {
                "left_foot": float(contact_np[:, 0].mean()), "right_foot": float(contact_np[:, 1].mean()),
            },
            "foot_sliding": foot_sliding,
            "ankle_mount_model": (
                "profile-native ankle mount"
                if self.options.hard_tracker_binding
                and self.options.hard_binding_use_ankle_rotation
                else "neutral anatomical lateral lower-leg anchor"
            ),
            "ground_model": "flat plane from initial ankle tracker height and shaped ankle-to-sole distance",
            "protocol_alignment": protocol.to_mapping() if protocol is not None else None,
            "dynamic_full_rate_optimization": protocol is not None,
            "observation_driven_protocol_priors": protocol is not None,
            "personal_locomotion_prior": {
                "available": bool(self.profile.locomotion_reference),
                "applied_to_ordinary_raw": bool(
                    protocol is None and self.profile.locomotion_reference
                    and self.options.locomotion_prior_weight > 0
                ),
                "weight": float(self.options.locomotion_prior_weight),
                "gait_seed_active_fraction": locomotion_diagnostics["active_fraction"],
                "maximum_swing_amount": locomotion_diagnostics["maximum_swing_amount"],
                "root_yaw_seed_range_deg": float(np.rad2deg(
                    np.ptp(np.unwrap(root_yaw_offset_np.astype(np.float64)))
                )),
                "source_sessions": self.profile.locomotion_reference.get("source_sessions", []),
            },
            "pose_representation": (
                "VPoser initialization plus full-body temporal IK corrections"
                if self.options.full_body_ik and hard_articulation > 0 else
                ("VPoser plus bounded tracker-chain IK corrections"
                if hard_articulation > 0 else
                ("VPoser decoder only" if self.options.natural_pose_only else "VPoser plus local corrections"))
            ),
            "hard_articulation_fraction": hard_articulation,
            "full_body_ik": bool(self.options.full_body_ik),
            "full_body_joint_weight": float(self.options.full_body_joint_weight),
            "full_body_joint_temporal_weight": float(self.options.full_body_joint_temporal_weight),
            "full_rate_pose_interpolation": (
                "latent then VPoser decode"
                if self.options.natural_pose_only and hard_articulation == 0 else
                "joint rotation SLERP"
            ),
            "post_articulation_refinement": bool(self.options.post_articulation_refinement and not self.options.natural_pose_only),
            "tracker_constraint": hard_binding_description,
            "tracker_hardness": float(self.options.tracker_hardness),
            "hard_tracker_binding": bool(self.options.hard_tracker_binding),
            "hard_tracker_binding_parameters": {
                "iterations": int(self.options.hard_binding_iterations),
                "batch_size": int(self.options.hard_binding_batch_size),
                "refinement_iterations": int(
                    self.options.hard_binding_refinement_iterations
                ),
                "learning_rate": float(self.options.hard_binding_learning_rate),
                "position_weight": float(self.options.hard_binding_position_weight),
                "head_position_multiplier": float(
                    self.options.hard_binding_head_position_multiplier
                ),
                "hand_position_multiplier": float(
                    self.options.hard_binding_hand_position_multiplier
                ),
                "foot_position_multiplier": float(
                    self.options.hard_binding_foot_position_multiplier
                ),
                "dynamic_foot_weight": float(
                    self.options.hard_binding_dynamic_foot_weight
                ),
                "rotation_weight": float(self.options.hard_binding_rotation_weight),
                "rotation_exact": bool(self.options.hard_binding_rotation_exact),
                "project_translation": bool(
                    self.options.hard_binding_project_translation
                    and self.options.hard_binding_release_all_body
                ),
                "translation_deviation_bound_m": float(
                    self.options.hard_binding_translation_bound_m
                ),
                "release_all_body": bool(self.options.hard_binding_release_all_body),
                "release_root": bool(self.options.hard_binding_release_root),
                "pure_equality": bool(self.options.hard_binding_pure_equality),
                "root_yaw_target_in_hard_binding": bool(
                    self.options.hard_binding_release_root and hard_root_yaw_target is not None
                ),
                "root_yaw_target_mode": (
                    "initialized_then_strongly_constrained"
                    if self.options.hard_binding_release_root and hard_root_yaw_target is not None
                    else "learned" if self.options.hard_binding_release_root else "anchored"
                ),
                "torso_axial_twist_limit_deg": 35.0,
                "torso_common_frame_limit_deg": 12.0,
                "torso_axis_barrier_weight": 20000.0,
                "torso_alignment_weight": float(self.options.hard_binding_torso_alignment_weight),
                "pelvis_ankle_horizontal_bound_m": float(
                    self.options.hard_binding_pelvis_ankle_bound_m
                ),
                "pelvis_ankle_center_weight": float(
                    self.options.hard_binding_pelvis_ankle_center_weight
                ),
                "squat_center_weight": float(
                    self.options.hard_binding_squat_center_weight
                ),
                "project_squat_center": bool(
                    self.options.hard_binding_project_squat_center
                ),
                "squat_torso_direction_weight": float(
                    self.options.hard_binding_squat_torso_direction_weight
                ),
                "knee_lateral_limit_m": float(
                    self.options.hard_binding_knee_lateral_limit_m
                ),
                "elbow_lateral_limit_m": float(
                    self.options.hard_binding_elbow_lateral_limit_m
                ),
                "limb_plane_weight": float(
                    self.options.hard_binding_limb_plane_weight
                ),
                "limb_center_weight": float(
                    self.options.hard_binding_limb_center_weight
                ),
                "limb_center_deadzone_m": float(
                    self.options.hard_binding_limb_center_deadzone_m
                ),
                "limb_precondition_iterations": int(
                    self.options.hard_binding_limb_precondition_iterations
                ),
                "right_knee_kick_deg": float(
                    self.options.hard_binding_right_knee_kick_deg
                ),
                "analytic_limb_weight": float(
                    self.options.hard_binding_analytic_limb_weight
                ),
                "analytic_arm_scale": float(
                    self.options.hard_binding_analytic_arm_scale
                ),
                "project_pelvis_horizontal": bool(
                    self.options.hard_binding_project_pelvis
                ),
                "torso_side_lean_limit_deg": float(
                    self.options.hard_binding_torso_side_lean_limit_deg
                ),
                "root_tilt_limit_deg": float(
                    self.options.hard_binding_root_tilt_limit_deg
                ),
                "neck_bend_limit_deg": float(
                    self.options.hard_binding_neck_bend_limit_deg
                ),
                "pelvis_head_min_m": float(
                    self.options.hard_binding_pelvis_head_min_m
                ),
                "pelvis_neck_min_m": float(
                    self.options.hard_binding_pelvis_neck_min_m
                ),
                "neck_twist_limit_deg": float(
                    self.options.hard_binding_neck_twist_limit_deg
                ),
                "neck_relative_angle_limit_deg": float(
                    self.options.hard_binding_neck_relative_angle_limit_deg
                ),
                "spine_bend_limit_deg": float(
                    self.options.hard_binding_spine_bend_limit_deg
                ),
                "spine_axis_tilt_limit_deg": float(
                    self.options.hard_binding_spine_axis_tilt_limit_deg
                ),
                "torso_direction_weight": float(
                    self.options.hard_binding_torso_direction_weight
                ),
                "proximal_anchor_weight": float(
                    self.options.hard_binding_proximal_anchor_weight
                ),
                "torso_local_twist_weight": float(
                    self.options.hard_binding_torso_local_twist_weight
                ),
                "spine_direction_weight": float(
                    self.options.hard_binding_spine_direction_weight
                ),
                "spine_opposite_weight": float(
                    self.options.hard_binding_spine_opposite_weight
                ),
                "spine_curve_limit_deg": float(
                    self.options.hard_binding_spine_curve_limit_deg
                ),
                "spine_curve_weight": float(
                    self.options.hard_binding_spine_curve_weight
                ),
                "root_yaw_only": bool(self.options.hard_binding_root_yaw_only),
                "barrier_scale_from_position_weight": max(
                    1.0, float(self.options.hard_binding_position_weight) / 1.0e6,
                ),
                "effective_torso_alignment_weight": float(
                    self.options.hard_binding_torso_alignment_weight
                ) * np.sqrt(max(
                    1.0, float(self.options.hard_binding_position_weight) / 1.0e6,
                )),
                "effective_anatomical_barrier_weight": 30000.0 * max(
                    1.0, float(self.options.hard_binding_position_weight) / 1.0e6,
                ),
                "effective_crossing_barrier_weight": 1000000.0 * max(
                    1.0, float(self.options.hard_binding_position_weight) / 1.0e6,
                ),
                "joint_step_limit_deg": float(self.options.hard_binding_joint_step_limit_deg),
                "effective_strict_smooth_weight": 800.0 * max(
                    1.0, float(self.options.hard_binding_position_weight) / 1.0e6,
                ),
                "effective_strict_jump_weight": 100000.0 * max(
                    1.0, float(self.options.hard_binding_position_weight) / 1.0e6,
                ),
                "hard_anatomical_barrier_weight": 30000.0,
                "hard_crossing_barrier_weight": 1000000.0,
                "hard_anatomical_barriers": [
                    "left-right ordering at hips/knees/ankles/shoulders/elbows/wrists",
                    "knee sagittal hinge and minimum angle",
                    "elbow sagittal hinge and minimum angle",
                    "broad torso forward/side-lean barrier",
                    "pelvis root up-axis tilt barrier",
                    "pelvis horizontal position near the rigid ankle midpoint",
                    "shoulder/hip transverse-axis roll barrier (20 deg)",
                    "pelvis-head/neck chain length and neck bend barrier",
                ],
                "foot_rotation_frame": (
                    "ankle_joint_7_8" if self.options.hard_binding_use_ankle_rotation
                    else "lower_shank_knee_frame_4_5"
                ),
                "released_body_pose_joints": list(released_body_names),
                "post_projection": "disabled in hard-binding mode",
            },
            "tracker_constraint_multipliers": {
                "head": float(self.options.head_tracker_constraint),
                "wrists": float(self.options.wrist_tracker_constraint),
                "feet": float(self.options.foot_tracker_constraint),
                "per_tracker": dict(zip(
                    TRACKER_NAMES,
                    _tracker_constraint_multipliers(self.options).astype(float).tolist(),
                    strict=True,
                )),
            },
            "sole_floor_constraint": {
                "enabled": bool(self.options.sole_floor_constraint),
                "contact_only": True,
                "virtual_points": "heel/forefoot from shaped ankle/toe-to-sole offsets",
                "normal_alignment": "foot local +Y to world up (10 cm lever equivalent)",
                "independent_penetration_and_lock": True,
            },
            "ik_branch_policy": (
                "nearest corrected local-joint solution on SO(3), with previous-frame "
                "and cross-window continuity"
            ),
            "joint_continuity_weight": float(self.options.joint_continuity_weight),
            "joint_jump_weight": float(self.options.joint_jump_weight),
            "maximum_joint_speed_rad_s": float(self.options.maximum_joint_speed_rad_s),
            "knee_hinge_weight": float(self.options.knee_hinge_weight),
            "joint_range_weight": float(self.options.joint_range_weight),
            "behavior_driven_protocol": bool(self.options.behavior_driven_protocol),
            "biomechanical_branch_repair": bool(self.options.biomechanical_branch_repair),
            "torso_bend_weight": float(self.options.torso_bend_weight),
            "branch_temporal_selector": {
                "enabled": bool(self.options.biomechanical_branch_repair),
                "joint_smooth_weight": float(self.options.branch_joint_smooth_weight),
                "joint_acceleration_weight": float(self.options.branch_joint_acceleration_weight),
                "leg_smooth_weight": float(self.options.branch_leg_smooth_weight),
                "root_acceleration_weight": float(self.options.branch_root_acceleration_weight),
                "max_joint_speed_rad_s": float(self.options.branch_max_joint_speed_rad_s),
                "max_joint_step_deg": float(self.options.branch_max_joint_step_deg),
                "leg_crossing_weight": float(self.options.leg_crossing_weight),
                "leg_crossing_margin_m": float(self.options.leg_crossing_margin_m),
                "semantics": "SO(3) continuity selects among tracker-compatible IK branches; no endpoint clamp",
            },
            "walking_fit_parameters": {
                "optimizer_learning_rate": float(self.options.optimizer_learning_rate),
                "tracker_rotation_weight": float(self.options.tracker_rotation_weight),
                "tracker_rotation_hardness_factor": float(.05 + .95 * np.clip(self.options.tracker_hardness, 0.0, 1.0)),
                "temporal_weight": float(self.options.temporal_weight),
                "pose_prior_weight": float(self.options.pose_prior_weight),
                "locomotion_prior_weight": float(self.options.locomotion_prior_weight),
                "upright_weight": float(self.options.upright_weight),
                "foot_contact_weight": float(self.options.foot_contact_weight),
                "sole_floor_constraint": bool(self.options.sole_floor_constraint),
                "foot_penetration_weight": float(self.options.foot_penetration_weight),
                "foot_lock_weight": float(self.options.foot_lock_weight),
                "contact_height_threshold_m": self.contact_height_threshold_m,
                "contact_speed_threshold_m_s": self.contact_speed_threshold_m_s,
                "requested_contact_height_threshold_m": float(self.options.contact_height_threshold_m),
                "requested_contact_speed_threshold_m_s": float(self.options.contact_speed_threshold_m_s),
                "root_tracking_weight": float(self.options.root_tracking_weight),
                "root_bound_weight": float(self.options.root_bound_weight),
                "root_bound_m": float(self.options.root_bound_m),
                "second_order_weight": float(self.options.second_order_weight),
                "leg_correction_weight": float(self.options.leg_correction_weight),
                "knee_hinge_weight": float(self.options.knee_hinge_weight),
                "joint_range_weight": float(self.options.joint_range_weight),
                "biomechanical_branch_repair": bool(self.options.biomechanical_branch_repair),
                "torso_bend_weight": float(self.options.torso_bend_weight),
            },
            "local_joint_step_diagnostics": _local_joint_step_summary(
                trajectory.global_orient, trajectory.body_pose, trajectory.timestamps,
            ),
            "effective_tracker_position_weight": float(
                self.options.tracker_position_weight
                * (10.0 ** (3.0 * (self.options.tracker_hardness - .5)))
            ),
            "first_window_seed_boundary": (
                f"free-start output blended into stable window over {self.options.initial_transition_s:.3g}s; "
                "stable bounded solve retained for later window warm starts"
            ),
            "root_projection_correction_cm": {
                "median": 100.0 * float(np.median(np.linalg.norm(translation_correction, axis=1))),
                "p95": 100.0 * float(np.percentile(np.linalg.norm(translation_correction, axis=1), 95)),
                "maximum": 100.0 * float(np.max(np.linalg.norm(translation_correction, axis=1))),
            },
            "foot_lock_projection_cm": {
                "median": 100.0 * float(np.median(np.linalg.norm(foot_lock_projection, axis=1))),
                "p95": 100.0 * float(np.percentile(np.linalg.norm(foot_lock_projection, axis=1), 95)),
                "maximum": 100.0 * float(np.max(np.linalg.norm(foot_lock_projection, axis=1))),
            },
            "tracker_position_rms_m": per_tracker,
            "tracker_rotation_rms_deg": {
                name: float(np.sqrt(np.nanmean(np.square(rotation_residual_deg[:, index]))))
                if np.isfinite(rotation_residual_deg[:, index]).any() else None
                for index, name in enumerate(TRACKER_NAMES)
            },
        }
        return ProfileFitResult(
            trajectory, output_frames, output_valid, predicted, residual, ground_y,
            summary, protocol, predicted_rotation, rotation_residual_deg,
        )

    def project_root_translation(
        self, trajectory: SmplTrajectory, target: np.ndarray, valid: np.ndarray,
    ) -> np.ndarray:
        """Eliminate common tracker drift with a closed-form root projection.

        For fixed pose, orientation, body shape and tracker mounts, root
        translation is linear.  Its weighted least-squares solution makes the
        model follow the measured tracker constellation exactly in common-mode
        motion, leaving only skeletal/pose incompatibility as visible residual.
        """
        predicted = self.predict_trackers(trajectory)
        tracker_weight = (
            np.asarray((1.25, 1.0, 1.0, 1.15, 1.15), dtype=np.float32)
            * _tracker_constraint_multipliers(self.options)
        )
        weight = valid.astype(np.float32) * tracker_weight[None]
        correction = (
            ((target - predicted) * weight[:, :, None]).sum(axis=1)
            / weight.sum(axis=1, keepdims=True).clip(min=1e-6)
        )
        return correction.astype(np.float32)

    def _foot_world_points(self, trajectory: SmplTrajectory) -> np.ndarray:
        """Return world-space left/right ankle and forefoot points."""
        import torch

        selected_ids = torch.as_tensor((7, 8, 10, 11), device=self.device)
        points = []
        for start in range(0, len(trajectory.timestamps), 256):
            end = min(len(trajectory.timestamps), start + 256)
            count = end - start
            zero3 = torch.zeros((count, 3), device=self.device)
            zero10 = torch.zeros((count, 10), device=self.device)
            with torch.no_grad():
                output = self.body(
                    betas=self.betas[None].expand(count, -1),
                    global_orient=torch.as_tensor(trajectory.global_orient[start:end], device=self.device),
                    transl=zero3,
                    body_pose=torch.as_tensor(trajectory.body_pose[start:end], device=self.device),
                    left_hand_pose=zero3.new_zeros((count, 45)),
                    right_hand_pose=zero3.new_zeros((count, 45)),
                    jaw_pose=zero3, leye_pose=zero3, reye_pose=zero3,
                    expression=zero10, return_verts=False,
                )
                joints = self.profile.scale * output.joints[:, selected_ids]
                joints += torch.as_tensor(trajectory.transl[start:end], device=self.device)[:, None]
            points.append(joints.cpu().numpy())
        return np.concatenate(points)

    def project_planted_feet(
        self, trajectory: SmplTrajectory, contact: np.ndarray,
    ) -> np.ndarray:
        """Apply a soft stance-run root correction after tracker projection.

        The differentiable foot-lock loss chooses a better pose during IK.
        Exact tracker root projection happens afterwards and can reintroduce a
        common horizontal drift, so this final bounded correction restores the
        requested user-visible lock. Its strength follows the same adjustable
        ``foot_lock_weight`` and remains zero when that control is disabled.
        """
        correction = np.zeros((len(trajectory.timestamps), 3), dtype=np.float32)
        if self.options.foot_lock_weight <= 0 or len(trajectory.timestamps) < 2:
            return correction
        world = self._foot_world_points(trajectory)
        centers = np.stack((
            world[:, (0, 2), :][:, :, (0, 2)].mean(axis=1),
            world[:, (1, 3), :][:, :, (0, 2)].mean(axis=1),
        ), axis=1)
        horizontal = np.zeros((len(world), 2), dtype=np.float32)
        count = np.zeros(len(world), dtype=np.float32)
        for side in range(2):
            indices = np.flatnonzero(contact[:, side])
            if not len(indices):
                continue
            runs = np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1)
            for run in runs:
                if trajectory.timestamps[run[-1]] - trajectory.timestamps[run[0]] < .10:
                    continue
                anchor = np.median(centers[run, side], axis=0)
                horizontal[run] += anchor - centers[run, side]
                count[run] += 1.0
        active = count > 0
        horizontal[active] /= count[active, None]
        step_interval = float(
            self.profile.locomotion_reference.get("gait", {}).get("step_interval_s", 1.0)
        )
        horizontal = _smooth_seconds(
            horizontal, trajectory.timestamps, np.clip(.10 * step_interval, .08, .16),
        )
        strength = float(self.options.foot_lock_weight) / (
            float(self.options.foot_lock_weight) + 100.0
        )
        projected = strength * horizontal
        maximum_shift = .020 + .040 * strength
        length = np.linalg.norm(projected, axis=1)
        scale = np.minimum(1.0, maximum_shift / np.maximum(length, 1e-8))
        correction[:, (0, 2)] = projected * scale[:, None]
        return correction

    def foot_sliding_summary(
        self, trajectory: SmplTrajectory, contact: np.ndarray,
    ) -> dict[str, dict[str, float | int]]:
        """Measure final ankle/toe horizontal speed during detected stance."""
        world = self._foot_world_points(trajectory)
        if len(world) < 2:
            empty = {"planted_pairs": 0, "rms_m_s": 0.0, "p95_m_s": 0.0, "maximum_m_s": 0.0}
            return {"left_foot": dict(empty), "right_foot": dict(empty)}
        dt = np.maximum(np.diff(trajectory.timestamps), 1e-4)
        result = {}
        for side, name, ids in ((0, "left_foot", (0, 2)), (1, "right_foot", (1, 3))):
            velocity = np.diff(world[:, ids][:, :, (0, 2)], axis=0) / dt[:, None, None]
            speed = np.sqrt(np.mean(np.square(velocity), axis=(1, 2)))
            planted = contact[1:, side] & contact[:-1, side] & (dt < .45)
            values = speed[planted]
            result[name] = {
                "planted_pairs": int(planted.sum()),
                "rms_m_s": float(np.sqrt(np.mean(np.square(values)))) if len(values) else 0.0,
                "p95_m_s": float(np.percentile(values, 95)) if len(values) else 0.0,
                "maximum_m_s": float(values.max()) if len(values) else 0.0,
            }
        return result

    def mesh_floor_height(self, trajectory: SmplTrajectory) -> np.ndarray:
        """Return the lowest body-mesh Y coordinate for every frame."""
        import torch

        result = []
        with torch.no_grad():
            for start in range(0, len(trajectory.timestamps), 128):
                end = min(len(trajectory.timestamps), start + 128)
                count = end - start
                zero3 = torch.zeros((count, 3), device=self.device)
                zero10 = torch.zeros((count, 10), device=self.device)
                output = self.body(
                    betas=self.betas[None].expand(count, -1),
                    global_orient=torch.as_tensor(trajectory.global_orient[start:end], device=self.device),
                    transl=zero3,
                    body_pose=torch.as_tensor(trajectory.body_pose[start:end], device=self.device),
                    left_hand_pose=torch.as_tensor(trajectory.left_hand_pose[start:end], device=self.device),
                    right_hand_pose=torch.as_tensor(trajectory.right_hand_pose[start:end], device=self.device),
                    jaw_pose=zero3, leye_pose=zero3, reye_pose=zero3,
                    expression=zero10, return_verts=True,
                )
                translation_y = torch.as_tensor(
                    trajectory.transl[start:end, 1], dtype=torch.float32, device=self.device,
                )
                height = trajectory.scale * output.vertices[:, :, 1].amin(dim=1) + translation_y
                result.append(height.cpu().numpy())
        return np.concatenate(result).astype(np.float32)

    def predict_trackers(self, trajectory: SmplTrajectory) -> np.ndarray:
        import torch

        result = []
        with torch.no_grad():
            for start in range(0, len(trajectory.timestamps), 100):
                end = min(len(trajectory.timestamps), start + 100); count = end - start
                zero3 = torch.zeros((count, 3), device=self.device); zero10 = torch.zeros((count, 10), device=self.device)
                global_orient = torch.as_tensor(trajectory.global_orient[start:end], device=self.device)
                body_pose = torch.as_tensor(trajectory.body_pose[start:end], device=self.device).reshape(count, 21, 3)
                output = self.body(
                    betas=self.betas[None].expand(count, -1), global_orient=global_orient, transl=zero3,
                    body_pose=body_pose.reshape(count, -1), left_hand_pose=torch.zeros((count, 45), device=self.device),
                    right_hand_pose=torch.zeros((count, 45), device=self.device), jaw_pose=zero3, leye_pose=zero3,
                    reye_pose=zero3, expression=zero10, return_verts=False,
                )
                rotation = _global_body_rotations(global_orient, body_pose, self.parents)[:, self.rotation_joint_ids]
                predicted = trajectory.scale * output.joints[:, self.position_joint_ids]
                predicted += torch.as_tensor(trajectory.transl[start:end], device=self.device)[:, None]
                predicted += (rotation @ self.mount_position[None, :, :, None]).squeeze(-1)
                result.append(predicted.cpu().numpy())
        return np.concatenate(result).astype(np.float32)

    def predict_tracker_rotations(self, trajectory: SmplTrajectory) -> np.ndarray:
        """Return the five predicted raw-convention tracker rotation matrices."""
        import torch

        result = []
        with torch.no_grad():
            for start in range(0, len(trajectory.timestamps), 128):
                end = min(len(trajectory.timestamps), start + 128)
                count = end - start
                global_orient = torch.as_tensor(
                    trajectory.global_orient[start:end], dtype=torch.float32, device=self.device,
                )
                body_pose = torch.as_tensor(
                    trajectory.body_pose[start:end], dtype=torch.float32, device=self.device,
                ).reshape(count, 21, 3)
                rotations = _global_body_rotations(global_orient, body_pose, self.parents)
                tracked = rotations[:, self.rotation_joint_ids]
                predicted = tracked @ self.mount_rotation[None]
                result.append(predicted.cpu().numpy())
        return np.concatenate(result).astype(np.float32)
