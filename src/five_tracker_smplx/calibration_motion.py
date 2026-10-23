"""Anatomical SMPL-X calibration motions generated in learned pose spaces.

The body is solved in VPoser's 32-D latent space against explicit joint
landmarks.  The hands are solved by MANO articulated-chain landmark IK from
the model's canonical flat-hand mean.
This deliberately avoids authored per-joint axis-angle tables: those are easy
to get mirrored or to rotate about the wrong local axis.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .geometry import matrix_to_quaternion_xyzw
from .model import ModelPaths, load_body_model, load_vposer
from .optimizer import _global_body_rotations, _rotation_distance_squared


_CACHE_VERSION = 10
_FLAT_FOOT_VERSION = 3
_FOOT_ACTION_VERSION = 2
_WRIST_ACTION_VERSION = 1
_NATURAL_FOOT_YAW = np.deg2rad(20.0)
_WRIST_ROLL_ANGLE = np.deg2rad(95.0)
_WRIST_PITCH_ANGLE = np.deg2rad(67.0)
_WRIST_ACTIONS = (
    ("wrist_pronate", "roll", -_WRIST_ROLL_ANGLE),
    ("wrist_supinate", "roll", _WRIST_ROLL_ANGLE),
    ("wrist_flex", "pitch", _WRIST_PITCH_ANGLE),
    ("wrist_extend", "pitch", -_WRIST_PITCH_ANGLE),
)
_LEGACY_BODY_KEYS = (
    "neutral", "a_pose", "t_pose", "forward", "elbow_left", "elbow_right",
    "squat", "left_leg", "right_leg", "heel_raise", "toe_raise",
    "head_left", "head_right", "head_down", "head_up",
    "wrist_pronate", "wrist_supinate", "wrist_flex", "wrist_extend",
)
_BODY_KEYS = _LEGACY_BODY_KEYS + (
    "walk_stride", "walk_left_swing", "walk_right_swing",
)
_FINGERS = ("index", "middle", "ring", "pinky")
_HAND_JOINTS = {
    "left": {
        "index": (25, 26, 27, 67), "middle": (28, 29, 30, 68),
        "pinky": (31, 32, 33, 70), "ring": (34, 35, 36, 69),
        "thumb": (37, 38, 39, 66),
    },
    "right": {
        "index": (40, 41, 42, 72), "middle": (43, 44, 45, 73),
        "pinky": (46, 47, 48, 75), "ring": (49, 50, 51, 74),
        "thumb": (52, 53, 54, 71),
    },
}
_WRIST = {"left": 20, "right": 21}


def _smooth(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def _pulse(phase: float, cycles: int = 1) -> float:
    """Smooth 0 -> 1 -> 0 pulse, repeated ``cycles`` times."""
    local = (phase * cycles) % 1.0
    return _smooth(0.5 - 0.5 * np.cos(2.0 * np.pi * local))


def _hold(phase: float) -> float:
    if phase < 0.2:
        return _smooth(phase / 0.2)
    if phase <= 0.8:
        return 1.0
    return _smooth((1.0 - phase) / 0.2)


def _protocol_body_value(values: dict[str, np.ndarray], motion: str, phase: float) -> np.ndarray:
    def blend(first: str, second: str, amount: float) -> np.ndarray:
        amount = _smooth(amount)
        return (1.0 - amount) * values[first] + amount * values[second]

    def path(keys: tuple[str, ...]) -> np.ndarray:
        position = np.clip(phase, 0.0, 1.0) * (len(keys) - 1)
        index = min(len(keys) - 2, int(position))
        return blend(keys[index], keys[index + 1], position - index)

    if motion == "head":
        return path(("neutral", "head_left", "neutral", "head_right", "neutral", "head_down", "neutral", "head_up", "neutral"))
    if motion in ("a_pose", "t_pose", "forward"):
        return blend("neutral", motion, _hold(phase))
    if motion == "elbows":
        return path((
            "t_pose", "elbow_right", "t_pose", "elbow_left", "t_pose",
            "elbow_right", "t_pose", "elbow_left", "t_pose",
        ))
    if motion == "wrists":
        return path(("forward", "wrist_pronate", "wrist_supinate", "wrist_flex", "wrist_extend", "forward"))
    if motion == "squat":
        return blend("neutral", "squat", _pulse(phase, 2))
    if motion in ("left_knee", "right_knee"):
        key = "left_leg" if motion == "left_knee" else "right_leg"
        return blend("neutral", key, _pulse(phase, 2))
    if motion == "ankles":
        return path(("neutral", "heel_raise", "neutral", "toe_raise", "neutral"))
    if motion in ("hand_open", "fist", "pinch", "fingers"):
        return values["forward"].copy()
    if motion == "turns":
        keys = ["neutral"]
        for step in range(16):
            keys.extend(("left_leg" if step % 2 == 0 else "right_leg", "neutral"))
        return path(tuple(keys))
    if motion == "walk":
        # Exactly two footfalls forward followed by two footfalls backward.
        # Earlier code repeated each left/right cycle twice (eight footfalls),
        # contradicting the written "前两步、后两步" instruction.
        return path((
            "neutral", "walk_left_swing", "walk_stride", "walk_right_swing", "neutral",
            "walk_right_swing", "walk_stride", "walk_left_swing", "neutral",
        ))
    return values["neutral"].copy()


def protocol_body_latent(body_latents: dict[str, np.ndarray], motion: str, phase: float) -> np.ndarray:
    """Return the VPoser seed corresponding to one aligned protocol phase."""
    # Isolated distal articulation is deliberately kept outside VPoser. Its
    # learned manifold otherwise distributes elbow/wrist motion into the
    # shoulders and torso. Local SMPL-X corrections carry those paths below.
    if motion == "elbows":
        return body_latents["t_pose"].copy()
    if motion == "wrists":
        return body_latents["forward"].copy()
    return _protocol_body_value(body_latents, motion, phase)


def protocol_body_correction(body_corrections: dict[str, np.ndarray], motion: str, phase: float) -> np.ndarray:
    """Return the joint-local correction paired with a protocol VPoser seed."""
    return _protocol_body_value(body_corrections, motion, phase)


def _unit(values: np.ndarray) -> np.ndarray:
    return values / max(float(np.linalg.norm(values)), 1e-8)


def _matrix_to_rotvec(matrix: np.ndarray) -> np.ndarray:
    """Convert an SO(3) matrix to the shortest, numerically stable rotvec."""
    quaternion = matrix_to_quaternion_xyzw(matrix)
    if quaternion[3] < 0.0:
        quaternion = -quaternion
    vector = quaternion[:3]
    sine = float(np.linalg.norm(vector))
    if sine < 1e-8:
        return (2.0 * vector).astype(np.float32)
    angle = 2.0 * np.arctan2(sine, float(quaternion[3]))
    return (vector * (angle / sine)).astype(np.float32)


@dataclass(frozen=True)
class CalibrationFrame:
    body_pose: np.ndarray
    left_hand_pose: np.ndarray
    right_hand_pose: np.ndarray


class CalibrationMotionLibrary:
    """Solve and play a standard calibration protocol in VPoser/MANO space."""

    def __init__(
        self,
        body_model_path: str | Path,
        vposer_path: str | Path,
        *,
        cache_path: str | Path | None = None,
        rebuild_cache: bool = False,
        device: str = "cpu",
        verbose: bool = True,
    ) -> None:
        self.device = torch.device(device)
        paths = ModelPaths(Path(body_model_path), Path(vposer_path))
        self.body = load_body_model(paths, device).eval()
        self.vposer = load_vposer(paths, device).eval()
        for model in (self.body, self.vposer):
            for parameter in model.parameters():
                parameter.requires_grad_(False)
        self.dtype = torch.float32
        self.verbose = verbose
        self.cache_path = Path(cache_path) if cache_path is not None else None
        # SMPL-X adds the MANO mean internally.  Its exact negative is the
        # canonical MANO flat/open hand (equivalent to flat_hand_mean=True).
        self.open_hand = {
            "left": self._project_flat_hand("left"),
            "right": self._project_flat_hand("right"),
        }
        self.body_latents: dict[str, np.ndarray] = {}
        self.body_corrections: dict[str, np.ndarray] = {}
        self.hand_poses: dict[str, dict[str, np.ndarray]] = {"left": {}, "right": {}}
        if not rebuild_cache and self._load_cache():
            return
        self._build_body_library()
        self._build_hand_library()
        self._save_cache()

    def _project_flat_hand(self, side: str) -> torch.Tensor:
        mean = getattr(self.body, f"{side}_hand_mean").detach().to(self.device)
        return -mean

    def _load_cache(self) -> bool:
        if self.cache_path is None or not self.cache_path.exists():
            return False
        with np.load(self.cache_path, allow_pickle=False) as cache:
            if int(cache["version"]) != _CACHE_VERSION:
                return False
            cache_keys = tuple(cache["body_keys"].tolist())
            if cache_keys not in (_LEGACY_BODY_KEYS, _BODY_KEYS):
                return False
            latents = cache["body_latents"].astype(np.float32)
            self.body_latents = dict(zip(cache_keys, latents, strict=True))
            corrections = cache["body_corrections"].astype(np.float32)
            self.body_corrections = dict(zip(cache_keys, corrections, strict=True))
            flat_foot_version = int(cache["flat_foot_version"]) if "flat_foot_version" in cache.files else 0
            foot_action_version = int(cache["foot_action_version"]) if "foot_action_version" in cache.files else 0
            wrist_action_version = int(cache["wrist_action_version"]) if "wrist_action_version" in cache.files else 0
            for side in ("left", "right"):
                keys = cache[f"{side}_hand_keys"].tolist()
                values = cache[f"{side}_hand_poses"].astype(np.float32)
                self.hand_poses[side] = dict(zip(keys, values, strict=True))
        changed = False
        if cache_keys == _LEGACY_BODY_KEYS:
            # Extend the already validated 01--15 cache with only three new
            # VPoser locomotion keys instead of rebuilding every body/hand key.
            self._build_locomotion_library()
            for key in _BODY_KEYS[len(_LEGACY_BODY_KEYS):]:
                self.body_corrections[key] = np.zeros(63, dtype=np.float32)
            changed = True
        if flat_foot_version < _FLAT_FOOT_VERSION:
            # Older caches aligned the whole mesh by its single lowest vertex.
            # That can leave one heel touching while the opposite forefoot is
            # several centimetres high. Migrate the cached poses once using
            # actual sole vertices and retain the result in the cache.
            self._apply_flat_foot_corrections()
            changed = True
        if foot_action_version < _FOOT_ACTION_VERSION:
            self._apply_foot_action_corrections()
            changed = True
        if wrist_action_version < _WRIST_ACTION_VERSION:
            # Preserve the expensive VPoser/foot/head solutions in existing
            # caches and update only the two local wrist joints.  The former
            # +/-49 and +/-30 degree reference actions were much smaller than
            # the measured 0827 protocol (about +/-95 and +/-67 degrees).
            self._apply_wrist_action_corrections()
            changed = True
        if changed:
            self._save_cache()
        if self.verbose:
            print(f"loaded VPoser/MANO calibration motions: {self.cache_path}")
        return True

    def _save_cache(self) -> None:
        if self.cache_path is None:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, np.ndarray | int] = {
            "version": np.asarray(_CACHE_VERSION),
            "flat_foot_version": np.asarray(_FLAT_FOOT_VERSION),
            "foot_action_version": np.asarray(_FOOT_ACTION_VERSION),
            "wrist_action_version": np.asarray(_WRIST_ACTION_VERSION),
            "body_keys": np.asarray(_BODY_KEYS),
            "body_latents": np.stack([self.body_latents[key] for key in _BODY_KEYS]),
            "body_corrections": np.stack([self.body_corrections[key] for key in _BODY_KEYS]),
        }
        for side in ("left", "right"):
            keys = tuple(self.hand_poses[side])
            payload[f"{side}_hand_keys"] = np.asarray(keys)
            payload[f"{side}_hand_poses"] = np.stack([self.hand_poses[side][key] for key in keys])
        np.savez_compressed(self.cache_path, **payload)
        if self.verbose:
            print(f"saved VPoser/MANO calibration motions: {self.cache_path}")

    def _decode(self, latent: torch.Tensor) -> torch.Tensor:
        # Keep VPoser's learned mean.  Subtracting decode(0) would turn the
        # natural standing prior back into SMPL-X's modelling T-pose.
        return self.vposer.decode(latent)["pose_body"].reshape(-1, 63)

    def _body_output(
        self,
        body_pose: torch.Tensor,
        left_hand: torch.Tensor | None = None,
        right_hand: torch.Tensor | None = None,
        *,
        return_verts: bool = False,
    ):
        count = body_pose.shape[0]
        zero3 = torch.zeros((count, 3), dtype=self.dtype, device=self.device)
        zero10 = torch.zeros((count, 10), dtype=self.dtype, device=self.device)
        if left_hand is None:
            left_hand = self.open_hand["left"].reshape(1, 45).expand(count, -1)
        if right_hand is None:
            right_hand = self.open_hand["right"].reshape(1, 45).expand(count, -1)
        return self.body(
            betas=zero10, global_orient=zero3, transl=zero3,
            body_pose=body_pose, left_hand_pose=left_hand, right_hand_pose=right_hand,
            jaw_pose=zero3, leye_pose=zero3, reye_pose=zero3,
            expression=zero10, return_verts=return_verts,
        )

    def _joints(self, latent: torch.Tensor) -> torch.Tensor:
        return self._body_output(self._decode(latent), return_verts=False).joints

    def _solve_body(
        self,
        name: str,
        targets: dict[int, np.ndarray],
        *,
        initial: np.ndarray | torch.Tensor | None = None,
        iterations: int = 420,
        latent_weight: float = 0.018,
        keep_sides: bool = False,
        side_margin: float = 0.055,
        joint_weights: dict[int, float] | None = None,
        rotation_targets: dict[int, np.ndarray] | None = None,
        palm_down: bool = False,
        hands_forward: bool = False,
    ) -> np.ndarray:
        if initial is None:
            initial_tensor = torch.zeros((1, 32), dtype=self.dtype, device=self.device)
        else:
            initial_tensor = torch.as_tensor(initial, dtype=self.dtype, device=self.device).reshape(1, 32)
        latent = torch.nn.Parameter(initial_tensor.clone())
        ids = torch.as_tensor(tuple(targets), dtype=torch.long, device=self.device)
        goal = torch.as_tensor(np.stack(tuple(targets.values())), dtype=self.dtype, device=self.device)
        weights = torch.as_tensor(
            [1.0 if joint_weights is None else joint_weights.get(joint, 1.0) for joint in targets],
            dtype=self.dtype, device=self.device,
        )
        rotation_ids = None
        rotation_goal = None
        if rotation_targets:
            rotation_ids = torch.as_tensor(tuple(rotation_targets), dtype=torch.long, device=self.device)
            rotation_goal = torch.as_tensor(np.stack(tuple(rotation_targets.values())), dtype=self.dtype, device=self.device)
        optimizer = torch.optim.Adam((latent,), lr=0.045)
        best_loss = float("inf")
        best = latent.detach().clone()
        for _ in range(iterations):
            optimizer.zero_grad(set_to_none=True)
            body_pose = self._decode(latent)
            joints = self._body_output(body_pose, return_verts=False).joints[0]
            point_error = (joints[ids] - goal).square().mean(dim=1)
            position = (point_error * weights).sum() / weights.sum()
            loss = 550.0 * position + latent_weight * latent.square().mean()
            if rotation_ids is not None and rotation_goal is not None:
                rotations = _global_body_rotations(
                    torch.zeros((1, 3), dtype=self.dtype, device=self.device),
                    body_pose.reshape(1, 21, 3), self.body.parents.to(self.device),
                )[0, rotation_ids]
                loss = loss + 8.0 * _rotation_distance_squared(rotations, rotation_goal).mean()
            if palm_down:
                palm_loss = torch.zeros((), dtype=self.dtype, device=self.device)
                direction_loss = torch.zeros((), dtype=self.dtype, device=self.device)
                for side, sign in (("left", 1.0), ("right", -1.0)):
                    hand = _HAND_JOINTS[side]
                    wrist = joints[_WRIST[side]]
                    direction = torch.nn.functional.normalize(joints[hand["middle"][-1]] - wrist, dim=0)
                    across = torch.nn.functional.normalize(joints[hand["index"][0]] - joints[hand["pinky"][0]], dim=0)
                    normal = sign * torch.nn.functional.normalize(torch.cross(direction, across, dim=0), dim=0)
                    palm_loss = palm_loss + (normal - torch.tensor((0., -1., 0.), dtype=self.dtype, device=self.device)).square().mean()
                    if hands_forward:
                        direction_loss = direction_loss + (direction - torch.tensor((0., 0., 1.), dtype=self.dtype, device=self.device)).square().mean()
                loss = loss + 28.0 * palm_loss
                if hands_forward:
                    loss = loss + 10.0 * direction_loss
            if keep_sides:
                # Anatomical left is +X in SMPL-X.  Apply this to knees,
                # ankles and forefeet so no solution can swap/cross the legs.
                left = joints[torch.as_tensor((4, 7, 10), device=self.device), 0]
                right = joints[torch.as_tensor((5, 8, 11), device=self.device), 0]
                side_loss = torch.relu(side_margin - left).square().mean()
                side_loss += torch.relu(side_margin + right).square().mean()
                # Both forefeet must point exactly along +Z rather than
                # inheriting an artificial outward toe angle.
                forward = torch.stack((joints[10, 2] - joints[7, 2], joints[11, 2] - joints[8, 2]))
                straight = torch.stack((joints[10, 0] - joints[7, 0], joints[11, 0] - joints[8, 0]))
                loss = loss + 220.0 * torch.relu(0.075 - forward).square().mean() + 90.0 * straight.square().mean() + 300.0 * side_loss
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                latent.clamp_(-4.0, 4.0)
                value = float(loss.detach())
                if value < best_loss:
                    best_loss = value
                    best.copy_(latent)
        result = best.detach().cpu().numpy()[0].astype(np.float32)
        if self.verbose:
            with torch.no_grad():
                achieved = self._joints(best)[0, ids]
                rms = torch.sqrt((achieved - goal).square().mean()).item()
            print(f"VPoser key {name:>15s}: joint RMS={rms:.4f} m, |z|={np.linalg.norm(result):.2f}")
        return result

    @staticmethod
    def _copy_targets(points: np.ndarray, ids: tuple[int, ...]) -> dict[int, np.ndarray]:
        return {joint: points[joint].copy() for joint in ids}

    @staticmethod
    def _arm_landmarks(
        side: str,
        shoulder: np.ndarray,
        direction: np.ndarray,
        upper_length: float,
        forearm_length: float,
        *,
        palm_normal: np.ndarray,
    ) -> dict[int, np.ndarray]:
        sign = 1.0 if side == "left" else -1.0
        shoulder_id, elbow_id, wrist_id = ((16, 18, 20) if side == "left" else (17, 19, 21))
        hand = _HAND_JOINTS[side]
        direction = _unit(direction)
        normal = _unit(palm_normal)
        across = _unit(np.cross(normal, direction)) * sign
        elbow = shoulder + upper_length * direction
        wrist = elbow + forearm_length * direction
        return {
            shoulder_id: shoulder.copy(), elbow_id: elbow, wrist_id: wrist,
            hand["middle"][0]: wrist + 0.052 * direction,
            hand["middle"][-1]: wrist + 0.175 * direction,
            hand["index"][0]: wrist + 0.050 * direction + 0.022 * across,
            hand["pinky"][0]: wrist + 0.045 * direction - 0.030 * across,
        }

    def _build_body_library(self) -> None:
        with torch.no_grad():
            raw = self._joints(torch.zeros((1, 32), dtype=self.dtype, device=self.device))[0].cpu().numpy()
        center_x = float(raw[0, 0])
        neutral_targets = self._copy_targets(raw, (0, 1, 2, 3, 6, 9, 12, 13, 14, 15, 16, 17))
        # Upright torso, relaxed shoulders and parallel feet.  These are
        # landmark targets, not local rotation guesses.
        for joint in (3, 6, 9, 12, 15):
            neutral_targets[joint][0] = center_x
        # Pelvis-to-head stays almost vertical; only retain a natural two
        # centimetre forward offset rather than VPoser's stronger slouch.
        for joint, dz in ((3, .002), (6, .005), (9, .009), (12, .014), (15, .019)):
            neutral_targets[joint][2] = raw[0, 2] + dz
        for joint, x, z in ((4, .105, .02), (5, -.105, .02), (7, .105, .0), (8, -.105, .0), (10, .105, .12), (11, -.105, .12)):
            neutral_targets[joint] = raw[joint].copy()
            neutral_targets[joint][0] = x
            neutral_targets[joint][2] = z
        neutral_targets.update({
            18: np.array((.255, raw[18, 1], .085), np.float32),
            19: np.array((-.255, raw[19, 1], .085), np.float32),
            20: np.array((.285, raw[20, 1] - .045, .105), np.float32),
            21: np.array((-.285, raw[21, 1] - .045, .105), np.float32),
        })
        neutral = self._solve_body("neutral", neutral_targets, keep_sides=True, iterations=520)
        self.body_latents["neutral"] = neutral
        with torch.no_grad():
            base = self._joints(torch.as_tensor(neutral, device=self.device)[None])[0].cpu().numpy()
        stable = self._copy_targets(base, (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15))
        upper_length = 0.5 * (np.linalg.norm(base[18] - base[16]) + np.linalg.norm(base[19] - base[17]))
        forearm_length = 0.5 * (np.linalg.norm(base[20] - base[18]) + np.linalg.norm(base[21] - base[19]))

        def arm_pose(name: str, left_direction: np.ndarray, right_direction: np.ndarray, palm_normal: np.ndarray, initial=neutral):
            targets = dict(stable)
            targets.update(self._arm_landmarks("left", base[16], left_direction, upper_length, forearm_length, palm_normal=palm_normal))
            targets.update(self._arm_landmarks("right", base[17], right_direction, upper_length, forearm_length, palm_normal=palm_normal))
            self.body_latents[name] = self._solve_body(
                name, targets, initial=initial, keep_sides=True,
                palm_down=True, hands_forward=(name == "forward"), iterations=560,
            )

        arm_pose("a_pose", np.array((1., -1., 0.)), np.array((-1., -1., 0.)), np.array((0., -1., 0.)))
        arm_pose("t_pose", np.array((1., 0., 0.)), np.array((-1., 0., 0.)), np.array((0., -1., 0.)))
        arm_pose("forward", np.array((0., 0., 1.)), np.array((0., 0., 1.)), np.array((0., -1., 0.)))
        t_targets = dict(stable)
        t_points = self._joints(torch.as_tensor(self.body_latents["t_pose"], device=self.device)[None]).detach().cpu().numpy()[0]
        t_targets.update(self._copy_targets(t_points, (16, 17, 18, 19, 20, 21, 25, 28, 31, 34, 40, 43, 46, 49)))
        for side, name in (("left", "elbow_left"), ("right", "elbow_right")):
            targets = dict(t_targets)
            shoulder_id, elbow_id = ((16, 18) if side == "left" else (17, 19))
            direction = np.array((0., 1., 0.))
            targets.update(self._arm_landmarks(side, t_points[shoulder_id],
                _unit(t_points[elbow_id] - t_points[shoulder_id]), upper_length, forearm_length,
                palm_normal=np.array((0., 0., 1.))))
            wrist_id = _WRIST[side]
            elbow_target = targets[elbow_id]
            wrist_target = elbow_target + forearm_length * direction
            targets[wrist_id] = wrist_target
            hand = _HAND_JOINTS[side]
            targets[hand["middle"][0]] = wrist_target + .052 * direction
            targets[hand["middle"][-1]] = wrist_target + .175 * direction
            self.body_latents[name] = self._solve_body(name, targets, initial=self.body_latents["t_pose"], keep_sides=True)

        squat_targets = self._copy_targets(base, (0, 1, 2, 3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21))
        for joint, x, dy, z in ((4,.17,.16,.19),(5,-.17,.16,.19),(7,.12,.25,.015),(8,-.12,.25,.015),(10,.12,.25,.14),(11,-.12,.25,.14)):
            squat_targets[joint] = base[joint].copy() + np.array((0., dy, 0.), np.float32)
            squat_targets[joint][0] = x; squat_targets[joint][2] = z
        self.body_latents["squat"] = self._solve_body(
            "squat", squat_targets, initial=neutral, keep_sides=True,
            side_margin=.10, iterations=620,
        )

        for side, name in (("left", "left_leg"), ("right", "right_leg")):
            lifted = (1, 4, 7, 10) if side == "left" else (2, 5, 8, 11)
            stance = (2, 5, 8, 11) if side == "left" else (1, 4, 7, 10)
            sign = 1.0 if side == "left" else -1.0
            targets = self._copy_targets(base, (0, 3, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21) + stance)
            targets[lifted[0]] = base[lifted[0]].copy()
            targets[lifted[1]] = np.array((.11*sign, base[lifted[0],1]-.22, .30), np.float32)
            targets[lifted[2]] = np.array((.11*sign, base[lifted[0],1]-.46, .16), np.float32)
            targets[lifted[3]] = np.array((.11*sign, base[lifted[0],1]-.50, .29), np.float32)
            self.body_latents[name] = self._solve_body(name, targets, initial=neutral, keep_sides=True, iterations=560)

        for name, ankle_up, toe_up in (("heel_raise", .075, 0.), ("toe_raise", 0., .155)):
            targets = self._copy_targets(base, (0, 3, 4, 5, 6, 9, 12, 15, 16, 17, 18, 19, 20, 21))
            for ankle, foot in ((7,10),(8,11)):
                targets[ankle] = base[ankle] + np.array((0., ankle_up, 0.), np.float32)
                targets[foot] = base[foot] + np.array((0., toe_up, 0.), np.float32)
                targets[foot][0] = targets[ankle][0]
                targets[foot][2] = targets[ankle][2] + .12
            weights = {7: 2.0, 8: 2.0, 10: 4.0, 11: 4.0} if name == "toe_raise" else {7: 3.0, 8: 3.0, 10: 3.0, 11: 3.0}
            self.body_latents[name] = self._solve_body(
                name, targets, initial=neutral, keep_sides=True,
                joint_weights=weights, iterations=560,
            )

        # Head motions are defined in the global body frame: +Y is up, +Z is
        # forward and +X is the subject's left.  This removes any ambiguity
        # from camera/viewer mirroring or neck-local axes.
        neutral_pose = self._decode(torch.as_tensor(neutral, device=self.device)[None]).reshape(1, 21, 3)
        with torch.no_grad():
            neutral_rotations = _global_body_rotations(
                torch.zeros((1, 3), dtype=self.dtype, device=self.device),
                neutral_pose, self.body.parents.to(self.device),
            )[0].cpu().numpy()

        def world_rotation(axis: str, angle: float) -> np.ndarray:
            c, s = np.cos(angle), np.sin(angle)
            if axis == "x":
                return np.asarray(((1.,0.,0.),(0.,c,-s),(0.,s,c)), np.float32)
            return np.asarray(((c,0.,s),(0.,1.,0.),(-s,0.,c)), np.float32)

        for name, axis, angle in (
            ("head_left", "y", np.deg2rad(32.)),
            ("head_right", "y", np.deg2rad(-32.)),
            ("head_down", "x", np.deg2rad(25.)),
            ("head_up", "x", np.deg2rad(-22.)),
        ):
            targets = self._copy_targets(base, tuple(range(0, 16)) + (16,17,18,19,20,21))
            rotation_target = world_rotation(axis, angle) @ neutral_rotations[15]
            self.body_latents[name] = self._solve_body(
                name, targets, initial=neutral, keep_sides=True,
                latent_weight=.025, rotation_targets={15: rotation_target},
                joint_weights={15: .2}, iterations=520,
            )

        with torch.no_grad():
            forward_points = self._joints(torch.as_tensor(self.body_latents["forward"], device=self.device)[None])[0].cpu().numpy()
        forward_keep = self._copy_targets(forward_points, tuple(range(0, 22)))
        # Define wrist rotations through MANO landmarks.  Targets are rotated
        # around the actual wrist-to-middle-finger direction, not a local axis.
        for name, kind, angle in _WRIST_ACTIONS:
            targets = dict(forward_keep)
            for side in ("left", "right"):
                wrist = forward_points[_WRIST[side]]
                ids = sum((_HAND_JOINTS[side][finger] for finger in ("index","middle","pinky","ring","thumb")), ())
                axis = np.array((0., 0., 1.)) if kind == "roll" else np.array((1., 0., 0.))
                side_angle = angle if kind == "pitch" else angle * (1.0 if side == "left" else -1.0)
                cross = np.array(((0.,-axis[2],axis[1]),(axis[2],0.,-axis[0]),(-axis[1],axis[0],0.)), np.float32)
                rotation = np.eye(3, dtype=np.float32) + np.sin(side_angle)*cross + (1.-np.cos(side_angle))*(cross@cross)
                for joint in ids:
                    targets[joint] = wrist + rotation @ (forward_points[joint] - wrist)
            self.body_latents[name] = self._solve_body(name, targets, initial=self.body_latents["forward"], keep_sides=True, iterations=480)

        self._build_locomotion_library()
        self.body_corrections = {key: np.zeros(63, dtype=np.float32) for key in _BODY_KEYS}
        # An isolated 90-degree elbow flexion sits close to the edge of
        # VPoser's learned whole-body manifold.  The latent-only solution
        # therefore leaves the forearm visibly too straight (especially on
        # the anatomical left side).  Keep VPoser as the whole-body prior and
        # solve only the distal elbow residual in SMPL-X local coordinates.
        # This is the same treatment used below for gaze and dorsiflexion.
        self.body_corrections["elbow_left"] += self._solve_elbow_correction("elbow_left", "left")
        self.body_corrections["elbow_right"] += self._solve_elbow_correction("elbow_right", "right")
        for key, kind, angle in _WRIST_ACTIONS:
            self.body_corrections[key] += self._solve_wrist_correction(key, kind, angle)
        canonical_head_targets = {
            key: np.eye(3, dtype=np.float32) for key in _BODY_KEYS
        }
        canonical_head_targets.update({
            "head_left": world_rotation("y", np.deg2rad(32.)),
            "head_right": world_rotation("y", np.deg2rad(-32.)),
            "head_down": world_rotation("x", np.deg2rad(25.)),
            "head_up": world_rotation("x", np.deg2rad(-22.)),
        })
        for key in _BODY_KEYS:
            self.body_corrections[key] += self._solve_head_correction(
                key, canonical_head_targets[key],
                latent_key=("t_pose" if key.startswith("elbow_") else "forward" if key.startswith("wrist_") else key),
            )
        # VPoser intentionally models the high-probability body manifold and
        # under-represents isolated ankle dorsiflexion.  Solve that distal
        # residual from the SMPL-X forefoot landmark instead of inventing a
        # local ankle axis or accepting an invisible toe raise.
        self.body_corrections["toe_raise"] += self._solve_ankle_correction("toe_raise")
        self._apply_flat_foot_corrections()
        self._apply_foot_action_corrections()

    def _build_locomotion_library(self) -> None:
        """Solve contact-compatible walking keys inside VPoser's pose space."""
        neutral = self.body_latents["neutral"]
        with torch.no_grad():
            base = self._joints(torch.as_tensor(neutral, device=self.device)[None])[0].cpu().numpy()

        upper = (3, 6, 9, 12, 13, 14, 15, 16, 17)
        hips = (1, 2)
        arms = (18, 19, 20, 21)

        stride = self._copy_targets(base, upper + hips + arms)
        # Double-support stride: anatomical left foot is one short step ahead,
        # right foot behind. Ankles rise slightly relative to the pelvis so the
        # VPoser solution flexes the knees instead of lengthening the legs.
        for knee, ankle, toe, dz, sign in (
            (4, 7, 10, .155, 1.0), (5, 8, 11, -.155, -1.0),
        ):
            stride[knee] = base[knee] + np.array((.008 * sign, .025, .055 * np.sign(dz)), np.float32)
            stride[ankle] = base[ankle] + np.array((0., .020, dz), np.float32)
            stride[toe] = base[toe] + np.array((0., .020, dz), np.float32)
        self.body_latents["walk_stride"] = self._solve_body(
            "walk_stride", stride, initial=neutral, keep_sides=True,
            joint_weights={7: 3., 8: 3., 10: 3., 11: 3.}, iterations=620,
        )

        for side, key in (("left", "walk_left_swing"), ("right", "walk_right_swing")):
            is_left = side == "left"
            swing = (1, 4, 7, 10) if is_left else (2, 5, 8, 11)
            stance = (2, 5, 8, 11) if is_left else (1, 4, 7, 10)
            side_sign = 1.0 if is_left else -1.0
            targets = self._copy_targets(base, upper + hips)
            # The root has already advanced by 8 cm at mid-swing. The stance
            # foot therefore sits 8 cm behind/ahead in pelvis coordinates;
            # the swing foot follows a 10 cm vertical clearance arc.
            stance_dz = -.080 if is_left else .080
            for joint in stance:
                targets[joint] = base[joint] + np.array((0., 0., stance_dz), np.float32)
            targets[swing[1]] = base[swing[1]] + np.array((.010 * side_sign, .085, .045 * side_sign), np.float32)
            targets[swing[2]] = base[swing[2]] + np.array((0., .105, .080 * side_sign), np.float32)
            targets[swing[3]] = base[swing[3]] + np.array((0., .105, .080 * side_sign), np.float32)

            # Contralateral arm swing is also solved as a landmark target,
            # keeping shoulders relaxed and avoiding hand-authored rotations.
            left_arm_dz = -.085 if is_left else .085
            right_arm_dz = -left_arm_dz
            for elbow, wrist, dz in ((18, 20, left_arm_dz), (19, 21, right_arm_dz)):
                targets[elbow] = base[elbow] + np.array((0., .010, .45 * dz), np.float32)
                targets[wrist] = base[wrist] + np.array((0., .020, dz), np.float32)
            self.body_latents[key] = self._solve_body(
                key, targets, initial=neutral, keep_sides=True,
                joint_weights={swing[2]: 3., swing[3]: 3., stance[2]: 3., stance[3]: 3.},
                iterations=680,
            )

    def _solve_elbow_correction(self, key: str, side: str) -> np.ndarray:
        """Fit a local elbow residual so the forearm reaches vertical."""
        latent = torch.as_tensor(self.body_latents["t_pose"], dtype=self.dtype, device=self.device)[None]
        base_pose = self._decode(latent).detach()
        pose_index = 17 if side == "left" else 18
        shoulder, elbow, wrist = ((16, 18, 20) if side == "left" else (17, 19, 21))
        with torch.no_grad():
            base_joints = self._body_output(base_pose, return_verts=False).joints[0]
            forearm_length = torch.linalg.vector_norm(base_joints[wrist] - base_joints[elbow])
            goal_wrist = base_joints[elbow] + torch.tensor(
                (0.0, float(forearm_length), 0.0), dtype=self.dtype, device=self.device,
            )
        residual = torch.nn.Parameter(torch.zeros((1, 3), dtype=self.dtype, device=self.device))
        optimizer = torch.optim.Adam((residual,), lr=.055)
        best_loss = float("inf")
        best = residual.detach().clone()
        for _ in range(420):
            optimizer.zero_grad(set_to_none=True)
            pose = base_pose.clone()
            begin = 3 * pose_index
            pose[:, begin:begin + 3] = base_pose[:, begin:begin + 3] + residual
            joints = self._body_output(pose, return_verts=False).joints[0]
            wrist_error = (joints[wrist] - goal_wrist).square().mean()
            elbow_stability = (joints[elbow] - base_joints[elbow]).square().mean()
            shoulder_stability = (joints[shoulder] - base_joints[shoulder]).square().mean()
            loss = 3600.0 * wrist_error + 500.0 * (elbow_stability + shoulder_stability) + .01 * residual.square().mean()
            loss.backward(); optimizer.step()
            with torch.no_grad():
                residual.clamp_(-2.4, 2.4)
                value = float(loss.detach())
                if value < best_loss:
                    best_loss = value; best.copy_(residual)
        correction = np.zeros(63, dtype=np.float32)
        begin = 3 * pose_index
        correction[begin:begin + 3] = best.detach().cpu().numpy()[0]
        if self.verbose:
            with torch.no_grad():
                pose = base_pose.clone(); pose[:, begin:begin + 3] += best
                joints = self._body_output(pose, return_verts=False).joints[0]
                first = joints[shoulder] - joints[elbow]
                second = joints[wrist] - joints[elbow]
                cosine = torch.dot(first, second) / (
                    torch.linalg.vector_norm(first) * torch.linalg.vector_norm(second)
                ).clamp_min(1e-6)
                angle = torch.rad2deg(torch.arccos(cosine.clamp(-1.0, 1.0)))
            print(f"SMPL-X elbow residual {key}: flexion angle={float(angle):.2f} deg")
        return correction

    def _solve_wrist_correction(self, key: str, kind: str, angle: float) -> np.ndarray:
        """Fit pure local wrist articulation on a fixed forward arm chain."""
        latent = torch.as_tensor(self.body_latents["forward"], dtype=self.dtype, device=self.device)[None]
        base_pose = self._decode(latent).detach()
        with torch.no_grad():
            base_rotations = _global_body_rotations(
                torch.zeros((1, 3), dtype=self.dtype, device=self.device),
                base_pose.reshape(1, 21, 3), self.body.parents.to(self.device),
            )[0]
        residual = torch.nn.Parameter(torch.zeros((2, 3), dtype=self.dtype, device=self.device))
        goals = []
        axis = "z" if kind == "roll" else "x"
        for side_index, wrist in enumerate((20, 21)):
            side_angle = angle if kind == "pitch" or side_index == 0 else -angle
            c, s = np.cos(side_angle), np.sin(side_angle)
            if axis == "z":
                world_delta = np.asarray(((c, -s, 0.), (s, c, 0.), (0., 0., 1.)), np.float32)
            else:
                world_delta = np.asarray(((1., 0., 0.), (0., c, -s), (0., s, c)), np.float32)
            goals.append(torch.as_tensor(world_delta, dtype=self.dtype, device=self.device) @ base_rotations[wrist])
        goal = torch.stack(goals)
        optimizer = torch.optim.Adam((residual,), lr=.06)
        best_loss = float("inf")
        best = residual.detach().clone()
        for _ in range(360):
            optimizer.zero_grad(set_to_none=True)
            pose = base_pose.clone()
            pose[:, 57:60] += residual[0]
            pose[:, 60:63] += residual[1]
            rotations = _global_body_rotations(
                torch.zeros((1, 3), dtype=self.dtype, device=self.device),
                pose.reshape(1, 21, 3), self.body.parents.to(self.device),
            )[0, torch.as_tensor((20, 21), device=self.device)]
            loss = 30.0 * _rotation_distance_squared(rotations, goal).mean() + .002 * residual.square().mean()
            loss.backward(); optimizer.step()
            with torch.no_grad():
                residual.clamp_(-1.8, 1.8)
                value = float(loss.detach())
                if value < best_loss:
                    best_loss = value; best.copy_(residual)
        correction = np.zeros(63, dtype=np.float32)
        correction[57:60] = best[0].detach().cpu().numpy()
        correction[60:63] = best[1].detach().cpu().numpy()
        if self.verbose:
            print(f"SMPL-X wrist residual {key}: error={np.rad2deg(np.sqrt(best_loss / 30.0)):.3f} deg")
        return correction

    def _apply_wrist_action_corrections(self) -> None:
        """Migrate cached wrist actions without disturbing other corrections."""
        for key, kind, angle in _WRIST_ACTIONS:
            correction = self._solve_wrist_correction(key, kind, angle)
            self.body_corrections[key][57:63] = correction[57:63]

    def _solve_head_correction(
        self, key: str, target_rotation: np.ndarray, *, latent_key: str | None = None,
    ) -> np.ndarray:
        """Fit the head-local rotation to an unambiguous world-frame gaze."""
        latent = torch.as_tensor(self.body_latents[latent_key or key], dtype=self.dtype, device=self.device)[None]
        base_pose = self._decode(latent).detach()
        head_local = torch.nn.Parameter(base_pose[:, 42:45].clone())
        goal = torch.as_tensor(target_rotation, dtype=self.dtype, device=self.device)[None]
        optimizer = torch.optim.Adam((head_local,), lr=.07)
        best_loss = float("inf")
        best = head_local.detach().clone()
        for _ in range(240):
            optimizer.zero_grad(set_to_none=True)
            pose = base_pose.clone(); pose[:, 42:45] = head_local
            predicted = _global_body_rotations(
                torch.zeros((1, 3), dtype=self.dtype, device=self.device),
                pose.reshape(1, 21, 3), self.body.parents.to(self.device),
            )[:, 15]
            loss = _rotation_distance_squared(predicted, goal).mean()
            loss.backward(); optimizer.step()
            with torch.no_grad():
                value = float(loss.detach())
                if value < best_loss:
                    best_loss = value; best.copy_(head_local)
        correction = np.zeros(63, dtype=np.float32)
        correction[42:45] = (best - base_pose[:, 42:45]).detach().cpu().numpy()[0]
        if self.verbose:
            print(f"SMPL-X gaze residual {key:>15s}: error={np.rad2deg(np.sqrt(best_loss)):.3f} deg")
        return correction

    def _solve_ankle_correction(self, key: str) -> np.ndarray:
        latent = torch.as_tensor(self.body_latents[key], dtype=self.dtype, device=self.device)[None]
        base_pose = self._decode(latent).detach()
        residual = torch.nn.Parameter(torch.zeros((2, 3), dtype=self.dtype, device=self.device))
        with torch.no_grad():
            key_joints = self._body_output(base_pose, return_verts=False).joints[0]
        targets = []
        for ankle, outward in ((7, .012), (8, -.012)):
            target = key_joints[ankle].clone()
            target[0] += outward
            target[1] += .045
            target[2] += .120
            targets.append(target)
        goal = torch.stack(targets)
        optimizer = torch.optim.Adam((residual,), lr=.045)
        best_loss = float("inf")
        best = residual.detach().clone()
        for _ in range(620):
            optimizer.zero_grad(set_to_none=True)
            pose = base_pose.clone()
            pose[:, 18:21] = base_pose[:, 18:21] + residual[0]
            pose[:, 21:24] = base_pose[:, 21:24] + residual[1]
            joints = self._body_output(pose, return_verts=False).joints[0]
            landmark = (joints[torch.as_tensor((10, 11), device=self.device)] - goal).square().mean()
            symmetry = (residual[0] - torch.tensor((1., -1., -1.), device=self.device) * residual[1]).square().mean()
            loss = 2600.0 * landmark + .025 * residual.square().mean() + .01 * symmetry
            loss.backward(); optimizer.step()
            with torch.no_grad():
                residual.clamp_(-1.35, 1.35)
                value = float(loss.detach())
                if value < best_loss:
                    best_loss = value; best.copy_(residual)
        correction = np.zeros(63, dtype=np.float32)
        correction[18:21] = best[0].detach().cpu().numpy()
        correction[21:24] = best[1].detach().cpu().numpy()
        if self.verbose:
            with torch.no_grad():
                pose = base_pose.clone(); pose[:, 18:21] += best[0]; pose[:, 21:24] += best[1]
                joints = self._body_output(pose, return_verts=False).joints[0]
                heights = joints[torch.as_tensor((10,11)), 1] - joints[torch.as_tensor((7,8)), 1]
            print(f"SMPL-X ankle residual {key}: forefoot above ankle={np.round(heights.cpu().numpy(), 3).tolist()} m")
        return correction

    def _sole_vertex_indices(self, body_pose: torch.Tensor, side: str) -> torch.Tensor:
        """Sample the underside across heel/arch/forefoot using SMPL-X skinning."""
        ankle, forefoot = ((7, 10) if side == "left" else (8, 11))
        weights = self.body.lbs_weights.detach().to(self.device)
        candidates = torch.nonzero(weights[:, ankle] + weights[:, forefoot] > .45).flatten()
        with torch.no_grad():
            vertices = self._body_output(body_pose, return_verts=True).vertices[0]
        points = vertices[candidates]
        # Fixed spatial bins stop the optimiser from flattening only the one
        # currently-lowest heel vertex. Two samples from each of twelve cells
        # cover the full width and length of the sole.
        x_edges = torch.linspace(points[:, 0].min(), points[:, 0].max(), 4, device=self.device)
        z_edges = torch.linspace(points[:, 2].min(), points[:, 2].max(), 5, device=self.device)
        selected: list[torch.Tensor] = []
        for x_index in range(3):
            for z_index in range(4):
                x_mask = (points[:, 0] >= x_edges[x_index]) & (
                    points[:, 0] <= x_edges[x_index + 1] if x_index == 2 else points[:, 0] < x_edges[x_index + 1]
                )
                z_mask = (points[:, 2] >= z_edges[z_index]) & (
                    points[:, 2] <= z_edges[z_index + 1] if z_index == 3 else points[:, 2] < z_edges[z_index + 1]
                )
                local = torch.nonzero(x_mask & z_mask).flatten()
                if local.numel():
                    count = min(2, int(local.numel()))
                    lowest = local[torch.topk(points[local, 1], count, largest=False).indices]
                    selected.append(candidates[lowest])
        if not selected:
            raise RuntimeError(f"could not identify {side} SMPL-X sole vertices")
        return torch.cat(selected)

    def _solve_flat_foot_correction(self, key: str) -> np.ndarray:
        """Fit both ankle orientations so each rendered sole lies in one plane."""
        latent = torch.as_tensor(self.body_latents[key], dtype=self.dtype, device=self.device)[None]
        base_pose = self._decode(latent).detach() + torch.as_tensor(
            self.body_corrections[key], dtype=self.dtype, device=self.device,
        )[None]
        sole_indices = {
            side: self._sole_vertex_indices(base_pose, side) for side in ("left", "right")
        }
        residual = torch.nn.Parameter(torch.zeros((2, 3), dtype=self.dtype, device=self.device))
        optimizer = torch.optim.Adam((residual,), lr=.045)
        best_loss = float("inf")
        best = residual.detach().clone()
        for _ in range(420):
            optimizer.zero_grad(set_to_none=True)
            pose = base_pose.clone()
            pose[:, 18:21] += residual[0]
            pose[:, 21:24] += residual[1]
            output = self._body_output(pose, return_verts=True)
            vertices = output.vertices[0]
            joints = output.joints[0]
            heights = [vertices[sole_indices[side], 1] for side in ("left", "right")]
            flatness = sum((height - height.mean()).square().mean() for height in heights)
            foot_directions = torch.stack((joints[10] - joints[7], joints[11] - joints[8]))
            headings = torch.atan2(foot_directions[:, 0], foot_directions[:, 2])
            desired_headings = torch.tensor(
                (_NATURAL_FOOT_YAW, -_NATURAL_FOOT_YAW),
                dtype=self.dtype, device=self.device,
            )
            straight_ahead = (headings - desired_headings).square().mean()
            forward = torch.relu(.085 - foot_directions[:, 2]).square().mean()
            # Both feet are expected on the same flat floor for every static
            # key except single-support leg/swing keys. Their lifted foot is
            # still levelled, but is not pulled toward the stance-foot plane.
            same_floor = torch.zeros((), dtype=self.dtype, device=self.device)
            if key not in ("left_leg", "right_leg", "walk_left_swing", "walk_right_swing"):
                same_floor = (heights[0].mean() - heights[1].mean()).square()
            loss = (
                18000.0 * flatness + 6000.0 * same_floor
                + 220.0 * straight_ahead + 8000.0 * forward
                + .012 * residual.square().mean()
            )
            loss.backward(); optimizer.step()
            with torch.no_grad():
                residual.clamp_(-1.0, 1.0)
                value = float(loss.detach())
                if value < best_loss:
                    best_loss = value; best.copy_(residual)
        correction = np.zeros(63, dtype=np.float32)
        correction[18:21] = best[0].detach().cpu().numpy()
        correction[21:24] = best[1].detach().cpu().numpy()
        if self.verbose:
            with torch.no_grad():
                pose = base_pose.clone(); pose[:, 18:21] += best[0]; pose[:, 21:24] += best[1]
                output = self._body_output(pose, return_verts=True)
                vertices = output.vertices[0]
                spreads = [
                    1000.0 * float(torch.quantile(vertices[sole_indices[side], 1], .9) - torch.quantile(vertices[sole_indices[side], 1], .1))
                    for side in ("left", "right")
                ]
                joints = output.joints[0]
                headings = [
                    float(torch.rad2deg(torch.atan2(joints[toe, 0] - joints[ankle, 0], joints[toe, 2] - joints[ankle, 2])))
                    for ankle, toe in ((7, 10), (8, 11))
                ]
            print(
                f"SMPL-X sole residual {key:>15s}: 10-90% spread={np.round(spreads, 2).tolist()} mm, "
                f"heading={np.round(headings, 2).tolist()} deg"
            )
        return correction

    def _apply_flat_foot_corrections(self) -> None:
        """Level every ordinary pose while preserving deliberate ankle raises."""
        for key in _BODY_KEYS:
            if key in ("heel_raise", "toe_raise"):
                continue
            self.body_corrections[key] += self._solve_flat_foot_correction(key)

    def _foot_action_samples(
        self, body_pose: torch.Tensor, side: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return fixed low heel/forefoot samples for a posed foot."""
        ankle, forefoot = ((7, 10) if side == "left" else (8, 11))
        indices = self._sole_vertex_indices(body_pose, side)
        with torch.no_grad():
            output = self._body_output(body_pose, return_verts=True)
            vertices = output.vertices[0]
            joints = output.joints[0]
            direction = joints[forefoot] - joints[ankle]
            direction[1] = 0.0
            direction = direction / torch.linalg.vector_norm(direction).clamp_min(1e-6)
            coordinate = vertices[indices] @ direction
            split = torch.median(coordinate)
            groups = (indices[coordinate <= split], indices[coordinate > split])
        return groups[0], groups[1]

    def _solve_foot_action_correction(
        self, key: str, *, mode: str, target_height: float,
    ) -> np.ndarray:
        """Balance heel/toe raises from actual mesh-sole height differences."""
        latent = torch.as_tensor(self.body_latents[key], dtype=self.dtype, device=self.device)[None]
        base_pose = self._decode(latent).detach() + torch.as_tensor(
            self.body_corrections[key], dtype=self.dtype, device=self.device,
        )[None]
        samples = {
            side: self._foot_action_samples(base_pose, side) for side in ("left", "right")
        }
        residual = torch.nn.Parameter(torch.zeros((2, 3), dtype=self.dtype, device=self.device))
        optimizer = torch.optim.Adam((residual,), lr=.045)
        best_loss = float("inf")
        best = residual.detach().clone()
        for _ in range(420):
            optimizer.zero_grad(set_to_none=True)
            pose = base_pose.clone()
            pose[:, 18:21] += residual[0]
            pose[:, 21:24] += residual[1]
            vertices = self._body_output(pose, return_verts=True).vertices[0]
            contacts = []
            lifts = []
            for side in ("left", "right"):
                heel_ids, front_ids = samples[side]
                heel = torch.quantile(vertices[heel_ids, 1], .2)
                front = torch.quantile(vertices[front_ids, 1], .2)
                if mode == "heel":
                    contacts.append(front); lifts.append(heel - front)
                else:
                    contacts.append(heel); lifts.append(front - heel)
            lifts_tensor = torch.stack(lifts)
            contacts_tensor = torch.stack(contacts)
            height_error = (lifts_tensor - target_height).square().mean()
            symmetry = (lifts_tensor[0] - lifts_tensor[1]).square()
            contact_level = (contacts_tensor[0] - contacts_tensor[1]).square()
            loss = (
                10000.0 * height_error + 5000.0 * symmetry
                + 3000.0 * contact_level + .018 * residual.square().mean()
            )
            loss.backward()
            with torch.no_grad():
                value = float(loss.detach())
                if value < best_loss:
                    best_loss = value; best.copy_(residual)
            optimizer.step()
            with torch.no_grad():
                residual.clamp_(-1.1, 1.1)
        correction = np.zeros(63, dtype=np.float32)
        correction[18:21] = best[0].detach().cpu().numpy()
        correction[21:24] = best[1].detach().cpu().numpy()
        if self.verbose:
            with torch.no_grad():
                pose = base_pose.clone(); pose[:, 18:21] += best[0]; pose[:, 21:24] += best[1]
                vertices = self._body_output(pose, return_verts=True).vertices[0]
                achieved = []
                for side in ("left", "right"):
                    heel_ids, front_ids = samples[side]
                    heel = torch.quantile(vertices[heel_ids, 1], .2)
                    front = torch.quantile(vertices[front_ids, 1], .2)
                    achieved.append(float(heel - front if mode == "heel" else front - heel))
            print(f"SMPL-X {mode} raise {key}: lift={np.round(1000.0 * np.asarray(achieved), 1).tolist()} mm")
        return correction

    def _apply_foot_action_corrections(self) -> None:
        self.body_corrections["heel_raise"] += self._solve_foot_action_correction(
            "heel_raise", mode="heel", target_height=.045,
        )
        self.body_corrections["toe_raise"] += self._solve_foot_action_correction(
            "toe_raise", mode="toe", target_height=.040,
        )

    def _stabilize_head(self, body_pose: np.ndarray) -> np.ndarray:
        """Keep the head upright/forward after nonlinear pose interpolation."""
        result = np.asarray(body_pose, dtype=np.float32).copy()
        pose = torch.as_tensor(result, dtype=self.dtype, device=self.device).reshape(1, 21, 3)
        with torch.no_grad():
            rotations = _global_body_rotations(
                torch.zeros((1, 3), dtype=self.dtype, device=self.device),
                pose, self.body.parents.to(self.device),
            )[0]
        parent = int(self.body.parents[15])
        # R_head_global = R_parent_global @ R_head_local. Solving the local
        # matrix analytically avoids axis-angle interpolation's roll/pitch
        # artefact between alternating walking keys.
        desired_local = rotations[parent].transpose(0, 1).cpu().numpy()
        result[42:45] = _matrix_to_rotvec(desired_local)
        return result

    def _straighten_feet(self, body_pose: np.ndarray) -> np.ndarray:
        """Keep a tiny natural toe-out without changing sole pitch/roll."""
        result = np.asarray(body_pose, dtype=np.float32).copy()
        pose = torch.as_tensor(result, dtype=self.dtype, device=self.device).reshape(1, 21, 3)
        with torch.no_grad():
            rotations = _global_body_rotations(
                torch.zeros((1, 3), dtype=self.dtype, device=self.device),
                pose, self.body.parents.to(self.device),
            )[0]
            joints = self._body_output(pose.reshape(1, 63), return_verts=False).joints[0]
        for side_index, (ankle, toe) in enumerate(((7, 10), (8, 11))):
            direction = joints[toe] - joints[ankle]
            heading = torch.atan2(direction[0], direction[2])
            target = heading.new_tensor(_NATURAL_FOOT_YAW if side_index == 0 else -_NATURAL_FOOT_YAW)
            correction = target - heading
            cosine, sine = torch.cos(correction), torch.sin(correction)
            yaw = torch.stack((
                torch.stack((cosine, torch.zeros_like(cosine), sine)),
                torch.stack((torch.zeros_like(cosine), torch.ones_like(cosine), torch.zeros_like(cosine))),
                torch.stack((-sine, torch.zeros_like(cosine), cosine)),
            ))
            desired_global = yaw @ rotations[ankle]
            parent = int(self.body.parents[ankle])
            desired_local = rotations[parent].transpose(0, 1) @ desired_global
            begin = 3 * (ankle - 1)
            result[begin:begin + 3] = _matrix_to_rotvec(desired_local.cpu().numpy())
        return result

    def _skeleton_positions(self, body_pose: torch.Tensor) -> torch.Tensor:
        """Fast differentiable SMPL-X FK for the 22 body-joint locations."""
        body_pose = body_pose.reshape(-1, 21, 3)
        count = body_pose.shape[0]
        rotations = _global_body_rotations(
            torch.zeros((count, 3), dtype=self.dtype, device=self.device),
            body_pose, self.body.parents.to(self.device),
        )
        if not hasattr(self, "_rest_body_offsets"):
            with torch.no_grad():
                regressor = self.body.J_regressor.to(dtype=self.dtype, device=self.device)
                template = self.body.v_template.to(dtype=self.dtype, device=self.device)
                rest = torch.einsum("jv,vk->jk", regressor, template)[:22]
                offsets = rest.clone()
                for joint in range(1, 22):
                    offsets[joint] = rest[joint] - rest[int(self.body.parents[joint])]
                offsets[0].zero_()
                self._rest_body_offsets = offsets
        positions = [torch.zeros((count, 3), dtype=self.dtype, device=self.device)]
        for joint in range(1, 22):
            parent = int(self.body.parents[joint])
            offset = self._rest_body_offsets[joint].expand(count, -1)
            positions.append(positions[parent] + torch.einsum("bij,bj->bi", rotations[:, parent], offset))
        return torch.stack(positions, dim=1)

    def _lock_static_feet(self, body_pose: np.ndarray) -> np.ndarray:
        """Use leg-chain IK to keep both ankles at the neutral floor anchors."""
        base_pose = torch.as_tensor(body_pose, dtype=self.dtype, device=self.device).reshape(1, 63).detach()
        with torch.no_grad():
            neutral = self._decode(torch.as_tensor(
                self.body_latents["neutral"], dtype=self.dtype, device=self.device,
            )[None]) + torch.as_tensor(
                self.body_corrections["neutral"], dtype=self.dtype, device=self.device,
            )[None]
            target = self._skeleton_positions(neutral)[0, torch.as_tensor((7, 8), device=self.device)][:, (0, 2)]
        # SMPL-X body-pose slices for left/right hip and left/right knee.
        begins = (0, 3, 9, 12)
        residual = torch.nn.Parameter(torch.zeros((4, 3), dtype=self.dtype, device=self.device))
        optimizer = torch.optim.Adam((residual,), lr=.065)
        best_loss = float("inf")
        best = residual.detach().clone()
        for _ in range(72):
            optimizer.zero_grad(set_to_none=True)
            pose = base_pose.clone()
            for row, begin in enumerate(begins):
                pose[:, begin:begin + 3] += residual[row]
            ankles = self._skeleton_positions(pose)[0, torch.as_tensor((7, 8), device=self.device)][:, (0, 2)]
            anchor_error = (ankles - target).square().mean()
            loss = 6000.0 * anchor_error + .025 * residual.square().mean()
            loss.backward()
            with torch.no_grad():
                value = float(loss.detach())
                if value < best_loss:
                    best_loss = value; best.copy_(residual)
            optimizer.step()
            with torch.no_grad():
                residual.clamp_(-.9, .9)
        result = base_pose.clone()
        for row, begin in enumerate(begins):
            result[:, begin:begin + 3] += best[row]
        return result[0].detach().cpu().numpy().astype(np.float32)

    def _orient_static_feet(self, body_pose: np.ndarray) -> np.ndarray:
        """Match both ankle/forefoot orientations to the neutral planted feet."""
        result = np.asarray(body_pose, dtype=np.float32).copy()
        current_pose = torch.as_tensor(result, dtype=self.dtype, device=self.device).reshape(1, 21, 3)
        with torch.no_grad():
            neutral_pose = self._decode(torch.as_tensor(
                self.body_latents["neutral"], dtype=self.dtype, device=self.device,
            )[None]) + torch.as_tensor(
                self.body_corrections["neutral"], dtype=self.dtype, device=self.device,
            )[None]
            current_rotations = _global_body_rotations(
                torch.zeros((1, 3), dtype=self.dtype, device=self.device),
                current_pose, self.body.parents.to(self.device),
            )[0]
            neutral_rotations = _global_body_rotations(
                torch.zeros((1, 3), dtype=self.dtype, device=self.device),
                neutral_pose.reshape(1, 21, 3), self.body.parents.to(self.device),
            )[0]
        for ankle, forefoot in ((7, 10), (8, 11)):
            parent = int(self.body.parents[ankle])
            desired_ankle_local = current_rotations[parent].transpose(0, 1) @ neutral_rotations[ankle]
            ankle_begin = 3 * (ankle - 1)
            result[ankle_begin:ankle_begin + 3] = _matrix_to_rotvec(desired_ankle_local.cpu().numpy())
            # The forefoot local rotation is relative to the ankle. Since the
            # ankle global orientation now equals neutral, copying this local
            # value fixes the whole distal foot transform rather than only its
            # ankle/toe landmark positions.
            forefoot_begin = 3 * (forefoot - 1)
            result[forefoot_begin:forefoot_begin + 3] = neutral_pose[0, forefoot_begin:forefoot_begin + 3].cpu().numpy()
        return result

    def _hand_output(self, side: str, body_pose: torch.Tensor, hand_pose: torch.Tensor) -> torch.Tensor:
        hand_pose = hand_pose.reshape(1, 45)
        if side == "left":
            return self._body_output(body_pose, hand_pose, self.open_hand["right"][None]).joints[0]
        return self._body_output(body_pose, self.open_hand["left"][None], hand_pose).joints[0]

    def _solve_hand(
        self,
        side: str,
        name: str,
        objective: str,
        finger: str | None = None,
        *,
        iterations: int = 460,
    ) -> np.ndarray:
        open_pose = torch.as_tensor(
            self.hand_poses[side].get("together", self.open_hand[side]),
            dtype=self.dtype, device=self.device,
        )
        # Optimise only the participating articulated MANO chains.  This is
        # still landmark IK (no guessed flexion axis), while inactive fingers
        # remain bit-for-bit in the canonical open pose.
        mano_group = {"index": 0, "middle": 1, "pinky": 2, "ring": 3, "thumb": 4}
        active = tuple(mano_group) if objective == "fist" else ((finger, "thumb") if objective == "pinch" else (finger,))
        active = tuple(value for value in active if value is not None)
        delta = torch.nn.Parameter(torch.zeros((len(active), 9), dtype=self.dtype, device=self.device))
        body_pose = self._decode(torch.as_tensor(self.body_latents["forward"], device=self.device)[None]).detach()
        with torch.no_grad():
            open_joints = self._hand_output(side, body_pose, open_pose)
        chains = _HAND_JOINTS[side]
        direction = torch.nn.functional.normalize(
            open_joints[chains["middle"][-1]] - open_joints[chains["middle"][0]], dim=0,
        )
        across = torch.nn.functional.normalize(
            open_joints[chains["index"][0]] - open_joints[chains["pinky"][0]], dim=0,
        )
        palm_normal = (1.0 if side == "left" else -1.0) * torch.nn.functional.normalize(
            torch.cross(direction, across, dim=0), dim=0,
        )

        def curl_targets(target_finger: str, angles=(48., 118., 158.)) -> torch.Tensor:
            chain_ids = chains[target_finger]
            points = [open_joints[chain_ids[0]]]
            for segment, angle in enumerate(angles):
                length = torch.linalg.vector_norm(open_joints[chain_ids[segment + 1]] - open_joints[chain_ids[segment]])
                radians = np.deg2rad(angle)
                segment_direction = np.cos(radians) * direction + np.sin(radians) * palm_normal
                points.append(points[-1] + length * segment_direction)
            return torch.stack(points[1:])

        full_curl = {target_finger: curl_targets(target_finger) for target_finger in _FINGERS}
        partial_curl = {target_finger: curl_targets(target_finger, (18., 42., 70.)) for target_finger in _FINGERS}
        optimizer = torch.optim.Adam((delta,), lr=.04)
        best_loss = float("inf")
        best = delta.detach().clone()
        for _ in range(iterations):
            optimizer.zero_grad(set_to_none=True)
            hand_pose = open_pose.clone()
            for row, active_finger in enumerate(active):
                start = 9 * mano_group[active_finger]
                hand_pose[start:start + 9] = open_pose[start:start + 9] + delta[row]
            joints = self._hand_output(side, body_pose, hand_pose)
            prior = delta.square().mean()
            smooth_chain = torch.zeros((), dtype=self.dtype, device=self.device)
            if delta.numel():
                chain = delta.reshape(len(active), 3, 3)
                smooth_chain = (chain[:, 1:] - chain[:, :-1]).square().mean()
            if objective == "fist":
                finger_loss = torch.stack([
                    joints[torch.as_tensor(chains[target_finger][1:], device=self.device)] - full_curl[target_finger]
                    for target_finger in _FINGERS
                ]).square().mean()
                thumb_goal = .5 * (joints[chains["index"][1]] + joints[chains["middle"][1]])
                thumb_loss = (joints[chains["thumb"][-1]] - thumb_goal).square().mean()
                loss = 2100.0 * finger_loss + 1100.0 * thumb_loss + .035 * prior + .012 * smooth_chain
            elif objective == "pinch":
                assert finger is not None
                thumb_tip = joints[chains["thumb"][-1]]
                finger_tip = joints[chains[finger][-1]]
                contact = .42 * open_joints[chains["thumb"][-1]] + .58 * open_joints[chains[finger][-1]]
                contact_loss = (thumb_tip - finger_tip).square().mean()
                location_loss = (thumb_tip - contact).square().mean() + (finger_tip - contact).square().mean()
                curve_loss = (joints[torch.as_tensor(chains[finger][1:3], device=self.device)] - partial_curl[finger][:2]).square().mean()
                loss = 1700.0 * contact_loss + 420.0 * location_loss + 260.0 * curve_loss + .055 * prior + .015 * smooth_chain
            elif objective == "flex":
                assert finger is not None
                chain_loss = (
                    joints[torch.as_tensor(chains[finger][1:], device=self.device)] - full_curl[finger]
                ).square().mean()
                loss = 2100.0 * chain_loss + .06 * prior + .018 * smooth_chain
            else:
                raise ValueError(objective)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                delta.clamp_(-1.65, 1.65)
                value = float(loss.detach())
                if value < best_loss:
                    best_loss = value; best.copy_(delta)
        pose_tensor = open_pose.clone()
        for row, active_finger in enumerate(active):
            start = 9 * mano_group[active_finger]
            pose_tensor[start:start + 9] = open_pose[start:start + 9] + best[row]
        pose = pose_tensor.detach().cpu().numpy().astype(np.float32)
        if self.verbose:
            print(f"MANO IK  {side:>5s}/{name:<12s}: |delta|={float(best.norm()):.2f}")
        return pose

    def _solve_hand_alignment(self, side: str, *, spread: bool) -> np.ndarray:
        """Solve straight, parallel fingers or a controlled planar spread."""
        open_pose = self.open_hand[side]
        groups = {"index": 0, "middle": 1, "ring": 3, "pinky": 2}
        fingers = tuple(groups)
        delta = torch.nn.Parameter(torch.zeros((4, 9), dtype=self.dtype, device=self.device))
        body_pose = self._decode(torch.as_tensor(self.body_latents["forward"], device=self.device)[None]).detach()
        chains = _HAND_JOINTS[side]
        with torch.no_grad():
            open_joints = self._hand_output(side, body_pose, open_pose)
        direction = torch.nn.functional.normalize(
            open_joints[chains["middle"][-1]] - open_joints[chains["middle"][0]], dim=0,
        )
        across = torch.nn.functional.normalize(
            open_joints[chains["index"][0]] - open_joints[chains["pinky"][0]], dim=0,
        )
        spread_amount = {"index": .020, "middle": .006, "ring": -.006, "pinky": -.020}
        targets: dict[str, torch.Tensor] = {}
        for finger in fingers:
            chain = chains[finger]
            lengths = torch.stack([
                torch.linalg.vector_norm(open_joints[chain[index + 1]] - open_joints[chain[index]])
                for index in range(3)
            ])
            cumulative = torch.cumsum(lengths, dim=0)
            ratio = cumulative / cumulative[-1]
            lateral = spread_amount[finger] if spread else 0.0
            targets[finger] = open_joints[chain[0]][None] + cumulative[:, None] * direction[None] + ratio[:, None] * lateral * across[None]
        optimizer = torch.optim.Adam((delta,), lr=.038)
        best_loss = float("inf")
        best = delta.detach().clone()
        for _ in range(520):
            optimizer.zero_grad(set_to_none=True)
            hand_pose = open_pose.clone()
            for row, finger in enumerate(fingers):
                start = 9 * groups[finger]
                hand_pose[start:start + 9] = open_pose[start:start + 9] + delta[row]
            joints = self._hand_output(side, body_pose, hand_pose)
            landmark_loss = torch.stack([
                joints[torch.as_tensor(chains[finger][1:], device=self.device)] - targets[finger]
                for finger in fingers
            ]).square().mean()
            loss = 2300.0 * landmark_loss + .10 * delta.square().mean()
            loss.backward(); optimizer.step()
            with torch.no_grad():
                delta.clamp_(-.85, .85)
                value = float(loss.detach())
                if value < best_loss:
                    best_loss = value; best.copy_(delta)
        result = open_pose.clone()
        for row, finger in enumerate(fingers):
            start = 9 * groups[finger]
            result[start:start + 9] = open_pose[start:start + 9] + best[row]
        return result.detach().cpu().numpy().astype(np.float32)

    def _build_hand_library(self) -> None:
        for side in ("left", "right"):
            self.hand_poses[side]["open"] = self.open_hand[side].detach().cpu().numpy().astype(np.float32)
            self.hand_poses[side]["relaxed"] = np.zeros(45, dtype=np.float32)
            self.hand_poses[side]["together"] = self._solve_hand_alignment(side, spread=False)
            self.hand_poses[side]["spread"] = self._solve_hand_alignment(side, spread=True)
            self.hand_poses[side]["fist"] = self._solve_hand(side, "fist", "fist")
            for finger in _FINGERS:
                self.hand_poses[side][f"pinch_{finger}"] = self._solve_hand(side, f"pinch_{finger}", "pinch", finger)
                self.hand_poses[side][f"flex_{finger}"] = self._solve_hand(side, f"flex_{finger}", "flex", finger)

    def _latent(self, key: str) -> np.ndarray:
        return self.body_latents[key]

    def _blend_body(self, first: str, second: str, amount: float) -> np.ndarray:
        amount = _smooth(amount)
        return (1.0 - amount) * self._latent(first) + amount * self._latent(second)

    def _blend_correction(self, first: str, second: str, amount: float) -> np.ndarray:
        amount = _smooth(amount)
        return (1.0 - amount) * self.body_corrections[first] + amount * self.body_corrections[second]

    def _body_path(self, phase: float, keys: tuple[str, ...]) -> np.ndarray:
        position = np.clip(phase, 0., 1.) * (len(keys) - 1)
        index = min(len(keys) - 2, int(position))
        return self._blend_body(keys[index], keys[index + 1], position - index)

    def _correction_path(self, phase: float, keys: tuple[str, ...]) -> np.ndarray:
        position = np.clip(phase, 0., 1.) * (len(keys) - 1)
        index = min(len(keys) - 2, int(position))
        return self._blend_correction(keys[index], keys[index + 1], position - index)

    def _hand_path(self, side: str, phase: float, keys: tuple[str, ...]) -> np.ndarray:
        position = np.clip(phase, 0., 1.) * (len(keys) - 1)
        index = min(len(keys) - 2, int(position))
        amount = _smooth(position - index)
        first, second = self.hand_poses[side][keys[index]], self.hand_poses[side][keys[index + 1]]
        return ((1.0 - amount) * first + amount * second).astype(np.float32)

    def frame(self, motion: str, phase: float) -> CalibrationFrame:
        neutral = "neutral"
        body_latent = self._latent(neutral).copy()
        body_correction = self.body_corrections[neutral].copy()
        left = self.hand_poses["left"]["relaxed"].copy()
        right = self.hand_poses["right"]["relaxed"].copy()
        if motion == "head":
            # Protocol semantics are anatomical, independent of whether a
            # particular viewer camera observes the subject from the front.
            head_keys = (neutral,"head_left",neutral,"head_right",neutral,"head_down",neutral,"head_up",neutral)
            body_latent = self._body_path(phase, head_keys)
            body_correction = self._correction_path(phase, head_keys)
        elif motion in ("a_pose", "t_pose", "forward"):
            amount = _hold(phase)
            body_latent = self._blend_body(neutral, motion, amount)
            body_correction = self._blend_correction(neutral, motion, amount)
            left = self.hand_poses["left"]["open"].copy(); right = self.hand_poses["right"]["open"].copy()
        elif motion == "elbows":
            # Alternate both sides twice, as instructed by the protocol.
            elbow_keys = (
                "t_pose", "elbow_right", "t_pose", "elbow_left", "t_pose",
                "elbow_right", "t_pose", "elbow_left", "t_pose",
            )
            body_latent = self._latent("t_pose").copy()
            body_correction = self._correction_path(phase, elbow_keys)
            left = self.hand_poses["left"]["open"].copy(); right = self.hand_poses["right"]["open"].copy()
        elif motion == "wrists":
            wrist_keys = ("forward","wrist_pronate","wrist_supinate","wrist_flex","wrist_extend","forward")
            body_latent = self._latent("forward").copy()
            body_correction = self._correction_path(phase, wrist_keys)
            left = self.hand_poses["left"]["open"].copy(); right = self.hand_poses["right"]["open"].copy()
        elif motion == "squat":
            amount = _pulse(phase, 2)
            body_latent = self._blend_body(neutral, "squat", amount)
            body_correction = self._blend_correction(neutral, "squat", amount)
        elif motion in ("left_knee", "right_knee"):
            # Keep anatomical SMPL-X side semantics. Camera/view mirroring is
            # a visualization concern and must never swap calibration joints.
            key = "left_leg" if motion == "left_knee" else "right_leg"
            amount = _pulse(phase, 2)
            body_latent = self._blend_body(neutral, key, amount)
            body_correction = self._blend_correction(neutral, key, amount)
        elif motion == "ankles":
            ankle_keys = (neutral,"heel_raise",neutral,"toe_raise",neutral)
            body_latent = self._body_path(phase, ankle_keys)
            body_correction = self._correction_path(phase, ankle_keys)
        elif motion in ("hand_open", "fist", "pinch", "fingers"):
            body_latent = self._latent("forward").copy()
            body_correction = self.body_corrections["forward"].copy()
            if motion == "hand_open":
                hand_keys = ("spread", "together", "spread", "together", "spread")
            elif motion == "fist":
                hand_keys = ("together", "fist", "together", "fist", "together")
            else:
                prefix = "pinch" if motion == "pinch" else "flex"
                hand_keys_list = ["together"]
                for finger in _FINGERS:
                    hand_keys_list.extend((f"{prefix}_{finger}", "together"))
                hand_keys = tuple(hand_keys_list)
            left = self._hand_path("left", phase, hand_keys)
            right = self._hand_path("right", phase, hand_keys)
        elif motion == "turns":
            turn_keys_list = [neutral]
            for step in range(16):
                turn_keys_list.extend(("left_leg" if step % 2 == 0 else "right_leg", neutral))
            turn_keys = tuple(turn_keys_list)
            lifted_latent = self._body_path(phase, turn_keys)
            lifted_correction = self._correction_path(phase, turn_keys)
            body_latent = .58 * self._latent(neutral) + .42 * lifted_latent
            body_correction = .58 * self.body_corrections[neutral] + .42 * lifted_correction
        elif motion == "walk":
            gait_keys = (
                neutral, "walk_left_swing", "walk_stride", "walk_right_swing", neutral,
                "walk_right_swing", "walk_stride", "walk_left_swing", neutral,
            )
            body_latent = self._body_path(phase, gait_keys)
            body_correction = self._correction_path(phase, gait_keys)
        latent_tensor = torch.as_tensor(body_latent, dtype=self.dtype, device=self.device)[None]
        with torch.no_grad():
            body_pose = self._decode(latent_tensor)[0].cpu().numpy().astype(np.float32) + body_correction
        static_feet = motion not in ("left_knee", "right_knee", "ankles", "turns", "walk")
        if static_feet:
            body_pose = self._lock_static_feet(body_pose)
            body_pose = self._orient_static_feet(body_pose)
            # The cached neutral foot transforms carry the sole pitch/roll.
            # Apply the requested natural toe-out afterwards as a pure world-Y
            # rotation, which keeps the planted sole and ankle anchor intact.
            body_pose = self._straighten_feet(body_pose)
        else:
            body_pose = self._straighten_feet(body_pose)
        if motion in ("turns", "walk"):
            body_pose = self._stabilize_head(body_pose)
        return CalibrationFrame(body_pose, left.astype(np.float32), right.astype(np.float32))

    def diagnostics(self) -> dict[str, dict[str, float]]:
        """Return landmark checks used to reject mirrored/crossed key poses."""
        result: dict[str, dict[str, float]] = {}
        with torch.no_grad():
            for key, latent in self.body_latents.items():
                joints = self._joints(torch.as_tensor(latent, device=self.device)[None])[0].cpu().numpy()
                result[key] = {
                    "left_knee_x": float(joints[4, 0]), "right_knee_x": float(joints[5, 0]),
                    "left_ankle_x": float(joints[7, 0]), "right_ankle_x": float(joints[8, 0]),
                    "left_foot_forward": float(joints[10, 2] - joints[7, 2]),
                    "right_foot_forward": float(joints[11, 2] - joints[8, 2]),
                    "left_knee_height": float(joints[4, 1]), "right_knee_height": float(joints[5, 1]),
                }
        return result
