"""Automatic alignment of the recorded 01--17 calibration protocol.

The instructional HTML is not itself machine-readable input.  This module
turns the same ordered protocol into per-recording time segments using the
tracker and MANUS signals, so calibration, fitting and visualization share one
set of labels even when the operator pauses between actions.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import itertools
from pathlib import Path
from typing import Sequence

import numpy as np

from .manus_hands import read_manus_npz, wrist_local_points
from .types import TrackerFrame


@dataclass(frozen=True)
class ProtocolDefinition:
    index: int
    key: str
    title: str
    motion: str
    body_calibration: bool


PROTOCOL = (
    ProtocolDefinition(1, "neutral", "01 自然站立", "neutral", True),
    ProtocolDefinition(2, "head", "02 头部标定", "head", True),
    ProtocolDefinition(3, "a_pose", "03 A 字站姿", "a_pose", True),
    ProtocolDefinition(4, "t_pose", "04 T 字站姿", "t_pose", True),
    ProtocolDefinition(5, "forward", "05 前平举", "forward", True),
    ProtocolDefinition(6, "elbows", "06 屈肘标定", "elbows", True),
    ProtocolDefinition(7, "wrists", "07 手腕标定", "wrists", True),
    ProtocolDefinition(8, "squat", "08 浅蹲", "squat", True),
    ProtocolDefinition(9, "left_knee", "09 左腿", "left_knee", True),
    ProtocolDefinition(10, "right_knee", "10 右腿", "right_knee", True),
    ProtocolDefinition(11, "ankles", "11 踝关节", "ankles", True),
    ProtocolDefinition(12, "hand_open", "12 张手与并指", "hand_open", False),
    ProtocolDefinition(13, "fist", "13 握拳", "fist", False),
    ProtocolDefinition(14, "pinch", "14 捏合与对掌", "pinch", False),
    ProtocolDefinition(15, "fingers", "15 单指屈伸", "fingers", False),
    # These final two stages validate dynamic root motion and temporal IK.
    # They are deliberately excluded from the static scale/shape equations:
    # each frame has a different pelvis transform, so treating them as fixed
    # calibration poses would bias the subject shape and tracker mounts.
    ProtocolDefinition(16, "turns", "16 左右转体 180°", "turns", False),
    ProtocolDefinition(17, "walk", "17 前后各走两步", "walk", False),
)
PROTOCOL_BY_KEY = {item.key: item for item in PROTOCOL}


@dataclass(frozen=True)
class ProtocolSegment:
    index: int
    key: str
    title: str
    motion: str
    start_s: float
    end_s: float
    active_start_s: float
    active_end_s: float
    confidence: str

    def phase(self, timestamp: float, *, active: bool = True) -> float:
        begin = self.active_start_s if active else self.start_s
        end = self.active_end_s if active else self.end_s
        return float(np.clip((timestamp - begin) / max(end - begin, 1e-6), 0.0, 1.0))


@dataclass(frozen=True)
class ProtocolAlignment:
    duration_s: float
    segments: tuple[ProtocolSegment, ...]
    method: str = "ordered tracker/MANUS feature alignment"

    def segment_at(self, timestamp: float) -> ProtocolSegment:
        for segment in self.segments:
            if timestamp < segment.end_s or segment is self.segments[-1]:
                return segment
        return self.segments[-1]

    def labels(self, timestamps: np.ndarray) -> np.ndarray:
        ends = np.asarray([item.end_s for item in self.segments], dtype=np.float64)
        return np.searchsorted(ends, np.asarray(timestamps), side="right").clip(0, len(self.segments) - 1).astype(np.int16)

    def to_mapping(self) -> dict:
        return {"duration_s": self.duration_s, "method": self.method, "segments": [asdict(item) for item in self.segments]}


def _smooth(values: np.ndarray, count: int) -> np.ndarray:
    count = max(1, int(count))
    if count <= 1:
        return values.copy()
    return np.convolve(values, np.ones(count, dtype=np.float64) / count, mode="same")


def _runs(mask: np.ndarray, timestamps: np.ndarray, *, minimum_s: float = .25, merge_s: float = 0.0) -> list[tuple[float, float]]:
    indices = np.flatnonzero(mask)
    if not len(indices):
        return []
    raw = np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1)
    merged: list[list[float]] = []
    for run in raw:
        begin, end = float(timestamps[run[0]]), float(timestamps[run[-1]])
        if merged and begin - merged[-1][1] <= merge_s:
            merged[-1][1] = end
        else:
            merged.append([begin, end])
    return [(begin, end) for begin, end in merged if end - begin >= minimum_s]


def _next_leg_raise(
    lift: np.ndarray,
    timestamps: np.ndarray,
    begin_s: float,
    *,
    threshold_m: float,
    search_duration_s: float = 18.0,
) -> list[tuple[float, float]]:
    """Find the next sustained foot lift without matching later foot motion.

    The ankle trackers do not necessarily have equal neutral heights.  In the
    2026-08-29 recordings the right-foot lift is a repeatable 4--5 cm while the
    left-foot lift exceeds 5.5 cm.  A bounded ordered search lets the lower
    threshold capture that deliberate action without confusing later MANUS or
    end-of-capture foot motion for protocol stage 10.
    """
    end_s = begin_s + search_duration_s
    search = (timestamps > begin_s) & (timestamps < end_s)
    return _runs(search & (lift > threshold_m), timestamps, minimum_s=.45, merge_s=2.4)


def _rotation_angle(rotation: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    relative = np.einsum("ji,tjk->tik", baseline, rotation)
    return np.arccos(np.clip((np.trace(relative, axis1=1, axis2=2) - 1.0) * .5, -1.0, 1.0))


def _rotation_speed(rotation: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
    relative = np.einsum("tji,tjk->tik", rotation[:-1], rotation[1:])
    angle = np.arccos(np.clip((np.trace(relative, axis1=1, axis2=2) - 1.0) * .5, -1.0, 1.0))
    return np.r_[0.0, angle / np.maximum(np.diff(timestamps), 1e-5)]


def _signal_peaks(values: np.ndarray, timestamps: np.ndarray, begin: float, end: float) -> list[tuple[float, float]]:
    """Return separated local maxima as ``(time, prominence)`` pairs."""
    cadence = float(np.median(np.diff(timestamps)))
    smoothed = _smooth(np.asarray(values, dtype=np.float64), max(3, round(.28 / cadence)))
    indices = np.flatnonzero(
        (timestamps >= begin) & (timestamps <= end)
        & (smoothed >= np.r_[smoothed[0], smoothed[:-1]])
        & (smoothed >= np.r_[smoothed[1:], smoothed[-1]])
    )
    candidates = []
    radius = max(2, round(1.0 / cadence))
    for index in indices:
        left = smoothed[max(0, index - radius):index + 1]
        right = smoothed[index:min(len(smoothed), index + radius + 1)]
        prominence = float(smoothed[index] - max(float(left.min()), float(right.min())))
        if prominence >= .025:
            candidates.append((int(index), prominence))
    retained = []
    for index, prominence in sorted(candidates, key=lambda item: item[1], reverse=True):
        if all(abs(timestamps[index] - timestamps[other]) >= 1.35 for other, _ in retained):
            retained.append((index, prominence))
    return sorted([(float(timestamps[index]), prominence) for index, prominence in retained[:8]])


def _ordered_elbow_peaks(
    left_peaks: Sequence[tuple[float, float]],
    right_peaks: Sequence[tuple[float, float]],
) -> tuple[float, float, float, float] | None:
    """Find the earliest complete right-left-right-left elbow sequence."""
    candidates = []
    for first, second, third, fourth in itertools.product(
        right_peaks, left_peaks, right_peaks, left_peaks,
    ):
        times4 = (first[0], second[0], third[0], fourth[0])
        gaps = np.diff(times4)
        if np.all(gaps >= 1.0) and np.all(gaps <= 6.5):
            prominence = first[1] + second[1] + third[1] + fourth[1]
            candidates.append((times4, prominence))
    if not candidates:
        return None
    # Wrist articulation later in the same interval can also form four large
    # peaks. Protocol order makes completion time the primary discriminator;
    # prominence only breaks ties among the early elbow candidates.
    return min(candidates, key=lambda item: (item[0][3], -item[1]))[0]


def _four_phase_window(
    peaks: Sequence[tuple[float, float]],
) -> tuple[float, float] | None:
    """Fit a window whose .2/.4/.6/.8 phases land on four strong peaks."""
    if len(peaks) < 4:
        return None
    selected = sorted(sorted(peaks, key=lambda item: item[1], reverse=True)[:4])
    times4 = np.asarray([item[0] for item in selected], dtype=np.float64)
    gaps = np.diff(times4)
    if np.any(gaps < 1.0) or np.any(gaps > 4.5):
        return None
    duration = float((times4[-1] - times4[0]) / .6)
    return float(times4[0] - .2 * duration), float(times4[-1] + .2 * duration)


def _static_holds(position: np.ndarray, timestamps: np.ndarray) -> dict[str, tuple[float, float]]:
    head, left, right = position[:, 0], position[:, 1], position[:, 2]
    separation = np.linalg.norm(left - right, axis=1)
    below = .5 * (left[:, 1] + right[:, 1]) - head[:, 1]
    speed = np.zeros(len(timestamps))
    speed[1:] = np.max(np.linalg.norm(np.diff(position, axis=0), axis=2) / np.maximum(np.diff(timestamps), 1e-5)[:, None], axis=1)
    stable = speed < .12
    candidates = {
        "a_pose": stable & (timestamps >= 15) & (timestamps <= 70) & (separation >= .80) & (separation <= 1.16) & (below >= -.62) & (below <= -.30),
        "t_pose": stable & (timestamps >= 20) & (timestamps <= 90) & (separation >= 1.05) & (below >= -.36),
        "forward": stable & (timestamps >= 30) & (timestamps <= 110) & (separation <= .36) & (below >= -.38),
    }
    result = {"neutral": (1.5, min(5.5, float(timestamps[-1])))}
    previous_end = 8.0
    for key in ("a_pose", "t_pose", "forward"):
        runs = _runs(candidates[key], timestamps, minimum_s=1.4, merge_s=.35)
        runs = [run for run in runs if run[0] > previous_end]
        if not runs:
            raise ValueError(f"cannot locate {key} hold in calibration recording")
        # The later wrist/hand stages can contain a much longer forward-arm
        # hold than 05.  Protocol order makes the first qualifying forward
        # hold the unambiguous choice once A- and T-pose have been located.
        run = runs[0] if key == "forward" else max(runs, key=lambda item: item[1] - item[0])
        begin, end = run[0] + .45, run[1] - .45
        if end <= begin:
            begin, end = run
        result[key] = (begin, end)
        previous_end = end
    return result


def _manus_energy(path: Path, target_time: np.ndarray) -> np.ndarray | None:
    if not path.exists():
        return None
    streams = read_manus_npz(path)
    energies = []
    origin = min(int(stream.timestamps_us[0]) for stream in streams.values())
    for side in ("left", "right"):
        points = wrist_local_points(streams[side])
        time = (streams[side].timestamps_us - origin) / 1e6
        speed = np.r_[0.0, np.max(np.linalg.norm(np.diff(points, axis=0), axis=2), axis=1) / np.maximum(np.diff(time), 1e-5)]
        energies.append(np.interp(target_time, time, speed, left=0.0, right=0.0))
    return _smooth(np.maximum(energies[0], energies[1]), max(3, round(.45 / np.median(np.diff(target_time)))))


def align_protocol(
    frames: Sequence[TrackerFrame], *, glove_path: str | Path | None = None,
    static_holds: dict[str, Sequence[float]] | None = None,
) -> ProtocolAlignment:
    """Locate all ordered protocol stages in one synchronized recording."""
    if len(frames) < 20:
        raise ValueError("protocol alignment needs at least 20 tracker frames")
    names = ("head", "left_hand", "right_hand", "left_foot", "right_foot")
    timestamps = np.asarray([frame.timestamp for frame in frames], dtype=np.float64)
    position = np.stack([[frame[name].position for name in names] for frame in frames]).astype(np.float64)
    rotation = np.stack([[frame[name].rotation for name in names] for frame in frames]).astype(np.float64)
    holds = {key: tuple(map(float, value)) for key, value in (static_holds or _static_holds(position, timestamps)).items()}
    neutral = (timestamps >= holds["neutral"][0]) & (timestamps <= holds["neutral"][1])
    baselines = []
    for index in range(5):
        average = rotation[neutral, index].mean(axis=0)
        u, _, vt = np.linalg.svd(average); value = u @ vt
        if np.linalg.det(value) < 0:
            u[:, -1] *= -1; value = u @ vt
        baselines.append(value)
    baselines = np.stack(baselines)

    # 02: four deliberate head excursions before the A-pose.
    head_angle = _rotation_angle(rotation[:, 0], baselines[0])
    head_runs = _runs((head_angle > np.deg2rad(18)) & (timestamps > 6) & (timestamps < holds["a_pose"][0]), timestamps, minimum_s=.45, merge_s=1.4)
    head_active = (head_runs[0][0], head_runs[-1][1]) if head_runs else (7.0, max(8.0, holds["a_pose"][0] - 2.0))

    base_position = np.median(position[neutral], axis=0)
    left_lift = position[:, 3, 1] - base_position[3, 1]
    right_lift = position[:, 4, 1] - base_position[4, 1]
    head_height = position[:, 0, 1] - .5 * (position[:, 3, 1] + position[:, 4, 1])
    base_head_height = float(np.median(head_height[neutral]))
    # Some valid recordings perform the same ordered protocol more quickly;
    # anchor the search to stage 05 instead of an absolute wall-clock time.
    squat_search = timestamps > holds["forward"][1] + 18.0
    squat_runs = _runs(
        squat_search & (head_height < base_head_height - .065) & (left_lift < .04) & (right_lift < .04),
        timestamps, minimum_s=.55, merge_s=2.8,
    )
    if not squat_runs:
        raise ValueError("cannot locate 08 squat pulses")
    squat_active = squat_runs[0]
    left_runs = _next_leg_raise(
        left_lift, timestamps, squat_active[1] + 1.5, threshold_m=.055,
    )
    if not left_runs:
        raise ValueError("cannot locate 09 left leg raises")
    left_active = left_runs[0]
    right_runs = _next_leg_raise(
        right_lift, timestamps, left_active[1] + 1.0, threshold_m=.035,
    )
    if not right_runs:
        raise ValueError("cannot locate 10 right leg raises")
    right_active = right_runs[0]

    # 06/07 are the two hand-motion blocks between forward hold and squat.
    hand_relative = position[:, 1:3] - position[:, :1]
    hand_speed = np.r_[0.0, np.max(np.linalg.norm(np.diff(hand_relative, axis=0), axis=2) / np.maximum(np.diff(timestamps), 1e-5)[:, None], axis=1)]
    hand_rotation_speed = np.maximum(_rotation_speed(rotation[:, 1], timestamps), _rotation_speed(rotation[:, 2], timestamps))
    cadence = np.median(np.diff(timestamps))
    activity = _smooth(hand_speed + .08 * hand_rotation_speed, max(3, round(.45 / cadence)))
    elbow_search_begin = holds["forward"][1] + .6
    elbow_search_end = squat_active[0] - 2.0
    between = (timestamps > elbow_search_begin) & (timestamps < elbow_search_end)
    candidates = _runs(between & (activity > .10), timestamps, minimum_s=.4, merge_s=1.5)
    if len(candidates) < 2:
        split = .5 * (holds["forward"][1] + squat_active[0])
        elbow_active, wrist_active = (holds["forward"][1] + 1.0, split - 1.0), (split + 1.0, squat_active[0] - 1.0)
    else:
        # Split at the largest inter-run pause; repeated movements inside one
        # stage have short pauses, while 06→07 has an operator instruction gap.
        gap_index = int(np.argmax([candidates[i + 1][0] - candidates[i][1] for i in range(len(candidates) - 1)]))
        elbow_active = (candidates[0][0], candidates[gap_index][1])
        wrist_active = (candidates[gap_index + 1][0], candidates[-1][1])
    # The protocol performs right-left-right-left elbow peaks. Refine the
    # coarse activity envelope so reference phases .125/.375/.625/.875 align
    # with the four observed wrist-height maxima rather than the preceding
    # forward→T transition.
    left_peaks = _signal_peaks(position[:, 1, 1], timestamps, elbow_search_begin, elbow_search_end)
    right_peaks = _signal_peaks(position[:, 2, 1], timestamps, elbow_search_begin, elbow_search_end)
    elbow_peaks = _ordered_elbow_peaks(left_peaks, right_peaks)
    if elbow_peaks is not None:
        action_duration = (elbow_peaks[3] - elbow_peaks[0]) / .75
        elbow_active = (elbow_peaks[0] - .125 * action_duration, elbow_peaks[3] + .125 * action_duration)

    # Wrist articulation is performed with both arms held forward.  Raw hand
    # activity alone also includes raising the arms into that pose and dropping
    # them afterwards; treating those transitions as wrist calibration kept the
    # fitted wrists half a metre in front of the measured trackers.  Intersect
    # the coarse activity block with the directly observed forward-arm geometry
    # and retain its longest continuous hold.
    hand_separation = np.linalg.norm(position[:, 1] - position[:, 2], axis=1)
    hand_below_head = .5 * (position[:, 1, 1] + position[:, 2, 1]) - position[:, 0, 1]
    forward_arms = (hand_separation < .48) & (hand_below_head > -.45)
    forward_runs = _runs(
        forward_arms
        & (timestamps >= wrist_active[0] - 1.0)
        & (timestamps <= wrist_active[1] + 1.0),
        timestamps, minimum_s=1.5, merge_s=.35,
    )
    if forward_runs:
        def overlap(run: tuple[float, float]) -> float:
            return max(0.0, min(run[1], wrist_active[1]) - max(run[0], wrist_active[0]))
        wrist_active = max(forward_runs, key=lambda run: (overlap(run), run[1] - run[0]))

    # Place the four reference phases directly on the observed articulation
    # extrema. This mainly corrects captures where raising the arms into the
    # forward hold was included at the beginning of the coarse activity run.
    forward_mask = (timestamps >= holds["forward"][0]) & (timestamps <= holds["forward"][1])
    forward_rotations = []
    for tracker_index in (1, 2):
        average = rotation[forward_mask, tracker_index].mean(axis=0)
        u, _, vt = np.linalg.svd(average)
        value = u @ vt
        if np.linalg.det(value) < 0:
            u[:, -1] *= -1
            value = u @ vt
        forward_rotations.append(value)
    wrist_angle = np.maximum(
        _rotation_angle(rotation[:, 1], forward_rotations[0]),
        _rotation_angle(rotation[:, 2], forward_rotations[1]),
    )
    wrist_peaks = _signal_peaks(
        wrist_angle, timestamps, wrist_active[0] - 1.0, wrist_active[1] + 1.0,
    )
    refined_wrist = _four_phase_window(wrist_peaks)
    if refined_wrist is not None:
        wrist_active = (
            max(elbow_active[1] + .5, refined_wrist[0]),
            min(squat_active[0] - 1.0, refined_wrist[1]),
        )

    # 11: both ankle straps rotate after the right-leg stage. Merge the heel
    # and toe sub-actions across their short neutral pause.
    foot_activity = _smooth(np.maximum(_rotation_speed(rotation[:, 3], timestamps), _rotation_speed(rotation[:, 4], timestamps)), max(3, round(.35 / cadence)))
    foot_runs = _runs((timestamps > right_active[1] + 1.0) & (foot_activity > .15), timestamps, minimum_s=.18, merge_s=4.0)
    foot_runs = [run for run in foot_runs if run[0] < right_active[1] + 28.0]
    ankle_active = max(foot_runs, key=lambda item: item[1] - item[0]) if foot_runs else (right_active[1] + 3.0, right_active[1] + 12.0)

    # 16 starts with a coherent whole-body turn.  Locate its first large
    # excursion before segmenting MANUS activity: glove-local noise can remain
    # above the hand threshold during turning and walking, and must not extend
    # stages 12--15 across the final whole-body actions.
    relative_angles = np.stack([
        _rotation_angle(rotation[:, tracker], baselines[tracker])
        for tracker in (0, 3, 4)
    ], axis=1)
    coherent_turn = np.median(relative_angles, axis=1)
    turn_search_start = ankle_active[1] + 20.0
    turn_runs = _runs(
        (timestamps > turn_search_start) & (coherent_turn > np.deg2rad(55.0)),
        timestamps, minimum_s=.65, merge_s=4.0,
    )
    provisional_turn = turn_runs[0] if turn_runs else None

    manus = _manus_energy(Path(glove_path), timestamps) if glove_path is not None else None
    hand_active: list[tuple[float, float]] = []
    if manus is not None:
        search = timestamps > ankle_active[1] + 1.0
        if provisional_turn is not None:
            search &= timestamps < provisional_turn[0] - .5
        raw_runs = _runs(search & (manus > .04), timestamps, minimum_s=.45, merge_s=2.5)
        raw_runs = [run for run in raw_runs if run[1] - run[0] >= 1.0]
        if raw_runs:
            begin, end = raw_runs[0][0], raw_runs[-1][1]
            expected = np.asarray((7, 14, 23), dtype=np.float64) / 32.0
            bounds = []
            for ratio in expected:
                guess = begin + ratio * (end - begin)
                mask = (timestamps >= guess - 3.0) & (timestamps <= guess + 3.0)
                local = np.flatnonzero(mask)
                bounds.append(float(timestamps[local[np.argmin(manus[local])]]) if len(local) else guess)
            edges = [begin, *bounds, end]
            hand_active = [(edges[index], edges[index + 1]) for index in range(4)]
    if len(hand_active) != 4:
        fallback_end = provisional_turn[0] - 1.0 if provisional_turn is not None else float(timestamps[-1])
        begin = min(float(fallback_end - 32.0), ankle_active[1] + 2.0)
        end = float(fallback_end)
        ratios = np.asarray((0, 7, 14, 23, 32), dtype=np.float64) / 32.0
        edges = begin + ratios * (end - begin)
        hand_active = [(float(edges[index]), float(edges[index + 1])) for index in range(4)]

    active = [
        holds["neutral"], head_active, holds["a_pose"], holds["t_pose"], holds["forward"],
        elbow_active, wrist_active, squat_active, left_active, right_active, ankle_active, *hand_active,
    ]
    definitions = PROTOCOL[:15]

    # 16: turn left 180 degrees, return to front, turn right 180 degrees and
    # return again.  A fixed tracker mount means the head and both ankle
    # orientations all make the same large world-yaw excursion.  Expand the
    # threshold crossing so the labelled interval also contains departure and
    # final settling.
    post_hands = hand_active[-1][1] + .35
    turn_runs = _runs(
        (timestamps > post_hands) & (coherent_turn > np.deg2rad(55.0)),
        timestamps, minimum_s=.65, merge_s=4.0,
    )
    if turn_runs:
        turn_active = (
            max(post_hands, turn_runs[0][0] - 1.4),
            min(float(timestamps[-1]), turn_runs[-1][1] + 1.4),
        )

        # 17: after returning to the original heading, walk two steps forward
        # and two steps backward to the floor mark.  Use the coherent
        # head/ankle horizontal translation, not hand swing, to find it.
        reference_mask = (timestamps >= holds["neutral"][0]) & (timestamps <= holds["neutral"][1])
        root_proxy = (position[:, 0] + position[:, 3] + position[:, 4]) / 3.0
        reference_root = np.median(root_proxy[reference_mask], axis=0)
        horizontal = root_proxy - reference_root
        horizontal[:, 1] = 0.0
        displacement = np.linalg.norm(horizontal, axis=1)
        root_speed = np.r_[
            0.0,
            np.linalg.norm(np.diff(root_proxy[:, (0, 2)], axis=0), axis=1)
            / np.maximum(np.diff(timestamps), 1e-5),
        ]
        walk_search = timestamps > turn_active[1] + .25
        walk_runs = _runs(
            walk_search & ((displacement > .075) | (root_speed > .16)),
            timestamps, minimum_s=.6, merge_s=1.5,
        )
        if walk_runs:
            walk_active = (
                max(turn_active[1] + .25, walk_runs[0][0] - .9),
                min(float(timestamps[-1]), walk_runs[-1][1] + 1.2),
            )
            # Require enough room for both actions.  This prevents an old
            # 01--15 recording with a brief end-of-capture bump from receiving
            # spurious stages 16 and 17.
            if turn_active[1] - turn_active[0] >= 4.0 and walk_active[1] - walk_active[0] >= 3.0:
                active.extend((turn_active, walk_active))
                definitions = PROTOCOL
    # Labels cover transitions and pauses. Boundaries are the midpoint of the
    # previous active end and the next active start, preserving exact order.
    boundaries = [float(timestamps[0])]
    for first, second in zip(active[:-1], active[1:]):
        boundaries.append(float(np.clip(.5 * (first[1] + second[0]), boundaries[-1] + 1e-3, timestamps[-1])))
    boundaries.append(float(timestamps[-1]))
    segments = []
    for index, definition in enumerate(definitions):
        if index < 11 or index >= 15:
            confidence = "high"
        else:
            confidence = "medium" if manus is not None else "low"
        segments.append(ProtocolSegment(
            definition.index, definition.key, definition.title, definition.motion,
            boundaries[index], boundaries[index + 1], float(active[index][0]), float(active[index][1]), confidence,
        ))
    return ProtocolAlignment(float(timestamps[-1] - timestamps[0]), tuple(segments))
