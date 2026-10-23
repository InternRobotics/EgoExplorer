"""Process-local, behavior-driven protocol refinement for IK experiments.

This module is intentionally not imported by the production alignment path.  It
uses the manually reviewed alignment only for broad search spans, then finds the
actual squat and leg-lift events from tracker trajectories.  The resulting
alignment is useful for isolated A/B runs where a single coarse segment covers
several repetitions.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Sequence

import numpy as np

from .protocol import ProtocolAlignment, ProtocolSegment
from .types import TrackerFrame


_NAMES = ("head", "left_hand", "right_hand", "left_foot", "right_foot")


def _smooth(values: np.ndarray, width: int) -> np.ndarray:
    width = max(1, int(width))
    if width <= 1:
        return np.asarray(values, dtype=np.float64).copy()
    kernel = np.ones(width, dtype=np.float64) / float(width)
    return np.convolve(np.asarray(values, dtype=np.float64), kernel, mode="same")


def _local_maxima(values: np.ndarray, timestamps: np.ndarray, begin: float, end: float,
                  *, min_distance_s: float = .8, prominence: float = .01) -> list[tuple[float, float]]:
    """Find separated peaks without adding a scipy dependency to the experiment."""
    values = np.asarray(values, dtype=np.float64)
    timestamps = np.asarray(timestamps, dtype=np.float64)
    if len(values) < 3:
        return []
    cadence = float(np.median(np.diff(timestamps)))
    width = max(3, round(.22 / max(cadence, 1e-4)))
    smooth = _smooth(values, width)
    indices = np.flatnonzero(
        (timestamps >= begin) & (timestamps <= end)
        & (smooth >= np.r_[smooth[0], smooth[:-1]])
        & (smooth >= np.r_[smooth[1:], smooth[-1]])
    )
    radius = max(2, round(.65 / max(cadence, 1e-4)))
    candidates: list[tuple[int, float]] = []
    for index in indices:
        left = smooth[max(0, index - radius):index + 1]
        right = smooth[index:min(len(smooth), index + radius + 1)]
        prom = float(smooth[index] - max(float(left.min()), float(right.min())))
        if prom >= prominence:
            candidates.append((int(index), prom))
    retained: list[tuple[int, float]] = []
    for index, prom in sorted(candidates, key=lambda item: (float(smooth[item[0]]), item[1]), reverse=True):
        if all(abs(float(timestamps[index] - timestamps[other])) >= min_distance_s for other, _ in retained):
            retained.append((index, prom))
    return sorted([(float(timestamps[index]), float(smooth[index])) for index, _ in retained])


def _select_events(signal: np.ndarray, timestamps: np.ndarray, begin: float, end: float,
                   count: int, *, min_distance_s: float = .9, prominence: float = .015) -> list[float]:
    """Select the strongest chronologically separated behavior events."""
    peaks = _local_maxima(signal, timestamps, begin, end, min_distance_s=min_distance_s, prominence=prominence)
    if len(peaks) < count:
        # A lower threshold is useful when one ankle is partly occluded.  It is
        # still a behavior peak, never a fixed timestamp fallback.
        peaks = _local_maxima(signal, timestamps, begin, end, min_distance_s=min_distance_s * .85,
                              prominence=max(.006, prominence * .45))
    if len(peaks) <= count:
        return [item[0] for item in peaks]
    # Keep the largest peaks while enforcing the temporal separation again.
    ranked = sorted(peaks, key=lambda item: item[1], reverse=True)
    chosen: list[tuple[float, float]] = []
    for time, score in ranked:
        if all(abs(time - other[0]) >= min_distance_s for other in chosen):
            chosen.append((time, score))
        if len(chosen) == count:
            break
    return sorted(item[0] for item in chosen)


def _active_window(center: float, start: float, end: float, half_width: float) -> tuple[float, float]:
    left = max(float(start), float(center - half_width))
    right = min(float(end), float(center + half_width))
    if right <= left:
        left, right = float(start), float(end)
    return left, right


def refine_alignment_by_behavior(
    frames: Sequence[TrackerFrame], base: ProtocolAlignment, *,
    diagnostics: dict | None = None,
) -> ProtocolAlignment:
    """Refine squat and leg-lift phases using tracker height trajectories.

    ``base`` supplies only broad stage boundaries.  Event centers are obtained
    from local extrema in the recorded foot/head signals, so pauses and capture
    speed do not move a pose to a timestamp copied from another session.
    """
    if len(frames) < 20 or len(base.segments) < 11:
        return base
    timestamps = np.asarray([frame.timestamp for frame in frames], dtype=np.float64)
    position = np.stack([[frame[name].position for name in _NAMES] for frame in frames]).astype(np.float64)
    dt = float(np.median(np.diff(timestamps))) if len(timestamps) > 1 else .066
    neutral = next((item for item in base.segments if item.key == "neutral"), base.segments[0])
    neutral_mask = (timestamps >= neutral.active_start_s) & (timestamps <= neutral.active_end_s)
    if int(neutral_mask.sum()) < 3:
        neutral_mask = timestamps <= min(6.0, float(timestamps[-1]))
    foot_baseline = np.median(position[neutral_mask][:, (3, 4), 1], axis=0)
    left_lift = position[:, 3, 1] - foot_baseline[0]
    right_lift = position[:, 4, 1] - foot_baseline[1]

    squat_segment = next((item for item in base.segments if item.key == "squat"), None)
    knee_segments = [item for item in base.segments if item.key in {"left_knee", "right_knee"}]
    if squat_segment is None or len(knee_segments) < 2:
        return base
    # Automatic protocol alignment can extend the squat transition well into
    # the first raise. Keep the reviewed/automatic knee span as a broad hint,
    # but always leave a short post-squat margin in the search so the first
    # behavior peak is not clipped by an overlong squat envelope.
    original_leg_start = min(item.start_s for item in knee_segments)
    search_leg_start = min(original_leg_start, squat_segment.active_end_s + 1.0)
    leg_end = max(item.end_s for item in knee_segments)

    # A torso-height drop has two local maxima for the requested shallow-squat
    # repetition.  The complete segment is searched, including the transition
    # that the old active range accidentally excluded.
    head_height = position[:, 0, 1] - .5 * (position[:, 3, 1] + position[:, 4, 1])
    baseline_head = float(np.median(head_height[neutral_mask]))
    squat_drop = baseline_head - head_height
    squat_centers = _select_events(
        squat_drop, timestamps, squat_segment.start_s, squat_segment.end_s,
        2, min_distance_s=.85, prominence=max(.012, .12 * float(np.ptp(squat_drop))),
    )
    if len(squat_centers) >= 2:
        squat_active = (
            max(squat_segment.start_s, squat_centers[0] - .65),
            min(squat_segment.end_s, squat_centers[-1] + .65),
        )
    else:
        squat_active = (squat_segment.start_s, squat_segment.end_s)

    # The raised-foot envelope gives four events.  We label the chronological
    # events according to the reviewed protocol semantics (right, right, left,
    # left), while retaining the observed per-peak side for diagnostics.
    raised = np.maximum(np.maximum(left_lift, 0.0), np.maximum(right_lift, 0.0))
    lift_centers = _select_events(
        raised, timestamps, search_leg_start, leg_end, 4,
        min_distance_s=.85, prominence=max(.012, .10 * float(np.ptp(raised))),
    )
    if len(lift_centers) < 4:
        # Use the union of each foot's peaks before giving up. This handles a
        # low-amplitude raise on one side without importing any time anchor.
        candidates = _local_maxima(left_lift, timestamps, search_leg_start, leg_end,
                                   min_distance_s=.85, prominence=.008)
        candidates += _local_maxima(right_lift, timestamps, search_leg_start, leg_end,
                                    min_distance_s=.85, prominence=.008)
        lift_centers = _select_events(
            np.maximum(left_lift, right_lift), timestamps, search_leg_start, leg_end, 4,
            min_distance_s=.75, prominence=.005,
        )
    if len(lift_centers) < 4:
        return base
    lift_centers = sorted(lift_centers[:4])
    observed_sides = ["left" if left_lift[np.argmin(abs(timestamps - center))] >= right_lift[np.argmin(abs(timestamps - center))] else "right"
                      for center in lift_centers]
    semantic_sides = ("right", "right", "left", "left")

    # Split the broad leg block at behavior-event midpoints.  Active windows
    # are local to each repetition; the intervals between events remain useful
    # transition frames but no longer borrow one action's pose prior.
    segment_leg_start = min(original_leg_start, lift_centers[0] - 2.0)
    segment_leg_start = max(float(squat_segment.start_s + .1), float(segment_leg_start))
    boundaries = [float(segment_leg_start)]
    boundaries.extend(float(.5 * (first + second)) for first, second in zip(lift_centers[:-1], lift_centers[1:]))
    boundaries.append(float(leg_end))
    leg_segments = []
    for index, (center, side) in enumerate(zip(lift_centers, semantic_sides)):
        begin, end = boundaries[index], boundaries[index + 1]
        active_begin, active_end = _active_window(center, begin, end, .72)
        side_text = "右" if side == "right" else "左"
        leg_segments.append(ProtocolSegment(
            9 + index, f"{side}_knee", f"{9 + index:02d} 抬{side_text}腿第{index % 2 + 1}次",
            f"{side}_knee", begin, end, active_begin, active_end, "behavior-detected",
        ))

    # Keep all non-knee stages and insert four local leg stages. Re-indexing is
    # metadata only; segment ordering (which drives ``segment_at``) is intact.
    first_knee_index = min(base.segments.index(item) for item in knee_segments)
    last_knee_index = max(base.segments.index(item) for item in knee_segments)
    rebuilt = list(base.segments[:first_knee_index]) + leg_segments + list(base.segments[last_knee_index + 1:])
    rebuilt = [replace(item, index=i + 1) for i, item in enumerate(rebuilt)]
    rebuilt = [
        replace(item, end_s=segment_leg_start, active_start_s=squat_active[0], active_end_s=squat_active[1])
        if item.key == "squat" else item for item in rebuilt
    ]
    if diagnostics is not None:
        diagnostics.update({
            "method": "tracker-behavior extrema; reviewed alignment used only as broad search spans",
            "squat_event_centers_s": [float(value) for value in squat_centers],
            "squat_active_s": [float(value) for value in squat_active],
            "leg_search_span_s": [float(search_leg_start), float(leg_end)],
            "leg_segment_span_s": [float(segment_leg_start), float(leg_end)],
            "leg_event_centers_s": [float(value) for value in lift_centers],
            "leg_observed_sides": observed_sides,
            "leg_semantic_sides": list(semantic_sides),
            "leg_segment_boundaries_s": [float(value) for value in boundaries],
        })
    return ProtocolAlignment(base.duration_s, tuple(rebuilt), method="manual broad spans + tracker behavior extrema")
