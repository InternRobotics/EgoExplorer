#!/usr/bin/env python3
"""Build a per-frame reference SMPL-X body pose for a recording, to be used as a low-weight
branch-selection prior by project_sparse_tracker_baseline_l2.py --pose-reference.

Two sources:
  --protocol-alignment <json>   the 01-17 calibration protocol (data0903-style manual alignment json);
                                poses come from the decoded protocol library (protocol_body_poses.json)
                                via the same phase functions as calibration_motion.py.
  --guide <B0x_vposer_smplx.npz> a GT-pilot full-body guide (static: 1 frame, dynamic: phase-indexed);
                                the active window is detected from tracker motion.
Output npz: body_pose_ref (T,63), mask (T,) float, joint_mask (21,) float, timestamps (T,).
"""
import argparse, json, numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation, Slerp

LEG_TRUNK = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)   # hips, spine1-3, knees, ankles, feet (body_pose joint order)

def _smooth(v): v = float(np.clip(v, 0, 1)); return v * v * (3 - 2 * v)
def _pulse(phase, cycles=1): l = (phase * cycles) % 1.0; return _smooth(0.5 - 0.5 * np.cos(2 * np.pi * l))
def _hold(phase):
    if phase < .2: return _smooth(phase / .2)
    if phase <= .8: return 1.0
    return _smooth((1 - phase) / .2)

def slerp_pose(a, b, amount):
    out = np.zeros(63, np.float32)
    for j in range(21):
        ra, rb = Rotation.from_rotvec(a[3*j:3*j+3]), Rotation.from_rotvec(b[3*j:3*j+3])
        out[3*j:3*j+3] = Slerp([0, 1], Rotation.concatenate([ra, rb]))([amount]).as_rotvec()[0]
    return out

def protocol_value(values, motion, phase):
    def blend(f, s, amt): return slerp_pose(values[f], values[s], _smooth(amt))
    def path(keys):
        pos = np.clip(phase, 0, 1) * (len(keys) - 1); i = min(len(keys) - 2, int(pos)); return blend(keys[i], keys[i + 1], pos - i)
    if motion == "head": return path(("neutral", "head_left", "neutral", "head_right", "neutral", "head_down", "neutral", "head_up", "neutral"))
    if motion in ("a_pose", "t_pose", "forward"): return blend("neutral", motion, _hold(phase))
    if motion == "elbows": return path(("t_pose", "elbow_right", "t_pose", "elbow_left", "t_pose", "elbow_right", "t_pose", "elbow_left", "t_pose"))
    if motion == "wrists": return path(("forward", "wrist_pronate", "wrist_supinate", "wrist_flex", "wrist_extend", "forward"))
    if motion == "squat": return blend("neutral", "squat", _pulse(phase, 2))
    if motion in ("left_knee", "right_knee"): return blend("neutral", "left_leg" if motion == "left_knee" else "right_leg", _pulse(phase, 2))
    if motion == "ankles": return path(("neutral", "heel_raise", "neutral", "toe_raise", "neutral"))
    if motion in ("hand_open", "fist", "pinch", "fingers"): return values["forward"].copy()
    if motion == "turns":
        keys = ["neutral"]
        for step in range(16): keys.extend(("left_leg" if step % 2 == 0 else "right_leg", "neutral"))
        return path(tuple(keys))
    if motion == "walk": return path(("neutral", "walk_left_swing", "walk_stride", "walk_right_swing", "neutral", "walk_right_swing", "walk_stride", "walk_left_swing", "neutral"))
    return values["neutral"].copy()

def tracker_times(tracker_npz):
    z = np.load(tracker_npz, allow_pickle=False); t = np.asarray(z["tracker_0__pose__t_host_us"], np.float64) * 1e-6
    pos = np.stack([z[f"tracker_0__pose.{r}.position"] for r in ("camera", "left_wrist", "right_wrist", "left_ankle", "right_ankle")], 1)
    return t - t[0], pos

def main():
    p = argparse.ArgumentParser(); p.add_argument("--tracker", type=Path, required=True); p.add_argument("--output", type=Path, required=True)
    p.add_argument("--protocol-alignment", type=Path); p.add_argument("--protocol-poses", type=Path, default=Path("assets/protocol_body_poses.json"))
    p.add_argument("--guide", type=Path); p.add_argument("--stand-guide", type=Path, help="guide npz whose first frame is the standing pose (default: B07)")
    p.add_argument("--active-threshold-m", type=float, default=0.08); p.add_argument("--joints", default="legs_trunk", choices=("legs_trunk", "all"))
    a = p.parse_args()
    ts, pos = tracker_times(a.tracker); T = len(ts); ref = np.zeros((T, 63), np.float32); mask = np.zeros(T, np.float32); note = {}
    if a.protocol_alignment:
        values = {k: np.asarray(v, np.float32) for k, v in json.load(open(a.protocol_poses))["poses"].items()}
        segs = json.load(open(a.protocol_alignment))["alignment"]["segments"]
        cache = {}
        for i, t in enumerate(ts):
            seg = next((s for s in segs if s["start_s"] <= t < s["end_s"]), None)
            if seg is None: ref[i] = values["neutral"]; continue
            s0, s1 = seg["active_start_s"], seg["active_end_s"]
            if t < s0 or t > s1: ref[i] = values["neutral"]; mask[i] = 1.0; continue
            phase = (t - s0) / max(1e-6, s1 - s0); key = (seg["key"], round(phase, 3))
            if key not in cache: cache[key] = protocol_value(values, seg["key"], phase)
            ref[i] = cache[key]; mask[i] = 1.0 if seg.get("confidence", "high") != "low" else 0.5
        note = {"source": "protocol", "alignment": str(a.protocol_alignment)}
    elif a.guide:
        g = np.load(a.guide, allow_pickle=True); gb = np.asarray(g["body_pose"], np.float32); ph = np.asarray(g["normalized_phase"], np.float64)
        stand = np.asarray(np.load(a.stand_guide or a.guide.with_name("B07_vposer_smplx.npz"), allow_pickle=True)["body_pose"], np.float32)[0]
        base = pos[:max(2, int(0.5 / max(np.median(np.diff(ts)), 1e-3)))].mean(0)
        dev = np.max(np.linalg.norm(pos - base[None], axis=-1), axis=1); active = dev > a.active_threshold_m
        if active.any():
            i0, i1 = int(np.argmax(active)), int(T - np.argmax(active[::-1]) - 1)
        else: i0, i1 = 0, T - 1
        for i, t in enumerate(ts):
            if i < i0 or i > i1: ref[i] = stand; mask[i] = 1.0; continue
            phase = (i - i0) / max(1, i1 - i0)
            if len(gb) == 1: ref[i] = slerp_pose(stand, gb[0], _hold(phase))
            else:
                k = np.searchsorted(ph, phase); k = int(np.clip(k, 1, len(ph) - 1)); amt = (phase - ph[k-1]) / max(1e-6, ph[k] - ph[k-1])
                ref[i] = slerp_pose(gb[k-1], gb[k], float(np.clip(amt, 0, 1)))
            mask[i] = 1.0
        note = {"source": "guide", "guide": str(a.guide), "active_frames": [i0, i1], "frames": T}
    else: raise SystemExit("give --protocol-alignment or --guide")
    jm = np.zeros(21, np.float32); jm[list(LEG_TRUNK) if a.joints == "legs_trunk" else range(21)] = 1.0
    a.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.output, body_pose_ref=ref, mask=mask, joint_mask=jm, timestamps=ts.astype(np.float64))
    kx = np.degrees(ref.reshape(T, 21, 3)[:, [3, 4], 0]); print(json.dumps({**note, "output": str(a.output), "mask_frames": int(mask.sum()), "knee_x_range_deg": [float(kx.min()), float(kx.max())]}, ensure_ascii=False))

if __name__ == "__main__": main()
