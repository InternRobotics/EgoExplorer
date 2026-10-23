#!/usr/bin/env python
"""Convert processed EgoExplorer sessions into a LeRobot v3.0 dataset.

Inputs are the outputs of this toolkit:
  --episodes      CSV, one row per episode: session, task_zh, task_en, environment_type
                  [, trim_head_s, trim_tail_s]  (row order = episode order)
  --raw-root      folder holding <session>/<session>.{mp4,video.npz,tracker.npz,npz,session.json}
                  (searched recursively; use the merged input view $WORK/input/ego_data_sync)
  --unified-root  unified_30hz/ exports (<session>.npz + .json)
  --projected-root (optional) projected/ results, for residual statistics in quality.json
  --captions-root (optional) <session>.en.jsonl step annotations

Layout produced:
  <out>/data/chunk-000/file-NNN.parquet
  <out>/videos/observation.images.camera_0/chunk-000/file-NNN.mp4
  <out>/meta/{info.json, tasks.parquet, stats.json, episodes/chunk-000/file-000.parquet}
  <out>/episode_meta/episode_XXXXXX/{metadata.json, calibration/calibration.json,
        annotations/instruction.json, annotations/segments.parquet,
        quality/quality.json, smplx/smplx_episode.npz}

The episode timeline is the 30 Hz video frame timeline: each video is re-encoded as
strict 30 fps CFR so that the parquet `timestamp` equals the mp4 pts (LeRobot reads
video with tolerance 1e-4 s).  Trackers are interpolated from their native samples onto
that timeline; hands, touch and SMPL-X come from the unified 30 Hz export.
Quaternions are written in w, x, y, z order.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

CONVERTER_VERSION = '3.0.0'
VIDEO_KEY = 'observation.images.camera_0'
TRACKER_ROLES = ['camera', 'left_ankle', 'left_wrist', 'right_ankle', 'right_wrist']
FPS = 30
TZ = timezone(timedelta(hours=8))
HANDS = ['left', 'right']
HAND_LETTER = {'left': 'L', 'right': 'R'}
ROT6D_NAMES = ['r00', 'r10', 'r20', 'r01', 'r11', 'r21']

TASK_ID = {
    '物料称量': 'weighing', '萃取': 'extraction', '开反应': 'reaction_setup',
    '旋蒸': 'rotary_evaporation', '烘箱取料': 'oven_drying', '过柱': 'column_chromatography',
    '折大纸箱': 'fold_large_cardboard_box', '书架整理': 'bookshelf_organization',
    '叠衣服': 'clothes_folding', '制作三明治': 'sandwich_making',
}
SCENE_ID = {'lab': 'chemistry_lab', 'home': 'home'}
TACTILE_REGIONS = ['thumb_distal', 'thumb_proximal', 'index_distal', 'index_middle',
                   'index_proximal', 'middle_distal', 'middle_middle', 'middle_proximal',
                   'ring_distal', 'ring_middle', 'ring_proximal', 'pinky_distal',
                   'pinky_middle', 'pinky_proximal', 'palm_radial', 'palm_central',
                   'palm_ulnar']
TACTILE_COARSE = ['thumb', 'index', 'middle', 'ring', 'pinky', 'palm']
SMPLX_SCALAR_KEYS = ['betas', 'scale', 'ground_y_m', 'coordinate_frame', 'attachment_position',
                     'attachment_rotation', 'tracker_roles', 'tactile_region_names',
                     'tactile_coarse_names', 'tactile_levels', 'glove_layout',
                     'hand_pose_source', 'rate_hz', 'video_native_hz', 'tracker_native_hz',
                     'export_note']

# Tracker interpolation guards: never interpolate through a native dropout or a jump.
GAP_HOLD_S = 0.25          # native dropout longer than this
JUMP_HOLD_M = 0.10         # > 6.5 m/s between native samples = relocalisation jump
JUMP_HOLD_DEG = 60.0       # implausible angular step between native samples


# ---------------------------------------------------------------- utilities
def sha256_of(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 22), b''):
            h.update(chunk)
    return h.hexdigest()


def nearest_idx(grid, t):
    """grid: sorted 1-D array; t: 1-D array -> indices of the nearest grid element."""
    i = np.searchsorted(grid, t)
    i0 = np.clip(i - 1, 0, len(grid) - 1)
    i1 = np.clip(i, 0, len(grid) - 1)
    return np.where(np.abs(grid[i1] - t) < np.abs(grid[i0] - t), i1, i0)


def quat_xyzw_to_rot6d(q):
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    r00 = 1 - 2 * (y * y + z * z)
    r10 = 2 * (x * y + z * w)
    r20 = 2 * (x * z - y * w)
    r01 = 2 * (x * y - z * w)
    r11 = 1 - 2 * (x * x + z * z)
    r21 = 2 * (y * z + x * w)
    return np.stack([r00, r10, r20, r01, r11, r21], axis=-1).astype(np.float32)


def slerp(q0, q1, a):
    """Shortest-path spherical interpolation. q0, q1: (n,4); a: (n,)."""
    d = np.sum(q0 * q1, axis=-1)
    q1 = np.where((d < 0)[:, None], -q1, q1)
    d = np.clip(np.abs(d), 0.0, 1.0)
    theta = np.arccos(d)
    sin_theta = np.sin(theta)
    small = sin_theta < 1e-6
    s = np.where(small, 1.0, sin_theta)
    w0 = np.where(small, 1.0 - a, np.sin((1.0 - a) * theta) / s)
    w1 = np.where(small, a, np.sin(a * theta) / s)
    q = w0[:, None] * q0 + w1[:, None] * q1
    n = np.linalg.norm(q, axis=-1, keepdims=True)
    return q / np.where(n < 1e-12, 1.0, n)


def iso_from_us(t_us):
    return datetime.fromtimestamp(t_us / 1e6, TZ).isoformat(timespec='microseconds')


def slug(text):
    return re.sub(r'[^a-z0-9]+', '_', str(text).lower()).strip('_') or 'task'


def video_frame_count(path):
    """Exact decoded frame count (container metadata can be wrong for VFR)."""
    import cv2
    cap = cv2.VideoCapture(str(path))
    n = 0
    while cap.grab():
        n += 1
    cap.release()
    return n


# ---------------------------------------------------------------- episode source
class Episode:
    def __init__(self, row, index, args):
        self.row = row
        self.index = index
        self.session = row['session']
        matches = sorted(args.raw_root.rglob(f'{self.session}.tracker.npz'))
        if not matches:
            raise FileNotFoundError(f'{self.session}: no tracker.npz under {args.raw_root}')
        self.raw_dir = matches[0].parent
        self.tracker_path = matches[0]
        self.session_json_path = self.raw_dir / f'{self.session}.session.json'
        self.glove_npz_path = self.raw_dir / f'{self.session}.npz'
        self.video_npz_path = self.raw_dir / f'{self.session}.video.npz'
        self.raw_video_path = self.raw_dir / f'{self.session}.mp4'
        self.smplx_npz_path = args.unified_root / f'{self.session}.npz'
        self.smplx_json_path = args.unified_root / f'{self.session}.json'
        self.diagnostics_path = (args.projected_root / self.session / 'tracker_diagnostics.npz'
                                 if args.projected_root else None)
        self.session_json = json.load(open(self.session_json_path, encoding='utf-8'))
        self.glove_type = self.session_json.get('glove_type', 'unknown')
        self.trim_head_s = float(row.get('trim_head_s') or 0.0)
        self.trim_tail_s = float(row.get('trim_tail_s') or 0.0)
        self.out_video_path = (args.output / 'videos' / VIDEO_KEY / 'chunk-000'
                               / f'file-{index:03d}.mp4')


# ---------------------------------------------------------------- video and frame grid
def encode_video(ep, ffmpeg, crop_top):
    """Strict 30 fps CFR re-encode, top `crop_top` px removed (GoPro overlay)."""
    if ep.out_video_path.exists():
        return
    ep.out_video_path.parent.mkdir(parents=True, exist_ok=True)
    w, h = ep.session_json['video']['size']
    t = np.load(ep.video_npz_path)['t_host_us'].astype(np.int64)
    duration = (t[-1] - t[0]) / 1e6 - ep.trim_head_s - ep.trim_tail_s
    tmp = ep.out_video_path.with_suffix('.tmp.mp4')
    subprocess.run([ffmpeg, '-hide_banner', '-loglevel', 'error', '-y',
                    '-i', str(ep.raw_video_path), '-ss', f'{ep.trim_head_s:.3f}', '-t', f'{duration:.3f}',
                    '-vf', f'crop={w}:{h - crop_top}:0:{crop_top},fps={FPS}', '-fps_mode', 'cfr',
                    '-c:v', 'libx264', '-crf', '18', '-preset', 'veryfast', '-pix_fmt', 'yuv420p',
                    '-an', str(tmp)], check=True)
    tmp.rename(ep.out_video_path)
    print(f'video encoded (30 fps CFR): {ep.out_video_path.name} <- {ep.session}', flush=True)


def episode_frame_grid(ep):
    """(timestamps_s float32[n], host_us int64[n]) of the re-encoded video frames.

    `host_us` is the true capture instant of each frame: the first kept frame's host
    timestamp advanced by exactly 1/30 s per frame.
    """
    t_host = np.load(ep.video_npz_path)['t_host_us'].astype(np.int64)
    n = video_frame_count(ep.out_video_path)
    expected = (t_host[-1] - t_host[0]) / 1e6 - ep.trim_head_s - ep.trim_tail_s
    if abs(n / FPS - expected) > 0.5:
        raise RuntimeError(f'{ep.session}: re-encoded {n} frames ({n / FPS:.2f} s) vs {expected:.2f} s')
    t0 = t_host[0] + int(round(ep.trim_head_s * 1e6))
    host_us = t0 + np.round(np.arange(n) / FPS * 1e6).astype(np.int64)
    return (np.arange(n) / FPS).astype(np.float32), host_us


# ---------------------------------------------------------------- tracker interpolation
def interpolate_tracker(trk, role, t_q):
    """Interpolate native tracker samples onto the query timeline t_q (host us).

    Position is linear, orientation shortest-path SLERP, both over valid samples only.
    Queries inside a native gap longer than GAP_HOLD_S, or across a position / orientation
    jump, hold the nearer bracketing sample instead of fabricating motion.
    Raw quaternions are wxyz; they are handled as xyzw internally.
    Returns (position[n,3] f32, quaternion_xyzw[n,4] f32, stats dict).
    """
    tt = trk['tracker_0__pose__t_host_us'].astype(np.float64)
    pos = np.asarray(trk[f'tracker_0__pose.{role}.position'], dtype=np.float64)
    quat = np.asarray(trk[f'tracker_0__pose.{role}.orientation'], dtype=np.float64)[:, [1, 2, 3, 0]]
    valid = np.asarray(trk[f'tracker_0__pose.{role}.valid'], dtype=bool)
    order = np.argsort(tt, kind='stable')
    tt, pos, quat, valid = tt[order], pos[order], quat[order], valid[order]
    good = valid & np.isfinite(pos).all(1) & np.isfinite(quat).all(1) & (np.linalg.norm(quat, axis=1) > 1e-6)
    tq = np.asarray(t_q, dtype=np.float64)
    held = held_gap = held_jump = np.zeros(len(tq), bool)

    if good.sum() >= 2:
        tg, pg, qg = tt[good], pos[good], quat[good]
        qg = qg / np.linalg.norm(qg, axis=1, keepdims=True)
        i1 = np.clip(np.searchsorted(tg, tq, side='left'), 0, len(tg) - 1)
        i0 = np.clip(i1 - 1, 0, len(tg) - 1)
        same = i0 == i1
        t0, t1 = tg[i0], tg[i1]
        span = np.where(t1 > t0, t1 - t0, 1.0)
        a = np.where(same, 0.0, np.clip((tq - t0) / span, 0.0, 1.0))
        held_gap = (~same) & (span > GAP_HOLD_S * 1e6)
        dpos = np.linalg.norm(pg[i1] - pg[i0], axis=1)
        dquat = 2.0 * np.arccos(np.clip(np.abs(np.sum(qg[i0] * qg[i1], axis=1)), 0.0, 1.0))
        held_jump = (~same) & ((dpos > JUMP_HOLD_M) | (dquat > np.deg2rad(JUMP_HOLD_DEG)))
        held = held_gap | held_jump
        a = np.where(held, np.where(tq - t0 < t1 - tq, 0.0, 1.0), a)
        P = pg[i0] + a[:, None] * (pg[i1] - pg[i0])
        Q = slerp(qg[i0], qg[i1], a)
        gap = np.diff(tg) / 1e6
    else:  # one valid sample or none: nearest sample, marked as not interpolated
        src = np.nonzero(good)[0] if good.any() else np.arange(len(tt))
        i = src[nearest_idx(tt[src], tq)]
        P = pos[i]
        Q = quat[i] / np.clip(np.linalg.norm(quat[i], axis=1, keepdims=True), 1e-12, None)
        held = np.ones(len(tq), bool)
        gap = np.array([0.0])

    stats = {
        'native_samples': int(len(tt)),
        'valid_samples': int(good.sum()),
        'valid_fraction': float(good.mean()),
        'native_rate_hz': float(len(tt) / max((tt[-1] - tt[0]) / 1e6, 1e-9)),
        'valid_on_query_fraction': float(valid[nearest_idx(tt, tq)].mean()),
        'median_sample_gap_s': float(np.median(gap)),
        'max_sample_gap_s': float(gap.max()),
        'gap_hold_threshold_s': GAP_HOLD_S,
        'jump_hold_threshold_m': JUMP_HOLD_M,
        'jump_hold_threshold_deg': JUMP_HOLD_DEG,
        'interpolated_fraction': float(1.0 - held.mean()),
        'held_across_gap_fraction': float(held_gap.mean()),
        'held_across_jump_fraction': float(held_jump.mean()),
    }
    return P.astype(np.float32), Q.astype(np.float32), stats


# ---------------------------------------------------------------- SMPL-X and touch
def resample_smplx(ep, host_us):
    """Unified 30 Hz arrays sampled onto the episode frame grid (nearest grid point)."""
    d = np.load(ep.smplx_npz_path, allow_pickle=True)
    t_s = d['timestamps'].astype(np.float64)
    idx = nearest_idx(t_s, host_us / 1e6)
    out = {k: d[k][idx] for k in d.files
           if k != 'timestamps' and d[k].ndim >= 1 and d[k].shape[0] == len(t_s)}
    out.update({k: d[k] for k in SMPLX_SCALAR_KEYS if k in d.files})
    return out, d, idx


class TouchPipeline:
    """The toolkit's wuji_correct L0->L4 pipeline, for hands missing from the export."""

    def __init__(self, tactile_dir):
        self.dir = Path(tactile_dir)
        self._process = None
        self._db = None

    def load(self):
        if self._process is None:
            sys.path.insert(0, str(self.dir))
            from wuji_correct import process
            self._process = process
            self._db = np.load(self.dir / 'health' / 'health_db.npz', allow_pickle=True)
        return self._process, self._db


def tactile_17zone(ep, side, smplx_raw, smplx_idx, touch):
    """17-zone touch for one hand on the episode grid.

    1. the unified 30 Hz export (`tactile_<H>_L1/L2/L4`)           -> 'unified_30hz'
    2. wuji_correct re-run on the raw glove npz (per-session health,
       else the newest day of the health table)                     -> 'reconstructed_wuji_correct'
    MANUS sessions have no touch                                     -> 'unavailable'
    Returns (dict of (n,K) arrays or None values, coverage (17,) or None, source).
    """
    L = HAND_LETTER[side]
    empty = dict(contact=None, score=None, residual=None, coarse_contact=None, coarse_score=None)
    if f'tactile_{L}_L4_contact' in smplx_raw.files:
        i = smplx_idx
        out = {'contact': smplx_raw[f'tactile_{L}_L4_contact'][i],
               'score': smplx_raw[f'tactile_{L}_L2_score'][i],
               'residual': smplx_raw[f'tactile_{L}_L1_residual'][i],
               'coarse_contact': smplx_raw[f'tactile_{L}_coarse_contact'][i],
               'coarse_score': smplx_raw[f'tactile_{L}_coarse_score'][i]}
        out = {k: np.asarray(v, np.float32) for k, v in out.items()}
        return out, np.asarray(smplx_raw[f'tactile_{L}_coverage'], np.float32), 'unified_30hz'
    device = next((g['device_name'] for g in ep.session_json.get('gloves', []) if g.get('slot') == side), None)
    if ep.glove_type != 'wuji' or device is None or not ep.glove_npz_path.exists():
        return empty, None, 'unavailable'
    z = np.load(ep.glove_npz_path)
    if f'{device}__tactile' not in z.files:
        return empty, None, 'unavailable'
    raw = np.asarray(z[f'{device}__tactile'], dtype=np.float32)
    t = z[f'{device}__tactile__t_host_us'].astype(np.int64)
    process, db = touch.load()
    sessions = db['sessions'].tolist()
    health = (db[f'{side}_health_session'][sessions.index(ep.session)] if ep.session in sessions
              else db[f'{side}_health_day'][-1])
    with np.errstate(all='ignore'):
        res = process(raw, 1e6 / float(np.median(np.diff(t))), health)
    grid_us = (smplx_raw['timestamps'][smplx_idx] * 1e6)
    i = nearest_idx(t.astype(np.float64), grid_us)
    out = {'contact': res['region_contact'][i], 'score': res['region_score'][i],
           'residual': res['region_res'][i], 'coarse_contact': res['coarse_contact'][i],
           'coarse_score': res['coarse_score'][i]}
    out = {k: np.asarray(v, np.float32) for k, v in out.items()}
    return out, np.asarray(res['coverage'], np.float32), 'reconstructed_wuji_correct'


# ---------------------------------------------------------------- per-frame features
def build_frame_data(ep, host_us, smplx_resampled, trk_stats, tactile):
    n = len(host_us)
    feats = {}
    trk = np.load(ep.tracker_path)
    for role in TRACKER_ROLES:
        pos, quat, stats = interpolate_tracker(trk, role, host_us.astype(np.float64))
        trk_stats[role] = stats
        feats[f'tracker.{role}.position'] = pos
        feats[f'tracker.{role}.orientation'] = quat[:, [3, 0, 1, 2]]      # released order: w, x, y, z
        feats[f'tracker.{role}.rotation_6d'] = quat_xyzw_to_rot6d(quat)
    feats['gravity.direction'] = np.tile(np.array([0, -1, 0], np.float32), (n, 1))
    for side in HANDS:
        pts = smplx_resampled.get(f'glove_{side}_points')
        feats[f'observation.hand.{side}.joint_positions'] = (
            np.asarray(pts, np.float32) if pts is not None and pts.shape[1] == 21
            else np.full((n, 21, 3), np.nan, np.float32))
        values = tactile[side][0]
        for suffix, width in [('contact', 17), ('score', 17), ('residual', 17),
                              ('coarse_contact', 6), ('coarse_score', 6)]:
            v = values.get(suffix)
            feats[f'observation.hand.{side}.tactile_17zone_{suffix}'] = (
                v if v is not None else np.full((n, width), np.nan, np.float32))
    return feats


# ---------------------------------------------------------------- parquet io
def f32list(values):
    arr = np.ascontiguousarray(values, dtype=np.float32)
    flat = pa.array(arr.reshape(-1), type=pa.float32())
    for dim in reversed(arr.shape[1:]):
        flat = pa.FixedSizeListArray.from_arrays(flat, dim)
    return flat


def write_episode_parquet(path, feats, timestamps, episode_index, task_index, index_offset):
    n = len(timestamps)
    cols = {
        'timestamp': pa.array(timestamps, type=pa.float32()),
        'frame_index': pa.array(np.arange(n, dtype=np.int64)),
        'episode_index': pa.array(np.full(n, episode_index, dtype=np.int64)),
        'index': pa.array(np.arange(index_offset, index_offset + n, dtype=np.int64)),
        'task_index': pa.array(np.full(n, task_index, dtype=np.int64)),
    }
    cols.update({k: f32list(v) for k, v in feats.items()})
    pq.write_table(pa.table(cols), path, compression='zstd')


class StatAcc:
    def __init__(self):
        self.n, self.sum, self.sumsq, self.min, self.max = 0, 0.0, 0.0, np.inf, -np.inf

    def update(self, arr):
        a = np.asarray(arr, dtype=np.float64).ravel()
        a = a[np.isfinite(a)]
        if a.size:
            self.n += a.size
            self.sum += a.sum()
            self.sumsq += (a * a).sum()
            self.min = min(self.min, float(a.min()))
            self.max = max(self.max, float(a.max()))

    def result(self, with_std=True):
        if self.n == 0:
            return {'mean': None, 'min': None, 'max': None, **({'std': None} if with_std else {})}
        mean = self.sum / self.n
        out = {'mean': float(mean), 'min': float(self.min), 'max': float(self.max)}
        if with_std:
            out['std'] = float(np.sqrt(max(self.sumsq / self.n - mean * mean, 0.0)))
        return out


# ---------------------------------------------------------------- annotations
SEGMENTS_SCHEMA = pa.schema([
    ('segment_index', pa.int32()), ('start', pa.float32()), ('end', pa.float32()),
    ('phase', pa.string()), ('phase_zh', pa.string()),
    ('action', pa.string()), ('action_zh', pa.string()),
    ('caption', pa.string()), ('caption_zh', pa.string()),
    ('confidence', pa.string()), ('qc_final', pa.string()),
    ('annotation_source', pa.string()), ('needs_human_review', pa.bool_()),
])


def load_segments(ep, captions_root):
    """Step annotations from <captions_root>/<session>.en.jsonl, shifted by the head trim."""
    if captions_root is None or not (captions_root / f'{ep.session}.en.jsonl').exists():
        return []
    rows = []
    for line in open(captions_root / f'{ep.session}.en.jsonl', encoding='utf-8'):
        r = json.loads(line)
        rows.append({
            'segment_index': int(r['segment_index']),
            'start': float(r['start']) - ep.trim_head_s, 'end': float(r['end']) - ep.trim_head_s,
            'phase': r.get('phase'), 'phase_zh': r.get('phase_zh'),
            'action': r.get('action'), 'action_zh': r.get('action_zh'),
            'caption': r.get('caption_en'), 'caption_zh': r.get('caption'),
            'confidence': r.get('confidence'), 'qc_final': r.get('qc_final'),
            'annotation_source': 'reviewed' if r.get('qc_final') == 'accept' else 'automatic',
            'needs_human_review': str(r.get('needs_human_review', '')).strip().lower() in ('true', '1'),
        })
    rows.sort(key=lambda r: r['segment_index'])
    return rows


# ---------------------------------------------------------------- quality
def residual_stats(ep):
    """Fit residuals of the projection (mm), from tracker_diagnostics.npz when available."""
    if ep.diagnostics_path is None or not ep.diagnostics_path.exists():
        return {}
    with np.load(ep.diagnostics_path) as d:
        r = np.linalg.norm(np.asarray(d['position_residual_m'], dtype=np.float64), axis=-1) * 1000
        out = {'residual_p50_mm': float(np.nanmedian(r)), 'residual_p95_mm': float(np.nanpercentile(r, 95))}
        if 'rotation_residual_deg' in d.files:
            out['rotation_deg'] = float(np.nanmean(d['rotation_residual_deg']))
    return {k: round(v, 2) for k, v in out.items()}


# ---------------------------------------------------------------- per-episode meta
def write_episode_meta(ep, ep_dir, args, timestamps, smplx_resampled, smplx_raw, segments,
                       src_sha, trk_stats, tactile):
    sj = ep.session_json
    stats_json = json.load(open(ep.smplx_json_path, encoding='utf-8')) if ep.smplx_json_path.exists() else {}
    task_zh, task_en = ep.row['task_zh'], ep.row.get('task_en') or ep.row['task_zh']
    env = ep.row.get('environment_type') or 'unknown'
    for sub in ['calibration', 'annotations', 'quality', 'smplx']:
        (ep_dir / sub).mkdir(parents=True, exist_ok=True)
    v_range = sj['video']['t_host_us_range']
    w, h = sj['video']['size']
    tactile_hands = [s for s in HANDS if tactile[s][0].get('contact') is not None]
    tactile_sources = {s: tactile[s][2] for s in HANDS}
    tactile_coverage = {s: (tactile[s][1].tolist() if tactile[s][1] is not None else None) for s in HANDS}

    metadata = {
        'episode_id': f'episode_{ep.index + 1:06d}',
        'episode_index': ep.index,
        'session_id': ep.session,
        'subject_id': None,
        'subject_anonymized': True,
        'scene_id': SCENE_ID.get(env, env),
        'task_id': TASK_ID.get(task_zh, slug(task_en)),
        'task_zh': task_zh,
        'task_en': task_en,
        'environment_type': env,
        'start_timestamp': iso_from_us(v_range[0]),
        'end_timestamp': iso_from_us(v_range[1]),
        'start_timestamp_host_us': int(v_range[0]),
        'end_timestamp_host_us': int(v_range[1]),
        'glove_type': ep.glove_type,
        'sensor_inventory': {
            'camera': [{'count': 1, 'width': w, 'height': h, 'capture_fps': sj['video'].get('fps'),
                        'video_native_hz': stats_json.get('video_native_hz'),
                        'crop': f'top {args.crop_top} px removed -> {w}x{h - args.crop_top}',
                        'timestamp_mode': sj['video'].get('timestamp_mode')}],
            'tracker': {'count': len(sj['tracker']['roles']), 'roles': sj['tracker']['roles'],
                        'device': sj['tracker'].get('device'), 'transport': sj['tracker'].get('transport'),
                        'record_fps_limit': sj['tracker'].get('record_fps_limit')},
            'gloves': [{'type': sj.get('glove_type'), 'device_name': g['device_name'],
                        'sn': g.get('sn'), 'slot': g.get('slot')} for g in sj.get('gloves', [])],
        },
        'license': args.license,
        'privacy_status': args.privacy_status,
        'episode_frames': len(timestamps),
        'episode_duration_s': float(timestamps[-1]),
        'trim_head_s': ep.trim_head_s,
        'trim_tail_s': ep.trim_tail_s,
        'tracker_processing': {
            'method': 'linear (position) + SLERP (orientation) interpolation onto the 30 Hz video grid',
            'per_role': trk_stats,
        },
        'tactile_17zone': {
            'hands_available': tactile_hands,
            'hand_source': tactile_sources,
            'coverage_17': tactile_coverage,
            'region_names': TACTILE_REGIONS,
            'coarse_names': TACTILE_COARSE,
            'missing_hands_are_nan': True,
        },
    }
    json.dump(metadata, open(ep_dir / 'metadata.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=2)

    sr = smplx_raw
    hands_with_calib = [L for L in ['L', 'R'] if f'tactile_{L}_sigma' in sr.files]
    wuji_calib = None
    if hands_with_calib:
        wuji_calib = {'tactile_hands': ['right' if L == 'R' else 'left' for L in hands_with_calib],
                      'tactile_region_names': TACTILE_REGIONS, 'tactile_coarse_names': TACTILE_COARSE}
        for L in hands_with_calib:
            wuji_calib[f'tactile_{L}_sigma_744'] = sr[f'tactile_{L}_sigma'].tolist()
            wuji_calib[f'tactile_{L}_coverage_17'] = sr[f'tactile_{L}_coverage'].tolist()
    intrinsics = ({'value': {'width': w, 'height': h - args.crop_top, 'fx': args.fx, 'fy': args.fy,
                             'cx': args.cx, 'cy': args.cy}, 'status': 'nominal'}
                  if args.fx else {'value': None, 'status': 'not_available'})
    calibration = {
        'camera_intrinsics': intrinsics,
        'camera_extrinsics': {'value': None, 'status': 'not_available'},
        'tracker_mount_transforms': {
            'source': 'unified_30hz attachment_position / attachment_rotation (subject profile)',
            'convention': 'tracker_world = joint_world @ attachment_rotation',
            'roles': [str(r) for r in sr['tracker_roles']],
            'attachment_position': np.asarray(sr['attachment_position']).tolist(),
            'attachment_rotation': np.asarray(sr['attachment_rotation']).tolist(),
        } if 'attachment_position' in sr.files else {'value': None, 'status': 'not_available'},
        'global_coordinate_convention': {
            'coordinate_frame': str(sr['coordinate_frame']) if 'coordinate_frame' in sr.files else 'raw_capture_y_up',
            'ground_y_m': float(sr['ground_y_m']) if 'ground_y_m' in sr.files else None,
            'axes': 'right-handed, Y-up, metres',
        },
        'subject': {
            'scale': float(sr['scale']) if 'scale' in sr.files else None,
            'smplx_betas': np.asarray(sr['betas']).tolist() if 'betas' in sr.files else None,
            'smplx_vertices': 'world = scale * SMPLX(transl=0).vertices + transl',
        },
        'wuji_sensor_calibration': wuji_calib,
        'sync_offsets': {
            'clock_note': sj.get('clock'),
            'video_t_host_us_range': [int(v_range[0]), int(v_range[1])],
            'glove_t_host_us_range': sj.get('glove_t_host_us_range'),
            'tracker_t_host_us_range': sj.get('tracker_t_host_us_range'),
            'video_time_offset_s_max': float(np.max(sr['video_time_offset_s']))
            if 'video_time_offset_s' in sr.files else None,
        },
    }
    json.dump(calibration, open(ep_dir / 'calibration' / 'calibration.json', 'w', encoding='utf-8'),
              ensure_ascii=False, indent=2)

    instruction = {'task_id': metadata['task_id'], 'instruction_zh': task_zh, 'instruction_en': task_en}
    json.dump(instruction, open(ep_dir / 'annotations' / 'instruction.json', 'w', encoding='utf-8'),
              ensure_ascii=False, indent=2)
    pq.write_table(pa.Table.from_pylist(segments, schema=SEGMENTS_SCHEMA),
                   ep_dir / 'annotations' / 'segments.parquet', compression='zstd')

    quality = {
        **residual_stats(ep),
        'unified_export': stats_json,
        'tracker_interpolation': trk_stats,
        'tactile_17zone_hand_source': tactile_sources,
        'tactile_17zone_coverage': tactile_coverage,
        'conversion': {
            'converter_version': CONVERTER_VERSION,
            'converted_at': datetime.now(TZ).isoformat(timespec='seconds'),
            'source_files_sha256': src_sha,
        },
    }
    json.dump(quality, open(ep_dir / 'quality' / 'quality.json', 'w', encoding='utf-8'),
              ensure_ascii=False, indent=2)
    np.savez_compressed(ep_dir / 'smplx' / 'smplx_episode.npz', **smplx_resampled)


# ---------------------------------------------------------------- main build
def build(args):
    rows = list(csv.DictReader(open(args.episodes, encoding='utf-8-sig')))
    if not rows:
        raise SystemExit(f'{args.episodes}: no episodes')
    tasks = list(dict.fromkeys(r['task_zh'] for r in rows))
    task_index_of = {t: i for i, t in enumerate(tasks)}
    out = args.output
    (out / 'data' / 'chunk-000').mkdir(parents=True, exist_ok=True)
    touch = TouchPipeline(args.tactile_dir)

    feature_stats, episodes_meta, tactile_map = {}, [], {}
    index_offset, total_duration, glove_types = 0, 0.0, set()
    for episode_index, row in enumerate(rows):
        ep = Episode(row, episode_index, args)
        glove_types.add(ep.glove_type)
        encode_video(ep, args.ffmpeg, args.crop_top)
        timestamps, host_us = episode_frame_grid(ep)
        n = len(timestamps)
        total_duration += float(timestamps[-1])

        smplx_resampled, smplx_raw, smplx_idx = resample_smplx(ep, host_us)
        tactile = {s: tactile_17zone(ep, s, smplx_raw, smplx_idx, touch) for s in HANDS}
        for s in HANDS:  # keep the per-episode SMPL-X npz self-contained
            contact = tactile[s][0].get('contact')
            smplx_resampled[f'contact_17zones_{s[0]}'] = (
                contact if contact is not None else np.full((n, 17), np.nan, np.float32))
        trk_stats = {}
        feats = build_frame_data(ep, host_us, smplx_resampled, trk_stats, tactile)
        write_episode_parquet(out / 'data' / 'chunk-000' / f'file-{episode_index:03d}.parquet', feats,
                              timestamps, episode_index, task_index_of[row['task_zh']], index_offset)

        src_files = [ep.session_json_path, ep.raw_video_path, ep.glove_npz_path, ep.video_npz_path,
                     ep.tracker_path, ep.smplx_npz_path]
        src_sha = {p.name: sha256_of(p) for p in src_files if p.exists()}
        write_episode_meta(ep, out / 'episode_meta' / f'episode_{episode_index + 1:06d}', args, timestamps,
                           smplx_resampled, smplx_raw, load_segments(ep, args.captions_root), src_sha,
                           trk_stats, tactile)

        arrays = {'timestamp': timestamps, 'frame_index': np.arange(n),
                  'episode_index': np.full(n, episode_index),
                  'index': np.arange(index_offset, index_offset + n),
                  'task_index': np.full(n, task_index_of[row['task_zh']]), **feats}
        for k, v in arrays.items():
            feature_stats.setdefault(k, StatAcc()).update(v)
        episodes_meta.append({
            'episode_index': episode_index, 'tasks': [row['task_zh']], 'length': n,
            'data/chunk_index': 0, 'data/file_index': episode_index,
            'dataset_from_index': index_offset, 'dataset_to_index': index_offset + n,
            'meta/episodes/chunk_index': 0, 'meta/episodes/file_index': 0,
            f'videos/{VIDEO_KEY}/chunk_index': 0, f'videos/{VIDEO_KEY}/file_index': episode_index,
            f'videos/{VIDEO_KEY}/from_timestamp': 0.0, f'videos/{VIDEO_KEY}/to_timestamp': float(timestamps[-1]),
        })
        index_offset += n
        tactile_map[f'episode_{episode_index:03d}'] = {s: tactile[s][2] for s in HANDS}
        print(f'episode {episode_index} {ep.session}: {n} frames, {timestamps[-1]:.1f} s, '
              f'touch={tactile_map[f"episode_{episode_index:03d}"]}', flush=True)

    ep_schema = pa.schema([
        ('episode_index', pa.int64()), ('tasks', pa.list_(pa.string())), ('length', pa.int64()),
        ('data/chunk_index', pa.int32()), ('data/file_index', pa.int32()),
        ('dataset_from_index', pa.int64()), ('dataset_to_index', pa.int64()),
        ('meta/episodes/chunk_index', pa.int32()), ('meta/episodes/file_index', pa.int32()),
        (f'videos/{VIDEO_KEY}/chunk_index', pa.int32()), (f'videos/{VIDEO_KEY}/file_index', pa.int32()),
        (f'videos/{VIDEO_KEY}/from_timestamp', pa.float32()), (f'videos/{VIDEO_KEY}/to_timestamp', pa.float32()),
    ])
    (out / 'meta' / 'episodes' / 'chunk-000').mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(episodes_meta, schema=ep_schema),
                   out / 'meta' / 'episodes' / 'chunk-000' / 'file-000.parquet', compression='zstd')
    pq.write_table(pa.table({'task_index': pa.array(range(len(tasks)), pa.int64()), 'task': tasks}),
                   out / 'meta' / 'tasks.parquet', compression='zstd')
    stats = {k: acc.result(with_std=k not in ('frame_index', 'episode_index', 'index', 'task_index'))
             for k, acc in feature_stats.items()}
    json.dump(stats, open(out / 'meta' / 'stats.json', 'w'), indent=1)

    features = {k: {'dtype': 'float32' if k == 'timestamp' else 'int64', 'shape': [1], 'names': None}
                for k in ('timestamp', 'frame_index', 'episode_index', 'index', 'task_index')}
    for role in TRACKER_ROLES:
        features[f'tracker.{role}.position'] = {'dtype': 'float32', 'shape': [3], 'names': None}
        features[f'tracker.{role}.orientation'] = {'dtype': 'float32', 'shape': [4], 'names': ['w', 'x', 'y', 'z']}
        features[f'tracker.{role}.rotation_6d'] = {'dtype': 'float32', 'shape': [6], 'names': ROT6D_NAMES}
    features['gravity.direction'] = {'dtype': 'float32', 'shape': [3], 'names': None}
    for side in HANDS:
        features[f'observation.hand.{side}.joint_positions'] = {'dtype': 'float32', 'shape': [21, 3], 'names': None}
        for suffix in ('contact', 'score', 'residual'):
            features[f'observation.hand.{side}.tactile_17zone_{suffix}'] = {
                'dtype': 'float32', 'shape': [17], 'names': TACTILE_REGIONS}
        for suffix in ('coarse_contact', 'coarse_score'):
            features[f'observation.hand.{side}.tactile_17zone_{suffix}'] = {
                'dtype': 'float32', 'shape': [6], 'names': TACTILE_COARSE}
    first = json.load(open(Episode(rows[0], 0, args).session_json_path, encoding='utf-8'))
    w, h = first['video']['size']
    features[VIDEO_KEY] = {
        'dtype': 'video', 'shape': [h - args.crop_top, w, 3], 'names': ['height', 'width', 'channels'],
        'info': {'video.height': h - args.crop_top, 'video.width': w, 'video.codec': 'h264',
                 'video.pix_fmt': 'yuv420p', 'video.is_depth_map': False, 'video.fps': FPS,
                 'video.channels': 3},
    }
    info = {
        'codebase_version': 'v3.0',
        'fps': FPS,
        'robot_type': None,
        'features': features,
        'total_episodes': len(rows),
        'total_frames': index_offset,
        'total_tasks': len(tasks),
        'splits': {'train': f'0:{len(rows)}'},
        'data_path': 'data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet',
        'video_path': 'videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4',
        'chunks_size': 1000,
        'data_files_size_in_mb': 100,
        'video_files_size_in_mb': 1000,
        'total_duration_s': round(total_duration, 1),
        'license': args.license,
        'privacy_status': args.privacy_status,
        'glove_type': glove_types.pop() if len(glove_types) == 1 else 'mixed',
        'coordinate_system': {
            'name': 'sdk-world-as-recorded',
            'description': 'tracker SDK world frame as recorded; no origin, yaw or axis transform applied',
            'vertical_axis': '+Y',
            'gravity_direction': [0.0, -1.0, 0.0],
            'transforms_applied': [],
            'quaternion_order': 'wxyz',
            'rotation_6d': {'encoding': 'rotation-matrix-first-two-columns', 'order': ROT6D_NAMES,
                            'matrix_direction': 'tracker-local-to-world'},
        },
        'tracker_processing': {
            'grid': '30 Hz video frame timeline (timestamp == mp4 pts)',
            'position': 'linear interpolation of valid native samples',
            'orientation': 'shortest-path SLERP of valid native quaternions',
            'guards': {'gap_hold_s': GAP_HOLD_S, 'jump_hold_m': JUMP_HOLD_M, 'jump_hold_deg': JUMP_HOLD_DEG,
                       'rule': 'across a native dropout longer than gap_hold_s, or a position / orientation '
                               'step larger than jump_hold_m / jump_hold_deg, the nearer native sample is '
                               'held instead of interpolating through the discontinuity'},
        },
        'tactile_processing': {
            'value_type': 'smplx_17_zone',
            'source': 'unified 30 Hz export (wuji_correct); hands missing from it are rebuilt from the raw '
                      'glove npz with the same pipeline (per-session health, else the newest day)',
            'levels': {'L1': 'residual (rolling p10 baseline removed)', 'L2': 'robust z-score (MAD)',
                       'L4': 'contact flag (region 80th percentile + hysteretic dual criterion)'},
            'region_names': TACTILE_REGIONS,
            'coarse_names': TACTILE_COARSE,
            'contact_dtype': 'float32 (0.0 / 1.0)',
            'missing_hand_value': 'NaN',
            'per_episode_hand_source': tactile_map,
        },
        'episode_extensions': 'episode_meta/episode_XXXXXX/{metadata.json, calibration/, annotations/, quality/, smplx/}',
        'converter_version': CONVERTER_VERSION,
    }
    json.dump(info, open(out / 'meta' / 'info.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
    print(f'TOTAL: {len(rows)} episodes, {index_offset} frames, {total_duration:.1f} s -> {out}')


def validate(out):
    """Cheap self-check: parquet rows == decoded video frames, timestamps on the 1/30 s grid."""
    info = json.load(open(out / 'meta' / 'info.json', encoding='utf-8'))
    problems = 0
    for i in range(info['total_episodes']):
        t = pq.read_table(out / 'data' / 'chunk-000' / f'file-{i:03d}.parquet', columns=['timestamp'])
        ts = t.column('timestamp').to_numpy()
        n_vid = video_frame_count(out / 'videos' / VIDEO_KEY / 'chunk-000' / f'file-{i:03d}.mp4')
        grid_err = float(np.max(np.abs(ts - np.arange(len(ts)) / FPS)))
        ok = n_vid == len(ts) and grid_err < 1e-4
        problems += not ok
        print(f'episode {i}: rows={len(ts)} video_frames={n_vid} max_grid_err={grid_err:.2e} '
              f'{"ok" if ok else "MISMATCH"}')
    return problems


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--episodes', type=Path, required=True)
    p.add_argument('--raw-root', type=Path, required=True)
    p.add_argument('--unified-root', type=Path, required=True)
    p.add_argument('--projected-root', type=Path, default=None)
    p.add_argument('--captions-root', type=Path, default=None)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--tactile-dir', type=Path,
                   default=Path(os.environ.get('EGOEXPLORER_TACTILE_DIR', Path(__file__).resolve().parents[1] / 'tactile')))
    p.add_argument('--ffmpeg', default=shutil.which('ffmpeg') or 'ffmpeg')
    p.add_argument('--crop-top', type=int, default=110, help='rows removed from the top (GoPro HDMI overlay)')
    p.add_argument('--fx', type=float, default=1012.0)
    p.add_argument('--fy', type=float, default=1016.0)
    p.add_argument('--cx', type=float, default=960.0)
    p.add_argument('--cy', type=float, default=430.0)
    p.add_argument('--license', default='internal-research-only')
    p.add_argument('--privacy-status', default='contains_person_view; faces_not_redacted')
    p.add_argument('--validate-only', action='store_true')
    args = p.parse_args()
    if not args.validate_only:
        build(args)
    raise SystemExit(1 if validate(args.output) else 0)


if __name__ == '__main__':
    main()
