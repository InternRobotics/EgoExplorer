#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
visualize_session.py

同步采集数据的可视化工具（兼容 Wuji 与 Manus 采集会话）。

它做三件事：
  1. 读取一个会话目录（采集程序落盘的 data/session_YYYYMMDD_HHMMSS_TASK/），
     解析 GoPro 视频、Wuji/Manus 手套骨骼与 Wuji 触觉矩阵(.npz)、
     tracker 位姿(.tracker.npz)。
  2. 把手套骨骼、触觉矩阵与 tracker 位姿导出成 viewer_data.json，并生成一个
     自包含的 viewer.html（用 Three.js + Canvas 渲染）。
  3. 起一个支持 HTTP Range 的本地服务器（这样 MP4 可以随意拖动/seek），并自动打开浏览器。

页面里同时展示：
  · 左上：GoPro 视频（原生播放，作为“主时钟”）
  · 右上：3D 世界空间下各个 tracker 的位姿（每个 role 一个 RGB 坐标轴 gizmo + 名称 + 运动轨迹）
  · 手套区：左右手骨骼（21 节点 MediaPipe 拓扑，旧版 Manus 数据为 25 节点）以及 Wuji 24×31 触觉压力热力图
  · 底部：统一时间轴（播放/暂停/拖动/倍速）。各视图共用主机墙钟，逐帧对齐。

用法：
    # 可视化一个真实会话
    python visualize_session.py data/session_20250101_120000

    # 没有真实数据？先跑 demo（会合成一份真实格式的 npz 再打开页面）
    python visualize_session.py data/demo_session --demo
    python visualize_session.py data/manus_demo --demo --demo-type manus

    # 常用开关
    python visualize_session.py <session_dir> --port 8000     # 指定端口
    python visualize_session.py <session_dir> --no-open       # 不自动开浏览器
    python visualize_session.py <session_dir> --max-fps 50    # 骨骼/tracker 采样上限(减小 JSON)
    python visualize_session.py <session_dir> --build-only    # 只生成 viewer 文件，不起服务

依赖：
    pip install numpy
    （视频解码/播放由浏览器完成；Three.js 从 CDN 加载，首次打开需联网。离线用法见文末说明。）
"""
from __future__ import annotations

import argparse
import functools
import glob
import http.server
import json
import os
import re
import shutil
import socketserver
import threading
import time
import webbrowser
from datetime import datetime

import numpy as np


# ============================================================
# 两种驱动的原生骨架拓扑
# ============================================================
WUJI_HAND_PARENTS = [
    -1,
    0, 1, 2, 3,
    0, 5, 6, 7,
    0, 9, 10, 11,
    0, 13, 14, 15,
    0, 17, 18, 19,
]
# Manus 数据在采集端已规约为与 wuji 相同的 21 节点 (MediaPipe, 丢弃 4 个掌骨关节);
# 旧版 25 节点数据通过节点数单独识别 (见 _detect_glove_type)。
MANUS_HAND_PARENTS = [
    -1,
    0, 1, 2, 3,
    0, 5, 6, 7,
    0, 9, 10, 11,
    0, 13, 14, 15,
    0, 17, 18, 19,
]
# 旧版 Manus 标准层级 (25 节点, 含 4 个掌骨关节); 仅用于无层级元信息的旧文件兜底。
MANUS_25_HAND_PARENTS = [
    -1,
    0, 1, 2, 3,
    0, 5, 6, 7, 8,
    0, 10, 11, 12, 13,
    0, 15, 16, 17, 18,
    0, 20, 21, 22, 23,
]
FINGER_NAMES = ["thumb", "index", "middle", "ring", "pinky"]
GLOVE_TYPES = ("manus", "wuji")


# ============================================================
# 小工具
# ============================================================

def _clean_list(arr, decimals):
    """numpy 数组 -> 干净的嵌套 list：非有限值置 0，四舍五入到 decimals 位。"""
    a = np.asarray(arr, dtype=np.float64)
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    a = np.round(a, decimals)
    return a.tolist()


def _meta_of(npz):
    """从一个已 load 的 npz 里取 meta（JSON 字符串 -> dict）；没有则空 dict。"""
    if "meta" in npz.files:
        try:
            return json.loads(str(npz["meta"].item()))
        except Exception:
            try:
                return json.loads(str(npz["meta"]))
            except Exception:
                return {}
    return {}


def _normalize_glove_type(value):
    """把元数据中的设备类型规约为 ``manus`` / ``wuji`` / ``None``。"""
    text = str(value or "").strip().lower()
    if "manus" in text:
        return "manus"
    if "wuji" in text or "无际" in text:
        return "wuji"
    return None


def _detect_glove_type(meta, node_count=None, hint=None):
    """综合会话提示、NPZ 元数据和原生节点数判断手套类型。"""
    # 旧版 Manus 数据为 25 节点 (含 4 个掌骨关节), 可直接判定; 现行两种驱动
    # 都是 21 节点 (MediaPipe), 节点数不再能区分类型, 交给元数据与会话提示。
    if node_count == 25:
        return "manus"
    candidates = [
        hint,
        meta.get("glove_type"),
        meta.get("device_type"),
        meta.get("type"),
    ]
    for candidate in candidates:
        detected = _normalize_glove_type(candidate)
        if detected:
            return detected
    return "unknown"


def _normalize_skeleton(value):
    """把常见骨架布局规约为 ``(T, N, C>=3)``，无法识别时返回 ``None``。"""
    try:
        array = np.asarray(value)
        if array.dtype == object:
            array = np.stack([np.asarray(frame) for frame in array], axis=0)
        array = np.asarray(array, dtype=np.float64)
    except Exception:
        return None

    if array.ndim == 2 and array.shape[1] in (3, 7):
        array = array[np.newaxis, ...]
    elif array.ndim == 2:
        width = 7 if array.shape[1] % 7 == 0 else (
            3 if array.shape[1] % 3 == 0 else None)
        if width is None:
            return None
        array = array.reshape(array.shape[0], -1, width)
    if array.ndim != 3 or array.shape[0] == 0 or array.shape[1] == 0:
        return None
    if array.shape[2] < 3:
        return None
    return array


def _normalize_tactile(value):
    """把 Wuji 触觉流规约为 ``(T, 24, 31|32)``，无法识别时返回 ``None``。"""
    try:
        array = np.asarray(value)
        if array.dtype == object:
            array = np.stack([np.asarray(frame) for frame in array], axis=0)
        array = np.asarray(array, dtype=np.float64)
    except Exception:
        return None

    if array.ndim == 2 and array.shape[0] == 24 and array.shape[1] in (31, 32):
        array = array[np.newaxis, ...]
    elif array.ndim == 1:
        array = array[np.newaxis, ...]
    if array.ndim < 2 or array.shape[0] == 0:
        return None
    array = array.reshape(array.shape[0], -1)
    if array.shape[1] not in (24 * 31, 24 * 32):
        return None
    cols = array.shape[1] // 24
    array = np.nan_to_num(array, nan=-1.0, posinf=1.0, neginf=-1.0)
    array = np.where(array < 0, -1.0, np.clip(array, 0.0, 1.0))
    return array.reshape(array.shape[0], 24, cols)


def _validated_parents(parents, node_count, glove_type):
    """校验元数据拓扑；非法时回退到对应驱动的原生拓扑。"""
    if glove_type == "manus" and node_count == 25:
        # 旧版 Manus 文件 (含 4 个掌骨关节) 使用其原生 25 节点层级兜底。
        fallback = MANUS_25_HAND_PARENTS
    else:
        fallback = (MANUS_HAND_PARENTS if glove_type == "manus"
                    else WUJI_HAND_PARENTS if glove_type == "wuji" else None)
    try:
        result = [int(value) for value in parents]
    except (TypeError, ValueError):
        result = []
    # MANUS SDK 允许根节点的 parentId 指向自身；发送端转成数组索引后
    # 仍是自引用，这里统一规约成 viewer 使用的 -1 根节点。
    result = [-1 if parent == child else parent
              for child, parent in enumerate(result)]
    valid = len(result) == node_count and result.count(-1) == 1
    if valid:
        valid = all(-1 <= parent < node_count and parent != child
                    for child, parent in enumerate(result))
    if valid:
        for child in range(node_count):
            seen = set()
            current = child
            while current >= 0:
                if current in seen:
                    valid = False
                    break
                seen.add(current)
                current = result[current]
            if not valid:
                break
    if valid:
        return result
    if fallback is not None and len(fallback) == node_count:
        return list(fallback)
    return [-1] * node_count


def _stream_time_axis(npz, device, stream, frame_count):
    """读取同步轴，优先主机墙钟，保证 Wuji 能与视频/Tracker 对齐。"""
    candidates = (
        (f"{device}__{stream}__t_host_us", "host"),
        (f"{device}__{stream}__t_us", "primary"),
        (f"{device}__{stream}__t_dev_us", "device"),
    )
    for key, source in candidates:
        if key not in npz.files:
            continue
        try:
            values = np.asarray(npz[key], dtype=np.int64).reshape(-1)
        except Exception:
            continue
        if values.size:
            count = min(frame_count, values.size)
            return values[:count], count, source
    # 旧文件没有时间戳时按 60 fps 构造相对微秒轴，至少保证播放速度合理。
    return (np.arange(frame_count, dtype=np.int64) * 16667,
            frame_count, "index")


def _infer_slot(device, glove_meta):
    """从显式 slot、逻辑设备名或 Wuji SN 推断左右手。"""
    text = str(glove_meta.get("slot") or "").lower()
    if "left" in text or "左" in text:
        return "left"
    if "right" in text or "右" in text:
        return "right"
    name = str(device).lower()
    if name.endswith("_0") or name == "left":
        return "left"
    if name.endswith("_1") or name == "right":
        return "right"
    serial = str(glove_meta.get("sn") or "").upper()
    if len(serial) > 3 and serial[3] == "J":
        return "left"
    if len(serial) > 3 and serial[3] == "K":
        return "right"
    return ""


def _device_streams(npz, device, meta):
    """列出该设备实际落盘的数据流，用于页面说明而不假设设备能力。"""
    channels = meta.get("channels") or {}
    streams = {
        str(key).split("/", 1)[1]
        for key in channels
        if str(key).startswith(f"{device}/") and "/" in str(key)
    }
    prefix = f"{device}__"
    for key in npz.files:
        if not key.startswith(prefix) or key.endswith(
                ("__t_us", "__t_host_us", "__t_dev_us", "__counts")):
            continue
        stream = key[len(prefix):].split(".", 1)[0]
        if stream:
            streams.add(stream)
    return sorted(streams)


def _finger_ids(parents):
    """每个节点属于哪根手指：手腕(根)=-1，其余=其“手指基节”(父为根的祖先)的出现序号。

    对任意层级都成立，不依赖固定节点数量。
    """
    n = len(parents)
    root = next((i for i, p in enumerate(parents) if p == -1), 0)
    bases = [i for i, p in enumerate(parents) if p == root and i != root]
    base_order = {b: k for k, b in enumerate(bases)}
    fid = [-1] * n
    for i in range(n):
        if i == root:
            continue
        j, guard = i, 0
        while j != -1 and parents[j] != root and guard < n:
            j = parents[j]
            guard += 1
        fid[i] = base_order.get(j, -1)
    return fid


def _node_labels(parents, fids):
    """给每个节点起一个可读名：根=wrist；末端(无子)=<finger>_tip；其余=<finger>_<深度>。"""
    n = len(parents)
    root = next((i for i, p in enumerate(parents) if p == -1), 0)
    children = {i: [] for i in range(n)}
    for c, p in enumerate(parents):
        if p >= 0:
            children[p].append(c)
    # 深度（到本手指基节的步数）
    depth = [0] * n
    for i in range(n):
        j, d = i, 0
        while parents[j] != -1 and parents[j] != root and d < n:
            j = parents[j]
            d += 1
        depth[i] = d
    labels = []
    for i in range(n):
        if i == root:
            labels.append("wrist")
            continue
        fid = fids[i]
        fname = FINGER_NAMES[fid] if 0 <= fid < len(FINGER_NAMES) else "f%d" % fid
        if not children[i]:
            labels.append("%s_tip" % fname)
        else:
            labels.append("%s_%d" % (fname, depth[i]))
    return labels


def _decimate_indices(t_us, max_fps):
    """按目标帧率对时间轴抽稀，返回保留的索引；始终包含首尾。"""
    t = np.asarray(t_us, dtype=np.float64)
    n = len(t)
    if n <= 2 or not max_fps or max_fps <= 0:
        return np.arange(n)
    dur = (t[-1] - t[0]) / 1e6
    if dur <= 0:
        return np.arange(n)
    cur_fps = (n - 1) / dur
    if cur_fps <= max_fps:
        return np.arange(n)
    step = max(1, int(round(cur_fps / max_fps)))
    idx = np.arange(0, n, step)
    if idx[-1] != n - 1:
        idx = np.append(idx, n - 1)
    return idx


# ============================================================
# 读取会话各部分
# ============================================================

def _read_gloves(npz_path, max_fps, glove_type_hint=None):
    """读取 Manus/Wuji 手套 NPZ，统一输出页面所需的原生骨架结构。"""
    out = {}
    if not npz_path or not os.path.exists(npz_path):
        return out
    with np.load(npz_path, allow_pickle=True) as npz:
        meta = _meta_of(npz)
        raw_gloves = meta.get("gloves") or []
        glove_meta = {
            str(item.get("device_name")): item
            for item in raw_gloves if isinstance(item, dict)
            and item.get("device_name") is not None
        }

        skel_keys = [key for key in npz.files
                     if key.endswith("__hand_skeleton")]
        for key in sorted(skel_keys):
            device = key[:-len("__hand_skeleton")]
            skeleton = _normalize_skeleton(npz[key])
            if skeleton is None:
                continue
            frame_count, node_count = skeleton.shape[:2]
            glove_type = _detect_glove_type(
                meta, node_count=node_count, hint=glove_type_hint)
            gm = glove_meta.get(device, {})
            parents = _validated_parents(
                gm.get("hand_hierarchy"), node_count, glove_type)
            finger_ids = _finger_ids(parents)
            labels = _node_labels(parents, finger_ids)

            t_us, count, time_source = _stream_time_axis(
                npz, device, "hand_skeleton", frame_count)
            skeleton = skeleton[:count]
            order = np.argsort(t_us, kind="stable")
            t_us = t_us[order]
            skeleton = skeleton[order]
            indices = _decimate_indices(t_us, max_fps)
            positions = skeleton[indices, :, :3]

            glove_out = {
                "type": glove_type,
                "slot": _infer_slot(device, gm),
                "serial": gm.get("sn"),
                "nodeCount": int(node_count),
                "parents": parents,
                "fingerOf": finger_ids,
                "labels": labels,
                "streams": _device_streams(npz, device, meta),
                "timeSource": time_source,
                "timeAbsolute": bool(
                    time_source == "host" or (t_us.size and t_us[0] > 10**14)),
                "t_us": t_us[indices].astype(np.int64).tolist(),
                "positions": _clean_list(positions, 4),
            }

            # Wuji 的触觉流与骨骼流有独立帧率和时间轴，必须分别抽稀、对齐。
            tactile_key = f"{device}__tactile"
            if tactile_key in npz.files:
                tactile = _normalize_tactile(npz[tactile_key])
                if tactile is not None:
                    tactile_t_us, tactile_count, tactile_time_source = \
                        _stream_time_axis(npz, device, "tactile", len(tactile))
                    tactile = tactile[:tactile_count]
                    tactile_order = np.argsort(tactile_t_us, kind="stable")
                    tactile_t_us = tactile_t_us[tactile_order]
                    tactile = tactile[tactile_order]
                    tactile_indices = _decimate_indices(tactile_t_us, max_fps)
                    glove_out["tactile"] = {
                        "rows": 24,
                        "cols": int(tactile.shape[2]),
                        "timeSource": tactile_time_source,
                        "timeAbsolute": bool(
                            tactile_time_source == "host" or (
                                tactile_t_us.size and tactile_t_us[0] > 10**14)),
                        "t_us": tactile_t_us[tactile_indices].astype(
                            np.int64).tolist(),
                        "values": _clean_list(
                            tactile[tactile_indices].reshape(
                                len(tactile_indices), -1), 3),
                    }

            out[device] = glove_out
    return out


def _read_trackers(npz_path, max_fps):
    """读 tracker .npz，返回 {device, t_us, roles:{role:{position,orientation,valid}}} 或 None。"""
    if not npz_path or not os.path.exists(npz_path):
        return None
    with np.load(npz_path, allow_pickle=True) as npz:
        pattern = re.compile(r"^(.+)__pose\.(.+)\.position$")
        matches = [pattern.match(key) for key in npz.files]
        matches = [match for match in matches if match]
        if not matches:
            return None
        device = matches[0].group(1)
        roles = sorted({match.group(2) for match in matches
                        if match.group(1) == device})
        if not roles:
            return None

        usable_roles = []
        frame_lengths = []
        for role in roles:
            pos_key = f"{device}__pose.{role}.position"
            quat_key = f"{device}__pose.{role}.orientation"
            if quat_key not in npz.files:
                continue
            pos_len = np.asarray(npz[pos_key]).shape[0]
            quat_len = np.asarray(npz[quat_key]).shape[0]
            lengths = [pos_len, quat_len]
            valid_key = f"{device}__pose.{role}.valid"
            if valid_key in npz.files:
                lengths.append(np.asarray(npz[valid_key]).reshape(-1).size)
            frame_lengths.append(min(lengths))
            usable_roles.append(role)
        if not usable_roles or not frame_lengths:
            return None
        roles = usable_roles
        frame_count = min(frame_lengths)
        t_us, count, time_source = _stream_time_axis(
            npz, device, "pose", frame_count)
        order = np.argsort(t_us, kind="stable")
        t_us = t_us[order]
        indices = _decimate_indices(t_us, max_fps)

        role_out = {}
        for role in roles:
            pos_key = f"{device}__pose.{role}.position"
            quat_key = f"{device}__pose.{role}.orientation"
            positions = np.asarray(npz[pos_key], dtype=np.float64)[:count][order]
            quaternions = np.asarray(npz[quat_key], dtype=np.float64)[:count][order]
            usable = min(len(t_us), len(positions), len(quaternions))
            selected = indices[indices < usable]
            positions = positions[selected]
            quaternions = quaternions[selected]
            valid_key = f"{device}__pose.{role}.valid"
            if valid_key in npz.files:
                valid_all = np.asarray(npz[valid_key]).astype(bool)[:count][order]
                valid = valid_all[selected]
            else:
                valid = np.isfinite(positions).all(axis=1)
            valid = (valid & np.isfinite(positions).all(axis=1)
                     & np.isfinite(quaternions).all(axis=1))
            role_out[role] = {
                "position": _clean_list(positions, 4),
                "orientation": _clean_list(quaternions, 5),
                "valid": valid.tolist(),
            }
        return {
            "device": device,
            "timeSource": time_source,
            "timeAbsolute": bool(
                time_source == "host" or (t_us.size and t_us[0] > 10**14)),
            "t_us": t_us[indices].astype(np.int64).tolist(),
            "roles": role_out,
        }


def _read_video(video_npz_path, session_dir):
    """读 .video.npz，返回 {src, t0_us, t1_us, fps, width, height} 或 None。"""
    if not video_npz_path or not os.path.exists(video_npz_path):
        return None
    with np.load(video_npz_path, allow_pickle=True) as npz:
        t = (np.asarray(npz["t_host_us"], dtype=np.int64).reshape(-1)
             if "t_host_us" in npz.files else np.empty(0, np.int64))
        if t.size == 0:
            return None
        t = np.sort(t)
        mp4_name = str(npz["mp4"].item()) if "mp4" in npz.files else None
        fps = float(npz["fps"]) if "fps" in npz.files else 0.0
        width = int(npz["width"]) if "width" in npz.files else 0
        height = int(npz["height"]) if "height" in npz.files else 0
    if not mp4_name or not os.path.exists(os.path.join(session_dir, mp4_name)):
        cands = sorted(glob.glob(os.path.join(session_dir, "*.mp4"))) or \
                sorted(glob.glob(os.path.join(session_dir, "*.avi")))
        mp4_name = os.path.basename(cands[0]) if cands else None
    if not mp4_name:
        return None
    return {"src": mp4_name, "t0_us": int(t[0]), "t1_us": int(t[-1]),
            "fps": fps, "width": width, "height": height,
            "timeSource": "host"}


def _find_session_files(session_dir):
    """在会话目录里定位 gloves/tracker/video 三类文件（优先用 session.json，其次按 npz 键判别）。"""
    base = os.path.basename(os.path.normpath(session_dir))
    files = {"gloves": None, "tracker": None, "video": None, "base": base,
             "created": None, "glove_type": None, "clock": None,
             "session_meta": None}

    sj = glob.glob(os.path.join(session_dir, "*.session.json"))
    if sj:
        try:
            with open(sj[0], "r", encoding="utf-8") as fp:
                meta = json.load(fp)
            files["base"] = meta.get("base", base)
            files["created"] = meta.get("created")
            files["overlap_s"] = meta.get("overlap_s")
            files["glove_type"] = _normalize_glove_type(meta.get("glove_type"))
            files["clock"] = meta.get("clock")
            files["session_meta"] = os.path.basename(sj[0])
            if meta.get("gloves_npz"):
                files["gloves"] = os.path.join(session_dir, meta["gloves_npz"])
            if meta.get("tracker_npz"):
                files["tracker"] = os.path.join(session_dir, meta["tracker_npz"])
            v = meta.get("video") or {}
            if v.get("timestamps_npz"):
                files["video"] = os.path.join(session_dir, v["timestamps_npz"])
        except Exception:
            pass

    # session.json 可能引用已移动或缺失的文件；这种情况也进入内容兜底。
    for kind in ("gloves", "tracker", "video"):
        if files[kind] and not os.path.isfile(files[kind]):
            files[kind] = None

    # 兜底 / 补全：按文件名与内容判别
    if files["tracker"] is None:
        for p in glob.glob(os.path.join(session_dir, "*.tracker.npz")):
            files["tracker"] = p
            break
    if files["video"] is None:
        for p in glob.glob(os.path.join(session_dir, "*.video.npz")):
            files["video"] = p
            break
    if files["gloves"] is None:
        for p in sorted(glob.glob(os.path.join(session_dir, "*.npz"))):
            if p.endswith(".tracker.npz") or p.endswith(".video.npz"):
                continue
            try:
                with np.load(p, allow_pickle=True) as npz:
                    if any(k.endswith("__hand_skeleton") for k in npz.files):
                        files["gloves"] = p
                        break
            except Exception:
                continue
    return files


def build_payload(session_dir, max_fps=60):
    """读取整个会话，组装成 viewer_data.json 的内容。"""
    session_dir = os.path.normpath(session_dir)
    if not os.path.isdir(session_dir):
        raise SystemExit("找不到会话目录: %s" % session_dir)

    files = _find_session_files(session_dir)
    gloves = _read_gloves(
        files["gloves"], max_fps, glove_type_hint=files.get("glove_type"))
    trackers = _read_trackers(files["tracker"], max_fps)
    video = _read_video(files["video"], session_dir)

    if not gloves and not trackers and not video:
        raise SystemExit(
            "会话目录里没找到可视化的数据 (手套 .npz / .tracker.npz / .video.npz)。\n"
            "  目录: %s\n  提示: 用 --demo 生成一份示例数据先看看效果。" % session_dir)

    # 全局时间范围（并集）。index 是缺少真实时间戳时的降级轴，不与墙钟混算。
    absolute_ranges, relative_ranges = [], []
    for g in gloves.values():
        streams = [g]
        if g.get("tactile"):
            streams.append(g["tactile"])
        for stream in streams:
            if stream["t_us"]:
                target = (absolute_ranges if stream["timeAbsolute"]
                          else relative_ranges)
                target.append((stream["t_us"][0], stream["t_us"][-1]))
    if trackers and trackers["t_us"]:
        target = absolute_ranges if trackers["timeAbsolute"] else relative_ranges
        target.append((trackers["t_us"][0], trackers["t_us"][-1]))
    if video:
        absolute_ranges.append((video["t0_us"], video["t1_us"]))

    selected_ranges = absolute_ranges or relative_ranges
    time_range = ({
        "t0_us": int(min(start for start, _ in selected_ranges)),
        "t1_us": int(max(end for _, end in selected_ranges)),
    } if selected_ranges else {"t0_us": 0, "t1_us": 0})

    # 单文件旧录制若缺少时间戳，平移到当前主轴起点，仍可独立播放。
    if absolute_ranges:
        for glove in gloves.values():
            if not glove["timeAbsolute"] and glove["t_us"]:
                offset = time_range["t0_us"] - glove["t_us"][0]
                glove["t_us"] = [int(value + offset) for value in glove["t_us"]]
            tactile = glove.get("tactile")
            if tactile and not tactile["timeAbsolute"] and tactile["t_us"]:
                offset = time_range["t0_us"] - tactile["t_us"][0]
                tactile["t_us"] = [
                    int(value + offset) for value in tactile["t_us"]]
        if trackers and not trackers["timeAbsolute"] and trackers["t_us"]:
            offset = time_range["t0_us"] - trackers["t_us"][0]
            trackers["t_us"] = [int(value + offset) for value in trackers["t_us"]]
        shifted_ends = [glove["t_us"][-1] for glove in gloves.values()
                        if glove["t_us"]]
        shifted_ends.extend(
            glove["tactile"]["t_us"][-1] for glove in gloves.values()
            if glove.get("tactile") and glove["tactile"]["t_us"])
        if trackers and trackers["t_us"]:
            shifted_ends.append(trackers["t_us"][-1])
        if shifted_ends:
            time_range["t1_us"] = max(time_range["t1_us"], max(shifted_ends))

    detected_types = sorted({glove["type"] for glove in gloves.values()
                             if glove["type"] != "unknown"})
    session_glove_type = (
        detected_types[0] if len(detected_types) == 1
        else "mixed" if detected_types else files.get("glove_type") or "unknown")

    payload = {
        "schemaVersion": 3,
        "base": files["base"],
        "created": files.get("created"),
        "overlap_s": files.get("overlap_s"),
        "gloveType": session_glove_type,
        "clock": files.get("clock"),
        "sources": {
            "session": files.get("session_meta"),
            "gloves": os.path.basename(files["gloves"]) if files["gloves"] else None,
            "tracker": os.path.basename(files["tracker"]) if files["tracker"] else None,
            "video": os.path.basename(files["video"]) if files["video"] else None,
        },
        "time": time_range,
        "video": video,
        "gloves": gloves,
        "trackers": trackers,
    }
    return payload


# ============================================================
# 生成 demo 数据（写成 record_session 相同键约定的真实 .npz）
# ============================================================

def _demo_hand(T, t_us, mirror=False, glove_type="wuji"):
    """按指定驱动的原生拓扑生成会攥拳/张开的示例手。"""
    parents = (MANUS_HAND_PARENTS if glove_type == "manus"
               else WUJI_HAND_PARENTS)
    P = parents
    N = len(P)
    seg = 0.028                                  # 每段骨长(米)
    bases = [index for index, parent in enumerate(P) if parent == 0]
    base_x = dict(zip(bases, (-0.030, -0.015, 0.0, 0.016, 0.032)))
    out = np.zeros((T, N, 7), np.float32)
    out[:, :, 6] = 1.0                           # 四元数 w=1(单位)
    for ti in range(T):
        phase = 0.5 - 0.5 * np.cos(2 * np.pi * ti / max(1, T - 1) * 2)  # 0->1 攥拳程度
        pos = np.zeros((N, 3), np.float32)
        # 五指基节
        for fbase, bx in base_x.items():
            pos[fbase] = [bx, 0.0, 0.03]
        # 沿链向指尖推进，逐关节增加弯曲
        order = [i for i in range(1, N) if i not in base_x]
        for c in order:
            p = P[c]
            depth_bend = phase * (0.6 + 0.18 * (c % 5))
            ang = depth_bend
            direction = np.array([0.0, -np.sin(ang), np.cos(ang)], np.float32)
            pos[c] = pos[p] + direction * seg
        if mirror:
            pos[:, 0] = -pos[:, 0]
        out[ti, :, 0:3] = pos
    return out


def make_demo(session_dir, glove_type="wuji"):
    """生成指定 Manus/Wuji 原生格式的示例会话。"""
    if glove_type not in GLOVE_TYPES:
        raise ValueError("demo glove_type 必须是 manus 或 wuji")
    os.makedirs(session_dir, exist_ok=True)
    base = os.path.basename(os.path.normpath(session_dir))
    T = 360
    t0 = int(time.time() * 1e6)
    dt = int(1e6 / 60)
    t_us = (t0 + np.arange(T) * dt).astype(np.int64)

    # ---- 手套（左右手合并到一个 npz） ----
    arrays = {}
    hierarchy = list(MANUS_HAND_PARENTS if glove_type == "manus"
                     else WUJI_HAND_PARENTS)
    for dev, mirror in (("glove_0", False), ("glove_1", True)):
        skel = _demo_hand(T, t_us, mirror=mirror, glove_type=glove_type)
        arrays["%s__hand_skeleton" % dev] = skel
        arrays["%s__hand_skeleton__t_us" % dev] = t_us
        arrays["%s__hand_skeleton__t_dev_us" % dev] = np.full(T, -1, np.int64)
        arrays["%s__hand_skeleton__t_host_us" % dev] = t_us
        if glove_type == "wuji":
            yy, xx = np.mgrid[0:24, 0:31]
            tactile = np.empty((T, 24, 31), np.float32)
            side_phase = np.pi if mirror else 0.0
            for ti in range(T):
                phase = 2 * np.pi * ti / max(1, T - 1) * 2 + side_phase
                cx, cy = 15 + 8 * np.sin(phase), 12 + 6 * np.cos(phase * .7)
                tactile[ti] = np.exp(
                    -((xx - cx) ** 2 + (yy - cy) ** 2) / 28.0) * (
                        .35 + .65 * (np.sin(phase * .5) ** 2))
            arrays["%s__tactile" % dev] = tactile.reshape(T, -1)
            arrays["%s__tactile__t_us" % dev] = t_us
            arrays["%s__tactile__t_dev_us" % dev] = np.full(T, -1, np.int64)
            arrays["%s__tactile__t_host_us" % dev] = t_us
    meta = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "device_type": f"{glove_type}_glove",
        "streams": (["hand_skeleton", "tactile"]
                    if glove_type == "wuji" else ["hand_skeleton"]),
        "gloves": [
            {"device_name": "glove_0", "sn": "DEMO-L", "slot": "left", "hand_hierarchy": hierarchy},
            {"device_name": "glove_1", "sn": "DEMO-R", "slot": "right", "hand_hierarchy": hierarchy},
        ],
        "note": "demo",
    }
    arrays["meta"] = np.array(json.dumps(meta, ensure_ascii=False))
    gloves_npz = os.path.join(session_dir, base + ".npz")
    np.savez_compressed(gloves_npz, **arrays)

    # ---- tracker（几个 role 在房间里运动） ----
    dev = "tracker_0"
    tarr = {"%s__pose__t_us" % dev: t_us,
            "%s__pose__t_dev_us" % dev: np.full(T, -1, np.int64),
            "%s__pose__t_host_us" % dev: t_us}
    phase = 2 * np.pi * np.arange(T) / (T - 1)

    def yaw_quat(a):
        return np.stack([np.cos(a / 2), np.zeros_like(a), np.sin(a / 2), np.zeros_like(a)], axis=1)

    roles = {
        "waist":      (np.stack([0.10 * np.sin(phase), np.full(T, 1.00), 0.10 * np.cos(phase)], 1), yaw_quat(phase)),
        "chest":      (np.stack([0.06 * np.sin(phase), np.full(T, 1.30), 0.05 * np.cos(phase)], 1), yaw_quat(0.5 * phase)),
        "left_foot":  (np.stack([np.full(T, -0.15), 0.05 + 0.05 * np.abs(np.sin(phase)), 0.20 * np.sin(phase)], 1), yaw_quat(0 * phase)),
        "right_foot": (np.stack([np.full(T, 0.15), 0.05 + 0.05 * np.abs(np.cos(phase)), 0.20 * np.cos(phase)], 1), yaw_quat(0 * phase)),
    }
    all_roles = []
    for r, (pos, quat) in roles.items():
        tarr["%s__pose.%s.position" % (dev, r)] = pos.astype(np.float32)
        tarr["%s__pose.%s.orientation" % (dev, r)] = quat.astype(np.float32)
        tarr["%s__pose.%s.valid" % (dev, r)] = np.ones(T, bool)
        all_roles.append(r)
    tmeta = {"created": datetime.now().isoformat(timespec="seconds"), "type": "tracker",
             "device": dev, "streams": ["pose"], "roles": sorted(all_roles),
             "frames": T, "ts_unit": "us"}
    tarr["meta"] = np.array(json.dumps(tmeta, ensure_ascii=False))
    tracker_npz = os.path.join(session_dir, base + ".tracker.npz")
    np.savez_compressed(tracker_npz, **tarr)

    # ---- session.json（无视频） ----
    sj = {
        "base": base,
        "created": datetime.now().isoformat(timespec="seconds"),
        "clock": "demo (host wall clock, µs)",
        "glove_type": glove_type,
        "video": {"mp4": None, "timestamps_npz": None, "frames": 0, "fps": None, "size": None,
                  "t_host_us_range": None},
        "gloves_npz": os.path.basename(gloves_npz),
        "gloves": meta["gloves"],
        "glove_t_host_us_range": [int(t_us[0]), int(t_us[-1])],
        "tracker_npz": os.path.basename(tracker_npz),
        "tracker": {"device": dev, "type": "tracker", "frames": T, "roles": sorted(all_roles)},
        "tracker_t_host_us_range": [int(t_us[0]), int(t_us[-1])],
        "overlap_s": None,
    }
    with open(os.path.join(session_dir, base + ".session.json"), "w", encoding="utf-8") as fp:
        json.dump(sj, fp, ensure_ascii=False, indent=2)

    tactile_text = " + 触觉矩阵" if glove_type == "wuji" else ""
    print("[demo] 已在 %s 生成 %s 示例会话 "
          "(无视频, 有左右手骨骼%s + 4 个 tracker role)"
          % (session_dir, glove_type.upper(), tactile_text))


# ============================================================
# 前端页面（自包含；数据经 viewer_data.json 用 fetch 载入，避免任何字符串转义问题）
# ============================================================
VIEWER_HTML = r"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>动捕会话可视化</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
<style>
  :root{
    --bg:#0e1116; --panel:#141a22; --panel-2:#0f141b; --line:#26313f;
    --ink:#d6dee8; --dim:#7d8a9c; --faint:#4b5563;
    --accent:#a78bfa; --accent-2:#c4b5fd; --live:#ff5d5d;
    --x:#ff5f56; --y:#7ddc7d; --z:#5aa2ff;
    --mono: ui-monospace,SFMono-Regular,"SF Mono",Menlo,Consolas,"Liberation Mono",monospace;
    --sans: ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,"PingFang SC","Microsoft YaHei",sans-serif;
  }
  *{box-sizing:border-box}
  html,body{height:100%}
  body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--sans);
       display:flex;flex-direction:column;overflow:hidden}
  a{color:var(--accent)}

  header{display:flex;align-items:baseline;gap:16px;padding:10px 16px;border-bottom:1px solid var(--line);
         background:linear-gradient(180deg,#121821,#0f141b)}
  header h1{font-size:14px;font-weight:600;letter-spacing:.02em;margin:0}
  header .base{font-family:var(--mono);font-size:12px;color:var(--accent-2)}
  header .spacer{flex:1}
  header .stat{font-family:var(--mono);font-size:11px;color:var(--dim)}
  header .stat b{color:var(--ink);font-weight:600}

  main{flex:1;display:grid;gap:10px;padding:10px;min-height:0;
       grid-template-columns:1fr 1fr;grid-template-rows:minmax(0,1.15fr) minmax(0,1fr);
       grid-template-areas:"video trackers" "gloves gloves";}
  main.with-tactile{grid-template-columns:minmax(0,1.35fr) minmax(520px,1fr);
       grid-template-rows:minmax(260px,.85fr) minmax(540px,1.7fr);
       grid-template-areas:"video trackers" "video gloves";overflow:auto}
  .panel{position:relative;background:var(--panel);border:1px solid var(--line);border-radius:10px;
         overflow:hidden;min-height:0;min-width:0}
  #panel-video{grid-area:video}
  #panel-trackers{grid-area:trackers}
  #panel-gloves{grid-area:gloves;padding:0}
  .glove-panel-tag{display:none}
  .glove-grid{display:grid;grid-template-columns:1fr 1fr;gap:8px;width:100%;height:100%;min-height:0}
  .glove-pair{min-width:0;min-height:0}
  #panel-gloves .hand{position:relative;width:100%;height:100%;min-width:0;min-height:0;
        background:var(--panel);border-radius:8px;overflow:hidden}
  .tactile-hand{display:none;grid-template-rows:auto minmax(0,1fr);gap:6px;min-width:0;min-height:0;
                padding:7px;border:1px solid var(--line);border-radius:8px;background:var(--panel-2)}
  main.with-tactile #panel-gloves{padding:34px 10px 10px}
  main.with-tactile .glove-panel-tag{display:block}
  main.with-tactile .glove-grid{grid-template-columns:1fr;grid-template-rows:repeat(2,minmax(240px,1fr));gap:10px}
  main.with-tactile .glove-pair{display:grid;grid-template-columns:minmax(190px,.8fr) minmax(280px,1.2fr);
                               gap:10px}
  main.with-tactile .hand{border:1px solid var(--line);background:var(--panel-2)}
  main.with-tactile .tactile-hand{display:grid}
  .tactile-head{display:flex;align-items:center;gap:10px;font:11px var(--mono);color:var(--dim)}
  .tactile-head b{color:var(--accent-2);font-weight:600}
  .tactile-head .tactile-stats{margin-left:auto;white-space:nowrap}
  .tactile-canvas-wrap{position:relative;min-width:0;min-height:0;height:100%;width:auto;max-width:100%;
                       aspect-ratio:31 / 24;justify-self:center;display:flex;align-items:center;
                       justify-content:center;overflow:hidden;border-radius:6px;background:#0a0e14}
  .panel .tactile-canvas{display:block;width:100%;height:100%;image-rendering:pixelated}
  .tactile-empty{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;
                 color:var(--dim);font:11px var(--mono);pointer-events:none;text-align:center;padding:12px}
  .tactile-scale{position:absolute;right:8px;bottom:7px;display:flex;align-items:center;gap:6px;
                 color:#d6dee8;font:9px var(--mono);text-shadow:0 1px 2px #000;pointer-events:none}
  .tactile-scale i{display:block;width:90px;height:7px;border-radius:4px;
                    background:linear-gradient(90deg,#102641,#0969a8,#13c8a3,#f6d743,#f04438)}

  .panel .tag{position:absolute;top:8px;left:10px;z-index:5;font-family:var(--mono);font-size:11px;
              letter-spacing:.04em;color:var(--dim);text-transform:uppercase;pointer-events:none}
  .panel .tag b{color:var(--accent-2);font-weight:600}
  .panel canvas{display:block;width:100%;height:100%}

  #panel-video{background:#000;display:flex;align-items:center;justify-content:center}
  video{max-width:100%;max-height:100%;display:block;background:#000}
  .novideo{color:var(--dim);font-family:var(--mono);font-size:12px;text-align:center;padding:24px;line-height:1.7}
  .novideo .big{color:var(--ink);font-size:13px}

  /* 图例 / 读数 */
  .legend{position:absolute;z-index:5;font-family:var(--mono);font-size:11px;line-height:1.5;
          background:rgba(10,14,20,.72);border:1px solid var(--line);border-radius:8px;padding:8px 10px;
          backdrop-filter:blur(3px)}
  #tracker-legend{top:8px;right:10px;max-width:46%;max-height:calc(100% - 16px);overflow:auto}
  #tracker-legend .row{display:flex;gap:8px;white-space:nowrap;align-items:center}
  #tracker-legend .name{color:var(--ink);min-width:92px}
  #tracker-legend .xyz{color:var(--dim)}
  #tracker-legend .xyz .off{color:var(--faint)}
  #axis-legend{bottom:8px;left:10px;color:var(--dim);display:flex;gap:12px}
  #axis-legend .ax{display:flex;gap:5px;align-items:center}
  #axis-legend i{width:14px;height:2px;display:inline-block;border-radius:2px}

  .glove-legend{top:8px;right:10px;display:flex;gap:12px;flex-wrap:wrap;
                justify-content:flex-end;max-width:calc(100% - 96px)}
  .glove-legend .chip{display:flex;gap:6px;align-items:center;color:var(--dim)}
  .glove-legend .dot{width:9px;height:9px;border-radius:50%}
  .empty-hint{position:absolute;inset:0;display:none;align-items:center;justify-content:center;
              color:var(--dim);font-family:var(--mono);font-size:12px;text-align:center;padding:24px}

  /* 底部走带 */
  .transport{display:flex;align-items:center;gap:14px;padding:10px 16px;border-top:1px solid var(--line);
             background:linear-gradient(0deg,#121821,#0f141b)}
  button.play{width:40px;height:40px;border-radius:50%;border:1px solid var(--line);background:var(--panel);
              color:var(--ink);cursor:pointer;font-size:15px;display:flex;align-items:center;justify-content:center;
              transition:border-color .15s,transform .05s}
  button.play:hover{border-color:var(--accent)}
  button.play:active{transform:scale(.94)}
  .tc{font-family:var(--mono);font-size:13px;color:var(--ink);white-space:nowrap;min-width:112px;text-align:center}
  .tc .sep{color:var(--faint)}
  .scrub-wrap{flex:1;display:flex;align-items:center}
  input[type=range].scrub{-webkit-appearance:none;appearance:none;width:100%;height:6px;border-radius:3px;
     background:linear-gradient(90deg,var(--accent) 0%,var(--accent) var(--pct,0%),#222b36 var(--pct,0%),#222b36 100%);
     outline:none;cursor:pointer}
  input[type=range].scrub::-webkit-slider-thumb{-webkit-appearance:none;width:15px;height:15px;border-radius:50%;
     background:var(--accent-2);border:2px solid #0e1116;box-shadow:0 0 0 1px var(--accent)}
  input[type=range].scrub::-moz-range-thumb{width:13px;height:13px;border-radius:50%;background:var(--accent-2);
     border:2px solid #0e1116}
  .ctl{display:flex;align-items:center;gap:7px;font-size:12px;color:var(--dim)}
  .ctl select,.ctl input[type=range]{accent-color:var(--accent)}
  .ctl select{background:var(--panel-2);color:var(--ink);border:1px solid var(--line);border-radius:6px;
     padding:3px 6px;font-family:var(--mono);font-size:12px}
  .toggles{display:flex;align-items:center;gap:14px;flex-wrap:wrap}
  .toggles label{display:flex;align-items:center;gap:5px;cursor:pointer;user-select:none}
  .toggles input{accent-color:var(--accent);cursor:pointer}
  .hint{font-family:var(--mono);font-size:10px;color:var(--faint)}

  #err{position:fixed;inset:0;background:rgba(8,10,14,.94);color:var(--ink);display:none;
       align-items:center;justify-content:center;z-index:99;padding:32px}
  #err .box{max-width:560px;font-family:var(--mono);font-size:13px;line-height:1.8;
       border:1px solid var(--line);border-radius:10px;padding:22px 26px;background:var(--panel)}
  #err h2{margin:0 0 10px;font-size:15px;color:var(--live)}

  @media (max-width: 860px){
    main{grid-template-columns:1fr;grid-template-rows:auto auto minmax(340px,auto);
         grid-template-areas:"video" "trackers" "gloves";overflow:auto}
    main.with-tactile{grid-template-columns:1fr;grid-template-rows:auto auto minmax(560px,auto);
         grid-template-areas:"video" "trackers" "gloves"}
    .panel{min-height:340px}
    #panel-gloves{min-height:340px}
    main.with-tactile #panel-gloves{min-height:560px}
    .transport{flex-wrap:wrap;gap:10px}
    .scrub-wrap{order:5;flex-basis:100%}
  }
  @media (max-width: 620px){
    main.with-tactile{grid-template-rows:auto auto auto}
    main.with-tactile #panel-gloves{min-height:1040px}
    main.with-tactile .glove-grid{grid-template-rows:repeat(2,minmax(490px,auto))}
    main.with-tactile .glove-pair{grid-template-columns:1fr;grid-template-rows:230px minmax(240px,auto)}
  }
</style>
</head>
<body>
  <header>
    <h1>动捕会话可视化</h1>
    <span class="base" id="h-base">—</span>
    <span class="spacer"></span>
    <span class="stat" id="h-stats"></span>
  </header>

  <main>
    <section class="panel" id="panel-video">
      <div class="tag"><b>视频</b> · GoPro</div>
      <video id="video" playsinline></video>
      <div class="novideo" id="novideo" style="display:none">
        <div class="big">本会话没有视频</div>
        <div>骨骼与 tracker 由内置虚拟时钟驱动，可正常播放。</div>
      </div>
    </section>

    <section class="panel" id="panel-trackers">
      <div class="tag"><b>Tracker</b> · 世界空间 (m)</div>
      <div class="legend" id="tracker-legend"></div>
      <div class="legend" id="axis-legend">
        <span class="ax"><i style="background:var(--x)"></i>X</span>
        <span class="ax"><i style="background:var(--y)"></i>Y↑</span>
        <span class="ax"><i style="background:var(--z)"></i>Z</span>
      </div>
    </section>

    <section class="panel" id="panel-gloves">
      <div class="tag glove-panel-tag"><b>Wuji 手套</b> · 骨骼与 24×31 触觉压力矩阵（0–1）</div>
      <div class="glove-grid">
        <div class="glove-pair">
          <div class="hand" id="hand-left">
            <div class="tag" id="tag-glove-left"><b>左手</b> · 手套骨骼</div>
            <div class="legend glove-legend" id="glove-legend-left"></div>
            <div class="empty-hint" id="empty-left">本会话无左手数据</div>
          </div>
          <div class="tactile-hand">
            <div class="tactile-head"><b>左手</b><span>压力热力图</span>
              <span class="tactile-stats" id="tactile-stats-left">等待数据…</span></div>
            <div class="tactile-canvas-wrap">
              <canvas class="tactile-canvas" id="tactile-left"></canvas>
              <div class="tactile-empty" id="tactile-empty-left">本会话无左手触觉数据</div>
              <span class="tactile-scale"><span>0</span><i></i><span>1</span></span>
            </div>
          </div>
        </div>
        <div class="glove-pair">
          <div class="hand" id="hand-right">
            <div class="tag" id="tag-glove-right"><b>右手</b> · 手套骨骼</div>
            <div class="legend glove-legend" id="glove-legend-right"></div>
            <div class="empty-hint" id="empty-right">本会话无右手数据</div>
          </div>
          <div class="tactile-hand">
            <div class="tactile-head"><b>右手</b><span>压力热力图</span>
              <span class="tactile-stats" id="tactile-stats-right">等待数据…</span></div>
            <div class="tactile-canvas-wrap">
              <canvas class="tactile-canvas" id="tactile-right"></canvas>
              <div class="tactile-empty" id="tactile-empty-right">本会话无右手触觉数据</div>
              <span class="tactile-scale"><span>0</span><i></i><span>1</span></span>
            </div>
          </div>
        </div>
      </div>
    </section>
  </main>

  <div class="transport">
    <button class="play" id="btn-play" title="播放/暂停 (空格)">▶</button>
    <div class="tc"><span id="tc-cur">0:00.000</span> <span class="sep">/</span> <span id="tc-tot">0:00.000</span></div>
    <div class="scrub-wrap"><input type="range" class="scrub" id="scrub" min="0" max="1000" value="0"/></div>
    <div class="ctl"><span>倍速</span>
      <select id="speed">
        <option value="0.25">0.25x</option><option value="0.5">0.5x</option>
        <option value="1" selected>1x</option><option value="2">2x</option><option value="4">4x</option>
      </select>
    </div>
    <div class="toggles">
      <label><input type="checkbox" id="t-grid" checked/>网格</label>
      <label><input type="checkbox" id="t-labels" checked/>名称</label>
      <label><input type="checkbox" id="t-trails" checked/>轨迹</label>
      <label style="gap:7px">轴长<input type="range" id="t-axis" min="4" max="40" value="15" style="width:78px"/></label>
    </div>
    <span class="hint">拖动=旋转 · 滚轮=缩放 · 右键拖=平移</span>
  </div>

  <div id="err"><div class="box"><h2>Three.js 未能加载</h2>
    <div>本页面通过 CDN 加载 3D 库，首次打开需要联网。</div>
    <div style="margin-top:10px;color:var(--dim)">离线方案：把 three.min.js (r128) 放到会话目录，
    并把本页 &lt;head&gt; 里的 CDN &lt;script src&gt; 改成 <b>./three.min.js</b> 后刷新。</div>
  </div>

<script>
"use strict";

// ---------- 通用小工具 ----------
const $ = (id) => document.getElementById(id);
function clamp(v, a, b){ return v < a ? a : (v > b ? b : v); }
function fmtTime(sec){
  sec = Math.max(0, sec);
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  const ms = Math.floor((sec - Math.floor(sec)) * 1000);
  return m + ":" + String(s).padStart(2, "0") + "." + String(ms).padStart(3, "0");
}
// 在升序数组 arr 中找最接近 T 的下标
function nearestIndex(arr, T){
  const n = arr.length;
  if (n === 0) return -1;
  if (T <= arr[0]) return 0;
  if (T >= arr[n-1]) return n-1;
  let lo = 0, hi = n - 1;
  while (hi - lo > 1){
    const mid = (lo + hi) >> 1;
    if (arr[mid] === T) return mid;
    if (arr[mid] < T) lo = mid; else hi = mid;
  }
  return (T - arr[lo] <= arr[hi] - T) ? lo : hi;
}

const FINGER_COLORS = [0xff6b6b, 0xffd166, 0x06d6a0, 0x4cc9f0, 0xb085f5]; // thumb..pinky
const ROOT_COLOR = 0xe6ebf2;
function fingerColor(fid){ return (fid >= 0 && fid < FINGER_COLORS.length) ? FINGER_COLORS[fid] : ROOT_COLOR; }

// ---------- 轻量轨道控制（自带，避免额外依赖） ----------
class Orbit {
  constructor(camera, dom, target){
    this.cam = camera; this.dom = dom;
    this.target = target ? target.clone() : new THREE.Vector3();
    const off = camera.position.clone().sub(this.target);
    this.sph = new THREE.Spherical().setFromVector3(off);
    this.minR = 1e-3; this.maxR = 1e6;
    this._drag = null; this._px = 0; this._py = 0;
    dom.style.touchAction = "none";
    dom.addEventListener("pointerdown", e => this._down(e));
    dom.addEventListener("pointermove", e => this._move(e));
    window.addEventListener("pointerup", () => this._drag = null);
    dom.addEventListener("wheel", e => this._wheel(e), {passive:false});
    dom.addEventListener("contextmenu", e => e.preventDefault());
    this.update();
  }
  _down(e){ this._drag = (e.button === 2 || e.shiftKey) ? "pan" : "rot";
            this._px = e.clientX; this._py = e.clientY; this.dom.setPointerCapture(e.pointerId); }
  _move(e){
    if(!this._drag) return;
    const dx = e.clientX - this._px, dy = e.clientY - this._py;
    this._px = e.clientX; this._py = e.clientY;
    if(this._drag === "rot"){
      this.sph.theta -= dx * 0.005;
      this.sph.phi = clamp(this.sph.phi - dy * 0.005, 0.02, Math.PI - 0.02);
    } else {
      const r = this.sph.radius;
      const panX = -dx * r * 0.0016, panY = dy * r * 0.0016;
      const te = this.cam.matrix.elements;
      const vx = new THREE.Vector3(te[0], te[1], te[2]);
      const vy = new THREE.Vector3(te[4], te[5], te[6]);
      this.target.addScaledVector(vx, panX).addScaledVector(vy, panY);
    }
    this.update();
  }
  _wheel(e){ e.preventDefault();
    this.sph.radius = clamp(this.sph.radius * (e.deltaY > 0 ? 1.1 : 0.9), this.minR, this.maxR);
    this.update(); }
  update(){
    const off = new THREE.Vector3().setFromSpherical(this.sph);
    this.cam.position.copy(this.target).add(off);
    this.cam.lookAt(this.target);
  }
}

// ---------- 一个可复用的 3D 视图 ----------
class Viewer3D {
  constructor(panel){
    this.panel = panel;
    this.canvas = document.createElement("canvas");
    panel.appendChild(this.canvas);
    this.renderer = new THREE.WebGLRenderer({canvas:this.canvas, antialias:true, alpha:false});
    this.renderer.setClearColor(0x0b0f14, 1);
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(45, 1, 0.001, 1000);
    this.camera.position.set(1, 1, 1.6);
    this.scene.add(new THREE.AmbientLight(0xffffff, 0.72));
    const dir = new THREE.DirectionalLight(0xffffff, 0.65); dir.position.set(2, 4, 3);
    this.scene.add(dir);
    this.controls = new Orbit(this.camera, this.canvas, new THREE.Vector3());
    this.grid = null; this.fitDiag = 1;
    this._ro = new ResizeObserver(() => this.resize());
    this._ro.observe(panel);
    this.resize();
  }
  resize(){
    const w = this.panel.clientWidth, h = this.panel.clientHeight;
    if(w === 0 || h === 0) return;
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h; this.camera.updateProjectionMatrix();
  }
  addGrid(center, minY, size){
    if(this.grid) this.scene.remove(this.grid);
    const div = Math.max(4, Math.min(40, Math.round(size)));
    const g = new THREE.GridHelper(size, div, 0x39485a, 0x1c2531);
    g.position.set(center.x, minY, center.z);
    g.material.opacity = 0.55; g.material.transparent = true;
    this.scene.add(g); this.grid = g;
  }
  frameTo(min, max){
    const center = new THREE.Vector3((min.x+max.x)/2, (min.y+max.y)/2, (min.z+max.z)/2);
    const size = new THREE.Vector3(max.x-min.x, max.y-min.y, max.z-min.z);
    let diag = size.length();
    if(!isFinite(diag) || diag < 1e-4) diag = 1;
    this.fitDiag = diag;
    this.controls.target.copy(center);
    this.controls.sph.radius = diag * 1.6;
    this.controls.sph.theta = 0.7; this.controls.sph.phi = 1.05;
    this.controls.minR = diag * 0.05; this.controls.maxR = diag * 40;
    this.controls.update();
    const gsize = Math.ceil(Math.max(Math.abs(min.x), Math.abs(max.x), Math.abs(min.z), Math.abs(max.z), diag) * 2);
    this.addGrid(center, min.y, Math.max(1, gsize));
  }
  render(){ this.renderer.render(this.scene, this.camera); }
}

function makeLabelSprite(text, colorHex){
  const pad = 8, fs = 44;
  const c = document.createElement("canvas");
  const ctx = c.getContext("2d");
  ctx.font = "600 " + fs + "px ui-monospace, monospace";
  const w = Math.ceil(ctx.measureText(text).width) + pad*2;
  const h = fs + pad*2;
  c.width = w; c.height = h;
  ctx.font = "600 " + fs + "px ui-monospace, monospace";
  ctx.fillStyle = "rgba(10,14,20,0.72)";
  ctx.fillRect(0, 0, w, h);
  ctx.strokeStyle = "rgba(120,140,165,0.5)"; ctx.lineWidth = 2; ctx.strokeRect(1,1,w-2,h-2);
  ctx.fillStyle = colorHex || "#d6dee8";
  ctx.textBaseline = "middle"; ctx.textAlign = "left";
  ctx.fillText(text, pad, h/2 + 2);
  const tex = new THREE.CanvasTexture(c);
  tex.minFilter = THREE.LinearFilter;
  const spr = new THREE.Sprite(new THREE.SpriteMaterial({map:tex, transparent:true, depthTest:false}));
  spr.userData.aspect = w / h;
  return spr;
}

// ---------- 全局状态 ----------
let DATA = null;
let trackerView = null;
const gloveViews = {};    // "left"/"right" -> Viewer3D（左右手各一个独立窗口）
const trackerObjs = {};   // role -> {group, sprite}
const gloveObjs = {};     // dev  -> {group, nodes[], boneLine, bonePairs, bonePos, positions, t_us}
const tactileObjs = {};   // "left"/"right" -> {rows, cols, values, t_us, lastIndex}
let trailWin = 90;
let labelsOn = true;

// 主时钟：有视频则以视频为准；否则用虚拟时钟
const video = $("video");
let videoUsable = false;                  // 仅当元数据成功、时长有限且 > 0 才为 true
const vclock = {t: 0, playing: false};    // 虚拟主时钟：始终作为播放的“真源”，3D 只读它
let masterRange = [0, 1];

// master(µs) <-> 视频秒 的映射（视频自身时间轴可能和数据略有出入）
function masterToVideoTime(T){
  const d = video.duration;
  if(!DATA.video || !isFinite(d) || d <= 0) return 0;
  const span = DATA.video.t1_us - DATA.video.t0_us;
  const f = span > 0 ? (T - DATA.video.t0_us) / span : 0;
  return clamp(f, 0, 1) * d;
}
function videoTimeToMaster(ct){
  const d = video.duration;
  if(!DATA.video || !isFinite(d) || d <= 0) return vclock.t;
  const f = ct / d;
  return DATA.video.t0_us + f * (DATA.video.t1_us - DATA.video.t0_us);
}
// 把视频对齐到主时钟：漂移超过阈值才 seek，避免每帧 seek 造成抖动
function syncVideoTo(T){
  if(!videoUsable) return;
  const target = masterToVideoTime(T);
  if(Math.abs(video.currentTime - target) > 0.30) video.currentTime = target;
}
// 视频不可用：隐藏视频并给出原因，但 3D 继续按虚拟时钟播放
function markVideoUnplayable(msg){
  videoUsable = false;
  try { video.pause(); } catch(e){}
  video.style.display = "none";
  const nv = $("novideo");
  nv.style.display = "block";
  nv.innerHTML = '<div class="big">视频无法播放</div><div>' + msg +
                 '；骨骼与 tracker 仍按内置虚拟时钟正常播放。</div>';
}
function setupVideo(src){
  video.muted = true;                     // 静音有利于点击/自动播放策略
  video.playsInline = true;
  video.preload = "auto";
  video.addEventListener("loadedmetadata", () => {
    if(isFinite(video.duration) && video.duration > 0){ videoUsable = true; }
    else { markVideoUnplayable("视频没有有效时长（可能是分段/流式 MP4）"); }
  });
  video.addEventListener("error", () =>
    markVideoUnplayable("浏览器无法解码该视频（GoPro 视频常为 HEVC/H.265，多数浏览器不支持）"));
  video.addEventListener("ended", () => { vclock.playing = false; syncPlayBtn(); });
  video.src = src;
  // 兜底：4 秒内既无元数据也无报错，就当作不可用（3D 照常播放）
  setTimeout(() => { if(!videoUsable) markVideoUnplayable("视频加载超时或无法解码"); }, 4000);
}

function getT(){ return vclock.t; }        // 3D 永远读虚拟主时钟
function setT(T){
  vclock.t = clamp(T, masterRange[0], masterRange[1]);
  if(videoUsable && video.paused) syncVideoTo(vclock.t);   // 暂停/拖动时立即对齐画面
}
function isPlaying(){ return vclock.playing; }
function tryPlayVideo(){
  if(!videoUsable) return;
  syncVideoTo(vclock.t);
  const p = video.play();
  if(p && p.catch) p.catch(() => markVideoUnplayable("浏览器阻止或无法播放该视频"));
}
function togglePlay(){
  vclock.playing = !vclock.playing;
  if(vclock.playing) tryPlayVideo();
  else if(videoUsable) video.pause();
  syncPlayBtn();
}
function syncPlayBtn(){ $("btn-play").textContent = isPlaying() ? "❚❚" : "▶"; }

// ---------- 构建 tracker 场景 ----------
function buildTrackers(){
  const panel = $("panel-trackers");
  if(!DATA.trackers){ $("tracker-legend").innerHTML = '<span style="color:var(--dim)">本会话无 tracker 数据</span>';
    trackerView = new Viewer3D(panel); trackerView.frameTo(new THREE.Vector3(-1,0,-1), new THREE.Vector3(1,2,1)); return; }
  trackerView = new Viewer3D(panel);

  const roles = DATA.trackers.roles;
  // 计算所有有效位置的包围盒用于取景
  const min = new THREE.Vector3(Infinity,Infinity,Infinity), max = new THREE.Vector3(-Infinity,-Infinity,-Infinity);
  let any = false;
  for(const r in roles){
    const pr = roles[r];
    for(let i=0;i<pr.position.length;i++){
      if(!pr.valid[i]) continue;
      const p = pr.position[i]; any = true;
      min.x=Math.min(min.x,p[0]); min.y=Math.min(min.y,p[1]); min.z=Math.min(min.z,p[2]);
      max.x=Math.max(max.x,p[0]); max.y=Math.max(max.y,p[1]); max.z=Math.max(max.z,p[2]);
    }
  }
  if(!any){ min.set(-1,0,-1); max.set(1,2,1); }
  trackerView.frameTo(min, max);

  const axisLen = trackerView.fitDiag * 0.12;
  const legend = $("tracker-legend"); legend.innerHTML = "";
  const roleNames = Object.keys(roles).sort();
  for(const r of roleNames){
    const group = new THREE.Group();
    const axes = new THREE.AxesHelper(axisLen);
    // 用统一的 X红/Y绿/Z蓝
    axes.material.vertexColors = true;
    group.add(axes);
    const dot = new THREE.Mesh(new THREE.SphereGeometry(axisLen*0.12, 12, 12),
                               new THREE.MeshStandardMaterial({color:0xffffff, emissive:0x222222}));
    group.add(dot);
    const spr = makeLabelSprite(r, "#c4b5fd");
    const lh = trackerView.fitDiag * 0.05;
    spr.scale.set(lh * spr.userData.aspect, lh, 1);
    trackerView.scene.add(spr);   // 挂场景, 每帧放到 marker 正上方(世界坐标), 不随姿态旋转
    // 轨迹
    const trailGeom = new THREE.BufferGeometry();
    const trailPos = new Float32Array((trailWin+1)*3);
    trailGeom.setAttribute("position", new THREE.BufferAttribute(trailPos, 3));
    trailGeom.setDrawRange(0, 0);
    const trail = new THREE.Line(trailGeom, new THREE.LineBasicMaterial({color:0x5a6b80, transparent:true, opacity:0.8}));
    trackerView.scene.add(trail);
    trackerView.scene.add(group);
    trackerObjs[r] = {group, spr, axes, trail, trailPos, trailGeom, labelY: axisLen*1.25};

    const row = document.createElement("div"); row.className = "row";
    row.innerHTML = '<span class="name">'+r+'</span><span class="xyz" id="xyz-'+r+'">—</span>';
    legend.appendChild(row);
  }
}

// ---------- 构建手套场景（左右手各一个独立窗口） ----------
// 判定一只手套属于左还是右：优先看 slot 文本，其次按出现顺序兜底(先左后右)。
function assignHandSides(devs){
  const bySide = {left:null, right:null};
  const unknown = [];
  for(const dev of devs){
    const slot = ((DATA.gloves[dev].slot || "") + "").toLowerCase();
    if(slot.indexOf("left") >= 0 || slot.indexOf("左") >= 0){ if(!bySide.left) bySide.left = dev; else unknown.push(dev); }
    else if(slot.indexOf("right") >= 0 || slot.indexOf("右") >= 0){ if(!bySide.right) bySide.right = dev; else unknown.push(dev); }
    else unknown.push(dev);
  }
  for(const dev of unknown){
    if(!bySide.left) bySide.left = dev;
    else if(!bySide.right) bySide.right = dev;
  }
  return bySide;
}

// 用一只手的所有帧算包围盒（这样每个窗口都能把该手撑满、互不干扰）
function handBounds(g){
  const min = new THREE.Vector3(Infinity,Infinity,Infinity);
  const max = new THREE.Vector3(-Infinity,-Infinity,-Infinity);
  for(let t=0;t<g.positions.length;t++){
    for(let n=0;n<g.positions[t].length;n++){
      const p = g.positions[t][n];
      min.x=Math.min(min.x,p[0]); min.y=Math.min(min.y,p[1]); min.z=Math.min(min.z,p[2]);
      max.x=Math.max(max.x,p[0]); max.y=Math.max(max.y,p[1]); max.z=Math.max(max.z,p[2]);
    }
  }
  if(!isFinite(min.x)){ min.set(-0.1,-0.1,-0.1); max.set(0.1,0.1,0.1); }
  return {min, max};
}

// 把一只手的节点/骨连线/腕标签建到指定视图里
function buildOneHand(dev, view){
  const g = DATA.gloves[dev];
  const N = g.parents.length;
  const b = handBounds(g);
  view.frameTo(b.min, b.max);
  const nodeR = Math.max(0.002, view.fitDiag * 0.018);

  const group = new THREE.Group();
  // 节点球
  const nodes = [];
  for(let i=0;i<N;i++){
    const col = fingerColor(g.fingerOf[i]);
    const isTip = g.labels[i] && g.labels[i].endsWith("_tip");
    const isRoot = g.parents[i] === -1;
    const r = isRoot ? nodeR*1.6 : (isTip ? nodeR*1.25 : nodeR);
    const m = new THREE.Mesh(new THREE.SphereGeometry(r, 12, 12),
      new THREE.MeshStandardMaterial({color:col, emissive:col, emissiveIntensity:0.28, roughness:0.5}));
    group.add(m); nodes.push(m);
  }
  // 骨连线
  const bonePairs = [];
  for(let c=0;c<N;c++){ if(g.parents[c] >= 0) bonePairs.push([c, g.parents[c]]); }
  const bonePos = new Float32Array(bonePairs.length*2*3);
  const boneColors = new Float32Array(bonePairs.length*2*3);
  for(let bi=0;bi<bonePairs.length;bi++){
    const col = new THREE.Color(fingerColor(g.fingerOf[bonePairs[bi][0]]));
    for(let k=0;k<2;k++){ boneColors[bi*6+k*3]=col.r; boneColors[bi*6+k*3+1]=col.g; boneColors[bi*6+k*3+2]=col.b; }
  }
  const boneGeom = new THREE.BufferGeometry();
  boneGeom.setAttribute("position", new THREE.BufferAttribute(bonePos, 3));
  boneGeom.setAttribute("color", new THREE.BufferAttribute(boneColors, 3));
  const boneLine = new THREE.LineSegments(boneGeom,
    new THREE.LineBasicMaterial({vertexColors:true, linewidth:2}));
  group.add(boneLine);
  // 腕部标签（受“名称”开关控制）
  const spr = makeLabelSprite("wrist", "#c4b5fd");
  const lh = view.fitDiag * 0.06;
  spr.scale.set(lh * spr.userData.aspect, lh, 1);
  spr.userData.baseH = lh;
  group.add(spr);

  view.scene.add(group);
  gloveObjs[dev] = {group, nodes, boneLine, boneGeom, bonePairs, bonePos, wristSpr: spr,
                    positions: g.positions, t_us: g.t_us};
}

// 填一个手指配色图例
function fillGloveLegend(elId){
  const legend = $(elId); if(!legend) return;
  legend.innerHTML = "";
  const names = ["拇指","食指","中指","无名","小指"];
  for(let i=0;i<5;i++){
    const chip = document.createElement("span"); chip.className = "chip";
    chip.innerHTML = '<span class="dot" style="background:#'+FINGER_COLORS[i].toString(16).padStart(6,"0")+'"></span>'+names[i];
    legend.appendChild(chip);
  }
}

function buildGloves(){
  const devs = DATA.gloves ? Object.keys(DATA.gloves) : [];
  const bySide = assignHandSides(devs);

  const sides = [
    {side:"left",  panelId:"hand-left",  tagId:"tag-glove-left",  legendId:"glove-legend-left",  emptyId:"empty-left",  name:"左手"},
    {side:"right", panelId:"hand-right", tagId:"tag-glove-right", legendId:"glove-legend-right", emptyId:"empty-right", name:"右手"},
  ];

  for(const s of sides){
    const dev = bySide[s.side];
    if(!dev){
      // 这一侧没有数据：显示提示，不建 3D 视图
      $(s.emptyId).style.display = "flex";
      $(s.tagId).innerHTML = "<b>"+s.name+"</b> · 无数据";
      $("tactile-stats-"+s.side).textContent = "无数据";
      continue;
    }
    const view = new Viewer3D($(s.panelId));
    gloveViews[s.side] = view;
    buildOneHand(dev, view);
    const g = DATA.gloves[dev];
    const typeName = (g.type || DATA.gloveType || "unknown").toUpperCase();
    const streamText = (g.streams || []).join(", ");
    $(s.tagId).innerHTML = "<b>"+s.name+"</b> · "+typeName+" · "+g.nodeCount+" 节点";
    $(s.tagId).title = dev + (g.serial ? (" · SN " + g.serial) : "") +
      (streamText ? (" · 数据流: " + streamText) : "");
    fillGloveLegend(s.legendId);
    if(g.tactile && g.tactile.values && g.tactile.values.length){
      tactileObjs[s.side] = Object.assign({lastIndex:-1}, g.tactile);
    } else {
      $("tactile-stats-"+s.side).textContent = "无数据";
    }
  }
}

// ---------- Wuji 触觉矩阵热力图（与采集页面使用同一配色和统计） ----------
function tactileColor(value){
  if(value < 0) return "#111821";              // SDK: -1 为无效触点
  const t = Math.pow(clamp(value, 0, 1), 0.55);
  const hue = 220 * (1 - t);                    // 低压蓝色，高压红色
  const light = 17 + 43 * t;
  return "hsl("+hue.toFixed(0)+",88%,"+light.toFixed(0)+"%)";
}

function drawTactile(side, T){
  const stream = tactileObjs[side];
  if(!stream) return;
  const idx = nearestIndex(stream.t_us, T);
  if(idx < 0) return;
  const canvas = $("tactile-"+side), wrap = canvas.parentElement;
  const rows = Number(stream.rows) || 24, cols = Number(stream.cols) || 31;
  wrap.style.aspectRatio = cols + " / " + rows;
  const ratio = Math.min(window.devicePixelRatio || 1, 2);
  const width = Math.max(1, Math.floor(wrap.clientWidth * ratio));
  const height = Math.max(1, Math.floor(wrap.clientHeight * ratio));
  const resized = canvas.width !== width || canvas.height !== height;
  if(!resized && stream.lastIndex === idx) return;
  if(resized){ canvas.width = width; canvas.height = height; }
  stream.lastIndex = idx;

  const values = stream.values[idx];
  if(!values || values.length !== rows * cols) return;
  const ctx = canvas.getContext("2d"); ctx.clearRect(0, 0, width, height);
  const pad = Math.max(4, Math.round(8 * ratio));
  const cellW = (width - pad * 2) / cols, cellH = (height - pad * 2) / rows;
  const gap = Math.max(.35, Math.min(cellW, cellH) * .08);
  let max = 0, sum = 0, valid = 0, active = 0;
  for(let row=0; row<rows; row++){
    for(let col=0; col<cols; col++){
      const value = Number(values[row * cols + col]);
      ctx.fillStyle = tactileColor(value);
      ctx.fillRect(pad + col * cellW + gap/2, pad + row * cellH + gap/2,
                   Math.max(.5, cellW-gap), Math.max(.5, cellH-gap));
      if(value >= 0){ valid++; sum += value; max = Math.max(max, value); if(value >= .05) active++; }
    }
  }
  $("tactile-empty-"+side).style.display = "none";
  $("tactile-stats-"+side).textContent =
    "max "+max.toFixed(2)+" · avg "+(valid ? sum/valid : 0).toFixed(3)+" · ≥.05 "+active;
}

// ---------- 每帧更新 ----------
function updateScene(T){
  // 手套
  for(const dev in gloveObjs){
    const o = gloveObjs[dev];
    const idx = nearestIndex(o.t_us, T);
    if(idx < 0) continue;
    const pos = o.positions[idx];
    for(let i=0;i<o.nodes.length;i++){ const p = pos[i]; o.nodes[i].position.set(p[0], p[1], p[2]); }
    let ptr = 0;
    for(let b=0;b<o.bonePairs.length;b++){
      const pc = pos[o.bonePairs[b][0]], pp = pos[o.bonePairs[b][1]];
      o.bonePos[ptr++]=pc[0]; o.bonePos[ptr++]=pc[1]; o.bonePos[ptr++]=pc[2];
      o.bonePos[ptr++]=pp[0]; o.bonePos[ptr++]=pp[1]; o.bonePos[ptr++]=pp[2];
    }
    o.boneGeom.attributes.position.needsUpdate = true;
    // 手别标签放在手腕(节点0)上方
    o.wristSpr.position.set(pos[0][0], pos[0][1] + o.wristSpr.userData.baseH*1.4, pos[0][2]);
  }

  drawTactile("left", T);
  drawTactile("right", T);

  // tracker
  if(DATA.trackers){
    const t_us = DATA.trackers.t_us;
    const idx = nearestIndex(t_us, T);
    for(const r in trackerObjs){
      const o = trackerObjs[r];
      const pr = DATA.trackers.roles[r];
      const ok = idx >= 0 && pr.valid[idx];
      o.group.visible = ok;
      const xyzEl = $("xyz-"+r);
      if(ok){
        const p = pr.position[idx], q = pr.orientation[idx];  // q = wxyz
        o.group.position.set(p[0], p[1], p[2]);
        o.group.quaternion.set(q[1], q[2], q[3], q[0]);       // three: x,y,z,w
        o.spr.position.set(p[0], p[1] + o.labelY, p[2]);
        o.spr.visible = labelsOn;
        if(xyzEl) xyzEl.innerHTML = p[0].toFixed(2)+", "+p[1].toFixed(2)+", "+p[2].toFixed(2);
        // 轨迹
        let n = 0;
        for(let k=Math.max(0, idx-trailWin); k<=idx; k++){
          if(!pr.valid[k]) continue;
          const pp = pr.position[k];
          o.trailPos[n*3]=pp[0]; o.trailPos[n*3+1]=pp[1]; o.trailPos[n*3+2]=pp[2]; n++;
        }
        o.trailGeom.setDrawRange(0, n);
        o.trailGeom.attributes.position.needsUpdate = true;
        o.trail.visible = $("t-trails").checked;
      } else {
        o.spr.visible = false;
        if(xyzEl) xyzEl.innerHTML = '<span class="off">— 未定位 —</span>';
        o.trail.visible = false;
      }
    }
  }
}

// ---------- UI 更新 ----------
let scrubbing = false;
function updateUI(T){
  const span = masterRange[1] - masterRange[0];
  const f = span > 0 ? (T - masterRange[0]) / span : 0;
  if(!scrubbing){
    const sc = $("scrub");
    sc.value = Math.round(clamp(f, 0, 1) * 1000);
    sc.style.setProperty("--pct", (f*100).toFixed(2) + "%");
  }
  $("tc-cur").textContent = fmtTime((T - masterRange[0]) / 1e6);
}

// ---------- 主循环 ----------
let last = performance.now();
function loop(now){
  const dt = (now - last) / 1000; last = now;
  if(vclock.playing){
    if(videoUsable && !video.paused){
      // 视频能正常播放时，用视频当前时间作为时钟（画面与数据天然同步）
      vclock.t = clamp(videoTimeToMaster(video.currentTime), masterRange[0], masterRange[1]);
    } else {
      // 否则用虚拟时钟推进——视频缺失/损坏/尚未解码，3D 也照样动
      let T = vclock.t + dt * 1e6 * (parseFloat($("speed").value) || 1);
      if(T >= masterRange[1]){ T = masterRange[1]; vclock.playing = false; syncPlayBtn(); }
      vclock.t = T;
      if(videoUsable && video.paused) tryPlayVideo();   // 元数据晚到时把视频拉起来跟随
    }
  }
  const T = getT();
  updateScene(T);
  updateUI(T);
  if(trackerView) trackerView.render();
  for(const s in gloveViews) gloveViews[s].render();
  requestAnimationFrame(loop);
}

// ---------- 初始化 ----------
function initControls(){
  $("btn-play").addEventListener("click", togglePlay);
  document.addEventListener("keydown", e => {
    if(e.code === "Space" && e.target.tagName !== "SELECT"){ e.preventDefault(); togglePlay(); }
  });
  const sc = $("scrub");
  const doSeek = () => {
    const f = sc.value / 1000;
    setT(masterRange[0] + f * (masterRange[1] - masterRange[0]));
    sc.style.setProperty("--pct", (f*100).toFixed(2) + "%");
  };
  sc.addEventListener("pointerdown", () => scrubbing = true);
  window.addEventListener("pointerup", () => scrubbing = false);
  sc.addEventListener("input", doSeek);
  $("speed").addEventListener("change", () => {
    const s = parseFloat($("speed").value) || 1;
    if(videoUsable) video.playbackRate = s;
  });
  $("t-grid").addEventListener("change", e => {
    if(trackerView && trackerView.grid) trackerView.grid.visible = e.target.checked;
    for(const s in gloveViews){ if(gloveViews[s].grid) gloveViews[s].grid.visible = e.target.checked; }
  });
  $("t-labels").addEventListener("change", e => {
    labelsOn = e.target.checked;                              // tracker 标签在 updateScene 里按此控制
    for(const d in gloveObjs) gloveObjs[d].wristSpr.visible = labelsOn;
  });
  $("t-axis").addEventListener("input", e => {
    const scale = parseInt(e.target.value, 10) / 15;
    for(const r in trackerObjs){
      const o = trackerObjs[r];
      o.axes.scale.setScalar(scale);
    }
  });
  syncPlayBtn();
}

function fillHeader(){
  $("h-base").textContent = DATA.base || "";
  const parts = [];
  const ng = DATA.gloves ? Object.keys(DATA.gloves).length : 0;
  const nt = DATA.trackers ? Object.keys(DATA.trackers.roles).length : 0;
  const gloveType = (DATA.gloveType || "unknown").toUpperCase();
  parts.push("手套 <b>"+ng+"</b> 只 · <b>"+gloveType+"</b>");
  parts.push("tracker <b>"+nt+"</b> role");
  if(DATA.video) parts.push("视频 <b>"+DATA.video.fps.toFixed(0)+"</b> fps · "+DATA.video.width+"×"+DATA.video.height);
  else parts.push("视频 <b>无</b>");
  if(DATA.overlap_s != null) parts.push("重叠 <b>"+Number(DATA.overlap_s).toFixed(1)+"</b>s");
  const dur = (DATA.time.t1_us - DATA.time.t0_us) / 1e6;
  parts.push("时长 <b>"+dur.toFixed(1)+"</b>s");
  $("h-stats").innerHTML = parts.join(" &nbsp;·&nbsp; ");
}

function start(){
  if(typeof THREE === "undefined"){ $("err").style.display = "flex"; return; }
  fetch("viewer_data.json", {cache:"no-store"})
    .then(r => r.json())
    .then(d => {
      DATA = d;
      fillHeader();
      if(DATA.gloveType === "wuji" || Object.values(DATA.gloves || {}).some(g => g.tactile)){
        document.querySelector("main").classList.add("with-tactile");
      }
      // 时长
      $("tc-tot").textContent = fmtTime((DATA.time.t1_us - DATA.time.t0_us) / 1e6);

      // 主时间轴统一用数据并集时间（与视频能否播放无关，保证 3D 一定能动）
      masterRange = [DATA.time.t0_us, DATA.time.t1_us];
      vclock.t = masterRange[0];
      $("tc-tot").textContent = fmtTime((masterRange[1] - masterRange[0]) / 1e6);

      // 视频作为“跟随”画面：能解码就跟随主时钟，不能解码也不影响 3D 播放
      if(DATA.video && DATA.video.src){
        setupVideo(DATA.video.src);
      } else {
        video.style.display = "none";
        $("novideo").style.display = "block";
      }

      buildTrackers();
      buildGloves();
      initControls();
      requestAnimationFrame(loop);
    })
    .catch(err => {
      console.error(err);
      $("err").style.display = "flex";
      $("err").querySelector(".box").innerHTML =
        '<h2>数据加载失败</h2><div>无法读取 viewer_data.json：'+err+'</div>';
    });
}
start();
</script>
</body>
</html>
"""


# ============================================================
# 支持 Range 的本地服务器（这样 MP4 能随意 seek）
# ============================================================

class RangeHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    """在 SimpleHTTPRequestHandler 基础上支持 HTTP Range，并禁用缓存。"""

    extensions_map = {
        **http.server.SimpleHTTPRequestHandler.extensions_map,
        ".mp4": "video/mp4", ".avi": "video/x-msvideo",
        ".json": "application/json", ".js": "text/javascript",
        ".css": "text/css", ".html": "text/html",
    }

    def end_headers(self):
        self.send_header("Cache-Control", "no-store, must-revalidate")
        self.send_header("Accept-Ranges", "bytes")
        super().end_headers()

    def send_head(self):
        path = self.translate_path(self.path)
        if os.path.isdir(path):
            return super().send_head()
        ctype = self.guess_type(path)
        try:
            f = open(path, "rb")
        except OSError:
            self.send_error(404, "File not found")
            return None
        fs = os.fstat(f.fileno())
        size = fs.st_size
        rng = self.headers.get("Range")
        self.range_length = None
        if rng:
            try:
                spec = rng.partition("=")[2]
                start_s, _, end_s = spec.partition("-")
                start = int(start_s) if start_s else 0
                end = int(end_s) if end_s else size - 1
                end = min(end, size - 1)
                if start < 0 or start > end:
                    start, end = 0, size - 1
            except Exception:
                start, end = 0, size - 1
            length = end - start + 1
            self.send_response(206)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Range", "bytes %d-%d/%d" % (start, end, size))
            self.send_header("Content-Length", str(length))
            self.end_headers()
            f.seek(start)
            self.range_length = length
            return f
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(size))
        self.end_headers()
        return f

    def copyfile(self, source, outputfile):
        # 浏览器在拖动/探测视频时经常主动断开连接，这属于正常现象，
        # 忽略由此产生的 ConnectionReset / BrokenPipe，避免刷屏的报错栈。
        try:
            length = getattr(self, "range_length", None)
            if length is None:
                shutil.copyfileobj(source, outputfile)
                return
            remaining = length
            while remaining > 0:
                chunk = source.read(min(64 * 1024, remaining))
                if not chunk:
                    break
                outputfile.write(chunk)
                remaining -= len(chunk)
        except (ConnectionResetError, BrokenPipeError, ConnectionAbortedError):
            pass

    def log_message(self, fmt, *args):
        pass  # 安静


class _ThreadingHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def serve(session_dir, port, open_browser=True):
    handler = functools.partial(RangeHTTPRequestHandler, directory=session_dir)
    # 端口被占用则顺延
    for p in range(port, port + 20):
        try:
            httpd = _ThreadingHTTPServer(("127.0.0.1", p), handler)
            port = p
            break
        except OSError:
            continue
    else:
        raise SystemExit("找不到可用端口 (%d..%d)" % (port, port + 20))

    url = "http://127.0.0.1:%d/viewer.html" % port
    print("\n  ✔ 可视化已就绪")
    print("    打开:  %s" % url)
    print("    根目录: %s" % os.path.abspath(session_dir))
    print("    Ctrl+C 结束服务。\n")
    if open_browser:
        threading.Thread(target=lambda: (time.sleep(0.6), webbrowser.open(url)), daemon=True).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n  已停止。")
    finally:
        httpd.server_close()


# ============================================================
# 入口
# ============================================================

def main():
    ap = argparse.ArgumentParser(
        description="同步采集会话的可视化：视频 + 手套骨骼 + 3D tracker 位姿（HTML/Three.js）")
    ap.add_argument("session", nargs="?",
                    help="会话目录 (data/session_YYYYMMDD_HHMMSS_TASK)。--demo 时作为示例数据的生成目录。")
    ap.add_argument("--demo", action="store_true",
                    help="生成一份示例会话数据(左右手骨骼 + 4 个 tracker role, 无视频)再打开页面。")
    ap.add_argument("--demo-type", choices=GLOVE_TYPES, default="wuji",
                    help="--demo 生成的手套格式：wuji(21节点，默认)或 manus(21节点)。")
    ap.add_argument("--port", type=int, default=8000, help="本地服务器端口 (默认 8000, 占用则顺延)。")
    ap.add_argument("--max-fps", type=float, default=60,
                    help="骨骼/tracker 导出的采样帧率上限, 用于减小 JSON (默认 60; 0=不抽稀)。")
    ap.add_argument("--no-open", action="store_true", help="不自动打开浏览器。")
    ap.add_argument("--build-only", action="store_true",
                    help="只生成 viewer.html + viewer_data.json, 不启动服务器。")
    args = ap.parse_args()

    session_dir = args.session
    if args.demo:
        session_dir = session_dir or os.path.join("data", "demo_session")
        make_demo(session_dir, glove_type=args.demo_type)
    if not session_dir:
        ap.error("请提供会话目录, 或用 --demo 生成示例。例如:\n"
                 "  python visualize_session.py data/session_20250101_120000\n"
                 "  python visualize_session.py data/demo_session --demo")

    payload = build_payload(session_dir, max_fps=args.max_fps)

    # 写出 viewer 文件到会话目录（MP4 用相对名引用, 服务根就是会话目录）
    with open(os.path.join(session_dir, "viewer_data.json"), "w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, separators=(",", ":"))
    with open(os.path.join(session_dir, "viewer.html"), "w", encoding="utf-8") as fp:
        fp.write(VIEWER_HTML)

    size_kb = os.path.getsize(os.path.join(session_dir, "viewer_data.json")) / 1024
    ng = len(payload.get("gloves") or {})
    nt = len(payload["trackers"]["roles"]) if payload.get("trackers") else 0
    print("[build] viewer_data.json (%.0f KB): %s 手套 %d 只 · "
          "tracker %d role · 视频 %s"
          % (size_kb, str(payload.get("gloveType", "unknown")).upper(),
             ng, nt, "有" if payload.get("video") else "无"))
    print("[build] viewer.html 已生成于 %s" % session_dir)

    if args.build_only:
        print("[build] --build-only: 未启动服务器。可用任意静态服务器托管该目录后打开 viewer.html。")
        return

    serve(session_dir, args.port, open_browser=not args.no_open)


if __name__ == "__main__":
    main()
