#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
devices/manus_glove.py

ManusGlove —— 单只 Manus 数据手套的驱动, 继承 BaseGlove。

数据来源不是本地 SDK 回调, 而是 Manus 官方示例(改版 SDKMinimalClient)通过 **UDP**
广播出来的原始骨骼帧 —— 与 manus_udp_receiver_vis.py 收到的是同一份流。

与 wuji_glove.py 的三点关键差异:

  1) **取数模型**: wuji 用 asyncio 协程 ``await sub.recv_async()``; Manus 用一个后台
     **线程**阻塞收 UDP 包, 再按 ``glove_id`` 把每只手的数据分发给对应的 ManusGlove。
     因为两只手共用同一个 UDP 端口(一个 socket 只能 bind 一次), 所以由一个共享的
     :class:`ManusReceiver`(角色类似 wuji 的 SdkManager)统一收包 + 路由; 每只手对应
     一只 ManusGlove(``device_name`` = ``glove_0`` / ``glove_1``)。

  2) **数据内容(触觉缺省)**: Manus 只提供**手指关节骨骼**(``hand_skeleton``)与
     **关节角度**(``hand_joint_angles``)两路真实数据, **没有触觉**。为了让产出的
     ``.npz`` 结构与 wuji 通用(下游 visualize / aligner 无差别处理), 触觉相关的三路
     (``tactile`` / ``tactile_zones`` / ``tactile_point_cloud``)仍**保留在接口与
     元信息里, 但缺省不产生任何数据** —— 见 :data:`RESERVED_STREAMS` 与已就位、
     但永不被调用的 :meth:`ManusGlove._feed_tactile`(将来固件真的下发触觉时, 一行即可接上)。

  3) **时间戳(同步的关键)**: Manus 的 UDP 帧里只有一个自增 ``frame`` 计数器,
     **没有微秒级设备时钟**。因此每帧的设备时间戳记 ``-1``(缺失), 主时间轴回退到
     **主机墙钟** ``t_host_us``(收到包的瞬间取 ``time.time()`` 微秒)。这与视频
     (``cap.read()`` 返回后取 ``time.time()``)、tracker(``poll()`` 后取 ``time.time()``)
     用的是**同一台电脑、同一个时钟**, 所以三者天然落在同一根时间轴上, 落盘即已对齐 ——
     无需 THM / 时区 / marker, 也无需任何换算。

``.npz`` 键约定与 read_recording.RecordingReader / wuji_glove **完全一致**:
    ``{device_name}__{stream}__t_us`` / ``__t_dev_us`` / ``__t_host_us``
    ``{device_name}__{subkey}``       (+ ``__counts``, 仅变长)

单只手套用法(两只手共享一个 receiver)::

    from devices.manus_glove import ManusReceiver, ManusGlove, MANUS_STREAMS

    rx = ManusReceiver(port=9000); rx.start()
    left, right = ManusGlove.discover(rx)      # 固定返回 [左手 glove_0, 右手 glove_1]
    for g in (left, right):
        g.connect()                            # 向 receiver 注册本手
        g.subscribe(MANUS_STREAMS)
        g.start_recording()
    ...                                        # 采集一段时间(后台线程持续喂正在录制的手)
    for g in (left, right):
        g.stop_recording()
        g.save_recording("left.npz")           # 单设备 .npz
    rx.stop()

多只手套合并成一个 ``.npz`` 并与视频 / tracker 同步: 见 ``record_session_manus.py``。

依赖:
    pip install numpy
    (纯 UDP, 无需任何 Manus 本地 SDK; 只要有程序按 manus_udp_receiver_vis.py 的格式
     往本机某端口发 UDP 即可。)
"""
from __future__ import annotations

import json
import os
import re
import socket
import threading
import time
from collections import defaultdict
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .base import BaseGlove


# ============================================================
# 配置 (UDP 需与 C++ 端 m_UdpAddress / m_UdpPort 一致)
# ============================================================
MANUS_UDP_IP = "0.0.0.0"          # 监听所有本地网卡
MANUS_UDP_PORT = 9000
RECV_BUFFER = 65536               # 单个数据报读取上限(一帧仅几 KB)
SOCK_RCVBUF = 4 * 1024 * 1024     # 放大 OS 接收缓冲, 降低丢包概率

SAVE_DIR = "./data"               # .npz 输出目录(与 wuji_glove 一致)

# --- 真实产生数据的两路流(Manus 只有这两路) ---
MANUS_STREAMS: List[str] = ["hand_skeleton", "hand_joint_angles"]

# --- 触觉相关的三路: Manus 没有, 保留占位但缺省不产生数据 ---
#     保留在 AVAILABLE_STREAMS / meta 里, 使 .npz 结构与 wuji 通用;
#     _feed_tactile 已就位, 将来固件若真的下发触觉, 一行即可接上。
RESERVED_STREAMS: List[str] = ["tactile", "tactile_zones", "tactile_point_cloud"]

# 关节角度: Manus ergonomics 每只手 20 个值 = 5 指 × 4 关节。
FINGERS = ["thumb", "index", "middle", "ring", "pinky"]

# Manus 原始骨骼为 25 节点 (腕1 + 拇指4 + 四指各5); 多出的 4 个是四指掌骨关节
# (原始索引 5/10/15/20)。采集端在 ManusReceiver 分发前丢弃这些掌骨节点并把
# parent 索引重映射到 21 节点空间 (MediaPipe 顺序), 与 wuji 完全一致:
#   旧 6-9 (index)  -> 新 5-8 ; 旧 11-14 (middle) -> 新 9-12 ;
#   旧 16-19 (ring) -> 新 13-16 ; 旧 21-24 (pinky) -> 新 17-20。
MANUS_METACARPAL_INDICES = frozenset((5, 10, 15, 20))


def _to_21_nodes(nodes: List[dict]) -> List[dict]:
    """把 Manus 原始 25 节点骨骼规约成 21 节点 (MediaPipe/wuji 顺序)。

    丢弃食指/中指/无名指/小指的掌骨关节; 原本挂在掌骨上的各指第一节改为
    直接挂手腕。已是 21 节点 (或未知长度) 的输入原样返回。
    """
    if len(nodes) != 25:
        return list(nodes)
    kept = [i for i in range(25) if i not in MANUS_METACARPAL_INDICES]
    remap = {raw: new for new, raw in enumerate(kept)}
    out = []
    for raw in kept:
        node = dict(nodes[raw])
        parent = node.get("parent")
        if parent is not None:
            # 掌骨的父节点是手腕 (原始索引 0): 上溯一层挂到腕上。
            while parent in MANUS_METACARPAL_INDICES:
                parent = 0
            node["parent"] = remap[parent]
        out.append(node)
    return out


# 标准 21 节点手部骨骼层级 (MediaPipe 顺序, 与 wuji 一致); 当 UDP 帧未逐节点
# 携带 parent 时的兜底。0 wrist ; thumb 1-4 ; index 5-8 ; middle 9-12 ;
# ring 13-16 ; pinky 17-20。
STANDARD_HAND_PARENTS = {
    0: -1,
    1: 0, 2: 1, 3: 2, 4: 3,
    5: 0, 6: 5, 7: 6, 8: 7,
    9: 0, 10: 9, 11: 10, 12: 11,
    13: 0, 14: 13, 15: 14, 16: 15,
    17: 0, 18: 17, 19: 18, 20: 19,
}


# ============================================================
# JSON 解析 (容忍 C++ ostringstream 可能吐出的 nan / inf)
# ============================================================
_BAD_NUMBER = re.compile(r'-?(?:nan(?:\(ind\))?|inf(?:inity)?)', re.IGNORECASE)


def loads_tolerant(text: str):
    """``json.loads``; 若因 nan/inf 等非有限数解析失败, 把它们替换为 0 再解析。"""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return json.loads(_BAD_NUMBER.sub("0", text))


def _host_ts_us() -> int:
    """主机(收 UDP 的这台电脑)墙钟时间戳, 微秒。

    用 ``time.time()`` 而非 ``perf_counter`` —— 要的是与视频 / tracker 可直接比较的
    **绝对时间戳**。在收到 UDP 包后立刻调用, 尽量贴近该帧真正到达主机的时刻。
    """
    return int(time.time() * 1_000_000)


# ============================================================
# 数据提取: 把一帧 UDP 内容转成 numpy 数组(与 wuji 的子通道同构)
# ============================================================

def _skeleton_to_array(nodes: List[dict]) -> np.ndarray:
    """把一只手的节点列表转成 ``(N, 7)``: 每行 ``[px,py,pz, qx,qy,qz,qw]``。

    与 wuji 的 ``hand_skeleton`` 同构(位置 + 四元数 xyzw), 下游可无差别处理。
    节点 0 为手腕; 21 节点 MediaPipe 顺序 (分发前已由 :func:`_to_21_nodes`
    丢弃 4 个掌骨关节, 与 wuji 一致)。
    """
    rows = []
    for n in nodes:
        pos = n.get("pos", [0.0, 0.0, 0.0])
        rot = n.get("rot", [0.0, 0.0, 0.0, 1.0])
        rows.append([float(pos[0]), float(pos[1]), float(pos[2]),
                     float(rot[0]), float(rot[1]), float(rot[2]), float(rot[3])])
    return np.asarray(rows, dtype=np.float32) if rows else np.zeros((0, 7), np.float32)


def _reported_parents_to_list(parents: Optional[dict], n: int) -> Optional[List[int]]:
    """规约并校验 SDK 上报的父索引；不完整或成环时返回 ``None``。

    MANUS 的根节点允许把 ``parentId`` 设为自身。Windows 发送端把 node ID
    换算成数组索引后，这会表现为 ``parent == child``；页面使用的约定则是
    根节点父索引为 ``-1``，因此需要在这里统一转换。
    """
    if not parents or len(parents) != n:
        return None
    lst = [-1] * n
    try:
        for c, p in parents.items():
            child = int(c)
            if not 0 <= child < n:
                return None
            parent = int(p) if p is not None else -1
            lst[child] = -1 if parent == child else parent
    except (TypeError, ValueError):
        return None
    if lst.count(-1) != 1 or any(parent < -1 or parent >= n for parent in lst):
        return None
    for child in range(n):
        current, seen = child, set()
        while current >= 0:
            if current in seen:
                return None
            seen.add(current)
            current = lst[current]
    return lst


def _parents_to_list(parents: Optional[dict], n: int) -> List[int]:
    """返回页面/元数据使用的父索引，优先采用 SDK 上报值。"""
    reported = _reported_parents_to_list(parents, n)
    if reported is not None:
        return reported
    if n == len(STANDARD_HAND_PARENTS):
        return [STANDARD_HAND_PARENTS[index] for index in range(n)]
    return [-1] * n


def _angles_to_arrays(values) -> Dict[str, np.ndarray]:
    """把一只手的 ergonomics 角度(通常 20 个)转成与 wuji 同构的子通道。

    - ``hand_joint_angles.angles`` : ``(K,)`` 一维, 5 指 × 4 关节, 顺序见
      5 根手指 × 4 个关节角度(K 通常 = 20)。
    - ``hand_joint_angles.conf``   : ``(5,)`` 每指置信度占位 —— Manus 不提供, 记 NaN
      (保留结构, 与 wuji 的 ``hand_joint_angles.conf`` 对齐)。
    """
    flat = np.asarray(list(values), dtype=np.float32)
    conf = np.full((len(FINGERS),), np.nan, dtype=np.float32)
    return {"hand_joint_angles.angles": flat, "hand_joint_angles.conf": conf}


# ============================================================
# 打包: 把一串单帧数组打包成一个数组用于保存 (与 wuji_glove 同一套规则)
# ============================================================
_OBJECT = object()  # 标记「变长且无法对齐」需要 object 数组的情形


def _pack(vals):
    """把一串单帧数组打包成一个数组用于保存。

    返回 ``(packed, counts)``:
      - 形状全一致 -> ``stack`` 成 ``(T, *shape)``, counts=None
      - 仅首维变化 -> 沿轴 0 拼接 + 返回每帧长度 counts, 便于切分
      - 完全不规则 -> object 数组(读取需 ``allow_pickle``)
    """
    first = vals[0]
    if all(v.shape == first.shape for v in vals):
        return np.stack(vals, axis=0), None
    if all(v.ndim == first.ndim and v.shape[1:] == first.shape[1:] for v in vals):
        counts = np.asarray([v.shape[0] for v in vals], dtype=np.int64)
        return np.concatenate(vals, axis=0), counts
    obj = np.empty(len(vals), dtype=object)
    for i, v in enumerate(vals):
        obj[i] = v
    return obj, _OBJECT


# ============================================================
# ManusReceiver: 共享的 UDP 收包 + 路由中枢
#   - 一个 socket / 一个后台线程, 服务两只手。
#   - 每收到一个包, 立刻打主机时间戳, 再按 glove_id -> slot(left/right)分发给
#     对应的 ManusGlove。单手包不会误伤另一只手(它保持不变), 也不会让手左右互换。
# ============================================================

class ManusReceiver:
    """接收 Manus UDP 流并把每只手的数据路由到对应 :class:`ManusGlove`。

    角色类似 wuji 的 ``SdkManager``: 用 :meth:`instance` 拿一个进程级单例, 或直接
    ``ManusReceiver(port=...)`` 自建。收包在后台线程里完成(阻塞式 ``recvfrom``),
    因此 :class:`ManusGlove` 的缓存写入发生在**接收线程**, 与主线程读取/落盘并发 ——
    故 ManusGlove 内部对缓存加了锁, 且导出时做一致性快照。
    """

    _instance: Optional["ManusReceiver"] = None

    @classmethod
    def instance(cls, **kwargs) -> "ManusReceiver":
        """进程级单例(懒创建)。首次调用可传 ip/port 等参数, 之后忽略。"""
        if cls._instance is None:
            cls._instance = cls(**kwargs)
        return cls._instance

    def __init__(self, ip: str = MANUS_UDP_IP, port: int = MANUS_UDP_PORT,
                 sock_rcvbuf: int = SOCK_RCVBUF, verbose: bool = True) -> None:
        self._ip = ip
        self._port = port
        self._sock_rcvbuf = sock_rcvbuf
        self.verbose = verbose

        self._sock: Optional[socket.socket] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False

        self._lock = threading.Lock()
        self._gloves: Dict[str, "ManusGlove"] = {}     # slot -> glove
        self._slot_by_id: Dict[Any, str] = {}          # glove_id -> slot
        self._explicit_id_pair: Optional[Tuple[Any, Any]] = None

        # 统计 / 诊断
        self._packets = 0
        self._last_frame: Optional[int] = None
        self._lost = 0
        self._parse_errors = 0
        self._seen_slots: set = set()
        self._data_event = threading.Event()

    # -- 生命周期 -----------------------------------------------------------

    def start(self) -> None:
        """bind UDP 端口并启动后台收包线程(幂等)。"""
        if self._running:
            return
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, self._sock_rcvbuf)
        except OSError:
            pass
        actual = sock.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
        sock.bind((self._ip, self._port))
        sock.settimeout(0.5)
        self._sock = sock
        self._running = True
        self._thread = threading.Thread(target=self._recv_loop, name="manus-recv", daemon=True)
        self._thread.start()
        print(f"[manus] 监听 UDP {self._ip}:{self._port} ...  (接收缓冲 {actual} 字节)")

    def stop(self) -> None:
        """停止收包线程并关闭 socket; 打印一段收包小结。"""
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._sock is not None:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None
        with self._lock:
            print(f"[manus] 已停止。packets={self._packets}, lost={self._lost}, "
                  f"parse_errors={self._parse_errors}, slots_seen={sorted(self._seen_slots)}")

    def is_running(self) -> bool:
        return self._running

    # -- 注册 / 查询 --------------------------------------------------------

    def register(self, glove: "ManusGlove") -> None:
        """登记一只手(按其 slot 路由)。同一 slot 后注册者覆盖前者。"""
        with self._lock:
            self._gloves[glove.slot] = glove
            if glove.glove_id is not None:
                self._bind_slot_locked(glove.glove_id, glove.slot, authoritative=True)

    def unregister(self, glove: "ManusGlove") -> None:
        with self._lock:
            if self._gloves.get(glove.slot) is glove:
                del self._gloves[glove.slot]

    def wait_for_data(self, timeout: float) -> bool:
        """阻塞至收到第一包有效数据或超时; 返回是否收到。供 discover 探测用。"""
        return self._data_event.wait(timeout)

    def seen_slots(self) -> set:
        """迄今出现过数据的 slot 集合(用于打印哪只手在流)。"""
        with self._lock:
            return set(self._seen_slots)

    def stats(self) -> dict:
        with self._lock:
            return {"packets": self._packets, "lost": self._lost,
                    "parse_errors": self._parse_errors}

    # -- glove_id -> slot 映射 --------------------------------------------

    def _bind_slot_locked(self, glove_id, slot: str, authoritative: bool) -> None:
        """绑定 glove_id 与 slot（调用方必须持有 ``self._lock``）。

        明确的 ergonomics 左右信息比启动时根据出现顺序做的推测
        更可信，因此会替换同一 slot 上的旧映射。
        """
        if glove_id is None:
            return
        if not authoritative and glove_id in self._slot_by_id:
            return
        if authoritative:
            for other_id, other_slot in list(self._slot_by_id.items()):
                if other_id != glove_id and other_slot == slot:
                    del self._slot_by_id[other_id]
        self._slot_by_id[glove_id] = slot

    def _learn_explicit_slots(self, left_id, right_id) -> None:
        """从 ergonomics 字段学习一次权威的实际佩戴手 ID。

        glove ID 是一轮采集期间稳定的设备身份。这里只接受第一组完整、互异的
        双手 ID，并原子替换启动阶段按出现顺序得到的猜测；后续包即使
        ergonomics 的 left/right 瞬时反转，也不能让两个可视化框交换数据。
        """
        # 单边缺失或两边为同一 ID 都不是一组可用于锁定手别的完整证据。
        if left_id is None or right_id is None or left_id == right_id:
            return
        with self._lock:
            if self._explicit_id_pair is not None:
                return
            # 一次性提交整个双手映射，避免先改一侧时出现短暂冲突。
            self._slot_by_id = {
                left_id: "left",
                right_id: "right",
            }
            self._explicit_id_pair = (left_id, right_id)

    def _resolve_slot(self, glove_id, left_id=None, right_id=None) -> Optional[str]:
        """解析 skeleton 所属的手，后到的明确信息可纠正早期推测。"""
        if left_id is not None or right_id is not None:
            self._learn_explicit_slots(left_id, right_id)
        if glove_id is None:
            return None
        with self._lock:
            if glove_id in self._slot_by_id:
                return self._slot_by_id[glove_id]
            slot = next(
                (candidate for candidate in ("left", "right")
                 if candidate not in self._slot_by_id.values()),
                None,
            )
            if slot is not None:
                self._bind_slot_locked(glove_id, slot, authoritative=False)
            return slot

    # -- 收包线程 -----------------------------------------------------------

    def _recv_loop(self) -> None:
        assert self._sock is not None
        while self._running:
            try:
                data, _ = self._sock.recvfrom(RECV_BUFFER)
            except socket.timeout:
                continue
            except OSError:
                break
            host_us = _host_ts_us()     # 收到即打时间戳, 最贴近到达时刻
            text = data.decode("utf-8", errors="replace")
            try:
                payload = loads_tolerant(text)
            except json.JSONDecodeError as exc:
                with self._lock:
                    self._parse_errors += 1
                if self.verbose:
                    print(f"[manus] JSON 解析失败: {exc}. 前 200 字符: {text[:200]!r}")
                continue
            self._dispatch(payload, host_us, len(data))

    def _dispatch(self, payload: dict, host_us: int, nbytes: int) -> None:
        frame = payload.get("frame")
        ergo = payload.get("ergonomics", {}) or {}
        left = ergo.get("left", {}) or {}
        right = ergo.get("right", {}) or {}
        left_id = left.get("glove_id")
        right_id = right.get("glove_id")

        # 必须在遍历 skeletons 前一次性学习左右 ID。否则首个 skeleton
        # 可能被启动早期的“左优先”推测锁死，导致右手永远路由到左手。
        self._learn_explicit_slots(left_id, right_id)

        # 本包里各实际佩戴手的角度。优先按已锁定的 glove ID 路由，
        # 不直接信任逐包 left/right 标签，避免其瞬时反转时角度也串手。
        angle_by_slot: Dict[str, list] = {}
        for source_side, entry in (("left", left), ("right", right)):
            values = entry.get("values")
            if not values:
                continue
            gid = entry.get("glove_id")
            slot = self._resolve_slot(gid) if gid is not None else None
            if slot is None:
                # 兼容旧发送端未附 glove_id 的 ergonomics 数据。
                slot = source_side
            angle_by_slot[slot] = values

        # 本包里各 slot 的骨骼(raw-skeleton 是"按设备"下发的, 有骨骼即代表该设备刚更新)
        skel_by_slot: Dict[str, list] = {}
        parents_by_slot: Dict[str, dict] = {}
        for skel in payload.get("skeletons", []):
            gid = skel.get("glove_id")
            slot = self._resolve_slot(gid)
            if slot is None:
                continue
            # 25 -> 21 节点规约在这里完成: 下游 (_feed_frame/网页/落盘) 全部只见 21 节点。
            nodes = _to_21_nodes(skel.get("nodes", []))
            skel_by_slot[slot] = nodes
            parents = {}
            for idx, node in enumerate(nodes):
                p = node.get("parent")
                if p is not None:
                    parents[idx] = p
            parents_by_slot[slot] = parents

        # 统计 + 丢包检测(C++ 的 frame 每发一包 +1, 跳变 >1 即丢包)
        with self._lock:
            self._packets += 1
            if skel_by_slot or angle_by_slot:
                self._data_event.set()
            for s in list(skel_by_slot.keys()) + list(angle_by_slot.keys()):
                self._seen_slots.add(s)
            if isinstance(frame, int):
                prev = self._last_frame
                if prev is not None:
                    gap = frame - prev - 1
                    if gap > 0:
                        self._lost += gap
                        if self.verbose:
                            print(f"[manus] *** 丢 {gap} 包: frame {prev} -> {frame} "
                                  f"(累计丢 {self._lost})")
                self._last_frame = frame
            gloves = dict(self._gloves)
            pkt_no = self._packets

        if self.verbose and (pkt_no == 1 or pkt_no % 300 == 0):
            counts = {s: len(n) for s, n in skel_by_slot.items()}
            print(f"[manus] 包 #{pkt_no}, {nbytes} 字节, frame {frame}, "
                  f"骨骼节点 {counts}, 角度 {sorted(angle_by_slot)}")

        # 分发(在锁外; 每只 glove 内部各自加锁保护缓存):
        #  1) 有骨骼的手 = 该设备本包刚更新 -> 骨骼 + 同一时刻的角度一起记(时间戳完全一致)
        handled = set()
        for slot, nodes in skel_by_slot.items():
            g = gloves.get(slot)
            if g is None or not nodes:
                continue
            g._feed_frame(nodes, parents_by_slot.get(slot), angle_by_slot.get(slot), host_us)
            handled.add(slot)
        #  2) 兜底: 某手本包只有角度、没有骨骼(正常 raw-skeleton 流几乎不会发生) -> 单独记角度,
        #     宁可多记也不丢数据; 因为只在"无骨骼却有角度"时触发, 正常情况不会造成重复。
        for slot, values in angle_by_slot.items():
            if slot in handled:
                continue
            g = gloves.get(slot)
            if g is not None:
                g._feed_angles_only(values, host_us)


# ============================================================
# ManusGlove: 单只手套设备(左手或右手)
# ============================================================

class ManusGlove(BaseGlove):
    """单只 Manus 数据手套(一只手)。

    通过共享的 :class:`ManusReceiver` 收数据:
      · ``ManusGlove(slot="left", device_name="glove_0", receiver=rx)``
        —— ``connect()`` 时向 ``rx`` 注册本手; 若未传 receiver, 则用
        ``ManusReceiver.instance()`` 单例(两只手会共享同一个)。

    对外接口全部同步且与 :class:`~devices.wuji_glove.WujiGlove` 一致(``connect`` /
    ``subscribe`` / ``start_recording`` / ``stop_recording`` / ``discard`` /
    ``recording`` / ``total_frames`` / ``glove_info`` / ``ts_host`` / ``export_arrays`` /
    ``save_recording``), 因此在 ``record_session`` 类脚本里可直接替换 WujiGlove。
    唯一不同: 不提供 ``create_recv_tasks`` —— 收数据由 receiver 后台线程完成。
    """

    #: 接口层面"支持"的全部流名 = 真实两路 + 触觉占位三路; subscribe 只接受其子集。
    AVAILABLE_STREAMS: Tuple[str, ...] = tuple(MANUS_STREAMS) + tuple(RESERVED_STREAMS)

    #: 触觉占位流(Manus 不下发, 缺省为空)。
    RESERVED_STREAMS: Tuple[str, ...] = tuple(RESERVED_STREAMS)

    def __init__(
        self,
        slot: str = "left",
        device_name: str = "glove_0",
        receiver: Optional[ManusReceiver] = None,
        glove_id: Any = None,
    ) -> None:
        """
        参数:
            slot        : ``"left"`` 或 ``"right"`` —— 本对象代表哪只手(receiver 按 slot 路由)。
            device_name : 逻辑设备名, 作 ``.npz`` 键前缀; 不能含 ``"__"``。
            receiver    : 共享的 :class:`ManusReceiver`; 省略则 connect() 时用单例。
            glove_id    : 已知的 Manus glove_id(可选); 一般留空, 由 receiver 从 ergonomics 学到。
        抛出:
            ValueError: ``device_name`` 含 ``"__"`` 或 ``slot`` 非法。
        """
        if "__" in device_name:
            raise ValueError(f"device_name 不能包含 '__', 收到: {device_name!r}")
        if slot not in ("left", "right"):
            raise ValueError(f"slot 必须是 'left' 或 'right', 收到: {slot!r}")

        self.slot = slot
        self.glove_id = glove_id
        self._name = device_name
        self._receiver = receiver

        self._connected = False
        self._recording = False
        self._streams: List[str] = []
        self._hierarchy: Optional[List[int]] = None   # 骨架父索引(从首个带 parent 的帧学到)
        self._latest_skeleton: Optional[np.ndarray] = None

        # 缓存(与 wuji 一致): 但因跨线程写入, 用 _lock 保护。
        self._lock = threading.Lock()
        self.data: Dict[str, list] = defaultdict(list)    # subkey -> [ndarray, ...]
        self.ts_dev: Dict[str, list] = defaultdict(list)  # stream -> [设备时间戳 或 None(Manus 恒 None)]
        self.ts_host: Dict[str, list] = defaultdict(list) # stream -> [主机时间戳 us]

    # -- BaseGlove 抽象属性 --------------------------------------------------

    @property
    def device_name(self) -> str:
        return self._name

    # -- 发现 ---------------------------------------------------------------

    @classmethod
    def discover(cls, receiver: ManusReceiver,
                 device_prefix: str = "glove") -> List["ManusGlove"]:
        """固定返回 ``[左手, 右手]`` 两只 ManusGlove(左 = ``glove_0``, 右 = ``glove_1``)。

        Manus 无法像 wuji 那样主动 scan —— 只能从 UDP 流里被动发现。为使编号确定、且
        无论实际有几只手在流都能正常工作, 这里**总是**建左右两只; 某只手全程无数据时,
        它录到 0 帧, 落盘阶段(record_session_manus 的 save_all)会自动跳过。
        用 ``receiver.seen_slots()`` 可查当前实际有哪只手在流。
        """
        return [
            cls(slot="left",  device_name=f"{device_prefix}_0", receiver=receiver),
            cls(slot="right", device_name=f"{device_prefix}_1", receiver=receiver),
        ]

    # -- 连接 ---------------------------------------------------------------

    def connect(self) -> None:
        """向共享 receiver 注册本手(必要时顺带启动 receiver)。"""
        if self._receiver is None:
            self._receiver = ManusReceiver.instance()
        if not self._receiver.is_running():
            self._receiver.start()
        self._receiver.register(self)
        self._connected = True

    def disconnect(self) -> None:
        """从 receiver 注销本手(不关闭 receiver —— 它可能还在服务另一只手)。"""
        if self._receiver is not None:
            self._receiver.unregister(self)
        self._connected = False

    def is_connected(self) -> bool:
        return (self._connected and self._receiver is not None
                and self._receiver.is_running())

    # -- 订阅 ---------------------------------------------------------------

    def subscribe(self, streams: List[str]) -> None:
        """订阅一路或多路数据流(:attr:`AVAILABLE_STREAMS` 的子集)。

        触觉三路属于**保留占位**: 允许订阅, 但 Manus 不下发, 故永远录不到帧(缺省为空)。
        抛出:
            RuntimeError: 未先 connect()。
            ValueError:   含未知流名。
        """
        if not self._connected:
            raise RuntimeError("请先 connect() 再 subscribe()。")
        unknown = set(streams) - set(self.AVAILABLE_STREAMS)
        if unknown:
            raise ValueError(
                f"未知的流: {sorted(unknown)}。可选: {list(self.AVAILABLE_STREAMS)}")
        self._streams = list(streams)
        reserved = [s for s in streams if s in self.RESERVED_STREAMS]
        if reserved:
            print(f"  [note] {self._name}: Manus 无触觉数据, 以下流保留占位但缺省为空: {reserved}")

    # -- 接收回调(由 ManusReceiver 后台线程调用) ---------------------------

    def _feed_frame(self, nodes: list, parents: Optional[dict],
                    values: Optional[list], host_us: int) -> None:
        """记一帧: 骨骼(必有) + 该手同一时刻的角度(若本包有)。二者共享同一主机时间戳。"""
        skeleton = _skeleton_to_array(nodes) if nodes else None
        with self._lock:
            # 实时预览不应依赖“正在录制”。连接阶段收到的首帧也要
            # 保留，避免 Web tap 安装前的启动数据永久丢失。
            if skeleton is not None:
                self._latest_skeleton = skeleton
                reported_hierarchy = _reported_parents_to_list(parents, len(nodes))
                if reported_hierarchy is not None:
                    # SDK 的节点布局可能变化；不能让启动阶段的硬编码兜底覆盖
                    # 后续收到的权威层级。
                    self._hierarchy = reported_hierarchy
                elif self._hierarchy is None:
                    self._hierarchy = _parents_to_list(None, len(nodes))
            if not self._recording:
                return
            if "hand_skeleton" in self._streams and skeleton is not None:
                self.data["hand_skeleton"].append(skeleton)
                self.ts_host["hand_skeleton"].append(int(host_us))
                self.ts_dev["hand_skeleton"].append(None)  # Manus 无微秒设备时钟
            if values and "hand_joint_angles" in self._streams:
                self._append_angles_locked(values, host_us)

    def _feed_angles_only(self, values: list, host_us: int) -> None:
        """兜底: 某手本包只有角度没有骨骼时, 单独记角度(正常流几乎不触发)。"""
        with self._lock:
            if not self._recording:
                return
            if values and "hand_joint_angles" in self._streams:
                self._append_angles_locked(values, host_us)

    def _append_angles_locked(self, values: list, host_us: int) -> None:
        """把一帧角度写入缓存(**调用方须已持有 self._lock**)。"""
        for k, v in _angles_to_arrays(values).items():
            self.data[k].append(v)
        self.ts_host["hand_joint_angles"].append(int(host_us))
        self.ts_dev["hand_joint_angles"].append(None)

    def _feed_tactile(self, tactile_values, host_us: int) -> None:
        """【保留占位】触觉回调 —— Manus 当前**不会调用**它。

        保留此钩子: 将来若有版本通过 UDP 下发触觉, 在 ManusReceiver 里解析出触觉数组、
        对已订阅 ``tactile`` 的手调用本方法即可无缝接上, 无需改动落盘 / 读取 / 对齐。
        """
        with self._lock:
            if not self._recording or "tactile" not in self._streams:
                return
            self.data["tactile"].append(np.asarray(list(tactile_values), dtype=np.float32))
            self.ts_host["tactile"].append(int(host_us))
            self.ts_dev["tactile"].append(None)

    # -- 录制状态 -----------------------------------------------------------

    def _reset(self) -> None:
        self.data = defaultdict(list)
        self.ts_dev = defaultdict(list)
        self.ts_host = defaultdict(list)

    def start_recording(self) -> None:
        """开始缓存帧。

        抛出:
            RuntimeError: 未先 subscribe()。
        """
        if not self._streams:
            raise RuntimeError("请先 subscribe() 再 start_recording()。")
        with self._lock:          # 清空 + 置位在同一把锁内, 相对接收线程原子
            self._reset()
            self._recording = True

    def stop_recording(self) -> None:
        """停止缓存; 数据保留在内存, 可随后 :meth:`save_recording`。"""
        with self._lock:
            self._recording = False

    def discard(self) -> None:
        """终止并丢弃缓存(不落盘)。"""
        with self._lock:
            self._recording = False
            self._reset()

    @property
    def recording(self) -> bool:
        return self._recording

    @property
    def latest_skeleton(self) -> Optional[np.ndarray]:
        """返回最新骨架快照；即使未开始录制也可用于实时预览。"""
        with self._lock:
            if self._latest_skeleton is None:
                return None
            return self._latest_skeleton.copy()

    def total_frames(self) -> int:
        with self._lock:
            return sum(len(v) for v in self.ts_host.values())

    # -- 导出 / 落盘 ---------------------------------------------------------

    def glove_info(self) -> Dict[str, Any]:
        """本设备元信息(用于 meta.gloves)。``sn`` 位置放 Manus 的 glove_id。"""
        info: Dict[str, Any] = {"device_name": self._name, "sn": self.glove_id,
                                "slot": self.slot}
        if self._hierarchy is not None:
            info["hand_hierarchy"] = self._hierarchy   # 父索引列表, 供重建骨架连线
        return info

    def export_arrays(self) -> Tuple[Dict[str, np.ndarray], Dict[str, dict], bool]:
        """把本设备缓存打包成 npz 键字典, 供单 / 多设备落盘复用。

        返回 ``(arrays, channels, need_pickle)``, 语义与
        :meth:`~devices.wuji_glove.WujiGlove.export_arrays` 一致(键前缀 = device_name)。
        导出时对缓存做一致性快照(锁内拷贝), 即使接收线程仍在并发 append 也只读到一致的前 n 帧。
        """
        dev = self._name
        with self._lock:      # 一致性快照
            ts_host = {k: list(v) for k, v in self.ts_host.items()}
            ts_dev = {k: list(v) for k, v in self.ts_dev.items()}
            data = {k: list(v) for k, v in self.data.items()}

        arrays: Dict[str, np.ndarray] = {}
        channels: Dict[str, dict] = {}
        need_pickle = False

        # 时间戳(按 stream 一份, 同流多子通道共享):
        #   __t_us      主时间轴: 设备优先, 缺失回退主机(Manus 恒回退主机)
        #   __t_dev_us  设备时钟原值, Manus 无微秒时钟 -> 恒 -1
        #   __t_host_us 主机墙钟(time.time), 逐帧
        for stream, host_list in ts_host.items():
            dev_list = ts_dev.get(stream, [])
            t_host = np.asarray(host_list, dtype=np.int64)
            t_dev = np.asarray(
                [-1 if t is None else int(t) for t in dev_list], dtype=np.int64)
            t_primary = np.where(t_dev >= 0, t_dev, t_host)

            arrays[f"{dev}__{stream}__t_us"] = t_primary
            arrays[f"{dev}__{stream}__t_dev_us"] = t_dev
            arrays[f"{dev}__{stream}__t_host_us"] = t_host

            channels[f"{dev}/{stream}"] = {
                "frames": int(len(t_host)),
                "duration_s": float((t_primary[-1] - t_primary[0]) / 1e6)
                if len(t_primary) > 1 else 0.0,
            }

        # 数据本体
        for subkey, vals in data.items():
            if not vals:
                continue
            packed, counts = _pack(vals)
            arrays[f"{dev}__{subkey}"] = packed
            if counts is _OBJECT:
                need_pickle = True
            elif counts is not None:
                arrays[f"{dev}__{subkey}__counts"] = counts

        return arrays, channels, need_pickle

    def host_ts_range(self) -> Optional[Tuple[int, int]]:
        """本次录制主机时间戳 ``[min, max]``(µs); 无帧则 None(线程安全)。"""
        with self._lock:
            lo = hi = None
            for lst in self.ts_host.values():
                if not lst:
                    continue
                a, b = lst[0], lst[-1]
                lo = a if lo is None else min(lo, a)
                hi = b if hi is None else max(hi, b)
        return None if lo is None else (int(lo), int(hi))

    def save_recording(self, path: Optional[str] = None,
                       save_dir: str = SAVE_DIR) -> Optional[str]:
        """把**本单只手套**缓存落盘为单设备 ``.npz``(会先 stop_recording)。

        多手合并请用 ``record_session_manus`` 里的 ``save_all``。
        参数:
            path : 目标路径; 缺省则在 save_dir 下按时间命名。
        返回:
            实际写入的路径; 无数据时返回 None。
        """
        self.stop_recording()
        arrays, channels, need_pickle = self.export_arrays()
        if not channels:
            print("\n  没有采集到数据, 未保存")
            self.discard()
            return None

        streams = sorted({k.split("/", 1)[1] for k in channels})
        meta = {
            "created": datetime.now().isoformat(timespec="seconds"),
            "device_type": "manus_glove",
            "streams": streams,
            "reserved_streams": list(self.RESERVED_STREAMS),
            "gloves": [self.glove_info()],
            "ts_unit": ("microseconds. t_us=device-preferred (host fallback); "
                        "t_dev_us=device clock (-1: Manus 无微秒设备时钟); "
                        "t_host_us=host computer wall clock (time.time)"),
            "note": ("Manus 仅有 hand_skeleton / hand_joint_angles; "
                     "触觉流为保留占位, 缺省不产生数据。"),
            "channels": channels,
        }
        arrays["meta"] = np.array(json.dumps(meta, ensure_ascii=False))

        if path is None:
            os.makedirs(save_dir, exist_ok=True)
            path = os.path.join(save_dir, datetime.now().strftime("%Y%m%d_%H%M%S") + ".npz")
        elif not path.endswith(".npz"):
            path += ".npz"
        np.savez_compressed(path, **arrays)

        size_mb = os.path.getsize(path) / 1024 / 1024
        total = sum(c["frames"] for c in channels.values())
        print(f"\n  ✔ 已保存 {path}")
        print(f"    {total} 帧 / {len(channels)} 个通道 / {size_mb:.2f} MB")
        if need_pickle:
            print("    注意: 含变长数组, 读取请用 np.load(path, allow_pickle=True)")

        self.discard()
        return path
