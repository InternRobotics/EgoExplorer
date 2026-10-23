#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
devices/wuji_glove.py

WujiGlove —— 单只 wuji 数据手套的驱动, 继承 BaseGlove。
由 subscribe_async_recorder.py 里的 Recorder + 提取函数独立而来 (功能不变)。

一只 WujiGlove 封装:
  - 通过 wuji_sdk 连接 / 断开一只手套 (SdkManager.instance().connect(...))
  - 默认订阅触觉矩阵与手部骨骼两路数据流
  - 持续接收帧的协程 (create_recv_tasks); 仅在录制状态才提取并缓存, 避免空转
  - 把本设备数据缓存到内存, 落盘为 .npz (键前缀 = device_name)

与 read_recording.RecordingReader 的 .npz 键约定完全一致:
  {device_name}__{stream}__t_us / __t_dev_us / __t_host_us
  {device_name}__{subkey}       (+ __counts, 仅变长)

数据提取 (EXTRACTORS): 每个函数把「一帧 SDK 对象」转成
  (dict[子通道名 -> ndarray], 设备时间戳_微秒 或 None)。
用「鸭子类型」访问字段, 字段名与官方示例一致; **若固件版本字段不同, 只需改这些函数。**

依赖:
    pip install numpy
    wuji_sdk (真机运行时必需; 未安装时本模块仍可 import, 便于离线测试打包/保存逻辑)

单只手套用法::

    from devices.wuji_glove import WujiGlove

    # 扫描并取回未连接的手套列表:
    gloves = WujiGlove.discover()          # [WujiGlove(sn=..., device_name='glove_0'), ...]
    g = gloves[0]
    g.connect()
    g.subscribe(["tactile", "hand_skeleton"])

    # 在事件循环里挂上接收协程:
    import asyncio
    stop = asyncio.Event()
    tasks = g.create_recv_tasks(stop)

    g.start_recording()
    ... # 采集一段时间
    g.stop_recording()
    g.save_recording("session.npz")        # 单设备 .npz
    g.disconnect()

多只手套合并成一个 .npz: 见 record_gloves.py (原交互式录制程序)。
"""
from __future__ import annotations

import asyncio
import json
import os
import struct
import time
from collections import defaultdict
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from .base import BaseGlove

# 允许在未安装 SDK 的环境里 import 本模块(仅用于离线测试数据/保存逻辑)。
# 真机运行时, wuji_sdk 必须可用。
try:
    from wuji_sdk import SdkManager
except Exception:  # pragma: no cover
    SdkManager = None

# DeviceType 是较新 SDK 才加入的扫描期类型标记。旧 SDK 仍可正常扫描/连接，
# 只是不能在 connect 前按类型过滤，因此必须与 SdkManager 分开导入。
try:
    from wuji_sdk import DeviceType
except Exception:  # pragma: no cover
    DeviceType = None


# ============================================================
# 配置
# ============================================================

# 默认只订阅最终数据集和实时页面实际使用的两路。Wuji 每路约 120 Hz，
# 同时订阅派生分区、关节角、EMF 和点云会重复解析大量 Python 对象，并与视频抓帧
# 争用 CPU/GIL。其它 extractor 保留，便于有明确需求时重新加入 STREAMS。
STREAMS = [
    "tactile",
    "hand_skeleton",
]

SAVE_DIR = "./data"          # .npz 输出目录

# tactile_zones 里可能存在的分区名(getattr 安全, 缺失会自动跳过)
ZONE_NAMES = ["palm", "thumb", "index", "middle", "ring", "pinky"]


# ============================================================
# 数据提取: 把一帧 SDK 对象转成 numpy 数组
# 每个提取函数返回 (dict[子通道名 -> ndarray], 时间戳_微秒 或 None)
# 这些函数用「鸭子类型」访问字段, 字段名与官方示例 / offline_pipeline 示例一致。
# 如果你的固件版本字段不同, 只需改这里。
# ============================================================

def _ts_us(frame):
    """优先取设备时间戳 header.timestamp_us(微秒); 取不到返回 None。"""
    h = getattr(frame, "header", None)
    if h is not None:
        t = getattr(h, "timestamp_us", None)
        if t is not None:
            return int(t)
    return None


def _host_ts_us():
    """采集主机(手套连接的这台电脑)的墙钟时间戳, 单位微秒。

    与设备时间戳 header.timestamp_us 同为微秒, 二者可直接相减,
    可用于对齐设备时钟与电脑时钟、或估算数据传输延迟。
    注意: 用 time.time()(墙钟)而非 perf_counter, 因为要的是电脑的绝对时间戳。
    """
    return int(time.time() * 1_000_000)


def _pos(p):
    """position 既支持索引 p[0..2], 也支持 .x/.y/.z。"""
    try:
        return [float(p[0]), float(p[1]), float(p[2])]
    except Exception:
        return [float(p.x), float(p.y), float(p.z)]


def _quat(q):
    """orientation 用 .x/.y/.z/.w(与 Quaternion 一致), 兜底支持索引。"""
    try:
        return [float(q.x), float(q.y), float(q.z), float(q.w)]
    except Exception:
        return [float(q[0]), float(q[1]), float(q[2]), float(q[3])]


def ex_tactile(f):
    # frame.data 是一维触觉读数
    return {"tactile": np.asarray(list(f.data), dtype=np.float32)}, _ts_us(f)


def ex_tactile_zones(z):
    out = {}
    for name in ZONE_NAMES:
        v = getattr(z, name, None)
        if v is None:
            continue
        out[f"tactile_zones.{name}"] = np.asarray(list(v), dtype=np.float32)
    return out, _ts_us(z)


def ex_emf_poses(a):
    # 每根手指一行: [px,py,pz, qx,qy,qz,qw, confidence]
    rows = []
    for p in a.poses:
        rows.append(_pos(p.pose.position) + _quat(p.pose.orientation)
                    + [float(getattr(p, "confidence", float("nan")))])
    arr = np.asarray(rows, dtype=np.float32) if rows else np.zeros((0, 8), np.float32)
    return {"emf_poses": arr}, _ts_us(a)


def ex_hand_joint_angles(a):
    # angles: 所有手指角度拼成一维; conf: 每根手指一个置信度
    parts, confs = [], []
    for fng in a.fingers:
        parts.append(np.asarray(list(fng.angles), dtype=np.float32))
        confs.append(float(getattr(fng, "confidence", float("nan"))))
    flat = np.concatenate(parts) if parts else np.zeros((0,), np.float32)
    return {
        "hand_joint_angles.angles": flat,
        "hand_joint_angles.conf": np.asarray(confs, dtype=np.float32),
    }, _ts_us(a)


def ex_hand_skeleton(s):
    # 每个关节一行: [px,py,pz, qx,qy,qz,qw]; joints[0] 为手腕
    rows = []
    for j in s.joints:
        rows.append(_pos(j.pose.position) + _quat(j.pose.orientation))
    arr = np.asarray(rows, dtype=np.float32) if rows else np.zeros((0, 7), np.float32)
    return {"hand_skeleton": arr}, _ts_us(s)


def ex_point_cloud(c):
    # 点云字段在不同版本里可能不同, 这里做 best-effort:
    # 取不到点数据时, 至少记录每帧点数, 保证时间轴完整。
    pts = None
    for attr in ("points", "xyz"):
        v = getattr(c, attr, None)
        if v is not None:
            pts = v
            break
    if pts is None:
        to_np = getattr(c, "to_numpy", None)
        if callable(to_np):
            try:
                pts = to_np()
            except Exception:
                pts = None
    if pts is None:
        # 当前 SDK 使用 PointCloud2 风格的自描述字节流。每个点一般为
        # x/y/z/pressure 四个 little-endian float32，但仍按 fields/stride 解码，
        # 避免把原始 byte 误当成浮点点坐标。
        raw_data = getattr(c, "data", None)
        stride = int(getattr(c, "point_stride", 0) or 0)
        fields = getattr(c, "fields", None) or []
        if raw_data is not None and stride > 0:
            raw = bytes(raw_data)
            count = (c.point_count() if callable(getattr(c, "point_count", None))
                     else len(raw) // stride)
            offsets = {str(getattr(f, "name", "")): int(getattr(f, "offset", -1))
                       for f in fields}
            rows = []
            for i in range(min(int(count), len(raw) // stride)):
                base = i * stride
                try:
                    rows.append([
                        struct.unpack_from("<f", raw, base + offsets[name])[0]
                        for name in ("x", "y", "z", "pressure")
                    ])
                except (KeyError, struct.error):
                    rows = []
                    break
            if rows:
                return {"tactile_point_cloud": np.asarray(rows, np.float32)}, _ts_us(c)
        n = c.point_count() if callable(getattr(c, "point_count", None)) else 0
        return {"tactile_point_cloud.count": np.asarray([int(n)], np.int64)}, _ts_us(c)
    return {"tactile_point_cloud": np.asarray(pts, dtype=np.float32)}, _ts_us(c)


EXTRACTORS = {
    "tactile": ex_tactile,
    "tactile_zones": ex_tactile_zones,
    "emf_poses": ex_emf_poses,
    "hand_joint_angles": ex_hand_joint_angles,
    "hand_skeleton": ex_hand_skeleton,
    "tactile_point_cloud": ex_point_cloud,
}


# ============================================================
# 打包: 把一串单帧数组打包成一个数组用于保存
# ============================================================

_OBJECT = object()  # 标记「变长且无法对齐」需要 object 数组的情形


def _pack(vals):
    """把一串单帧数组打包成一个数组用于保存。
    返回 (packed, counts):
      - 形状全一致      -> stack 成 (T, *shape), counts=None
      - 仅首维变化      -> 沿轴 0 拼接 + 返回每帧长度 counts, 便于切分
      - 完全不规则      -> object 数组(需 allow_pickle 读取)
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
# WujiGlove: 单只手套设备
# 所有缓存访问都在同一个事件循环线程里完成, 无需加锁。
# ============================================================

class WujiGlove(BaseGlove):
    """单只 wuji 数据手套。

    连接方式二选一:
      · ``WujiGlove(sn=..., device_name="glove_0", manager=None)``
        —— connect() 时通过 ``SdkManager.instance().connect(sn=..., device_name=...)`` 打开
      · ``WujiGlove(handle=<已连接的 sdk 手套对象>, device_name="glove_0")``
        —— 直接包装已有句柄 (适合外部已完成扫描/连接)

    对外接口全部同步; 数据接收由 :meth:`create_recv_tasks` 返回的 asyncio 任务驱动。
    """

    #: 固件支持的全部流名; subscribe 只接受其子集。
    AVAILABLE_STREAMS: Tuple[str, ...] = tuple(STREAMS)

    #: 各流对应的提取函数 (固件字段变更时改 EXTRACTORS 即可)。
    EXTRACTORS = EXTRACTORS

    def __init__(
        self,
        sn: Optional[str] = None,
        device_name: str = "glove_0",
        handle: Any = None,
        manager: Any = None,
        slot: Optional[str] = None,
    ) -> None:
        """
        参数:
            sn         : 手套序列号 (SdkManager.scan() 得到的 dev.sn); 用 handle 时可省。
            device_name: 逻辑设备名, 作为 .npz 键前缀; 不能含 "__"。
            handle     : 已连接的 sdk 手套对象; 给了它则 connect() 不再自行连接。
            manager    : SdkManager 实例; 缺省用 SdkManager.instance()。
        抛出:
            ValueError: device_name 含 "__"。
        """
        if "__" in device_name:
            raise ValueError(f"device_name 不能包含 '__', 收到: {device_name!r}")
        self._sn = sn
        self._name = device_name
        self._handle = handle
        self._manager = manager
        self.slot = slot or self._side_from_sn(sn)

        self._connected: bool = handle is not None
        self._recording: bool = False
        self._streams: List[str] = []
        self._subs: Dict[str, Any] = {}         # stream -> sub 对象
        self.t0_perf: float = 0.0

        # 缓存(与原 Recorder 一致, 只是限定单设备):
        self.data: Dict[str, list] = defaultdict(list)   # subkey        -> [ndarray, ...]
        self.ts_dev: Dict[str, list] = defaultdict(list) # stream        -> [设备时间戳 us 或 None]
        self.ts_host: Dict[str, list] = defaultdict(list)# stream        -> [主机时间戳 us]
        # 页面实时预览使用；无论是否处于录制状态，都只保留各流最后一帧。
        self.latest: Dict[str, np.ndarray] = {}
        self.latest_t_host_us: Dict[str, int] = {}
        # 可选桥接组件使用的轻量帧监听器。回调发生在 SDK 接收事件循环中，
        # 必须保持非阻塞；普通本地采集不注册监听器，因此没有额外开销。
        self._frame_listeners: List[Callable] = []

    # -- BaseGlove 抽象属性 --------------------------------------------------

    @property
    def device_name(self) -> str:
        return self._name

    # -- 发现 / 连接 ---------------------------------------------------------

    @staticmethod
    def _side_from_sn(sn: Optional[str]) -> Optional[str]:
        """按 Wuji SN 约定识别左右手：第 4 个字符 J=左、K=右。"""
        text = str(sn or "").upper()
        if len(text) > 3 and text[3] == "J":
            return "left"
        if len(text) > 3 and text[3] == "K":
            return "right"
        return None

    @classmethod
    def scan(cls, manager: Any = None) -> list:
        """扫描在线设备, 返回 SDK 的设备描述列表 (每个含 .sn / .address)。"""
        if SdkManager is None:
            raise RuntimeError("未找到 wuji_sdk, 请先安装/配置 SDK 后在真机上运行。")
        mgr = manager or SdkManager.instance()
        return mgr.scan()

    @classmethod
    def discover(cls, manager: Any = None) -> List["WujiGlove"]:
        """扫描 Wuji 手套并按左、右、未知的稳定顺序构造对象。"""
        mgr = manager or (SdkManager.instance() if SdkManager else None)
        devices = cls.scan(mgr)
        if DeviceType is not None:
            devices = [dev for dev in devices
                       if getattr(dev, "device_type", DeviceType.WujiGlove)
                       == DeviceType.WujiGlove]
        rank = {"left": 0, "right": 1, None: 2}
        devices.sort(key=lambda dev: (rank[cls._side_from_sn(dev.sn)], str(dev.sn)))
        gloves = []
        for i, dev in enumerate(devices):
            side = cls._side_from_sn(dev.sn)
            name = "glove_0" if side == "left" else (
                "glove_1" if side == "right" else f"glove_{i}")
            gloves.append(cls(sn=dev.sn, device_name=name, manager=mgr, slot=side))
        return gloves

    def connect(self) -> None:
        """连接手套 (阻塞)。已给 handle 则直接标记为已连接。"""
        if self._handle is None:
            if SdkManager is None:
                raise RuntimeError("未找到 wuji_sdk, 请先安装/配置 SDK 后在真机上运行。")
            if self._sn is None:
                raise ValueError("未提供 sn 也未提供 handle, 无法连接。")
            mgr = self._manager or SdkManager.instance()
            self._handle = mgr.connect(sn=self._sn, device_name=self._name)
        # 若序列号未知, 从句柄补齐
        if self._sn is None:
            self._sn = getattr(self._handle, "serial_number", None)
        self._connected = True

    def disconnect(self) -> None:
        """关闭所有订阅并断开连接。"""
        for sub in self._subs.values():
            close = getattr(sub, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass
        self._subs.clear()
        self._streams.clear()

        # 仅关闭 subscription 不会终止 SDK 为设备创建的 bridge/后台任务；
        # 健康设备会因此让 Python 进程一直无法退出。优先让设备句柄执行官方
        # disconnect，旧版/测试句柄没有该方法时再通过 manager 按别名断开。
        handle = self._handle
        try:
            disconnect = getattr(handle, "disconnect", None)
            if callable(disconnect):
                disconnect()
            elif handle is not None and self._manager is not None:
                self._manager.disconnect(device_name=self._name)
        except Exception:
            # 设备可能已经因掉线被 SDK 清理；disconnect 仍须保持幂等。
            pass
        finally:
            self._handle = None
            self._recording = False
            self.latest.clear()
            self.latest_t_host_us.clear()
        self._connected = False

    def is_connected(self) -> bool:
        return self._connected

    # -- 订阅 ---------------------------------------------------------------

    def subscribe(self, streams: List[str]) -> None:
        """订阅一路或多路数据流。

        参数:
            streams: :attr:`AVAILABLE_STREAMS` 的子集。
        抛出:
            RuntimeError: 未先 connect()。
            ValueError:   含未知流名。
        """
        if not self._connected or self._handle is None:
            raise RuntimeError("请先 connect() 再 subscribe()。")
        unknown = set(streams) - set(self.AVAILABLE_STREAMS)
        if unknown:
            raise ValueError(
                f"未知的流: {sorted(unknown)}。可选: {list(self.AVAILABLE_STREAMS)}")
        self._streams = []
        for stream in streams:
            try:
                sub = getattr(self._handle, stream)().subscribe()
            except Exception as ex:
                print(f"  [warn] {self._name} 订阅 {stream} 失败: {ex}")
                continue
            self._subs[stream] = sub
            self._streams.append(stream)

    # -- 接收协程 -----------------------------------------------------------

    def create_recv_tasks(self, stop_event: "asyncio.Event") -> List["asyncio.Task"]:
        """为每一路已订阅的流建立一个接收协程任务, 返回任务列表。

        持续接收; 只有在录制状态才提取并缓存(避免空转开销)。调用方负责
        在退出时 cancel 这些任务。
        """
        tasks = []
        for stream, sub in self._subs.items():
            tasks.append(asyncio.create_task(self._recv_loop(stream, sub, stop_event)))
        return tasks

    async def _recv_loop(self, stream: str, sub: Any, stop_event: "asyncio.Event") -> None:
        extractor = self.EXTRACTORS[stream]
        frames_since_yield = 0
        while not stop_event.is_set():
            try:
                frame = await sub.recv_async()
            except asyncio.CancelledError:
                raise                                  # 正常取消(退出时), 照常传播
            except Exception as ex:
                # 手套断线 / 流被关闭: 本条流优雅退出, 不拖垮其它流与键盘响应
                print(f"\n[glove] 数据流 {stream} 接收结束/断开 "
                      f"({type(ex).__name__}: {ex}); 该流停止。")
                return
            if frame is None:
                await asyncio.sleep(0)                 # 防御: 万一 recv 立刻返回 None, 让出事件循环
                continue
            # 电脑(手套连接的这台主机)时间戳: 收到帧后立即采集, 最接近到达时刻
            host_us = _host_ts_us()
            try:
                subdict, dev_us = extractor(frame)     # dev_us 为设备时间戳, 可能为 None
            except Exception as ex:
                print(f"\n[glove] 解析 {stream} 帧失败 ({type(ex).__name__}: {ex}); 跳过该帧。")
                continue
            # 实时页面始终读取最新值；录制缓存仍由 _feed 内的 recording 开关控制。
            self.latest.update(subdict)
            self.latest_t_host_us[stream] = host_us
            for listener in tuple(self._frame_listeners):
                try:
                    listener(self, stream, subdict, dev_us, host_us)
                except Exception as ex:
                    print(f"\n[glove] 帧监听器异常 ({type(ex).__name__}: {ex}); "
                          "已忽略本帧回调。")
            self._feed(stream, subdict, dev_us, host_us)
            # SDK 内部队列一旦积压，recv_async() 可能连续返回已经就绪的 Future，
            # 此时单个数据流会长时间不把控制权还给 asyncio，终端按键与录制控制便会
            # 跟着延迟。每处理一小批帧显式让出一次，既保留全部录制帧又保证控制响应。
            frames_since_yield += 1
            if frames_since_yield >= 8:
                frames_since_yield = 0
                await asyncio.sleep(0)

    def _feed(self, stream: str, subdict: Dict[str, np.ndarray],
              dev_us: Optional[int], host_us: int) -> None:
        """接收回调: 把一帧写入缓存 (由 _recv_loop 调用)。"""
        if not self._recording:
            return
        self.ts_dev[stream].append(None if dev_us is None else int(dev_us))
        self.ts_host[stream].append(int(host_us))
        for k, v in subdict.items():
            self.data[k].append(v)

    # -- 录制状态 -----------------------------------------------------------

    def total_frames(self) -> int:
        return sum(len(v) for v in self.ts_host.values())

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
        self._reset()
        self.t0_perf = time.perf_counter()
        self._recording = True

    def stop_recording(self) -> None:
        """停止缓存; 数据保留在内存, 可随后 save_recording()。"""
        self._recording = False

    def discard(self) -> None:
        """终止并丢弃缓存(不落盘)。"""
        self._recording = False
        self._reset()

    @property
    def recording(self) -> bool:
        return self._recording

    # -- 导出 / 落盘 ---------------------------------------------------------

    def glove_info(self) -> Dict[str, Any]:
        """本设备的元信息 {device_name, sn, slot} (用于 meta.gloves)。"""
        return {"device_name": self._name, "sn": self._sn, "slot": self.slot}

    def add_frame_listener(self, listener: Callable) -> None:
        """注册实时帧监听器，供可选网络桥接组件转发 SDK 数据。"""
        if listener not in self._frame_listeners:
            self._frame_listeners.append(listener)

    def remove_frame_listener(self, listener: Callable) -> None:
        """移除先前注册的实时帧监听器。"""
        try:
            self._frame_listeners.remove(listener)
        except ValueError:
            pass

    def latest_stream(self, stream: str) -> Optional[Dict[str, np.ndarray]]:
        """返回某一路流的最新提取结果，供实时页面轻量读取。"""
        prefix = stream + "."
        out = {key: value for key, value in self.latest.items()
               if key == stream or key.startswith(prefix)}
        return out or None

    def host_ts_range(self) -> Optional[Tuple[int, int]]:
        """返回本次录制缓存中全部流的主机时间戳范围。"""
        values = [timestamp for stream_values in self.ts_host.values()
                  for timestamp in stream_values]
        if not values:
            return None
        return int(min(values)), int(max(values))

    def export_arrays(self) -> Tuple[Dict[str, np.ndarray], Dict[str, dict], bool]:
        """把本设备缓存打包成 npz 键字典, 供单/多设备落盘复用。

        返回 (arrays, channels, need_pickle):
          arrays      : {"{dev}__..." -> ndarray} 时间戳 + 数据本体(+counts)
          channels    : {"{dev}/{stream}" -> {frames, duration_s, [host_minus_dev_us]}}
          need_pickle : 是否含 object 数组(读取需 allow_pickle=True)
        """
        dev = self._name
        arrays: Dict[str, np.ndarray] = {}
        channels: Dict[str, dict] = {}
        need_pickle = False

        # 时间戳(按 stream 存一份, 同一流的多个子通道共享):
        #   __t_us      主时间轴: 优先设备时间戳, 设备缺失时回退电脑时间戳(向后兼容旧读法)
        #   __t_dev_us  设备(手套)时间戳原值, 缺失的帧用 -1 占位
        #   __t_host_us 电脑(手套连接的这台主机)墙钟时间戳, 逐帧记录
        for stream, host_list in self.ts_host.items():
            dev_list = self.ts_dev[stream]
            t_host = np.asarray(host_list, dtype=np.int64)
            t_dev = np.asarray(
                [-1 if t is None else int(t) for t in dev_list], dtype=np.int64)
            t_primary = np.where(t_dev >= 0, t_dev, t_host)

            arrays[f"{dev}__{stream}__t_us"] = t_primary
            arrays[f"{dev}__{stream}__t_dev_us"] = t_dev
            arrays[f"{dev}__{stream}__t_host_us"] = t_host

            ch = {
                "frames": int(len(t_host)),
                "duration_s": float((t_primary[-1] - t_primary[0]) / 1e6)
                if len(t_primary) > 1 else 0.0,
            }
            valid = t_dev >= 0
            if valid.any():
                # 电脑时钟 − 设备时钟 的中位数(微秒), 用于估算两时钟偏移 / 传输延迟
                ch["host_minus_dev_us"] = int(np.median(t_host[valid] - t_dev[valid]))
            channels[f"{dev}/{stream}"] = ch

        # 数据本体
        for subkey, vals in self.data.items():
            packed, counts = _pack(vals)
            arrays[f"{dev}__{subkey}"] = packed
            if counts is _OBJECT:
                need_pickle = True
            elif counts is not None:
                arrays[f"{dev}__{subkey}__counts"] = counts

        return arrays, channels, need_pickle

    def save_recording(self, path: Optional[str] = None,
                       save_dir: str = SAVE_DIR) -> Optional[str]:
        """把本设备缓存落盘为单设备 ``.npz`` (会先 stop_recording)。

        参数:
            path    : 目标路径; 缺省则在 save_dir 下按时间命名。
        返回:
            实际写入的路径; 无数据时返回 None。
        """
        self._recording = False
        if not self.ts_host:
            print("\n  没有采集到数据, 未保存")
            self._reset()
            return None

        arrays, channels, need_pickle = self.export_arrays()
        meta = {
            "created": datetime.now().isoformat(timespec="seconds"),
            "streams": list(self._streams),
            "gloves": [self.glove_info()],
            "ts_unit": ("microseconds. t_us=device-preferred (host fallback); "
                        "t_dev_us=device clock (-1 if missing); "
                        "t_host_us=host computer wall clock (time.time)"),
            "channels": channels,
        }
        arrays["meta"] = np.array(json.dumps(meta, ensure_ascii=False))

        if path is None:
            os.makedirs(save_dir, exist_ok=True)
            path = os.path.join(save_dir, datetime.now().strftime("%Y%m%d_%H%M%S") + ".npz")
        elif not path.endswith(".npz"):
            path = path + ".npz"
        np.savez_compressed(path, **arrays)

        size_mb = os.path.getsize(path) / 1024 / 1024
        total = sum(c["frames"] for c in channels.values())
        print(f"\n  ✔ 已保存 {path}")
        print(f"    {total} 帧 / {len(channels)} 个通道 / {size_mb:.2f} MB")
        if need_pickle:
            print("    注意: 含变长数组, 读取请用 np.load(path, allow_pickle=True)")

        self._reset()
        return path
