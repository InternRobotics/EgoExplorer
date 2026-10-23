#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
devices/base.py

相机与手套设备的抽象基类, 便于后续扩展不同硬件。

  Frame       —— 相机解码出的一帧 (index / image / time_sec / timestamp)
  BaseCamera  —— 随机读取视频/相机帧的统一接口
  BaseGlove   —— 数据手套的统一接口 (连接 / 订阅 / 录制 / 落盘)
  BaseTracker —— 位姿追踪器的统一接口 (连接 / 轮询 / 录制 / 落盘)

新增一台相机:
    class MyCamera(BaseCamera):
        @property
        def fps(self): ...
        @property
        def frame_count(self): ...
        def get_frame(self, index): ...
        def close(self): ...
    # __len__ / __getitem__(支持切片) / __iter__ / with 语句 自动可用

新增一款手套:
    class MyGlove(BaseGlove):
        @property
        def device_name(self): ...
        def connect(self): ...
        def disconnect(self): ...
        def is_connected(self): ...
        def subscribe(self, streams): ...
        def start_recording(self): ...
        def stop_recording(self): ...
        def save_recording(self, path): ...

新增一台追踪器 (只实现少量硬件相关成员, 轮询/录制/落盘由基类免费提供):
    class MyTracker(BaseTracker):
        @property
        def device_name(self): ...   # .npz 键前缀, 不含 "__"
        def connect(self): ...
        def disconnect(self): ...
        def is_connected(self): ...
        def _refresh(self): ...       # 前进一帧, 刷新并返回 self.latest_poses
    # poll / get_all_poses / get_pose_by_role / start_recording /
    # stop_recording / save_recording / with 语句 自动可用
    # (务必在自己的 __init__ 里调用 super().__init__())
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from collections import namedtuple
from typing import Any, Dict, Iterator, List, Optional


# ---------------------------------------------------------------------------
# 共享数据类型
# ---------------------------------------------------------------------------

# 一帧的返回结果: .image(BGR ndarray) / .index / .time_sec / .timestamp(datetime|None)
Frame = namedtuple("Frame", ["index", "image", "time_sec", "timestamp"])
"""相机解码出的单帧。

    index     (int):           从 0 开始的帧号。
    image     (np.ndarray):    BGR 图像 (OpenCV 约定)。
    time_sec  (float):         距视频开头的秒数。
    timestamp (datetime|None): 真实拍摄时刻; 无法确定时为 None。
"""


# ---------------------------------------------------------------------------
# BaseCamera
# ---------------------------------------------------------------------------

class BaseCamera(ABC):
    """所有相机 / 取流设备的抽象接口。

    子类必须实现下面 4 个抽象成员; 一旦实现, 便可免费获得
    __len__ / __getitem__(支持整数与切片) / __iter__ / with 上下文管理。

    例::

        class MyCamera(BaseCamera):
            def __init__(self, path):
                self._cap = cv2.VideoCapture(path)
            @property
            def fps(self): return self._cap.get(cv2.CAP_PROP_FPS)
            @property
            def frame_count(self): return int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
            def get_frame(self, index): ...
            def close(self): self._cap.release()
    """

    # -- 抽象属性 -----------------------------------------------------------

    @property
    @abstractmethod
    def fps(self) -> float:
        """视频帧率 (帧/秒)。"""

    @property
    @abstractmethod
    def frame_count(self) -> int:
        """总帧数 (未知/实时流时为 0)。"""

    # -- 抽象方法 -----------------------------------------------------------

    @abstractmethod
    def get_frame(self, index: int) -> Frame:
        """定位并解码单帧。

        参数:
            index: 从 0 开始的帧号; 支持负索引 (如 -1 表示最后一帧)。
        返回:
            :class:`Frame` 具名元组。
        抛出:
            IndexError: 帧号越界或无法解码。
        """

    @abstractmethod
    def close(self) -> None:
        """释放底层硬件/文件句柄。"""

    # -- Pythonic 便捷方法 ---------------------------------------------------

    def __len__(self) -> int:
        return self.frame_count

    def __getitem__(self, index):
        """支持 video[i] 以及切片 video[start:stop:step]。"""
        if isinstance(index, slice):
            return [self.get_frame(i) for i in range(*index.indices(self.frame_count))]
        return self.get_frame(index)

    def __iter__(self) -> Iterator[Frame]:
        """按帧号顺序迭代整段视频。"""
        for i in range(self.frame_count):
            yield self.get_frame(i)

    def __enter__(self) -> "BaseCamera":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(frames={self.frame_count}, fps={self.fps:.3f})"


# ---------------------------------------------------------------------------
# BaseGlove
# ---------------------------------------------------------------------------

class BaseGlove(ABC):
    """所有数据手套设备的抽象接口。

    子类职责:
      1. 连接 / 断开物理设备或网络中枢。
      2. 订阅一路或多路命名数据流。
      3. 在一次录制会话中把帧缓存到内存。
      4. 把录制结果按 :class:`~read_recording.RecordingReader` 约定的
         ``.npz`` 键格式落盘。

    ``.npz`` 键约定 (save_recording 必须遵守; 设备名内不含 "__"):

    ============================================ ==============================
    键                                            含义
    ============================================ ==============================
    ``{device_name}__{stream}__t_us``            主时间轴(µs): 设备优先, 主机回退
    ``{device_name}__{stream}__t_dev_us``        设备时钟(µs); 缺失帧为 -1
    ``{device_name}__{stream}__t_host_us``       主机墙钟(time.time, µs), 逐帧
    ``{device_name}__{subkey}``                  该子通道的数据本体
    ``{device_name}__{subkey}__counts``          (仅变长)每帧首维长度
    ``meta``                                     JSON 字符串, 元信息
    ============================================ ==============================
    """

    # -- 抽象属性 -----------------------------------------------------------

    @property
    @abstractmethod
    def device_name(self) -> str:
        """逻辑设备名 (如 ``'glove_0'``)。

        用作 ``.npz`` 里的键前缀, 因此 **不能包含双下划线 "__"**。
        """

    # -- 抽象方法 -----------------------------------------------------------

    @abstractmethod
    def connect(self) -> None:
        """建立与物理设备的连接。"""

    @abstractmethod
    def disconnect(self) -> None:
        """优雅关闭连接并释放资源。"""

    @abstractmethod
    def is_connected(self) -> bool:
        """当前设备是否已连接可用。"""

    @abstractmethod
    def subscribe(self, streams: List[str]) -> None:
        """订阅一路或多路命名数据流。

        参数:
            streams: 要启用的流名, 如
                     ``["tactile", "hand_joint_angles", "hand_skeleton"]``。
        抛出:
            ValueError:   请求了未知流名。
            RuntimeError: 未先 :meth:`connect` 便调用。
        """

    @abstractmethod
    def start_recording(self) -> None:
        """开始把接收到的帧缓存到内存。

        抛出:
            RuntimeError: 未先 :meth:`subscribe` 便调用。
        """

    @abstractmethod
    def stop_recording(self) -> None:
        """停止缓存; 数据保留在内存直到 :meth:`save_recording`。"""

    @abstractmethod
    def save_recording(self, path: str) -> None:
        """把缓存的录制写入 ``.npz`` 文件。

        键布局必须遵守类文档中的约定, 以便
        :class:`~read_recording.RecordingReader` 能读取。

        参数:
            path: 目标路径 (无 ``.npz`` 后缀时自动补上)。
        """

    # -- 上下文管理 ---------------------------------------------------------

    def __enter__(self) -> "BaseGlove":
        self.connect()
        return self

    def __exit__(self, *exc: Any) -> None:
        try:
            if self.is_connected():
                self.disconnect()
        except Exception:
            pass

    def __repr__(self) -> str:
        state = "connected" if self.is_connected() else "disconnected"
        return f"{self.__class__.__name__}(device_name={self.device_name!r}, {state})"


# ---------------------------------------------------------------------------
# BaseTracker
# ---------------------------------------------------------------------------

class BaseTracker(ABC):
    """所有位姿追踪器 (如 HTC Vive Tracker) 的抽象接口。

    与 :class:`BaseGlove` 的关键差异在于**取数模型**: 追踪器是**帧驱动**的 —— 每调用
    一次 :meth:`poll` 前进一帧并刷新位姿缓存 ``latest_poses``。因此录制不像手套那样由
    异步回调驱动, 而是: 只要处于录制态, 每次 :meth:`poll` 后就把当前这一帧的位姿追加进
    内存缓存; :meth:`save_recording` 再按下方 ``.npz`` 约定落盘。

    子类只需实现少量**硬件相关**成员:
      · :attr:`device_name`                          —— ``.npz`` 键前缀 (不含 ``"__"``)。
      · :meth:`connect` / :meth:`disconnect` / :meth:`is_connected`
      · :meth:`_refresh`                             —— 前进一帧, 更新并返回 ``latest_poses``。

    基类已实现 (与硬件无关, 免费获得):
      · :meth:`poll`                                 —— 调 ``_refresh``; 录制态下自动缓存该帧。
      · :meth:`get_all_poses` / :meth:`get_pose_by_role`
      · :meth:`start_recording` / :meth:`stop_recording` / :meth:`discard`
      · :meth:`save_recording`                       —— 落盘为 :class:`RecordingReader` 可读的 ``.npz``。
      · :meth:`total_frames` / :meth:`host_ts_range` / :meth:`tracker_info`
      · ``with`` 上下文管理 (进入即 connect, 退出即 disconnect)。

    .. note::
        子类务必在自己的 ``__init__`` 里调用 ``super().__init__()``, 以初始化
        ``latest_poses`` 与录制缓存。

    位姿格式 (``latest_poses`` 及各 getter 的返回): 以 role 名 (如 ``"waist"``) 为键,
    值为::

        {"position":    {"x": .., "y": .., "z": ..},
         "orientation": {"w": .., "x": .., "y": .., "z": ..}}

    ``.npz`` 键约定 (与 :class:`BaseGlove` 同一套, 因此追踪器文件可被 ``RecordingReader``
    直接读取; 追踪器只有一路名为 ``pose`` 的流, 数据本体按 role 拆成子通道):

    ==================================================== ==============================
    键                                                    含义
    ==================================================== ==============================
    ``{device_name}__pose__t_us``                        主时间轴(µs): 设备优先, 主机回退
    ``{device_name}__pose__t_dev_us``                    设备时钟(µs); 未采集时为 -1
    ``{device_name}__pose__t_host_us``                   主机墙钟(time.time, µs), 逐帧
    ``{device_name}__pose__t_source_host_us``            可选远端原始墙钟(µs); 本地源不写
    ``{device_name}__pose.{role}.position``              (T, 3) 位置 xyz; 该帧无效则为 NaN
    ``{device_name}__pose.{role}.orientation``           (T, 4) 四元数 wxyz; 无效则为 NaN
    ``{device_name}__pose.{role}.valid``                 (T,) bool, 该帧该 role 是否有效
    ``meta``                                             JSON 字符串, 元信息
    ==================================================== ==============================
    """

    # -- 初始化 (录制状态) --------------------------------------------------
    def __init__(self) -> None:
        # role -> {"position": {...}, "orientation": {...}}: 最近一帧的有效位姿。
        self.latest_poses: Dict[str, dict] = {}
        self._recording: bool = False
        self._rec_t_host_us: List[int] = []   # 逐帧主机时间戳 (µs)
        self._rec_t_dev_us: List[int] = []    # 逐帧设备时间戳 (µs); 无则 -1
        self._rec_t_source_host_us: List[int] = []  # 可选远端原始墙钟; 无则 -1
        self._rec_poses: List[dict] = []      # 逐帧位姿快照

    # -- 抽象属性 -----------------------------------------------------------

    @property
    @abstractmethod
    def device_name(self) -> str:
        """逻辑设备名 (如 ``'tracker_0'``)。

        用作 ``.npz`` 里的键前缀, 因此 **不能包含双下划线 "__"**。
        """

    # -- 抽象方法 (硬件相关) ------------------------------------------------

    @abstractmethod
    def connect(self) -> None:
        """建立与 runtime / 物理设备的连接。"""

    @abstractmethod
    def disconnect(self) -> None:
        """优雅断开连接并释放资源。"""

    @abstractmethod
    def is_connected(self) -> bool:
        """当前是否已连接可用。"""

    @abstractmethod
    def _refresh(self) -> Dict[str, dict]:
        """前进一帧, 刷新并返回 ``self.latest_poses``。

        由 :meth:`poll` 调用。子类在此完成一次"取一帧最新位姿"的硬件动作 (例如 OpenXR
        的 wait/begin/locate/end frame), 把结果写入 ``self.latest_poses``
        (role -> {position, orientation}) 并返回它。
        """

    # -- 读取位姿 -----------------------------------------------------------

    def poll(self) -> Dict[str, dict]:
        """前进一帧并刷新位姿缓存; 录制态下顺带把该帧写入缓存。

        每轮循环调用一次。返回与 :meth:`get_all_poses` 相同的 dict。
        """
        poses = self._refresh()
        if self._recording:
            self.record_frame(poses)
        return poses

    def get_all_poses(self, refresh: bool = True) -> Dict[str, dict]:
        """获取所有 role 的位姿 (role -> {position, orientation})。

        refresh=True 时先 :meth:`poll` 一帧使数值最新。想读同一瞬间的多个 role,
        先 :meth:`poll` 一次再以 ``refresh=False`` 读, 避免多推进几帧。
        """
        if refresh:
            self.poll()
        return dict(self.latest_poses)

    def get_pose_by_role(self, role: str, refresh: bool = True) -> Optional[dict]:
        """按 role 名取单个位姿; 该 role 当前无有效位姿时返回 None。"""
        if refresh:
            self.poll()
        return self.latest_poses.get(role)

    # -- 录制 ---------------------------------------------------------------

    @property
    def recording(self) -> bool:
        """当前是否处于录制态。"""
        return self._recording

    def start_recording(self) -> None:
        """开始把每次 :meth:`poll` 得到的位姿缓存到内存。

        抛出:
            RuntimeError: 未先 :meth:`connect` 便调用。
        """
        if not self.is_connected():
            raise RuntimeError("start_recording() 前请先 connect()。")
        self._rec_t_host_us = []
        self._rec_t_dev_us = []
        self._rec_t_source_host_us = []
        self._rec_poses = []
        self._recording = True

    def stop_recording(self) -> None:
        """停止缓存; 数据保留在内存直到 :meth:`save_recording` 或 :meth:`discard`。"""
        self._recording = False

    def discard(self) -> None:
        """停止并丢弃缓存 (不落盘)。"""
        self._recording = False
        self._rec_t_host_us = []
        self._rec_t_dev_us = []
        self._rec_t_source_host_us = []
        self._rec_poses = []

    def total_frames(self) -> int:
        """本次录制已缓存的帧数。"""
        return len(self._rec_poses)

    def host_ts_range(self) -> Optional[tuple]:
        """本次录制的主机时间戳 ``[min, max]`` (µs); 无帧则 ``None``。

        与手套的 ``t_host_us`` 同源 (``time.time``), 可直接用于和视频对齐/求重叠。
        """
        if not self._rec_t_host_us:
            return None
        return (int(min(self._rec_t_host_us)), int(max(self._rec_t_host_us)))

    def tracker_info(self) -> dict:
        """本次录制的简要元信息 (设备名 / 帧数 / 出现过的 role)。供上层写会话 JSON。"""
        roles = sorted({r for frame in self._rec_poses for r in frame})
        return {
            "device": self.device_name,
            "type": "tracker",
            "frames": len(self._rec_poses),
            "roles": roles,
        }

    def record_frame(self, poses: Optional[dict] = None,
                     t_host_us: Optional[int] = None,
                     t_dev_us: Optional[int] = None,
                     t_source_host_us: Optional[int] = None) -> None:
        """把一帧位姿追加进录制缓存。

        通常无需手动调用 —— 录制态下 :meth:`poll` 会自动调它。仅当你在缓存帧的同时手上
        有更精确的设备时间戳 (µs) 时, 才需要手动传 ``t_dev_us``。

        参数:
            poses:     该帧位姿 (省略则取当前 ``self.latest_poses``)。
            t_host_us: 主机时间戳µs (省略则取 ``time.time()``)。
            t_dev_us:  设备时间戳µs (省略则记 -1, 表示未采集)。
            t_source_host_us: 网络数据源的原始墙钟µs；本地源省略。
        """
        if poses is None:
            poses = self.latest_poses
        if t_host_us is None:
            t_host_us = int(time.time() * 1e6)
        self._rec_t_host_us.append(int(t_host_us))
        self._rec_t_dev_us.append(int(t_dev_us) if t_dev_us is not None else -1)
        self._rec_t_source_host_us.append(
            int(t_source_host_us) if t_source_host_us is not None else -1)
        # 存快照 (深拷贝一层), 避免后续帧改写同一个 dict。
        self._rec_poses.append({
            role: {"position": dict(p["position"]),
                   "orientation": dict(p["orientation"])}
            for role, p in poses.items()
        })

    def save_recording(self, path: str) -> None:
        """把缓存的位姿写入 ``.npz`` 文件。

        键布局遵守类文档中的约定, 以便 :class:`~read_recording.RecordingReader` 能读取
        (设备 = :attr:`device_name`, 单流 ``pose``, 每个 role 三个子通道)。

        参数:
            path: 目标路径 (无 ``.npz`` 后缀时自动补上)。
        """
        import json
        from datetime import datetime

        import numpy as np  # 延迟导入: 未落盘时 base 不强依赖 numpy

        if not path.endswith(".npz"):
            path += ".npz"

        dev = self.device_name

        # 切片快照: 即便有后台线程在录制中并发 append, 也只读到一致的前 n 帧
        # (record_frame 里位姿是最后追加的, 故以 len(_rec_poses) 为准)。
        n = len(self._rec_poses)
        rec_poses = self._rec_poses[:n]
        t_host = np.asarray(self._rec_t_host_us[:n], dtype=np.int64)
        t_dev = np.asarray(self._rec_t_dev_us[:n], dtype=np.int64)
        t_source_host = np.asarray(
            self._rec_t_source_host_us[:n], dtype=np.int64)
        # 主时间轴: 设备时钟优先, 缺失 (-1) 回退主机时钟。
        t_primary = np.where(t_dev >= 0, t_dev, t_host).astype(np.int64)

        # 本次录制中出现过 (至少一帧有效) 的所有 role。
        roles = sorted({r for frame in rec_poses for r in frame})

        arrays: Dict[str, Any] = {
            f"{dev}__pose__t_us": t_primary,
            f"{dev}__pose__t_dev_us": t_dev,
            f"{dev}__pose__t_host_us": t_host,
        }
        if np.any(t_source_host >= 0):
            arrays[f"{dev}__pose__t_source_host_us"] = t_source_host
        for role in roles:
            pos = np.full((n, 3), np.nan, dtype=np.float32)
            quat = np.full((n, 4), np.nan, dtype=np.float32)
            valid = np.zeros(n, dtype=bool)
            for i, frame in enumerate(rec_poses):
                p = frame.get(role)
                if p is None:
                    continue
                pp, qq = p["position"], p["orientation"]
                pos[i] = (pp["x"], pp["y"], pp["z"])
                quat[i] = (qq["w"], qq["x"], qq["y"], qq["z"])
                valid[i] = True
            arrays[f"{dev}__pose.{role}.position"] = pos
            arrays[f"{dev}__pose.{role}.orientation"] = quat
            arrays[f"{dev}__pose.{role}.valid"] = valid

        meta = {
            "created": datetime.now().isoformat(timespec="seconds"),
            "type": "tracker",
            "device": dev,
            "streams": ["pose"],
            "roles": roles,
            "frames": int(n),
            "ts_unit": "us",
        }
        arrays["meta"] = np.array(json.dumps(meta, ensure_ascii=False))

        np.savez_compressed(path, **arrays)

    # -- 上下文管理 ---------------------------------------------------------

    def __enter__(self) -> "BaseTracker":
        self.connect()
        return self

    def __exit__(self, *exc: Any) -> None:
        try:
            if self.is_connected():
                self.disconnect()
        except Exception:
            pass

    def __repr__(self) -> str:
        state = "connected" if self.is_connected() else "disconnected"
        return f"{self.__class__.__name__}(device_name={self.device_name!r}, {state})"
