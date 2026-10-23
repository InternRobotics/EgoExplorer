#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
devices/gopro_camera.py

两个相机类, 都继承 BaseCamera:

  GoProCamera      离线读 MP4 —— 按帧号随机读取已录制的视频, 返回图像与时间戳。
                   (由原 gopro_video.GoProVideo 重构而来; 别名 GoProVideo)
                   供 aligner.Aligner 做逐帧对齐。
  GoProHDMICamera  实时采集 —— 从 HDMI 采集卡读 GoPro 直播画面, 录制为 MP4,
                   并逐帧记录主机时间戳; 供 record_session.py 与手套同步采集。

--- GoProCamera (离线读 MP4) ---
时间戳含义:
  - time_sec   : 距视频开头的秒数 (float), 只要有 MP4 就可用
  - timestamp  : 真实拍摄时刻 (datetime), 需要 THM 或 MP4 元数据里的起始时间; 无则 None
THM 是 GoPro 随视频生成的缩略图(带 EXIF 的 JPEG), 读帧本身不需要它, 但其 EXIF 里
通常记录了拍摄起始时刻, 用于把相对时间换算成真实时间。

--- GoProHDMICamera (实时采集) ---
硬件链路: GoPro + Media Mod(micro-HDMI) → HDMI 线 → USB HDMI 采集卡 → 电脑。
采集卡在 Linux 上是标准 UVC 设备 (/dev/videoN)。先确认设备编号:
    sudo apt install v4l-utils
    v4l2-ctl --list-devices
    v4l2-ctl -d /dev/videoN --list-formats-ext

依赖:
  pip install opencv-python pillow numpy
  (可选) 系统安装 ffprobe, 用于在没有 THM 时读取 MP4 的拍摄时间

用法::

    # 离线读已录制的 MP4 (随机读取)
    from devices.gopro_camera import GoProCamera
    with GoProCamera("GX012458.MP4", thm_path="GX012458.THM") as cam:
        f = cam.get_frame(100)      # 或 cam[100]; cam[-1]; cam[::30]

    # 实时采集并录制
    from devices.gopro_camera import GoProHDMICamera
    cam = GoProHDMICamera(device=0, width=1920, height=1080, fps=60).open()
    cam.start_recording("out.mp4")
    ... # 采集一段时间(后台线程写盘)
    mp4, ts_us = cam.save_recording()   # ts_us: 每帧主机时间戳, 供与手套对齐

    # 兼容旧代码: GoProVideo 是 GoProCamera 的别名
    from devices.gopro_camera import GoProVideo
"""
from __future__ import annotations

import os
import queue
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime, timedelta
from fractions import Fraction
from functools import lru_cache
from typing import Optional, Tuple

import cv2
import numpy as np

try:
    import av
except Exception:  # pragma: no cover - requirements 未安装完整时走 CFR 兼容路径
    av = None

from .base import BaseCamera, Frame


def _host_ts_us() -> int:
    """主机墙钟时间戳 (µs)。

    **必须与 WujiGlove 采集端的主机时钟一致** (都用 ``time.time()``),
    这样视频帧与手套帧才落在同一根时间轴上, 落盘后可直接按时间戳对齐。
    """
    return int(time.time() * 1_000_000)


def _default_cam_api() -> int:
    """采集卡默认后端(按操作系统自动选择):

      - Windows : DirectShow (``CAP_DSHOW``) —— 采集卡在 Windows 上最稳的后端
      - macOS   : AVFoundation
      - 其它(Linux): V4L2

    仍可在 __init__/autodetect 里显式传 ``api`` 覆盖(如 Windows 上想用 ``cv2.CAP_MSMF``)。
    """
    if os.name == "nt":
        return cv2.CAP_DSHOW
    if sys.platform == "darwin":
        return getattr(cv2, "CAP_AVFOUNDATION", cv2.CAP_ANY)
    return cv2.CAP_V4L2


@lru_cache(maxsize=None)
def _ffmpeg_has_encoder(encoder: str) -> bool:
    """系统 PATH 里是否有 ffmpeg, 且它支持给定编码器(如 ``"libx264"``)。结果缓存。

    注意: 这里查的是**系统 ffmpeg** (``/usr/bin/ffmpeg`` 等), 与 opencv-python wheel
    **自带**的那个精简 ffmpeg 无关 —— 后者常常只编进了硬件编码器(如 h264_v4l2m2m),
    没有软件 libx264/libx265, 导致 ``cv2.VideoWriter("avc1")`` 打不开、静默退回 mp4v。
    """
    if shutil.which("ffmpeg") is None:
        return False
    try:
        out = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"],
                             capture_output=True, text=True, timeout=5).stdout
    except Exception:
        return False
    return encoder in out


class _PyAVVideoWriter:
    """使用抓帧时刻作为 PTS 的 MP4 写入器。

    ``rawvideo`` 管道只能按固定输入帧率生成时间戳；编码队列一旦出现空洞，视频就会
    加速。PyAV 允许给每个 ``VideoFrame`` 显式设置微秒级 PTS，因此编码发生得早或晚
    都不会改变采集时间轴。
    """

    _TIME_BASE = Fraction(1, 1_000_000)

    def __init__(self, path: str, width: int, height: int, fps: float,
                 encoder: str = "libx264", crf: int = 20,
                 preset: str = "ultrafast", pix_fmt: str = "yuv420p",
                 threads: int = 0) -> None:
        self.path = path
        self.encoder = encoder
        self.error = ""
        self._ok = False
        self._container = None
        self._stream = None
        self._first_ts_us: Optional[int] = None
        self._last_pts = -1
        self._fallback_index = 0
        self._fps = float(fps) if fps and fps > 0 else 30.0
        if av is None:
            self.error = "未安装 PyAV（pip install av）"
            return
        try:
            options = {}
            if encoder in ("libx264", "libx265"):
                options = {"preset": str(preset), "crf": str(int(crf))}
            self._container = av.open(
                path, mode="w", options={"movflags": "+faststart"})
            rate = Fraction(str(self._fps)).limit_denominator(1_000_000)
            self._stream = self._container.add_stream(
                encoder, rate=rate, options=options)
            self._stream.width = int(width)
            self._stream.height = int(height)
            self._stream.pix_fmt = pix_fmt
            self._stream.time_base = self._TIME_BASE
            self._stream.codec_context.time_base = self._TIME_BASE
            if int(threads) > 0:
                self._stream.codec_context.thread_count = int(threads)
            self._ok = True
        except Exception as ex:
            self.error = f"{type(ex).__name__}: {ex}"
            self._ok = False
            if self._container is not None:
                try:
                    self._container.close()
                except Exception:
                    pass
            self._container = None
            self._stream = None

    def isOpened(self) -> bool:
        return bool(self._ok and self._container is not None and self._stream is not None)

    def write(self, frame, timestamp_us: Optional[int] = None) -> bool:
        if not self.isOpened():
            return False
        try:
            if timestamp_us is None:
                pts = int(round(self._fallback_index * 1_000_000 / self._fps))
            else:
                timestamp_us = int(timestamp_us)
                if self._first_ts_us is None:
                    self._first_ts_us = timestamp_us
                pts = timestamp_us - self._first_ts_us
            # 墙钟被校时或分辨率不足时仍须保证编码器看到严格递增的 PTS。
            pts = max(self._last_pts + 1, int(pts))
            video_frame = av.VideoFrame.from_ndarray(
                np.ascontiguousarray(frame), format="bgr24")
            video_frame.pts = pts
            video_frame.time_base = self._TIME_BASE
            for packet in self._stream.encode(video_frame):
                self._container.mux(packet)
            self._last_pts = pts
            self._fallback_index += 1
            return True
        except Exception as ex:
            self.error = f"{type(ex).__name__}: {ex}"
            self._ok = False
            return False

    def release(self) -> None:
        if self._container is None:
            return
        try:
            if self._stream is not None and self._ok:
                for packet in self._stream.encode():
                    self._container.mux(packet)
        except Exception as ex:
            self.error = f"{type(ex).__name__}: {ex}"
            self._ok = False
        finally:
            try:
                self._container.close()
            except Exception as ex:
                if not self.error:
                    self.error = f"{type(ex).__name__}: {ex}"
            self._container = None
            self._stream = None


class _FFmpegVideoWriter:
    """把 BGR 帧经管道喂给**系统 ffmpeg** 编码写 MP4; 接口 duck-type ``cv2.VideoWriter``
    (``write`` / ``isOpened`` / ``release``), 供 :class:`GoProHDMICamera` 在需要 H.264/H.265
    时替代 ``cv2.VideoWriter`` 使用。

    为什么需要它: opencv-python 的 wheel 自带一个**精简 ffmpeg**, 很多机器上只编进了硬件
    编码器(如 ``h264_v4l2m2m``, 需真实 V4L2 硬件), 没有软件 ``libx264``/``libx265``; 于是
    ``cv2.VideoWriter("avc1")`` 直接打不开, 静默退回 ``mp4v``(MPEG-4 Part 2), 拿不到 H.264。
    改用**系统 ffmpeg**(通常带 libx264/libx265)编码, 就能稳定产出标准 H.264(avc1),
    与 opencv 的编码器构建无关。

    只应在系统 ffmpeg 且该编码器可用时创建(用 :func:`_ffmpeg_has_encoder` 判断)。
    """

    def __init__(self, path: str, width: int, height: int, fps: float,
                 encoder: str = "libx264", crf: int = 20, preset: str = "veryfast",
                 pix_fmt: str = "yuv420p", threads: int = 0) -> None:
        fps = float(fps) if fps and fps > 0 else 30.0
        cmd = ["ffmpeg", "-y", "-loglevel", "error",
               "-f", "rawvideo", "-pix_fmt", "bgr24",
               "-s", f"{int(width)}x{int(height)}", "-r", f"{fps:.6g}",
               "-i", "pipe:0", "-an", "-c:v", encoder, "-pix_fmt", pix_fmt]
        # libx264/libx265 用 -crf + -preset 控质量/速度; 硬件编码器(nvenc/vaapi/qsv)不认 crf
        if encoder in ("libx264", "libx265"):
            cmd += ["-preset", preset, "-crf", str(int(crf))]
            if int(threads) > 0:
                cmd += ["-threads", str(int(threads))]
        cmd += ["-movflags", "+faststart", path]

        self.path = path
        self.encoder = encoder
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                      stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        # 参数错/编码器缺失时 ffmpeg 会立刻退出; 给一点点时间再用 poll() 判定是否真起来了
        time.sleep(0.08)
        self._ok = self._proc.poll() is None
        self.error = ""
        if not self._ok:
            try:
                self.error = self._proc.stderr.read().decode(errors="ignore").strip()
            except Exception:
                pass

    def isOpened(self) -> bool:
        return self._ok and self._proc.poll() is None

    def write(self, frame, timestamp_us: Optional[int] = None) -> bool:
        if not self._ok:
            return False
        try:
            self._proc.stdin.write(np.ascontiguousarray(frame).tobytes())
            return True
        except (BrokenPipeError, ValueError, OSError):
            self._ok = False       # 管道断了(多半是 ffmpeg 侧报错退出), 后续帧丢弃
            return False

    def release(self) -> None:
        if self._proc.stdin:
            try:
                self._proc.stdin.close()      # 关 stdin -> ffmpeg 冲刷缓冲、写尾并退出
            except Exception:
                pass
        try:
            self._proc.wait(timeout=10)
        except Exception:
            try:
                self._proc.kill()
            except Exception:
                pass


class GoProCamera(BaseCamera):
    """GoPro MP4 视频的随机读取器。

    真实拍摄起始时间的确定顺序 (优先级从高到低):

      1. ``thm_path`` 的 EXIF ``DateTimeOriginal`` / ``DateTime``
      2. MP4 ``creation_time`` 标签 (借助 ``ffprobe``)
      3. ``None`` —— 此时只有相对时间 ``time_sec`` 可用

    GoPro 为长 GOP 编码, 本地已验证按帧号 (``CAP_PROP_POS_FRAMES``) 定位是精准的。
    关键: 真正的帧时间戳在 ``cap.read()`` 之后再读取 (``CAP_PROP_POS_MSEC``),
    这样得到的才是刚解码这一帧的时刻, 而非定位目标。
    """

    def __init__(self, video_path: str, thm_path: Optional[str] = None) -> None:
        """
        参数:
            video_path: GoPro ``.MP4`` 路径。
            thm_path  : 可选的 ``.THM`` 缩略图路径, 用其 EXIF 确定拍摄起始时间;
                        没有则留空。
        抛出:
            RuntimeError: 无法打开视频。
        """
        self._video_path = video_path
        self._cap = cv2.VideoCapture(video_path)
        if not self._cap.isOpened():
            raise RuntimeError(f"无法打开视频: {video_path}")

        self._fps: float = self._cap.get(cv2.CAP_PROP_FPS)
        self._frame_count: int = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

        # 拍摄起始时间: 优先 THM, 其次 MP4 元数据, 都没有则 None
        self._start_time: Optional[datetime] = None
        if thm_path and os.path.exists(thm_path):
            self._start_time = self._start_time_from_thm(thm_path)
        if self._start_time is None:
            self._start_time = self._start_time_from_mp4(video_path)

    # -- BaseCamera 抽象实现 -------------------------------------------------

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def get_frame(self, index: int) -> Frame:
        """定位到目标帧再解码, 返回一个 Frame。

        参数:
            index: 从 0 开始的帧号; 支持负索引 (如 -1 表示最后一帧)。
        返回:
            :class:`~devices.base.Frame` 具名元组。
        抛出:
            IndexError: 帧号越界或无法读取。
        """
        if index < 0:
            index += self._frame_count  # 支持负索引
        if self._frame_count and not (0 <= index < self._frame_count):
            raise IndexError(f"帧号越界: {index} (总帧数 {self._frame_count})")

        # 定位到目标帧再解码; GoPro 为长 GOP 编码, 本地已验证按帧号定位是精准的
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ret, image = self._cap.read()
        if not ret or image is None:
            raise IndexError(f"无法读取帧号 {index}")

        # 关键: read() 之后读取 POS_MSEC, 得到的才是刚读取这一帧的时间戳
        time_sec = self._cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
        timestamp = None
        if self._start_time is not None:
            timestamp = self._start_time + timedelta(seconds=time_sec)

        return Frame(index=index, image=image, time_sec=time_sec, timestamp=timestamp)

    def close(self) -> None:
        """释放视频句柄。"""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    # -- 额外访问器 ----------------------------------------------------------

    @property
    def start_time(self) -> Optional[datetime]:
        """真实拍摄起始时刻; 无法确定时为 None。(Aligner 依赖此属性)"""
        return self._start_time

    @property
    def video_path(self) -> str:
        """源 MP4 路径。"""
        return self._video_path

    # -- 起始时间读取 --------------------------------------------------------

    @staticmethod
    def _parse_exif_dt(value) -> Optional[datetime]:
        if not value:
            return None
        try:
            return datetime.strptime(str(value).strip(), "%Y:%m:%d %H:%M:%S")
        except ValueError:
            return None

    @classmethod
    def _start_time_from_thm(cls, thm_path: str) -> Optional[datetime]:
        try:
            from PIL import Image
        except ImportError:
            print("  提示: 未安装 Pillow, 无法读取 THM 时间 (pip install pillow)")
            return None
        try:
            exif = Image.open(thm_path).getexif()
            sub = exif.get_ifd(0x8769)  # Exif 子目录
            return (cls._parse_exif_dt(sub.get(0x9003))       # DateTimeOriginal
                    or cls._parse_exif_dt(exif.get(0x0132)))  # DateTime
        except Exception as e:
            print(f"  读取 THM 失败: {e}")
            return None

    @staticmethod
    def _start_time_from_mp4(video_path: str) -> Optional[datetime]:
        if shutil.which("ffprobe") is None:
            return None
        try:
            out = subprocess.check_output(
                ["ffprobe", "-v", "quiet",
                 "-show_entries", "format_tags=creation_time",
                 "-of", "default=noprint_wrappers=1:nokey=1", video_path],
                text=True,
            ).strip()
            if not out:
                return None
            out = out.replace("Z", "").split(".")[0]
            return datetime.strptime(out, "%Y-%m-%dT%H:%M:%S")
        except Exception:
            return None


# ===========================================================================
# GoProHDMICamera —— 实时 HDMI 采集 + 录制为 MP4
# ===========================================================================

class GoProHDMICamera(BaseCamera):
    """从 HDMI 采集卡实时读取 GoPro 画面, 录制为 MP4, 并逐帧记录主机时间戳。

    这是 :class:`GoProCamera` (离线读 MP4) 的**采集端**姊妹类。硬件链路::

        GoPro + Media Mod(micro-HDMI) → HDMI 线 → USB HDMI 采集卡 → 电脑

    采集卡在 Linux 上是标准 UVC 设备 (``/dev/videoN``), OpenCV 当普通摄像头读即可。
    先用 ``v4l2-ctl --list-devices`` 确认编号, 把 ``device`` 改成对应的 N。

    与随机读取的 :class:`GoProCamera` 不同, 本类是**直播流**:
      - ``read()``            顺序抓取一帧 + 其主机时间戳 (µs)
      - ``for f in cam``      顺序迭代直播帧 (Frame)
      - ``get_frame(i)``      **不支持** —— 直播无法按帧号 seek
      - ``start_recording()`` / ``stop_recording()`` 用独立抓帧/编码线程写入 MP4,
        并把逐帧主机墙钟同时写入 VFR PTS，供播放和手套数据对齐。

    录制结束后, 若要按帧号随机读取那段视频, 用 ``GoProCamera(mp4_path)``。

    时间戳:
      每一帧在 ``cap.read()`` 返回后立即取 ``time.time()`` 作为主机时间戳 (µs),
      与 :class:`~devices.wuji_glove.WujiGlove` 的 ``t_host_us`` 同一时钟, 因此视频帧
      与手套帧可直接按时间戳最近邻对齐 (见 :meth:`frame_timestamps`)。

    依赖:
        pip install opencv-python av
    系统 ffmpeg 仅作为 PyAV 不可用时的固定帧率兼容路径。
    """

    # -- 写 MP4 的视频编码设置 (目标: 稳定拿到 H.264) --------------------------
    # H.264 优先经 PyAV/libx264 编码(见 _make_writer)，并写入逐帧真实 PTS。
    # 需要时可在实例或类上覆盖这几个属性来调质量/速度/编码器。
    _H264_TAGS = ("avc1", "H264", "h264", "X264", "x264")  # 视 fourcc_out 为 H.264 的标签
    _H264_ENCODER = "libx264"    # ffmpeg 编码器; 可换硬件的 "h264_nvenc"/"h264_vaapi"/"h264_qsv"
    _H264_CRF = 20               # 画质: 0(无损)~51(最差); 18~23 常用, 越小越清晰、文件越大
    _H264_PRESET = "ultrafast"   # 实时采集优先，避免软件编码挤占手套接收事件循环
    _H264_THREADS = 4             # 给 SDK/网页/控制循环保留 CPU；0 表示 ffmpeg 自动选择
    _ENCODE_QUEUE_SIZE = 16       # 1440p BGR 每帧约 11 MB；16 帧可吸收短时调度抖动
    _QUEUE_STOP = object()

    def __init__(self, device: int = 0, width: int = 2560, height: int = 1440,
                 fps: int = 60, fourcc_in: str = "MJPG", fourcc_out: str = "avc1",
                 api: Optional[int] = None) -> None:
        """
        参数:
            device    : 采集卡设备。可为整数索引(如 ``0``, 对应 /dev/video0)或**路径字符串**
                        (如 ``"/dev/video1"``)。OpenCV 有时用整数索引打不开、但用路径能打开,
                        open() 会自动两种都试。用 v4l2-ctl --list-devices 确认。
            width/height/fps : 期望的采集分辨率与帧率。
            fourcc_in : 采集卡输入编码; 廉价卡默认多为 YUYV, 上不去高分辨率高帧率,
                        切成 ``"MJPG"`` 才能跑 1080p60。
            fourcc_out: 写 MP4 的目标编码 (默认 ``"avc1"`` = H.264)。优先使用 PyAV
                        生成带真实抓帧 PTS 的 MP4；PyAV 不可用时回退系统 ffmpeg/OpenCV。
                        H.264 的质量/速度可用类属性 ``_H264_CRF``/``_H264_PRESET`` 或
                        ``_H264_ENCODER`` 调整。
            api       : OpenCV 后端; ``None``=按操作系统自动选(Windows→DSHOW,
                        macOS→AVFoundation, Linux→V4L2)。也可显式传, 如 ``cv2.CAP_MSMF``。
        """
        self._device = device
        self._req = (int(width), int(height), int(fps))
        self._fourcc_in = fourcc_in
        self._fourcc_out = fourcc_out
        self._api = _default_cam_api() if api is None else api
        self._opened_target = None             # 实际成功打开用的目标(路径/索引+后端)

        self._cap: Optional[cv2.VideoCapture] = None
        self._w = 0
        self._h = 0
        self._fps = float(fps)
        self._t0_us: Optional[int] = None      # 首帧主机时间戳, 用于算相对 time_sec

        # 录制状态
        self._recording = False
        self._writer: Optional[cv2.VideoWriter] = None
        self._thread: Optional[threading.Thread] = None  # 兼容旧代码：指向抓帧线程
        self._capture_thread: Optional[threading.Thread] = None
        self._encoder_thread: Optional[threading.Thread] = None
        self._encode_queue: Optional[queue.Queue] = None
        self._frame_ts: list = []              # 每写入一帧的主机时间戳 (µs)
        self._mp4_path: Optional[str] = None
        self._n = 0                            # 已成功编码帧数
        self._captured_n = 0                   # 已从 UVC 成功取到的帧数
        self._read_failures = 0
        self._queue_dropped = 0
        self._writer_failures = 0
        self._max_queue_depth = 0
        self._timestamp_mode = "unknown"
        self._latest = None                    # 最新一帧(供预览; 录制线程更新)

    # -- 打开 / 关闭 ---------------------------------------------------------

    def open(self) -> "GoProHDMICamera":
        """打开采集卡并按期望参数配置; 返回 self 以便链式调用。

        依次尝试 路径+V4L2 / 索引+V4L2 / 路径+默认后端 / 索引+默认后端, 因为 OpenCV 常出现
        "用整数索引打不开、用 /dev/videoN 路径能打开" 的情况。全部失败才抛异常。
        """
        cap, used = self._open_capture(self._device, self._api)
        if cap is None:
            tried = ", ".join(str(t) for t, _ in self._open_attempts(self._device, self._api))
            if os.name == "nt":
                hint = (" Windows 上采集卡请用**整数索引**(如 --cam-device 0/1/2), 不要用 /dev/videoN;"
                        " 在设备管理器里确认采集卡已作为“摄像头/图像设备”出现;"
                        " 换个索引或改用 CAP_MSMF 后端再试。")
            else:
                hint = (" 用 `v4l2-ctl --list-devices` 确认设备号; 若是权限问题,"
                        " 把当前用户加入 video 组: `sudo usermod -aG video $USER`(需重新登录);"
                        " 并确认此 OpenCV 支持 V4L2。")
            raise RuntimeError(
                f"无法打开采集卡 {self._device!r} (尝试过: {tried})。" + hint)

        w, h, fps = self._req
        # 廉价采集卡默认常是 YUYV, 分辨率/帧率上不去; 切成 MJPG 才能跑高分辨率高帧率。
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*self._fourcc_in))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        cap.set(cv2.CAP_PROP_FPS, fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)    # 只保留最新 1 帧, 压低延迟

        self._cap = cap
        self._opened_target = used
        self._w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or w
        self._h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or h
        self._fps = float(cap.get(cv2.CAP_PROP_FPS) or fps)
        self._t0_us = None
        print(f"实际采集: {self._w}x{self._h} @ {self._fps:.0f}fps  (来源 {used})")
        return self

    @staticmethod
    def _open_attempts(device, api):
        """按优先级列出 (打开目标, 后端) 组合。

        - 字符串设备(如 "/dev/video1"): 原样试 指定后端 / 自动后端。
        - 整数设备 + Linux: 额外试 /dev/videoN 路径(OpenCV 有时索引打不开、路径能打开)。
        - 整数设备 + Windows/macOS: 没有 /dev 节点, 只试 整数索引 + 指定后端 / 自动后端。
        """
        attempts = []
        if isinstance(device, str):
            attempts += [(device, api), (device, None)]
        elif os.name == "nt" or sys.platform == "darwin":
            attempts += [(device, api), (device, None)]
        else:
            attempts += [(f"/dev/video{device}", api), (device, api),
                         (f"/dev/video{device}", None), (device, None)]
        seen, uniq = set(), []
        for t in attempts:
            if t not in seen:
                seen.add(t)
                uniq.append(t)
        return uniq

    @classmethod
    def _open_capture(cls, device, api):
        """依次尝试各打开方式; 返回 (可用 VideoCapture 或 None, 描述用的目标字符串)。"""
        names = {cv2.CAP_V4L2: "V4L2", getattr(cv2, "CAP_DSHOW", -101): "DSHOW",
                 getattr(cv2, "CAP_MSMF", -102): "MSMF",
                 getattr(cv2, "CAP_AVFOUNDATION", -103): "AVFoundation"}
        for target, ap in cls._open_attempts(device, api):
            cap = cv2.VideoCapture(target, ap) if ap is not None else cv2.VideoCapture(target)
            if cap is not None and cap.isOpened():
                tag = "auto" if ap is None else names.get(ap, f"api{ap}")
                return cap, f"{target}+{tag}"
            if cap is not None:
                cap.release()
        return None, ""

    @classmethod
    def autodetect(cls, width: int = 1920, height: int = 1080, fps: int = 60,
                   prefer=None, fourcc_in: str = "MJPG", fourcc_out: str = "avc1",
                   api: Optional[int] = None, probe_frames: int = 10,
                   verbose: bool = True, max_index: int = 10):
        """扫描可用采集口, 返回第一个"能打开且能读到画面"的**已打开**相机; 找不到返回 None。

        - Linux: 扫描 ``/dev/video*``。
        - Windows/macOS: 没有 /dev 节点, 按**整数索引** 0..max_index-1 依次尝试。
        prefer: 优先尝试的设备(int 或路径); 其余按上述顺序补齐。用于自动挑选采集口
        (很多 UVC 采集卡会占用两个节点, 只有一个能出画面)。
        """
        cands = []
        if prefer is not None:
            cands.append(prefer)
        if os.name == "nt" or sys.platform == "darwin":
            for i in range(max_index):            # Windows/macOS: 按整数索引扫描
                if i not in cands:
                    cands.append(i)
        else:
            import glob
            for p in sorted(glob.glob("/dev/video*")):
                if p not in cands:
                    cands.append(p)
        for dev in cands:
            try:
                cam = cls(dev, width, height, fps, fourcc_in=fourcc_in,
                          fourcc_out=fourcc_out, api=api).open()
            except Exception as ex:
                if verbose:
                    print(f"[autodetect] {dev}: 打不开 ({type(ex).__name__})")
                continue
            img = None
            for _ in range(probe_frames):
                img, _ = cam.read()
                if img is not None:
                    break
            if img is not None:
                if verbose:
                    print(f"[autodetect] 选用 {dev} (来源 {cam._opened_target}): "
                          f"可出画面 {img.shape[1]}x{img.shape[0]}")
                return cam
            if verbose:
                print(f"[autodetect] {dev}: 能打开但读不到画面, 跳过")
            cam.close()
        return None

    def is_open(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    def close(self) -> None:
        """停止录制 (若在录) 并释放采集卡。"""
        self.stop_recording()
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    # -- BaseCamera 抽象实现 -------------------------------------------------

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def frame_count(self) -> int:
        """已采集帧数 (直播流没有"总帧数", 返回目前抓到的帧数)。"""
        return self._n

    def get_frame(self, index: int) -> Frame:
        """**直播流不支持按帧号随机读取** (无法 seek)。

        请改用 :meth:`read` 顺序取帧, 或迭代 ``for f in cam``;
        录制结束后要随机读那段视频, 用 ``GoProCamera(mp4_path)``。
        """
        raise NotImplementedError(
            "GoProHDMICamera 是直播采集, 无法按帧号随机读取; "
            "用 read() 顺序取帧, 或录制结束后用 GoProCamera(mp4_path) 读取 MP4。")

    # -- 顺序读取 -----------------------------------------------------------

    def read(self) -> Tuple[Optional["np.ndarray"], int]:
        """抓取一帧, 返回 ``(image_bgr 或 None, host_us)``。

        ``host_us`` 是 ``cap.read()`` 返回后立即取的主机墙钟时间戳 (µs),
        与手套 ``t_host_us`` 同一时钟。
        """
        if not self.is_open():
            raise RuntimeError("请先 open() 再 read()。")
        ok, frame = self._cap.read()
        host_us = _host_ts_us()
        if self._t0_us is None and ok:
            self._t0_us = host_us
        return (frame if ok else None), host_us

    def __iter__(self):
        """顺序迭代直播帧 (仅在已 open 时); 每次 yield 一个 :class:`Frame`。"""
        while self.is_open():
            img, host_us = self.read()
            if img is None:
                continue
            self._n += 1
            t0 = self._t0_us if self._t0_us is not None else host_us
            time_sec = (host_us - t0) / 1e6
            ts = datetime(1970, 1, 1) + timedelta(microseconds=host_us)
            yield Frame(index=self._n - 1, image=img, time_sec=time_sec, timestamp=ts)

    # -- 录制 (后台线程写 MP4) ----------------------------------------------

    def _make_writer(self, path: str):
        """创建视频写入器, 返回 ``(writer 或 None, 实际输出路径, 编码描述)``。

        编码选择顺序 (目标: 稳定拿到 H.264，并保留逐帧真实 PTS):

          1. **PyAV + libx264/mpeg4** —— 首选。每帧 PTS 来自抓帧时的 Linux 墙钟，
             队列延迟或输入掉帧不会压缩 MP4 时间轴。
          2. **系统 ffmpeg + libx264** 固定帧率兼容路径。
          3. **cv2.VideoWriter 按 fourcc 兜底**: 先试 ``fourcc_out`` 及其它 H.264 标签
             (opencv 自带编码器能出 H.264 时用), 再退 ``mp4v``(MPEG-4 Part 2, 几乎总可用),
             最后 AVI(MJPG/XVID)。

        返回的 writer 都实现 ``write/isOpened/release``。先确保输出目录存在(**目录不存在
        是 writer 打不开的最常见原因**)。
        """
        base, _ = os.path.splitext(path)
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)      # 关键: 目录不存在会让 writer 打不开

        fps = self._fps if self._fps and self._fps > 0 else 30.0
        want_h264 = self._fourcc_out in self._H264_TAGS

        # --- 1) 首选: PyAV 显式设置每帧抓取 PTS，真正保留可变帧率时间轴 ---
        pyav_encoder = self._H264_ENCODER if want_h264 else "mpeg4"
        timestamped = _PyAVVideoWriter(
            base + ".mp4", self._w, self._h, fps,
            encoder=pyav_encoder, crf=self._H264_CRF,
            preset=self._H264_PRESET, threads=self._H264_THREADS)
        if timestamped.isOpened():
            self._timestamp_mode = "capture-vfr"
            codec_text = "H.264" if want_h264 else "MPEG-4"
            print(f"[相机] 录制编码: {codec_text} (PyAV / {pyav_encoder}, "
                  f"真实抓帧 PTS, preset={self._H264_PRESET}, "
                  f"threads={self._H264_THREADS or 'auto'}) -> {base + '.mp4'}")
            return timestamped, base + ".mp4"
        print(f"[相机] ⚠ PyAV VFR 写入器不可用，回退固定帧率编码: "
              f"{timestamped.error or '未知错误'}")
        self._timestamp_mode = "fixed-fps-fallback"

        # --- 2) 兼容: 系统 ffmpeg + libx264(或指定的 H.264 编码器) ---
        if want_h264:
            enc = self._H264_ENCODER
            if _ffmpeg_has_encoder(enc):
                out_path = base + ".mp4"
                ff = _FFmpegVideoWriter(out_path, self._w, self._h, fps, encoder=enc,
                                        crf=self._H264_CRF, preset=self._H264_PRESET,
                                        threads=self._H264_THREADS)
                if ff.isOpened():
                    extra = (f", crf={self._H264_CRF}, preset={self._H264_PRESET}, "
                             f"threads={self._H264_THREADS or 'auto'}"
                             if enc in ("libx264", "libx265") else "")
                    print(f"[相机] 录制编码: H.264 (系统 ffmpeg / {enc}{extra}) -> {out_path}")
                    return ff, out_path
                if ff.error:                   # 起不来: 打印原因, 落到 cv2 兜底
                    print(f"[相机] ⚠ 系统 ffmpeg 用 {enc} 起不来, 回退 opencv 编码器: "
                          f"{ff.error[:200]}")
            else:
                print(f"[相机] ⚠ 系统未找到 ffmpeg 或其无 {enc}; 改用 opencv 自带编码器写 H.264 "
                      f"(很多构建不可用, 可能退回 mp4v)。稳妥做法: `sudo apt install ffmpeg`。")

        # --- 3) 兜底: cv2.VideoWriter 按 fourcc(H.264 优先, 再 mp4v / AVI) ---
        candidates = []
        for c in (self._fourcc_out, "avc1", "H264", "h264", "X264"):
            if c and (c, ".mp4") not in candidates:
                candidates.append((c, ".mp4"))
        for c in ("mp4v",):                    # MPEG-4 Part 2, 软件编码几乎总可用
            if (c, ".mp4") not in candidates:
                candidates.append((c, ".mp4"))
        candidates += [("MJPG", ".avi"), ("XVID", ".avi")]

        for fourcc, ext in candidates:
            out_path = base + ext
            try:
                vw = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*fourcc),
                                     fps, (self._w, self._h))
            except Exception:
                vw = None
            if vw is not None and vw.isOpened():
                if fourcc in self._H264_TAGS:
                    print(f"[相机] 录制编码: H.264 (opencv 自带编码器 / {fourcc}) -> {out_path}")
                elif want_h264:
                    print(f"[相机] ⚠ 本机无可用 H.264 编码器, 退回 {fourcc}{ext}(非 H.264)。"
                          f" 想要 H.264: 装系统 ffmpeg(`sudo apt install ffmpeg`, 带 libx264), "
                          f"或录完用 `ffmpeg -i 输入 -c:v libx264 输出.mp4` 转一次。")
                else:
                    print(f"[相机] 录制编码: {fourcc}{ext} -> {out_path}")
                return vw, out_path
            if vw is not None:
                vw.release()
            # 失败时 OpenCV 可能留下 0 字节残file, 删掉
            if os.path.exists(out_path) and os.path.getsize(out_path) == 0:
                try:
                    os.remove(out_path)
                except OSError:
                    pass
        return None, path

    def start_recording(self, mp4_path: str) -> str:
        """开始把画面录制到 ``mp4_path``。

        抓帧与编码分别运行在线程中，中间使用有界队列。抓帧线程只负责持续清空 UVC
        缓冲，编码抖动不会再直接阻塞 ``cap.read()``；队列溢出会被显式计数。

        未 open 会自动 open。返回实际写入的 MP4 路径。
        """
        if self._recording:
            return self._mp4_path
        if not self.is_open():
            self.open()
        if not mp4_path.endswith(".mp4"):
            mp4_path = mp4_path + ".mp4"

        writer, out_path = self._make_writer(mp4_path)
        if writer is None:
            raise RuntimeError(
                f"无法创建 VideoWriter ({self._w}x{self._h}@{self._fps:.0f}fps): "
                f"试过的编码/容器都打不开。多半是这个 OpenCV 没带可用的 MP4 编码器 "
                f"(FFmpeg 后端)。修法: `pip install --upgrade --force-reinstall "
                f"opencv-python`(别用 headless 精简版), 或系统装 ffmpeg "
                f"(`sudo apt install ffmpeg`)。用探测脚本可看哪种编码能用。")

        self._writer = writer
        self._mp4_path = out_path
        self._frame_ts = []
        self._n = 0
        self._captured_n = 0
        self._read_failures = 0
        self._queue_dropped = 0
        self._writer_failures = 0
        self._max_queue_depth = 0
        self._t0_us = None
        self._latest = None
        self._encode_queue = queue.Queue(maxsize=max(1, int(self._ENCODE_QUEUE_SIZE)))
        self._recording = True
        self._encoder_thread = threading.Thread(
            target=self._encode_loop, name="camera-encoder", daemon=True)
        self._capture_thread = threading.Thread(
            target=self._capture_loop, name="camera-capture", daemon=True)
        self._thread = self._capture_thread
        self._encoder_thread.start()
        self._capture_thread.start()
        return out_path

    def _capture_loop(self) -> None:
        """抓帧线程：持续清空 UVC 缓冲，把 ``(帧, 抓取时间)`` 放入编码队列。"""
        while self._recording:
            ok, frame = self._cap.read()
            host_us = _host_ts_us()
            if not self._recording:
                break
            if not ok:
                self._read_failures += 1
                time.sleep(0.001)
                continue
            if self._t0_us is None:
                self._t0_us = host_us
            self._captured_n += 1
            self._latest = frame        # 供预览用；不会再等待编码完成
            try:
                self._encode_queue.put_nowait((frame, host_us))
                self._max_queue_depth = max(
                    self._max_queue_depth, self._encode_queue.qsize())
            except queue.Full:
                # 不阻塞抓帧线程；VFR PTS 会在下一帧中保留这段时间空洞。
                self._queue_dropped += 1

    def _encode_loop(self) -> None:
        """编码线程：按抓帧时间写帧，队列清空并收到哨兵后退出。"""
        while True:
            item = self._encode_queue.get()
            if item is self._QUEUE_STOP:
                return
            frame, host_us = item
            try:
                try:
                    result = self._writer.write(frame, host_us)
                except TypeError:
                    # OpenCV VideoWriter 兼容路径没有 timestamp 参数。
                    result = self._writer.write(frame)
                opened = getattr(self._writer, "isOpened", lambda: True)()
                if result is False or not opened:
                    self._writer_failures += 1
                    continue
            except Exception:
                self._writer_failures += 1
                continue
            self._frame_ts.append(int(host_us))
            self._n += 1

    def grab_preview(self):
        """给预览用: 返回一帧(BGR)或 None。

        录制中返回后台线程缓存的最新帧(**不与录制抢帧**); 空闲时自己 read() 抓一帧。
        调用方若要在帧上叠加文字, 请先 ``.copy()`` —— 录制中返回的是共享帧。
        """
        if self._recording:
            return self._latest
        if not self.is_open():
            return None
        img, _ = self.read()
        return img

    def stop_recording(self) -> None:
        """停止抓帧，排空编码队列并 finalize MP4 (幂等)。"""
        if (not self._recording and self._capture_thread is None
                and self._encoder_thread is None and self._writer is None):
            return
        self._recording = False
        if self._capture_thread is not None:
            self._capture_thread.join(timeout=2.0)
        if self._encode_queue is not None and self._encoder_thread is not None:
            # 即使抓帧驱动短暂阻塞，循环返回后也会先检查 recording，不会在哨兵后入队。
            self._encode_queue.put(self._QUEUE_STOP)
            self._encoder_thread.join()
        if self._writer is not None:
            self._writer.release()
            self._writer = None
        self._capture_thread = None
        self._encoder_thread = None
        self._thread = None
        self._encode_queue = None
        if self._queue_dropped or self._read_failures or self._writer_failures:
            print(f"[相机] ⚠ 抓帧/编码统计: captured={self._captured_n}, "
                  f"encoded={self._n}, queue_dropped={self._queue_dropped}, "
                  f"read_failures={self._read_failures}, "
                  f"writer_failures={self._writer_failures}")

    def save_recording(self, path: Optional[str] = None) -> Tuple[Optional[str], "np.ndarray"]:
        """停止录制并返回 ``(mp4_path, 每帧主机时间戳数组)``。

        MP4 在录制过程中已逐帧写盘, 这里只负责收尾。返回的时间戳数组与手套
        ``t_host_us`` 同一时钟, 用于落盘后对齐 (见 record_session.py)。
        """
        self.stop_recording()
        return self._mp4_path, self.frame_timestamps()

    # -- 时间戳 / 计数 -------------------------------------------------------

    def frame_timestamps(self) -> "np.ndarray":
        """本次录制每一帧的主机时间戳 (µs, int64), 长度 == MP4 帧数。"""
        return np.asarray(self._frame_ts, dtype=np.int64)

    def total_frames(self) -> int:
        return self._n

    def capture_stats(self) -> dict:
        """返回本次录制的抓帧、队列和编码统计。"""
        return {
            "captured_frames": int(self._captured_n),
            "encoded_frames": int(self._n),
            "queue_dropped_frames": int(self._queue_dropped),
            "read_failures": int(self._read_failures),
            "writer_failures": int(self._writer_failures),
            "max_queue_depth": int(self._max_queue_depth),
            "queue_capacity": int(self._ENCODE_QUEUE_SIZE),
            "timestamp_mode": self._timestamp_mode,
        }

    @property
    def recording(self) -> bool:
        return self._recording

    @property
    def size(self) -> Tuple[int, int]:
        """实际采集分辨率 (width, height)。"""
        return (self._w, self._h)

    # -- 快速预览 (源自 gopro_hdmi.py, 用于确认采集卡是否正常) ---------------

    @staticmethod
    def preview(device: int = 0, width: int = 1920, height: int = 1080,
                fps: int = 60, process=None) -> None:
        """打开采集卡实时预览, 窗口里按 q 退出。用于确认设备编号/画面是否正常。

        ``process`` 可选: 传入 BGR 图像、返回要显示的图像 (如转灰度)。
        注意: 需要有 GUI 环境 (cv2.imshow); 无显示器的服务器上不要用。
        """
        cam = GoProHDMICamera(device, width, height, fps).open()
        print("按 q 退出预览。")
        try:
            while True:
                img, _ = cam.read()
                if img is None:
                    continue
                cv2.imshow("GoPro HDMI", process(img) if process else img)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        finally:
            cam.close()
            cv2.destroyAllWindows()


# 向后兼容: 旧代码 import 的名字是 GoProVideo
GoProVideo = GoProCamera


# --------------------------------- 用法示例 --------------------------------- #
if __name__ == "__main__":
    video_path = "./data/GX012458.MP4"
    thm_path = "./data/GX012458.THM"
    with GoProCamera(video_path, thm_path) as video:
        print(f"总帧数: {len(video)}, 帧率: {video.fps:.3f} fps")
        f = video.get_frame(100)            # 或 video[100]
        print(f"第 {f.index} 帧: shape={f.image.shape}, "
              f"相对时间={f.time_sec:.3f}s, 真实时刻={f.timestamp}")
