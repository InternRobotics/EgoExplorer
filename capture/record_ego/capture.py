"""设备采集、会话落盘以及录制控制。"""

import asyncio
import csv
import json
import math
import os
import sys
import threading
import time
from datetime import date, datetime

import numpy as np

from devices.gopro_camera import GoProHDMICamera
from devices.manus_glove import RESERVED_STREAMS
from .config import (
    CAM_DEVICE,
    CAM_WIDTH,
    CAM_HEIGHT,
    CAM_FPS,
    MP4_CRF,
    H264_PRESET,
    H264_THREADS,
    DEFAULT_SAVE_DIR as SAVE_DIR,
    STATUS_INTERVAL,
)
from .tasks import task_filename_component


RECORDING_LOG_DIRNAME = "record_logs"
RECORDING_LOG_FILENAME_TEMPLATE = "recording_log_{date}.csv"
RECORDING_LOG_FIELDS = (
    "session",
    "started_at",
    "ended_at",
    "start_time_us",
    "end_time_us",
    "duration_seconds",
    "duration_minutes",
    "video_frames",
    "glove_frames",
    "tracker_frames",
    "session_dir",
)

# ============================================================
# 终端键盘 + 相机初始化
# ============================================================

def setup_keyboard(loop):
    """跨平台单键读取，返回 ``(asyncio.Queue, restore)``。"""
    q = asyncio.Queue()

    # 服务进程、IDE 后台任务和重定向输入时 stdin 可能不是终端。网页控制仍然
    # 使用同一个队列，因此此时只需禁用本地单键监听，不能让整个服务退出。
    if not sys.stdin.isatty():
        return q, lambda: None

    if os.name == "nt":
        import msvcrt

        stop = threading.Event()

        def _reader():
            while not stop.is_set():
                if msvcrt.kbhit():
                    ch = msvcrt.getwch()
                    if ch in ("\x00", "\xe0"):
                        if msvcrt.kbhit():
                            msvcrt.getwch()
                        continue
                    loop.call_soon_threadsafe(q.put_nowait, ch)
                else:
                    time.sleep(0.01)

        threading.Thread(target=_reader, name="kbd", daemon=True).start()

        def restore():
            stop.set()

        return q, restore

    import termios
    import tty

    fd = sys.stdin.fileno()
    try:
        old = termios.tcgetattr(fd)
    except (OSError, termios.error):
        return q, lambda: None
    tty.setcbreak(fd)

    def on_key():
        try:
            ch = os.read(fd, 1).decode(errors="ignore")
        except OSError:
            return
        if ch:
            q.put_nowait(ch)

    loop.add_reader(fd, on_key)

    def restore():
        try:
            loop.remove_reader(fd)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)

    return q, restore


def _probe_frame(camera, n=15):
    for _ in range(n):
        image, _ = camera.read()
        if image is not None:
            return image
    return None


def open_camera(device=CAM_DEVICE, width=CAM_WIDTH, height=CAM_HEIGHT, fps=CAM_FPS,
                allow_scan=True, fourcc_out="avc1", crf=MP4_CRF,
                h264_preset=H264_PRESET, h264_threads=H264_THREADS):
    """打开采集卡并打印状态；失败时返回 ``None``，其它设备继续工作。"""
    print("-" * 52)
    print(f"[相机] 尝试打开采集卡 {device!r}  请求 {width}x{height}@{fps}fps (后端 V4L2)  "
          f"编码={fourcc_out}" + (
              f"(crf={crf}, preset={h264_preset}, threads={h264_threads or 'auto'})"
              if fourcc_out != "mp4v" else ""))
    try:
        camera = GoProHDMICamera(
            device, width, height, fps, fourcc_out=fourcc_out).open()
        camera._H264_CRF = int(crf)
        camera._H264_PRESET = str(h264_preset)
        camera._H264_THREADS = int(h264_threads)
    except Exception as ex:
        print(f"[相机] ✖ 指定设备打开失败: {type(ex).__name__}: {ex}")
        camera = None

    if camera is not None:
        actual_width, actual_height = camera.size
        print(f"[相机] ✔ 已打开 (来源 {camera._opened_target})  "
              f"实际 {actual_width}x{actual_height}@{camera.fps:.0f}fps")
        image = _probe_frame(camera)
        if image is not None:
            print(f"[相机]   画面读取正常 (首帧 {image.shape[1]}x{image.shape[0]})。")
            print("-" * 52)
            return camera
        print("[相机] ⚠ 打开成功但读不到画面 —— GoPro 未开机 / HDMI 未连 / "
              "输入选错，或此节点非采集口。")
        if not allow_scan:
            print("[相机]   仍返回该设备；若始终无帧，保存时视频帧数为 0。")
            print("-" * 52)
            return camera
        camera.close()

    if allow_scan:
        print("[相机] 自动扫描 /dev/video* 寻找可用采集口 …")
        camera = GoProHDMICamera.autodetect(
            width, height, fps, prefer=device, fourcc_out=fourcc_out)
        if camera is not None:
            camera._H264_CRF = int(crf)
            camera._H264_PRESET = str(h264_preset)
            camera._H264_THREADS = int(h264_threads)
            actual_width, actual_height = camera.size
            print(f"[相机] ✔ 自动选用 (来源 {camera._opened_target})  "
                  f"实际 {actual_width}x{actual_height}@{camera.fps:.0f}fps")
            print("-" * 52)
            return camera

    print("[相机] ✖ 未找到可用采集卡 → 本次不生成 .mp4 / .video.npz。")
    print("[相机]   排查: `v4l2-ctl --list-devices` 看设备号；或用 "
          "--cam-device 指定，如 --cam-device /dev/video1")
    print("-" * 52)
    return None


def apply_mp4_fps(camera, mp4_fps):
    """覆盖编码器名义帧率；VFR 模式仍以逐帧抓取时间决定实际播放时刻。"""
    if camera is None or mp4_fps is None:
        return None if camera is None else float(camera.fps)
    native = float(camera.fps)
    try:
        camera.fps = float(mp4_fps)
    except (AttributeError, TypeError):
        try:
            camera._fps = float(mp4_fps)
        except Exception:
            pass
    effective = float(camera.fps)
    if abs(effective - float(mp4_fps)) > 1e-6:
        print(f"[相机] ⚠ 无法覆盖 MP4 帧率，仍按 {effective:g} fps 写入。")
    else:
        print(f"[相机] MP4 名义帧率 = {effective:g} fps  "
              f"(采集请求 {native:g} fps；VFR 逐帧 PTS 对齐不受影响)")
    return effective


# ============================================================
# 会话落盘 + 录制控制
# ============================================================

def glove_type_of(gloves):
    """按驱动类型返回 ``manus`` / ``wuji``，供元数据与页面标识使用。"""
    for glove in gloves:
        cls = type(glove)
        text = f"{cls.__module__}.{cls.__name__}".lower()
        if "wuji" in text:
            return "wuji"
    return "manus"


def save_all(gloves, save_dir=SAVE_DIR, path=None, clear_after=False):
    """把多只统一 BaseGlove 驱动的录制缓存合并为一个 ``.npz``。"""
    active = [glove for glove in gloves if glove.total_frames() > 0]
    if not active:
        print("\n  没有采集到手套数据，未写 .npz")
        if clear_after:
            for glove in gloves:
                glove.discard()
        return None

    arrays = {}
    channels = {}
    for glove in active:
        glove_arrays, glove_channels, _ = glove.export_arrays()
        arrays.update(glove_arrays)
        channels.update(glove_channels)

    streams = sorted({key.split("/", 1)[1] for key in channels})
    glove_type = glove_type_of(active)
    meta = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "device_type": f"{glove_type}_glove",
        "streams": streams,
        "gloves": [glove.glove_info() for glove in active],
        "ts_unit": ("microseconds. t_us=device-preferred (host fallback); "
                    "t_dev_us=device clock (-1 if unavailable); "
                    "t_host_us=host computer wall clock (time.time)"),
        "channels": channels,
    }
    if glove_type == "manus":
        meta["reserved_streams"] = list(RESERVED_STREAMS)
        meta["note"] = ("Manus 仅有 hand_skeleton / hand_joint_angles; "
                        "触觉流为保留占位，缺省不产生数据。")
    else:
        meta["note"] = ("Wuji SDK 精简流：原始触觉矩阵与手部骨骼；"
                        "为保证视频实时抓帧，默认不订阅派生分区、关节角、EMF 和点云。")
    arrays["meta"] = np.array(json.dumps(meta, ensure_ascii=False))

    if path is None:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(
            save_dir, datetime.now().strftime("%Y%m%d_%H%M%S") + ".npz")
    elif not path.endswith(".npz"):
        path += ".npz"
    np.savez_compressed(path, **arrays)

    if clear_after:
        for glove in gloves:
            glove.discard()
    return path


def _session_dir(save_dir, base):
    return os.path.join(save_dir, base)


def _recording_log_date(value=None):
    """把日期参数统一为本地日期；默认使用今天。"""
    if value is None:
        return datetime.now().astimezone().date()
    if isinstance(value, datetime):
        return value.astimezone().date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    for pattern in ("%Y-%m-%d", "%Y%m%d"):
        try:
            return datetime.strptime(text, pattern).date()
        except ValueError:
            pass
    raise ValueError(f"无效的录制日志日期: {value!r}")


def recording_log_directory(save_dir=SAVE_DIR):
    """返回数据目录同级的独立日志目录。"""
    data_dir = os.path.abspath(os.fspath(save_dir))
    return os.path.join(os.path.dirname(data_dir), RECORDING_LOG_DIRNAME)


def recording_log_path(save_dir=SAVE_DIR, record_date=None):
    day = _recording_log_date(record_date)
    filename = RECORDING_LOG_FILENAME_TEMPLATE.format(date=day.isoformat())
    return os.path.join(recording_log_directory(save_dir), filename)


def load_recording_log_total(save_dir=SAVE_DIR, record_date=None):
    """从指定日期的 CSV 恢复当天保存时长和会话数。损坏行会被忽略。"""
    path = recording_log_path(save_dir, record_date)
    total_seconds = 0.0
    rows = 0
    try:
        with open(path, "r", encoding="utf-8", newline="") as file_obj:
            for row in csv.DictReader(file_obj):
                try:
                    seconds = float(row.get("duration_seconds") or "")
                    if not math.isfinite(seconds) or seconds < 0:
                        raise ValueError
                except (TypeError, ValueError):
                    try:
                        seconds = float(row.get("duration_minutes") or "") * 60.0
                    except (TypeError, ValueError):
                        continue
                    if not math.isfinite(seconds) or seconds < 0:
                        continue
                total_seconds += seconds
                rows += 1
    except OSError:
        pass
    return total_seconds, rows


def append_recording_log(save_dir, entry, record_date=None):
    """向对应日期的录制日志追加一个已保存会话并立即刷盘。"""
    if record_date is None:
        end_time_us = entry.get("end_time_us")
        try:
            record_date = datetime.fromtimestamp(
                int(end_time_us) / 1_000_000).astimezone().date()
        except (TypeError, ValueError, OverflowError, OSError):
            record_date = None
    path = recording_log_path(save_dir, record_date)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    needs_header = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", encoding="utf-8", newline="") as file_obj:
        writer = csv.DictWriter(file_obj, fieldnames=RECORDING_LOG_FIELDS)
        if needs_header:
            writer.writeheader()
        writer.writerow({name: entry.get(name, "") for name in RECORDING_LOG_FIELDS})
        file_obj.flush()
        os.fsync(file_obj.fileno())
    return path


def refresh_daily_recording_total(state, save_dir=SAVE_DIR, now=None):
    """日期变化时把内存累计值切换为当天日志中的统计。"""
    day = _recording_log_date(now)
    day_text = day.isoformat()
    if state.recording_log_date != day_text:
        (state.total_recording_seconds,
         state.saved_recordings) = load_recording_log_total(save_dir, day)
        state.recording_log_date = day_text
    return day


def _host_range(gloves):
    """返回所有手套主机时间戳的 ``(min, max)``，无数据时返回 ``None``。"""
    lower = upper = None
    for glove in gloves:
        timestamp_range = glove.host_ts_range()
        if timestamp_range is None:
            continue
        start, end = timestamp_range
        lower = start if lower is None else min(lower, start)
        upper = end if upper is None else max(upper, end)
    return None if lower is None else (int(lower), int(upper))


def save_session(camera, gloves, base, save_dir=SAVE_DIR, tracker=None,
                 task_name=None, glove_type=None):
    """保存 MP4、手套 NPZ、tracker NPZ、视频时间戳和会话元信息。"""
    session_dir = _session_dir(save_dir, base)
    os.makedirs(session_dir, exist_ok=True)
    out = {
        "mp4": None,
        "npz": None,
        "video_npz": None,
        "tracker_npz": None,
        "meta": None,
    }

    # 先关闭所有数据源的录制开关，再做视频收尾和 NPZ 压缩。否则前面的慢 I/O
    # 会让后面的设备继续缓存数秒，造成各路结束时间不一致和额外负载。
    camera_was_recording = camera is not None and camera.recording
    tracker_was_recording = tracker is not None and tracker.recording
    for glove in gloves:
        glove.stop_recording()
    if tracker_was_recording:
        tracker.stop_recording()

    video_ts = np.empty(0, dtype=np.int64)
    if camera_was_recording:
        mp4_path, video_ts = camera.save_recording()
        out["mp4"] = mp4_path

    if gloves:
        out["npz"] = save_all(
            gloves,
            save_dir=session_dir,
            path=os.path.join(session_dir, base + ".npz"),
            clear_after=False,
        )

    tracker_info = None
    tracker_range = None
    if tracker_was_recording:
        tracker_info = tracker.tracker_info()
        tracker_range = tracker.host_ts_range()
        tracker_npz = os.path.join(session_dir, base + ".tracker.npz")
        tracker.save_recording(tracker_npz)
        out["tracker_npz"] = tracker_npz

    if out["mp4"] is not None:
        capture_stats_fn = getattr(camera, "capture_stats", None)
        capture_stats = (dict(capture_stats_fn())
                         if callable(capture_stats_fn) else {})
        video_npz = os.path.join(session_dir, base + ".video.npz")
        np.savez_compressed(
            video_npz,
            t_host_us=video_ts,
            fps=np.array(float(camera.fps)),
            width=np.array(int(camera.size[0])),
            height=np.array(int(camera.size[1])),
            mp4=np.array(os.path.basename(out["mp4"])),
            timestamp_mode=np.array(
                capture_stats.get("timestamp_mode", "unknown")),
            captured_frames=np.array(
                int(capture_stats.get("captured_frames", video_ts.size))),
            queue_dropped_frames=np.array(
                int(capture_stats.get("queue_dropped_frames", 0))),
            read_failures=np.array(
                int(capture_stats.get("read_failures", 0))),
            writer_failures=np.array(
                int(capture_stats.get("writer_failures", 0))),
            created=np.array(datetime.now().isoformat(timespec="seconds")),
        )
        out["video_npz"] = video_npz
    else:
        capture_stats = {}

    video_range = (
        (int(video_ts[0]), int(video_ts[-1])) if video_ts.size else None)
    glove_range = _host_range(gloves)
    overlap_s = None
    if video_range and glove_range:
        lower = max(video_range[0], glove_range[0])
        upper = min(video_range[1], glove_range[1])
        overlap_s = max(0.0, (upper - lower) / 1e6)

    glove_type = glove_type or glove_type_of(gloves)
    if glove_type == "wuji":
        clock_description = (
            "video and Wuji use Linux host wall clock; Windows tracker timestamps "
            "are converted to the Linux wall clock by NTP-style sampling (all in µs)")
    else:
        clock_description = (
            "video, Manus glove and local tracker use host wall clock "
            "(time.time, µs); already aligned")
    meta = {
        "base": base,
        "task_name": task_name,
        "created": datetime.now().isoformat(timespec="seconds"),
        "clock": clock_description,
        "glove_type": glove_type,
        "video": {
            "mp4": os.path.basename(out["mp4"]) if out["mp4"] else None,
            "timestamps_npz": (
                os.path.basename(out["video_npz"]) if out["video_npz"] else None),
            "frames": int(video_ts.size),
            "fps": float(camera.fps) if camera is not None else None,
            "size": ([int(camera.size[0]), int(camera.size[1])]
                     if camera is not None else None),
            "t_host_us_range": list(video_range) if video_range else None,
            "timestamp_mode": capture_stats.get("timestamp_mode"),
            "captured_frames": capture_stats.get("captured_frames"),
            "queue_dropped_frames": capture_stats.get("queue_dropped_frames", 0),
            "read_failures": capture_stats.get("read_failures", 0),
            "writer_failures": capture_stats.get("writer_failures", 0),
            "max_queue_depth": capture_stats.get("max_queue_depth"),
            "queue_capacity": capture_stats.get("queue_capacity"),
        },
        "gloves_npz": os.path.basename(out["npz"]) if out["npz"] else None,
        "gloves": [glove.glove_info() for glove in gloves],
        "glove_t_host_us_range": list(glove_range) if glove_range else None,
        "tracker_npz": (
            os.path.basename(out["tracker_npz"]) if out["tracker_npz"] else None),
        "tracker": tracker_info,
        "tracker_t_host_us_range": list(tracker_range) if tracker_range else None,
        "overlap_s": overlap_s,
    }
    meta_path = os.path.join(session_dir, base + ".session.json")
    with open(meta_path, "w", encoding="utf-8") as file_obj:
        json.dump(meta, file_obj, ensure_ascii=False, indent=2)
    out["meta"] = meta_path

    print(f"\n  ✔ 会话已保存: {base}/   (文件夹: {session_dir})")
    print(f"    视频 {meta['video']['frames']} 帧 -> {meta['video']['mp4']}")
    if camera is not None and video_ts.size >= 2:
        span_s = (int(video_ts[-1]) - int(video_ts[0])) / 1e6
        if span_s > 0:
            measured = (video_ts.size - 1) / span_s
            print(f"    实测平均帧率 ≈ {measured:.2f} fps   "
                  f"(写入 MP4 用 {float(camera.fps):g} fps)")
    if capture_stats:
        print(f"    相机抓取 {capture_stats.get('captured_frames', 0)} 帧 / "
              f"编码 {capture_stats.get('encoded_frames', video_ts.size)} 帧 / "
              f"队列丢弃 {capture_stats.get('queue_dropped_frames', 0)} 帧 / "
              f"时间轴 {capture_stats.get('timestamp_mode', 'unknown')}")
    print(f"    手套 -> {meta['gloves_npz']}   "
          f"视频时间戳 -> {meta['video']['timestamps_npz']}")
    if out["tracker_npz"]:
        roles = tracker_info["roles"] if tracker_info else []
        print(f"    tracker 位姿 -> {os.path.basename(out['tracker_npz'])}"
              f"   ({tracker_info['frames']} 帧, roles={roles})")
    if overlap_s is not None:
        print(f"    视频/手套重叠 ≈ {overlap_s:.2f}s (同一主机时钟，已对齐)")
    print(f"    关联信息 -> {os.path.basename(meta_path)}")

    for glove in gloves:
        glove.discard()
    if tracker is not None:
        tracker.discard()
    return out


def save_session_and_log(camera, gloves, base, save_dir, tracker, log_entry,
                         task_name=None, glove_type=None):
    """在同一个后台工作项中保存会话并追加时长日志。"""
    out = save_session(
        camera, gloves, base, save_dir, tracker, task_name, glove_type)
    try:
        log_path = append_recording_log(save_dir, log_entry)
        log_error = None
    except Exception as exc:
        log_path = None
        log_error = f"{type(exc).__name__}: {exc}"
    return out, log_path, log_error


def _discard_session(camera, gloves, base, save_dir=SAVE_DIR, tracker=None):
    """终止并丢弃当前录制，删除半成品视频和空会话目录。"""
    video_count = 0
    session_dir = _session_dir(save_dir, base) if base else None
    if camera is not None and camera.recording:
        video_count = camera.total_frames()
        camera.stop_recording()
        mp4_path = (
            os.path.join(session_dir, base + ".mp4") if session_dir else None)
        if mp4_path and os.path.exists(mp4_path):
            try:
                os.remove(mp4_path)
            except OSError:
                pass

    glove_count = sum(glove.total_frames() for glove in gloves)
    for glove in gloves:
        glove.discard()
    if tracker is not None and tracker.recording:
        tracker.discard()
    if session_dir and os.path.isdir(session_dir):
        try:
            if not os.listdir(session_dir):
                os.rmdir(session_dir)
        except OSError:
            pass
    return video_count, glove_count


class RecordingState:
    def __init__(self, countdown_seconds=0.0, storage_limit_bytes=0,
                 task_names=None, task_setting_path=None):
        self.base = None
        self.t0_perf = 0.0
        self.started_at = None
        self.start_time_us = 0
        self.total_recording_seconds = 0.0
        self.saved_recordings = 0
        self.recording_log_date = None
        countdown_seconds = float(countdown_seconds)
        self.countdown_default = (
            min(3600.0, max(0.0, countdown_seconds))
            if math.isfinite(countdown_seconds) else 0.0)
        self.countdown_deadline = 0.0
        self.countdown_total = 0.0
        self.saving = False
        self.storage_limit_bytes = max(0, int(storage_limit_bytes))
        self.storage_bytes = 0
        self.storage_files = 0
        self.storage_over_limit = False
        self.storage_scanned_at = None
        self.storage_message = "等待首次容量检查"
        self.task_names = tuple(task_names or ())
        self.task_setting_path = task_setting_path
        self.task_index = 0
        self.active_task_name = None

    @property
    def current_task_name(self):
        if not self.task_names:
            return None
        return self.task_names[self.task_index]

    @property
    def displayed_task_name(self):
        return self.active_task_name or self.current_task_name

    def prepare_task(self):
        self.active_task_name = self.current_task_name
        return self.active_task_name

    def abandon_task(self):
        """取消或丢弃时保留任务序号，让下次继续采集同一任务。"""
        self.active_task_name = None

    def finish_task(self):
        """完成保存后推进一项；到列表末尾后从第一项继续。"""
        finished = self.active_task_name
        if self.task_names:
            self.task_index = (self.task_index + 1) % len(self.task_names)
        self.active_task_name = None
        return finished

    def select_relative_task(self, offset):
        """空闲时按相对位置手动选择任务；任务列表首尾相接。"""
        if self.active_task_name is not None:
            raise RuntimeError("当前任务已经开始，不能切换")
        if not self.task_names:
            return None
        self.task_index = (self.task_index + int(offset)) % len(self.task_names)
        return self.current_task_name

    @property
    def counting_down(self):
        return self.countdown_deadline > 0.0

    def countdown_remaining(self):
        if not self.counting_down:
            return 0.0
        return max(0.0, self.countdown_deadline - time.perf_counter())

    def begin_countdown(self, seconds=None):
        seconds = self.countdown_default if seconds is None else float(seconds)
        seconds = min(3600.0, max(0.0, seconds))
        self.countdown_total = seconds
        self.countdown_deadline = time.perf_counter() + seconds if seconds > 0 else 0.0
        return seconds

    def cancel_countdown(self):
        was_counting = self.counting_down
        self.countdown_deadline = 0.0
        self.countdown_total = 0.0
        return was_counting

    def clear_active_recording(self):
        self.base = None
        self.t0_perf = 0.0
        self.started_at = None
        self.start_time_us = 0


def any_recording(camera, gloves, tracker=None):
    camera_recording = camera is not None and camera.recording
    tracker_recording = tracker is not None and tracker.recording
    return camera_recording or tracker_recording or any(
        glove.recording for glove in gloves)


def video_frames(camera):
    return camera.total_frames() if camera is not None else 0


def glove_frames(gloves):
    return sum(glove.total_frames() for glove in gloves)


def tracker_frames(tracker):
    return tracker.total_frames() if tracker is not None else 0


def _emit_start_beep():
    """倒计时结束提示音；网页端还会播放更可靠的 Web Audio 鸣笛。"""
    if os.name == "nt":
        try:
            import winsound
            winsound.Beep(1200, 450)
            return
        except Exception:
            pass
    # Linux 终端是否实际响铃取决于终端/桌面设置，但不会引入额外依赖。
    sys.stdout.write("\a")
    sys.stdout.flush()


def _start_recording(camera, gloves, state, save_dir, tracker=None):
    """真正启动各设备记录；只在倒计时结束后调用。"""
    if camera is None and not gloves and tracker is None:
        state.cancel_countdown()
        state.abandon_task()
        print("\n  ⚠ 当前没有可用采集设备，不能开始记录。")
        return
    refresh_daily_recording_total(state, save_dir)
    _emit_start_beep()
    state.cancel_countdown()
    task_component = task_filename_component(
        state.active_task_name or state.current_task_name)
    base = datetime.now().strftime("session_%Y%m%d_%H%M%S_") + task_component
    state.base = base
    session_dir = _session_dir(save_dir, base)
    os.makedirs(session_dir, exist_ok=True)
    if camera is not None:
        camera.start_recording(os.path.join(session_dir, base + ".mp4"))
    for glove in gloves:
        glove.start_recording()
    if tracker is not None:
        tracker.start_recording()
    state.t0_perf = time.perf_counter()
    start_wall = time.time()
    state.start_time_us = int(start_wall * 1_000_000)
    state.started_at = datetime.fromtimestamp(
        start_wall).astimezone().isoformat(timespec="milliseconds")

    parts = []
    if camera is not None:
        parts.append("视频")
    if gloves:
        parts.append(f"手套×{len(gloves)}")
    if tracker is not None:
        parts.append("tracker")
    print(f"\n  ● 开始记录 [{' + '.join(parts) if parts else '无设备'}] "
          f"(session={base})")


def _record_command(value):
    """兼容终端字符串指令和网页传入的 ``(指令, 倒计时秒数)``。"""
    if isinstance(value, (tuple, list)):
        ch = str(value[0]).lower()[:1] if value else ""
        seconds = value[1] if len(value) > 1 else None
        return ch, seconds
    return str(value).lower()[:1], None


async def control_loop(q, camera, gloves, state, save_dir, stop_event, tracker=None,
                       uploading_fn=None, saved_fn=None, glove_type=None):
    while not stop_event.is_set():
        if state.counting_down and state.countdown_remaining() <= 0.0:
            _start_recording(camera, gloves, state, save_dir, tracker)
            continue
        try:
            command = await asyncio.wait_for(q.get(), timeout=0.05)
        except asyncio.TimeoutError:
            continue
        ch, requested_countdown = _record_command(command)

        if ch == "r":
            upload_running = bool(uploading_fn and uploading_fn())
            if (any_recording(camera, gloves, tracker) or state.counting_down
                    or state.saving or upload_running):
                print("\n  当前正忙（记录/倒计时/保存/上传），不能开始新记录")
                continue
            task_name = state.prepare_task()
            print(f"\n  开始采集{task_name}任务")
            seconds = state.begin_countdown(requested_countdown)
            if seconds > 0:
                print(f"\n  ◷ {seconds:g} 秒后开始记录（按 d 可取消）")
            else:
                _start_recording(camera, gloves, state, save_dir, tracker)

        elif ch == "d":
            if state.cancel_countdown():
                state.abandon_task()
                print("\n  ■ 已取消开始记录倒计时")
                continue
            if not any_recording(camera, gloves, tracker):
                print("\n  当前未在记录")
                continue
            tracker_count = (
                tracker_frames(tracker)
                if tracker is not None and tracker.recording else 0)
            video_count, glove_count = _discard_session(
                camera, gloves, state.base, save_dir, tracker)
            state.clear_active_recording()
            state.abandon_task()
            tracker_text = (
                f" / tracker {tracker_count} 帧" if tracker is not None else "")
            print(f"\n  ■ 已终止并丢弃，未落盘(视频 {video_count} 帧 / "
                  f"手套 {glove_count} 帧{tracker_text})")

        elif ch == "e":
            if state.counting_down:
                print("\n  正在倒计时；按 d 可取消")
                continue
            if not any_recording(camera, gloves, tracker):
                print("\n  当前未在记录")
                continue
            ended_perf = time.perf_counter()
            ended_wall = time.time()
            ended_datetime = datetime.fromtimestamp(ended_wall).astimezone()
            refresh_daily_recording_total(state, save_dir, ended_datetime)
            duration_seconds = max(0.0, ended_perf - state.t0_perf)
            base = state.base
            task_name = state.active_task_name
            counts = {
                "video_frames": video_frames(camera),
                "glove_frames": glove_frames(gloves),
                "tracker_frames": tracker_frames(tracker),
            }
            log_entry = {
                "session": base,
                "started_at": state.started_at,
                "ended_at": ended_datetime.isoformat(timespec="milliseconds"),
                "start_time_us": state.start_time_us,
                "end_time_us": int(ended_wall * 1_000_000),
                "duration_seconds": f"{duration_seconds:.6f}",
                "duration_minutes": f"{duration_seconds / 60.0:.6f}",
                **counts,
                "session_dir": _session_dir(save_dir, base),
            }
            state.saving = True
            try:
                _out, log_path, log_error = (
                    await asyncio.get_running_loop().run_in_executor(
                        None, save_session_and_log, camera, gloves, base,
                        save_dir, tracker, log_entry, task_name, glove_type))
                if log_path:
                    print(f"    录制日志 -> {log_path}  "
                          f"({duration_seconds / 60.0:.3f} 分钟)")
                else:
                    print(f"\n  ⚠ 写入录制日志失败: {log_error}")
                state.total_recording_seconds += duration_seconds
                state.saved_recordings += 1
                state.clear_active_recording()
                state.finish_task()
                if saved_fn is not None:
                    saved_fn()
            finally:
                state.saving = False

        elif ch == "q":
            if state.cancel_countdown():
                state.abandon_task()
                print("\n  已取消开始记录倒计时")
            if any_recording(camera, gloves, tracker):
                print("\n  ⚠ 正在记录，退出将丢弃未保存的数据")
                _discard_session(camera, gloves, state.base, save_dir, tracker)
                state.clear_active_recording()
                state.abandon_task()
            print("\n  退出。")
            stop_event.set()
            return


async def status_loop(camera, gloves, state, stop_event, tracker=None):
    while not stop_event.is_set():
        await asyncio.sleep(STATUS_INTERVAL)
        if any_recording(camera, gloves, tracker):
            duration = (
                time.perf_counter() - state.t0_perf if state.t0_perf > 0 else 0.0)
            video_text = str(video_frames(camera)) if camera is not None else "off"
            tracker_text = (
                f"  tracker={tracker_frames(tracker)}" if tracker is not None else "")
            sys.stdout.write(
                f"\r  ● REC {duration:6.1f}s  video={video_text}  "
                f"glove={glove_frames(gloves)}{tracker_text}      ")
            sys.stdout.flush()
