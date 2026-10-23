"""Linux 采集端的 Windows Tracker TCP 接收器。"""

from __future__ import annotations

import math
import socket
import threading
import time
from collections import deque
from contextlib import suppress

from devices.base import BaseTracker
from network_bridge.protocol import PROTOCOL_VERSION, recv_message, send_message


def _now_us():
    return time.time_ns() // 1_000


REMOTE_CONTROL_COMMANDS = frozenset(("r", "e"))


def _normalise_poses(value):
    """校验网络输入并只保留有限浮点数的标准位姿。"""
    if not isinstance(value, dict):
        return {}
    result = {}
    for raw_role, raw_pose in value.items():
        if not isinstance(raw_pose, dict):
            continue
        position = raw_pose.get("position")
        orientation = raw_pose.get("orientation")
        if not isinstance(position, dict) or not isinstance(orientation, dict):
            continue
        try:
            p = {key: float(position[key]) for key in ("x", "y", "z")}
            q = {key: float(orientation[key]) for key in ("w", "x", "y", "z")}
        except (KeyError, TypeError, ValueError):
            continue
        if not all(math.isfinite(item) for item in (*p.values(), *q.values())):
            continue
        role = str(raw_role).strip()
        if role:
            result[role] = {"position": p, "orientation": q}
    return result


class NetworkTrackerReceiver(BaseTracker):
    """接收 Windows OpenXR 位姿，并换算到 Linux 协调端墙钟。"""

    background_driven = True

    def __init__(self, host="0.0.0.0", port=8766, token="",
                 device_name="tracker_0", max_record_fps=120.0):
        super().__init__()
        self.host = str(host)
        self.port = int(port)
        self.token = str(token or "")
        self._device_name = str(device_name)
        self._lock = threading.RLock()
        self._send_lock = threading.Lock()
        self._stop = threading.Event()
        self._server = None
        self._client = None
        self._accept_thread = None
        self._client_thread = None
        self._listening = False
        self._sender_connected = False
        self._peer = None
        self._last_error = "等待 Windows 发送端"
        self._last_frame_us = 0
        self._last_sequence = None
        self._dropped_frames = 0
        self._ping_id = 0
        self._offset_samples = deque(maxlen=20)
        self._windows_minus_linux_us = None
        self._clock_rtt_us = None
        self._control_handler = None
        self._max_record_fps = max(0.0, float(max_record_fps))
        self._min_record_interval_us = (
            int(round(1_000_000 / self._max_record_fps))
            if self._max_record_fps > 0 else 0)
        self._last_record_received_us = 0
        self._rate_limited_frames = 0

    @property
    def device_name(self):
        return self._device_name

    def connect(self):
        if self._listening:
            return
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((self.host, self.port))
        server.listen(2)
        server.settimeout(0.5)
        self.port = int(server.getsockname()[1])
        self._server = server
        self._listening = True
        self._stop.clear()
        self._accept_thread = threading.Thread(
            target=self._accept_loop, name="tracker-net-accept", daemon=True)
        self._accept_thread.start()

    def disconnect(self):
        self._stop.set()
        with self._lock:
            server, self._server = self._server, None
            client, self._client = self._client, None
            self._listening = False
            self._sender_connected = False
        for sock in (client, server):
            if sock is None:
                continue
            with suppress(OSError):
                sock.shutdown(socket.SHUT_RDWR)
            with suppress(OSError):
                sock.close()
        for thread in (self._client_thread, self._accept_thread):
            if thread is not None and thread is not threading.current_thread():
                thread.join(timeout=2.0)

    def is_connected(self):
        # 服务已监听即可开始一次会话；Windows 稍后上线的帧也能进入同一会话。
        return self._listening

    def sender_connected(self):
        with self._lock:
            return self._sender_connected

    def set_control_handler(self, handler):
        """设置 Windows 控制指令回调，签名为 ``handler(command, countdown)``。"""
        if handler is not None and not callable(handler):
            raise TypeError("control handler 必须可调用或为 None")
        with self._lock:
            self._control_handler = handler

    def _refresh(self):
        with self._lock:
            return dict(self.latest_poses)

    def poll(self):
        return self._refresh()

    def get_all_poses(self, refresh=True):
        return self._refresh()

    def start_recording(self):
        with self._lock:
            super().start_recording()
            self._last_record_received_us = 0
            self._rate_limited_frames = 0

    def stop_recording(self):
        with self._lock:
            super().stop_recording()

    def discard(self):
        with self._lock:
            super().discard()

    def total_frames(self):
        with self._lock:
            return super().total_frames()

    def host_ts_range(self):
        with self._lock:
            return super().host_ts_range()

    def tracker_info(self):
        with self._lock:
            return {
                **super().tracker_info(),
                "transport": "windows-tcp",
                "source_clock": "windows-wall-clock",
                "host_clock": "linux-wall-clock",
                "dropped_frames": int(self._dropped_frames),
                "record_fps_limit": self._max_record_fps,
                "record_rate_limited": int(self._rate_limited_frames),
            }

    def save_recording(self, path):
        with self._lock:
            return super().save_recording(path)

    def status(self):
        now = _now_us()
        with self._lock:
            return {
                "mode": "windows-network",
                "listening": self._listening,
                "connected": self._sender_connected,
                "endpoint": f"{self.host}:{self.port}",
                "peer": self._peer,
                "last_error": self._last_error,
                "last_frame_age_ms": (
                    round((now - self._last_frame_us) / 1000.0, 1)
                    if self._last_frame_us else None),
                "clock_offset_us": (
                    round(self._windows_minus_linux_us, 1)
                    if self._windows_minus_linux_us is not None else None),
                "clock_rtt_us": self._clock_rtt_us,
                "transport_dropped": int(self._dropped_frames),
                "remote_control": self._control_handler is not None,
                "record_fps_limit": self._max_record_fps,
                "record_rate_limited": int(self._rate_limited_frames),
            }

    def _accept_loop(self):
        while not self._stop.is_set():
            try:
                client, peer = self._server.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            with self._lock:
                busy = self._sender_connected
            if busy:
                with suppress(Exception):
                    send_message(client, {"type": "error", "message": "已有发送端连接"})
                client.close()
                continue
            client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            thread = threading.Thread(
                target=self._client_loop, args=(client, peer),
                name="tracker-net-client", daemon=True)
            self._client_thread = thread
            thread.start()

    def _authenticate(self, client):
        client.settimeout(5.0)
        auth = recv_message(client)
        valid = (
            auth.get("type") == "auth"
            and int(auth.get("protocol", -1)) == PROTOCOL_VERSION
            and (not self.token or auth.get("token") == self.token)
        )
        if not valid:
            send_message(client, {"type": "error", "message": "认证失败或协议不兼容"})
            raise ConnectionError("tracker 发送端认证失败")
        send_message(client, {
            "type": "hello",
            "protocol": PROTOCOL_VERSION,
            "service": "linux-tracker-receiver",
            "t_linux_us": _now_us(),
        })
        client.settimeout(None)

    def _client_loop(self, client, peer):
        ping_thread = None
        try:
            self._authenticate(client)
            with self._lock:
                self._client = client
                self._sender_connected = True
                self._peer = f"{peer[0]}:{peer[1]}"
                self._last_error = ""
                self._last_sequence = None
                self._offset_samples.clear()
                self._windows_minus_linux_us = None
                self._clock_rtt_us = None
            print(f"[tracker] ✔ Windows 发送端已连接: {self._peer}")
            ping_thread = threading.Thread(
                target=self._ping_loop, args=(client,),
                name="tracker-net-clock", daemon=True)
            ping_thread.start()
            while not self._stop.is_set():
                message = recv_message(client)
                received_us = _now_us()
                kind = message.get("type")
                if kind == "frame":
                    self._handle_frame(message, received_us)
                elif kind == "pong":
                    self._handle_pong(message, received_us)
                elif kind == "control":
                    self._handle_control(client, message)
        except Exception as exc:
            if not self._stop.is_set():
                print(f"[tracker] ⚠ Windows 发送端断开: {exc}")
                with self._lock:
                    self._last_error = f"{type(exc).__name__}: {exc}"
        finally:
            with self._lock:
                if client is self._client:
                    self._client = None
                    self._sender_connected = False
                    self._peer = None
            with suppress(OSError):
                client.close()
            if ping_thread is not None and ping_thread is not threading.current_thread():
                ping_thread.join(timeout=2.5)

    def _send(self, client, message):
        with self._send_lock:
            send_message(client, message)

    def _ping_loop(self, client):
        while not self._stop.is_set():
            with self._lock:
                if client is not self._client:
                    return
                self._ping_id += 1
                ping_id = self._ping_id
            try:
                self._send(client, {
                    "type": "ping", "id": ping_id, "t0_linux_us": _now_us()})
            except Exception:
                return
            if self._stop.wait(2.0):
                return

    def _handle_pong(self, message, received_us):
        t0 = int(message.get("t0_linux_us", 0))
        t1 = int(message.get("t1_windows_us", 0))
        t2 = int(message.get("t2_windows_us", 0))
        if t0 <= 0 or t1 <= 0 or t2 < t1:
            return
        rtt = max(0, (received_us - t0) - (t2 - t1))
        offset = ((t1 - t0) + (t2 - received_us)) / 2.0
        with self._lock:
            self._offset_samples.append((rtt, offset))
            best_rtt, best_offset = min(self._offset_samples, key=lambda item: item[0])
            self._clock_rtt_us = int(best_rtt)
            self._windows_minus_linux_us = float(best_offset)

    def _handle_control(self, client, message):
        """校验并转发来自已认证 Windows Tracker 连接的 r/e 指令。"""
        request_id = message.get("id")
        command = str(message.get("command") or "").strip().lower()
        countdown = message.get("countdown")
        response = {
            "type": "control_result",
            "id": request_id,
            "command": command,
            "ok": False,
        }
        try:
            if command not in REMOTE_CONTROL_COMMANDS:
                raise ValueError("只允许远程发送 r/e")
            if command == "r" and countdown is not None:
                countdown = float(countdown)
                if not math.isfinite(countdown) or not 0 <= countdown <= 3600:
                    raise ValueError("倒计时必须在 0～3600 秒之间")
            else:
                countdown = None
            with self._lock:
                handler = self._control_handler
            if handler is None:
                raise RuntimeError("Linux 采集端 采集控制尚未就绪")
            handler(command, countdown)
            response.update(ok=True, message="Linux 采集端 已接收控制指令")
            print(f"[control] ✔ Windows -> Linux 采集端: {command.upper()}")
        except Exception as exc:
            response["message"] = str(exc)
            print(f"[control] ⚠ 拒绝 Windows 控制指令: {exc}")
        self._send(client, response)

    def _linux_timestamp(self, windows_us, received_us):
        with self._lock:
            offset = self._windows_minus_linux_us
        if offset is None:
            return int(received_us)
        converted = int(round(int(windows_us) - offset))
        if abs(converted - int(received_us)) > 5_000_000:
            return int(received_us)
        return converted

    def _handle_frame(self, message, received_us):
        poses = _normalise_poses(message.get("poses"))
        source_us = int(message.get("t_source_host_us") or received_us)
        host_us = self._linux_timestamp(source_us, received_us)
        sequence = int(message.get("sequence", 0))
        with self._lock:
            if (self._last_sequence is not None
                    and sequence > self._last_sequence + 1):
                self._dropped_frames += sequence - self._last_sequence - 1
            self._last_sequence = sequence
            self._last_frame_us = received_us
            self.latest_poses = poses
            if self._recording:
                if (self._min_record_interval_us > 0
                        and self._last_record_received_us > 0
                        and received_us - self._last_record_received_us
                        < self._min_record_interval_us):
                    self._rate_limited_frames += 1
                    return
                # Tracker 没有稳定的跨进程设备时钟，主时间轴使用换算后的 Linux 墙钟。
                self._last_record_received_us = received_us
                super().record_frame(
                    poses, t_host_us=host_us, t_dev_us=None,
                    t_source_host_us=source_us)
