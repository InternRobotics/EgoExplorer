#!/usr/bin/env python3
"""在 Windows 上读取 SteamVR/OpenXR Tracker，并发送给 Linux 采集端。"""

from __future__ import annotations

import argparse
import math
import os
import socket
import threading
import time
from contextlib import suppress

from network_bridge.protocol import PROTOCOL_VERSION, recv_message, send_message


def _now_us():
    return time.time_ns() // 1_000


class TrackerSenderClient:
    def __init__(self, host, port=8766, token=""):
        self.host = str(host)
        self.port = int(port)
        self.token = str(token or "")
        self._sock = None
        self._lock = threading.RLock()
        self._send_lock = threading.Lock()
        self._reader = None
        self._stop = threading.Event()
        self._control_id = 0
        self.last_error = ""

    @property
    def connected(self):
        with self._lock:
            return self._sock is not None

    def connect(self, timeout=3.0):
        sock = socket.create_connection((self.host, self.port), timeout=timeout)
        try:
            send_message(sock, {
                "type": "auth", "protocol": PROTOCOL_VERSION, "token": self.token})
            hello = recv_message(sock)
            if (hello.get("type") != "hello"
                    or int(hello.get("protocol", -1)) != PROTOCOL_VERSION):
                raise ConnectionError(hello.get("message") or "Linux 接收端握手失败")
            sock.settimeout(None)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except Exception:
            sock.close()
            raise
        with self._lock:
            old, self._sock = self._sock, sock
            self.last_error = ""
        if old is not None:
            with suppress(OSError):
                old.close()
        self._reader = threading.Thread(
            target=self._reader_loop, args=(sock,), name="tracker-sender-reader",
            daemon=True)
        self._reader.start()

    def send_frame(self, sequence, poses, timestamp_us):
        self._send({
            "type": "frame",
            "sequence": int(sequence),
            "t_source_host_us": int(timestamp_us),
            "poses": poses,
        })

    def send_control(self, command, countdown=None):
        """通过 Tracker TCP 连接向 Linux 采集端 发送 r/e 采集指令。"""
        command = str(command or "").strip().lower()
        if command not in ("r", "e"):
            raise ValueError("只支持 r/e 控制指令")
        if command == "r" and countdown is not None:
            countdown = float(countdown)
            if not math.isfinite(countdown) or not 0 <= countdown <= 3600:
                raise ValueError("倒计时必须在 0～3600 秒之间")
        else:
            countdown = None
        with self._lock:
            self._control_id += 1
            request_id = self._control_id
        message = {
            "type": "control",
            "id": request_id,
            "command": command,
        }
        if countdown is not None:
            message["countdown"] = countdown
        self._send(message)
        return request_id

    def _send(self, message):
        with self._send_lock:
            with self._lock:
                sock = self._sock
            if sock is None:
                raise ConnectionError("尚未连接 Linux 接收端")
            try:
                send_message(sock, message)
            except Exception as exc:
                with self._lock:
                    if sock is self._sock:
                        self._sock = None
                        self.last_error = f"{type(exc).__name__}: {exc}"
                with suppress(OSError):
                    sock.shutdown(socket.SHUT_RDWR)
                with suppress(OSError):
                    sock.close()
                raise

    def _reader_loop(self, sock):
        try:
            while not self._stop.is_set():
                message = recv_message(sock)
                if message.get("type") == "ping":
                    t1 = _now_us()
                    self._send({
                        "type": "pong",
                        "id": message.get("id"),
                        "t0_linux_us": int(message.get("t0_linux_us", 0)),
                        "t1_windows_us": t1,
                        "t2_windows_us": _now_us(),
                    })
                elif message.get("type") == "control_result":
                    command = str(message.get("command") or "?").upper()
                    if message.get("ok"):
                        print(f"[control] {command} ✔ Linux 采集端 已接收")
                    else:
                        detail = message.get("message") or "Linux 采集端 拒绝了指令"
                        print(f"[control] {command} ✖ {detail}")
        except Exception as exc:
            if not self._stop.is_set():
                self.last_error = f"{type(exc).__name__}: {exc}"
        finally:
            with self._lock:
                if sock is self._sock:
                    self._sock = None
            with suppress(OSError):
                sock.close()

    def close(self):
        self._stop.set()
        with self._lock:
            sock, self._sock = self._sock, None
        if sock is not None:
            with suppress(OSError):
                sock.shutdown(socket.SHUT_RDWR)
            with suppress(OSError):
                sock.close()
        if self._reader is not None and self._reader is not threading.current_thread():
            self._reader.join(timeout=2.0)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Windows SteamVR/OpenXR Tracker -> Linux EgoDataCollection")
    parser.add_argument("--host", required=True, help="Linux 采集机的局域网 IP")
    parser.add_argument("--port", type=int, default=8766, help="TCP 端口，默认 8766")
    parser.add_argument(
        "--token", default=os.environ.get("EGO_TRACKER_TOKEN", ""),
        help="可选共享密钥；也可用 EGO_TRACKER_TOKEN")
    parser.add_argument(
        "--reconnect", type=float, default=2.0, help="断线重连间隔秒数，默认 2")
    parser.add_argument(
        "--countdown", type=float, default=None,
        help="按 r 时覆盖 Linux 采集端 的开始倒计时（0～3600 秒）")
    parser.add_argument(
        "--send-fps", type=float, default=90.0,
        help="Tracker 网络发送帧率上限，默认 90；避免 headless OpenXR 空转发送")
    parser.add_argument(
        "--no-keyboard", action="store_true",
        help="禁用 Windows 终端的 r/e 远程采集快捷键")
    return parser


def _read_windows_key():
    try:
        import msvcrt
    except ImportError:
        raise RuntimeError("键盘快捷键只能在 Windows CMD/PowerShell 中使用") from None
    key = msvcrt.getwch()
    if key in ("\x00", "\xe0"):
        msvcrt.getwch()
        return ""
    return key.lower()


def run_keyboard_controls(sender, stop_event=None, countdown=None, read_key=None):
    """把 Windows 终端按键发送到 Linux 采集端；read_key 用于自动化测试。"""
    stop_event = stop_event or threading.Event()
    read_key = read_key or _read_windows_key
    countdown_text = "Linux 采集端 默认值" if countdown is None else f"{countdown:g} 秒"
    print("\n[control] Windows 远程采集键盘已启用：")
    print(f"[control]   R = 开始采集（倒计时：{countdown_text}）")
    print("[control]   E = 结束、保存本次采集")
    print("[control]   Esc = 关闭快捷键监听（Tracker 继续发送）\n")
    while not stop_event.is_set():
        try:
            key = read_key()
        except (EOFError, StopIteration):
            return
        if key == "\x1b":
            print("[control] 键盘快捷键监听已关闭；Tracker 继续发送。")
            return
        if key not in ("r", "e"):
            continue
        try:
            sender.send_control(key, countdown if key == "r" else None)
            print(f"[control] {key.upper()} → 指令已发送，等待 Linux 采集端 确认")
        except (ConnectionError, OSError, ValueError) as exc:
            print(f"[control] {key.upper()} ✖ 未发送：{exc}")


def main(argv=None):
    args = build_parser().parse_args(argv)
    if (args.countdown is not None
            and (not math.isfinite(args.countdown)
                 or not 0 <= args.countdown <= 3600)):
        raise SystemExit("--countdown 必须是 0～3600 之间的有限数字")
    if (not math.isfinite(args.send_fps)
            or not 0 < args.send_fps <= 240):
        raise SystemExit("--send-fps 必须是大于 0 且不超过 240 的有限数字")
    try:
        from devices.vive_tracker import ViveTracker
    except Exception as exc:
        raise SystemExit(
            "无法加载 pyopenxr（xr 模块）。请在 Windows 执行：\n"
            "  python -m pip install -r requirements-tracker-windows.txt\n"
            f"原始错误: {type(exc).__name__}: {exc}") from None
    reconnect = max(0.2, float(args.reconnect))
    sender = TrackerSenderClient(args.host, args.port, args.token)
    tracker = ViveTracker()
    sequence = 0
    next_reconnect = 0.0
    stop_event = threading.Event()
    keyboard_thread = None
    send_interval = 1.0 / args.send_fps
    try:
        tracker.connect()
        print("[tracker] ✔ SteamVR/OpenXR 已启动。")
        print(f"[network] Tracker 发送帧率上限: {args.send_fps:g} fps")
        for item in tracker.get_connected_trackers():
            print(f"[tracker]   serial={item['serial']}  role={item['role']}")
        if not args.no_keyboard:
            keyboard_thread = threading.Thread(
                target=run_keyboard_controls,
                args=(sender, stop_event, args.countdown),
                name="tracker-control-keyboard",
                daemon=True,
            )
            keyboard_thread.start()
        while not tracker.should_exit and not stop_event.is_set():
            cycle_started = time.perf_counter()
            poses = tracker.poll()
            source_us = _now_us()
            if not sender.connected and time.monotonic() >= next_reconnect:
                try:
                    sender.connect()
                    print(f"[network] ✔ 已连接 Linux {args.host}:{args.port}")
                except Exception as exc:
                    next_reconnect = time.monotonic() + reconnect
                    print(f"[network] 等待 Linux {args.host}:{args.port}: {exc}")
            if sender.connected:
                sequence += 1
                try:
                    sender.send_frame(sequence, poses, source_us)
                except Exception as exc:
                    next_reconnect = time.monotonic() + reconnect
                    print(f"[network] 发送中断，将自动重连: {exc}")
            remaining = send_interval - (time.perf_counter() - cycle_started)
            if remaining > 0:
                stop_event.wait(remaining)
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        sender.close()
        tracker.disconnect()
        print("[tracker] 已退出。")


if __name__ == "__main__":
    main()
