"""Camera、Manus/Wuji 手套和 tracker 的连接与生命周期管理。"""

import asyncio
import threading
import traceback
from dataclasses import dataclass, field

import cv2

from devices.manus_glove import MANUS_STREAMS, ManusGlove, ManusReceiver
from devices.wuji_connect import connect_wuji_local
from .capture import apply_mp4_fps, open_camera


@dataclass
class ConnectedDevices:
    """一次采集进程中的设备集合，负责 tracker 轮询和统一清理。"""

    camera: object = None
    receiver: object = None
    gloves: list = field(default_factory=list)
    tracker: object = None
    glove_link: object = None
    shutdown_glove_link: bool = False
    _tracker_thread: threading.Thread = None
    _tracker_stop: threading.Event = field(default_factory=threading.Event)
    _glove_tasks: list = field(default_factory=list)

    @property
    def available(self):
        return self.camera is not None or bool(self.gloves) or self.tracker is not None

    def start_tracker_polling(self):
        """启动 tracker 后台接收/轮询。网络 tracker 自带接收线程，无需重复轮询。"""
        if self.tracker is None or self._tracker_thread is not None:
            return
        if getattr(self.tracker, "background_driven", False):
            return

        def poll_loop():
            while (not self._tracker_stop.is_set()
                   and not getattr(self.tracker, "should_exit", False)):
                try:
                    self.tracker.poll()
                except Exception:
                    print("\n[tracker] 轮询线程异常退出:", flush=True)
                    traceback.print_exc()
                    break

        self._tracker_thread = threading.Thread(
            target=poll_loop, name="tracker-poll", daemon=True)
        self._tracker_thread.start()
        print("[tracker] 位姿轮询线程已启动。")

    def start_glove_receivers(self, stop_event):
        """启动 Wuji 的异步接收任务；Manus 由 UDP 后台线程接收。"""
        if self._glove_tasks:
            return self._glove_tasks
        for glove in self.gloves:
            create_tasks = getattr(glove, "create_recv_tasks", None)
            if callable(create_tasks):
                self._glove_tasks.extend(create_tasks(stop_event))
        return self._glove_tasks

    async def stop_glove_receivers(self):
        """取消并等待全部异步手套接收任务退出。"""
        tasks, self._glove_tasks = self._glove_tasks, []
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def close(self):
        """按 tracker、手套、receiver、camera 的顺序关闭所有资源。"""
        self._tracker_stop.set()
        if self._tracker_thread is not None:
            self._tracker_thread.join(timeout=2.0)

        if self.tracker is not None:
            try:
                self.tracker.disconnect()
            except Exception:
                pass
        for glove in self.gloves:
            try:
                glove.disconnect()
            except Exception:
                pass
        if self.glove_link is not None:
            try:
                self.glove_link.close(shutdown_agent=self.shutdown_glove_link)
            except Exception:
                pass
        if self.receiver is not None:
            try:
                self.receiver.stop()
            except Exception:
                pass
        if self.camera is not None:
            self.camera.close()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


def _connect_camera(args):
    if args.no_camera:
        print("[相机] --no-camera：跳过相机。")
        return None

    codec = "avc1" if args.video_codec == "h264" else "mp4v"
    device = int(args.cam_device) if str(args.cam_device).isdigit() else args.cam_device
    camera = open_camera(
        device,
        args.cam_width,
        args.cam_height,
        args.cam_fps,
        allow_scan=not args.no_cam_scan,
        fourcc_out=codec,
        crf=args.crf,
        h264_preset=args.h264_preset,
        h264_threads=args.h264_threads,
    )
    if camera is not None and args.mp4_fps is not None:
        if args.mp4_fps <= 0:
            print(f"[相机] ⚠ 忽略非法 --mp4-fps {args.mp4_fps}，"
                  f"使用采集帧率 {float(camera.fps):g}。")
        else:
            apply_mp4_fps(camera, args.mp4_fps)
    return camera


def _connect_manus(args):
    """启动 Manus UDP 接收器并构造固定的左右手设备。"""
    receiver = ManusReceiver(ip=args.manus_ip, port=args.manus_port, verbose=True)
    try:
        receiver.start()
    except OSError as ex:
        print(f"[manus] ✖ 无法在 {args.manus_ip}:{args.manus_port} 监听 UDP: {ex}")
        print("[manus]   端口被占用? 换 --manus-port，或先关掉别的接收程序。")
        return None, []

    gloves = ManusGlove.discover(receiver)
    for glove in gloves:
        glove.connect()
        glove.subscribe(MANUS_STREAMS)

    if args.manus_wait and args.manus_wait > 0:
        got_data = receiver.wait_for_data(args.manus_wait)
        seen = receiver.seen_slots()
        if got_data and seen:
            print(f"[manus] 检测到数据: {sorted(seen)} 手在流。")
        else:
            print(f"[manus] {args.manus_wait:g}s 内暂未收到 UDP 数据 —— "
                  f"确认发送端已启动并发往 {args.manus_ip}:{args.manus_port}; "
                  "录制开始后一旦有数据会自动缓存。")
    return receiver, gloves


def _connect_local_tracker(args):
    """连接 Manus 模式原有的本机 OpenXR Tracker。"""
    try:
        from devices.vive_tracker import ViveTracker
        tracker = ViveTracker(sample_fps=args.tracker_sample_fps)
        tracker.connect()
        print("[tracker] ✔ 已连接 HTC Vive Tracker (OpenXR)。")
        try:
            for item in tracker.get_connected_trackers():
                print(f"[tracker]   connected: serial={item['serial']}  "
                      f"role={item['role']}")
        except Exception as ex:
            print(f"[tracker]   (暂时无法枚举 tracker: {ex!r})")
        return tracker
    except Exception as ex:
        print(f"[warn] 连接 Vive tracker 失败，本次不记录位姿: {ex}")
        return None


def _connect_network_tracker(args):
    """启动 Wuji 模式使用的 Windows Tracker 网络接收器。"""
    if args.no_tracker:
        print("[tracker] --no-tracker：跳过网络接收服务。")
        return None
    try:
        from devices.network_tracker import NetworkTrackerReceiver
        tracker = NetworkTrackerReceiver(
            host=args.tracker_host,
            port=args.tracker_port,
            token=args.tracker_token,
            max_record_fps=args.tracker_record_fps,
        )
        tracker.connect()
        print(f"[tracker] ✔ 等待 Windows OpenXR 发送端连接 "
              f"{args.tracker_host}:{args.tracker_port}。")
        return tracker
    except Exception as ex:
        print(f"[warn] 启动 tracker 网络接收服务失败，本次不记录位姿: {ex}")
        return None


def connect_devices(args, glove_type=None):
    """按手套类型连接设备，并隔离两种采集链路。"""
    glove_type = glove_type or getattr(args, "glove_type", "manus")
    camera = _connect_camera(args)
    if glove_type == "wuji":
        receiver, glove_link = None, None
        gloves = connect_wuji_local(args)
        tracker = _connect_network_tracker(args)
    else:
        receiver, gloves = _connect_manus(args)
        glove_link = None
        tracker = None if args.no_tracker else _connect_local_tracker(args)
    return ConnectedDevices(
        camera=camera,
        receiver=receiver,
        gloves=gloves,
        tracker=tracker,
        glove_link=glove_link,
        shutdown_glove_link=False,
    )
