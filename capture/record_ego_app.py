"""Ego 数据采集应用的设备生命周期与主流程编排。"""

import asyncio
import math
import os
import threading
import time
import webbrowser
from concurrent.futures import Future, TimeoutError as FutureTimeoutError
from datetime import datetime, time as datetime_time

from record_ego.capture import (
    RecordingState,
    any_recording,
    control_loop,
    glove_frames,
    refresh_daily_recording_total,
    recording_log_directory,
    recording_log_path,
    setup_keyboard,
    status_loop,
    tracker_frames,
    video_frames,
)
from record_ego.config import parse_args
from record_ego.devices import connect_devices
from record_ego.tasks import load_task_setting
from record_ego.upload import RcloneUploader, directory_usage
from record_ego.web import (
    CameraPump,
    HandTap,
    TactileTap,
    WebContext,
    assign_sides,
    hand_parents_snapshot,
    install_live_skeleton_tap,
    make_server,
    read_tracker_snapshot,
)


async def main(glove_type="manus", argv=None):
    args = parse_args(argv, glove_type=glove_type)
    glove_type = args.glove_type
    try:
        task_setting_path, task_names = load_task_setting(args.task_setting)
    except ValueError as exc:
        raise SystemExit(f"--task-setting 无效：{exc}") from None
    loop = asyncio.get_running_loop()
    save_dir = args.save_dir
    if not math.isfinite(args.storage_limit_gb) or args.storage_limit_gb < 0:
        raise SystemExit("--storage-limit-gb 必须是大于等于 0 的有限数字")
    if (not math.isfinite(args.storage_check_interval)
            or args.storage_check_interval <= 0):
        raise SystemExit("--storage-check-interval 必须是大于 0 的有限数字")
    if args.h264_threads < 0:
        raise SystemExit("--h264-threads 必须是大于等于 0 的整数")
    if args.preview_width < 0 or args.preview_height < 0:
        raise SystemExit("--preview-width 和 --preview-height 必须是大于等于 0 的整数")
    if (glove_type == "wuji"
            and (not math.isfinite(args.tracker_record_fps)
                 or args.tracker_record_fps < 0)):
        raise SystemExit("--tracker-record-fps 必须是大于等于 0 的有限数字")
    if (glove_type == "manus"
            and (not math.isfinite(args.tracker_sample_fps)
                 or not 0 <= args.tracker_sample_fps <= 240)):
        raise SystemExit("--tracker-sample-fps 必须是 0～240 之间的有限数字")
    try:
        storage_limit_bytes = int(args.storage_limit_gb * (1024 ** 3))
    except (OverflowError, ValueError):
        raise SystemExit("--storage-limit-gb 数值过大") from None
    storage_check_interval_minutes = args.storage_check_interval
    storage_check_interval_seconds = max(
        1.0, storage_check_interval_minutes * 60.0)

    stop_event = asyncio.Event()
    devices = connect_devices(args, glove_type=glove_type)
    camera = devices.camera
    gloves = devices.gloves
    tracker = devices.tracker
    install_live_skeleton_tap(gloves)
    hand_sides = assign_sides(gloves)
    devices.start_glove_receivers(stop_event)

    if not devices.available:
        print("[warn] 当前没有可用的采集卡、手套或 tracker；"
              "网页仍会启动并显示离线状态。")

    storage_check_event = asyncio.Event()
    state = RecordingState(
        args.record_countdown, storage_limit_bytes,
        task_names=task_names, task_setting_path=task_setting_path)
    os.makedirs(recording_log_directory(save_dir), exist_ok=True)
    refresh_daily_recording_total(state, save_dir)
    uploader = RcloneUploader(
        save_dir,
        args.rclone_remote,
        executable=args.rclone_bin,
        config_path=args.rclone_config,
        on_finished=lambda: loop.call_soon_threadsafe(storage_check_event.set),
    )

    # 终端键盘：尽早建好控制队列 q（网页 /cmd 的 cmd_fn 会把指令塞进同一个 q，
    # 与终端键盘、网页键盘三者并行汇入 control_loop）
    q, restore = setup_keyboard(loop)

    def enqueue_control(ch, countdown=None):
        """把本机、网页或 Windows Tracker 指令汇入同一采集状态机。"""
        command = (ch, countdown) if ch == "r" and countdown is not None else ch
        loop.call_soon_threadsafe(q.put_nowait, command)

    set_control_handler = getattr(tracker, "set_control_handler", None)
    if glove_type == "wuji" and callable(set_control_handler):
        set_control_handler(enqueue_control)

    devices.start_tracker_polling()

    # --- 实时取数 & 状态 ---
    hand_tap = HandTap(gloves, hand_sides, use_export_fallback=not args.no_hand_export)
    tactile_tap = TactileTap(gloves, hand_sides)
    pump = None

    def glove_status():
        if glove_type == "wuji":
            status = getattr(devices.glove_link, "status", None)
            if callable(status):
                result = dict(status())
                result.setdefault("label", "Wuji")
                return result
            return {
                "label": "Wuji",
                "mode": "local",
                "connected": any(glove.is_connected() for glove in gloves),
                "endpoint": None,
            }
        try:
            seen_slots = sorted(devices.receiver.seen_slots())
        except Exception:
            seen_slots = []
        return {
            "label": "Manus",
            "mode": "udp",
            "connected": bool(seen_slots),
            "endpoint": f"{args.manus_ip}:{args.manus_port}",
            "seenSlots": seen_slots,
        }

    def tracker_status():
        status = getattr(tracker, "status", None)
        if callable(status):
            return status()
        return {
            "mode": "disabled" if tracker is None else "local",
            "listening": tracker is not None,
            "connected": tracker is not None,
            "endpoint": None,
        }

    def status_fn():
        rec_on = any_recording(camera, gloves, tracker)
        dur = ((time.perf_counter() - state.t0_perf)
               if rec_on and state.t0_perf > 0 else 0.0)
        return rec_on, dur

    def live_fn():
        now = datetime.now().astimezone()
        today = refresh_daily_recording_total(state, save_dir, now)
        rec_on = any_recording(camera, gloves, tracker)
        dur = ((time.perf_counter() - state.t0_perf)
               if rec_on and state.t0_perf > 0 else 0.0)
        countdown = state.countdown_remaining()
        today_active_seconds = 0.0
        if rec_on and state.start_time_us > 0:
            today_start = datetime.combine(
                today, datetime_time.min, tzinfo=now.tzinfo).timestamp()
            today_active_seconds = max(
                0.0, time.time() - max(state.start_time_us / 1_000_000,
                                       today_start))
        total_seconds = state.total_recording_seconds + today_active_seconds
        return {
            "rec": bool(rec_on),
            "dur": round(dur, 2),
            "countingDown": state.counting_down,
            "countdown": round(countdown, 2),
            "saving": state.saving,
            "taskName": state.displayed_task_name,
            "taskIndex": state.task_index + 1,
            "taskCount": len(state.task_names),
            "totalRecordingMinutes": round(total_seconds / 60.0, 4),
            "savedRecordings": state.saved_recordings,
            "recordingDate": state.recording_log_date,
            "recordingLog": recording_log_path(save_dir, today),
            "storage": {
                "bytes": state.storage_bytes,
                "files": state.storage_files,
                "limitBytes": state.storage_limit_bytes,
                "overLimit": state.storage_over_limit,
                "scannedAt": state.storage_scanned_at,
                "message": state.storage_message,
            },
            "frames": {
                "video": video_frames(camera),
                "glove": glove_frames(gloves),
                "tracker": tracker_frames(tracker),
            },
            "hands": hand_tap.snapshot(),
            # Manus 的节点顺序由 SDK 决定；随实时快照下发，数据晚到或
            # 层级更新后页面可立即重建正确连线。
            "handParents": hand_parents_snapshot(
                gloves, hand_sides, glove_type),
            "tactile": tactile_tap.snapshot() if glove_type == "wuji" else {},
            "trackers": read_tracker_snapshot(tracker),
            "hasTracker": tracker is not None,
            "hasCaptureDevice": devices.available,
            "cameraAvailable": camera is not None,
            "cameraOnline": bool(pump is not None and pump.online()),
            "gloveStatus": glove_status(),
            "wujiStatus": glove_status() if glove_type == "wuji" else {},
            "trackerNetwork": tracker_status(),
        }

    def meta_fn():
        cam_meta = None
        preview_meta = None
        if camera is not None:
            try:
                cam_meta = {"width": int(camera.size[0]), "height": int(camera.size[1]),
                            "fps": int(round(float(camera.fps)))}
                preview_width, preview_height = pump.preview_size(
                    camera.size[0], camera.size[1])
                preview_meta = {
                    "width": preview_width,
                    "height": preview_height,
                    "fps": max(1, int(args.stream_fps)),
                    "default_visible": glove_type == "wuji",
                }
            except Exception:
                cam_meta = {"width": 0, "height": 0, "fps": 0}
        parents = hand_parents_snapshot(gloves, hand_sides, glove_type)
        return {
            "camera": cam_meta,
            "camera_preview": preview_meta,
            "gloves": len(gloves),
            "tracker": tracker is not None,
            "has_capture_device": devices.available,
            "glove_type": glove_type,
            "glove_transport": glove_status(),
            "wuji_transport": (glove_status() if glove_type == "wuji" else None),
            "tracker_transport": tracker_status(),
            "udp": (f"{args.manus_ip}:{args.manus_port}"
                    if glove_type == "manus" else None),
            "wuji_calibration": (
                args.wuji_calibration if glove_type == "wuji"
                and not args.no_wuji_calibration else None),
            "save_dir": save_dir,
            "record_countdown": state.countdown_default,
            "task_setting": state.task_setting_path,
            "task_count": len(state.task_names),
            "upload_enabled": bool(args.rclone_remote),
            "upload_remote": args.rclone_remote,
            "storage_limit_gb": args.storage_limit_gb,
            "storage_check_interval": storage_check_interval_minutes,
            "storage_check_interval_minutes": storage_check_interval_minutes,
            "recording_log": recording_log_path(save_dir),
            "handParents": parents,
        }

    def cmd_fn(ch, countdown=None):
        # 把网页按钮/键盘指令塞进与终端键盘同一个控制队列（线程安全跨到 asyncio 循环）
        enqueue_control(ch, countdown)
        return {
            "r": f"开始采集{state.current_task_name}任务",
            "d": "已发送：终止并丢弃",
            "e": "已发送：完成并保存",
            "q": "已发送：退出",
        }.get(ch, "")

    def task_fn(offset):
        """在 asyncio 主线程中切换任务，避免与采集状态转换发生竞争。"""
        result = Future()

        def select_task():
            if (any_recording(camera, gloves, tracker) or state.counting_down
                    or state.saving or uploader.status()["running"]):
                result.set_result({
                    "ok": False,
                    "msg": "当前正忙，不能切换采集任务",
                })
                return
            try:
                task_name = state.select_relative_task(offset)
            except (RuntimeError, TypeError, ValueError) as exc:
                result.set_result({"ok": False, "msg": str(exc)})
                return
            if task_name is None:
                result.set_result({"ok": False, "msg": "任务列表为空"})
                return
            result.set_result({
                "ok": True,
                "msg": (f"已切换到 [{state.task_index + 1}/"
                        f"{len(state.task_names)}] {task_name}"),
                "taskName": task_name,
                "taskIndex": state.task_index + 1,
                "taskCount": len(state.task_names),
            })

        loop.call_soon_threadsafe(select_task)
        try:
            return result.result(timeout=2.0)
        except FutureTimeoutError:
            return {"ok": False, "msg": "任务切换超时，请重试"}

    def upload_fn(start):
        if not start:
            return {"ok": True, **uploader.status()}
        if state.counting_down:
            return {"ok": False, "message": "开始记录倒计时期间不能上传"}
        if any_recording(camera, gloves, tracker):
            return {"ok": False, "message": "记录数据期间不能上传"}
        if state.saving:
            return {"ok": False, "message": "正在保存数据，请稍后上传"}
        ok, message = uploader.start()
        return {"ok": ok, **uploader.status(), "message": message}

    async def storage_monitor_loop():
        """定期统计数据目录；超过上限时在空闲状态启动 rclone move。"""
        last_auto_attempt = 0.0
        retry_seconds = max(300.0, storage_check_interval_seconds)
        while not stop_event.is_set():
            storage_check_event.clear()
            try:
                usage_bytes, file_count = await loop.run_in_executor(
                    None, directory_usage, save_dir)
                state.storage_bytes = int(usage_bytes)
                state.storage_files = int(file_count)
                state.storage_scanned_at = time.time()
                state.storage_over_limit = bool(
                    state.storage_limit_bytes > 0
                    and usage_bytes > state.storage_limit_bytes)

                if state.storage_limit_bytes <= 0:
                    state.storage_message = "容量自动清理已禁用"
                elif not state.storage_over_limit:
                    state.storage_message = "存储占用正常"
                else:
                    upload_status = uploader.status()
                    capture_busy = (
                        state.counting_down or state.saving
                        or any_recording(camera, gloves, tracker))
                    if not args.rclone_remote:
                        state.storage_message = "已超过上限，但未配置 rclone 远端"
                    elif upload_status["running"]:
                        state.storage_message = "已超过上限，上传清理正在运行"
                    elif capture_busy:
                        state.storage_message = "已超过上限，等待本次记录/保存完成"
                    elif time.monotonic() - last_auto_attempt < retry_seconds:
                        state.storage_message = "自动清理失败或未释放空间，稍后重试"
                    else:
                        ok, message = uploader.start(cleanup=True, automatic=True)
                        state.storage_message = message
                        if ok:
                            last_auto_attempt = time.monotonic()
            except Exception as exc:
                state.storage_message = f"容量检查失败：{type(exc).__name__}: {exc}"

            try:
                await asyncio.wait_for(
                    storage_check_event.wait(), timeout=storage_check_interval_seconds)
            except asyncio.TimeoutError:
                pass

    # --- 相机泵 ---
    pump = CameraPump(camera, status_fn, quality=args.jpeg_quality,
                      target_fps=args.stream_fps, overlay=True,
                      max_width=args.preview_width,
                      max_height=args.preview_height)
    pump.start()

    # --- 网页服务 ---
    if not os.path.exists(args.page):
        print(f"[web] ⚠ 未找到页面文件: {args.page}")
        print("[web]   请确认 record_ego_page.html 与本脚本同目录，或用 --page 指定。")
    ctx = WebContext(
        args.page, pump, live_fn, meta_fn, cmd_fn, upload_fn, task_fn)
    httpd, web_port = make_server(args.web_port, ctx, host=args.host)
    srv_thread = threading.Thread(target=httpd.serve_forever, name="ego-web", daemon=True)
    srv_thread.start()

    shown_host = "127.0.0.1" if args.host in ("0.0.0.0", "") else args.host
    url = "http://%s:%d/" % (shown_host, web_port)

    # --- 控制 + 状态（q 已在前面建好；网页/终端/网页键盘共用之）---
    ctrl = asyncio.create_task(
        control_loop(q, camera, gloves, state, save_dir, stop_event, tracker,
                     lambda: uploader.status()["running"], storage_check_event.set,
                     glove_type))
    status = asyncio.create_task(
        status_loop(camera, gloves, state, stop_event, tracker))
    storage = asyncio.create_task(storage_monitor_loop())

    # --- 启动横幅 ---
    cam_txt = "就绪" if camera is not None else "无"
    startup_tracker = tracker_status()
    if glove_type == "wuji":
        trk_txt = ("Windows 已连接" if startup_tracker.get("connected") else
                   ("等待 Windows" if tracker is not None else "无"))
    else:
        trk_txt = "已连接" if tracker is not None else "无"
    print("\n" + "=" * 60)
    print("  Ego 数据采集 —— 浏览器实时监看已就绪")
    print("    打开:  %s" % url)
    if args.host in ("0.0.0.0", ""):
        print("    (已监听 0.0.0.0：同局域网设备可用本机 IP 访问，如 http://<本机IP>:%d/)" % web_port)
    glove_label = "Manus" if glove_type == "manus" else "Wuji"
    transport_text = (f"   UDP: {args.manus_ip}:{args.manus_port}"
                      if glove_type == "manus" else "")
    print("    视频: %s   %s 手套: %d 只(左/右)   tracker: %s%s"
          % (cam_txt, glove_label, len(gloves), trk_txt, transport_text))
    if glove_type == "wuji" and tracker is not None:
        print("    Tracker 接收端: %s   状态: %s"
              % (startup_tracker.get("endpoint"), trk_txt))
        print("    Windows Tracker 键盘: r 开始采集   e 结束并保存")
    print("    控制: 网页大按钮「开始/完成」 · 网页键盘 · 终端键盘  r/d/e/q  三者并行")
    print("      r 开始记录   d 终止丢弃   e 完成保存   q 退出")
    clock_text = ("统一换算到 Linux 墙钟" if glove_type == "wuji"
                  else "同一主机时钟，天然对齐")
    print("    保存: %s/session_YYYYMMDD_HHMMSS_TASK/ (%s)" % (save_dir, clock_text))
    print("    倒计时: %g 秒（网页可调整）" % state.countdown_default)
    print("    任务流程: %s（%d 个任务，完成后循环）"
          % (state.task_setting_path, len(state.task_names)))
    print("    上传: %s" % (args.rclone_remote or "未配置（使用 --rclone-remote）"))
    print("    存储上限: %g GB，超过后自动上传并删除已上传文件（检查间隔 %g 分钟）"
          % (args.storage_limit_gb, storage_check_interval_minutes))
    print("    今日录制: %.3f 分钟（%d 条，日志: %s）"
          % (state.total_recording_seconds / 60.0,
             state.saved_recordings, recording_log_path(save_dir)))
    print("=" * 60 + "\n")

    if not args.no_open:
        threading.Thread(
            target=lambda: (time.sleep(0.6), webbrowser.open(url)), daemon=True).start()

    def _report_dead(task):
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            import traceback
            print("\n[!] 后台任务异常退出:", flush=True)
            traceback.print_exception(type(exc), exc, exc.__traceback__)
            stop_event.set()
    ctrl.add_done_callback(_report_dead)
    status.add_done_callback(_report_dead)
    storage.add_done_callback(_report_dead)

    try:
        await stop_event.wait()
    finally:
        stop_event.set()
        ctrl.cancel()
        status.cancel()
        storage.cancel()
        await asyncio.gather(ctrl, status, storage, return_exceptions=True)
        # 关网页与相机泵
        try:
            httpd.shutdown()
        except Exception:
            pass
        try:
            httpd.server_close()
        except Exception:
            pass
        pump.stop()
        restore()
        await devices.stop_glove_receivers()
        devices.close()
