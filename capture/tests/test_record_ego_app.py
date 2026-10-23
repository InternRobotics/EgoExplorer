import asyncio
import json
import os
import tempfile
import time
import unittest
from unittest import mock

import av
import record_ego_app
import tracker_sender_windows
import visualize_session
import numpy as np
from devices.gopro_camera import GoProHDMICamera, _PyAVVideoWriter
from devices.manus_glove import ManusGlove, _parents_to_list
from devices.network_tracker import NetworkTrackerReceiver
from devices.wuji_glove import WujiGlove
from record_ego.capture import save_session
from record_ego.config import parse_args
from record_ego.devices import ConnectedDevices
from record_ego.web import (
    CameraPump,
    HandTap,
    hand_parents_snapshot,
    standard_hand_parents,
)


class _FakeHttpServer:
    def serve_forever(self):
        return None

    def shutdown(self):
        return None

    def server_close(self):
        return None


class NoDeviceWebStartupTests(unittest.IsolatedAsyncioTestCase):
    async def test_wuji_web_starts_and_reports_offline_without_devices(self):
        captured = {}

        def fake_keyboard(_loop):
            queue = asyncio.Queue()
            queue.put_nowait("q")
            return queue, lambda: None

        def fake_server(port, context, host="127.0.0.1"):
            captured["context"] = context
            return _FakeHttpServer(), port

        with tempfile.TemporaryDirectory() as save_dir, mock.patch.object(
            record_ego_app, "connect_devices", return_value=ConnectedDevices()
        ), mock.patch.object(
            record_ego_app, "setup_keyboard", side_effect=fake_keyboard
        ), mock.patch.object(
            record_ego_app, "make_server", side_effect=fake_server
        ), mock.patch.object(
            record_ego_app, "directory_usage", return_value=(0, 0)
        ):
            task_setting = os.path.join(save_dir, "tasks.json")
            with open(task_setting, "w", encoding="utf-8") as file_obj:
                json.dump(["test"], file_obj)
            await record_ego_app.main(
                "wuji",
                [
                    "--no-camera",
                    "--no-open",
                    "--storage-limit-gb", "0",
                    "--save-dir", save_dir,
                    "--task-setting", task_setting,
                ],
            )

        meta = captured["context"].meta_fn()
        live = captured["context"].live_fn()
        self.assertEqual(meta["glove_type"], "wuji")
        self.assertEqual(meta["gloves"], 0)
        self.assertFalse(meta["has_capture_device"])
        self.assertFalse(live["hasCaptureDevice"])
        self.assertFalse(live["cameraAvailable"])
        self.assertFalse(live["cameraOnline"])
        self.assertEqual(live["hands"], {})
        self.assertEqual(live["tactile"], {})

    async def test_manus_web_starts_and_keeps_manus_metadata(self):
        captured = {}

        def fake_keyboard(_loop):
            queue = asyncio.Queue()
            queue.put_nowait("q")
            return queue, lambda: None

        def fake_server(port, context, host="127.0.0.1"):
            captured["context"] = context
            return _FakeHttpServer(), port

        with tempfile.TemporaryDirectory() as save_dir, mock.patch.object(
            record_ego_app, "connect_devices", return_value=ConnectedDevices()
        ), mock.patch.object(
            record_ego_app, "setup_keyboard", side_effect=fake_keyboard
        ), mock.patch.object(
            record_ego_app, "make_server", side_effect=fake_server
        ), mock.patch.object(
            record_ego_app, "directory_usage", return_value=(0, 0)
        ):
            task_setting = os.path.join(save_dir, "tasks.json")
            with open(task_setting, "w", encoding="utf-8") as file_obj:
                json.dump(["test"], file_obj)
            await record_ego_app.main(
                "manus",
                [
                    "--no-camera",
                    "--no-tracker",
                    "--no-open",
                    "--storage-limit-gb", "0",
                    "--save-dir", save_dir,
                    "--task-setting", task_setting,
                ],
            )

        meta = captured["context"].meta_fn()
        live = captured["context"].live_fn()
        self.assertEqual(meta["glove_type"], "manus")
        self.assertEqual(meta["udp"], "0.0.0.0:9000")
        self.assertEqual(meta["glove_transport"]["label"], "Manus")
        self.assertEqual(live["tactile"], {})


class CameraPageTests(unittest.TestCase):
    def test_wuji_defaults_to_low_resolution_preview(self):
        wuji = parse_args(["--task-setting", "tasks.json"], glove_type="wuji")
        manus = parse_args(["--task-setting", "tasks.json"], glove_type="manus")

        self.assertEqual((wuji.preview_width, wuji.preview_height), (640, 360))
        self.assertEqual((manus.preview_width, manus.preview_height), (0, 0))

    def test_camera_pump_online_requires_a_recent_real_frame(self):
        pump = CameraPump(object(), lambda: (False, 0.0))
        self.assertFalse(pump.online())
        pump._last_frame_monotonic = time.monotonic()
        self.assertTrue(pump.online())
        pump._last_frame_monotonic = time.monotonic() - 5.0
        self.assertFalse(pump.online(timeout=2.0))

    def test_camera_pump_downscales_preview_without_changing_aspect_ratio(self):
        pump = CameraPump(
            object(), lambda: (False, 0.0), max_width=640, max_height=360)
        source = np.zeros((1440, 2560, 3), dtype=np.uint8)

        preview = pump._resize_preview(source)

        self.assertEqual(preview.shape, (360, 640, 3))
        self.assertEqual(source.shape, (1440, 2560, 3))
        self.assertEqual(pump.preview_size(640, 480), (480, 360))

    def test_camera_pump_does_not_upscale_small_preview(self):
        pump = CameraPump(
            object(), lambda: (False, 0.0), max_width=640, max_height=360)
        source = np.zeros((180, 320, 3), dtype=np.uint8)

        self.assertIs(pump._resize_preview(source), source)

    def test_page_has_camera_toggle_and_online_light(self):
        page_path = os.path.join(
            os.path.dirname(record_ego_app.__file__), "record_ego_page.html")
        with open(page_path, encoding="utf-8") as file_obj:
            html = file_obj.read()
        self.assertIn('id="btn-camera-view"', html)
        self.assertIn('id="camera-light"', html)
        self.assertIn('id="camera-state-text"', html)
        self.assertIn('data-src="stream.mjpg"', html)
        self.assertIn('m.glove_type==="wuji"&&!cameraPreferenceTouched', html)


class WujiSkeletonTests(unittest.TestCase):
    def test_live_wuji_skeleton_keeps_native_21_nodes(self):
        glove = WujiGlove(sn="WG1JA00000000000", slot="left")
        raw = np.zeros((21, 7), dtype=np.float32)
        raw[:, 0] = np.arange(21, dtype=np.float32)
        raw[:, 6] = 1.0
        glove.latest["hand_skeleton"] = raw

        points = HandTap([glove], {id(glove): "left"}).snapshot()["left"]

        self.assertEqual(len(points), 21)
        self.assertEqual(len(standard_hand_parents()), 21)
        self.assertEqual([point[0] for point in points], list(range(21)))
        self.assertEqual(glove.latest["hand_skeleton"].shape, (21, 7))


class ManusSkeletonTests(unittest.TestCase):
    def test_live_manus_skeleton_uses_21_nodes_like_wuji(self):
        glove = ManusGlove(slot="left", device_name="glove_0")
        raw = np.zeros((21, 7), dtype=np.float32)
        raw[:, 0] = np.arange(21, dtype=np.float32)
        raw[:, 6] = 1.0
        glove._latest_skeleton = raw

        points = HandTap([glove], {id(glove): "left"}).snapshot()["left"]

        self.assertEqual(len(points), 21)
        # 丢弃 4 个掌骨关节后, Manus 与 Wuji 使用同一套 21 节点拓扑。
        self.assertEqual(standard_hand_parents("manus"),
                         standard_hand_parents("wuji"))
        self.assertEqual([point[0] for point in points], list(range(21)))

    def test_manus_self_parent_root_is_normalized(self):
        reported = {index: (index if index == 0 else index - 1)
                    for index in range(21)}

        parents = _parents_to_list(reported, 21)

        self.assertEqual(parents[0], -1)
        self.assertEqual(parents[1:], list(range(20)))

    def test_live_page_uses_reported_manus_hierarchy(self):
        glove = ManusGlove(slot="left", device_name="glove_0")
        # 构造一个有效但不同于硬编码布局的层级，模拟 SDK 节点顺序变化。
        reported = {index: (index if index == 0 else index - 1)
                    for index in range(21)}
        nodes = [
            {"pos": [float(index), 0.0, 0.0],
             "rot": [0.0, 0.0, 0.0, 1.0]}
            for index in range(21)
        ]
        glove._feed_frame(nodes, reported, None, 1)

        parents = hand_parents_snapshot(
            [glove], {id(glove): "left"}, "manus")["left"]

        self.assertEqual(parents[0], -1)
        self.assertEqual(parents[1:], list(range(20)))


class ManusTo21NodesTests(unittest.TestCase):
    """Manus 原始 25 节点 -> 21 节点 (MediaPipe) 规约。"""

    @staticmethod
    def _raw_25_nodes():
        nodes = []
        for i in range(25):
            node = {"pos": [float(i), 0.0, 0.0], "rot": [0.0, 0.0, 0.0, 1.0]}
            # 复刻 Manus 原始层级: 拇指4节与四指掌骨挂腕, 其余沿链挂前一节。
            if i == 0 or i in (1, 5, 10, 15, 20):
                node["parent"] = 0
            else:
                node["parent"] = i - 1
            nodes.append(node)
        return nodes

    def test_metacarpal_nodes_dropped_and_parents_remapped(self):
        from devices.manus_glove import _to_21_nodes

        out = _to_21_nodes(self._raw_25_nodes())

        self.assertEqual(len(out), 21)
        # MediaPipe 顺序: 0腕 1-4拇指 5-8食指 9-12中指 13-16无名指 17-20小指。
        self.assertEqual(
            [node["pos"][0] for node in out],
            [0, 1, 2, 3, 4,
             6, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18, 19, 21, 22, 23, 24])
        # 各指第一节原本挂在掌骨上, 规约后直接挂腕 (MediaPipe)。
        for mcp in (5, 9, 13, 17):
            self.assertEqual(out[mcp]["parent"], 0)
        # 拇指链与其余指节保持逐级父子关系。
        self.assertEqual([node["parent"] for node in out[1:5]], [0, 1, 2, 3])
        self.assertEqual([node["parent"] for node in out[6:9]], [5, 6, 7])

    def test_to_21_nodes_passthrough_non_25(self):
        from devices.manus_glove import _to_21_nodes

        nodes = [{"pos": [0.0, 0.0, 0.0], "rot": [0.0, 0.0, 0.0, 1.0],
                  "parent": 0}] * 21
        out = _to_21_nodes(nodes)
        self.assertEqual(len(out), 21)
        self.assertIsNot(out, nodes)

    def test_feed_frame_with_raw_25_nodes_stores_21(self):
        from devices.manus_glove import _to_21_nodes

        glove = ManusGlove(slot="left", device_name="glove_0")
        glove._streams = ["hand_skeleton"]
        nodes = _to_21_nodes(self._raw_25_nodes())
        glove._feed_frame(nodes, None, None, 1)

        self.assertEqual(glove.latest_skeleton.shape, (21, 7))

    def test_page_keeps_and_rebuilds_dynamic_parent_relationships(self):
        page_path = os.path.join(
            os.path.dirname(record_ego_app.__file__), "record_ego_page.html")
        with open(page_path, encoding="utf-8") as file_obj:
            html = file_obj.read()

        self.assertIn("const MANUS_21", html)
        self.assertIn("LIVE={...LIVE,...d}", html)
        self.assertIn("handObjs[side].parentKey!==parentKey", html)

    def test_offline_viewer_accepts_manus_self_parent_root(self):
        reported = [0] + list(range(20))

        parents = visualize_session._validated_parents(
            reported, 21, "manus")

        self.assertEqual(parents, [-1] + list(range(20)))


class SessionMetadataTests(unittest.TestCase):
    def test_requested_wuji_type_is_kept_without_connected_gloves(self):
        with tempfile.TemporaryDirectory() as save_dir:
            output = save_session(
                None, [], "offline_wuji", save_dir=save_dir, glove_type="wuji")
            with open(output["meta"], encoding="utf-8") as file_obj:
                metadata = json.load(file_obj)

        self.assertEqual(metadata["glove_type"], "wuji")
        self.assertIn("Windows tracker", metadata["clock"])

    def test_save_stops_glove_and_tracker_before_video_finalize(self):
        events = []

        class FakeCamera:
            recording = True
            fps = 30.0
            size = (1920, 1080)

            def save_recording(self):
                events.append("camera-save")
                self.recording = False
                return None, np.empty(0, dtype=np.int64)

        class FakeGlove:
            recording = True

            def stop_recording(self):
                events.append("glove-stop")
                self.recording = False

            def host_ts_range(self):
                return None

            def glove_info(self):
                return {"device_name": "glove_0", "slot": "left", "sn": "TEST"}

            def discard(self):
                events.append("glove-discard")

        class FakeTracker:
            recording = True

            def stop_recording(self):
                events.append("tracker-stop")
                self.recording = False

            def tracker_info(self):
                return {"frames": 1, "roles": []}

            def host_ts_range(self):
                return None

            def save_recording(self, _path):
                events.append("tracker-save")

            def discard(self):
                events.append("tracker-discard")

        with tempfile.TemporaryDirectory() as save_dir, mock.patch(
                "record_ego.capture.save_all", return_value=None):
            save_session(
                FakeCamera(), [FakeGlove()], "ordered_stop",
                save_dir=save_dir, tracker=FakeTracker(), glove_type="wuji")

        self.assertLess(events.index("glove-stop"), events.index("camera-save"))
        self.assertLess(events.index("tracker-stop"), events.index("camera-save"))


class TrackerRateLimitTests(unittest.TestCase):
    def test_linux_receiver_limits_legacy_high_rate_sender_while_recording(self):
        receiver = NetworkTrackerReceiver(max_record_fps=100)
        receiver._listening = True
        receiver.start_recording()
        pose = {
            "camera": {
                "position": {"x": 0, "y": 0, "z": 0},
                "orientation": {"w": 1, "x": 0, "y": 0, "z": 0},
            },
        }
        for index in range(100):
            timestamp = 1_000_000 + index * 1_000
            receiver._handle_frame({
                "sequence": index,
                "t_source_host_us": timestamp,
                "poses": pose,
            }, timestamp)

        self.assertEqual(receiver.total_frames(), 10)
        self.assertEqual(receiver.status()["record_rate_limited"], 90)

    def test_windows_sender_defaults_to_90_fps(self):
        args = tracker_sender_windows.build_parser().parse_args(["--host", "127.0.0.1"])
        self.assertEqual(args.send_fps, 90.0)


class TrackerSampleFpsTests(unittest.TestCase):
    def test_manus_local_tracker_defaults_to_90_fps_sampling(self):
        args = parse_args(["--task-setting", "tasks.json"], glove_type="manus")
        self.assertEqual(args.tracker_sample_fps, 90.0)

    def test_manus_local_tracker_sampling_is_configurable(self):
        args = parse_args(
            ["--task-setting", "tasks.json", "--tracker-sample-fps", "120"],
            glove_type="manus")
        self.assertEqual(args.tracker_sample_fps, 120.0)

    def test_wuji_has_no_local_sample_fps_option(self):
        args = parse_args(["--task-setting", "tasks.json"], glove_type="wuji")
        self.assertFalse(hasattr(args, "tracker_sample_fps"))


def _load_vive_tracker_with_fake_xr():
    """注入一个最小的假 ``xr`` 模块后导入 devices.vive_tracker。

    这样无需 pyopenxr/SteamVR 即可测试采样节拍逻辑。返回 (module, counters),
    counters["locate"] 统计 xrLocateSpace 调用次数, counters["sync"] 统计
    xrSyncActions 调用次数。
    """
    import ctypes
    import importlib
    import sys
    import types

    fake_xr = types.ModuleType("xr")
    fake_xr.FormFactor = types.SimpleNamespace(HEAD_MOUNTED_DISPLAY=0)
    fake_xr.SessionState = types.SimpleNamespace(
        IDLE=0, READY=1, SYNCHRONIZED=2, VISIBLE=3, FOCUSED=4,
        STOPPING=5, EXITING=6, LOSS_PENDING=7)
    fake_xr.SPACE_LOCATION_POSITION_VALID_BIT = 0b0001
    fake_xr.StructureType = types.SimpleNamespace(
        EVENT_DATA_INSTANCE_LOSS_PENDING=1,
        EVENT_DATA_SESSION_STATE_CHANGED=2)

    class _EventUnavailable(Exception):
        pass

    fake_xr.EventUnavailable = _EventUnavailable

    def _no_events(_instance):
        raise _EventUnavailable()

    fake_xr.poll_event = _no_events
    fake_xr.ActiveActionSet = lambda **kwargs: ctypes.c_int(0)
    fake_xr.ActionsSyncInfo = lambda **kwargs: None

    counters = {"locate": 0, "sync": 0}

    def _sync_actions(**kwargs):
        counters["sync"] += 1

    fake_xr.sync_actions = _sync_actions

    class _Pose:
        class position:
            x = y = z = 0.0

        class orientation:
            w = 1.0
            x = y = z = 0.0

    class _Location:
        location_flags = 0b0001
        pose = _Pose()

    def _locate_space(**kwargs):
        counters["locate"] += 1
        return _Location()

    fake_xr.locate_space = _locate_space
    fake_xr.convert_win32_performance_counter_to_time_khr = (
        lambda instance, counter: int(counter.value))

    saved = sys.modules.get("xr")
    sys.modules["xr"] = fake_xr
    try:
        module = importlib.import_module("devices.vive_tracker")
        if module.__dict__.get("xr") is not fake_xr:
            # 已被其它用例以真实/其它 xr 导入过: 重新绑定到本测试的 fake xr.
            module = importlib.reload(module)
    finally:
        if saved is not None:
            sys.modules["xr"] = saved
        else:
            sys.modules.pop("xr", None)
    return module, counters


# devices.vive_tracker 只在首次导入时绑定 xr; 全测试进程共享一份 fake xr.
_VIVE_TRACKER_FAKE = {}


def _vive_tracker_module():
    if "counters" not in _VIVE_TRACKER_FAKE:
        (_VIVE_TRACKER_FAKE["module"],
         _VIVE_TRACKER_FAKE["counters"]) = _load_vive_tracker_with_fake_xr()
    return _VIVE_TRACKER_FAKE["module"], _VIVE_TRACKER_FAKE["counters"]


@unittest.skipUnless(os.name == "nt", "OpenXR tracker sampling uses Win32 APIs (ctypes.windll)")
class ViveTrackerSamplingTests(unittest.TestCase):
    """本机 tracker 的主机时钟采样必须与合成器帧节拍解耦 (headless 可慢至 ~15 Hz)。"

    用 66 ms 阻塞的假 _pump_frame 模拟慢速合成器; 若采样仍按帧节拍走,
    90 次 poll 需要 ~6 s, 主机时钟采样则应在 ~2 s 内完成。
    """

    def test_host_clock_sampling_ignores_slow_compositor(self):
        module, counters = _vive_tracker_module()
        counters["locate"] = 0
        tracker = module.ViveTracker(sample_fps=90)
        tracker.session_is_running = True
        tracker.session_state = module.xr.SessionState.FOCUSED
        tracker.role_strings = ["a", "b", "c"]
        tracker.tracker_action_spaces = [object(), object(), object()]
        tracker.space = object()

        pump_calls = []

        def slow_pump():
            pump_calls.append(time.perf_counter())
            time.sleep(0.066)  # 模拟 headless 合成器 ~15 Hz 的阻塞

        tracker._pump_frame = slow_pump

        polls = 90
        started = time.perf_counter()
        for _ in range(polls):
            tracker.poll()
        elapsed = time.perf_counter() - started

        self.assertGreater(elapsed, 0.5)
        self.assertLess(elapsed, 2.0)
        self.assertTrue(2 <= len(pump_calls) <= 4, len(pump_calls))
        # 每次 poll 都应在主机时钟上做一次 3-role 定位.
        self.assertGreaterEqual(counters["locate"], (polls - 2) * 3)

    def test_sample_fps_zero_means_unlimited(self):
        module, counters = _vive_tracker_module()
        counters["locate"] = 0
        tracker = module.ViveTracker(sample_fps=0)
        tracker.session_is_running = True
        tracker.session_state = module.xr.SessionState.FOCUSED
        tracker.role_strings = ["a"]
        tracker.tracker_action_spaces = [object()]
        tracker.space = object()
        tracker._pump_frame = lambda: None

        started = time.perf_counter()
        for _ in range(50):
            tracker.poll()
        elapsed = time.perf_counter() - started
        self.assertLess(elapsed, 0.2)
        self.assertGreaterEqual(counters["locate"], 50)

    def test_hot_path_never_calls_sync_actions(self):
        """回归: 采样热路径不得调用 xrSyncActions。

        SteamVR 帧外的 sync 会阻塞到下一个合成器帧 (~66 ms, 实测
        session_20260912_163132_task1), 热路径一旦 sync, 采样率即被拖回
        ~15 Hz。sync 只允许出现在保活帧内 (_pump_frame); 这里桩掉
        _pump_frame 后 poll 50 次, 不得产生任何 sync 调用。
        """
        module, counters = _vive_tracker_module()
        counters["sync"] = 0
        tracker = module.ViveTracker(sample_fps=90)
        tracker.session_is_running = True
        tracker.session_state = module.xr.SessionState.FOCUSED
        tracker.role_strings = ["a"]
        tracker.tracker_action_spaces = [object()]
        tracker.space = object()
        tracker._pump_frame = lambda: None

        for _ in range(50):
            tracker.poll()
        self.assertEqual(counters["sync"], 0)

    def test_invalid_sample_fps_rejected(self):
        module, _ = _vive_tracker_module()
        for bad in (-1, 241, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                module.ViveTracker(sample_fps=bad)


class VisualizeSessionCompatibilityTests(unittest.TestCase):
    @staticmethod
    def _write_session(directory, glove_type, nodes, flattened=False):
        base = "session_test"
        device = "glove_0"
        frames = 4
        skeleton = np.zeros((frames, nodes, 7), dtype=np.float32)
        skeleton[:, :, 0] = np.arange(nodes, dtype=np.float32)
        skeleton[:, :, 6] = 1.0
        if flattened:
            skeleton = skeleton.reshape(frames, nodes * 7)
        host_time = np.array(
            [1_800_000_000_000_000 + index * 20_000
             for index in range(frames)], dtype=np.int64)
        device_time = np.array(
            [100_000 + index * 20_000 for index in range(frames)],
            dtype=np.int64)
        meta = {
            "device_type": f"{glove_type}_glove",
            "gloves": [{
                "device_name": device,
                "slot": "left",
                "sn": "TEST",
            }],
            "channels": {
                f"{device}/hand_skeleton": {"frames": frames},
            },
        }
        arrays = {
            f"{device}__hand_skeleton": skeleton,
            f"{device}__hand_skeleton__t_us": device_time,
            f"{device}__hand_skeleton__t_dev_us": device_time,
            f"{device}__hand_skeleton__t_host_us": host_time,
        }
        if glove_type == "wuji":
            meta["channels"][f"{device}/tactile"] = {"frames": frames}
            tactile = np.linspace(-0.2, 1.2, frames * 24 * 31,
                                  dtype=np.float32).reshape(frames, 24, 31)
            arrays.update({
                f"{device}__tactile": tactile,
                f"{device}__tactile__t_us": device_time,
                f"{device}__tactile__t_dev_us": device_time,
                f"{device}__tactile__t_host_us": host_time,
            })
        arrays["meta"] = np.array(json.dumps(meta))
        glove_file = os.path.join(directory, base + ".npz")
        np.savez_compressed(glove_file, **arrays)
        with open(os.path.join(directory, base + ".session.json"),
                  "w", encoding="utf-8") as file_obj:
            json.dump({
                "base": base,
                "glove_type": glove_type,
                "gloves_npz": os.path.basename(glove_file),
                "video": {},
            }, file_obj)
        return host_time

    def test_manus_uses_native_25_nodes_and_host_clock(self):
        with tempfile.TemporaryDirectory() as directory:
            host_time = self._write_session(directory, "manus", 25)
            payload = visualize_session.build_payload(directory, max_fps=0)

        glove = payload["gloves"]["glove_0"]
        self.assertEqual(payload["gloveType"], "manus")
        self.assertEqual(glove["type"], "manus")
        self.assertEqual(glove["nodeCount"], 25)
        self.assertEqual(glove["timeSource"], "host")
        self.assertEqual(glove["t_us"], host_time.tolist())
        self.assertEqual(len(glove["parents"]), 25)

    def test_wuji_accepts_flattened_21_nodes_and_lists_tactile(self):
        with tempfile.TemporaryDirectory() as directory:
            host_time = self._write_session(
                directory, "wuji", 21, flattened=True)
            payload = visualize_session.build_payload(directory, max_fps=0)

        glove = payload["gloves"]["glove_0"]
        self.assertEqual(payload["gloveType"], "wuji")
        self.assertEqual(glove["type"], "wuji")
        self.assertEqual(glove["nodeCount"], 21)
        self.assertEqual(glove["t_us"], host_time.tolist())
        self.assertIn("tactile", glove["streams"])
        self.assertEqual(len(glove["positions"][0]), 21)
        self.assertEqual(glove["tactile"]["t_us"], host_time.tolist())
        self.assertEqual(glove["tactile"]["rows"], 24)
        self.assertEqual(glove["tactile"]["cols"], 31)
        self.assertEqual(len(glove["tactile"]["values"][0]), 24 * 31)
        self.assertEqual(glove["tactile"]["values"][0][0], -1.0)
        self.assertEqual(glove["tactile"]["values"][-1][-1], 1.0)

    def test_both_demo_formats_build(self):
        for glove_type, nodes in (("manus", 21), ("wuji", 21)):
            with self.subTest(glove_type=glove_type), \
                    tempfile.TemporaryDirectory() as directory:
                visualize_session.make_demo(directory, glove_type=glove_type)
                payload = visualize_session.build_payload(directory, max_fps=10)
                self.assertEqual(payload["gloveType"], glove_type)
                self.assertEqual(
                    payload["gloves"]["glove_0"]["nodeCount"], nodes)
                if glove_type == "wuji":
                    self.assertIn("tactile", payload["gloves"]["glove_0"])


class TimestampedWriterTests(unittest.TestCase):
    def test_pyav_writer_preserves_irregular_capture_pts(self):
        offsets_us = [0, 33_333, 66_667, 200_000, 233_333, 500_000]
        base_us = 1_800_000_000_000_000
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "vfr.mp4")
            writer = _PyAVVideoWriter(
                path, 64, 48, 30, encoder="libx264",
                preset="ultrafast", threads=2)
            self.assertTrue(writer.isOpened(), writer.error)
            for index, offset_us in enumerate(offsets_us):
                image = np.full((48, 64, 3), index * 30, dtype=np.uint8)
                self.assertTrue(writer.write(image, base_us + offset_us))
            writer.release()

            with av.open(path) as container:
                stream = container.streams.video[0]
                actual_us = [
                    int(round(float(frame.pts * frame.time_base) * 1_000_000))
                    for frame in container.decode(stream)
                ]

        self.assertEqual(actual_us, offsets_us)


class QueuedCaptureTests(unittest.TestCase):
    class _FastCapture:
        def __init__(self):
            self.reads = 0

        def isOpened(self):
            return True

        def read(self):
            time.sleep(0.001)
            self.reads += 1
            return True, np.zeros((12, 16, 3), dtype=np.uint8)

        def release(self):
            pass

    class _SlowWriter:
        def __init__(self):
            self.timestamps = []
            self.opened = True

        def write(self, _frame, timestamp_us):
            time.sleep(0.01)
            self.timestamps.append(timestamp_us)
            return True

        def isOpened(self):
            return self.opened

        def release(self):
            self.opened = False

    def test_capture_continues_while_slow_encoder_queue_is_bounded(self):
        camera = GoProHDMICamera(width=16, height=12, fps=30)
        capture = self._FastCapture()
        writer = self._SlowWriter()
        camera._cap = capture
        camera._w, camera._h, camera._fps = 16, 12, 30.0
        camera._ENCODE_QUEUE_SIZE = 4
        camera._make_writer = lambda path: (writer, path)

        with tempfile.TemporaryDirectory() as directory:
            camera.start_recording(os.path.join(directory, "queued.mp4"))
            time.sleep(0.08)
            camera.stop_recording()

        stats = camera.capture_stats()
        self.assertGreater(stats["captured_frames"], stats["encoded_frames"])
        self.assertGreater(stats["queue_dropped_frames"], 0)
        self.assertLessEqual(stats["max_queue_depth"], 4)
        self.assertEqual(stats["encoded_frames"], len(writer.timestamps))
        self.assertEqual(len(camera.frame_timestamps()), len(writer.timestamps))
        self.assertTrue(all(a < b for a, b in zip(
            writer.timestamps, writer.timestamps[1:])))


class WujiStreamLoadTests(unittest.TestCase):
    def test_default_streams_only_keep_tactile_and_skeleton(self):
        self.assertEqual(
            WujiGlove.AVAILABLE_STREAMS,
            ("tactile", "hand_skeleton"),
        )


if __name__ == "__main__":
    unittest.main()
