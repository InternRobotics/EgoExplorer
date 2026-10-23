"""
vive_tracker.py

A small wrapper class around the pyopenxr HTC Vive tracker example
(test_vivie.py). It exposes tracker poses through a simple API:

    with ViveTracker() as vt:
        while not vt.should_exit:
            vt.poll()                              # refresh poses (host-clock sampled)
            all_poses = vt.get_all_poses()         # {role: {position, orientation}}
            waist = vt.get_pose_by_role("waist")   # a single role, or None

Why the poll()/frame model:
    OpenXR is frame driven. A tracker only reports a pose while the session is
    in the FOCUSED state. Call poll() once per loop iteration to refresh the
    cached poses; the two getters can poll for you (refresh=True, the default)
    or read poses from a loop you already drove (refresh=False).

Sampling is driven by the host clock, not by the compositor:
    Headless runtimes can pace compositor frames as slowly as ~15 Hz (measured
    on the Manus collection PC) while the trackers themselves report at
    ~90+ Hz. xrLocateSpace() accepts any XrTime, so poll() locates tracker
    poses at ``sample_fps`` on the host clock and only pumps a real
    wait/begin/end frame every _FRAME_PUMP_KEEPALIVE_S to keep the session
    alive. poll() paces itself: callers may keep calling it in a tight loop.

Gotchas carried over from the example:
    * A tracker only reports a pose when the role you assigned it in SteamVR is
      present in ROLES_FULL / ROLES_CLASSIC below. If a tracker never shows up,
      check that its SteamVR role is in the list (see get_connected_trackers()).
    * Trackers whose SteamVR role is "Handheld object" are not reported.
    * This targets a headless Windows SteamVR session and uses the Win32
      performance-counter extension to derive a valid XrTime, matching the
      original example.
"""

import ctypes
from ctypes import byref, cast, POINTER
import math
import os
import time

import xr

from devices.base import BaseTracker


class ViveTracker(BaseTracker):
    # Revision-3 role set (includes wrist/ankle). Tried first.
    ROLES_FULL = [
        "handheld_object", "left_foot", "right_foot",
        "left_shoulder", "right_shoulder",
        "left_elbow", "right_elbow",
        "left_knee", "right_knee",
        "left_wrist", "right_wrist",
        "left_ankle", "right_ankle",
        "waist", "chest", "camera", "keyboard",
    ]
    # Classic role set (revision 1/2). Used if the runtime rejects the full set.
    ROLES_CLASSIC = [
        "handheld_object", "left_foot", "right_foot",
        "left_shoulder", "right_shoulder",
        "left_elbow", "right_elbow",
        "left_knee", "right_knee",
        "waist", "chest", "camera", "keyboard",
    ]

    _RENDERABLE_STATES = (
        xr.SessionState.READY,
        xr.SessionState.SYNCHRONIZED,
        xr.SessionState.VISIBLE,
        xr.SessionState.FOCUSED,
    )

    # 帧泵保活间隔 (秒): 完整 wait/begin/end 帧只用于维持会话活性, 与采样解耦.
    # 某些 runtime 的 xrWaitFrame 会阻塞到下一个合成器帧 (headless 实测 ~66 ms),
    # 太频繁会把主机时钟采样拖回合成器帧节拍.
    _FRAME_PUMP_KEEPALIVE_S = 0.5

    def __init__(self, prefer_full_roles=True, device_name="tracker_0",
                 form_factor=xr.FormFactor.HEAD_MOUNTED_DISPLAY,
                 sample_fps=90.0):
        super().__init__()            # 初始化 latest_poses 与录制缓存 (BaseTracker)
        self.prefer_full_roles = prefer_full_roles
        self._device_name = device_name
        self.form_factor = form_factor

        sample_fps = float(sample_fps)
        if not math.isfinite(sample_fps) or not 0.0 <= sample_fps <= 240.0:
            raise ValueError("sample_fps 必须是 0～240 之间的有限数字")
        self.sample_fps = sample_fps
        # 主机时钟采样间隔; 0 表示不限制 (每次 poll 都定位, 由调用方自行定速).
        self._sample_interval = 1.0 / sample_fps if sample_fps > 0.0 else 0.0
        self._last_sample_perf = 0.0
        self._last_frame_pump_perf = 0.0
        self._timer_resolution_raised = False

        self.instance = None
        self.system_id = None
        self.session = None
        self.space = None
        self.action_set = None
        self.pose_action = None

        self.role_strings = []        # e.g. ["waist", "chest", ...]
        self.role_path_strings = []   # e.g. ["/user/vive_tracker_htcx/role/waist", ...]
        self.tracker_action_spaces = None

        self.session_state = xr.SessionState.IDLE
        self.session_is_running = False
        self.should_exit = False

        # self.latest_poses 由 BaseTracker.__init__ 初始化; _locate_spaces 每次
        # 采样 (帧内或主机时钟) 把它更新为 role -> {"position": {x, y, z},
        # "orientation": {w, x, y, z}} (仅包含报告了有效位姿的 role)。

        self._enumerate_paths_fn = None

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #
    def start(self):
        """Create the OpenXR instance/session and bind the tracker roles."""
        self.instance = xr.create_instance(
            create_info=xr.InstanceCreateInfo(
                enabled_extension_names=[
                    xr.MND_HEADLESS_EXTENSION_NAME,
                    xr.HTCX_VIVE_TRACKER_INTERACTION_EXTENSION_NAME,
                    xr.KHR_WIN32_CONVERT_PERFORMANCE_COUNTER_TIME_EXTENSION_NAME,
                ],
            ),
        )
        self.system_id = xr.get_system(
            instance=self.instance,
            get_info=xr.SystemGetInfo(form_factor=self.form_factor),
        )
        # Headless session: no graphics binding in `next`.
        self.session = xr.create_session(
            instance=self.instance,
            create_info=xr.SessionCreateInfo(system_id=self.system_id),
        )
        # Reference space that tracker poses are reported relative to. This uses
        # the same default the original example did; change the reference space
        # type here if you want STAGE-relative coordinates instead.
        self.space = xr.create_reference_space(
            session=self.session,
            create_info=xr.ReferenceSpaceCreateInfo(),
        )
        self.action_set = xr.create_action_set(
            instance=self.instance,
            create_info=xr.ActionSetCreateInfo(
                action_set_name="vive_tracker_action_set",
                localized_action_set_name="Vive Tracker Action Set",
                priority=0,
            ),
        )

        # Bind the roles, trying the full (rev-3) set first and falling back to
        # the classic set if the runtime rejects the newer role paths.
        candidate_sets = []
        if self.prefer_full_roles:
            candidate_sets.append(("tracker_pose_full", self.ROLES_FULL))
        candidate_sets.append(("tracker_pose_classic", self.ROLES_CLASSIC))

        last_error = None
        for action_name, roles in candidate_sets:
            try:
                (self.role_path_strings,
                 self.pose_action,
                 self.tracker_action_spaces) = self._setup_tracker_bindings(action_name, roles)
                self.role_strings = list(roles)
                last_error = None
                break
            except Exception as exc:  # runtime rejected these role paths
                last_error = exc
                continue
        if last_error is not None:
            raise last_error

        # Attach the action set to the session exactly once.
        xr.attach_session_action_sets(
            session=self.session,
            attach_info=xr.SessionActionSetsAttachInfo(
                count_action_sets=1,
                action_sets=(xr.ActionSet * 1)(self.action_set),
            ),
        )

        # Function pointer used by get_connected_trackers().
        self._enumerate_paths_fn = cast(
            xr.get_instance_proc_addr(self.instance, "xrEnumerateViveTrackerPathsHTCX"),
            xr.PFN_xrEnumerateViveTrackerPathsHTCX,
        )
        self._raise_timer_resolution()
        return self

    def close(self):
        """Tear down the OpenXR objects."""
        self._restore_timer_resolution()
        if self.action_set is not None:
            xr.destroy_action_set(self.action_set)
            self.action_set = None
        if self.space is not None:
            xr.destroy_space(self.space)
            self.space = None
        if self.session is not None:
            xr.destroy_session(self.session)
            self.session = None
        if self.instance is not None:
            xr.destroy_instance(self.instance)
            self.instance = None

    # ------------------------------------------------------------------ #
    # BaseTracker interface
    # ------------------------------------------------------------------ #
    # poll() / get_all_poses() / get_pose_by_role() 以及 with 上下文、录制相关方法
    # (start_recording / stop_recording / save_recording / discard ...) 全部由
    # BaseTracker 提供。这里只实现 5 个硬件相关成员:

    @property
    def device_name(self):
        """逻辑设备名, 用作 .npz 键前缀 (不含 "__")。"""
        return self._device_name

    def connect(self):
        """建立 OpenXR 实例/会话并绑定 tracker 角色 (BaseTracker 接口); 等价于 start()。"""
        self.start()

    def disconnect(self):
        """销毁 OpenXR 对象 (BaseTracker 接口); 等价于 close()。"""
        self.close()

    def is_connected(self):
        """OpenXR 会话是否已创建。"""
        return self.session is not None

    def _refresh(self):
        """按主机时钟采样并刷新位姿缓存 (BaseTracker 接口)。

        由基类的 poll()/getter 以紧循环调用。采样节奏由 sample_fps 决定:
        仅在 FOCUSED 且到达下一个采样时刻时执行主机时钟 locate (见 _sample_now,
        热路径不含 xrSyncActions), 并按采样间隔自行 sleep 定速; 完整的合成器帧
        (wait/begin/end + 帧内 sync) 每 _FRAME_PUMP_KEEPALIVE_S 才泵一次用于
        会话保活。返回 self.latest_poses。
        """
        self._poll_events()
        if self.should_exit:
            return self.latest_poses
        if not self.session_is_running:
            # 会话还没 READY;  gentle on the CPU while we wait.
            time.sleep(0.05)
            return self.latest_poses
        if self.session_state not in self._RENDERABLE_STATES:
            time.sleep(0.01)
            return self.latest_poses

        now = time.perf_counter()
        if now - self._last_frame_pump_perf >= self._FRAME_PUMP_KEEPALIVE_S:
            # 保活帧可能阻塞到下一个合成器帧 (headless 下实测 ~66 ms);
            # 期间只是少采几帧, 不会把整体采样拖回合成器节拍.
            self._pump_frame()
            self._last_frame_pump_perf = time.perf_counter()
            now = self._last_frame_pump_perf

        if self.session_state == xr.SessionState.FOCUSED:
            if (self._sample_interval <= 0.0
                    or now - self._last_sample_perf >= self._sample_interval):
                self._sample_now()
                self._last_sample_perf = time.perf_counter()
            if self._sample_interval > 0.0:
                wait_s = (self._sample_interval
                          - (time.perf_counter() - self._last_sample_perf))
                if wait_s > 0.0:
                    time.sleep(wait_s)
        return self.latest_poses

    # ------------------------------------------------------------------ #
    # ViveTracker-specific API
    # ------------------------------------------------------------------ #
    def get_connected_trackers(self):
        """Return a list of every currently connected tracker as
        [{"serial": <persistent path>, "role": <role path or None>}, ...],
        regardless of whether it reports a pose. Handy for discovering serials
        and diagnosing role mismatches.
        """
        trackers = []
        n = ctypes.c_uint32(0)
        result = self._enumerate_paths_fn(self.instance, 0, byref(n), None)
        if xr.check_result(result).is_exception():
            raise RuntimeError(f"xrEnumerateViveTrackerPathsHTCX failed: {xr.Result(result)}")
        paths = (xr.ViveTrackerPathsHTCX * n.value)(*([xr.ViveTrackerPathsHTCX()] * n.value))
        result = self._enumerate_paths_fn(self.instance, n, byref(n), paths)
        if xr.check_result(result).is_exception():
            raise RuntimeError(f"xrEnumerateViveTrackerPathsHTCX failed: {xr.Result(result)}")
        for tp in paths:
            serial = xr.path_to_string(self.instance, tp.persistent_path)
            role = (xr.path_to_string(self.instance, tp.role_path)
                    if tp.role_path != xr.NULL_PATH else None)
            trackers.append({"serial": serial, "role": role})
        return trackers

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _setup_tracker_bindings(self, action_name, roles):
        rp_strings = [f"/user/vive_tracker_htcx/role/{r}" for r in roles]
        rps = (xr.Path * len(rp_strings))(
            *[xr.string_to_path(self.instance, s) for s in rp_strings],
        )
        action = xr.create_action(
            action_set=self.action_set,
            create_info=xr.ActionCreateInfo(
                action_type=xr.ActionType.POSE_INPUT,
                action_name=action_name,
                localized_action_name="Tracker Pose",
                count_subaction_paths=len(rps),
                subaction_paths=rps,
            ),
        )
        binding_paths = (xr.ActionSuggestedBinding * len(rp_strings))(
            *[xr.ActionSuggestedBinding(
                action,
                xr.string_to_path(self.instance, f"{s}/input/grip/pose"))
              for s in rp_strings],
        )
        xr.suggest_interaction_profile_bindings(
            instance=self.instance,
            suggested_bindings=xr.InteractionProfileSuggestedBinding(
                interaction_profile=xr.string_to_path(
                    self.instance, "/interaction_profiles/htc/vive_tracker_htcx"),
                count_suggested_bindings=len(binding_paths),
                suggested_bindings=binding_paths,
            ),
        )
        spaces = (xr.Space * len(rps))(
            *[xr.create_action_space(
                session=self.session,
                create_info=xr.ActionSpaceCreateInfo(action=action, subaction_path=rp),
            ) for rp in rps],
        )
        return rp_strings, action, spaces

    def _poll_events(self):
        while True:
            try:
                event_buffer = xr.poll_event(self.instance)
            except xr.EventUnavailable:
                break
            event_type = xr.StructureType(event_buffer.type)
            if event_type == xr.StructureType.EVENT_DATA_INSTANCE_LOSS_PENDING:
                self.should_exit = True
            elif (event_type == xr.StructureType.EVENT_DATA_SESSION_STATE_CHANGED
                  and self.session is not None):
                event = cast(byref(event_buffer),
                             POINTER(xr.EventDataSessionStateChanged)).contents
                self.session_state = xr.SessionState(event.state)
                if self.session_state == xr.SessionState.READY:
                    xr.begin_session(
                        session=self.session,
                        begin_info=xr.SessionBeginInfo(
                            xr.ViewConfigurationType.PRIMARY_STEREO),
                    )
                    self.session_is_running = True
                elif self.session_state == xr.SessionState.STOPPING:
                    self.session_is_running = False
                    xr.end_session(self.session)
                elif self.session_state in (xr.SessionState.EXITING,
                                            xr.SessionState.LOSS_PENDING):
                    self.should_exit = True

    def _host_xr_time(self):
        # Headless sessions can report predicted_display_time == 0, which is an
        # invalid XrTime. Derive a valid time from the Win32 performance counter.
        counter = ctypes.c_longlong()
        ctypes.windll.kernel32.QueryPerformanceCounter(ctypes.byref(counter))
        return xr.convert_win32_performance_counter_to_time_khr(self.instance, counter)

    def _locate_time(self, frame_state):
        if frame_state.predicted_display_time > 0:
            return frame_state.predicted_display_time
        return self._host_xr_time()

    def _raise_timer_resolution(self):
        # time.sleep() 的默认粒度 (~15.6 ms) 会把 90 Hz 采样压到 ~64 Hz;
        # 在进程内把 Windows 计时器分辨率提到 1 ms, close() 时恢复.
        if self._timer_resolution_raised or os.name != "nt":
            return
        try:
            ctypes.windll.winmm.timeBeginPeriod(1)
            self._timer_resolution_raised = True
        except Exception:
            pass

    def _restore_timer_resolution(self):
        if not self._timer_resolution_raised:
            return
        self._timer_resolution_raised = False
        try:
            ctypes.windll.winmm.timeEndPeriod(1)
        except Exception:
            pass

    def _pump_frame(self):
        self._poll_events()
        if self.should_exit:
            return
        if not self.session_is_running:
            # Nothing to render yet; be gentle on the CPU while we wait for the
            # runtime to move the session to READY.
            time.sleep(0.05)
            return
        if self.session_state not in self._RENDERABLE_STATES:
            return

        frame_state = xr.wait_frame(self.session)
        xr.begin_frame(self.session)
        if self.session_state == xr.SessionState.FOCUSED:
            self._sync_and_locate(frame_state)
        xr.end_frame(
            self.session,
            frame_end_info=xr.FrameEndInfo(
                display_time=frame_state.predicted_display_time,
                environment_blend_mode=xr.EnvironmentBlendMode.OPAQUE,
                layers=[],
            ),
        )

    def _sync_actions(self):
        active = xr.ActiveActionSet(
            action_set=self.action_set,
            subaction_path=xr.NULL_PATH,
        )
        xr.sync_actions(
            session=self.session,
            sync_info=xr.ActionsSyncInfo(
                count_active_action_sets=1,
                active_action_sets=ctypes.pointer(active),
            ),
        )

    def _locate_spaces(self, locate_time):
        poses = {}
        for index, space in enumerate(self.tracker_action_spaces):
            loc = xr.locate_space(space=space, base_space=self.space, time=locate_time)
            if loc.location_flags & xr.SPACE_LOCATION_POSITION_VALID_BIT:
                p = loc.pose
                poses[self.role_strings[index]] = {
                    "position": {"x": p.position.x, "y": p.position.y, "z": p.position.z},
                    "orientation": {
                        "w": p.orientation.w, "x": p.orientation.x,
                        "y": p.orientation.y, "z": p.orientation.z,
                    },
                }
        self.latest_poses = poses

    def _sync_and_locate(self, frame_state):
        """帧内采样 (保活帧路径): sync 后按帧的 locate_time 定位。"""
        self._sync_actions()
        self._locate_spaces(self._locate_time(frame_state))

    def _sample_now(self):
        """主机时钟采样: 在"当前时刻"定位, 不要求位于帧内。

        Headless 下合成器帧节拍可慢至 ~15 Hz, 但 tracker 设备以 ~90+ Hz
        上报, 而 xrLocateSpace 接受任意 XrTime, 因此按主机时钟采样即可
        绕过合成器帧节流。

        注意: 这里**不要**调用 xrSyncActions。帧外的 sync 在 SteamVR 上会
        阻塞到下一个合成器帧 (headless 实测 ~66 ms), 每次采样都 sync 会把
        采样率拖回合成器节拍 (~15 Hz; 见 session_20260912_163132_task1 回归)。
        action 同步只做在保活帧内 (_pump_frame -> _sync_and_locate), 已足够
        维持 action space 激活; locate 按 XrTime 从追踪缓冲取值, 不依赖每次
        sync 的新鲜动作状态。
        """
        self._locate_spaces(self._host_xr_time())


if __name__ == "__main__":
    with ViveTracker() as vt:
        print("Reading Vive tracker poses. Ctrl+C to stop.")

        # One-time: list every connected tracker with its serial + SteamVR role.
        # If a tracker's role is not in ROLES_FULL/ROLES_CLASSIC, it won't
        # produce a pose until you add that role.
        try:
            for t in vt.get_connected_trackers():
                print(f"  connected: serial={t['serial']}  role={t['role']}")
        except Exception as exc:
            print(f"  (could not enumerate trackers yet: {exc!r})")

        try:
            while not vt.should_exit:
                time.sleep(1)
                # Method 2: all trackers at once -> {role: {position, orientation}}
                poses = vt.get_all_poses()          # samples on the host clock
                for role, pose in poses.items():
                    p, q = pose["position"], pose["orientation"]
                    print(f"{role:>15}: pos=({p['x']:.3f}, {p['y']:.3f}, {p['z']:.3f}) "
                          f"quat=({q['w']:.3f}, {q['x']:.3f}, {q['y']:.3f}, {q['z']:.3f})")
                if poses:
                    print("-" * 70)

                # Method 1: a single role by name. Read from the frame we just
                # pumped (refresh=False) so we don't advance an extra frame.
                # waist = vt.get_pose_by_role("waist", refresh=False)
                # if waist is not None:
                #     print("waist ->", waist)
        except KeyboardInterrupt:
            print("\nstopped.")