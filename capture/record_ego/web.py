"""Ego 实时手部/tracker 快照、相机 MJPEG 泵和 HTTP 服务。"""

import json
import math
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np


# Manus 与 Wuji 现在使用同一套 21 节点 MediaPipe 拓扑: Manus 原始 25 节点骨骼
# 在 receiver 分发前已丢弃 4 个掌骨关节 (devices/manus_glove.py::_to_21_nodes)。
MANUS_HAND_PARENTS = [
    -1,
    0, 1, 2, 3,
    0, 5, 6, 7,
    0, 9, 10, 11,
    0, 13, 14, 15,
    0, 17, 18, 19,
]

WUJI_HAND_PARENTS = [
    -1,
    0, 1, 2, 3,
    0, 5, 6, 7,
    0, 9, 10, 11,
    0, 13, 14, 15,
    0, 17, 18, 19,
]


def standard_hand_parents(glove_type="wuji"):
    """按手套类型返回页面使用的原生骨架父节点索引。"""
    return list(MANUS_HAND_PARENTS if glove_type == "manus" else WUJI_HAND_PARENTS)


def _normalized_hand_parents(parents):
    """把设备层级规约为页面约定并拒绝越界、森林或成环关系。"""
    try:
        result = [int(parent) for parent in parents]
    except (TypeError, ValueError):
        return None
    node_count = len(result)
    if not node_count:
        return None
    result = [-1 if parent == child else parent
              for child, parent in enumerate(result)]
    if result.count(-1) != 1:
        return None
    if any(parent < -1 or parent >= node_count for parent in result):
        return None
    for child in range(node_count):
        current, seen = child, set()
        while current >= 0:
            if current in seen:
                return None
            seen.add(current)
            current = result[current]
    return result


def hand_parents_snapshot(gloves, sides, glove_type="wuji"):
    """返回每只手当前的真实父索引；尚未收到层级时使用驱动标准拓扑。"""
    fallback = standard_hand_parents(glove_type)
    result = {}
    for glove in gloves:
        side = sides.get(id(glove))
        if side not in ("left", "right"):
            continue
        info = None
        try:
            info = glove.glove_info()
        except Exception:
            pass
        parents = _normalized_hand_parents(
            info.get("hand_hierarchy") if isinstance(info, dict) else None)
        result[side] = parents if parents is not None else list(fallback)
    return result


# ============================================================
# 左右手判定（与 visualizer 的 assignHandSides 同逻辑：先看 slot 文本，再按出现顺序兜底）
# ============================================================

def _slot_text(g):
    """尽力取到一只手套的“左右”线索文本（slot / device_name）。"""
    txt = ""
    try:
        info = g.glove_info() if hasattr(g, "glove_info") else None
        if isinstance(info, dict):
            txt = str(info.get("slot", "") or "")
    except Exception:
        txt = ""
    if not txt:
        txt = str(getattr(g, "slot", "") or "")
    if not txt:
        try:
            txt = str(getattr(g, "device_name", "") or "")
        except Exception:
            txt = ""
    return txt


def assign_sides(gloves):
    """返回 {id(glove): 'left'/'right'}。左=glove_0、右=glove_1；文本不明时按出现顺序补位。"""
    sides = {}
    left = right = None
    unknown = []
    for g in gloves:
        s = _slot_text(g).lower()
        if ("left" in s) or ("左" in s) or ("glove_0" in s) or s.endswith("_0"):
            if left is None:
                left = g; sides[id(g)] = "left"
            else:
                unknown.append(g)
        elif ("right" in s) or ("右" in s) or ("glove_1" in s) or s.endswith("_1"):
            if right is None:
                right = g; sides[id(g)] = "right"
            else:
                unknown.append(g)
        else:
            unknown.append(g)
    for g in unknown:
        if left is None:
            left = g; sides[id(g)] = "left"
        elif right is None:
            right = g; sides[id(g)] = "right"
        else:
            sides[id(g)] = "left"
    return sides


# ============================================================
# 实时取数：tracker 位姿
#   用 BaseTracker 文档化接口 get_all_poses()/latest_poses：
#   {role: {"position": {x,y,z}, "orientation": {w,x,y,z}}}
# ============================================================

def read_tracker_snapshot(tracker):
    """返回 {role: {"p":[x,y,z], "q":[w,x,y,z], "valid":bool}}；无 tracker 返回 {}。"""
    if tracker is None:
        return {}
    poses = None
    for name in ("get_all_poses", "latest_poses"):
        try:
            obj = getattr(tracker, name, None)
            poses = obj() if callable(obj) else obj
            if poses:
                break
        except Exception:
            poses = None
    out = {}
    if not poses:
        return out
    try:
        items = poses.items()
    except Exception:
        return out
    for role, pose in items:
        try:
            if not isinstance(pose, dict):
                continue
            pos = pose.get("position") or {}
            ori = pose.get("orientation") or {}
            p = [float(pos.get("x", 0.0)), float(pos.get("y", 0.0)), float(pos.get("z", 0.0))]
            q = [float(ori.get("w", 1.0)), float(ori.get("x", 0.0)),
                 float(ori.get("y", 0.0)), float(ori.get("z", 0.0))]
            valid = bool(pose.get("valid", True))
            out[str(role)] = {"p": [round(v, 4) for v in p],
                              "q": [round(v, 5) for v in q], "valid": valid}
        except Exception:
            continue
    return out


# ============================================================
# 实时取数：Manus/Wuji 手套骨骼
#
#   WujiGlove.latest 与 ManusGlove.latest_skeleton 都会保留最新帧。
#   install_live_skeleton_tap 继续兼容旧 Manus 驱动；HandTap 保留
#   export_arrays() 兜底。两种手套均保持原生节点数，不修改落盘数据。
# ============================================================

def _nodes_to_xyz(nodes):
    """从 Manus UDP 节点列表提取位置。"""
    points = []
    for node in nodes:
        try:
            pos = node.get("pos", (0.0, 0.0, 0.0))
            points.append([float(pos[0]), float(pos[1]), float(pos[2])])
        except Exception:
            points.append([0.0, 0.0, 0.0])
    return points


def install_live_skeleton_tap(gloves):
    """为旧式 Manus ``_feed_frame`` 驱动缓存未录制时的最新骨架。"""
    for glove in gloves:
        if getattr(glove, "_live_tap_installed", False):
            continue
        original = getattr(glove, "_feed_frame", None)
        if not callable(original):
            continue

        def make_wrapper(original_fn, target):
            def wrapped(nodes, parents, values, host_us):
                if nodes:
                    try:
                        target._live_skeleton = _nodes_to_xyz(nodes)
                    except Exception:
                        pass
                return original_fn(nodes, parents, values, host_us)
            return wrapped

        glove._live_skeleton = None
        glove._feed_frame = make_wrapper(original, glove)
        glove._live_tap_installed = True
    return gloves


class HandTap:
    def __init__(self, gloves, sides, use_export_fallback=True):
        self.gloves = gloves
        self.sides = sides                       # {id(glove): 'left'/'right'}
        self.use_export_fallback = use_export_fallback
        self._last_export = 0.0

    @staticmethod
    def _to_xyz(arr):
        """把 (N,7)/(N,3)/(N,>=3) 的 ndarray 或嵌套 list 规约为 [[x,y,z], ...]；失败 None。"""
        try:
            a = np.asarray(arr, dtype=np.float64)
        except Exception:
            return None
        if a.ndim == 1:
            if a.size and a.size % 7 == 0:
                a = a.reshape(-1, 7)
            elif a.size and a.size % 3 == 0:
                a = a.reshape(-1, 3)
            else:
                return None
        if a.ndim != 2 or a.shape[0] == 0 or a.shape[1] < 3:
            return None
        a = np.nan_to_num(a[:, :3], nan=0.0, posinf=0.0, neginf=0.0)
        return np.round(a, 4).tolist()

    def _lightweight(self, g):
        """轻量：尝试常见“最新帧”属性/方法。命中返回 [[x,y,z]...]，否则 None。"""
        # 1) 直接属性：优先 Manus 实时缓存，再兼容其它常见名。
        for name in ("_live_skeleton", "latest_skeleton", "latest_hand_skeleton",
                     "last_skeleton", "current_skeleton", "skeleton"):
            v = getattr(g, name, None)
            if v is not None:
                xyz = self._to_xyz(v)
                if xyz:
                    return xyz
        # 2) dict 形态：g.latest["hand_skeleton"] / g.last["hand_skeleton"]
        for name in ("latest", "last", "current"):
            d = getattr(g, name, None)
            if isinstance(d, dict):
                hv = d.get("hand_skeleton")
                if hv is not None:
                    xyz = self._to_xyz(hv)
                    if xyz:
                        return xyz
        # 3) 方法形态：g.get_latest("hand_skeleton") / g.peek(...) / g.snapshot(...)
        for name in ("get_latest", "latest_of", "peek", "snapshot", "get_stream"):
            f = getattr(g, name, None)
            if callable(f):
                for call in (lambda: f("hand_skeleton"), lambda: f()):
                    try:
                        r = call()
                    except TypeError:
                        continue
                    except Exception:
                        break
                    if isinstance(r, dict):
                        r = r.get("hand_skeleton")
                    if isinstance(r, (tuple, list)) and r and not np.isscalar(r[0]):
                        # 可能是 (data, ts) 元组
                        if len(r) == 2 and np.ndim(r[0]) >= 1:
                            r = r[0]
                    xyz = self._to_xyz(r) if r is not None else None
                    if xyz:
                        return xyz
        return None

    def _from_export(self, g):
        """文档化兜底：export_arrays() 取 *__hand_skeleton 的最后一帧（较重，故限流调用）。"""
        f = getattr(g, "export_arrays", None)
        if not callable(f):
            return None
        try:
            a, _ch, _npk = f()
        except Exception:
            return None
        for k, v in (a or {}).items():
            if str(k).endswith("__hand_skeleton"):
                arr = np.asarray(v)
                if arr.ndim == 3 and arr.shape[0] > 0:
                    return self._to_xyz(arr[-1])
        return None

    def snapshot(self):
        """返回 {"left":[[x,y,z]...], "right":[...]}，仅含当前取到数据的手。"""
        out = {}
        allow_export = self.use_export_fallback and (time.time() - self._last_export > 0.3)
        used_export = False
        for g in self.gloves:
            side = self.sides.get(id(g), "left")
            xyz = None
            try:
                xyz = self._lightweight(g)
            except Exception:
                xyz = None
            if xyz is None and allow_export:
                xyz = self._from_export(g)
                used_export = True
            if xyz:
                out[side] = xyz
        if used_export:
            self._last_export = time.time()
        return out


class TactileTap:
    """读取 WujiGlove 保留的最新 24×31 触觉矩阵，供网页热力图展示。"""

    def __init__(self, gloves, sides):
        self.gloves = gloves
        self.sides = sides

    @staticmethod
    def _latest_tactile(glove):
        latest_stream = getattr(glove, "latest_stream", None)
        if callable(latest_stream):
            try:
                result = latest_stream("tactile") or {}
                if "tactile" in result:
                    return result["tactile"]
            except Exception:
                pass
        latest = getattr(glove, "latest", None)
        if isinstance(latest, dict):
            return latest.get("tactile")
        return None

    def snapshot(self):
        """返回按左右手组织的 row-major 触觉矩阵；无数据的手不出现在结果中。"""
        out = {}
        for glove in self.gloves:
            values = self._latest_tactile(glove)
            if values is None:
                continue
            try:
                array = np.asarray(values, dtype=np.float32).reshape(-1)
            except Exception:
                continue
            if array.size == 0 or array.size % 24 != 0:
                continue
            # 当前固件为 24×31；保留对旧 24×32 录制/固件的兼容显示。
            cols = int(array.size // 24)
            if cols not in (31, 32):
                continue
            finite = np.nan_to_num(array, nan=-1.0, posinf=1.0, neginf=-1.0)
            finite = np.where(finite < 0, -1.0, np.clip(finite, 0.0, 1.0))
            side = self.sides.get(id(glove), "left")
            out[side] = {
                "rows": 24,
                "cols": cols,
                "values": np.round(finite, 3).tolist(),
                "tHostUs": int(getattr(glove, "latest_t_host_us", {}).get(
                    "tactile", 0)),
            }
        return out


# ============================================================
# 相机泵：后台持续取最新帧编码成 JPEG，供所有 MJPEG 客户端共享
#   —— 单读者，避免多个 /stream.mjpg 连接互相抢帧；录制中用相机缓存帧（不抢录制）。
# ============================================================

class CameraPump(threading.Thread):
    def __init__(self, camera, status_fn, quality=80, target_fps=30, overlay=True,
                 max_width=0, max_height=0):
        super().__init__(name="cam-pump", daemon=True)
        self.camera = camera
        self.status_fn = status_fn            # () -> (rec: bool, dur: float)
        self.quality = int(quality)
        self.interval = 1.0 / max(1, int(target_fps))
        self.overlay = overlay
        self.max_width = max(0, int(max_width or 0))
        self.max_height = max(0, int(max_height or 0))
        self._jpg = self._placeholder(
            "等待相机画面…" if camera is not None else "无相机 (--no-camera)")
        self._seq = 0
        self._last_frame_monotonic = None
        self._cond = threading.Condition()
        self._stop = threading.Event()

    @staticmethod
    def _placeholder(text):
        img = np.full((360, 640, 3), (24, 28, 34), np.uint8)
        cv2.putText(img, text, (40, 190), cv2.FONT_HERSHEY_SIMPLEX,
                    0.85, (150, 158, 168), 2, cv2.LINE_AA)
        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        return buf.tobytes() if ok else b""

    def preview_size(self, source_width, source_height):
        """返回保持宽高比、且不放大原图的监看流尺寸。"""
        width, height = int(source_width), int(source_height)
        if width <= 0 or height <= 0:
            return width, height
        scales = [1.0]
        if self.max_width:
            scales.append(self.max_width / width)
        if self.max_height:
            scales.append(self.max_height / height)
        scale = min(scales)
        return max(1, round(width * scale)), max(1, round(height * scale))

    def _resize_preview(self, frame):
        height, width = frame.shape[:2]
        target_width, target_height = self.preview_size(width, height)
        if (target_width, target_height) == (width, height):
            return frame
        return cv2.resize(
            frame, (target_width, target_height), interpolation=cv2.INTER_AREA)

    def run(self):
        while not self._stop.is_set():
            t0 = time.time()
            frame = None
            if self.camera is not None:
                try:
                    frame = self.camera.grab_preview()
                except Exception:
                    frame = None
            if frame is not None:
                view = self._resize_preview(frame)
                if self.overlay:
                    try:
                        rec_on, dur = self.status_fn()
                        if rec_on:
                            view = view.copy()
                            cv2.putText(view, "REC %5.1fs" % dur, (18, 42),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
                            cv2.circle(view, (view.shape[1] - 28, 30), 10, (0, 0, 255), -1)
                    except Exception:
                        view = self._resize_preview(frame)
                ok, buf = cv2.imencode(".jpg", view,
                                       [int(cv2.IMWRITE_JPEG_QUALITY), self.quality])
                if ok:
                    with self._cond:
                        self._jpg = buf.tobytes()
                        self._seq += 1
                        self._last_frame_monotonic = time.monotonic()
                        self._cond.notify_all()
            dt = time.time() - t0
            if dt < self.interval:
                self._stop.wait(self.interval - dt)

    def frame(self, last_seq):
        """阻塞直到有新帧(或 1s 超时)；返回 (jpg_bytes, seq)。"""
        with self._cond:
            if self._seq == last_seq:
                self._cond.wait(timeout=1.0)
            return self._jpg, self._seq

    def online(self, timeout=2.0):
        """最近 ``timeout`` 秒内成功取得相机帧时返回 True。"""
        with self._cond:
            last_frame = self._last_frame_monotonic
        return bool(
            self.camera is not None
            and last_frame is not None
            and time.monotonic() - last_frame <= max(0.1, float(timeout)))

    def stop(self):
        self._stop.set()
        with self._cond:
            self._cond.notify_all()


# ============================================================
# HTTP 服务：/(页面) /stream.mjpg /live.json /meta.json /cmd
# ============================================================

class WebContext:
    def __init__(self, html_path, pump, live_fn, meta_fn, cmd_fn, upload_fn=None,
                 task_fn=None):
        self.html_path = html_path
        self.pump = pump
        self.live_fn = live_fn      # () -> dict
        self.meta_fn = meta_fn      # () -> dict
        self.cmd_fn = cmd_fn        # (ch, countdown) -> str(msg)
        self.upload_fn = upload_fn  # (start: bool) -> dict
        self.task_fn = task_fn      # (offset: -1 | 1) -> dict


class EgoHandler(BaseHTTPRequestHandler):
    ctx = None                      # 由子类注入
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):      # 安静
        pass

    # -- 工具 --
    def _route(self):
        u = urlparse(self.path)
        return u.path, parse_qs(u.query)

    def _write(self, body, code=200, ctype="application/octet-stream", extra=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        if extra:
            for k, v in extra.items():
                self.send_header(k, v)
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass

    def _json(self, obj, code=200):
        self._write(json.dumps(obj, ensure_ascii=False).encode("utf-8"),
                    code=code, ctype="application/json; charset=utf-8")

    # -- 路由 --
    def do_GET(self):
        path, qs = self._route()
        if path in ("/", "/index.html", "/record_ego_page.html", "/ego.html"):
            return self._serve_html()
        if path == "/stream.mjpg":
            return self._serve_mjpeg()
        if path == "/live.json":
            return self._json(self.ctx.live_fn())
        if path == "/meta.json":
            return self._json(self.ctx.meta_fn())
        if path == "/upload.json" and self.ctx.upload_fn is not None:
            return self._json(self.ctx.upload_fn(False))
        if path == "/cmd":
            return self._do_cmd((qs.get("k") or [""])[0], qs)
        if path == "/favicon.ico":
            self.send_response(204); self.send_header("Content-Length", "0"); self.end_headers()
            return
        self._write(b"not found", code=404, ctype="text/plain")

    def do_POST(self):
        path, qs = self._route()
        if path == "/cmd":
            return self._do_cmd((qs.get("k") or [""])[0], qs)
        if path == "/task" and self.ctx.task_fn is not None:
            return self._do_task(qs)
        if path == "/upload" and self.ctx.upload_fn is not None:
            result = self.ctx.upload_fn(True)
            return self._json(result, code=200 if result.get("ok") else 409)
        self._write(b"not found", code=404, ctype="text/plain")

    def _do_cmd(self, k, qs=None):
        k = (k or "").strip().lower()[:1]
        if k in ("r", "d", "e", "q"):
            countdown = None
            if k == "r" and qs and "countdown" in qs:
                try:
                    countdown = float(qs["countdown"][0])
                    if not math.isfinite(countdown) or not 0 <= countdown <= 3600:
                        raise ValueError
                except (TypeError, ValueError, IndexError):
                    return self._json(
                        {"ok": False, "msg": "倒计时必须是 0～3600 秒"}, code=400)
            msg = self.ctx.cmd_fn(k, countdown) or ""
            return self._json({"ok": True, "msg": msg})
        return self._json({"ok": False, "msg": "未知指令"}, code=400)

    def _do_task(self, qs):
        try:
            offset = int((qs.get("offset") or [""])[0])
            if offset not in (-1, 1):
                raise ValueError
        except (TypeError, ValueError, IndexError):
            return self._json(
                {"ok": False, "msg": "任务切换方向必须是 -1 或 1"}, code=400)
        result = self.ctx.task_fn(offset) or {
            "ok": False, "msg": "任务切换失败"
        }
        return self._json(result, code=200 if result.get("ok") else 409)

    def _serve_html(self):
        try:
            with open(self.ctx.html_path, "rb") as f:
                body = f.read()
        except OSError:
            body = ("<h1>找不到 record_ego_page.html</h1>"
                    "<p>请把页面文件与 record_ego_data.py 放在同一目录，"
                    "或用 --page 指定路径。</p>").encode("utf-8")
        self._write(body, ctype="text/html; charset=utf-8")

    def _serve_mjpeg(self):
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        pump = self.ctx.pump
        seq = -1
        try:
            while True:
                jpg, seq = pump.frame(seq)
                if not jpg:
                    time.sleep(0.05)
                    continue
                self.wfile.write(b"--frame\r\n")
                self.wfile.write(b"Content-Type: image/jpeg\r\n")
                self.wfile.write(("Content-Length: %d\r\n\r\n" % len(jpg)).encode("ascii"))
                self.wfile.write(jpg)
                self.wfile.write(b"\r\n")
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass
        except Exception:
            pass


def make_server(port, ctx, host="127.0.0.1"):
    handler = type("EgoHandlerBound", (EgoHandler,), {"ctx": ctx})
    for p in range(port, port + 20):
        try:
            httpd = ThreadingHTTPServer((host, p), handler)
            return httpd, p
        except OSError:
            continue
    raise SystemExit("找不到可用网页端口 (%d..%d)" % (port, port + 20))
