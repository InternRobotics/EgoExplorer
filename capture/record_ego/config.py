"""Ego 数据采集的默认配置与命令行参数。"""

import argparse
import os


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_PAGE = os.path.join(PROJECT_ROOT, "record_ego_page.html")
DEFAULT_WUJI_CALIBRATION = os.path.join(
    PROJECT_ROOT, "wuji_calibration", "calibration_bundle.zip")
DEFAULT_WEB_PORT = 8770
DEFAULT_SAVE_DIR = "./data"
DEFAULT_RECORD_COUNTDOWN = 3.0
DEFAULT_STORAGE_LIMIT_GB = 50.0
DEFAULT_STORAGE_CHECK_INTERVAL_MINUTES = 30.0

CAM_DEVICE = 0
CAM_WIDTH, CAM_HEIGHT = 2560, 1440
CAM_FPS = 60
MP4_FPS = None
MP4_CODEC = "h264"
MP4_CRF = 20
H264_PRESET = "ultrafast"
H264_THREADS = 4
WUJI_PREVIEW_WIDTH = 640
WUJI_PREVIEW_HEIGHT = 360

STATUS_INTERVAL = 0.5
TRACKER_LISTEN_HOST = "0.0.0.0"
TRACKER_LISTEN_PORT = 8766
TRACKER_RECORD_FPS = 120.0
TRACKER_SAMPLE_FPS = 90.0
MANUS_IP = "0.0.0.0"
MANUS_PORT = 9000
MANUS_WAIT = 1.5


def parse_args(argv=None, glove_type="manus"):
    """解析采集、编码和网页服务参数。"""
    glove_type = str(glove_type).lower()
    if glove_type not in ("manus", "wuji"):
        raise ValueError(f"不支持的手套类型: {glove_type!r}")
    glove_label = "Manus" if glove_type == "manus" else "Wuji"
    parser = argparse.ArgumentParser(
        description=f"同步采集 GoPro + {glove_label} 手套 + Tracker"
                    "（页面按钮 & 终端键盘 r/d/e/q 并行）")

    camera = parser.add_argument_group("相机与编码")
    camera.add_argument("--cam-device", default=str(CAM_DEVICE),
                        help=f"采集卡设备：索引(如 0)或路径(如 /dev/video1)。默认 {CAM_DEVICE}")
    camera.add_argument("--cam-width", type=int, default=CAM_WIDTH,
                        help=f"默认 {CAM_WIDTH}")
    camera.add_argument("--cam-height", type=int, default=CAM_HEIGHT,
                        help=f"默认 {CAM_HEIGHT}")
    camera.add_argument("--cam-fps", type=int, default=CAM_FPS,
                        help=f"默认 {CAM_FPS}")
    camera.add_argument("--mp4-fps", type=float, default=MP4_FPS,
                        help=("编码器名义帧率；默认跟随 --cam-fps。实际 MP4 播放时间由"
                              "逐帧抓取 PTS 决定，不影响对齐"))
    camera.add_argument("--video-codec", default=MP4_CODEC,
                        choices=["h264", "mp4v"],
                        help=f"写 MP4 的编码，默认 {MP4_CODEC}")
    camera.add_argument("--crf", type=int, default=MP4_CRF,
                        help=f"H.264 画质 0~51，默认 {MP4_CRF}")
    camera.add_argument(
        "--h264-preset", default=H264_PRESET,
        choices=["ultrafast", "superfast", "veryfast", "faster", "fast", "medium"],
        help=f"软件 H.264 编码速度档位，默认 {H264_PRESET}")
    camera.add_argument(
        "--h264-threads", type=int, default=H264_THREADS,
        help=f"软件 H.264 编码线程数；0 表示 ffmpeg 自动选择，默认 {H264_THREADS}")
    camera.add_argument("--no-cam-scan", action="store_true",
                        help="指定设备失败时不扫描其它 /dev/video* 节点")
    camera.add_argument("--no-camera", action="store_true",
                        help="不开相机（手套和 tracker 仍会自动连接）")

    if glove_type == "manus":
        manus = parser.add_argument_group("Manus")
        manus.add_argument("--manus-ip", default=MANUS_IP,
                           help=f"UDP 监听地址，默认 {MANUS_IP}")
        manus.add_argument("--manus-port", type=int, default=MANUS_PORT,
                           help=f"UDP 监听端口，默认 {MANUS_PORT}")
        manus.add_argument("--manus-wait", type=float, default=MANUS_WAIT,
                           help=f"启动数据探测秒数，默认 {MANUS_WAIT}")
        manus.add_argument(
            "--tracker-sample-fps", type=float, default=TRACKER_SAMPLE_FPS,
            help=("本机 OpenXR Tracker 的位姿采样帧率，按主机时钟驱动，"
                  "不受 headless 合成器帧节拍 (~15 Hz) 限制。"
                  f"0 表示不限制，默认 {TRACKER_SAMPLE_FPS:g}"))
    else:
        wuji = parser.add_argument_group("Wuji")
        wuji.add_argument(
            "--wuji-calibration", default=DEFAULT_WUJI_CALIBRATION,
            help=f"Linux 本地 SDK 使用的标定 bundle。默认 {DEFAULT_WUJI_CALIBRATION}")
        wuji.add_argument(
            "--no-wuji-calibration", action="store_true",
            help="不导入标定 bundle，使用 Linux SDK 当前用户")

        tracker = parser.add_argument_group("Windows Tracker 网络接收")
        tracker.add_argument(
            "--tracker-host", default=TRACKER_LISTEN_HOST,
            help=f"Linux tracker 接收服务监听地址，默认 {TRACKER_LISTEN_HOST}")
        tracker.add_argument(
            "--tracker-port", type=int, default=TRACKER_LISTEN_PORT,
            help=f"Linux tracker 接收服务 TCP 端口，默认 {TRACKER_LISTEN_PORT}")
        tracker.add_argument(
            "--tracker-token", default=os.environ.get("EGO_TRACKER_TOKEN", ""),
            help="可选共享密钥；也可用 EGO_TRACKER_TOKEN")
        tracker.add_argument(
            "--tracker-record-fps", type=float, default=TRACKER_RECORD_FPS,
            help=("Linux 端写入 Tracker 录制缓存的最高帧率；防止旧版发送端无上限发送。"
                  f"0 表示不限速，默认 {TRACKER_RECORD_FPS:g}"))

    tracker_common = parser.add_argument_group("Tracker")
    tracker_common.add_argument(
        "--no-tracker", action="store_true",
        help=("不启动 Windows tracker 网络接收服务" if glove_type == "wuji"
              else "不连接本机 OpenXR Tracker"))

    web = parser.add_argument_group("存储与网页")
    web.add_argument(
        "--task-setting", required=True,
        help="任务流程 JSON 文件路径；内容必须是非空任务名称字符串数组")
    web.add_argument("--save-dir", default=DEFAULT_SAVE_DIR,
                     help=f"保存目录，默认 {DEFAULT_SAVE_DIR}")
    web.add_argument("--record-countdown", type=float,
                     default=DEFAULT_RECORD_COUNTDOWN,
                     help=f"开始记录前的倒计时秒数，网页中可临时修改；默认 {DEFAULT_RECORD_COUNTDOWN:g}")
    web.add_argument("--rclone-remote", default=os.environ.get("EGO_RCLONE_REMOTE", ""),
                     help="rclone 远端目标，如 myremote:bucket/ego；也可用 EGO_RCLONE_REMOTE")
    web.add_argument("--rclone-bin", default=os.environ.get("EGO_RCLONE_BIN", "rclone"),
                     help="rclone 可执行文件名或完整路径；默认 rclone")
    web.add_argument("--rclone-config", default=os.environ.get("EGO_RCLONE_CONFIG") or None,
                     help="可选的 rclone.conf 路径；也可用 EGO_RCLONE_CONFIG")
    web.add_argument("--storage-limit-gb", type=float, default=DEFAULT_STORAGE_LIMIT_GB,
                     help=f"数据目录自动上传清理上限(GB)，0 表示禁用；默认 {DEFAULT_STORAGE_LIMIT_GB:g}")
    web.add_argument("--storage-check-interval", type=float,
                     default=DEFAULT_STORAGE_CHECK_INTERVAL_MINUTES,
                     help=("数据目录容量检查间隔分钟数；默认 "
                           f"{DEFAULT_STORAGE_CHECK_INTERVAL_MINUTES:g}"))
    web.add_argument("--web-port", type=int, default=DEFAULT_WEB_PORT,
                     help=f"本地网页端口，默认 {DEFAULT_WEB_PORT}，占用则顺延")
    web.add_argument("--host", default="127.0.0.1",
                     help="网页监听地址；用 0.0.0.0 允许局域网访问")
    web.add_argument("--page", default=DEFAULT_PAGE,
                     help="页面 HTML 文件路径")
    web.add_argument("--jpeg-quality", type=int, default=80,
                     help="相机直播 JPEG 质量 1~100，默认 80")
    web.add_argument("--stream-fps", type=int, default=30,
                     help="相机直播目标帧率，默认 30")
    preview_width = WUJI_PREVIEW_WIDTH if glove_type == "wuji" else 0
    preview_height = WUJI_PREVIEW_HEIGHT if glove_type == "wuji" else 0
    web.add_argument(
        "--preview-width", type=int, default=preview_width,
        help=("相机监看流最大宽度；0 表示保持采集分辨率。"
              f"当前模式默认 {preview_width}"))
    web.add_argument(
        "--preview-height", type=int, default=preview_height,
        help=("相机监看流最大高度；0 表示保持采集分辨率。"
              f"当前模式默认 {preview_height}"))
    web.add_argument("--no-open", action="store_true",
                     help="不自动打开浏览器")
    web.add_argument("--no-hand-export", action="store_true",
                     help="关闭手套 export_arrays() 兜底取数")

    args = parser.parse_args(argv)
    args.glove_type = glove_type
    return args
