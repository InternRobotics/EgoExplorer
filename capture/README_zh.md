# 采集程序

[English](README.md) | **中文**

第一视角数据的同步采集程序：头部相机、五个 VIVE tracker 和一副手套同时录制，统一到同一台主机的时钟上。支持两套互不干扰的手套方案：

| 方案 | 手套数据 | 机器 | 入口 |
| :--- | :--- | :--- | :--- |
| **Wuji** | 21 节点骨架 + 触觉 | Linux 采集端 + Windows tracker 发送端 | `record_ego_wuji.py`（Linux）、`tracker_sender_windows.py`（Windows） |
| **MANUS** | 21 节点骨架 + 关节角 | 一台 Windows | `record_ego_manus.py` |

两种方案的手部骨架都以 21 节点（MediaPipe 拓扑）落盘，下游处理不区分手套类型。

## 目录

- [工作原理](#工作原理)
- [文件结构](#文件结构)
- [Wuji 方案](#wuji-方案)
- [MANUS 方案](#manus-方案)
- [录制](#录制)
- [会话输出与时间戳](#会话输出与时间戳)
- [上传数据](#上传数据)
- [回看会话](#回看会话)
- [故障排查](#故障排查)

## 工作原理

Wuji 方案中 Linux 是唯一的采集协调端：

```text
Wuji 手套 ─UDP/Zenoh─> Linux Wuji SDK ────────────────> gloves.npz
HDMI 采集卡 ─────────> Linux GoProHDMICamera ─────────> MP4 + video.npz
HTC Vive Tracker ────> Windows SteamVR/OpenXR ─TCP───> Linux NetworkTracker
                                                        ├──> tracker.npz
                                                        └──> session.json
```

**时钟同步。** Windows 随每帧位姿发送自己的墙钟时间。两端每 2 秒做一次 NTP 式四时间戳采样，Linux 用 RTT 最低的样本估计 `Windows − Linux` 时钟偏移，把 tracker 时间换算到 Linux 墙钟，与视频和 Wuji 的 `t_host_us` 对齐。校时完成前使用 Linux 收包时间；检测到超过 5 秒的异常偏移时也会回退到收包时间。

MANUS 方案在一台 Windows 上完成：MANUS 发送端通过 UDP 推送手部数据，tracker 由本机 OpenXR 直接读取。

## 文件结构

```text
capture/
├── record_ego_wuji.py               # Wuji 方案主入口（Linux）
├── record_ego_manus.py              # MANUS 方案主入口（Windows）
├── record_ego_app.py                # 设备、网页服务和录制任务编排
├── record_ego_page.html             # 实时监看与控制页面
├── tracker_sender_windows.py        # Windows OpenXR tracker 发送端
├── visualize_session.py             # 会话回看
├── record_ego/                      # 配置、设备生命周期、录制、上传、网页服务
├── devices/
│   ├── wuji_glove.py                # Wuji SDK 驱动
│   ├── manus_glove.py               # MANUS UDP 接收与打包
│   ├── network_tracker.py           # Linux 端 tracker TCP 接收
│   ├── vive_tracker.py              # Windows OpenXR 驱动
│   └── gopro_camera.py              # 采集卡驱动
├── network_bridge/protocol.py       # TCP/MessagePack 分帧协议
├── ManusClient/                     # MANUS UDP 发送端（需自行下载，见 ManusClient/README_zh.md）
├── wuji_calibration/                # Wuji 标定 bundle 的导入/导出（bundle 本身不入库）
├── scripts/
│   ├── setup_wuji_gloves.sh         # Linux 上 Wuji 网卡配置与检查
│   ├── wuji_calibration_tool.py     # Wuji 手套标定、bundle 导出/导入
│   ├── wuji_launch.sh               # Wuji 方案一键启动
│   ├── start_tracker_sender.ps1     # Windows tracker 发送端启动脚本
│   ├── rclone_linux.sh              # 按天批量上传 data/（Linux）
│   └── rclone_win.bat               # 按天批量上传 data/（Windows）
├── task_setting/example.json        # 任务列表示例：JSON 字符串数组
├── tests/                           # 单元测试
├── requirements.txt                 # 采集端依赖
└── requirements-tracker-windows.txt # Windows 发送端依赖
```

以下命令都在 `capture/` 目录下运行。

## Wuji 方案

### 1. 安装 Linux 采集端

建议使用 Wuji SDK 支持的 Ubuntu 22.04 或更新版本，Python 3.10/3.11。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 2. 配置手套网络

连接手套、网线和 USB 网卡后运行自动配置脚本。脚本会排除默认路由和业务网卡，逐个物理网口探测左手 `192.168.1.100` 与右手 `192.168.1.101`，创建持久化的 NetworkManager 配置和 `/32` 手套路由，最后通过 Wuji SDK 做扫描、连接、断开验证。

```bash
./scripts/setup_wuji_gloves.sh                                             # 自动配置
./scripts/setup_wuji_gloves.sh --check                                     # 只检查，不修改 NetworkManager
./scripts/setup_wuji_gloves.sh --interface enx001 --interface enx002       # 自动候选不明确时显式指定 USB 网卡
./scripts/setup_wuji_gloves.sh --expected-gloves 2 --allow-ufw             # 要求两只手都在线；UFW 启用时放行 SDK 的局域网 UDP 端口
```

- 脚本不会执行官方示例中的 `ip addr flush`，不会误删业务网卡地址；自动模式也不会修改承载默认路由或已有其他网段 IPv4 的接口。
- 手套 IP 改过时，用 `--left-ip`、`--right-ip` 和 `--host-prefix` 指定实际网段。
- 完整参数见 `./scripts/setup_wuji_gloves.sh --help`。

### 3. 生成手套标定 bundle

bundle 是 Wuji SDK 导出的某个 SDK 用户的全部手部模型、触觉模型和标定记录，让每台采集机使用与所连手套一致的标定。运行前关闭 Wuji Studio。

```bash
python scripts/wuji_calibration_tool.py users                       # 列出 SDK 用户（* 为当前用户）
python scripts/wuji_calibration_tool.py create-user "rig-A"         # 为这套手套新建用户
python scripts/wuji_calibration_tool.py calibrate --user <user_id>  # 逐只手套标定，按 SDK 提示摆姿势
python scripts/wuji_calibration_tool.py export --user <user_id>     # 导出到 wuji_calibration/calibration_bundle.zip
python scripts/wuji_calibration_tool.py preview                     # 校验 bundle（不导入）
```

也可以在 Wuji Studio 里完成标定，再用 `export` 导出同一个用户。把 bundle 拷到其它采集机的 `wuji_calibration/` 下即可。

采集程序启动时会自动导入 `wuji_calibration/calibration_bundle.zip` 并切换到 bundle 所属的用户；`--wuji-calibration` 可指定其它路径，`--no-wuji-calibration` 跳过导入、使用 SDK 当前用户。

### 4. 启动 Linux 采集端

```bash
python record_ego_wuji.py \
  --task-setting task_setting/example.json \
  --tracker-host 0.0.0.0 \
  --tracker-port 8766 \
  --tracker-token "<共享密钥>"
```

- `--tracker-host 0.0.0.0` 监听所有网卡；Linux 的局域网 IP 用 `hostname -I` 查看。
- 密钥也可以用环境变量传入，避免留在命令历史里：`export EGO_TRACKER_TOKEN="<共享密钥>"`。
- 控制页默认是 `http://127.0.0.1:8770/`。要从其他电脑打开，加 `--host 0.0.0.0` 并在防火墙里放行网页端口。
- 启用了 UFW 时，放行 Windows 机器访问 TCP 8766：

  ```bash
  sudo ufw allow from <WINDOWS_IP> to any port 8766 proto tcp
  ```

> [!WARNING]
> 不要把 8766 暴露到公网。即使在可信局域网里，也建议设置一段足够长的 `--tracker-token`。

### 5. 启动 Windows tracker 发送端

Windows 需要安装 SteamVR，并把 SteamVR 设为当前 OpenXR Runtime；给每个 tracker 分配受支持的 role。安装独立环境：

```powershell
python -m venv .venv-tracker
.\.venv-tracker\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-tracker-windows.txt
```

先启动 Linux 采集端，再在 Windows 上运行（两种方式等价）：

```powershell
.\scripts\start_tracker_sender.ps1 -LinuxHost 192.168.1.20 -Port 8766 -Token "<共享密钥>"
python tracker_sender_windows.py --host 192.168.1.20 --port 8766 --token "<共享密钥>"
```

- 两端的 host、port、token 必须一致。
- 连接中断后 Windows 每 2 秒自动重连；Linux 采集不会退出，视频和手套照常录制。
- Windows Defender 防火墙通常只需允许 Python 访问局域网出站连接，不需要开入站端口。

**发送帧率。** tracker 默认最多以 90 FPS 发送（`-SendFps` / `--send-fps`）。这个上限是必要的：部分 headless OpenXR Runtime 的 `wait_frame` 会立即返回，不限速会产生近千 FPS 的重复位姿，持续抢占 Linux 上手套接收与控制的事件循环。Linux 端另外默认以 120 FPS 为上限写入 tracker 缓存（`--tracker-record-fps`），兼容旧版发送端。

**用 Windows 键盘控制录制。** 发送端会监听 Windows 终端的键盘，复用同一条已认证的 TCP 连接控制 Linux 采集端：

| 按键 | 操作 |
| :--- | :--- |
| `r` | 开始录制（倒计时默认用 Linux 端的 `--record-countdown`） |
| `e` | 结束并保存 |
| `Esc` | 关闭键盘监听；tracker 位姿继续发送 |

`-Countdown 3` / `--countdown 3` 可以从 Windows 覆盖倒计时；不需要键盘控制时加 `-NoKeyboard` / `--no-keyboard`。两端设置相同且非空的 token，可以防止局域网里的其他设备发送控制指令。

## MANUS 方案

1. 下载 MANUS UDP 发送端（`SDKMinimalClient_Windows.exe` 和 `ManusSDK.dll`），放到 `ManusClient/`。这两个文件不随仓库分发，下载方式和数据包格式见 [`ManusClient/README_zh.md`](ManusClient/README_zh.md)。
2. 让发送端的目标 IP 和端口与采集端一致，然后启动采集程序：

   ```bash
   python record_ego_manus.py \
     --task-setting task_setting/example.json \
     --manus-ip 0.0.0.0 \
     --manus-port 9000
   ```

- 采集端接收 MANUS 的 25 节点骨骼，丢掉 4 个非拇指掌骨节点后以 21 节点（与 Wuji / MediaPipe 一致）落盘，同时保存关节角；不生成触觉通道。
- tracker 由本机 `pyopenxr` 驱动；没有 tracker 时加 `--no-tracker`，不影响相机和手套。
- 完整参数见 `python record_ego_manus.py --help`。

**tracker 采样。** 本机 tracker 按主机时钟采样（`--tracker-sample-fps`，默认 90，0 为不限）。headless SteamVR 的合成器可能只以 ~15 Hz 出帧，但 tracker 本身以 90+ Hz 上报，`xrLocateSpace` 可以随时查询，所以采样不受合成器节拍限制；完整的 wait/begin/end 帧每 0.5 s 才执行一次，只用来维持会话。`xrSyncActions` 也只在这些保活帧里调用：在帧外调用时，SteamVR 会阻塞到下一个合成器帧（实测约 66 ms），每次采样都同步会把采样率拖回 ~15 Hz。

## 录制

**开始前检查**

1. Wuji：关闭 Wuji Studio，确认骨骼和触觉数据正常。MANUS：确认 UDP 发送端在运行。
2. 启动 SteamVR 和 Windows 发送端，确认页面顶部显示“Tracker 已连接”。
3. 检查相机预览和 tracker 3D 位姿，再开始录制。

Wuji 页面每次启动默认显示相机监看画面，监看流限制在 640×360，不影响原始录制分辨率；可在页面右上角临时隐藏。

**控制页按键**

| 按键 | 操作 |
| :--- | :--- |
| `r` | 倒计时后开始新会话 |
| `e` | 结束并保存当前会话 |
| `d` | 终止并丢弃当前会话 |
| `q` | 退出；未保存的录制会被丢弃 |

**常用参数**

| 参数 | 默认值 | 说明 |
| :--- | :--- | :--- |
| `--task-setting` | 必填 | JSON 任务名称数组 |
| `--save-dir` | `./data` | 会话保存根目录 |
| `--record-countdown` | `3` | 录制倒计时（秒） |
| `--host` | `127.0.0.1` | 控制网页监听地址 |
| `--cam-device` | `0` | 采集卡索引或 `/dev/video*` 路径 |
| `--cam-width` / `--cam-height` | `2560` / `1440` | 请求分辨率 |
| `--cam-fps` | `60` | 请求采集帧率 |
| `--video-codec` | `h264` | `h264` 或 `mp4v` |
| `--h264-preset` | `ultrafast` | 软件 H.264 实时编码速度档位 |
| `--h264-threads` | `4` | H.264 编码线程数，`0` 为自动 |
| `--no-camera` | 关 | 不连接相机 |
| `--preview-width` / `--preview-height` | Wuji `640` / `360`，MANUS `0` / `0` | 网页监看流最大尺寸，`0` 为采集分辨率；不影响录制 |
| `--tracker-host` | `0.0.0.0` | tracker TCP 监听地址（Wuji） |
| `--tracker-port` | `8766` | tracker TCP 端口（Wuji） |
| `--tracker-token` | 空 | 共享密钥，也可用 `EGO_TRACKER_TOKEN` |
| `--tracker-record-fps` | `120` | tracker 录制缓存帧率上限，`0` 为不限（Wuji） |
| `--no-tracker` | 关 | 不使用 tracker |
| `--wuji-calibration` | 默认 bundle | Wuji 标定 bundle 路径 |
| `--no-wuji-calibration` | 关 | 不导入 bundle，使用 SDK 当前用户 |

完整参数见 `python record_ego_wuji.py --help` 或 `python record_ego_manus.py --help`。

## 会话输出与时间戳

```text
data/session_YYYYMMDD_HHMMSS_TASK/
├── session_....mp4            # 头部相机视频
├── session_....video.npz      # 逐帧时间戳
├── session_....npz            # 手套：骨架、关节角 / 触觉
├── session_....tracker.npz    # 五个 tracker 的位姿
└── session_....session.json   # 会话元信息
```

- 视频和 Wuji 的 `t_host_us` 直接取自 Linux `time.time()`。
- Wuji 默认只订阅数据集用到的 `tactile` 和 `hand_skeleton`，避免其他约 120 Hz 的派生流和视频抓帧争用 CPU。
- 视频抓帧与编码使用独立线程和有界队列；MP4 的可变帧率 PTS 来自逐帧 `t_host_us`，即使输入有帧间空洞，播放时长也不会被压缩。
- tracker 的原始发送时间来自 Windows `time.time()`；接收端校正后写入 `tracker_0__pose__t_host_us`，Windows 原始墙钟保存在 `tracker_0__pose__t_source_host_us`，供同步诊断。
- tracker 没有独立硬件时钟，因此 `t_dev_us` 为 `-1`，`t_us` 取校正后的 `t_host_us`。
- tracker 的传输丢帧数和传输方式写在会话的 tracker 元信息里。

## 上传数据

**批量上传。** `scripts/rclone_linux.sh` 和 `scripts/rclone_win.bat` 是 `rclone copy --progress` 的简单封装。打开脚本，把开头的 `LOCAL_PATH`（本地目录）和 `REMOTE_PATH`（远端桶路径）改成自己的，然后直接运行（`bash scripts/rclone_linux.sh`，或双击 `scripts\rclone_win.bat`）。要加过滤（`--include`）或上传后删除本地（改用 `move`），修改脚本里那条 rclone 命令即可。

**采集程序内上传。** 采集程序也可以在后台用 rclone 上传已完成的会话：

| 参数 | 说明 |
| :--- | :--- |
| `--rclone-remote` | 远端目标，如 `myremote:bucket/ego`；也可用 `EGO_RCLONE_REMOTE` |
| `--rclone-bin` | rclone 可执行文件，默认 `rclone`；也可用 `EGO_RCLONE_BIN` |
| `--rclone-config` | 可选的 `rclone.conf` 路径；也可用 `EGO_RCLONE_CONFIG` |
| `--storage-limit-gb` | 数据目录超过此容量时自动上传并清理，默认 50，`0` 为禁用 |
| `--storage-check-interval` | 容量检查间隔（分钟），默认 30 |

## 回看会话

```bash
python visualize_session.py data/session_YYYYMMDD_HHMMSS_TASK
```

脚本根据 `session.json`、手套 NPZ 元数据和节点数自动识别手套类型：现行 MANUS 与 Wuji 数据都是 21 节点，旧版 MANUS 数据按 25 节点识别；两者都优先用 `t_host_us` 与视频和 tracker 对齐。页面标题显示识别出的类型、节点数和实际落盘的数据流。

可用参数：`--no-open`、`--port 8000`、`--max-fps 30`、`--build-only`。也可以生成两种手套的演示会话：

```bash
python visualize_session.py /tmp/manus_demo --demo --demo-type manus
python visualize_session.py /tmp/wuji_demo --demo --demo-type wuji
```

## 故障排查

**Linux 一直显示“等待 Tracker”**

- Windows 命令里的 `--host` 应是 Linux 的局域网 IP，不是 `127.0.0.1`。
- 两端的端口和 token 必须完全一致。
- 在 Windows 上测试：`Test-NetConnection <LINUX_IP> -Port 8766`。
- 检查 Linux 防火墙，以及两台机器是否在可互通的局域网 / VLAN。
- 确认先启动了 Linux 接收端。

**Windows 读不到 tracker 位姿**

- 确认装了 `requirements-tracker-windows.txt`。
- SteamVR 必须是当前 OpenXR Runtime，tracker 已配对并分配 role。
- “Handheld object” role 不产生可用位姿，请改成 waist、wrist、ankle 等。
- 发送端会打印枚举到的 serial 和 role，可以据此排查 SteamVR 配置。

**Linux 找不到 Wuji 手套**

- 确认原生 Linux 上装了 `wuji-sdk`，并关闭其他占用 SDK 的程序。
- 用 `./scripts/setup_wuji_gloves.sh --check` 检查网卡、路由、ping 和 SDK 扫描。
- 检查左右手序列号，以及当前 SDK 用户的标定数据。
