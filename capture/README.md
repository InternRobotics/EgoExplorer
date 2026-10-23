# Capture

**English** | [中文](README_zh.md)

Synchronised egocentric capture: a head camera, five VIVE trackers and a pair of gloves recorded together on one host clock. Two independent glove rigs are supported:

| Rig | Glove data | Machines | Entry point |
| :--- | :--- | :--- | :--- |
| **Wuji** | 21-node skeleton + touch | Linux recorder + Windows tracker sender | `record_ego_wuji.py` (Linux), `tracker_sender_windows.py` (Windows) |
| **MANUS** | 21-node skeleton + joint angles | One Windows machine | `record_ego_manus.py` |

Both rigs store the hand skeleton as 21 nodes (MediaPipe topology), so the processing pipeline treats them the same way.

## Contents

- [How it works](#how-it-works)
- [Files](#files)
- [Wuji rig](#wuji-rig)
- [MANUS rig](#manus-rig)
- [Recording](#recording)
- [Session output and timestamps](#session-output-and-timestamps)
- [Uploading data](#uploading-data)
- [Replaying a session](#replaying-a-session)
- [Troubleshooting](#troubleshooting)

## How it works

In the Wuji rig, Linux is the only capture coordinator:

```text
Wuji gloves ─UDP/Zenoh─> Linux Wuji SDK ──────────────> gloves.npz
HDMI capture card ─────> Linux GoProHDMICamera ───────> MP4 + video.npz
HTC Vive Tracker ──────> Windows SteamVR/OpenXR ─TCP─> Linux NetworkTracker
                                                        ├──> tracker.npz
                                                        └──> session.json
```

**Clock sync.** Windows sends its wall-clock time with every pose. Every 2 s the two machines exchange an NTP-style four-timestamp sample; Linux estimates the `Windows − Linux` offset from the lowest-RTT sample and maps tracker times onto the Linux wall clock, the same clock as the video and the Wuji `t_host_us`. Until the offset is known, the Linux receive time is used; it is also the fallback when the estimated offset jumps by more than 5 s.

The MANUS rig runs on one Windows machine: the MANUS sender pushes hand data over UDP, and the trackers are read locally through OpenXR.

## Files

```text
capture/
├── record_ego_wuji.py               # Wuji rig entry point (Linux)
├── record_ego_manus.py              # MANUS rig entry point (Windows)
├── record_ego_app.py                # orchestrates devices, web server and recording tasks
├── record_ego_page.html             # live monitoring and control page
├── tracker_sender_windows.py        # Windows OpenXR tracker sender
├── visualize_session.py             # session replay
├── record_ego/                      # config, device lifecycle, recording, upload, web server
├── devices/
│   ├── wuji_glove.py                # Wuji SDK driver
│   ├── manus_glove.py               # MANUS UDP receiver and packing
│   ├── network_tracker.py           # Linux-side tracker TCP receiver
│   ├── vive_tracker.py              # Windows OpenXR driver
│   └── gopro_camera.py              # capture-card driver
├── network_bridge/protocol.py       # TCP/MessagePack framing
├── ManusClient/                     # MANUS UDP sender (download separately, see ManusClient/README.md)
├── wuji_calibration/                # import/export of Wuji calibration bundles (bundles are not committed)
├── scripts/
│   ├── setup_wuji_gloves.sh         # Wuji network setup and checks on Linux
│   ├── wuji_calibration_tool.py     # Wuji glove calibration, bundle export/import
│   ├── wuji_launch.sh               # one-command start for the Wuji rig
│   ├── start_tracker_sender.ps1     # Windows tracker sender launcher
│   ├── rclone_linux.sh              # bulk upload of data/ by day (Linux)
│   └── rclone_win.bat               # bulk upload of data/ by day (Windows)
├── task_setting/example.json        # example task list: a JSON array of strings
├── tests/                           # unit tests
├── requirements.txt                 # recorder dependencies
└── requirements-tracker-windows.txt # Windows sender dependencies
```

All commands below are run from `capture/`.

## Wuji rig

### 1. Install the Linux recorder

Use Ubuntu 22.04 or later (as supported by the Wuji SDK) and Python 3.10/3.11.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 2. Set up the glove network

With the gloves, cables and USB Ethernet adapters connected, run the setup script. It skips the default-route and other in-use interfaces, probes each physical port for the left glove at `192.168.1.100` and the right glove at `192.168.1.101`, creates persistent NetworkManager profiles and `/32` glove routes, and finally scans, connects and disconnects through the Wuji SDK to verify.

```bash
./scripts/setup_wuji_gloves.sh                                             # configure automatically
./scripts/setup_wuji_gloves.sh --check                                     # check only, do not touch NetworkManager
./scripts/setup_wuji_gloves.sh --interface enx001 --interface enx002       # name the USB adapters when auto-detection is ambiguous
./scripts/setup_wuji_gloves.sh --expected-gloves 2 --allow-ufw             # require both gloves; open the SDK's LAN UDP ports if UFW is on
```

- The script never runs the `ip addr flush` from the official example, so it cannot wipe addresses on your other interfaces; automatic mode also leaves alone any interface that carries the default route or another IPv4 subnet.
- If the glove IPs were changed, pass the actual subnet with `--left-ip`, `--right-ip` and `--host-prefix`.
- See `./scripts/setup_wuji_gloves.sh --help` for all options.

### 3. Create a glove calibration bundle

A bundle is everything the Wuji SDK holds for one SDK user — hand models, touch models and calibration records — so that every capture machine uses the calibration of the gloves attached to it. Close Wuji Studio first.

```bash
python scripts/wuji_calibration_tool.py users                       # list SDK users (* = current)
python scripts/wuji_calibration_tool.py create-user "rig-A"         # create a user for this pair of gloves
python scripts/wuji_calibration_tool.py calibrate --user <user_id>  # calibrate each glove, following the SDK prompts
python scripts/wuji_calibration_tool.py export --user <user_id>     # export to wuji_calibration/calibration_bundle.zip
python scripts/wuji_calibration_tool.py preview                     # inspect the bundle without importing it
```

You can also calibrate in Wuji Studio and then `export` the same user. To set up another capture machine, copy the bundle into its `wuji_calibration/`.

On start-up the recorder imports `wuji_calibration/calibration_bundle.zip` and switches to the bundle's user. Use `--wuji-calibration` for a different path, or `--no-wuji-calibration` to skip the import and keep the SDK's current user.

### 4. Start the Linux recorder

```bash
python record_ego_wuji.py \
  --task-setting task_setting/example.json \
  --tracker-host 0.0.0.0 \
  --tracker-port 8766 \
  --tracker-token "<shared secret>"
```

- `--tracker-host 0.0.0.0` listens on every interface; find the Linux LAN IP with `hostname -I`.
- To keep the secret out of your shell history, set it as an environment variable instead: `export EGO_TRACKER_TOKEN="<shared secret>"`.
- The control page is at `http://127.0.0.1:8770/`. To open it from another computer, add `--host 0.0.0.0` and allow the web port through the firewall.
- If UFW is enabled, allow the Windows machine to reach TCP 8766:

  ```bash
  sudo ufw allow from <WINDOWS_IP> to any port 8766 proto tcp
  ```

> [!WARNING]
> Never expose port 8766 to the internet. Even on a trusted LAN, set a long `--tracker-token`.

### 5. Start the Windows tracker sender

Windows needs SteamVR set as the active OpenXR runtime, with a supported role assigned to every tracker. Install a separate environment:

```powershell
python -m venv .venv-tracker
.\.venv-tracker\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-tracker-windows.txt
```

Start the Linux recorder first, then run either of these on Windows:

```powershell
.\scripts\start_tracker_sender.ps1 -LinuxHost 192.168.1.20 -Port 8766 -Token "<shared secret>"
python tracker_sender_windows.py --host 192.168.1.20 --port 8766 --token "<shared secret>"
```

- Host, port and token must match on both sides.
- If the connection drops, Windows reconnects every 2 s; the Linux recorder keeps running and keeps recording video and gloves.
- Windows Defender usually only needs to allow Python's outbound LAN connections; no inbound port is opened on Windows.

**Send rate.** Trackers are sent at up to 90 FPS by default (`-SendFps` / `--send-fps`). The cap matters: on some headless OpenXR runtimes `wait_frame` returns immediately, and without a limit the sender produces nearly a thousand duplicate poses per second, starving the Linux event loop that receives glove data and control events. Linux additionally caps the tracker buffer at 120 FPS by default (`--tracker-record-fps`) to stay compatible with older senders.

**Recording from the Windows keyboard.** The sender listens to the Windows terminal and controls the Linux recorder over the same authenticated TCP connection:

| Key | Action |
| :--- | :--- |
| `r` | Start recording (countdown from the Linux `--record-countdown`) |
| `e` | Stop and save |
| `Esc` | Stop listening to the keyboard; poses keep streaming |

Override the countdown from Windows with `-Countdown 3` / `--countdown 3`, or disable keyboard control with `-NoKeyboard` / `--no-keyboard`. Setting the same non-empty token on both sides stops other devices on the LAN from sending control commands.

## MANUS rig

1. Download the MANUS UDP sender (`SDKMinimalClient_Windows.exe` and `ManusSDK.dll`) into `ManusClient/`. They are not distributed with this repository; see [`ManusClient/README.md`](ManusClient/README.md) for how to get them and for the packet format.
2. Point the sender at the recorder's IP and port, then start the recorder:

   ```bash
   python record_ego_manus.py \
     --task-setting task_setting/example.json \
     --manus-ip 0.0.0.0 \
     --manus-port 9000
   ```

- The recorder receives MANUS's 25-node skeleton, drops the four non-thumb metacarpals and stores 21 nodes (the same layout as Wuji / MediaPipe), together with the joint angles. No touch channels are produced.
- Trackers are driven locally through `pyopenxr`; pass `--no-tracker` to record without them — camera and gloves are unaffected.
- See `python record_ego_manus.py --help` for all options.

**Tracker sampling.** Local trackers are sampled on the host clock (`--tracker-sample-fps`, default 90; 0 = unlimited). Headless SteamVR may only produce compositor frames at ~15 Hz, but the trackers themselves report at 90+ Hz and `xrLocateSpace` can be queried at any time, so sampling is decoupled from the compositor. A full wait/begin/end frame runs only every 0.5 s to keep the session alive, and `xrSyncActions` is called only inside those keep-alive frames: outside a frame SteamVR blocks it until the next compositor frame (~66 ms measured), which would drag sampling back down to ~15 Hz.

## Recording

**Before you start**

1. Wuji: close Wuji Studio and check that skeleton and touch data look right. MANUS: check that the UDP sender is running.
2. Start SteamVR and the Windows sender, and check that the top of the page shows the tracker as connected.
3. Check the camera preview and the 3D tracker poses, then start recording.

The Wuji page shows the camera preview by default, limited to 640×360 (the recording keeps its full resolution); it can be hidden from the top-right corner.

**Control-page keys**

| Key | Action |
| :--- | :--- |
| `r` | Start a new session after the countdown |
| `e` | Stop and save the current session |
| `d` | Abort and discard the current session |
| `q` | Quit; any unsaved recording is discarded |

**Common options**

| Option | Default | Description |
| :--- | :--- | :--- |
| `--task-setting` | required | JSON array of task names |
| `--save-dir` | `./data` | Root folder for sessions |
| `--record-countdown` | `3` | Countdown before recording, in seconds |
| `--host` | `127.0.0.1` | Address the control page listens on |
| `--cam-device` | `0` | Capture-card index or `/dev/video*` path |
| `--cam-width` / `--cam-height` | `2560` / `1440` | Requested resolution |
| `--cam-fps` | `60` | Requested capture frame rate |
| `--video-codec` | `h264` | `h264` or `mp4v` |
| `--h264-preset` | `ultrafast` | Software H.264 speed preset |
| `--h264-threads` | `4` | H.264 encoder threads; `0` = auto |
| `--no-camera` | off | Do not connect a camera |
| `--preview-width` / `--preview-height` | Wuji `640` / `360`, MANUS `0` / `0` | Maximum preview size; `0` = capture resolution; does not affect the recording |
| `--tracker-host` | `0.0.0.0` | Tracker TCP listen address (Wuji) |
| `--tracker-port` | `8766` | Tracker TCP port (Wuji) |
| `--tracker-token` | empty | Shared secret; or `EGO_TRACKER_TOKEN` |
| `--tracker-record-fps` | `120` | Tracker buffer rate cap; `0` = unlimited (Wuji) |
| `--no-tracker` | off | Record without trackers |
| `--wuji-calibration` | default bundle | Wuji calibration bundle path |
| `--no-wuji-calibration` | off | Skip the bundle and use the SDK's current user |

See `python record_ego_wuji.py --help` or `python record_ego_manus.py --help` for everything else.

## Session output and timestamps

```text
data/session_YYYYMMDD_HHMMSS_TASK/
├── session_....mp4            # head-camera video
├── session_....video.npz      # per-frame timestamps
├── session_....npz            # gloves: skeleton, joint angles / touch
├── session_....tracker.npz    # poses of the five trackers
└── session_....session.json   # session metadata
```

- Video and Wuji `t_host_us` come straight from Linux `time.time()`.
- Wuji subscribes only to the `tactile` and `hand_skeleton` streams the dataset uses, so the other ~120 Hz derived streams do not compete with video capture for CPU.
- Video capture and encoding run on separate threads with a bounded queue. MP4 timestamps (variable frame rate) come from the per-frame `t_host_us`, so gaps in the input never shorten playback.
- Tracker send times come from Windows `time.time()`. After correction they are written to `tracker_0__pose__t_host_us`; the raw Windows wall clock is kept in `tracker_0__pose__t_source_host_us` for sync diagnostics.
- Trackers have no hardware clock of their own, so `t_dev_us` is `-1` and `t_us` falls back to the corrected `t_host_us`.
- Dropped tracker packets and the transport used are recorded in the session's tracker metadata.

## Uploading data

**Bulk upload.** `scripts/rclone_linux.sh` and `scripts/rclone_win.bat` are thin wrappers around `rclone copy --progress`. Edit `LOCAL_PATH` (local folder) and `REMOTE_PATH` (remote bucket path) at the top of the script, then run it (`bash scripts/rclone_linux.sh`, or double-click `scripts\rclone_win.bat`). To filter files (`--include`) or delete local copies after upload (`move`), edit the single rclone command in the script.

**Upload from the recorder.** The recorder can also upload finished sessions in the background with rclone:

| Option | Description |
| :--- | :--- |
| `--rclone-remote` | Remote target, e.g. `myremote:bucket/ego`; or `EGO_RCLONE_REMOTE` |
| `--rclone-bin` | rclone executable, default `rclone`; or `EGO_RCLONE_BIN` |
| `--rclone-config` | Optional `rclone.conf` path; or `EGO_RCLONE_CONFIG` |
| `--storage-limit-gb` | Upload and clean up automatically once the data folder exceeds this size; default 50, `0` disables |
| `--storage-check-interval` | Minutes between size checks; default 30 |

## Replaying a session

```bash
python visualize_session.py data/session_YYYYMMDD_HHMMSS_TASK
```

The glove type is detected from `session.json`, the glove NPZ metadata and the node count: current MANUS and Wuji data both use 21 nodes, and older MANUS data is recognised by its 25 nodes. Both are aligned to video and trackers by `t_host_us`. The page title shows the detected type, node count and the streams actually recorded.

Options: `--no-open`, `--port 8000`, `--max-fps 30`, `--build-only`. Demo sessions can be generated for either glove:

```bash
python visualize_session.py /tmp/manus_demo --demo --demo-type manus
python visualize_session.py /tmp/wuji_demo --demo --demo-type wuji
```

## Troubleshooting

**Linux keeps waiting for the tracker**

- The `--host` on Windows must be the Linux LAN IP, not `127.0.0.1`.
- Port and token must match exactly on both sides.
- Test from Windows: `Test-NetConnection <LINUX_IP> -Port 8766`.
- Check the Linux firewall, and that both machines are on a LAN / VLAN that can reach each other.
- Make sure the Linux receiver was started first.

**No tracker poses on Windows**

- Check that `requirements-tracker-windows.txt` is installed.
- SteamVR must be the active OpenXR runtime, with every tracker paired and assigned a role.
- The "Handheld object" role produces no usable pose; use waist, wrist, ankle or similar.
- The sender prints every serial and role it finds, which helps debug the SteamVR setup.

**Linux cannot find the Wuji gloves**

- Check that `wuji-sdk` is installed on native Linux and that no other program holds the SDK.
- Run `./scripts/setup_wuji_gloves.sh --check` to test interfaces, routes, ping and the SDK scan.
- Check the left/right serial numbers and the current SDK user's calibration data.
