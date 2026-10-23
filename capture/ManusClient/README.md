# MANUS UDP sender

**English** | [中文](README_zh.md)

The MANUS rig needs a Windows program that reads the gloves through the MANUS SDK and streams every frame to the recorder over UDP. The binaries are **not** in this repository (MANUS SDK licence); download them and put them in this folder.

## 1. Download

1. Install **MANUS Core** from the MANUS website (account required) and pair the gloves.
2. Download the Windows **MANUS SDK** from the same site. It ships `ManusSDK.dll` and the `SDKMinimalClient` sample.
3. Put the UDP sender `SDKMinimalClient_Windows.exe` and `ManusSDK.dll` in this folder. The sender is `SDKMinimalClient` extended to send the packets described below; the build we use is available from the project maintainers.

```text
capture/ManusClient/
├── SDKMinimalClient_Windows.exe
└── ManusSDK.dll
```

## 2. Run

Start the sender and point it at the recorder's IP and port (default 9000):

```bash
python record_ego_manus.py --task-setting task_setting/example.json --manus-ip 0.0.0.0 --manus-port 9000
```

## 3. Packet format

One UTF-8 JSON object per UDP datagram (NaN / Inf are tolerated and read as 0). Any sender that produces this format works with `devices/manus_glove.py`.

```json
{
  "frame": 12345,
  "ergonomics": {
    "left":  {"glove_id": 1234, "values": [20 floats: 5 fingers x 4 joint angles, thumb..pinky]},
    "right": {"glove_id": 5678, "values": [20 floats]}
  },
  "skeletons": [
    {"glove_id": 1234,
     "nodes": [{"pos": [x, y, z], "rot": [qx, qy, qz, qw], "parent": 0}, "... 25 nodes"]},
    {"glove_id": 5678, "nodes": ["..."]}
  ]
}
```

| Field | Meaning |
| :--- | :--- |
| `frame` | Counter incremented once per packet; gaps are counted as dropped packets |
| `glove_id` | Identifies the physical glove; the `ergonomics.left/right` entries tell the recorder which ID is which hand |
| `nodes` | MANUS raw skeleton, 25 nodes (wrist + thumb 4 + four fingers × 5). Node 0 is the wrist, `rot` is in **x, y, z, w** order, `parent` is a node index (the root may name itself) |

- The recorder drops the four non-thumb metacarpals (raw indices 5 / 10 / 15 / 20) and stores 21 nodes in MediaPipe order.
- There is no device clock; packets are stamped with the receiving host's wall clock.
