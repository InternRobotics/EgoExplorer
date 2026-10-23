# MANUS UDP 发送端

[English](README.md) | **中文**

MANUS 方案需要一个 Windows 程序：通过 MANUS SDK 读取手套，把每一帧用 UDP 发给采集程序。这些二进制文件**不随仓库分发**（受 MANUS SDK 许可约束），请自行下载后放到本目录。

## 1. 下载

1. 从 MANUS 官网（需注册账号）安装 **MANUS Core** 并配对手套。
2. 在同一网站下载 Windows 版 **MANUS SDK**，其中包含 `ManusSDK.dll` 和 `SDKMinimalClient` 示例。
3. 把 UDP 发送端 `SDKMinimalClient_Windows.exe` 和 `ManusSDK.dll` 放到本目录。发送端是在 `SDKMinimalClient` 基础上加了下文的 UDP 输出；我们使用的版本可向项目维护者索取。

```text
capture/ManusClient/
├── SDKMinimalClient_Windows.exe
└── ManusSDK.dll
```

## 2. 运行

启动发送端，把发送目标设为采集程序的 IP 和端口（默认 9000）：

```bash
python record_ego_manus.py --task-setting task_setting/example.json --manus-ip 0.0.0.0 --manus-port 9000
```

## 3. 数据包格式

每个 UDP 包是一个 UTF-8 JSON 对象（NaN / Inf 按 0 读入）。任何按此格式发送的程序都能被 `devices/manus_glove.py` 接收。

```json
{
  "frame": 12345,
  "ergonomics": {
    "left":  {"glove_id": 1234, "values": [20 个浮点数：5 根手指 × 4 个关节角，拇指到小指]},
    "right": {"glove_id": 5678, "values": [20 个浮点数]}
  },
  "skeletons": [
    {"glove_id": 1234,
     "nodes": [{"pos": [x, y, z], "rot": [qx, qy, qz, qw], "parent": 0}, "... 共 25 个节点"]},
    {"glove_id": 5678, "nodes": ["..."]}
  ]
}
```

| 字段 | 含义 |
| :--- | :--- |
| `frame` | 每包加 1 的计数器，跳变计为丢包 |
| `glove_id` | 区分实体手套；`ergonomics.left/right` 用来确定哪个 ID 是哪只手 |
| `nodes` | MANUS 原始骨骼，25 个节点（手腕 + 拇指 4 + 其余四指各 5）。节点 0 为手腕，`rot` 按 **x, y, z, w** 顺序，`parent` 为父节点索引（根节点可指向自己） |

- 采集端丢掉 4 个非拇指掌骨节点（原始索引 5 / 10 / 15 / 20），以 21 节点（MediaPipe 顺序）落盘。
- 没有设备时钟，按接收主机的墙钟打时间戳。
