# 手套骨架 → MANO / SMPL-X

[English](glove_to_mano.md) | **中文**

说明手套的手部骨架如何转成 SMPL-X 手部姿态。代码见 `src/five_tracker_smplx/hands.py` 和 `src/five_tracker_smplx/manus_hands.py`。

## 输入格式

`hand_skeleton` 每帧形状为 `[N, 7]`：前 3 维是节点位置，后 4 维是 `x, y, z, w` 四元数。格式由节点数 `N` 决定：

| 格式 | N | 来源 | 节点 |
| :--- | :---: | :--- | :--- |
| `wuji21` | 21 | Wuji 手套，以及现行 MANUS 采集程序 | MediaPipe 手部关键点：手腕 + 每指 4 个 |
| `manus25` | 25 | 旧版 MANUS 录制 | MANUS 原始骨骼：手腕 + 拇指 4 + 其余四指各 5（多一个掌骨节点） |

```text
wuji21                                     manus25
0       wrist                              0       wrist
1..4    thumb:  cmc, mcp, ip, tip          1..4    thumb:  metacarpal, proximal, distal, tip
5..8    index:  mcp, pip, dip, tip         5..9    index:  metacarpal, proximal, intermediate, distal, tip
9..12   middle: mcp, pip, dip, tip         10..14  middle: metacarpal, proximal, intermediate, distal, tip
13..16  ring:   mcp, pip, dip, tip         15..19  ring:   metacarpal, proximal, intermediate, distal, tip
17..20  pinky:  mcp, pip, dip, tip         20..24  pinky:  metacarpal, proximal, intermediate, distal, tip
```

两种格式的手指链都挂在手腕上。如果录制里的 `hand_hierarchy` 与预期格式不符，会直接报错。

## 腕局部坐标

采集端保存的是以腕点为原点的**全局**节点变换，而不是逐父节点的局部偏移。先用节点 0（手腕）的旋转消去整只手的朝向：

```text
p_wrist_local[i] = R_wrist(xyzw)^T (p[i] - p[0])
```

这里的 `R_wrist^T` 属于手套骨架自身的约定，不是 VIVE tracker 的安装外参公式。tracker 使用采集端的右作用矩阵约定；不能因为两者都用 `xyzw` 四元数，就混用矩阵乘法顺序。

## 映射到 MANO 21 个关键点

MANO 有手腕、每根手指 3 个旋转关节和 1 个指尖，共 21 个关键点。除手腕外的 20 个点进入拟合：

| MANO / SMPL-X | `wuji21` | `manus25` |
| :--- | :--- | :--- |
| wrist | `0` | `0` |
| thumb 1, 2, 3, tip | `1, 2, 3, 4` | `1, 2, 3, 4` |
| index 1, 2, 3, tip | `5, 6, 7, 8` | `6, 7, 8, 9` |
| middle 1, 2, 3, tip | `9, 10, 11, 12` | `11, 12, 13, 14` |
| ring 1, 2, 3, tip | `13, 14, 15, 16` | `16, 17, 18, 19` |
| pinky 1, 2, 3, tip | `17, 18, 19, 20` | `21, 22, 23, 24` |

MANUS 的 4 个非拇指掌骨节点（`5, 10, 15, 20`）是掌内锚点，MANO 没有对应的独立关节。它们不参与拟合，只保留在可视化中，用来检查掌宽、掌骨方向和手套解算是否稳定。

> [!IMPORTANT]
> SMPL-X 的 45 维 `left_hand_pose` / `right_hand_pose` 是 15 个局部 axis-angle，顺序为 **index, middle, pinky, ring, thumb**，不是手套的 thumb, index, middle, ring, pinky。不能直接按顺序拷贝。

## 拟合方法

手部用 3D 关键点的可微逆运动学拟合，而不是把手套的关节角直接塞进 MANO：

1. 去掉手腕的全局旋转。
2. 从手指最伸直的若干帧估计每只手一个固定的手套 → SMPL-X 相似变换。
3. 固定受试者的 SMPL-X `betas` 和人体全局尺度。
4. 在短时间窗内优化每只手 45 维 MANO axis-angle。
5. 损失包括 20 个点的位置、骨段方向、时间平滑、姿态先验和大角度软限制。

相似变换中的尺度只把受试者的手套骨长归一到已标定的 SMPL-X 手部尺寸；它不是人体全局尺度，也不逐帧变化。

## 对单条录制运行

管线的 `fuse` 阶段会对每个 session 自动完成这一步。要单独检查一个手套文件：

```bash
PYTHONPATH=src python -m five_tracker_smplx.cli fit-manus-hands \
  --input <session>/<session>.npz \
  --subject-profile profiles/collector_a_subject_profile.json \
  --output-dir out/hand_fit
```

输出 NPZ 包含 `left_hand_pose` / `right_hand_pose` `[T, 45]`、原始腕局部点、对齐后的点、拟合得到的 MANO 关键点和逐帧误差；HTML 页面同时显示手套骨架、MANO 21 个关键点、手部网格及两者的残差。

## 参考资料

- MANUS Core Knowledge Center，*Raw Skeleton*：25 个节点的 ID、父节点、链类型和关节类型。
- MANUS SDK `ManusSDKTypes.h`：`SkeletonNode` / `ManusTransform` 与 SDK 坐标系。
- MediaPipe Hands：21 个关键点的布局。
- SMPL-X `joint_names.py`、`body_models.py`、`vertex_ids.py`：关节顺序、15 个手部关节和指尖顶点。
- ContactPose `utilities/mano_fitting.py`：从 3D 手部关键点拟合 MANO 姿态和形状的现成参考。
