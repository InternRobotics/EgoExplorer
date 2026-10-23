# Glove skeleton → MANO / SMPL-X

**English** | [中文](glove_to_mano_zh.md)

How the glove hand skeletons are turned into SMPL-X hand poses. Code: `src/five_tracker_smplx/hands.py` and `src/five_tracker_smplx/manus_hands.py`.

## Input layouts

Each frame of `hand_skeleton` has shape `[N, 7]`: node position (3) followed by an `x, y, z, w` quaternion (4). The layout is chosen by the node count `N`:

| Layout | N | Recorded by | Nodes |
| :--- | :---: | :--- | :--- |
| `wuji21` | 21 | Wuji gloves, and the current MANUS recorder | MediaPipe hand landmarks: wrist + 4 per finger |
| `manus25` | 25 | Older MANUS recordings | MANUS raw skeleton: wrist + thumb 4 + four fingers × 5 (one extra metacarpal each) |

```text
wuji21                                     manus25
0       wrist                              0       wrist
1..4    thumb:  cmc, mcp, ip, tip          1..4    thumb:  metacarpal, proximal, distal, tip
5..8    index:  mcp, pip, dip, tip         5..9    index:  metacarpal, proximal, intermediate, distal, tip
9..12   middle: mcp, pip, dip, tip         10..14  middle: metacarpal, proximal, intermediate, distal, tip
13..16  ring:   mcp, pip, dip, tip         15..19  ring:   metacarpal, proximal, intermediate, distal, tip
17..20  pinky:  mcp, pip, dip, tip         20..24  pinky:  metacarpal, proximal, intermediate, distal, tip
```

In both layouts the finger chains are parented to the wrist. A recorded `hand_hierarchy` that contradicts the expected layout is rejected.

## Wrist-local frame

The recorder stores wrist-rooted **global** node transforms, not per-parent offsets. The whole-hand orientation is removed first with the wrist rotation (node 0):

```text
p_wrist_local[i] = R_wrist(xyzw)^T (p[i] - p[0])
```

`R_wrist^T` belongs to the glove skeleton's own convention. It is not the VIVE tracker mounting formula: trackers use the recorder's right-acting matrix convention, and the two must not share a multiplication order just because both use `xyzw` quaternions.

## Mapping to MANO's 21 keypoints

MANO has a wrist, three rotational joints per finger and a fingertip, 21 keypoints in all. The 20 non-wrist points enter the fit:

| MANO / SMPL-X | `wuji21` | `manus25` |
| :--- | :--- | :--- |
| wrist | `0` | `0` |
| thumb 1, 2, 3, tip | `1, 2, 3, 4` | `1, 2, 3, 4` |
| index 1, 2, 3, tip | `5, 6, 7, 8` | `6, 7, 8, 9` |
| middle 1, 2, 3, tip | `9, 10, 11, 12` | `11, 12, 13, 14` |
| ring 1, 2, 3, tip | `13, 14, 15, 16` | `16, 17, 18, 19` |
| pinky 1, 2, 3, tip | `17, 18, 19, 20` | `21, 22, 23, 24` |

The four non-thumb MANUS metacarpals (`5, 10, 15, 20`) are palm anchors with no separate MANO joint. They are left out of the fit and kept only in the viewer, as a check on palm width, metacarpal direction and glove stability.

> [!IMPORTANT]
> SMPL-X's 45-D `left_hand_pose` / `right_hand_pose` are 15 local axis-angles in the order **index, middle, pinky, ring, thumb** — not the glove order thumb, index, middle, ring, pinky. Never copy values across directly.

## Fitting

The hands are fitted by differentiable inverse kinematics on 3D keypoints, rather than by feeding the glove's joint angles into MANO:

1. Remove the wrist's global rotation.
2. From the straightest frames, estimate one fixed glove → SMPL-X similarity transform per hand.
3. Fix the subject's SMPL-X `betas` and global body scale.
4. Optimise the 45-D MANO axis-angle of each hand over short time windows.
5. Losses: 20-point positions, bone directions, temporal smoothness, a pose prior and a soft limit on large angles.

The scale in the similarity transform only normalises the subject's glove bone lengths to the calibrated SMPL-X hand size. It is neither the global body scale nor per-frame.

## Running it on one recording

The pipeline's `fuse` stage does this for every session. To inspect a single glove file:

```bash
PYTHONPATH=src python -m five_tracker_smplx.cli fit-manus-hands \
  --input <session>/<session>.npz \
  --subject-profile profiles/collector_a_subject_profile.json \
  --output-dir out/hand_fit
```

The output NPZ holds `left_hand_pose` / `right_hand_pose` `[T, 45]`, the raw wrist-local points, the aligned points, the fitted MANO keypoints and per-frame errors. The HTML viewer shows the glove skeleton, the 21 MANO keypoints, the hand mesh and the residual between them.

## References

- MANUS Core Knowledge Center, *Raw Skeleton*: IDs, parents, chain types and joint types of the 25 nodes.
- MANUS SDK `ManusSDKTypes.h`: `SkeletonNode` / `ManusTransform` and the SDK coordinate system.
- MediaPipe Hands: the 21-landmark layout.
- SMPL-X `joint_names.py`, `body_models.py` and `vertex_ids.py`: joint order, the 15 hand joints and fingertip vertices.
- ContactPose `utilities/mano_fitting.py`: an existing route for fitting MANO pose and shape to 3D hand keypoints.
