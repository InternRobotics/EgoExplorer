#!/usr/bin/env python3
"""Render a Chinese SMPL-X mesh reference video for five-tracker calibration."""

from __future__ import annotations

import argparse
import base64
from dataclasses import dataclass
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.font_manager import FontProperties
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import torch

from five_tracker_smplx.calibration_motion import CalibrationMotionLibrary
from five_tracker_smplx.coordinates import DISPLAY_ROTATION


@dataclass(frozen=True)
class Segment:
    title: str
    instruction: str
    duration: float
    motion: str
    closeup: bool = False


# This is a calibration procedure, not a motion-performance clip.  The holds
# make it possible to identify a single fixed tracker-to-segment transform.
SEGMENTS = (
    Segment("01 自然站立", "双脚平行；双臂自然下垂；目视前方；保持 3 秒不动", 7, "neutral"),
    Segment("02 头部标定", "躯干不动：缓慢左看、右看、低头、抬头，各停 1 秒", 9, "head"),
    Segment("03 A 字站姿", "双臂向下外展约 45°；肘伸直；手心向下", 7, "a_pose"),
    Segment("04 T 字站姿", "双臂水平侧平举；肘伸直；手心向下；保持稳定", 7, "t_pose"),
    Segment("05 前平举", "双臂向身体前方至肩高；肘伸直；手心向下；保持稳定", 7, "forward"),
    Segment("06 屈肘标定", "双臂侧平举并翻掌向上；先屈右肘、再屈左肘，交替两轮", 8, "elbows"),
    Segment("07 手腕标定", "双臂前平举；手腕依次掌心向下、翻转、俯、仰", 8, "wrists"),
    Segment("08 浅蹲", "双脚不移动；躯干尽量竖直；慢速浅蹲两次", 8, "squat"),
    Segment("09 左腿", "抬起受试者自己的左腿（正面观察时在画面右侧）后放下两次；避免侧倾", 7, "left_knee"),
    Segment("10 右腿", "抬起受试者自己的右腿（正面观察时在画面左侧）后放下两次；避免侧倾", 7, "right_knee"),
    Segment("11 踝关节", "原地先抬脚跟、再抬脚尖；不要移动脚的位置", 7, "ankles"),
    Segment("12 张手与并指", "双手前举：五指自然张开、并拢；手掌始终可见", 7, "hand_open", True),
    Segment("13 握拳", "缓慢握拳、完全张开，重复两次；手腕保持稳定", 7, "fist", True),
    Segment("14 捏合与对掌", "拇指依次触碰食指、中指、无名指、小指，再完全张开", 9, "pinch", True),
    Segment("15 单指屈伸", "食指、中指、无名指、小指依次屈伸；动作慢且幅度完整", 9, "fingers", True),
    Segment("16 左右转体 180°", "小步原地转身：先向左 180°、回正，再向右 180°、回正；头和躯干同向", 16, "turns"),
    Segment("17 前后各走两步", "沿脚尖方向连续向前迈两步，再连续向后退两步回到地面标记；始终朝前", 14, "walk"),
)


def _smooth_step(value: float) -> float:
    value = float(np.clip(value, 0.0, 1.0))
    return value * value * (3.0 - 2.0 * value)


def _root_motion(motion: str, phase: float) -> tuple[np.ndarray, np.ndarray]:
    """Return SMPL-X global axis-angle and translation for dynamic stages."""
    orient = np.zeros(3, dtype=np.float32)
    translation = np.zeros(3, dtype=np.float32)
    if motion == "turns":
        # Sixteen 45-degree footfalls: four left, four back to front, four
        # right, four back. Small increments avoid a stiff 90-degree pivot.
        section = min(15, int(np.clip(phase, 0.0, 1.0 - 1e-8) * 16.0))
        amount = _smooth_step(phase * 16.0 - section)
        yaw_keys = np.deg2rad((
            0., 45., 90., 135., 180., 135., 90., 45., 0.,
            -45., -90., -135., -180., -135., -90., -45., 0.,
        ))
        orient[1] = (1.0 - amount) * yaw_keys[section] + amount * yaw_keys[section + 1]
    elif motion == "walk":
        # Exactly two footfalls move 32 cm forward, followed by two backward
        # footfalls returning to the mark. The former 16-section path encoded
        # eight footfalls and contradicted the written four-step instruction.
        section = min(7, int(np.clip(phase, 0.0, 1.0 - 1e-8) * 8.0))
        amount = _smooth_step(phase * 8.0 - section)
        z_keys = (
            0., .08, .16, .24, .32, .24, .16, .08, 0.,
        )
        translation[2] = (1.0 - amount) * z_keys[section] + amount * z_keys[section + 1]
    return orient, translation


def _stance_contact(motion: str, phase: float) -> int:
    """Return the SMPL-X ankle joint that must remain planted, or -1."""
    if motion == "turns":
        section = min(15, int(np.clip(phase, 0.0, 1.0 - 1e-8) * 16.0))
        # Even steps swing the left foot; odd steps swing the right.
        return 8 if section % 2 == 0 else 7
    if motion == "walk":
        section = min(7, int(np.clip(phase, 0.0, 1.0 - 1e-8) * 8.0))
        # Forward: left/right swing. Backward: right/left swing. The opposite
        # ankle remains the hard anchor.
        return (
            8, 8, 7, 7, 7, 7, 8, 8,
        )[section]
    return -1


def sequence(fps: int, library: CalibrationMotionLibrary):
    body, left, right, orient, transl, contacts, titles, instructions, closeups, elapsed = [], [], [], [], [], [], [], [], [], []
    time = 0.0
    for segment in SEGMENTS:
        count = max(2, round(segment.duration * fps))
        for index in range(count):
            phase = index / (count - 1)
            frame = library.frame(segment.motion, phase)
            body.append(frame.body_pose); left.append(frame.left_hand_pose); right.append(frame.right_hand_pose)
            root_orient, root_translation = _root_motion(segment.motion, phase)
            orient.append(root_orient); transl.append(root_translation)
            contacts.append(_stance_contact(segment.motion, phase))
            titles.append(segment.title); instructions.append(segment.instruction); closeups.append(segment.closeup)
            elapsed.append(time + phase * segment.duration)
        time += segment.duration
    return (np.stack(body), np.stack(left), np.stack(right), np.stack(orient), np.stack(transl), np.asarray(contacts), titles, instructions, closeups, np.asarray(elapsed), time)


def compressed_mesh(vertices: np.ndarray, faces: np.ndarray, size: float = .028):
    """Voxel-cluster a dense SMPL-X mesh once for fast software rendering."""
    cells = np.floor((vertices - vertices.min(axis=0)) / size).astype(np.int64)
    _, inverse = np.unique(cells, axis=0, return_inverse=True)
    count = inverse.max() + 1
    clustered_faces = inverse[faces]
    valid = np.array([len(set(face)) == 3 for face in clustered_faces])
    clustered_faces = clustered_faces[valid]
    _, first = np.unique(np.sort(clustered_faces, axis=1), axis=0, return_index=True)
    return inverse, np.sort(first), clustered_faces[np.sort(first)], count


def reduce_vertices(vertices: np.ndarray, inverse: np.ndarray, count: int) -> np.ndarray:
    denominator = np.bincount(inverse, minlength=count).astype(np.float32)
    return np.stack([np.bincount(inverse, weights=vertices[:, axis], minlength=count) / denominator for axis in range(3)], axis=1)


def mesh_sequence(
    fps: int,
    body_model_path: Path,
    vposer_path: Path,
    motion_cache_path: Path,
    *,
    rebuild_motion_cache: bool = False,
    voxel_size: float = .028,
):
    library = CalibrationMotionLibrary(
        body_model_path, vposer_path, cache_path=motion_cache_path,
        rebuild_cache=rebuild_motion_cache, device="cpu",
    )
    poses, left_hands, right_hands, orientations, translations, contacts, titles, instructions, closeups, elapsed, total = sequence(fps, library)
    body = library.body
    vertex_batches, joint_batches = [], []
    with torch.no_grad():
        for start in range(0, len(poses), 48):
            end = min(len(poses), start + 48); count = end - start
            out = body(
                # SMPL-X is rooted around the pelvis.  The per-frame root is
                # on a level floor, including the downward pelvis travel in a squat.
                betas=torch.zeros((count, 10)), global_orient=torch.as_tensor(orientations[start:end]), transl=torch.as_tensor(translations[start:end]),
                body_pose=torch.as_tensor(poses[start:end]), left_hand_pose=torch.as_tensor(left_hands[start:end]), right_hand_pose=torch.as_tensor(right_hands[start:end]),
                jaw_pose=torch.zeros((count, 3)), leye_pose=torch.zeros((count, 3)), reye_pose=torch.zeros((count, 3)), expression=torch.zeros((count, 10)), return_verts=True,
            )
            vertex_batches.append(out.vertices.cpu().numpy())
            joint_batches.append(out.joints[:, :22].cpu().numpy())
    vertices = np.concatenate(vertex_batches)
    joints = np.concatenate(joint_batches)

    # Contact-aware root correction. A stance ankle keeps one fixed horizontal
    # world point until the next footfall; on support changes the new anchor is
    # initialized under the already corrected foot, so root motion is C0 and
    # the turn/walk cannot skate across the floor.
    correction = np.zeros(2, dtype=np.float32)
    anchor = None
    previous_contact = -1
    for index, contact in enumerate(contacts):
        contact = int(contact)
        if contact < 0:
            correction[:] = 0.; anchor = None; previous_contact = -1
            continue
        point = joints[index, contact, (0, 2)]
        if contact != previous_contact or anchor is None:
            anchor = point + correction
        correction = anchor - point
        vertices[index, :, 0] += correction[0]; vertices[index, :, 2] += correction[1]
        joints[index, :, 0] += correction[0]; joints[index, :, 2] += correction[1]
        previous_contact = contact

    # Keep the mesh on one fixed horizontal ground plane. Horizontal contact
    # has already been solved above; this vertical shift only removes tiny
    # VPoser/mesh-sole height differences between key poses.
    vertices[:, :, 1] -= vertices[:, :, 1].min(axis=1, keepdims=True)
    # Never merge vertices across a moving SMPL-X joint: that produces false
    # "sheets" between limbs in an animated mesh.  Keep the original topology
    # and sample only faces for a lighter viewer/video.
    face_stride = 1 if voxel_size >= .04 else 4
    faces = np.asarray(body.faces, dtype=np.int32)[::face_stride]
    # Use a proper right-handed rotation into left/back/up display space.
    # The previous bare (X,Z,Y) swap had det=-1 and mirrored anatomy.
    vertices = [frame @ DISPLAY_ROTATION.T for frame in vertices]
    return vertices, faces, titles, instructions, closeups, elapsed, total


def render(output: Path, fps: int, body_model_path: Path, vposer_path: Path, motion_cache_path: Path, rebuild_motion_cache: bool = False) -> None:
    vertices, faces, titles, instructions, closeups, elapsed, total = mesh_sequence(
        fps, body_model_path, vposer_path, motion_cache_path, rebuild_motion_cache=rebuild_motion_cache,
    )
    font = FontProperties(fname="/System/Library/Fonts/STHeiti Medium.ttc", size=15)
    small_font = FontProperties(fname="/System/Library/Fonts/STHeiti Light.ttc", size=11)
    fig = plt.figure(figsize=(10, 7), facecolor="#f7f8fa")
    ax = fig.add_subplot(111, projection="3d")
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = FFMpegWriter(fps=fps, bitrate=2200)
    with writer.saving(fig, str(output), dpi=120):
        for index, frame in enumerate(vertices):
            ax.clear()
            mesh = Poly3DCollection(frame[faces], facecolor="#d9a37b", edgecolor="#875f48", linewidth=.08, alpha=1.0)
            ax.add_collection3d(mesh)
            if closeups[index]:
                ax.set(xlim=(-.95, .95), ylim=(-1.10, .25), zlim=(.65, 1.92))
                # A three-quarter close-up leaves the forward-extended hands
                # visible rather than hiding them behind the chest.
                ax.view_init(elev=7, azim=-135)
            else:
                ax.set(xlim=(-1.15, 1.15), ylim=(-1.25, .75), zlim=(-.04, 1.95))
                # Forward-reaching poses need a three-quarter view; a pure
                # frontal view would hide the arms behind the torso.
                side_view = titles[index].startswith(("05", "06", "07"))
                ax.view_init(elev=10, azim=-135 if side_view else -92)
            # Flat ground is intentionally only a visual reference; no tracker
            # positions are rendered in this instructional video.
            ground_x, ground_y = np.meshgrid(np.linspace(-1.15, 1.15, 2), np.linspace(-1.25, .75, 2))
            ax.plot_surface(ground_x, ground_y, np.zeros_like(ground_x), color="#b9c1c8", alpha=.18, shade=False)
            ax.set_box_aspect((1.9, 1.0, 2.0)); ax.set_axis_off()
            fig.suptitle(f"五个 Vive Tracker：人体与手指标定动作参考\n{titles[index]}", fontproperties=font, y=.965)
            ax.text2D(.5, .94, instructions[index], transform=ax.transAxes, ha="center", fontproperties=small_font)
            ax.text2D(.5, .02, f"{elapsed[index]:05.1f} / {total:.0f} 秒    无镜像：正面观察时，受试者左侧在画面右边", transform=ax.transAxes, ha="center", fontproperties=small_font)
            writer.grab_frame()
    plt.close(fig)


def render_multiview(
    output: Path,
    fps: int,
    body_model_path: Path,
    vposer_path: Path,
    motion_cache_path: Path,
    rebuild_motion_cache: bool = False,
) -> None:
    """Render synchronized front, anatomical-left and anatomical-right views."""
    vertices, faces, titles, instructions, _, elapsed, total = mesh_sequence(
        fps, body_model_path, vposer_path, motion_cache_path,
        rebuild_motion_cache=rebuild_motion_cache, voxel_size=.045,
    )
    font = FontProperties(fname="/System/Library/Fonts/STHeiti Medium.ttc", size=15)
    label_font = FontProperties(fname="/System/Library/Fonts/STHeiti Medium.ttc", size=12)
    small_font = FontProperties(fname="/System/Library/Fonts/STHeiti Light.ttc", size=11)
    fig = plt.figure(figsize=(16, 6.6), facecolor="#f7f8fa")
    views = (("正面（无镜像）", -90), ("受试者左侧面", 0), ("受试者右侧面", 180))
    axes = [fig.add_subplot(1, 3, index + 1, projection="3d") for index in range(3)]
    meshes = []
    ground_x, ground_y = np.meshgrid(np.linspace(-1.15, 1.15, 2), np.linspace(-1.25, .75, 2))
    first = vertices[0]
    for axis, (label, azimuth) in zip(axes, views, strict=True):
        mesh = Poly3DCollection(
            first[faces], facecolor="#d9a37b", edgecolor="#875f48",
            linewidth=.018, alpha=1.0,
        )
        axis.add_collection3d(mesh); meshes.append(mesh)
        axis.plot_surface(
            ground_x, ground_y, np.zeros_like(ground_x),
            color="#b9c1c8", alpha=.20, shade=False,
        )
        axis.set(xlim=(-1.15, 1.15), ylim=(-1.25, .75), zlim=(-.04, 1.95))
        axis.set_box_aspect((1.8, 1.55, 2.0))
        axis.view_init(elev=8, azim=azimuth)
        axis.set_axis_off()
        axis.set_title(label, fontproperties=label_font, pad=2)
    title_text = fig.suptitle("", fontproperties=font, y=.975)
    instruction_text = fig.text(.5, .925, "", ha="center", fontproperties=small_font, color="#263647")
    time_text = fig.text(.5, .025, "", ha="center", fontproperties=small_font, color="#5d6975")
    fig.subplots_adjust(left=.01, right=.99, bottom=.06, top=.89, wspace=.01)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = FFMpegWriter(
        fps=fps, bitrate=5200,
        metadata={"title": "SMPL-X 五 Tracker 标定动作三视角参考"},
    )
    with writer.saving(fig, str(output), dpi=120):
        for index, frame in enumerate(vertices):
            polygons = frame[faces]
            for mesh in meshes:
                mesh.set_verts(polygons)
            title_text.set_text(f"五个 Vive Tracker：人体与手指标定动作参考 · {titles[index]}")
            instruction_text.set_text(instructions[index])
            time_text.set_text(
                f"{elapsed[index]:05.1f} / {total:.0f} 秒    无镜像：正面中受试者左侧在画面右边"
            )
            writer.grab_frame()
    plt.close(fig)


def render_html(output: Path, fps: int, body_model_path: Path, vposer_path: Path, motion_cache_path: Path, rebuild_motion_cache: bool = False) -> None:
    """Write a rotatable, text-synchronised mesh viewer (no tracker markers)."""
    vertices, faces, titles, instructions, _, elapsed, total = mesh_sequence(
        float(fps), body_model_path, vposer_path, motion_cache_path,
        rebuild_motion_cache=rebuild_motion_cache, voxel_size=.045,
    )
    vertex_scale = 1e-4
    quantized_vertices = np.rint(np.stack(vertices) / vertex_scale).astype("<i2")
    payload = {
        "vertex_data_b64": base64.b64encode(quantized_vertices.tobytes()).decode("ascii"),
        "vertex_scale": vertex_scale,
        "vertex_count": int(quantized_vertices.shape[1]),
        "frame_count": int(quantized_vertices.shape[0]),
        "faces": np.asarray(faces, dtype=np.int32).tolist(),
        "titles": titles, "instructions": instructions,
        "times": np.round(elapsed, 3).tolist(), "total": float(total),
        "frame_interval": float(total / len(vertices)),
    }
    document = """<!doctype html><html lang=\"zh-CN\"><head><meta charset=\"utf-8\">
<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\"><title>SMPL-X 标定动作参考</title>
<script src=\"https://cdn.plot.ly/plotly-2.35.2.min.js\"></script>
<style>body{margin:0;background:#f7f8fa;color:#18212b;font:15px -apple-system,BlinkMacSystemFont,'PingFang SC','Microsoft YaHei',sans-serif}#bar{padding:12px 20px;background:#fff;border-bottom:1px solid #dfe3e8}h1{font-size:20px;margin:0 0 8px}#instruction{font-size:17px;margin:4px 0;color:#263647}#frame{width:min(760px,75vw);vertical-align:middle}button{padding:5px 12px;margin-right:6px}.hint{color:#5d6975;font-size:13px;margin-left:10px}.contract{color:#b42318;font-weight:650;margin-left:12px}#plot{width:100vw;height:calc(100vh - 125px)}</style></head>
<body><div id=\"bar\"><h1 id=\"title\"></h1><div id=\"instruction\"></div><button id=\"play\">播放</button><button id=\"prev\">上一帧</button><button id=\"next\">下一帧</button><input id=\"frame\" type=\"range\" min=\"0\" step=\"1\"><span id=\"time\"></span><span class=\"contract\">无镜像：正面时受试者左=画面右</span><span class=\"hint\">拖动旋转 · 滚轮缩放 · 右键拖动平移</span></div><div id=\"plot\"></div>
<script>const data=__PAYLOAD__, slider=document.querySelector('#frame'),title=document.querySelector('#title'),instruction=document.querySelector('#instruction'),time=document.querySelector('#time'),play=document.querySelector('#play');slider.max=data.frame_count-1;let timer=null;
const encoded=atob(data.vertex_data_b64),bytes=new Uint8Array(encoded.length);for(let n=0;n<encoded.length;n++)bytes[n]=encoded.charCodeAt(n);const packed=new Int16Array(bytes.buffer),faceI=data.faces.map(a=>a[0]),faceJ=data.faces.map(a=>a[1]),faceK=data.faces.map(a=>a[2]);
function frameVertices(n){const x=new Array(data.vertex_count),y=new Array(data.vertex_count),z=new Array(data.vertex_count),offset=n*data.vertex_count*3,s=data.vertex_scale;for(let k=0;k<data.vertex_count;k++){x[k]=packed[offset+3*k]*s;y[k]=packed[offset+3*k+1]*s;z[k]=packed[offset+3*k+2]*s;}return{x,y,z};}
function traces(n){const v=frameVertices(n);return [{type:'mesh3d',x:v.x,y:v.y,z:v.z,i:faceI,j:faceJ,k:faceK,color:'#d9a37b',flatshading:false,lighting:{ambient:.55,diffuse:.7,specular:.05,roughness:.9},name:'SMPL-X 网格',hoverinfo:'skip'}, {type:'mesh3d',x:[-1.2,1.2,1.2,-1.2],y:[-1.4,-1.4,.8,.8],z:[0,0,0,0],i:[0,0],j:[1,2],k:[2,3],color:'#aab4bf',opacity:.22,hoverinfo:'skip',name:'地面'}]}
const layout={margin:{l:0,r:0,b:0,t:0},showlegend:false,uirevision:'calibration-camera',scene:{aspectmode:'manual',aspectratio:{x:1,y:.9,z:1.15},xaxis:{title:'+X 受试者左 / -X 受试者右（m）',range:[-1.2,1.2]},yaxis:{title:'+Y 身体后方 / -Y 身体前方（m）',range:[-1.4,.8]},zaxis:{title:'+Z 向上（m）',range:[-.1,2]},camera:{eye:{x:0,y:-2.2,z:1.05}}}};
function draw(n){n=Math.max(0,Math.min(data.frame_count-1,n));slider.value=n;title.textContent=data.titles[n];instruction.textContent=data.instructions[n];time.textContent=` ${data.times[n].toFixed(1)} / ${data.total.toFixed(0)} 秒`;Plotly.react('plot',traces(n),layout,{responsive:true,scrollZoom:true});}slider.oninput=()=>draw(+slider.value);document.querySelector('#prev').onclick=()=>draw(+slider.value-1);document.querySelector('#next').onclick=()=>draw(+slider.value+1);play.onclick=()=>{if(timer){clearInterval(timer);timer=null;play.textContent='播放';return;}play.textContent='暂停';timer=setInterval(()=>draw((+slider.value+1)%data.frame_count),Math.max(80,data.frame_interval*1000));};draw(0);</script></body></html>""".replace("__PAYLOAD__", json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document, encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="outputs/calibration_protocol_reference.mp4")
    parser.add_argument("--multiview-output", help="write a front/left/right synchronized MP4")
    parser.add_argument("--html-output", help="write an interactive 3D HTML viewer instead of an MP4")
    parser.add_argument("--fps", type=int, default=3)
    parser.add_argument("--body-model-path", default="models/smplx")
    parser.add_argument("--vposer-path", default="models/vposer/V02_05")
    parser.add_argument("--motion-cache", default="assets/calibration_motion_vposer_mano.npz")
    parser.add_argument("--rebuild-motion-cache", action="store_true")
    args = parser.parse_args()
    if args.multiview_output:
        render_multiview(Path(args.multiview_output), args.fps, Path(args.body_model_path), Path(args.vposer_path), Path(args.motion_cache), args.rebuild_motion_cache)
    elif args.html_output:
        render_html(Path(args.html_output), args.fps, Path(args.body_model_path), Path(args.vposer_path), Path(args.motion_cache), args.rebuild_motion_cache)
    else:
        render(Path(args.output), args.fps, Path(args.body_model_path), Path(args.vposer_path), Path(args.motion_cache), args.rebuild_motion_cache)
