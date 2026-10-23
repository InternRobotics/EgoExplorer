"""Batch profile-driven reconstruction and a multi-recording mesh viewer."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from .coordinates import DISPLAY_ROTATION, recorded_tracker_axes_in_world
from .export import write_smplx_npz
from .model import ModelPaths, load_body_model
from .optimizer import _global_body_rotations
from .profile_fitting import ProfileDrivenFitter, ProfileFitOptions, ProfileFitResult, SubjectProfile
from .protocol import ProtocolAlignment, align_protocol
from .behavior_protocol import refine_alignment_by_behavior
from .raw_trackers import read_raw_tracker_npz
from .subject_calibration import TRACKER_NAMES


_TRACKED_IDS = np.asarray((15, 20, 21, 7, 8), dtype=np.int64)
_BODY_EDGES = ((0, 1), (0, 2), (0, 3), (1, 4), (2, 5), (3, 6), (4, 7), (5, 8), (6, 9), (7, 10), (8, 11), (9, 12), (9, 13), (9, 14), (12, 15), (13, 16), (14, 17), (16, 18), (17, 19), (18, 20), (19, 21))


def _session_files(roots: Sequence[str | Path]) -> list[Path]:
    result = []
    for value in roots:
        root = Path(value)
        if root.is_file() and root.name.endswith(".tracker.npz"):
            result.append(root)
        elif root.is_dir():
            result.extend(root.glob("*.tracker.npz"))
            result.extend(root.glob("*/*.tracker.npz"))
        else:
            raise FileNotFoundError(root)
    unique = {path.resolve(): path for path in result}
    return sorted(unique.values(), key=lambda path: path.stem)


def _session_id(path: Path) -> str:
    suffix = ".tracker.npz"
    return path.name[: -len(suffix)] if path.name.endswith(suffix) else path.stem


def reconstruct_profile_batch(
    profile_path: str | Path,
    input_roots: Sequence[str | Path],
    output_dir: str | Path,
    *,
    options: ProfileFitOptions | None = None,
    viewer_max_frames: int = 280,
    use_protocol: bool = True,
    write_viewer: bool = True,
    reuse_existing: bool = False,
    continue_on_error: bool = False,
) -> dict:
    """Fit every discovered tracker NPZ and write one selectable HTML viewer.

    ``use_protocol=False`` is the inference path for ordinary recordings.  It
    deliberately prevents an arbitrary trajectory from being segmented as the
    01--17 calibration protocol, while still applying the already calibrated
    subject shape, metric scale and fixed tracker mounts.
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if reuse_existing and write_viewer:
        raise ValueError("reuse_existing requires write_viewer=False; build the consolidated viewer after fitting")
    profile = SubjectProfile.load(profile_path)
    options = options or ProfileFitOptions()
    fitter = ProfileDrivenFitter(profile, options)
    files = _session_files(input_roots)
    if not files:
        raise FileNotFoundError(f"no *.tracker.npz below {input_roots}")
    results: list[tuple[str, Path, ProfileFitResult]] = []
    report_entries: list[dict] = []
    errors: list[dict] = []
    progress_path = output / "progress_report.json"

    def write_progress(status: str) -> None:
        progress = {
            "status": status,
            "subject_profile": str(Path(profile_path).resolve()),
            "protocol_mode": "calibration" if use_protocol else "unlabelled_inference",
            "total_sequences": len(files),
            "completed_sequences": len(report_entries),
            "failed_sequences": len(errors),
            "sequences": report_entries,
            "errors": errors,
        }
        progress_path.write_text(json.dumps(progress, ensure_ascii=False, indent=2), encoding="utf-8")

    write_progress("running")
    for file_index, path in enumerate(files, start=1):
        session_id = _session_id(path)
        session_output = output / session_id
        session_output.mkdir(parents=True, exist_ok=True)
        trajectory_path = session_output / "smplx_profile_driven.npz"
        summary_path = trajectory_path.with_suffix(".json")
        diagnostics_path = session_output / "tracker_diagnostics.npz"
        if reuse_existing and trajectory_path.exists() and summary_path.exists() and diagnostics_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            report_entries.append({
                "source": str(path), "output": str(session_output), "reused": True, **summary,
            })
            write_progress("running")
            print(f"[reconstruct-profile] {file_index}/{len(files)} reused {session_id}", flush=True)
            continue
        try:
            frames = read_raw_tracker_npz(path)
            glove_path = path.with_name(f"{session_id}.npz")
            protocol = (
                align_protocol(frames, glove_path=glove_path if glove_path.exists() else None)
                if use_protocol else None
            )
            if protocol is not None and options is not None and options.behavior_driven_protocol:
                protocol = refine_alignment_by_behavior(frames, protocol)
            result = fitter.fit(frames, session_id=session_id, protocol=protocol)
            protocol_path = session_output / "protocol_alignment.json"
            if protocol is not None:
                protocol_path.write_text(
                    json.dumps(protocol.to_mapping(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
                )
            elif protocol_path.exists():
                # Avoid a stale calibration marker when an output directory is
                # intentionally reused for ordinary inference.
                protocol_path.unlink()
            summary = {
                **result.summary, "source": str(path.resolve()),
                "subject_profile": str(Path(profile_path).resolve()),
                "coordinate_frame": "raw global, +Y up",
                "translation_semantics": "world translation applied after metric scale",
            }
            write_smplx_npz(trajectory_path, result.trajectory, metadata=summary)
            np.savez_compressed(
                diagnostics_path, valid=result.valid,
                predicted_tracker=result.predicted_tracker, position_residual_m=result.position_residual_m,
                predicted_tracker_rotation=result.predicted_tracker_rotation
                if result.predicted_tracker_rotation is not None else np.empty((0, 5, 3, 3), np.float32),
                rotation_residual_deg=result.rotation_residual_deg
                if result.rotation_residual_deg is not None else np.empty((0, 5), np.float32),
            )
            if write_viewer:
                results.append((session_id, path, result))
            report_entries.append({"source": str(path), "output": str(session_output), **result.summary})
            write_progress("running")
            print(f"[reconstruct-profile] {file_index}/{len(files)} completed {session_id}", flush=True)
        except Exception as exc:
            error = {"session": session_id, "source": str(path), "error": repr(exc)}
            errors.append(error)
            write_progress("running_with_errors")
            print(f"[reconstruct-profile] {file_index}/{len(files)} FAILED {session_id}: {exc!r}", flush=True)
            if not continue_on_error:
                raise
    viewer_path = output / "all_sequences.html"
    if write_viewer:
        write_multi_sequence_html(
            viewer_path, results, profile,
            body_model_path=fitter.options.body_model_path, device=fitter.device,
            max_frames=viewer_max_frames,
        )
    report = {
        "subject_profile": str(Path(profile_path).resolve()),
        "viewer": str(viewer_path) if write_viewer else None,
        "protocol_mode": "calibration" if use_protocol else "unlabelled_inference",
        "sequences": report_entries,
        "errors": errors,
    }
    (output / "batch_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    write_progress("complete_with_errors" if errors else "complete")
    return report


def _cluster_topology(body, template_vertices: np.ndarray, cell_size: float = .025):
    weights = body.lbs_weights.detach().cpu().numpy()
    dominant_joint = weights.argmax(axis=1).astype(np.int32)
    cell = np.floor((template_vertices - template_vertices.min(axis=0)) / cell_size).astype(np.int32)
    _, inverse = np.unique(np.column_stack((cell, dominant_joint)), axis=0, return_inverse=True)
    count = int(inverse.max()) + 1
    mapped = inverse[np.asarray(body.faces, dtype=np.int64)]
    valid = np.asarray([len(set(face)) == 3 for face in mapped], dtype=bool)
    mapped = mapped[valid]
    _, first = np.unique(np.sort(mapped, axis=1), axis=0, return_index=True)
    faces = mapped[np.sort(first)].astype(np.int32)
    denominator = np.bincount(inverse, minlength=count).astype(np.float32)
    return inverse, denominator, faces


def _reduce_vertices(vertices: np.ndarray, inverse: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return np.stack([
        np.bincount(inverse, weights=vertices[:, axis], minlength=len(denominator)) / denominator
        for axis in range(3)
    ], axis=1).astype(np.float32)


def _viewer_selection(
    count: int, maximum: int, timestamps: np.ndarray | None = None,
    protocol: ProtocolAlignment | None = None,
) -> np.ndarray:
    if maximum <= 0 or count <= maximum:
        return np.arange(count, dtype=np.int64)
    if protocol is not None and timestamps is not None:
        action_quota = max(8, (maximum // 2) // len(protocol.segments))
        action = []
        for segment in protocol.segments:
            candidates = np.flatnonzero(
                (timestamps >= segment.active_start_s) & (timestamps <= segment.active_end_s)
            )
            if len(candidates):
                action.extend(candidates[np.unique(np.rint(np.linspace(0, len(candidates) - 1, min(action_quota, len(candidates)))).astype(int))])
        action = np.unique(np.asarray(action, dtype=np.int64))
        remaining = max(2, maximum - len(action))
        base = np.unique(np.rint(np.linspace(0, count - 1, remaining)).astype(np.int64))
        selected = np.unique(np.r_[action, base])
        if len(selected) <= maximum:
            return selected
    return np.unique(np.rint(np.linspace(0, count - 1, maximum)).astype(np.int64))


def write_multi_sequence_html(
    path: str | Path,
    results: Sequence[tuple[str, Path, ProfileFitResult]],
    profile: SubjectProfile,
    *,
    body_model_path: str | Path,
    device: str = "cpu",
    max_frames: int = 280,
) -> None:
    """Render skin-segment-aware reduced SMPL-X meshes for all sessions."""
    import torch

    body = load_body_model(ModelPaths(Path(body_model_path), None, "smplx", "neutral"), device).eval()
    for parameter in body.parameters():
        parameter.requires_grad_(False)
    betas = torch.as_tensor(profile.betas, dtype=torch.float32, device=device)
    with torch.no_grad():
        zero3 = torch.zeros((1, 3), device=device); zero10 = torch.zeros((1, 10), device=device)
        template = body(
            betas=betas[None], global_orient=zero3, transl=zero3, body_pose=torch.zeros((1, 63), device=device),
            left_hand_pose=torch.zeros((1, 45), device=device), right_hand_pose=torch.zeros((1, 45), device=device),
            jaw_pose=zero3, leye_pose=zero3, reye_pose=zero3, expression=zero10, return_verts=True,
        ).vertices[0].cpu().numpy()
    inverse, denominator, faces = _cluster_topology(body, template)
    parents = body.parents.to(device)
    display_rotation = DISPLAY_ROTATION
    sequences = []
    for session_id, source_path, result in results:
        trajectory = result.trajectory
        selection = _viewer_selection(
            len(trajectory.timestamps), max_frames, trajectory.timestamps, result.protocol,
        )
        vertices_all, joints_all, rotations_all = [], [], []
        for chunk_start in range(0, len(selection), 80):
            selected = selection[chunk_start:chunk_start + 80]
            count = len(selected)
            zero3 = torch.zeros((count, 3), device=device); zero10 = torch.zeros((count, 10), device=device)
            global_orient = torch.as_tensor(trajectory.global_orient[selected], dtype=torch.float32, device=device)
            body_pose = torch.as_tensor(trajectory.body_pose[selected], dtype=torch.float32, device=device).reshape(count, 21, 3)
            left_hand_pose = torch.as_tensor(trajectory.left_hand_pose[selected], dtype=torch.float32, device=device)
            right_hand_pose = torch.as_tensor(trajectory.right_hand_pose[selected], dtype=torch.float32, device=device)
            translation = np.asarray(trajectory.transl[selected], dtype=np.float32)
            with torch.no_grad():
                output = body(
                    betas=betas[None].expand(count, -1), global_orient=global_orient, transl=zero3,
                    body_pose=body_pose.reshape(count, -1), left_hand_pose=left_hand_pose,
                    right_hand_pose=right_hand_pose, jaw_pose=zero3, leye_pose=zero3,
                    reye_pose=zero3, expression=zero10, return_verts=True,
                )
                world_vertices = trajectory.scale * output.vertices.cpu().numpy() + translation[:, None]
                world_joints = trajectory.scale * output.joints.cpu().numpy() + translation[:, None]
                rotations = _global_body_rotations(global_orient, body_pose, parents).cpu().numpy()
            vertices_all.extend(_reduce_vertices(frame, inverse, denominator) for frame in world_vertices)
            joints_all.append(world_joints[:, :22]); rotations_all.append(rotations)
        vertices = np.stack(vertices_all)
        joints = np.concatenate(joints_all)
        joint_rotations = np.concatenate(rotations_all)
        tracker = np.stack([[result.frames[index][name].position for name in TRACKER_NAMES] for index in selection]).astype(np.float32)
        tracker_rotation = np.stack([[result.frames[index][name].rotation for name in TRACKER_NAMES] for index in selection]).astype(np.float32)
        valid = result.valid[selection]
        predicted = result.predicted_tracker[selection]
        residual = result.position_residual_m[selection]
        floor = float(result.ground_y_m)
        for values in (vertices, joints, tracker, predicted):
            values[..., 1] -= floor
        vertices = vertices @ display_rotation.T
        joints = joints @ display_rotation.T
        tracker = tracker @ display_rotation.T
        predicted = predicted @ display_rotation.T
        tracker_rotation = np.einsum(
            "ij,ftjk->ftik", display_rotation,
            recorded_tracker_axes_in_world(tracker_rotation, convention="columns"),
        )
        initial_count = min(20, len(joints))
        lateral = np.mean(
            .5 * (
                (joints[:initial_count, 1, :2] - joints[:initial_count, 2, :2])
                + (joints[:initial_count, 16, :2] - joints[:initial_count, 17, :2])
            ),
            axis=0,
        )
        lateral /= max(float(np.linalg.norm(lateral)), 1e-8)
        default_camera_xy = np.asarray((lateral[1], -lateral[0]))
        visible_tracker = np.where(valid[:, :, None], tracker, np.nan)
        visible_points = np.concatenate((joints, visible_tracker), axis=1)
        lower = np.nanmin(visible_points, axis=(0, 1)); upper = np.nanmax(visible_points, axis=(0, 1))
        padding = np.maximum(.12, .08 * (upper - lower))
        lower -= padding; upper += padding; lower[2] = min(0.0, lower[2])
        center = .5 * (vertices.min(axis=(0, 1)) + vertices.max(axis=(0, 1)))
        extent = float(np.max(np.abs(vertices - center)))
        quantization = max(1e-4, extent / 32000.0)
        packed = np.rint((vertices - center) / quantization).astype("<i2")
        packed_base64 = base64.b64encode(packed.tobytes()).decode("ascii")
        action_indices = (
            result.protocol.labels(trajectory.timestamps[selection]).tolist()
            if result.protocol is not None else [0] * len(selection)
        )
        protocol_segments = (
            result.protocol.to_mapping()["segments"] if result.protocol is not None else
            [_unlabelled_segment(trajectory.timestamps)]
        )
        sequences.append({
            "id": session_id, "source": str(source_path), "timestamps": trajectory.timestamps[selection].round(6).tolist(),
            "source_indices": selection.tolist(), "action_indices": action_indices,
            "protocol_segments": protocol_segments,
            "mesh": packed_base64, "mesh_origin": center.round(6).tolist(), "mesh_scale": quantization,
            "joints": joints.round(5).tolist(), "tracker": tracker.round(5).tolist(),
            "tracker_rotation": tracker_rotation.round(6).tolist(), "predicted": predicted.round(5).tolist(),
            "valid": valid.tolist(), "residual": np.where(np.isfinite(residual), residual, -1).round(5).tolist(),
            "range": np.stack((lower, upper), axis=1).round(4).tolist(),
            "default_camera_xy": default_camera_xy.round(6).tolist(), "summary": result.summary,
        })
    payload = json.dumps({
        "sequences": sequences, "faces": faces.tolist(), "vertex_count": int(len(denominator)),
        "tracker_names": TRACKER_NAMES, "tracker_joint_ids": _TRACKED_IDS.tolist(),
        "edges": _BODY_EDGES,
    }, ensure_ascii=False, separators=(",", ":"))
    document = """<!doctype html><html lang=\"zh-CN\"><head><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">
<title>Subject-profile driven SMPL-X</title><script src=\"https://cdn.plot.ly/plotly-2.35.2.min.js\"></script>
<style>body{margin:0;background:#090f1d;color:#e5e7eb;font:14px -apple-system,BlinkMacSystemFont,\"PingFang SC\",sans-serif}#bar{display:flex;gap:10px;align-items:center;flex-wrap:wrap;padding:9px 14px;background:#111827;border-bottom:1px solid #334155}select,button{color:#e5e7eb;background:#243146;border:1px solid #475569;border-radius:5px;padding:6px 10px}#frame{width:min(520px,38vw)}#plot{height:calc(100vh - 94px)}#status{padding:5px 14px;color:#cbd5e1;background:#0f172a}.tag{color:#93c5fd}label{white-space:nowrap}</style></head><body>
<div id=\"bar\"><span>序列</span><select id=\"sequence\"></select><span>动作</span><select id=\"action\"></select><button id=\"play\">播放</button><input id=\"frame\" type=\"range\" min=\"0\" step=\"1\"><span id=\"time\"></span><label><input id=\"mesh\" type=\"checkbox\" checked> Mesh</label><label><input id=\"axes\" type=\"checkbox\"> Tracker 三轴</label><label><input id=\"residual\" type=\"checkbox\" checked> 残差线</label><label><input id=\"follow\" type=\"checkbox\"> 跟随人体</label><span style=\"color:#fca5a5\">相机按首帧正面；无镜像：受试者左=画面右</span></div><div id=\"status\"></div><div id=\"plot\"></div>
<script>const data=__PAYLOAD__;const colors=['#f43f5e','#22d3ee','#818cf8','#fbbf24','#a3e635'];const sel=document.getElementById('sequence'),actionSel=document.getElementById('action'),slider=document.getElementById('frame'),play=document.getElementById('play'),time=document.getElementById('time'),meshToggle=document.getElementById('mesh'),axesToggle=document.getElementById('axes'),resToggle=document.getElementById('residual'),followToggle=document.getElementById('follow'),status=document.getElementById('status');let timer=null,meshCache={};
data.sequences.forEach((s,i)=>{const o=document.createElement('option');o.value=i;o.textContent=s.id;sel.appendChild(o)});
function packed(si){if(meshCache[si])return meshCache[si];const raw=atob(data.sequences[si].mesh),bytes=new Uint8Array(raw.length);for(let i=0;i<raw.length;i++)bytes[i]=raw.charCodeAt(i);return meshCache[si]=new Int16Array(bytes.buffer);}
function verts(si,fi){const s=data.sequences[si],q=packed(si),n=data.vertex_count,off=fi*n*3,x=[],y=[],z=[];for(let i=0;i<n;i++){x.push(s.mesh_origin[0]+s.mesh_scale*q[off+3*i]);y.push(s.mesh_origin[1]+s.mesh_scale*q[off+3*i+1]);z.push(s.mesh_origin[2]+s.mesh_scale*q[off+3*i+2]);}return{x,y,z};}
function segments(p,e){const x=[],y=[],z=[];for(const a of e){x.push(p[a[0]][0],p[a[1]][0],null);y.push(p[a[0]][1],p[a[1]][1],null);z.push(p[a[0]][2],p[a[1]][2],null);}return{x,y,z};}
function trackerAxes(s,f){const out=[];for(let a=0;a<3;a++){const x=[],y=[],z=[];for(let n=0;n<5;n++){if(!s.valid[f][n])continue;const p=s.tracker[f][n],r=s.tracker_rotation[f][n],d=[r[0][a]*.11,r[1][a]*.11,r[2][a]*.11];x.push(p[0],p[0]+d[0],null);y.push(p[1],p[1]+d[1],null);z.push(p[2],p[2]+d[2],null);}out.push({type:'scatter3d',mode:'lines',x,y,z,line:{color:['#ef4444','#22c55e','#3b82f6'][a],width:5},name:['Tracker +X','Tracker +Y','Tracker +Z'][a]});}return out;}
function traces(si,f){const s=data.sequences[si],v=verts(si,f),j=s.joints[f],t=s.tracker[f],p=s.predicted[f],sk=segments(j,data.edges),mx=[],my=[],mz=[],rx=[],ry=[],rz=[],tx=[],ty=[],tz=[],tc=[],tt=[];for(let n=0;n<5;n++){if(!s.valid[f][n])continue;mx.push(j[data.tracker_joint_ids[n]][0],p[n][0],null);my.push(j[data.tracker_joint_ids[n]][1],p[n][1],null);mz.push(j[data.tracker_joint_ids[n]][2],p[n][2],null);rx.push(p[n][0],t[n][0],null);ry.push(p[n][1],t[n][1],null);rz.push(p[n][2],t[n][2],null);tx.push(t[n][0]);ty.push(t[n][1]);tz.push(t[n][2]);tc.push(colors[n]);tt.push(`${data.tracker_names[n]}<br>残差 ${(s.residual[f][n]*100).toFixed(1)} cm`);}const F=data.faces,gx=s.range[0],gy=s.range[1];const out=[{type:'mesh3d',x:[gx[0],gx[1],gx[1],gx[0]],y:[gy[0],gy[0],gy[1],gy[1]],z:[0,0,0,0],i:[0,0],j:[1,2],k:[2,3],color:'#31506f',opacity:.32,name:'地面',hoverinfo:'skip'}];if(meshToggle.checked)out.push({type:'mesh3d',x:v.x,y:v.y,z:v.z,i:F.map(q=>q[0]),j:F.map(q=>q[1]),k:F.map(q=>q[2]),color:'#aeb8c5',opacity:.72,flatshading:false,lighting:{ambient:.55,diffuse:.75,specular:.15,roughness:.8},name:'SMPL-X mesh'});out.push({type:'scatter3d',mode:'lines',x:sk.x,y:sk.y,z:sk.z,line:{color:'#111827',width:5},name:'SMPL-X 骨架'},{type:'scatter3d',mode:'lines',x:mx,y:my,z:mz,line:{color:'#2563eb',width:4},name:'固定安装偏置'},{type:'scatter3d',mode:'markers',x:p.map(q=>q[0]),y:p.map(q=>q[1]),z:p.map(q=>q[2]),marker:{size:4,color:colors,symbol:'diamond'},name:'模型预测 Tracker'});if(resToggle.checked)out.push({type:'scatter3d',mode:'lines',x:rx,y:ry,z:rz,line:{color:'#f97316',width:4,dash:'dot'},name:'拟合残差'});out.push({type:'scatter3d',mode:'markers+text',x:tx,y:ty,z:tz,text:tt,hovertemplate:'%{text}<extra></extra>',marker:{size:6,color:tc,symbol:'x',line:{width:2}},name:'实测 Tracker'});if(axesToggle.checked)out.push(...trackerAxes(s,f));return out;}
function layout(si,f){const s=data.sequences[si],r=s.range,c=s.joints[f][0],useFollow=followToggle.checked,rx=useFollow?[c[0]-1.0,c[0]+1.0]:r[0],ry=useFollow?[c[1]-1.0,c[1]+1.0]:r[1],rz=useFollow?[-.08,1.75]:r[2],ca=s.default_camera_xy;return{margin:{l:0,r:0,b:0,t:35},paper_bgcolor:'#090f1d',font:{color:'#e5e7eb'},title:{text:s.id,font:{size:15}},legend:{x:.01,y:.99,bgcolor:'rgba(9,15,29,.55)'},uirevision:'session-'+si+'-'+useFollow,scene:{xaxis:{title:'参考左 +X / 右 -X',range:rx,gridcolor:'#334155',backgroundcolor:'#090f1d'},yaxis:{title:'参考后方 +Y / 前方 -Y',range:ry,gridcolor:'#334155',backgroundcolor:'#090f1d'},zaxis:{title:'向上 +Z',range:rz,gridcolor:'#334155',backgroundcolor:'#090f1d'},aspectmode:'data',camera:{eye:{x:2.35*ca[0],y:2.35*ca[1],z:1.25},up:{x:0,y:0,z:1}}}};}
function draw(){const si=+sel.value,s=data.sequences[si],f=Math.min(+slider.value,s.timestamps.length-1);slider.value=f;const action=s.protocol_segments[s.action_indices[f]],dt=f?s.timestamps[f]-s.timestamps[f-1]:0,gap=dt>.5?`　<span style=\"color:#fb923c\">显示抽帧 Δt=${dt.toFixed(2)}s</span>`:'';if(action)actionSel.value=s.action_indices[f];time.textContent=`源帧 ${s.source_indices[f]}/${s.summary.source_frames-1} · t=${s.timestamps[f].toFixed(2)} s`;const vals=s.residual[f].filter(x=>x>=0),rms=Math.sqrt(vals.reduce((a,x)=>a+x*x,0)/Math.max(1,vals.length));status.innerHTML=`<span class=tag>${action?action.title:'未标注'}</span>　整体位置 RMS ${(s.summary.position_rms_m*100).toFixed(2)} cm　当前帧 ${(rms*100).toFixed(2)} cm　完整输出 ${s.summary.fitted_frames}/${s.summary.source_frames} 帧　最大原始缺口 ${s.summary.maximum_gap_s.toFixed(2)}s　排除 Tracker：${s.summary.excluded_trackers.join(', ')||'无'}${gap}　源文件：${s.source}`;Plotly.react('plot',traces(si,f),layout(si,f),{responsive:true,scrollZoom:true});}
function fillActions(){const s=data.sequences[+sel.value];actionSel.innerHTML='';s.protocol_segments.forEach((a,i)=>{const o=document.createElement('option');o.value=i;o.textContent=a.title;actionSel.appendChild(o)});}
function jumpAction(){const s=data.sequences[+sel.value],i=+actionSel.value,a=s.protocol_segments[i],mid=.5*(a.active_start_s+a.active_end_s);let best=0,d=Infinity;s.timestamps.forEach((t,k)=>{if(s.action_indices[k]===i&&Math.abs(t-mid)<d){best=k;d=Math.abs(t-mid)}});slider.value=best;draw();}
function change(){const s=data.sequences[+sel.value];slider.max=s.timestamps.length-1;slider.value=0;fillActions();draw();}sel.onchange=change;actionSel.onchange=jumpAction;slider.oninput=draw;meshToggle.onchange=draw;axesToggle.onchange=draw;resToggle.onchange=draw;followToggle.onchange=draw;play.onclick=()=>{if(timer){clearInterval(timer);timer=null;play.textContent='播放';return}play.textContent='暂停';timer=setInterval(()=>{const s=data.sequences[+sel.value];slider.value=(+slider.value+1)%s.timestamps.length;draw();},1000/12)};change();</script></body></html>""".replace("__PAYLOAD__", payload)
    destination = Path(path); destination.parent.mkdir(parents=True, exist_ok=True); destination.write_text(document, encoding="utf-8")


def _unlabelled_segment(timestamps: np.ndarray) -> dict:
    """Return one viewer-compatible segment for a non-calibration recording."""
    start = float(timestamps[0]) if len(timestamps) else 0.0
    end = float(timestamps[-1]) if len(timestamps) else start
    return {
        "index": 0,
        "key": "unlabelled_sequence",
        "title": "完整测试轨迹（不参与标定）",
        "motion": "unlabelled",
        "start_s": start,
        "end_s": end,
        "active_start_s": start,
        "active_end_s": end,
        "confidence": "not_applicable",
    }
