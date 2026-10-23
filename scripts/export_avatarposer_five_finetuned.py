#!/usr/bin/env python3
"""Run the locally fine-tuned five-tracker AvatarPoser model."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from five_tracker_smplx.model import ModelPaths, load_body_model
from tracker_streams import RAW_ROLES, wxyz_to_matrix as _wxyz_to_matrix
from train_avatarposer_five_tracker import pack_five_features, sixd

def model_cls():
 import torch
 from torch import nn
 class Net(nn.Module):
  def __init__(self):
   super().__init__(); self.linear_embedding=nn.Linear(90,256); self.transformer_encoder=nn.TransformerEncoder(nn.TransformerEncoderLayer(256,nhead=8),num_layers=3); self.stabilizer=nn.Sequential(nn.Linear(256,256),nn.ReLU(),nn.Linear(256,6)); self.joint_rotation_decoder=nn.Sequential(nn.Linear(256,256),nn.ReLU(),nn.Linear(256,126)); self.contact=nn.Linear(256,2)
  def forward(self,x):
   h=self.transformer_encoder(self.linear_embedding(x).permute(1,0,2)).permute(1,0,2)[:,-1]; return torch.cat((self.stabilizer(h),self.joint_rotation_decoder(h),self.contact(h)),dim=-1)
 return Net()
def decode(v):
 v=v.reshape(-1,22,6); a=v[...,:3]; a=a/np.maximum(np.linalg.norm(a,axis=-1,keepdims=True),1e-8); b=v[...,3:]-np.sum(v[...,3:]*a,axis=-1,keepdims=True)*a; b=b/np.maximum(np.linalg.norm(b,axis=-1,keepdims=True),1e-8); return np.stack((a,b,np.cross(a,b)),axis=-1)
def main():
 p=argparse.ArgumentParser(); p.add_argument('--reference',type=Path,required=True); p.add_argument('--checkpoint',type=Path,default=Path('outputs/avatarposer_five_finetuned_20260905/model.pt')); p.add_argument('--output',type=Path,default=Path('outputs/avatarposer_five_finetuned_20260905/exported')); p.add_argument('--device',default='cpu'); p.add_argument('--batch-size',type=int,default=128); p.add_argument('--window',type=int,default=40); p.add_argument('--root-local',action='store_true'); p.add_argument('--three-anchor',action='store_true'); args=p.parse_args(); import torch
 meta=json.loads(args.reference.with_suffix('.json').read_text()); z=np.load(args.reference); ref={k:np.asarray(z[k]).copy() for k in z.files}; ts=ref['timestamps'].astype(float); ts-=ts[0]
 raw=np.load(meta['source'],allow_pickle=False); pos=np.stack([raw[f'tracker_0__pose.{r}.position'][:len(ts)] for r in RAW_ROLES],1).astype(np.float32); _q=[]; _qok=[]
 for r in RAW_ROLES:
  _qq=np.asarray(raw[f'tracker_0__pose.{r}.orientation'][:len(ts)],dtype=np.float64); _ok=np.isfinite(_qq).all(axis=-1)&(np.linalg.norm(_qq,axis=-1)>1e-8); _q.append(np.where(_ok[:,None],_qq,np.array([1.,0.,0.,0.]))); _qok.append(_ok)
 rot=np.stack([_wxyz_to_matrix(x) for x in _q],1); valid=np.stack([(np.asarray(raw[f'tracker_0__pose.{r}.valid'][:len(ts)],bool) if f'tracker_0__pose.{r}.valid' in raw else np.ones(len(ts),bool)) & ok for r,ok in zip(RAW_ROLES,_qok)],1); raw.close()
 finite=valid & np.isfinite(pos).all(axis=-1) & np.isfinite(rot).all(axis=(-2,-1))
 for j in range(len(RAW_ROLES)):
  good=np.flatnonzero(finite[:,j])
  if len(good)<2: raise ValueError(f"too few valid tracker samples for {RAW_ROLES[j]}")
  clipped=np.clip(ts,ts[good[0]],ts[good[-1]])
  rot[:,j]=Slerp(ts[good],Rotation.from_matrix(rot[good,j]))(clipped).as_matrix()
  for a in range(3): pos[:,j,a]=np.interp(ts,ts[good],pos[good,j,a])
 mountp=ref['attachment_position']; mountr=ref['attachment_rotation']; jointrot=rot@mountr.transpose(0,2,1)[None]; jointpos=pos-(jointrot@mountp[None,:,:,None]).squeeze(-1); q=Rotation.from_euler('x',90,degrees=True).as_matrix().astype(np.float32); origin=np.array([jointpos[0,0,0],meta.get('ground_y_m',float(jointpos[0,0,1])),jointpos[0,0,2]],np.float32); pp=np.einsum('ij,tnj->tni',q,jointpos-origin); rr=q@jointrot
 ut=np.unique(np.append(np.arange(0,ts[-1],1/60),ts[-1])); up=np.stack([np.stack([np.interp(ut,ts,pp[:,j,a]) for a in range(3)],-1) for j in range(5)],1); ur=np.stack([Slerp(ts,Rotation.from_matrix(rr[:,j]))(ut).as_matrix() for j in range(5)],1); head_ur=ur[:,0].copy();
 if args.root_local:
  ur=np.einsum('tij,tnjk->tnik',head_ur.transpose(0,2,1),ur); up=np.einsum('tij,tnj->tni',head_ur.transpose(0,2,1),up-up[:,0:1])
 dr=np.concatenate((np.broadcast_to(np.eye(3,dtype=np.float32),ur[:1].shape),ur[:-1].transpose(0,1,3,2)@ur[1:]),0); dp=np.diff(up,axis=0,prepend=up[:1]); feat=pack_five_features(ur,dr,up,dp).astype(np.float32)
 net=model_cls().to(args.device).eval(); ck=torch.load(args.checkpoint,map_location=args.device,weights_only=False); net.load_state_dict(ck['model']); out=[]; cont=[]
 if args.three_anchor or ck.get('three_anchor'): feat[:,54:]=0.0; print('three-anchor: foot channels zeroed')
 with torch.inference_mode():
  for i in range(min(args.window-1,len(feat))):
   y=net(torch.from_numpy(feat[None,:i+1]).to(args.device))[0].cpu().numpy(); out.append(y[None,:132]); cont.append(y[None,132:])
  for s in range(args.window-1,len(feat),args.batch_size):
   es=range(s,min(len(feat),s+args.batch_size)); w=np.stack([feat[i-args.window+1:i+1] for i in es]); y=net(torch.from_numpy(w).to(args.device)).cpu().numpy(); out.append(y[:,:132]); cont.append(y[:,132:])
 pred=np.concatenate(out); logits=np.concatenate(cont); dec=decode(pred); dec0=np.stack([Slerp(ut,Rotation.from_matrix(dec[:,j]))(ts).as_matrix() for j in range(22)],1); root_dec=dec[:,0];
 if args.root_local: root_dec=np.einsum('tij,tjk->tik',head_ur,root_dec); dec0[:,0]=q.T@np.stack([Slerp(ut,Rotation.from_matrix(root_dec))(ts).as_matrix()],0)[0]
 else: dec0[:,0]=q.T@dec0[:,0]
 aa=Rotation.from_matrix(dec0.reshape(-1,3,3)).as_rotvec().reshape(-1,22,3).astype(np.float32); body=load_body_model(ModelPaths(Path('models/smplx'),None,'smplx','neutral'),args.device).eval(); trans=[]
 with torch.no_grad():
  for s in range(0,len(aa),args.batch_size):
   e=min(len(aa),s+args.batch_size); n=e-s; z3=torch.zeros((n,3),device=args.device); o=body(betas=torch.as_tensor(ref['betas'],device=args.device)[None].expand(n,-1),global_orient=torch.from_numpy(aa[s:e,0]).to(args.device),body_pose=torch.from_numpy(aa[s:e,1:].reshape(n,-1)).to(args.device),transl=z3,left_hand_pose=torch.zeros((n,45),device=args.device),right_hand_pose=torch.zeros((n,45),device=args.device),jaw_pose=z3,leye_pose=z3,reye_pose=z3,expression=torch.zeros((n,10),device=args.device),return_verts=False); trans.append(jointpos[s:e,0]-float(ref['scale'])*o.joints[:,15].cpu().numpy())
 ref.update(global_orient=aa[:,0],body_pose=aa[:,1:].reshape(-1,63),transl=np.concatenate(trans).astype(np.float32)); args.output.mkdir(parents=True,exist_ok=True); np.savez_compressed(args.output/'smplx_profile_driven.npz',**ref); np.save(args.output/'contact_probability.npy',(1/(1+np.exp(-np.clip(logits,-30,30)))).astype(np.float32)); report={'method':'five-tracker AvatarPoser proposal (EgoExplorer C3, Complete stage)','source':meta['source'],'session':meta['session'],'ground_y_m':float(meta.get('ground_y_m',0.0)),'optimization_ground_y_m':float(meta.get('optimization_ground_y_m',meta.get('ground_y_m',0.0))),'checkpoint':str(args.checkpoint),'frames':len(aa),'inference_fps':60,'window':args.window,'input_trackers':list(RAW_ROLES)}; (args.output/'smplx_profile_driven.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n'); print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
