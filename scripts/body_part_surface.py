"""General sampled body-part clearance, matching Appendix K's ordered-pair mean.

Six regions, all 30 directed different-region pairs, 96 vertices each. Samples
use skin-weight interiors and deterministic spatial coverage in the fixed,
calibrated neutral mesh. This is still unsigned clearance, not containment.
"""
from __future__ import annotations
from itertools import combinations
import numpy as np
import torch

PARTS={'torso':(0,3,6,9,12),'head':(15,),
       'left_arm':(13,16,18),'right_arm':(14,17,19),
       'left_leg':(1,4,7,10),'right_leg':(2,5,8,11)}


def farthest_samples(vertices,ids,count):
    ids=np.asarray(ids,dtype=np.int64)
    if len(ids)<=count:return ids
    p=np.asarray(vertices)[ids];centroid=p.mean(0)
    chosen=[int(np.argmax(np.sum((p-centroid)**2,axis=1)))]
    nearest=np.sum((p-p[chosen[0]])**2,axis=1)
    for _ in range(1,count):
        nxt=int(np.argmax(nearest));chosen.append(nxt)
        nearest=np.minimum(nearest,np.sum((p-p[nxt])**2,axis=1))
        nearest[chosen]=-1
    return ids[chosen]


def interior_samples(neutral_vertices,skin_weights,faces,count=96,erosion_steps=2):
    v=np.asarray(neutral_vertices);weights=np.asarray(skin_weights)
    faces=np.asarray(faces)
    scores=np.stack([weights[:,ids].sum(1) for ids in PARTS.values()],1)
    labels=scores.argmax(1);labels[scores.max(1)<.8]=-1
    neighbours=[set() for _ in range(len(v))]
    for a,b,c in faces:
        neighbours[a].update((int(b),int(c)));neighbours[b].update((int(a),int(c)));neighbours[c].update((int(a),int(b)))
    selected={}
    for k,name in enumerate(PARTS):
        keep=labels==k
        for _ in range(erosion_steps):
            remove=[i for i in np.flatnonzero(keep) if any(not keep[j] for j in neighbours[i])]
            keep[remove]=False
        ids=np.flatnonzero(keep)
        if len(ids)<count:raise ValueError(f'{name}: only {len(ids)} interior vertices, expected {count}; inspect partition')
        selected[name]=farthest_samples(v,ids,count)
    return selected


def ordered_pair_mean(world_vertices,samples,margin=.018):
    """Mean over frames, vertices and 30 directed part pairs, exactly as defined."""
    terms=[]
    for a,b in combinations(samples,2):
        d=torch.cdist(world_vertices[:,samples[a]],world_vertices[:,samples[b]])
        terms.append(torch.relu(margin-d.amin(dim=2)).square().mean())
        terms.append(torch.relu(margin-d.amin(dim=1)).square().mean())
    if not terms:raise ValueError('surface loss requires at least two nonempty parts')
    return torch.stack(terms).mean()


class BodyPartSurface:
    def __init__(self,neutral_vertices,skin_weights,faces,count=96):
        device=neutral_vertices.device
        arrays=interior_samples(neutral_vertices.detach().cpu().numpy(),skin_weights.detach().cpu().numpy(),faces,count)
        self.samples={k:torch.as_tensor(ids,dtype=torch.long,device=device) for k,ids in arrays.items()}
        self.manifest=dict(parts=list(PARTS),directed_pairs=[[a,b] for a in PARTS for b in PARTS if a!=b],
                           sample_count=count,weight_confidence=.8,boundary_erosion_hops=2,
                           sampler='deterministic farthest-point coverage in calibrated neutral mesh',
                           sample_ids={k:ids.tolist() for k,ids in arrays.items()},
                           reduction='mean over frames, vertices, and all 30 directed pairs',
                           unsigned_distance=True,complete_containment_guarantee=False)

    def __call__(self,world_vertices,margin=.018):return ordered_pair_mean(world_vertices,self.samples,margin)
