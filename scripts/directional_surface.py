"""Exploratory directional clearance; independent triangle checks remain the evaluator.

Nearest correspondences and area-weighted normals are recomputed but detached:
this optimizes a local point/plane surrogate, not an exact signed distance field.
"""
from itertools import combinations
import numpy as np
import torch
from body_part_surface import interior_samples


def local_penalty(query, target, normals, nearest, margin=.018, support=.1):
    near = target.gather(1, nearest[..., None].expand(-1, -1, 3))
    normal = normals.gather(1, nearest[..., None].expand(-1, -1, 3))
    delta = query - near
    signed = (delta * normal).sum(-1)
    gate = delta.detach().square().sum(-1) < support ** 2
    return torch.relu(margin - signed).square() * gate


class ExploratorySurface:
    def __init__(self, neutral_vertices, skin_weights, faces, mode, count=256):
        self.mode = mode
        device = neutral_vertices.device
        arrays = interior_samples(neutral_vertices.detach().cpu().numpy(), skin_weights.detach().cpu().numpy(), faces, count)
        self.samples = {k: torch.as_tensor(v, device=device, dtype=torch.long) for k, v in arrays.items()}
        self.faces = torch.as_tensor(np.asarray(faces, dtype=np.int64).copy(), device=device, dtype=torch.long)
        self.manifest = dict(mode=mode, sample_count=count, sample_ids={k:v.tolist() for k,v in arrays.items()},
            directed_pair_count=30, margin_m=.018, local_support_m=.1,
            correspondence='nearest sampled vertex; recomputed every iteration, detached',
            normals='area-weighted mesh vertex normals, detached',
            reduction='top 8 per directed pair per frame' if mode=='signed_top8' else 'all points mean',
            limitations='local oriented proxy, not exact SDF or complete containment test')

    def __call__(self, vertices, margin=.018):
        points = {k: vertices[:, ids] for k, ids in self.samples.items()}
        normals = None
        if self.mode != 'unsigned_dense':
            with torch.no_grad():
                tri = vertices.detach()[:, self.faces]
                area = torch.cross(tri[:, :, 1]-tri[:, :, 0], tri[:, :, 2]-tri[:, :, 0], dim=-1)
                all_normals = torch.zeros_like(vertices)
                for j in range(3): all_normals.index_add_(1, self.faces[:, j], area)
                all_normals = torch.nn.functional.normalize(all_normals, dim=-1, eps=1e-10)
                normals = {k: all_normals[:, ids] for k,ids in self.samples.items()}
        terms=[]
        for a,b in combinations(points, 2):
            # Find matches without retaining the quadratic cdist backward graph.
            with torch.no_grad():
                distances=torch.cdist(points[a].detach(), points[b].detach())
                ab=distances.argmin(-1); ba=distances.argmin(-2)
            for src,dst,idx in [(a,b,ab),(b,a,ba)]:
                if self.mode=='unsigned_dense':
                    near=points[dst].gather(1,idx[...,None].expand(-1,-1,3))
                    loss=torch.relu(margin-torch.linalg.vector_norm(points[src]-near,dim=-1)).square()
                else:
                    loss=local_penalty(points[src],points[dst],normals[dst],idx,margin)
                if self.mode=='signed_top8': loss=loss.topk(min(8,loss.shape[-1]),dim=-1).values
                terms.append(loss.mean())
        return torch.stack(terms).mean()
