"""Extract a palm-view 2D hand template (both hands) from SMPL-X.

Output: hand_template.json
  hands: {left,right}: {tris: [[x0,y0,x1,y1,x2,y2],...], tri_region: [int,...],
                        regions: [name,...], outline: [[x,y],...] per region boundary edges}
Regions match wuji_layout.ZONES (17 anatomical zones).
"""
import json, os, numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.environ.get('SMPLX_MODEL', os.path.join(HERE, '..', 'models', 'smplx', 'SMPLX_NEUTRAL.npz'))
OUT = os.path.join(HERE, 'hand_template.json')

d = np.load(MODEL, allow_pickle=True)
V = d['v_template'].astype(np.float64)
F = d['f'].astype(np.int64)
W = d['weights'].astype(np.float64)
Jreg = d['J_regressor'].astype(np.float64)
J = Jreg @ V                      # (55,3) joint positions of the template

dom = W.argmax(1)                 # dominant joint per vertex

HANDS = {
    'left':  dict(wrist=20, index=(25, 26, 27), middle=(28, 29, 30),
                  pinky=(31, 32, 33), ring=(34, 35, 36), thumb=(37, 38, 39)),
    'right': dict(wrist=21, index=(40, 41, 42), middle=(43, 44, 45),
                  pinky=(46, 47, 48), ring=(49, 50, 51), thumb=(52, 53, 54)),
}
# glove zone names (see wuji_layout.ZONES)
SEG_NAME = {0: 'proximal', 1: 'middle', 2: 'distal'}

out = {'hands': {}}
for side, cfg in HANDS.items():
    jmap = {}                     # joint id -> region name
    for fng in ('index', 'middle', 'ring', 'pinky'):
        for k, jid in enumerate(cfg[fng]):
            jmap[jid] = f'{fng}_{SEG_NAME[k]}'
    # thumb: glove only has distal/proximal -> thumb1,2 = proximal, thumb3 = distal
    jmap[cfg['thumb'][0]] = 'thumb_proximal'
    jmap[cfg['thumb'][1]] = 'thumb_proximal'
    jmap[cfg['thumb'][2]] = 'thumb_distal'
    jmap[cfg['wrist']] = 'palm'   # split later

    keep_j = set(jmap)
    vsel = np.array([i for i in range(len(V)) if dom[i] in keep_j])
    vset = set(vsel.tolist())

    # anatomical frame
    Wj = J[cfg['wrist']]
    u = J[cfg['middle'][0]] - Wj                # wrist -> middle MCP ("up")
    u /= np.linalg.norm(u)
    q = J[cfg['index'][0]] - J[cfg['pinky'][0]]  # pinky -> index ("radial")
    q = q - u * (q @ u)
    q /= np.linalg.norm(q)
    # outward palm normal: left = q x u, right = u x q
    p = np.cross(q, u) if side == 'left' else np.cross(u, q)
    p /= np.linalg.norm(p)
    x_ax = np.cross(u, p)                        # screen right
    x_ax /= np.linalg.norm(x_ax)

    P = np.stack([(V - Wj) @ x_ax, (V - Wj) @ u], 1)   # 2D screen coords
    depth = (V - Wj) @ p
    radial = (V - Wj) @ q                              # anatomical pinky->index

    # palm sub-zones from radial extent of the palm vertices
    palm_v = np.array([i for i in vsel if dom[i] == cfg['wrist']])
    lo, hi = np.percentile(radial[palm_v], [2, 98])
    t1, t2 = lo + 0.29 * (hi - lo), lo + 0.76 * (hi - lo)

    # faces fully inside the hand, palm-facing only (backface cull)
    tris, treg = [], []
    regions = []
    rid = {}
    for f in F:
        a, b, c = int(f[0]), int(f[1]), int(f[2])
        if a not in vset or b not in vset or c not in vset:
            continue
        n = np.cross(V[b] - V[a], V[c] - V[a])
        if n @ p <= 0:
            continue                                    # dorsal side
        js = [dom[a], dom[b], dom[c]]
        jid = max(set(js), key=js.count)
        name = jmap[jid]
        if name == 'palm':
            r = radial[[a, b, c]].mean()
            name = 'palm_ulnar' if r < t1 else ('palm_central' if r < t2 else 'palm_radial')
        if name not in rid:
            rid[name] = len(regions); regions.append(name)
        tris.append([P[a, 0], P[a, 1], P[b, 0], P[b, 1], P[c, 0], P[c, 1]])
        treg.append(rid[name])
    tris = np.array(tris); treg = np.array(treg)

    # normalise to a 0..1 box (shared scale across hands handled client-side)
    out['hands'][side] = dict(
        tris=[[round(float(v), 5) for v in t] for t in tris],
        tri_region=treg.tolist(), regions=regions,
        n_tris=len(tris),
        bbox=[float(tris[:, 0::2].min()), float(tris[:, 1::2].min()),
              float(tris[:, 0::2].max()), float(tris[:, 1::2].max())])
    print(side, 'tris', len(tris), 'regions', regions)

json.dump(out, open(OUT, 'w'))
print('wrote', OUT)
