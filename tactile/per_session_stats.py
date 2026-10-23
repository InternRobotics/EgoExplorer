"""Per-session / per-taxel tactile statistics for the wuji glove recordings.

For every session and both gloves we compute, per taxel (744 = 24x31):
  valid      fraction of frames with a value (>=0)
  base       session baseline = p10 of raw value      (non-contact offset)
  base_sd    spread of the block-wise baseline        (within-session drift)
  noise      robust sigma of the baseline-removed residual on quiet frames
  r99 / r999 high quantiles of the residual           (achieved response)
  act        fraction of frames with residual > max(5*noise, 0.02)
  gain       neighbourhood-conditioned relative gain (see below)
  gain_n     number of contact frames the gain was estimated from

`gain` is the key degradation observable: for each anatomical region k we take
the per-frame p75 over that region's taxels as a usage reference; on frames
where that reference is clearly in contact we take the median ratio
residual_i / reference_k.  It is invariant to how hard / how often the wearer
pressed, so it isolates the sensor's own responsivity.
"""
import os, sys, glob, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
# usage: python tactile/per_session_stats.py <folder with Wuji session dirs> [stats out dir]
BASE = sys.argv[1]
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, 'stats')
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, HERE)
from wuji_layout import channel_regions, REGION_NAMES  # noqa

STEP = 2          # frame subsample
BLK = 300         # baseline block length (frames, post-subsample) ~5 s at 60 Hz eff.
REG = np.array([REGION_NAMES.index(r) if r else -1 for r in channel_regions()])
NREG = len(REGION_NAMES)


def block_baseline(Xn, blk=BLK):
    """Piecewise-linear p10 baseline per channel."""
    T, C = Xn.shape
    nb = max(1, int(np.ceil(T / blk)))
    pad = nb * blk - T
    Z = np.concatenate([Xn, np.full((pad, C), np.nan, Xn.dtype)], 0) if pad else Xn
    with np.errstate(all='ignore'):
        bq = np.nanpercentile(Z.reshape(nb, blk, C), 10, axis=1)     # (nb, C)
    if nb == 1:
        return np.repeat(bq, T, axis=0)
    ctr = (np.arange(nb) + 0.5) * blk
    t = np.arange(T, dtype=np.float64)
    out = np.empty((T, C), np.float32)
    for c in range(C):
        col = bq[:, c]
        good = np.isfinite(col)
        out[:, c] = np.interp(t, ctr[good], col[good]) if good.any() else np.nan
    return out, bq


def hand_stats(X):
    X = X[::STEP].astype(np.float32)
    T, C = X.shape
    Xn = np.where(X >= 0, X, np.nan)
    valid = np.isfinite(Xn).mean(0)
    base_f, bq = block_baseline(Xn)
    R = Xn - base_f                                   # residual
    with np.errstate(all='ignore'):
        base = np.nanpercentile(Xn, 10, axis=0)
        base_sd = np.nanstd(bq, axis=0)
        q70 = np.nanpercentile(R, 70, axis=0)
        quiet = np.where(R <= q70, R, np.nan)
        med = np.nanmedian(quiet, axis=0)
        noise = 1.4826 * np.nanmedian(np.abs(quiet - med), axis=0)
        r99 = np.nanpercentile(R, 99, axis=0)
        r999 = np.nanpercentile(R, 99.9, axis=0)
    noise = np.where(np.isfinite(noise) & (noise > 1e-4), noise, 1e-4)
    thr = np.maximum(5 * noise, 0.02)
    act = np.nanmean(R > thr, axis=0)

    # ---- neighbourhood-conditioned relative gain -------------------------
    gain = np.full(C, np.nan, np.float32)
    gain_n = np.zeros(C, np.int32)
    for k in range(NREG):
        idx = np.where(REG == k)[0]
        if idx.size < 6:
            continue
        Rk = R[:, idx]
        if not np.isfinite(Rk).any():
            continue
        with np.errstate(all='ignore'):
            ref = np.nanpercentile(Rk, 75, axis=1)         # per-frame usage reference
        sel = np.isfinite(ref) & (ref > 0.05)
        if sel.sum() < 30:
            continue
        with np.errstate(all='ignore'):
            ratio = Rk[sel] / ref[sel, None]
            g = np.nanmedian(ratio, axis=0)
        gain[idx] = g
        gain_n[idx] = int(sel.sum())
    return dict(valid=valid.astype(np.float32), base=base.astype(np.float32),
                base_sd=base_sd.astype(np.float32), noise=noise.astype(np.float32),
                r99=r99.astype(np.float32), r999=r999.astype(np.float32),
                act=act.astype(np.float32), gain=gain, gain_n=gain_n,
                frames=np.int32(T * STEP))


def main():
    files = sorted(glob.glob(os.path.join(BASE, 'session_*', 'session_*.npz')))
    files = [f for f in files if '.video.' not in f and '.tracker.' not in f]
    ns = int(os.environ.get('NSHARD', '1')); sh = int(os.environ.get('SHARD', '0'))
    files = [f for i, f in enumerate(files) if i % ns == sh]
    print('sessions', len(files), 'shard', sh, '/', ns, flush=True)
    for n, f in enumerate(files):
        name = os.path.basename(os.path.dirname(f))
        dst = os.path.join(OUT, name + '.npz')
        if os.path.exists(dst):
            continue
        try:
            z = np.load(f, allow_pickle=True)
        except Exception as e:
            print('skip', name, e, flush=True); continue
        rec = {}
        for g in ('glove_0', 'glove_1'):
            key = g + '__tactile'
            if key not in z.files:
                continue
            try:
                s = hand_stats(z[key])
            except Exception as e:
                print('fail', name, g, e, flush=True); continue
            for k, v in s.items():
                rec[f'{g}_{k}'] = v
        if rec:
            np.savez_compressed(dst, **rec)
        if n % 10 == 0:
            print(n, name, flush=True)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
