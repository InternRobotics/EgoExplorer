"""Raw -> corrected -> region-contact pipeline for the wuji tactile glove.

Stages (each one is a separate, independently publishable data product):

  L0 raw        as recorded, -1 preserved.                     [ship as-is]
  L1 residual   baseline-removed:  r = x - b(t),  b = rolling p10.
                Removes the non-zero no-contact offset and its slow drift.
  L2 z          noise-normalised:  z = r / sigma_i, sigma_i = robust MAD of r
                on quiet frames.  Unit-free, comparable across taxels.
  L3 corrected  gain-compensated:  c = z / max(h_i, H_FLOOR), taxels with
                h_i < H_DEAD masked to NaN.  h_i = per-taxel health coefficient.
  L4 contact    per-region contact score + hysteresis binary + coverage.

Everything above L0 is reversible / recomputable from L0 + the health table,
so the release only needs L0 + health + this code.
"""
from __future__ import annotations
import numpy as np

from wuji_layout import ROWS, COLS, NCH, REGION_NAMES, region_index, FINGER_OF

REG = np.array(region_index())
NREG = len(REGION_NAMES)

# --- tunables (all reported in the paper) ---------------------------------
BASE_WIN_S = 10.0     # rolling-baseline window
BASE_Q = 10           # rolling-baseline quantile (%)
H_FLOOR = 0.15        # max gain boost = 1/0.15 ~ 6.7x
H_DEAD = 0.10         # below this a taxel is declared dead and masked
H_MIN_REGION = 0.25   # taxels below this are excluded from region scores
LSB = 5e-4            # quantisation step of the stored 0..1 values
SIG_FLOOR = 2 * LSB   # noise floor: never trust a sigma below 2 LSB
Z_ON, Z_OFF = 8.0, 4.0        # relative (SNR) criterion
R_ON, R_OFF = 0.030, 0.015    # absolute criterion, in raw units
MIN_ON_S, MIN_OFF_S = 0.05, 0.05
REG_Q = 0.80          # within-region quantile used as the region score


def rolling_baseline(X, fps, win_s=BASE_WIN_S, q=BASE_Q, blocks_per_win=4):
    """Piecewise-linear rolling low quantile, NaN-safe. X: (T,C) with NaN."""
    T, C = X.shape
    blk = max(8, int(round(fps * win_s / blocks_per_win)))
    nb = max(1, int(np.ceil(T / blk)))
    pad = nb * blk - T
    Z = np.concatenate([X, np.full((pad, C), np.nan, X.dtype)], 0) if pad else X
    with np.errstate(all='ignore'):
        bq = np.nanpercentile(Z.reshape(nb, blk, C), q, axis=1)
    # smooth across blocks with a running min-of-medians to stay under contacts
    if nb >= 3:
        bq = np.minimum(bq, np.nanmedian(
            np.stack([np.roll(bq, k, 0) for k in (-1, 0, 1)]), axis=0))
    if nb == 1:
        return np.repeat(bq, T, axis=0).astype(np.float32)
    ctr = (np.arange(nb) + 0.5) * blk
    t = np.arange(T, dtype=np.float64)
    out = np.empty((T, C), np.float32)
    for c in range(C):
        col = bq[:, c]
        g = np.isfinite(col)
        out[:, c] = np.interp(t, ctr[g], col[g]) if g.any() else np.nan
    return out


def residual(raw, fps):
    """L0 -> L1.  raw: (T,744) float with -1 for invalid."""
    X = np.where(raw >= 0, raw, np.nan).astype(np.float32)
    b = rolling_baseline(X, fps)
    return X - b, b


def noise_sigma(r):
    """Robust per-taxel noise AND residual offset, estimated on quiet frames.

    A p10 rolling baseline sits below the true no-contact level, so the
    residual keeps a small positive offset; we measure it on the quiet half of
    the frames and return it so it can be removed.
    """
    with np.errstate(all='ignore'):
        q70 = np.nanpercentile(r, 70, axis=0)
        quiet = np.where(r <= q70, r, np.nan)
        med = np.nanmedian(quiet, axis=0)
        s = 1.4826 * np.nanmedian(np.abs(quiet - med), axis=0)
    s = np.where(np.isfinite(s), np.maximum(s, SIG_FLOOR), SIG_FLOOR)
    med = np.where(np.isfinite(med), med, 0.0)
    return s.astype(np.float32), med.astype(np.float32)


def corrected(r, sigma, health):
    """L1 -> L3.  health: (744,) in [0,1]; NaN/dead taxels -> NaN."""
    z = r / sigma[None, :]
    h = np.where(np.isfinite(health), health, 0.0)
    c = z / np.maximum(h, H_FLOOR)[None, :]
    c[:, h < H_DEAD] = np.nan
    return z.astype(np.float32), c.astype(np.float32)


def region_scores(z, r, health, present=None, q=REG_Q):
    """(T,744) -> (T,NREG) health-gated within-region high quantile.

    Returns the SNR score, the matching absolute-residual score and the
    coverage, i.e. the fraction of *physically present* taxels of the zone
    that are still healthy enough to be used.
    """
    T = z.shape[0]
    out = np.full((T, NREG), np.nan, np.float32)
    outr = np.full((T, NREG), np.nan, np.float32)
    cov = np.zeros(NREG, np.float32)
    nuse = np.zeros(NREG, np.int32)
    if present is None:
        present = np.isfinite(health)
    for k in range(NREG):
        idx = np.where((REG == k) & (health >= H_MIN_REGION) & np.isfinite(health))[0]
        allk = np.where((REG == k) & present)[0]
        cov[k] = len(idx) / max(len(allk), 1)
        nuse[k] = len(idx)
        if idx.size == 0:
            continue
        with np.errstate(all='ignore'):
            out[:, k] = np.nanquantile(z[:, idx], q, axis=1)
            outr[:, k] = np.nanquantile(r[:, idx], q, axis=1)
    return out, outr, cov, nuse


def hysteresis(score, rscore, fps, on=Z_ON, off=Z_OFF, ron=R_ON, roff=R_OFF,
               min_on=MIN_ON_S, min_off=MIN_OFF_S):
    """(T,K) scores -> (T,K) bool contact.

    Dual criterion: a contact must be both statistically significant (z > on)
    and physically non-trivial (residual > ron); it is released when either
    drops below its off-threshold.  Hysteresis + minimum durations suppress
    chatter.
    """
    T, K = score.shape
    out = np.zeros((T, K), bool)
    n_on, n_off = max(1, int(fps * min_on)), max(1, int(fps * min_off))
    for k in range(K):
        s = score[:, k]
        if not np.isfinite(s).any():
            continue
        state, run = False, 0
        col = out[:, k]
        for t in range(T):
            v = s[t]
            if not np.isfinite(v):
                col[t] = state; continue
            w = rscore[t, k]
            if state:
                run = run + 1 if (v < off or w < roff) else 0
                if run >= n_off:
                    state, run = False, 0
            else:
                run = run + 1 if (v > on and w > ron) else 0
                if run >= n_on:
                    state, run = True, 0
            col[t] = state
    return out


def coarse_from_regions(contact, scores):
    """17 zones -> 6 coarse parts (thumb/index/middle/ring/pinky/palm)."""
    parts = ['thumb', 'index', 'middle', 'ring', 'pinky', 'palm']
    C = np.zeros((contact.shape[0], len(parts)), bool)
    S = np.full((scores.shape[0], len(parts)), np.nan, np.float32)
    for i, p in enumerate(parts):
        cols = [k for k, n in enumerate(REGION_NAMES) if FINGER_OF[n] == p]
        C[:, i] = contact[:, cols].any(1)
        with np.errstate(all='ignore'):
            S[:, i] = np.nanmax(scores[:, cols], axis=1)
    return parts, C, S


def process(raw, fps, health):
    """Full L0->L4 for one glove.  Returns a dict of every level."""
    present = (raw >= 0).any(0)
    r0, b = residual(raw, fps)
    sigma, off = noise_sigma(r0)
    r = r0 - off[None, :]            # re-centred residual (L1)
    z, c = corrected(r, sigma, health)
    s, sr, cov, nuse = region_scores(z, r, health, present)
    on = hysteresis(s, sr, fps)
    parts, cC, cS = coarse_from_regions(on, s)
    return dict(residual=r, baseline=b + off[None, :], sigma=sigma, offset=off,
                z=z, corrected=c, present=present,
                region_score=s, region_res=sr, region_contact=on,
                coverage=cov, n_used=nuse,
                coarse_parts=parts, coarse_contact=cC, coarse_score=cS)
