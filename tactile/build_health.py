"""Aggregate per-session taxel stats into a per-taxel health table.

health h_{d,i} in [0,1] = responsivity of taxel i on day d relative to its own
day-1 responsivity, estimated from the neighbourhood-conditioned relative gain
(usage-invariant).  Also emits:
  * per-day / per-hand median health, dead-taxel counts
  * row x column separability of log-gain (are the losses on the drive lines?)
  * per-region coverage (fraction of a zone's taxels still usable)
  * baseline-offset drift per day
"""
import os, re, glob, json
import numpy as np
import sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from wuji_layout import ROWS, COLS, NCH, REGION_NAMES, region_index

# usage: python tactile/build_health.py [stats dir] [health out dir]
STATS = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, 'stats')
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, 'health')
os.makedirs(OUT, exist_ok=True)
REG = np.array(region_index())
GAIN_MIN_N = 200          # min contact frames for a per-session gain estimate
HANDS = {'glove_0': 'left', 'glove_1': 'right'}


def session_day(name):
    m = re.search(r'session_(\d{8})_(\d{6})', name)
    return (m.group(1), m.group(1) + m.group(2)) if m else (None, None)


def main():
    files = sorted(glob.glob(os.path.join(STATS, 'session_*.npz')))
    names = [os.path.basename(f)[:-4] for f in files]
    meta = [session_day(n) for n in names]
    keep = [i for i, m in enumerate(meta) if m[0]]
    files = [files[i] for i in keep]; names = [names[i] for i in keep]
    days = np.array([meta[i][0] for i in keep])
    order = np.argsort([meta[i][1] for i in keep])
    files = [files[i] for i in order]; names = [names[i] for i in order]; days = days[order]
    uday = sorted(set(days.tolist()))
    print('sessions', len(files), 'days', uday, flush=True)

    S = len(files)
    keys = ['gain', 'gain_n', 'base', 'noise', 'r999', 'r99', 'act', 'valid', 'base_sd']
    A = {g: {k: np.full((S, NCH), np.nan, np.float32) for k in keys} for g in HANDS}
    for s, f in enumerate(files):
        z = np.load(f)
        for g in HANDS:
            for k in keys:
                kk = f'{g}_{k}'
                if kk in z.files:
                    A[g][k][s] = z[kk]

    res = {'sessions': names, 'days': days.tolist(), 'uday': uday,
           'regions': REGION_NAMES}
    for g, side in HANDS.items():
        gain = A[g]['gain'].copy()
        gain[A[g]['gain_n'] < GAIN_MIN_N] = np.nan
        # per-day pooled gain (robust to per-session usage gaps)
        dg = np.full((len(uday), NCH), np.nan, np.float32)
        for d, day in enumerate(uday):
            sel = days == day
            with np.errstate(all='ignore'):
                dg[d] = np.nanmedian(gain[sel], axis=0)
        ref = dg[0].copy()                       # day-1 responsivity
        # taxels with no day-1 estimate: fall back to best day observed
        with np.errstate(all='ignore'):
            bad = ~np.isfinite(ref) | (ref <= 1e-3)
            ref[bad] = np.nanmax(dg[:, bad], axis=0) if bad.any() else ref[bad]
        with np.errstate(all='ignore'):
            h = np.clip(dg / np.maximum(ref, 1e-3), 0, 1.0)
        # never-valid taxels stay NaN
        with np.errstate(all='ignore'):
            never = np.nanmax(A[g]['valid'], axis=0) < 0.5
        h[:, never] = np.nan
        # dead also if the taxel cannot exceed its own noise
        with np.errstate(all='ignore'):
            snr = np.nanmedian(A[g]['r999'] / np.maximum(A[g]['noise'], 1e-4), axis=0)

        # ---- fall back to the coarse part for taxels with no estimate ----
        from wuji_layout import FINGER_OF, REGION_NAMES as RN
        part_of = np.array([{'thumb':0,'index':1,'middle':2,'ring':3,'pinky':4,'palm':5}[FINGER_OF[RN[k]]]
                            if k >= 0 else -1 for k in REG])
        h_src = np.zeros_like(h, np.int8)          # 0 direct, 1 part-fallback, 2 unknown
        for d in range(len(uday)):
            row = h[d]
            miss = ~np.isfinite(row) & ~never
            for p in range(6):
                sel = miss & (part_of == p)
                if not sel.any():
                    continue
                with np.errstate(all='ignore'):
                    v = np.nanmedian(row[(part_of == p) & np.isfinite(row)])
                row[sel] = v if np.isfinite(v) else np.nan
                h_src[d, sel] = 1 if np.isfinite(v) else 2
            h_src[d, never] = 2
        res[f'{side}_health_src'] = h_src
        res[f'{side}_present'] = (~never).astype(np.int8)
        res[f'{side}_health_day'] = h
        res[f'{side}_gain_day'] = dg
        res[f'{side}_gain_ref'] = ref
        res[f'{side}_snr'] = snr
        res[f'{side}_base_day'] = np.stack(
            [np.nanmedian(A[g]['base'][days == d], axis=0) for d in uday])
        res[f'{side}_noise_day'] = np.stack(
            [np.nanmedian(A[g]['noise'][days == d], axis=0) for d in uday])
        res[f'{side}_act_day'] = np.stack(
            [np.nanmedian(A[g]['act'][days == d], axis=0) for d in uday])
        res[f'{side}_valid_day'] = np.stack(
            [np.nanmedian(A[g]['valid'][days == d], axis=0) for d in uday])

        # ---- row x column separability of the loss --------------------
        sep = []
        for d in range(len(uday)):
            L = np.log(np.clip(h[d], 0.02, 1.0)).reshape(ROWS, COLS)
            M = np.isfinite(L) & np.isfinite(h[d].reshape(ROWS, COLS))
            if M.sum() < 50:
                sep.append(np.nan); continue
            Lm = np.where(M, L, np.nan)
            with np.errstate(all='ignore'):
                mu = np.nanmean(Lm)
                a = np.nanmean(Lm, axis=1, keepdims=True) - mu
                b = np.nanmean(Lm, axis=0, keepdims=True) - mu
                fit = mu + a + b
                sse = np.nansum((Lm - fit) ** 2)
                sst = np.nansum((Lm - mu) ** 2)
            sep.append(float(1 - sse / sst) if sst > 0 else np.nan)
        res[f'{side}_rowcol_r2'] = np.array(sep, np.float32)

        # ---- per-region coverage per day ------------------------------
        cov = np.zeros((len(uday), len(REGION_NAMES)), np.float32)
        npres = np.zeros(len(REGION_NAMES), np.int32)
        for k in range(len(REGION_NAMES)):
            npres[k] = int(((REG == k) & ~never).sum())
        for d in range(len(uday)):
            for k in range(len(REGION_NAMES)):
                idx = np.where((REG == k) & ~never)[0]
                hv = np.nan_to_num(h[d, idx])
                cov[d, k] = (hv >= 0.25).sum() / max(npres[k], 1)
        res[f'{side}_coverage_day'] = cov
        res[f'{side}_n_present'] = npres

        # ---- per-SESSION health (the level the pipeline actually uses) ----
        # Drive lines change state between recordings (re-donning / reconnect),
        # so a per-day average is the wrong granularity.  Fallback chain:
        # direct per-session estimate -> that day's estimate -> coarse-part
        # median of the session -> unknown.
        hs = np.clip(gain / np.maximum(ref, 1e-3)[None, :], 0, 1.0)
        hs_src = np.zeros_like(hs, np.int8)
        day_index = {d: i for i, d in enumerate(uday)}
        for si in range(S):
            row = hs[si]
            miss = ~np.isfinite(row) & ~never
            if miss.any():
                dayrow = h[day_index[days[si]]]
                take = miss & np.isfinite(dayrow)
                row[take] = dayrow[take]; hs_src[si, take] = 1
                miss = ~np.isfinite(row) & ~never
            for p in range(6):
                sel = miss & (part_of == p)
                if not sel.any():
                    continue
                with np.errstate(all='ignore'):
                    v = np.nanmedian(row[(part_of == p) & np.isfinite(row)])
                row[sel] = v if np.isfinite(v) else np.nan
                hs_src[si, sel] = 2 if np.isfinite(v) else 3
            hs[si, never] = np.nan; hs_src[si, never] = 3
        res[f'{side}_health_session'] = hs.astype(np.float32)
        res[f'{side}_health_session_src'] = hs_src
        covs = np.zeros((S, len(REGION_NAMES)), np.float32)
        for si in range(S):
            for k in range(len(REGION_NAMES)):
                idx = np.where((REG == k) & ~never)[0]
                covs[si, k] = (np.nan_to_num(hs[si, idx]) >= 0.25).sum() / max(npres[k], 1)
        res[f'{side}_coverage_session'] = covs
        direct = (hs_src == 0) & ~never[None, :]
        print(side, 'per-session health: direct estimates =',
              round(float(direct.sum() / max((~never).sum() * S, 1)), 3),
              '| median coverage per session =',
              round(float(np.median(covs)), 3), flush=True)
        # per-day fractions of dead / degraded taxels (over present taxels)
        frac = np.zeros((len(uday), 3), np.float32)
        for d in range(len(uday)):
            hv = np.nan_to_num(h[d, ~never])
            frac[d] = [(hv < 0.10).mean(), (hv < 0.50).mean(), (hv >= 0.80).mean()]
        res[f'{side}_frac_dead_degraded_good'] = frac
        print(side, 'dead/degraded/good per day:',
              [[round(float(x), 3) for x in frac[d]] for d in range(len(uday))], flush=True)
        print(side, 'median health per day',
              [round(float(np.nanmedian(h[d])), 3) for d in range(len(uday))],
              'rowcol_r2', [round(float(x), 3) for x in sep], flush=True)

    np.savez_compressed(os.path.join(OUT, 'health_db.npz'),
                        **{k: v for k, v in res.items()
                           if isinstance(v, np.ndarray)},
                        sessions=np.array(names), days=days, uday=np.array(uday),
                        regions=np.array(REGION_NAMES))
    json.dump({'sessions': names, 'days': days.tolist(), 'uday': uday},
              open(os.path.join(OUT, 'index.json'), 'w'))
    print('wrote health_db.npz', flush=True)


if __name__ == '__main__':
    main()
