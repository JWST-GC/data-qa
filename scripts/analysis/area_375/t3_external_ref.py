"""Test 3 (#375): per-frame magnitudes against external Ks (VIRAC2, GNS DR1) and the pixel-area term.

Per frame row (same per-frame table as test 2): d = m_frame (AB, PIXAR_SR) - Ks (Vega), fitted as
    d = z_frame + cK (Ks - 15.5) + cHK (H - Ks - 1.5) + sum_det k_det * term_i * 1(det_i = det)
term = a = 2.5 log10 AREA (model 'a') or pT = a + sT (model 'pT', the pre-#1154 PSF error).
z_frame (one per exposure/detector) absorbs the AB-Vega offset and the tile/exposure zero point,
so k comes from the within-frame spatial pattern against an external reference (absolute side;
any true flat-field / image-level pattern within a detector enters here, unlike test 2).
VIRAC2: tile cache (PM-propagated to the epoch in its file name), Ks 14.5-16.5, nearest VIRAC
neighbour > 1".  GNS: J/A+A/631/A20 central (epoch ~2015.5, no PM), Ks 14.5-17, no GNS star within
0.5" brighter than Ks + 3.  H - Ks from the same reference.  Matching: tile star list (i2d
source_catalog positions) after a median bulk offset, 0.12" (GNS) / 0.15" (VIRAC), unique.
Errors: bootstrap over tiles.
Usage: python t3_external_ref.py  -> test3.json
"""
import glob
import json
import os
import re

import numpy as np
from astropy.table import Table
from scipy.spatial import cKDTree

import common as C

CEN = json.load(open(os.path.join(C.HERE, 'centres.json')))
VC = '/orange/adamginsburg/jwst/gc-treasury/catalogs/virac2_ks_cache/'
# GNS Ks tiles cut around each pointing (built for #362); set AREA375_GNS_DIR.
GNS = os.environ.get('AREA375_GNS_DIR', os.path.join(C.WORK, 'gns_tiles')) + '/'
rng = np.random.default_rng(2)


def F(c):
    return np.asarray(np.ma.filled(np.ma.asarray(c, dtype=float), np.nan), float)


def ref_tile(obs, which):
    ra0, de0 = CEN[obs][0], CEN[obs][1]
    cd = np.cos(np.radians(de0))
    if which == 'virac':
        f = glob.glob(VC + f'virac2_ks_*_o{obs}.fits')[0]
        ep = float(re.search(r'epoch(\d+\.\d+)', f).group(1))
        v = Table.read(f)
        ks, h = F(v['Ksmag']), F(v['Hmag'])
        dec = F(v['DEJ2000'])
        ra = F(v['RAJ2000']) + np.nan_to_num(F(v['pmRA'])) * (ep - 2014.0) / 3.6e6 / np.cos(np.radians(dec))
        dec = dec + np.nan_to_num(F(v['pmDE'])) * (ep - 2014.0) / 3.6e6
        ok = np.isfinite(ks) & np.isfinite(ra)
        R = np.c_[(ra - ra0) * cd * 3600, (dec - de0) * 3600][ok]
        ks, h = ks[ok], h[ok]
        dd, _ = cKDTree(R).query(R, k=2)
        iso = dd[:, 1] > 1.0
        rng_ = (14.5, 16.5)
        rad = 0.15
    else:
        g = Table.read(GNS + f'{obs}.fits')
        if len(g) == 0:
            return None
        ks, h = F(g['Ksmag']), F(g['Hmag'])
        ok = np.isfinite(ks)
        R = np.c_[(F(g['RAJ2000']) - ra0) * cd * 3600, (F(g['DEJ2000']) - de0) * 3600][ok]
        ks, h = ks[ok], h[ok]
        nb = cKDTree(R).query_ball_point(R, r=0.5)
        iso = np.array([np.all(ks[[j for j in n if j != i]] > ks[i] + 3) for i, n in enumerate(nb)])
        rng_ = (14.5, 17.0)
        rad = 0.12
    sel = iso & (ks > rng_[0]) & (ks < rng_[1])
    return R, ks, h, sel, rad


def match(D, which):
    """Ks, H per row of D (NaN when unmatched)."""
    Ks = np.full(len(D['x']), np.nan)
    H = np.full(len(D['x']), np.nan)
    for obs in np.unique(D['obs']):
        o = f'{obs:03d}'
        r = ref_tile(o, which)
        if r is None:
            continue
        R, ks, h, sel, rad = r
        q = np.flatnonzero(D['obs'] == obs)
        sid, first = np.unique(D['sid'][q], return_index=True)
        P = np.c_[D['X'][q][first], D['Y'][q][first]] * 60
        tr = cKDTree(R)
        d0, j0 = tr.query(P, distance_upper_bound=0.3)
        ok = np.isfinite(d0)
        off = np.median(P[ok] - R[j0[ok]], axis=0)
        d, j = tr.query(P - off, distance_upper_bound=rad)
        ok = np.isfinite(d)
        u, c = np.unique(j[ok], return_counts=True)
        ok &= ~np.isin(j, u[c > 1])
        ok[ok] &= sel[j[ok]]
        lut_k = dict(zip(sid[ok], ks[j[ok]]))
        lut_h = dict(zip(sid[ok], h[j[ok]]))
        Ks[q] = np.array([lut_k.get(s, np.nan) for s in D['sid'][q]])
        H[q] = np.array([lut_h.get(s, np.nan) for s in D['sid'][q]])
    return Ks, H


out = {}
for filt in ('f212n', 'f480m'):
    D = C.add_terms(C.load(filt), filt)
    dets = C.DETS[filt]
    nd = len(dets)
    detidx = np.array([dets.index(d) for d in D['det']])
    frame = C.group_index(D['obs'], D['det'], D['exp'])
    for which in ('virac', 'gns'):
        Ks, H = match(D, which)
        hk = H - Ks
        for phot, col in C.PHOT.items():
            y = D[col] - Ks
            base = np.isfinite(y) & np.isfinite(hk) & (D['area'] > 0.5)
            base &= np.abs(y - np.nanmedian(y[base])) < 1.5
            if base.sum() < 1000:
                continue
            res = {}
            for model, term in (('a', D['a']), ('pT', D['pT'])):
                X = np.c_[np.stack([term * (detidx == j) for j in range(nd)], 1), Ks - 15.5, hk - 1.5]
                beta, err, w, sig, yd, Xd = C.fe_fit(y, X, frame, frame, base)
                tiles = np.unique(D['obs'][w])
                XX = np.stack([Xd[w & (D['obs'] == t)].T @ Xd[w & (D['obs'] == t)] for t in tiles])
                Xy = np.stack([Xd[w & (D['obs'] == t)].T @ yd[w & (D['obs'] == t)] for t in tiles])
                bs = []
                for _ in range(300):
                    s = rng.integers(0, len(tiles), len(tiles))
                    bs.append(np.linalg.solve(XX[s].sum(0), Xy[s].sum(0)))
                be = np.std(bs, 0)
                # pooled k: one term column
                Xp = np.c_[term, Ks - 15.5, hk - 1.5]
                bp, ep, wp, _, ydp, Xdp = C.fe_fit(y, Xp, frame, frame, base)
                XXp = np.stack([Xdp[wp & (D['obs'] == t)].T @ Xdp[wp & (D['obs'] == t)] for t in tiles])
                Xyp = np.stack([Xdp[wp & (D['obs'] == t)].T @ ydp[wp & (D['obs'] == t)] for t in tiles])
                bsp = []
                for _ in range(300):
                    s = rng.integers(0, len(tiles), len(tiles))
                    bsp.append(np.linalg.solve(XXp[s].sum(0), Xyp[s].sum(0))[0])
                res[model] = dict(N=int(w.sum()), n_tile=int(len(tiles)), sigma_mmag=sig * 1e3,
                                  k={d: [float(beta[j]), float(be[j])] for j, d in enumerate(dets)},
                                  k_pooled=[float(bp[0]), float(np.std(bsp))],
                                  cKs=float(beta[nd]), cHK=float(beta[nd + 1]))
                print(f"{filt.upper()} vs {which:5s} {C.PHOT_LABEL[phot]:24s} model {model:2s}: N={w.sum()} "
                      f"pooled k={bp[0]:+.3f}±{np.std(bsp):.3f}  per det: " +
                      ' '.join(f"{d[3:]}={beta[j]:+.2f}±{be[j]:.2f}" for j, d in enumerate(dets)) +
                      f"  cKs={beta[nd]:+.3f} cHK={beta[nd+1]:+.3f} sigma={sig*1e3:.0f} mmag", flush=True)
            out[f'{filt}:{which}:{phot}'] = res
json.dump(out, open('test3.json', 'w'), indent=1)
