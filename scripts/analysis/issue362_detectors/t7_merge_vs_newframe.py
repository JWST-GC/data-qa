"""#362: does the merge add per-detector structure?  Catalogue-only version of t6 that reads the
CURRENT per-frame hybpsf m7 catalogues (the merge inputs) instead of the 10-17 frame tables.

d = mag_vega_<filt>(merged hybpsf m7) - m_frame,  m_frame = -2.5 log10(flux_fit * PIXAR_SR * 1e6) + 8.90
(AB, PIXAR_SR from the crf SCI header, as build_frames.py).  A pure inverse-variance average gives
one constant (AB-Vega); per-detector structure in d is added between the per-frame file and the
merged column.  Matching: per-frame skycoord_centroid -> merged skycoord_<filt>, bulk offset then 0.06".
Usage: python t7_merge_vs_newframe.py <catalog_dir> <out_dir> obs [obs ...]
"""
import glob
import json
import os
import sys

import numpy as np
from astropy.io import fits
from astropy.table import Table
from scipy.spatial import cKDTree

CATDIR, OUT = sys.argv[1], sys.argv[2]
OBS = sys.argv[3:]
B = '/orange/adamginsburg/jwst/gc-treasury/'
DETS = {'f212n': [f'nrc{m}{n}' for m in 'ab' for n in range(1, 5)], 'f480m': ['nrcalong', 'nrcblong']}


def F(c):
    return np.asarray(np.ma.filled(np.ma.asarray(c, dtype=float), np.nan), float)


def sc(s):
    return F(s.ra.deg), F(s.dec.deg)


out = {}
for filt in ('f212n', 'f480m'):
    rows = []   # (obs, det, d) per (tile, det) median
    for o in OBS:
        mf = os.path.join(CATDIR, f'basic_merged_indivexp_photometry_tables_merged_resbgsub_hybpsf_m7_o{o}.fits')
        if not os.path.exists(mf):
            continue
        t = Table.read(mf)
        ra, dec = sc(t[f'skycoord_{filt}'])
        mv = F(t[f'mag_vega_{filt}'])
        ok = np.isfinite(ra) & np.isfinite(dec) & np.isfinite(mv)
        ra0, de0 = np.nanmedian(ra[ok]), np.nanmedian(dec[ok])
        cd = np.cos(np.radians(de0))
        R = np.c_[(ra[ok] - ra0) * cd * 3600, (dec[ok] - de0) * 3600]
        mv = mv[ok]
        tr = cKDTree(R)
        per = {}
        for f in sorted(glob.glob(B + f'{filt.upper()}/{filt}_nrc*_o{o}_visit*_vgroup*_exp*_resbgsub_hybpsf_m7_daophot_basic.fits')):
            ft = Table.read(f)
            det = ft.meta['DETECTOR'].lower()
            with fits.open(ft.meta['FILENAME'].replace('//', '/')) as h:
                pixar = float(h['SCI'].header['PIXAR_SR'])
            fl, ef = F(ft['flux_fit']), F(ft['flux_err'])
            fra, fde = sc(ft['skycoord_centroid'])
            g = np.isfinite(fl) & (fl > 0) & np.isfinite(fra) & (fl / ef > 50)
            if g.sum() < 30:
                continue
            P = np.c_[(fra[g] - ra0) * cd * 3600, (fde[g] - de0) * 3600]
            m = -2.5 * np.log10(fl[g] * pixar * 1e6) + 8.90
            d0, j0 = tr.query(P, distance_upper_bound=0.3)
            k = np.isfinite(d0)
            off = np.median(P[k] - R[j0[k]], axis=0)
            d, j = tr.query(P - off, distance_upper_bound=0.06)
            k = np.isfinite(d)
            per.setdefault(det, []).append(mv[j[k]] - m[k])
        for det, ds in per.items():
            dd = np.concatenate(ds)
            dd = dd[np.abs(dd - np.median(dd)) < 0.5]
            if len(dd) >= 30:
                rows.append((o, det, float(np.median(dd)), len(dd)))
        print(filt, o, {r[1]: round(r[2], 4) for r in rows if r[0] == o}, flush=True)
    dets = DETS[filt]
    tiles = sorted({r[0] for r in rows})
    M = np.full((len(tiles), len(dets)), np.nan)
    for (o, det, v, n) in rows:
        if det in dets:
            M[tiles.index(o), dets.index(det)] = v
    v = np.nanmedian(M, 0)
    vc = v - np.nanmean(v)
    rng = np.random.default_rng(7)
    bs = []
    for _ in range(300):
        Mb = M[rng.integers(0, len(tiles), len(tiles))]
        vb = np.nanmedian(Mb, 0)
        bs.append(vb - np.nanmean(vb))
    bs = np.array(bs)
    out[filt] = dict(tiles=tiles, const=float(np.nanmean(v)),
                     det={d: [float(vc[k]), float(np.std(bs[:, k]))] for k, d in enumerate(dets)},
                     tile_det_rms=float(np.nanstd(M - v)))
    print(filt.upper(), 'tiles', len(tiles), f"const {np.nanmean(v):+.4f}  merged(Vega) - new frame hybrid(AB), centred mmag: " +
          ' '.join(f"{d[3:].upper()}={vc[k] * 1e3:+.1f}±{np.std(bs[:, k]) * 1e3:.1f}" for k, d in enumerate(dets)),
          f"tile_det_rms {out[filt]['tile_det_rms'] * 1e3:.1f}", flush=True)
json.dump(out, open(os.path.join(OUT, 't7.json'), 'w'), indent=1)
