"""#362: does the per-detector pattern enter at the merge?  Per-frame hybrid-PSF magnitude (AB,
frame table) against the merged hybpsf m7 catalogue magnitude (Vega) of the same star.

d = mag_vega_<filt>(merged hybpsf m7) - m_PSF,frame (AB): one constant (AB-Vega + nothing else) if
the merge only averages; any per-detector structure is added by the merge step.  Per detector:
median over tiles of the per-(tile,det) medians, centred; bootstrap over tiles.  Matching: frame
rows carry the tile star-list position; merged skycoord_<filt> after a median bulk offset, 0.06".
Usage: AREA375_WORKDIR=<workdir> python t6_merge_vs_frame.py <catalog_dir>  -> t6.json, t6_*.png
"""
import glob
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from astropy.table import Table  # noqa: E402
from scipy.spatial import cKDTree  # noqa: E402
from scipy.stats import binned_statistic_2d  # noqa: E402

A375 = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'area_375')
sys.path.insert(0, A375)
import common as C  # noqa: E402

CEN = json.load(open(os.path.join(A375, 'centres.json')))
CATDIR = sys.argv[1]
rng = np.random.default_rng(6)


def F(c):
    return np.asarray(np.ma.filled(np.ma.asarray(c, dtype=float), np.nan), float)


def merged_cat(obs):
    # hybpsf_m7 first.  m8_dedup is only for the #362 s13zp_hyb snapshot directory, whose files are
    # hybpsf m7 merged catalogues renamed m8_dedup (its manifest.json records each `src`); a live
    # m8_dedup from the 10-17..20 rebuilds is STPSF-filled (jwst-gc-pipeline #1214) and is the wrong input.
    for tok in ('hybpsf_m7', 'm8_dedup'):
        fs = sorted(glob.glob(os.path.join(CATDIR, f'basic_merged_indivexp_photometry_tables_merged_resbgsub_{tok}_o{obs}.fits')))
        if fs:
            return fs[0]
    return None


def sc_cols(t, filt):
    if f'skycoord_{filt}' in t.colnames:
        s = t[f'skycoord_{filt}']
        return F(s.ra.deg), F(s.dec.deg)
    return F(t[f'skycoord_{filt}.ra']), F(t[f'skycoord_{filt}.dec'])


out = {}
for filt in ('f212n', 'f480m'):
    D = C.load(filt)
    dets = C.DETS[filt]
    dm = np.full(len(D['x']), np.nan)
    srcs = {}
    for obs in np.unique(D['obs']):
        o = f'{obs:03d}'
        f = merged_cat(o)
        if f is None:
            continue
        srcs[o] = os.path.basename(f)
        t = Table.read(f)
        ra, dec = sc_cols(t, filt)
        mv = F(t[f'mag_vega_{filt}'])
        ok = np.isfinite(ra) & np.isfinite(dec) & np.isfinite(mv)
        ra0, de0 = CEN[o][0], CEN[o][1]
        cd = np.cos(np.radians(de0))
        R = np.c_[(ra[ok] - ra0) * cd * 3600, (dec[ok] - de0) * 3600]
        mv = mv[ok]
        q = np.flatnonzero(D['obs'] == obs)
        P = np.c_[D['X'][q], D['Y'][q]] * 60
        tr = cKDTree(R)
        d0, j0 = tr.query(P, distance_upper_bound=0.3)
        g = np.isfinite(d0)
        off = np.median(P[g] - R[j0[g]], axis=0)
        d, j = tr.query(P - off, distance_upper_bound=0.06)
        g = np.isfinite(d)
        dm[q[g]] = mv[j[g]] - D['mhyb'][q[g]]
    base = np.isfinite(dm) & (D['snr'] > 50)
    base &= np.abs(dm - np.nanmedian(dm[base])) < 0.5
    tiles = np.unique(D['obs'][base])
    M = np.full((len(tiles), len(dets)), np.nan)
    for i, tl in enumerate(tiles):
        qt = base & (D['obs'] == tl)
        for k, det in enumerate(dets):
            qq = qt & (D['det'] == det)
            if qq.sum() >= 30:
                M[i, k] = np.median(dm[qq])

    def centred(MM):
        v = np.nanmedian(MM, 0)
        return v - np.nanmean(v)
    v = centred(M)
    bs = np.array([centred(M[rng.integers(0, len(tiles), len(tiles))]) for _ in range(300)])
    res = dict(n=int(base.sum()), n_tile=int(len(tiles)), const=float(np.nanmedian(M)),
               det={det: [float(v[k]), float(np.std(bs[:, k]))] for k, det in enumerate(dets)},
               tile_det_rms=float(np.nanstd(M - np.nanmedian(M, 0))), sources=srcs)
    out[filt] = res
    print(filt.upper(), 'N', res['n'], 'tiles', res['n_tile'], f"const {res['const']:+.4f}", flush=True)
    print('  merged(Vega) - frame hybrid PSF(AB), centred, mmag: ' +
          ' '.join(f"{d[3:].upper()}={res['det'][d][0] * 1e3:+.1f}±{res['det'][d][1] * 1e3:.1f}" for d in dets), flush=True)
    bx = np.arange(-1.5, 1.51, 0.15)
    by = np.arange(-2.4, 2.41, 0.15)
    r = dm[base] - np.median(dm[base])
    med = binned_statistic_2d(D['X'][base], D['Y'][base], r, 'median', bins=[bx, by]).statistic
    n = binned_statistic_2d(D['X'][base], D['Y'][base], r, 'count', bins=[bx, by]).statistic
    med[n < 30] = np.nan
    fig, ax = plt.subplots(figsize=(6, 8), constrained_layout=True)
    im = ax.imshow(med.T * 1e3, origin='lower', extent=[bx[0], bx[-1], by[0], by[-1]], cmap='RdBu_r',
                   vmin=-40, vmax=40, aspect='equal')
    ax.set_xlabel('east offset from tile centre (arcmin)')
    ax.set_ylabel('north offset from tile centre (arcmin)')
    ax.set_title(f'{filt.upper()}: merged hybpsf m7 (Vega) − per-frame hybrid PSF (AB)\n'
                 f'68 tiles stacked, 0.15′ cells; median removed', fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.7, label='mmag')
    fig.savefig(os.path.join(C.WORK, f't6_{filt}_merge_vs_frame.png'), dpi=110)
    plt.close(fig)
json.dump(out, open(os.path.join(C.WORK, 't6.json'), 'w'), indent=1)
