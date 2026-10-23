"""#362 open item 2: per-SW-detector offsets.  Per-frame hybrid-PSF (post-#1154) photometry against
aperture photometry of the same stars on the same crf frame, and each against VIRAC2 Ks.

Frame tables: build_frames.py (area_375) rebuilt 2026-10-22 on the current hybpsf m7 per-frame fits.
All frame magnitudes are AB (PIXAR_SR, +8.90); aperture fluxes are multiplied by the mean AREA in r1
(m_ap,true = m_ap - 2.5 log10 AREA), the per-pixel solid-angle correction the PSF chain applies.

(A) internal: d = m_PSF - m_ap,true per row; per detector the median over tiles of the per-(tile,det)
    medians, minus the mean over detectors (the absolute aperture correction is one constant).
(B) external: y = m - Ks = z_(tile,exposure) + z_det + cKs (Ks-15.5) + cHK (H-Ks-1.5), robust fixed-
    effects fit (common.fe_fit); z_det re-centred to zero mean over the detectors of the filter.
    Errors: bootstrap over tiles (300).
(C) tile-frame stacked maps of (A) in the #362 figure-C cells.
Usage: AREA375_WORKDIR=<workdir> python t5_detector_offsets.py   -> t5.json, t5_*.png
"""
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from astropy.io import fits  # noqa: E402
from astropy.wcs import WCS  # noqa: E402
from scipy.stats import binned_statistic_2d  # noqa: E402

A375 = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'area_375')
sys.path.insert(0, A375)
import common as C  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import t3_lib as T3  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CEN = json.load(open(os.path.join(A375, 'centres.json')))
FIGC = json.load(open(os.path.join(HERE, 'results', 'figC_detectors.json')))
PIPE = '/orange/adamginsburg/jwst/gc-treasury/{F}/pipeline/jw10678066001_02101_00001_{det}_destreak_o066_crf.fits'
rng = np.random.default_rng(5)
NB = 300


def det_polys(filt):
    c = CEN['066']
    out = {}
    for det in C.DETS[filt]:
        w = WCS(fits.getheader(PIPE.format(F=filt.upper(), det=det), 'SCI'))
        fp = w.calc_footprint()
        out[det] = np.c_[(fp[:, 0] - c[0]) * np.cos(np.radians(c[1])) * 60, (fp[:, 1] - c[1]) * 60]
    return out


def internal(D, filt, mcol, apcol, base):
    """Per-detector median offset of m_PSF - m_ap,true; bootstrap over tiles."""
    d = D[mcol] - (D[apcol] - D['a_ap'])
    ok = base & np.isfinite(d)
    dets = C.DETS[filt]
    tiles = np.unique(D['obs'][ok])
    M = np.full((len(tiles), len(dets)), np.nan)
    for i, t in enumerate(tiles):
        qt = ok & (D['obs'] == t)
        for j, det in enumerate(dets):
            q = qt & (D['det'] == det)
            if q.sum() >= 30:
                M[i, j] = np.median(d[q])
    def centred(MM):
        v = np.nanmedian(MM, 0)
        return v - np.nanmean(v)
    v = centred(M)
    bs = np.array([centred(M[rng.integers(0, len(tiles), len(tiles))]) for _ in range(NB)])
    return {det: [float(v[j]), float(np.std(bs[:, j])), int(np.isfinite(M[:, j]).sum())]
            for j, det in enumerate(dets)}, float(np.nanmedian(M)), d, ok


def external(D, filt, y, base, Ks, hk):
    dets = C.DETS[filt]
    nd = len(dets)
    detidx = np.array([dets.index(x) for x in D['det']])
    grp = C.group_index(D['obs'], D['exp'])
    X = np.c_[np.stack([(detidx == j).astype(float) for j in range(1, nd)], 1), Ks - 15.5, hk - 1.5]
    beta, err, w, sig, yd, Xd = C.fe_fit(y, X, grp, grp, base)
    tiles = np.unique(D['obs'][w])
    XX = np.stack([Xd[w & (D['obs'] == t)].T @ Xd[w & (D['obs'] == t)] for t in tiles])
    Xy = np.stack([Xd[w & (D['obs'] == t)].T @ yd[w & (D['obs'] == t)] for t in tiles])

    def zdet(b):
        z = np.r_[0.0, b[:nd - 1]]
        return z - z.mean()
    z = zdet(beta)
    bs = np.array([zdet(np.linalg.solve(XX[s].sum(0), Xy[s].sum(0)))
                   for s in (rng.integers(0, len(tiles), len(tiles)) for _ in range(NB))])
    return dict(N=int(w.sum()), n_tile=int(len(tiles)), sigma_mmag=sig * 1e3,
                z={det: [float(z[j]), float(np.std(bs[:, j]))] for j, det in enumerate(dets)},
                cKs=float(beta[nd - 1]), cHK=float(beta[nd]))


def figc_ref(filt):
    v = {det: FIGC[f'{filt}:hyb:{det}'][0] for det in C.DETS[filt]}
    mu = np.mean(list(v.values()))
    return {det: v[det] - mu for det in v}


out = {}
for filt in ('f212n', 'f480m'):
    D = C.add_terms(C.load(filt), filt)
    dets = C.DETS[filt]
    base = (D['area'] > 0.5) & (D['snr'] > 50)
    res = dict(n_rows=int(len(D['x'])), n_tiles=int(len(np.unique(D['obs']))))
    # (A) internal
    for mcol, mlab in (('mhyb', 'hyb'), ('mpsf', 'psf')):
        for apcol in ('map1', 'map2'):
            v, zero, d, ok = internal(D, filt, mcol, apcol, base)
            res[f'int:{mlab}-{apcol}'] = dict(det=v, median_abs=zero)
            if mlab == 'hyb' and apcol == 'map2':
                dA, okA = d, ok
    # (B) external
    Ks, H = T3.match(D, 'virac')
    hk = H - Ks
    for col, lab in (('mhyb', 'hyb'), ('mpsf', 'psf'), ('map1', 'ap1'), ('map2', 'ap2')):
        y = D[col] - Ks
        if lab.startswith('ap'):
            y = y - D['a_ap']
        b = base & np.isfinite(y) & np.isfinite(hk)
        b &= np.abs(y - np.nanmedian(y[b])) < 1.5
        res[f'ext:{lab}'] = external(D, filt, y, b, Ks, hk)
    res['figC_hyb_centred'] = figc_ref(filt)
    out[filt] = res
    print(filt.upper(), flush=True)
    hdr = f"{'':26s}" + ''.join(f'{d[3:].upper():>9s}' for d in dets)
    print(hdr)
    print(f"{'#362 figC hyb (vs Ks)':26s}" + ''.join(f'{res["figC_hyb_centred"][d] * 1e3:+9.1f}' for d in dets))
    for k in ('ext:hyb', 'ext:psf', 'ext:ap1', 'ext:ap2'):
        print(f'{k + " (vs Ks)":26s}' + ''.join(f'{res[k]["z"][d][0] * 1e3:+6.1f}±{res[k]["z"][d][1] * 1e3:<3.1f}'[:9].rjust(9)
                                              for d in dets))
    for k in ('int:hyb-map1', 'int:hyb-map2', 'int:psf-map1', 'int:psf-map2'):
        print(f'{k + " (internal)":26s}' + ''.join(f'{res[k]["det"][d][0] * 1e3:+9.1f}' for d in dets))

    # (C) figure: tile-frame maps of hybrid PSF - aperture r2 (internal) + bar chart
    bx = np.arange(-1.5, 1.51, 0.15)
    by = np.arange(-2.4, 2.41, 0.15)
    x, y_, r = D['X'][okA], D['Y'][okA], dA[okA]
    r = r - np.median(r)
    med = binned_statistic_2d(x, y_, r, 'median', bins=[bx, by]).statistic
    n = binned_statistic_2d(x, y_, r, 'count', bins=[bx, by]).statistic
    med[n < 30] = np.nan
    res['map_rms_mmag'] = float(np.nanstd(med) * 1e3)
    fig = plt.figure(figsize=(14, 7.5), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.6])
    ax = fig.add_subplot(gs[0])
    im = ax.imshow(med.T * 1e3, origin='lower', extent=[bx[0], bx[-1], by[0], by[-1]], cmap='RdBu_r',
                   vmin=-40, vmax=40, aspect='equal')
    for det, p in det_polys(filt).items():
        ax.add_patch(plt.Polygon(p, fill=False, ec='k', lw=0.8))
        ax.text(*p.mean(0), f'{det.replace("nrc", "").upper()}\n{res["int:hyb-map2"]["det"][det][0] * 1e3:+.0f}',
                ha='center', va='center', fontsize=8, bbox=dict(fc='w', alpha=0.6, lw=0))
    ax.set_xlabel('east offset from tile centre (arcmin)')
    ax.set_ylabel('north offset from tile centre (arcmin)')
    ax.set_title(f'{filt.upper()}: hybrid PSF − aperture r2 (×AREA), same crf frame\n'
                 f'68 tiles stacked, 0.15′ cells, binned-map RMS {res["map_rms_mmag"]:.1f} mmag', fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.7, label='m_PSF − m_ap (both AB, same frame; mmag, median removed)')
    ax2 = fig.add_subplot(gs[1])
    series = [('figC', '#362 fig C: hybrid m7 merged − Ks (Vega)', None),
              ('ext:hyb', 'per-frame hybrid PSF − Ks', 'z'),
              ('ext:ap2', 'per-frame aperture r2 ×AREA − Ks', 'z'),
              ('ext:ap1', 'per-frame aperture r1 ×AREA − Ks', 'z'),
              ('int:hyb-map2', 'hybrid PSF − aperture r2 (internal)', 'det'),
              ('int:hyb-map1', 'hybrid PSF − aperture r1 (internal)', 'det')]
    wbar = 0.8 / len(series)
    xi = np.arange(len(dets))
    for s, (k, lab, kk) in enumerate(series):
        if k == 'figC':
            v = np.array([res['figC_hyb_centred'][d] for d in dets])
            e = np.array([FIGC[f'{filt}:hyb:{d}'][1] for d in dets])
        else:
            v = np.array([res[k][kk][d][0] for d in dets])
            e = np.array([res[k][kk][d][1] for d in dets])
        ax2.bar(xi + (s - len(series) / 2 + 0.5) * wbar, v * 1e3, wbar, yerr=e * 1e3, label=lab, capsize=2)
    ax2.axhline(0, color='k', lw=0.6)
    ax2.set_xticks(xi, [d[3:].upper() for d in dets])
    ax2.set_ylabel('per-detector offset, mean over detectors removed (mmag; + = fainter)')
    ax2.set_title(f'{filt.upper()} per-detector offsets.  F − Ks: F in AB (frames) or Vega (fig C), Ks Vega; '
                  'colour terms + (tile, exposure) offsets fitted\nerrors: bootstrap over tiles', fontsize=9)
    ax2.legend(fontsize=8)
    fig.suptitle('#362: per-detector offsets — hybrid PSF (post-#1154) vs crf aperture photometry, 10678', fontsize=11)
    fig.savefig(os.path.join(C.WORK, f't5_{filt}_detector_offsets.png'), dpi=110)
    plt.close(fig)

json.dump(out, open(os.path.join(C.WORK, 't5.json'), 'w'), indent=1)
