"""Shared helpers for the #375 analysis: load the per-frame tables, model terms, two-way
fixed-effects regression, detector binning with a shuffled null."""
import glob
import os

import numpy as np
from scipy.interpolate import RegularGridInterpolator

HERE = os.path.dirname(os.path.abspath(__file__))
# Intermediate products (frames/, *.npz, *.json, net_*/) go to AREA375_WORKDIR, default the cwd.
WORK = os.environ.get('AREA375_WORKDIR', os.getcwd())
DETS = {'f212n': ['nrca1', 'nrca2', 'nrca3', 'nrca4', 'nrcb1', 'nrcb2', 'nrcb3', 'nrcb4'],
        'f480m': ['nrcalong', 'nrcblong']}
PHOT = {'psf': 'mpsf', 'hyb': 'mhyb', 'ap1': 'map1', 'ap2': 'map2'}
PHOT_LABEL = {'psf': 'PSF (STPSF grid, m7)', 'hyb': 'PSF (hybrid grid, m7)',
              'ap1': 'aperture r1 on crf', 'ap2': 'aperture r2 on crf'}


def load(filt):
    fs = sorted(glob.glob(os.path.join(WORK, 'frames', f'{filt}_o*.npz')))
    parts = [dict(np.load(f)) for f in fs]
    D = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    D['det'] = D['det'].astype(str)
    return D


_NODES = None


def grid_sum_term(filt, det, x, y, assign):
    """s = 2.5 log10(sum of the STPSF grid ePSF) at detector (x, y), bilinear between the 4x4
    nodes (clamped at the outer nodes, as GriddedPSFModel does), for plane assignment
    'computed' (post-#1154) or 'transposed' (pre-#1154 to_griddedpsfmodel)."""
    global _NODES
    if _NODES is None:
        _NODES = dict(np.load(os.path.join(WORK, 't1_psfgrid_nodes.npz')))
    n = _NODES[f'{filt.upper()}:{det}:{assign}']
    xs, ys = np.unique(n[:, 0]), np.unique(n[:, 1])
    g = np.full((len(ys), len(xs)), np.nan)
    for xx, yy, s in n:
        g[np.searchsorted(ys, yy), np.searchsorted(xs, xx)] = s
    f = RegularGridInterpolator((ys, xs), g)
    return f(np.c_[np.clip(y, ys[0], ys[-1]), np.clip(x, xs[0], xs[-1])])


def add_terms(D, filt):
    """a = 2.5 log10 AREA (positive where pixels are large; k=1 means the term is fully missing,
    m_frame = m_true + k a).  sT/sC = grid-sum term with transposed / computed plane pairing.
    pT = a + sT: the error a PSF fit with the pre-#1154 grid carries (m_fit = m_true + a + s)."""
    D['a'] = 2.5 * np.log10(D['area'])
    D['a_ap'] = 2.5 * np.log10(D['area_ap'])
    D['sT'] = np.zeros(len(D['x']))
    D['sC'] = np.zeros(len(D['x']))
    for det in DETS[filt]:
        q = D['det'] == det
        D['sT'][q] = grid_sum_term(filt, det, D['x'][q], D['y'][q], 'transposed')
        D['sC'][q] = grid_sum_term(filt, det, D['x'][q], D['y'][q], 'computed')
    D['pT'] = D['a'] + D['sT']
    D['pC'] = D['a'] + D['sC']
    return D


def group_index(*keys):
    k = np.rec.fromarrays(keys)
    _, inv = np.unique(k, return_inverse=True)
    return inv.ravel()


def demean2(M, g1, g2, w, niter=50, tol=1e-9):
    """Two-way within transform (alternating projections) of the columns of M over groups g1
    (star) and g2 (frame), using only rows with w=True (others returned as NaN)."""
    M = np.where(w[:, None], M, 0.0).astype(float)
    n1, n2 = g1.max() + 1, g2.max() + 1
    c1 = np.bincount(g1, w, n1)
    c2 = np.bincount(g2, w, n2)
    c1[c1 == 0] = np.inf   # empty groups: mean 0 (rows are masked anyway)
    c2[c2 == 0] = np.inf
    for _ in range(niter):
        old = M.copy()
        for j in range(M.shape[1]):
            M[:, j] -= (np.bincount(g1, M[:, j] * w, n1) / c1)[g1]
            M[:, j] -= (np.bincount(g2, M[:, j] * w, n2) / c2)[g2]
        M = np.where(w[:, None], M, 0.0)
        if np.nanmax(np.abs(M - old)) < tol:
            break
    M[~w] = np.nan
    return M


def fe_fit(y, X, g1, g2, base, clip=4.0, niter=4):
    """y = mu_star + z_frame + X beta.  Robust (4 sigma MAD clip); returns beta, formal err,
    kept mask, sigma, and the within-transformed y and X (on the kept rows)."""
    w = base.copy()
    for _ in range(niter):
        M = demean2(np.c_[y, X], g1, g2, w)
        yd, Xd = M[:, 0], M[:, 1:]
        beta, *_ = np.linalg.lstsq(Xd[w], yd[w], rcond=None)
        r = yd - Xd @ beta
        sig = 1.4826 * np.nanmedian(np.abs(r[w]))
        neww = base & np.isfinite(r) & (np.abs(r) < clip * sig)
        if np.array_equal(neww, w):
            break
        w = neww
    M = demean2(np.c_[y, X], g1, g2, w)
    yd, Xd = M[:, 0], M[:, 1:]
    beta, *_ = np.linalg.lstsq(Xd[w], yd[w], rcond=None)
    r = yd - Xd @ beta
    sig = float(np.std(r[w]))
    cov = np.linalg.inv(Xd[w].T @ Xd[w]) * sig ** 2
    return beta, np.sqrt(np.diag(cov)), w, sig, yd, Xd


def cell_index(x, y, nb, size=2048):
    ix = np.clip((x / size * nb).astype(int), 0, nb - 1)
    iy = np.clip((y / size * nb).astype(int), 0, nb - 1)
    return iy * nb + ix


def binned(v, cell, nb, minn=30):
    cnt = np.bincount(cell, minlength=nb * nb)
    order = np.argsort(cell, kind='stable')
    bounds = np.searchsorted(cell[order], np.arange(nb * nb + 1))
    vs = v[order]
    med = np.full(nb * nb, np.nan)
    for c in range(nb * nb):
        if cnt[c] >= minn:
            med[c] = np.median(vs[bounds[c]:bounds[c + 1]])
    return med.reshape(nb, nb)


def map_stats(v, x, y, nb=8, n_perm=10, seed=0):
    """RMS (mmag) of the nb x nb binned median map of v, its shuffled-position null and the
    debiased RMS sqrt(rms^2 - null^2)."""
    cell = cell_index(x, y, nb)
    m = binned(v, cell, nb)
    rms = float(np.sqrt(np.nanmean(m ** 2)) * 1e3)
    rng = np.random.default_rng(seed)
    nulls = [float(np.sqrt(np.nanmean(binned(v[rng.permutation(len(v))], cell, nb) ** 2)) * 1e3)
             for _ in range(n_perm)]
    null = float(np.median(nulls))
    return dict(rms=rms, null=null, deb=float(np.sqrt(max(rms ** 2 - null ** 2, 0.0))))
