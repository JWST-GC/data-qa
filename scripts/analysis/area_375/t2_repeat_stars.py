"""Test 2 (#375): per-frame magnitude minus tile consensus against the pixel-area term.

Model per filter and photometry type (PSF STPSF m7, PSF hybrid m7, aperture r1, aperture r2):
    m_i = mu_star + z_frame + sum_det [k_a,det * a_i + k_s,det * sT_i] * 1(det_i = det)
with a = 2.5 log10 AREA(x, y) and sT the STPSF grid-sum term under the pre-#1154 (transposed)
plane pairing.  mu_star is the tile consensus (one free magnitude per star; stars with >= 3 frames
after a 0.2 mag flux vet), z_frame one zero point per exposure and detector.  The fixed effects are
removed by the two-way within transform, so k is measured from frame-to-frame differences of the
same star only (no dilution: the regressor is differenced the same way).  k_a = 1: the area term
is fully missing; k_a = 0: already applied.

Models reported:  'a'  (k_a only, per detector and pooled)            -- the issue's k
                  'pT' (one k on pT = a + sT: the pre-#1154 PSF error)
                  'a+sT' (both, per detector)
Errors: bootstrap over tiles (tiles are independent: every fixed effect is tile-specific).
Cross-validation: tiles split into two halves (alternate obs ids); k fitted on one half, binned
8x8 detector residual RMS (mmag) of the other half before / after, against a shuffled null.
All magnitudes are frame-to-frame differences of one system (AB from PIXAR_SR); constants cancel.

Usage: python t2_repeat_stars.py   -> test2.json, test2_maps.npz
"""
import json
import os

import numpy as np

import common as C

NBOOT = int(os.environ.get('NBOOT', 1000))
SUFFIX = os.environ.get('SUFFIX', '')
rng = np.random.default_rng(1)


def prep(D, col):
    y = D[col].copy()
    star = C.group_index(D['obs'], D['sid'])
    frame = C.group_index(D['obs'], D['det'], D['exp'])
    ok = np.isfinite(y) & (D['area'] > 0.5) & np.isfinite(D['a'])
    # flux vet against the star median, then >= 3 frames per star
    for _ in range(2):
        idx = np.flatnonzero(ok)
        order = np.argsort(star[idx], kind='stable')
        si = star[idx][order]
        yi = y[idx][order]
        bounds = np.flatnonzero(np.r_[True, si[1:] != si[:-1], True])
        med = np.empty(len(si))
        for b0, b1 in zip(bounds[:-1], bounds[1:]):
            med[b0:b1] = np.median(yi[b0:b1])
        bad = np.abs(yi - med) > 0.2
        ok[idx[order][bad]] = False
        n = np.bincount(star[ok], minlength=star.max() + 1)
        ok &= n[star] >= 3
    return y, star, frame, ok


def normal_eq(Xd, yd, w, tile):
    """Per-tile X'X and X'y over kept rows."""
    tiles = np.unique(tile[w])
    XX = np.zeros((len(tiles), Xd.shape[1], Xd.shape[1]))
    Xy = np.zeros((len(tiles), Xd.shape[1]))
    for i, t in enumerate(tiles):
        q = w & (tile == t)
        XX[i] = Xd[q].T @ Xd[q]
        Xy[i] = Xd[q].T @ yd[q]
    return tiles, XX, Xy


def solve(XX, Xy, sel, cols):
    A = XX[sel][:, cols][:, :, cols].sum(0)
    b = Xy[sel][:, cols].sum(0)
    return np.linalg.solve(A, b)


def boot(XX, Xy, cols):
    nt = XX.shape[0]
    bs = np.array([solve(XX, Xy, rng.integers(0, nt, nt), cols) for _ in range(NBOOT)])
    return bs.std(0)


out, maps = {}, {}
for filt in ('f212n', 'f480m'):
    D = C.add_terms(C.load(filt), filt)
    dets = C.DETS[filt]
    nd = len(dets)
    detidx = np.array([dets.index(d) for d in D['det']])
    Xa = np.stack([D['a'] * (detidx == j) for j in range(nd)], 1)
    Xs = np.stack([D['sT'] * (detidx == j) for j in range(nd)], 1)
    Xfull = np.c_[Xa, Xs]
    tile = D['obs']
    tl = np.unique(tile)
    halfA = np.isin(tile, tl[0::2])
    for phot, col in C.PHOT.items():
        y, star, frame, base = prep(D, col)
        if base.sum() < 1000:
            continue
        beta, err, w, sig, yd, Xd = C.fe_fit(y, Xfull, star, frame, base)
        tiles, XX, Xy = normal_eq(Xd, yd, w, tile)
        allt = np.arange(len(tiles))
        ia, isx = list(range(nd)), list(range(nd, 2 * nd))
        res = dict(N=int(w.sum()), n_star=int(len(np.unique(star[w]))), n_tile=int(len(tiles)),
                   sigma_mmag=sig * 1e3)
        # model 'a' per detector and pooled
        ka = solve(XX, Xy, allt, ia)
        ka_e = boot(XX, Xy, ia)
        # pooled: a single column = sum of the per-detector a columns
        XXp = np.stack([[[XX[t][np.ix_(ia, ia)].sum()]] for t in allt])
        Xyp = np.stack([[Xy[t][ia].sum()] for t in allt])
        kpool = solve(XXp, Xyp, allt, [0])[0]
        kpool_e = boot(XXp, Xyp, [0])[0]
        # model 'pT': pT = a + sT, a single column per detector = a-col + s-col
        T = np.zeros((2 * nd, nd))
        for j in range(nd):
            T[j, j] = T[nd + j, j] = 1.0
        XXt = np.einsum('ij,tik,kl->tjl', T, XX, T)
        Xyt = Xy @ T
        kp = solve(XXt, Xyt, allt, list(range(nd)))
        kp_e = boot(XXt, Xyt, list(range(nd)))
        Tp = T.sum(1, keepdims=True)
        XXtp = np.einsum('ij,tik,kl->tjl', Tp, XX, Tp)
        kpp = solve(XXtp, Xy @ Tp, allt, [0])[0]
        kpp_e = boot(XXtp, Xy @ Tp, [0])[0]
        # model 'a+sT'
        kas = solve(XX, Xy, allt, ia + isx)
        kas_e = boot(XX, Xy, ia + isx)
        # dilution: variance of the within-transformed a relative to the raw a (per detector)
        dil = {}
        for j, d in enumerate(dets):
            q = w & (detidx == j)
            dil[d] = float(np.std(Xd[q, j]) / np.std(D['a'][q]))
        # cross-validation over tile halves
        isA = np.isin(tiles, np.unique(tile[halfA]))
        cv = {}
        for d_j, d in enumerate(dets):
            cv[d] = {}
        stats = {lab: {d: [] for d in dets} for lab in ('before', 'k1_area', 'cv_a', 'cv_pT')}
        kcv = {}
        for train, test, lab in ((isA, ~isA, 'A->B'), (~isA, isA, 'B->A')):
            k_a = solve(XX, Xy, np.flatnonzero(train), ia)
            k_p = solve(XXt, Xyt, np.flatnonzero(train), list(range(nd)))
            kcv[lab] = dict(k_a=k_a.tolist(), k_pT=k_p.tolist())
            rows_test = w & np.isin(tile, tiles[test])
            for j, d in enumerate(dets):
                q = rows_test & (detidx == j)
                yy, aa, pp = yd[q], Xd[q, j], Xd[q, j] + Xd[q, nd + j]
                for slab, v in (('before', yy), ('k1_area', yy - aa), ('cv_a', yy - k_a[j] * aa),
                                ('cv_pT', yy - k_p[j] * pp)):
                    stats[slab][d].append(C.map_stats(v, D['x'][q], D['y'][q]))
        mstats = {slab: {d: {k: float(np.mean([s[k] for s in v])) for k in ('rms', 'null', 'deb')}
                         for d, v in sv.items()} for slab, sv in stats.items()}
        res.update(k_a={d: [float(ka[j]), float(ka_e[j])] for j, d in enumerate(dets)},
                   k_a_formal_err={d: float(err[j]) for j, d in enumerate(dets)},
                   k_a_pooled=[float(kpool), float(kpool_e)],
                   k_pT={d: [float(kp[j]), float(kp_e[j])] for j, d in enumerate(dets)},
                   k_pT_pooled=[float(kpp), float(kpp_e)],
                   k_a_sT={d: dict(k_a=[float(kas[j]), float(kas_e[j])],
                                   k_s=[float(kas[nd + j]), float(kas_e[nd + j])])
                           for j, d in enumerate(dets)},
                   dilution=dil, cv_k=kcv, binned_rms_mmag=mstats)
        out[f'{filt}:{phot}'] = res
        # 16x16 maps per detector for the figure (all tiles)
        for j, d in enumerate(dets):
            q = w & (detidx == j)
            cell = C.cell_index(D['x'][q], D['y'][q], 16)
            maps[f'{filt}:{phot}:{d}:obs'] = C.binned(yd[q], cell, 16)
            maps[f'{filt}:{phot}:{d}:pred_a1'] = C.binned(Xd[q, j], cell, 16)
            maps[f'{filt}:{phot}:{d}:pred_pT1'] = C.binned(Xd[q, j] + Xd[q, nd + j], cell, 16)
            maps[f'{filt}:{phot}:{d}:resid_a'] = C.binned(yd[q] - ka[j] * Xd[q, j], cell, 16)
            maps[f'{filt}:{phot}:{d}:resid_pT'] = C.binned(yd[q] - kp[j] * (Xd[q, j] + Xd[q, nd + j]), cell, 16)
            if phot == 'psf':
                maps[f'{filt}:{d}:a_raw'] = C.binned(D['a'][q], cell, 16)
                maps[f'{filt}:{d}:pT_raw'] = C.binned(D['pT'][q], cell, 16)
        print(f"\n{filt.upper()} {C.PHOT_LABEL[phot]}: N={res['N']} stars={res['n_star']} tiles={res['n_tile']} sigma={sig*1e3:.1f} mmag")
        print(f"  pooled k_a = {kpool:+.3f} ± {kpool_e:.3f}   pooled k_pT = {kpp:+.3f} ± {kpp_e:.3f}")
        for j, d in enumerate(dets):
            ms = {s: mstats[s][d] for s in mstats}
            print(f"  {d:9s} k_a={ka[j]:+.3f}±{ka_e[j]:.3f}  k_pT={kp[j]:+.3f}±{kp_e[j]:.3f}  "
                  f"[a+sT: k_a={kas[j]:+.3f}±{kas_e[j]:.3f} k_s={kas[nd+j]:+.3f}±{kas_e[nd+j]:.3f}]  dil={dil[d]:.2f}  "
                  f"rms8x8 before {ms['before']['rms']:.2f} (null {ms['before']['null']:.2f}) "
                  f"k=1 {ms['k1_area']['rms']:.2f} cv_a {ms['cv_a']['rms']:.2f} cv_pT {ms['cv_pT']['rms']:.2f} mmag", flush=True)
json.dump(out, open(f'test2{SUFFIX}.json', 'w'), indent=1)
np.savez(f'test2_maps{SUFFIX}.npz', **maps)
