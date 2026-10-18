"""Empirical test of distortion corrections for 10678 on the per-frame catalogues.

Inputs: matches/matches_<filt>.npz (extract_matches.py; frame <-> tile consensus pairs) and
matches/refmatches_<filt>.npz (extract_refcat.py; frame <-> Gaia DR3/VIRAC2 refcat pairs).

Solutions (corrections c(x, y) in detector-axis mas added to each detection's position):
  crds       : c = 0 (positions as produced with jwst_1596.pmap distortion)
  gdc_v1/v2  : c = delta_STDGDC (affine-anchored, fields.DeltaField mode='affine'; F212N only)
  dqmap_cv   : c = -M, M = data-qa-estimator binned map (16x16) built from the OTHER fold of tiles
  loomap_cv  : c = -M, M = leave-one-out-estimator binned map built from the OTHER fold
  dqmap_full : c = -M from all tiles (circular; shown only to size the circularity gain)
Folds: tiles (obs) sorted, fold = rank % 2.

Estimators on the consensus pairs (s = frame - official consensus on the tangent plane):
  dq  : e_i = s_i + c_i - <c>_star          (data-qa: consensus includes the frame itself)
  loo : e_i = s'_i - mean_{j != i, same star} s'_j,  s' = s + c   (needs >= 2 detections)
Absolute (refcat): e_i = s_i + c_i.
Per frame (one detector, one exposure) a shift or a similarity (shift + rotation + scale) is fitted
to the detector-axis residual and removed, iterated 3 times; the result is binned per detector
and summarised with data_qa.intradetector.stack_detector (median per cell; RMS of the binned map;
shuffled-position null; debiased RMS = sqrt(rms^2 - null^2)).

Dilution: a known field f is injected with no noise into the real detection geometry and passed
through the same estimator + per-frame removal; recovery slope = <R.T>/<T.T> where T is the
injected field binned on the same cells with the same per-detector transform removed.
  inject 'gdc'  : f = -delta_STDGDC v1 (what CRDS - truth would be if STDGDC were the truth)
  inject 'dqmap': f = the all-tile dq-estimator map (shape of the measured pattern)

Usage:
  python empirical_test.py --filters F212N F480M --matchdir <distcmp>/matches --outdir <distcmp>/empirical
Outputs: <outdir>/empirical_<filt>.json, <outdir>/maps_<filt>.npz
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
import fields as F                                     # noqa: E402
from data_qa import intradetector as I                 # noqa: E402
from extract_matches import DETS                       # noqa: E402

NB = 16
NB_TILE = 8
NB_REF = 8
N_ITER = 3


class Data:
    """Per-detection arrays plus per-frame transforms (detector mas <-> sky arcsec)."""

    def __init__(self, path, consensus=True):
        z = np.load(path, allow_pickle=False)
        self.obs = z["obs"].astype(int); self.frame = z["frame"].astype(int)
        self.det = z["det"].astype(int)
        self.x = z["x"].astype(float); self.y = z["y"].astype(float)
        self.s = np.column_stack([z["sxi"], z["seta"]]).astype(float)
        J = z["frame_J"]; ps = z["frame_pixscale"]
        M = J * (ps * 1000.0)[:, None, None]          # sky arcsec -> detector mas
        self.Mf = M; self.Mfi = np.linalg.inv(M)
        self.nframe = len(ps)
        if consensus:
            key = self.obs.astype(np.int64) * 10_000_000 + z["star"].astype(np.int64)
            _, self.g = np.unique(key, return_inverse=True)
            self.ng = np.bincount(self.g)
        self.N = len(self.x)
        uo = np.unique(self.obs)
        rank = {o: i for i, o in enumerate(uo)}
        self.fold = np.array([rank[o] % 2 for o in self.obs])
        self.tiles = uo

    def to_det(self, s):
        M = self.Mf[self.frame]
        return np.column_stack([M[:, 0, 0] * s[:, 0] + M[:, 0, 1] * s[:, 1],
                                M[:, 1, 0] * s[:, 0] + M[:, 1, 1] * s[:, 1]])

    def to_sky(self, r):
        M = self.Mfi[self.frame]
        return np.column_stack([M[:, 0, 0] * r[:, 0] + M[:, 0, 1] * r[:, 1],
                                M[:, 1, 0] * r[:, 0] + M[:, 1, 1] * r[:, 1]])


def frame_fit(D, r, mode):
    """Per-frame least-squares shift or similarity fit to detector-axis residuals r; returns the
    model evaluated at each detection (N x 2)."""
    f = D.frame; n = np.bincount(f, minlength=D.nframe).astype(float)
    n[n == 0] = 1
    mx = np.bincount(f, r[:, 0], D.nframe) / n; my = np.bincount(f, r[:, 1], D.nframe) / n
    if mode == "shift":
        return np.column_stack([mx[f], my[f]])
    cx = np.bincount(f, D.x, D.nframe) / n; cy = np.bincount(f, D.y, D.nframe) / n
    xt = D.x - cx[f]; yt = D.y - cy[f]
    rx = r[:, 0] - mx[f]; ry = r[:, 1] - my[f]
    den = np.bincount(f, xt * xt + yt * yt, D.nframe); den[den == 0] = 1
    a = np.bincount(f, xt * rx + yt * ry, D.nframe) / den
    b = np.bincount(f, xt * ry - yt * rx, D.nframe) / den
    return np.column_stack([mx[f] + a[f] * xt - b[f] * yt, my[f] + b[f] * xt + a[f] * yt])


def residuals(D, c_det, estimator, mode, s=None):
    """Detector-axis residual (mas) of every detection under correction c_det (N x 2, mas).
    Returns (r, valid)."""
    s0 = D.s if s is None else s
    c_sky = D.to_sky(c_det)
    valid = np.ones(D.N, bool)
    if estimator in ("dq", "loo"):
        ng = D.ng
        valid = ng[D.g] >= 2 if estimator == "loo" else valid
    sp = s0 + c_sky
    for _ in range(N_ITER):
        if estimator == "dq":
            mc = np.column_stack([np.bincount(D.g, c_sky[:, k], len(D.ng)) / D.ng for k in (0, 1)])
            # consensus includes the frame itself; frame-shift updates are carried in sp - s0 - c
            e = sp - mc[D.g]
        elif estimator == "loo":
            S = np.column_stack([np.bincount(D.g, sp[:, k], len(D.ng)) for k in (0, 1)])
            den = np.maximum(D.ng[D.g] - 1, 1)[:, None]
            e = sp - (S[D.g] - sp) / den
        else:
            e = sp
        r = D.to_det(e)
        r[~valid] = 0.0
        m = frame_fit_masked(D, r, mode, valid)
        sp = sp - D.to_sky(m)
    if estimator == "dq":
        mc = np.column_stack([np.bincount(D.g, c_sky[:, k], len(D.ng)) / D.ng for k in (0, 1)])
        e = sp - mc[D.g]
    elif estimator == "loo":
        S = np.column_stack([np.bincount(D.g, sp[:, k], len(D.ng)) for k in (0, 1)])
        den = np.maximum(D.ng[D.g] - 1, 1)[:, None]
        e = sp - (S[D.g] - sp) / den
    else:
        e = sp
    return D.to_det(e), valid


def frame_fit_masked(D, r, mode, valid):
    """frame_fit restricted to valid detections (invalid rows get the frame model too)."""
    if valid.all():
        return frame_fit(D, r, mode)
    f = D.frame; w = valid.astype(float)
    n = np.bincount(f, w, D.nframe); n[n == 0] = 1
    mx = np.bincount(f, w * r[:, 0], D.nframe) / n; my = np.bincount(f, w * r[:, 1], D.nframe) / n
    if mode == "shift":
        return np.column_stack([mx[f], my[f]])
    cx = np.bincount(f, w * D.x, D.nframe) / n; cy = np.bincount(f, w * D.y, D.nframe) / n
    xt = D.x - cx[f]; yt = D.y - cy[f]
    rx = r[:, 0] - mx[f]; ry = r[:, 1] - my[f]
    den = np.bincount(f, w * (xt * xt + yt * yt), D.nframe); den[den == 0] = 1
    a = np.bincount(f, w * (xt * rx + yt * ry), D.nframe) / den
    b = np.bincount(f, w * (xt * ry - yt * rx), D.nframe) / den
    return np.column_stack([mx[f] + a[f] * xt - b[f] * yt, my[f] + b[f] * xt + a[f] * yt])


def det_list(filt):
    return F.SW_DETS if filt.upper() == "F212N" else F.LW_DETS


def gdc_correction(D, filt, version):
    c = np.zeros((D.N, 2))
    for d in det_list(filt):
        sel = D.det == DETS.index(d.lower())
        if not sel.any():
            continue
        dx, dy = F.DeltaField(d, filt, version, mode="affine")(D.x[sel], D.y[sel])
        c[sel, 0] = np.nan_to_num(dx); c[sel, 1] = np.nan_to_num(dy)
    return c


def build_maps(D, r, sel, filt, nb=NB):
    """{det: (mdx, mdy, cnt)} binned medians of r over detections `sel`."""
    out = {}
    for d in det_list(filt):
        m = sel & (D.det == DETS.index(d.lower()))
        if m.sum() < I.MIN_PER_CELL * 4:
            continue
        out[d] = I.binned_map(D.x[m], D.y[m], r[m, 0], r[m, 1], nb=nb)
    return out


def map_correction(D, maps_by_fold, filt, apply_fold):
    """c = -M evaluated at each detection, M taken from maps_by_fold[k] for detections whose
    fold == apply_fold[k] mapping (dict train_fold -> maps applied to the other fold)."""
    c = np.zeros((D.N, 2))
    for test_fold, maps in maps_by_fold.items():
        for d, (mdx, mdy, _) in maps.items():
            sel = (D.det == DETS.index(d.lower())) & (D.fold == test_fold if apply_fold else True)
            if not sel.any():
                continue
            fx, fy = F.BinnedMapField(mdx, mdy)(D.x[sel], D.y[sel])
            c[sel, 0] = -fx; c[sel, 1] = -fy
    return c


def summarise(D, r, valid, filt, nb, n_perm=10, sel=None):
    out = {}
    for d in det_list(filt):
        m = valid & (D.det == DETS.index(d.lower()))
        if sel is not None:
            m &= sel
        if m.sum() < I.MIN_PER_CELL * 4:
            continue
        st = I.stack_detector(D.x[m], D.y[m], r[m, 0], r[m, 1], nb=nb, n_perm=n_perm)
        out[d] = dict(n=st["n"], sysrms_mas=st["sysrms_mas"], null_rms_mas=st["null_rms_mas"],
                      sysrms_debiased_mas=st["sysrms_debiased_mas"],
                      perstar_rms_mas=st["perstar_rms_mas"], cells_used=st["cells_used"],
                      mdx=st["mdx"], mdy=st["mdy"])
    return out


def per_tile(D, r, valid, filt, nb=NB_TILE):
    """{det: {obs: debiased RMS}} for each tile separately."""
    out = {}
    for d in det_list(filt):
        md = valid & (D.det == DETS.index(d.lower()))
        row = {}
        for o in D.tiles:
            m = md & (D.obs == o)
            if m.sum() < I.MIN_PER_CELL * 4 * 4:
                continue
            st = I.stack_detector(D.x[m], D.y[m], r[m, 0], r[m, 1], nb=nb, n_perm=5)
            row[int(o)] = (st["sysrms_mas"], st["null_rms_mas"], st["sysrms_debiased_mas"])
        out[d] = row
    return out


def recovery(D, f_det, estimator, mode, filt, nb=NB):
    """Inject field f (detector mas) with no noise, return per-detector recovery numbers."""
    # perfect positions (s = 0) displaced by the field: passing f as the "correction" moves every
    # detection by f and, for the dq estimator, moves the consensus by the mean of f over the
    # star's detections (consensus includes the frame itself); loo uses the other detections.
    r, valid = residuals(D, f_det, estimator, mode, s=np.zeros((D.N, 2)))
    out = {}
    for d in det_list(filt):
        m = valid & (D.det == DETS.index(d.lower()))
        if m.sum() < I.MIN_PER_CELL * 4:
            continue
        rx, ry, cnt = I.binned_map(D.x[m], D.y[m], r[m, 0], r[m, 1], nb=nb)
        tx, ty, _ = I.binned_map(D.x[m], D.y[m], f_det[m, 0], f_det[m, 1], nb=nb)
        ok = np.isfinite(rx) & np.isfinite(tx)
        cxx = (np.arange(nb) + 0.5) * F.NPIX / nb
        XX, YY = np.meshgrid(cxx, cxx, indexing="ij")
        tmode = "shift" if mode == "shift" else "similarity"
        ttx, tty = F.remove_transform_detmas(XX[ok], YY[ok], tx[ok], ty[ok], tmode)
        rrx, rry = rx[ok], ry[ok]
        T = np.r_[ttx, tty]; R = np.r_[rrx, rry]
        slope = float(np.dot(R, T) / np.dot(T, T)) if np.dot(T, T) > 0 else float("nan")
        corr = float(np.corrcoef(R, T)[0, 1]) if T.std() > 0 else float("nan")
        out[d] = dict(slope=slope, corr=corr, truth_rms_mas=float(np.sqrt(np.mean(ttx**2 + tty**2))),
                      recovered_rms_mas=float(np.sqrt(np.mean(rrx**2 + rry**2))),
                      rec_mdx=rx, rec_mdy=ry, truth_mdx=tx, truth_mdy=ty)
    return out


def strip(d):
    """Drop arrays for JSON."""
    if isinstance(d, dict):
        return {k: strip(v) for k, v in d.items() if not isinstance(v, np.ndarray)}
    return d


def run(filt, matchdir, outdir):
    os.makedirs(outdir, exist_ok=True)
    D = Data(os.path.join(matchdir, f"matches_{filt.lower()}.npz"))
    print(f"[{filt}] {D.N} detections, {D.nframe} frames, {len(D.tiles)} tiles", flush=True)
    zero = np.zeros((D.N, 2))
    res = dict(filter=filt, n_det=int(D.N), n_frames=int(D.nframe), n_tiles=int(len(D.tiles)),
               nb=NB, nb_tile=NB_TILE, nb_ref=NB_REF)
    mapsave = {}

    # training maps (CRDS positions, shift removal, as data-qa)
    r_dq0, v_dq0 = residuals(D, zero, "dq", "shift")
    r_lo0, v_lo0 = residuals(D, zero, "loo", "shift")
    maps_dq = {1 - k: build_maps(D, r_dq0, v_dq0 & (D.fold == k), filt) for k in (0, 1)}
    maps_lo = {1 - k: build_maps(D, r_lo0, v_lo0 & (D.fold == k), filt) for k in (0, 1)}
    full_dq = build_maps(D, r_dq0, v_dq0, filt)
    full_lo = build_maps(D, r_lo0, v_lo0, filt)
    for d, (mx, my, cn) in full_dq.items():
        mapsave[f"dqfull_{d}_dx"] = mx; mapsave[f"dqfull_{d}_dy"] = my; mapsave[f"dqfull_{d}_n"] = cn
    for d, (mx, my, cn) in full_lo.items():
        mapsave[f"loofull_{d}_dx"] = mx; mapsave[f"loofull_{d}_dy"] = my
    # train-fold maps agreement (split-half repeatability of the measured pattern)
    rep = {}
    for d in full_dq:
        a = maps_dq[1][d] if d in maps_dq[1] else None   # built from fold 0
        b = maps_dq[0][d] if d in maps_dq[0] else None   # built from fold 1
        if a is None or b is None:
            continue
        va = np.r_[a[0].ravel(), a[1].ravel()]; vb = np.r_[b[0].ravel(), b[1].ravel()]
        ok = np.isfinite(va) & np.isfinite(vb)
        rep[d] = dict(r_splithalf=float(np.corrcoef(va[ok], vb[ok])[0, 1]),
                      rms_fold0=float(np.sqrt(np.nanmean(a[0]**2 + a[1]**2))),
                      rms_fold1=float(np.sqrt(np.nanmean(b[0]**2 + b[1]**2))),
                      rms_halfdiff=float(np.sqrt(np.nanmean((a[0]-b[0])**2 + (a[1]-b[1])**2)) / 2))
    res["dqmap_splithalf"] = rep

    corr = {"crds": zero}
    if filt.upper() == "F212N":
        corr["gdc_v1"] = gdc_correction(D, filt, "v1")
        corr["gdc_v2"] = gdc_correction(D, filt, "v2")
    corr["dqmap_cv"] = map_correction(D, maps_dq, filt, True)
    corr["loomap_cv"] = map_correction(D, maps_lo, filt, True)
    corr["dqmap_full"] = map_correction(D, {None: full_dq}, filt, False)

    res["consensus"] = {}
    pertile = {}
    for est in ("loo", "dq"):
        for mode in ("shift", "similarity"):
            key = f"{est}_{mode}"
            res["consensus"][key] = {}
            pertile[key] = {}
            for sol, c in corr.items():
                r, v = residuals(D, c, est, mode)
                sm = summarise(D, r, v, filt, NB)
                res["consensus"][key][sol] = strip(sm)
                for d, sd in sm.items():
                    mapsave[f"cons_{key}_{sol}_{d}_dx"] = sd["mdx"]
                    mapsave[f"cons_{key}_{sol}_{d}_dy"] = sd["mdy"]
                pertile[key][sol] = per_tile(D, r, v, filt)
                print(f"  [{filt}] {key} {sol}: " + ", ".join(
                    f"{d} {sd['sysrms_debiased_mas']:.2f}" for d, sd in sm.items()), flush=True)
    # per-tile consistency vs crds
    res["per_tile"] = {}
    for key, bysol in pertile.items():
        res["per_tile"][key] = {}
        for sol, bydet in bysol.items():
            if sol == "crds":
                continue
            res["per_tile"][key][sol] = {}
            for d, row in bydet.items():
                base = bysol["crds"][d]
                common = sorted(set(row) & set(base))
                dr = np.array([row[o][0] - base[o][0] for o in common])        # raw binned RMS
                dd = np.array([row[o][2] - base[o][2] for o in common])        # debiased
                res["per_tile"][key][sol][d] = dict(
                    n_tiles=len(common),
                    median_delta_rms_mas=float(np.median(dr)) if len(dr) else None,
                    frac_tiles_reduced=float(np.mean(dr < 0)) if len(dr) else None,
                    median_delta_debiased_mas=float(np.median(dd)) if len(dd) else None,
                    crds_tile_debiased_median_mas=float(np.median([base[o][2] for o in common]))
                    if common else None)
        res["per_tile"][key]["crds_raw"] = {
            d: {str(o): v for o, v in row.items()} for d, row in bysol["crds"].items()}

    # dilution / recovery
    res["recovery"] = {}
    inj = {"dqmap": -corr["dqmap_full"]}     # -(-M) = M : the measured pattern itself
    if filt.upper() == "F212N":
        inj["gdc"] = -corr["gdc_v1"]
    for name, f in inj.items():
        res["recovery"][name] = {}
        for est in ("dq", "loo"):
            for mode in ("shift", "similarity"):
                rc = recovery(D, f, est, mode, filt)
                res["recovery"][name][f"{est}_{mode}"] = strip(rc)
                for d, x in rc.items():
                    mapsave[f"rec_{name}_{est}_{mode}_{d}_dx"] = x["rec_mdx"]
                    mapsave[f"rec_{name}_{est}_{mode}_{d}_dy"] = x["rec_mdy"]
                print(f"  [{filt}] recovery {name} {est}_{mode}: " + ", ".join(
                    f"{d} {x['slope']:.2f}" for d, x in rc.items()), flush=True)

    # data-qa map vs predicted map if STDGDC were the truth (dq estimator, shift removal)
    if filt.upper() == "F212N":
        cmp = {}
        for d, (mx, my, _) in full_dq.items():
            px = mapsave[f"rec_gdc_dq_shift_{d}_dx"]; py = mapsave[f"rec_gdc_dq_shift_{d}_dy"]
            M = np.r_[mx.ravel(), my.ravel()]; P = np.r_[px.ravel(), py.ravel()]
            ok = np.isfinite(M) & np.isfinite(P)
            cmp[d] = dict(corr=float(np.corrcoef(M[ok], P[ok])[0, 1]),
                          slope_M_on_P=float(np.dot(M[ok], P[ok]) / np.dot(P[ok], P[ok])),
                          rms_M=float(np.sqrt(np.mean(M[ok]**2) * 2)),
                          rms_P=float(np.sqrt(np.mean(P[ok]**2) * 2)),
                          rms_M_minus_P=float(np.sqrt(np.mean((M[ok] - P[ok])**2) * 2)))
        res["dqmap_vs_gdc_prediction"] = cmp

    # absolute test against the refcat
    rp = os.path.join(matchdir, f"refmatches_{filt.lower()}.npz")
    if os.path.exists(rp):
        R = Data(rp, consensus=False)
        # corrections on the refcat detections
        rc = {"crds": np.zeros((R.N, 2))}
        if filt.upper() == "F212N":
            rc["gdc_v1"] = gdc_correction(R, filt, "v1")
            rc["gdc_v2"] = gdc_correction(R, filt, "v2")
        # the consensus folds are defined on the same sorted obs list -> same fold labels
        rc["dqmap_cv"] = map_correction(R, maps_dq, filt, True)
        rc["loomap_cv"] = map_correction(R, maps_lo, filt, True)
        rc["dqmap_full"] = map_correction(R, {None: full_dq}, filt, False)
        res["refcat"] = dict(n_det=int(R.N), n_frames=int(R.nframe))
        for mode in ("shift", "similarity"):
            res["refcat"][mode] = {}
            for nb in (NB_REF, 4):
                res["refcat"][mode][f"nb{nb}"] = {}
                for sol, c in rc.items():
                    r, v = residuals(R, c, "abs", mode)
                    sm = summarise(R, r, v, filt, nb, n_perm=20)
                    res["refcat"][mode][f"nb{nb}"][sol] = strip(sm)
                    for d, sd in sm.items():
                        mapsave[f"ref_{mode}_nb{nb}_{sol}_{d}_dx"] = sd["mdx"]
                        mapsave[f"ref_{mode}_nb{nb}_{sol}_{d}_dy"] = sd["mdy"]
                    print(f"  [{filt}] refcat {mode} nb{nb} {sol}: " + ", ".join(
                        f"{d} {sd['sysrms_debiased_mas']:.2f}" for d, sd in sm.items()), flush=True)
        # predicted refcat map if STDGDC were the truth: -delta binned on refcat detections
        if filt.upper() == "F212N":
            res["refcat"]["gdc_truth_prediction"] = {}
            for d in det_list(filt):
                m = R.det == DETS.index(d.lower())
                f = -rc["gdc_v1"][m]
                for mode in ("shift", "similarity"):
                    tx, ty, _ = I.binned_map(R.x[m], R.y[m], f[:, 0], f[:, 1], nb=NB_REF)
                    ok = np.isfinite(tx)
                    cxx = (np.arange(NB_REF) + 0.5) * F.NPIX / NB_REF
                    XX, YY = np.meshgrid(cxx, cxx, indexing="ij")
                    ttx, tty = F.remove_transform_detmas(XX[ok], YY[ok], tx[ok], ty[ok], mode)
                    mm = res["refcat"][mode][f"nb{NB_REF}"]["crds"].get(d)
                    mdx = mapsave.get(f"ref_{mode}_nb{NB_REF}_crds_{d}_dx")
                    mdy = mapsave.get(f"ref_{mode}_nb{NB_REF}_crds_{d}_dy")
                    entry = dict(pred_rms_mas=float(np.sqrt(np.mean(ttx**2 + tty**2))))
                    if mm is not None:
                        Mv = np.r_[mdx[ok], mdy[ok]]; Pv = np.r_[ttx, tty]
                        g = np.isfinite(Mv)
                        entry["corr_crdsmap_vs_pred"] = float(np.corrcoef(Mv[g], Pv[g])[0, 1])
                        entry["slope_crdsmap_on_pred"] = float(np.dot(Mv[g], Pv[g]) / np.dot(Pv[g], Pv[g]))
                    res["refcat"]["gdc_truth_prediction"].setdefault(d, {})[mode] = entry
    jp = os.path.join(outdir, f"empirical_{filt.lower()}.json")
    with open(jp, "w") as fh:
        json.dump(res, fh, indent=1, default=float)
    np.savez(os.path.join(outdir, f"maps_{filt.lower()}.npz"), **mapsave)
    print(f"[{filt}] wrote {jp}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--filters", nargs="+", default=["F212N", "F480M"])
    ap.add_argument("--matchdir", required=True)
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()
    for f in a.filters:
        run(f, a.matchdir, a.outdir)


if __name__ == "__main__":
    main()
