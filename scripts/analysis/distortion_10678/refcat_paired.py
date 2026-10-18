"""Paired regression test against Gaia DR3/VIRAC2: does each correction predict the CRDS residual?

The per-star reference noise (~50-60 mas, dominated by VIRAC2 proper-motion propagation over
~12 yr) makes the debiased binned RMS of the frame-minus-refcat map noise-limited for SW.
This test uses every cell instead: for each detector, the binned CRDS residual map m (median per
cell; frame - refcat, per-frame similarity removed) is regressed on the predicted residual p that
CRDS would leave if the correction were the truth (p = -c passed through the same per-frame
similarity removal, binned by mean).  beta = sum(m p) / sum(p p): beta ~ 1 means the correction
predicts the absolute residual at full amplitude, beta ~ 0 means no relation.  Uncertainties come
from 300 bootstrap resamples of the tiles (obs) with replacement.

Corrections: gdc_v1, gdc_v2 (F212N), dqmap / loomap (all-tile maps from empirical/maps_<filt>.npz,
built from frame-vs-consensus residuals only, independent of the refcat).

Usage:
  python refcat_paired.py --filters F212N F480M --matchdir <distcmp>/matches \
      --empdir <distcmp>/empirical --out <distcmp>/empirical/refcat_paired.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
import fields as F                                                  # noqa: E402
from data_qa import intradetector as I                              # noqa: E402
from empirical_test import Data, residuals, gdc_correction, det_list  # noqa: E402
from extract_matches import DETS                                    # noqa: E402

NBOOT = 300


def map_corr(D, maps, filt, prefix):
    c = np.zeros((D.N, 2))
    for d in det_list(filt):
        k = f"{prefix}_{d}_dx"
        if k not in maps:
            continue
        sel = D.det == DETS.index(d.lower())
        fx, fy = F.BinnedMapField(maps[k], maps[f"{prefix}_{d}_dy"])(D.x[sel], D.y[sel])
        c[sel, 0] = -fx; c[sel, 1] = -fy
    return c


def cell_stats(x, y, r, p, nb, idx):
    mx, my, cnt = I.binned_map(x[idx], y[idx], r[idx, 0], r[idx, 1], nb=nb, minn=20)
    # mean of the (smooth) prediction per cell
    k = np.clip((x[idx] / F.NPIX * nb).astype(int), 0, nb - 1) * nb + \
        np.clip((y[idx] / F.NPIX * nb).astype(int), 0, nb - 1)
    n = np.bincount(k, minlength=nb * nb).astype(float); n[n == 0] = np.nan
    px = (np.bincount(k, p[idx, 0], nb * nb) / n).reshape(nb, nb)
    py = (np.bincount(k, p[idx, 1], nb * nb) / n).reshape(nb, nb)
    ok = np.isfinite(mx) & np.isfinite(px)
    M = np.r_[mx[ok], my[ok]]; P = np.r_[px[ok], py[ok]]
    beta = float(M @ P / (P @ P)) if (P @ P) > 0 else np.nan
    corr = float(np.corrcoef(M, P)[0, 1]) if P.std() > 0 else np.nan
    # variance explained: mean square of the map before / after subtracting the prediction
    dv = float(np.mean(M ** 2) - np.mean((M - P) ** 2))
    return beta, corr, dv, float(np.sqrt(np.mean(M ** 2) * 2)), float(np.sqrt(np.mean(P ** 2) * 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--filters", nargs="+", default=["F212N", "F480M"])
    ap.add_argument("--matchdir", required=True)
    ap.add_argument("--empdir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rng = np.random.default_rng(42)
    out = {}
    for filt in a.filters:
        R = Data(os.path.join(a.matchdir, f"refmatches_{filt.lower()}.npz"), consensus=False)
        maps = dict(np.load(os.path.join(a.empdir, f"maps_{filt.lower()}.npz")))
        zero = np.zeros((R.N, 2))
        r0, _ = residuals(R, zero, "abs", "similarity")
        corr = {}
        if filt.upper() == "F212N":
            corr["gdc_v1"] = gdc_correction(R, filt, "v1")
            corr["gdc_v2"] = gdc_correction(R, filt, "v2")
        corr["dqmap"] = map_corr(R, maps, filt, "dqfull")
        corr["loomap"] = map_corr(R, maps, filt, "loofull")
        preds = {}
        for k, c in corr.items():
            rp, _ = residuals(R, c, "abs", "similarity", s=zero)
            preds[k] = -rp
        tiles = np.unique(R.obs)
        by_tile = {o: np.nonzero(R.obs == o)[0] for o in tiles}
        res = {}
        for d in det_list(filt):
            dm = R.det == DETS.index(d.lower())
            res[d] = {}
            for nb in (8, 4):
                for k, p in preds.items():
                    idx = np.nonzero(dm)[0]
                    beta, cc, dv, rms_m, rms_p = cell_stats(R.x, R.y, r0, p, nb, idx)
                    boots = []
                    for _ in range(NBOOT):
                        pick = rng.choice(tiles, len(tiles), replace=True)
                        bi = np.concatenate([by_tile[o] for o in pick])
                        bi = bi[dm[bi]]
                        boots.append(cell_stats(R.x, R.y, r0, p, nb, bi)[:3])
                    boots = np.array(boots)
                    res[d][f"{k}_nb{nb}"] = dict(
                        beta=beta, beta_err=float(np.nanstd(boots[:, 0])), corr=cc,
                        corr_err=float(np.nanstd(boots[:, 1])), var_explained_mas2=dv,
                        var_explained_err=float(np.nanstd(boots[:, 2])), rms_crds_map_mas=rms_m,
                        rms_pred_mas=rms_p, n=int(dm.sum()))
            print(filt, d, {k: (round(v["beta"], 2), round(v["beta_err"], 2)) for k, v in res[d].items()},
                  flush=True)
        # all SW (or LW) detectors combined: inverse-variance mean of beta
        comb = {}
        for key in res[det_list(filt)[0]]:
            b = np.array([res[d][key]["beta"] for d in res]); e = np.array([res[d][key]["beta_err"] for d in res])
            w = 1 / e ** 2
            comb[key] = dict(beta=float(np.sum(w * b) / np.sum(w)), beta_err=float(1 / np.sqrt(np.sum(w))))
        res["combined"] = comb
        print(filt, "combined", comb, flush=True)
        out[filt] = res
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
