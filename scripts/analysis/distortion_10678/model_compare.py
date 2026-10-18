"""Model-to-model comparison of intra-detector distortion solutions for 10678.

Per detector:
  * STDGDC (v1, v2; F212N) minus CRDS (jwst_1596.pmap R_DISTOR files) on a 128 x 128 pixel grid
    (16-px step), after removing the least-squares similarity (shift + rotation + scale) or the
    6-parameter affine between the STDGDC corrected frame and CRDS V2V3 (fields.DeltaField).
    Displacements are in detector-axis mas (pixel offset x CRDS pixel scale).
  * data-qa stacked residual map (intradet_<filt>.json from data_qa.intradetector_distortion;
    16 x 16 cells; already a residual relative to CRDS) with a similarity / affine removed.
  * data-qa map M versus -delta_STDGDC (the residual CRDS would leave if STDGDC were the truth),
    -delta averaged over each 128-px cell, both with a similarity removed: Pearson r, slope of M
    on -delta, RMS of the difference.

Usage:
  python model_compare.py --dqdir <distcmp>/dataqa_stack --outdir <distcmp>/model_compare
Outputs: <outdir>/model_compare.json, <outdir>/model_fields.npz
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fields as F   # noqa: E402

NG = 128
EDGE = 32   # px excluded for the 'interior' statistics


def stats(dx, dy, x=None, y=None):
    a = np.hypot(dx, dy)
    ok = np.isfinite(a)
    out = dict(rms_mas=float(np.sqrt(np.mean(a[ok] ** 2))), max_mas=float(np.max(a[ok])),
               p99_mas=float(np.percentile(a[ok], 99)), n=int(ok.sum()))
    if x is not None:
        inner = ok & (x >= EDGE) & (x <= F.NPIX - 1 - EDGE) & (y >= EDGE) & (y <= F.NPIX - 1 - EDGE)
        out.update(rms_interior_mas=float(np.sqrt(np.mean(a[inner] ** 2))),
                   max_interior_mas=float(np.max(a[inner])))
    return out


def cell_average(field, nb=16, sub=8):
    """Average of a callable detector-axis field over each of nb x nb cells (sub x sub samples)."""
    cs = F.NPIX / nb
    off = (np.arange(sub) + 0.5) * cs / sub
    mdx = np.full((nb, nb), np.nan); mdy = np.full((nb, nb), np.nan)
    for i in range(nb):
        for j in range(nb):
            xx, yy = np.meshgrid(i * cs + off, j * cs + off, indexing="ij")
            dx, dy = field(xx.ravel(), yy.ravel())
            mdx[i, j] = np.nanmean(dx); mdy[i, j] = np.nanmean(dy)
    return mdx, mdy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dqdir", required=True)
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    g = np.linspace(0, F.NPIX - 1, NG)
    GX, GY = np.meshgrid(g, g, indexing="ij")
    gx, gy = GX.ravel(), GY.ravel()
    out = dict(grid=f"{NG}x{NG}", edge_px=EDGE, crds_files={d: os.path.basename(F.crds_path(d))
                                                            for d in F.R_DISTOR},
               pixscale_arcsec={d: F.pixscale(d) for d in F.R_DISTOR}, detectors={})
    save = {}
    dq = {}
    for filt in ("F212N", "F480M"):
        p = os.path.join(a.dqdir, f"intradet_{filt.lower()}.json")
        if os.path.exists(p):
            with open(p) as fh:
                dq[filt] = json.load(fh)
    for det in F.SW_DETS + F.LW_DETS:
        filt = "F212N" if det in F.SW_DETS else "F480M"
        row = {}
        if det in F.SW_DETS:
            fl = {}
            for ver in ("v1", "v2"):
                for mode in ("similarity", "affine"):
                    fld = F.DeltaField(det, filt, ver, mode)
                    dx, dy = fld(gx, gy)
                    row[f"stdgdc_{ver}_minus_crds_{mode}"] = stats(dx, dy, gx, gy)
                    row[f"stdgdc_{ver}_invalid_anchor_points"] = fld.n_invalid_anchor
                    save[f"{det}_{ver}_{mode}_dx"] = dx.reshape(NG, NG)
                    save[f"{det}_{ver}_{mode}_dy"] = dy.reshape(NG, NG)
                    fl[(ver, mode)] = (dx, dy)
            for mode in ("similarity", "affine"):
                ddx = fl[("v2", mode)][0] - fl[("v1", mode)][0]
                ddy = fl[("v2", mode)][1] - fl[("v1", mode)][1]
                row[f"stdgdc_v2_minus_v1_{mode}"] = stats(ddx, ddy, gx, gy)
        d = dq.get(filt, {}).get("detectors", {}).get(det.lower())
        if d is not None:
            M = (np.array(d["map_dx_mas"], float), np.array(d["map_dy_mas"], float))
            nb = M[0].shape[0]
            c = (np.arange(nb) + 0.5) * F.NPIX / nb
            CX, CY = np.meshgrid(c, c, indexing="ij")
            ok = np.isfinite(M[0]) & np.isfinite(M[1])
            null = float(d["null_rms_mas"])
            row["dataqa_map_raw"] = dict(stats(M[0][ok], M[1][ok]), null_rms_mas=null,
                                         sysrms_debiased_mas=float(d["sysrms_debiased_mas"]),
                                         n_pairs=int(d["n"]), cells=int(ok.sum()))
            for mode in ("similarity", "affine"):
                rx, ry = F.remove_transform_detmas(CX[ok], CY[ok], M[0][ok], M[1][ok], mode)
                st = stats(rx, ry)
                st["debiased_rms_mas"] = float(np.sqrt(max(st["rms_mas"] ** 2 - null ** 2, 0)))
                row[f"dataqa_map_minus_crds_{mode}"] = st
                save[f"{det}_dqmap_{mode}_dx"] = np.where(ok, 0, np.nan) + _fill(ok, rx)
                save[f"{det}_dqmap_{mode}_dy"] = np.where(ok, 0, np.nan) + _fill(ok, ry)
            if det in F.SW_DETS:
                fld = F.DeltaField(det, filt, "v1", "similarity")
                ex, ey = cell_average(lambda x, y: tuple(-v for v in fld(x, y)), nb=nb)
                good = ok & np.isfinite(ex) & np.isfinite(ey)
                mx, my = F.remove_transform_detmas(CX[good], CY[good], M[0][good], M[1][good],
                                                   "similarity")
                px, py = F.remove_transform_detmas(CX[good], CY[good], ex[good], ey[good],
                                                   "similarity")
                Mv = np.r_[mx, my]; Pv = np.r_[px, py]
                row["dataqa_map_vs_minus_stdgdc_delta"] = dict(
                    pearson_r=float(np.corrcoef(Mv, Pv)[0, 1]),
                    slope_map_on_pred=float(np.dot(Mv, Pv) / np.dot(Pv, Pv)),
                    rms_map_mas=float(np.sqrt(np.mean(mx ** 2 + my ** 2))),
                    rms_pred_mas=float(np.sqrt(np.mean(px ** 2 + py ** 2))),
                    rms_map_minus_pred_mas=float(np.sqrt(np.mean((mx - px) ** 2 + (my - py) ** 2))),
                    null_rms_mas=null)
                save[f"{det}_minusdelta_cell_dx"] = ex; save[f"{det}_minusdelta_cell_dy"] = ey
        out["detectors"][det] = row
        print(det, json.dumps({k: (v.get("rms_mas") if isinstance(v, dict) and "rms_mas" in v
                                   else v) for k, v in row.items()}, default=float), flush=True)
    with open(os.path.join(a.outdir, "model_compare.json"), "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    np.savez(os.path.join(a.outdir, "model_fields.npz"), grid=g, **save)
    print("wrote", os.path.join(a.outdir, "model_compare.json"))


def _fill(ok, v):
    out = np.full(ok.shape, np.nan)
    out[ok] = v
    return out


if __name__ == "__main__":
    main()
