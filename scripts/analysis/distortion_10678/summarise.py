"""Collect every number for REPORT.md into tables (markdown) and one JSON.

Reads model_compare/model_compare.json, dataqa_stack/intradet_*.json, empirical/empirical_*.json,
empirical/maps_*.npz and empirical/catalog_impact.json.  Adds one derived comparison: the
all-tile measured map M (data-qa estimator 'dq' and leave-one-out 'loo') against the map that the
same estimator would measure if STDGDC v1 were the truth (the injected -delta_STDGDC passed
through the estimator, 'rec_gdc_<est>_shift'), both with a similarity removed per detector.

Usage:
  python summarise.py --distcmp <distcmp>
Outputs: <distcmp>/tables.md, <distcmp>/summary.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fields as F   # noqa: E402

SOLS = ["crds", "gdc_v1", "gdc_v2", "dqmap_cv", "loomap_cv", "dqmap_full"]


def sim_rm(dx, dy, mode="similarity"):
    nb = dx.shape[0]
    c = (np.arange(nb) + 0.5) * F.NPIX / nb
    XX, YY = np.meshgrid(c, c, indexing="ij")
    ok = np.isfinite(dx) & np.isfinite(dy)
    rx, ry = F.remove_transform_detmas(XX[ok], YY[ok], dx[ok], dy[ok], mode)
    return ok, rx, ry


def f2(v):
    return "–" if v is None else f"{v:.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--distcmp", required=True)
    a = ap.parse_args()
    dc = a.distcmp
    mc = json.load(open(os.path.join(dc, "model_compare", "model_compare.json")))["detectors"]
    E = {f: json.load(open(os.path.join(dc, "empirical", f"empirical_{f.lower()}.json")))
         for f in ("F212N", "F480M")}
    MP = {f: dict(np.load(os.path.join(dc, "empirical", f"maps_{f.lower()}.npz")))
          for f in ("F212N", "F480M")}
    ci_path = os.path.join(dc, "empirical", "catalog_impact.json")
    CI = json.load(open(ci_path)) if os.path.exists(ci_path) else {}
    S = {}
    L = []
    dets = F.SW_DETS + F.LW_DETS
    filt_of = {d: ("F212N" if d in F.SW_DETS else "F480M") for d in dets}

    # ---- (a) model to model
    L.append("### (a) Model-to-model (detector-axis mas; 128x128 grid; data-qa map 16x16 cells)\n")
    L.append("| det | STDGDC v1−CRDS sim RMS / max | aff RMS / max | interior (32 px) sim max | "
             "v2−v1 RMS | data-qa map sim RMS / max | aff RMS | r(map, CRDS−STDGDC) | slope | "
             "r(map, predicted) | slope (pred.) |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    S["model"] = {}
    for d in dets:
        r = mc[d]; f = filt_of[d]
        row = {}
        g = r.get("stdgdc_v1_minus_crds_similarity")
        ga = r.get("stdgdc_v1_minus_crds_affine")
        v21 = r.get("stdgdc_v2_minus_v1_similarity")
        q = r.get("dataqa_map_minus_crds_similarity"); qa = r.get("dataqa_map_minus_crds_affine")
        x = r.get("dataqa_map_vs_minus_stdgdc_delta")
        # measured map vs estimator-propagated STDGDC prediction (similarity removed)
        pr = None
        if f"rec_gdc_dq_shift_{d}_dx" in MP[f]:
            mx, my = MP[f][f"dqfull_{d}_dx"], MP[f][f"dqfull_{d}_dy"]
            px, py = MP[f][f"rec_gdc_dq_shift_{d}_dx"], MP[f][f"rec_gdc_dq_shift_{d}_dy"]
            good = np.isfinite(mx) & np.isfinite(px)
            mx = np.where(good, mx, np.nan); px = np.where(good, px, np.nan)
            my = np.where(good, my, np.nan); py = np.where(good, py, np.nan)
            _, a1, b1 = sim_rm(mx, my); _, a2, b2 = sim_rm(px, py)
            M = np.r_[a1, b1]; P = np.r_[a2, b2]
            pr = dict(r=float(np.corrcoef(M, P)[0, 1]), slope=float(M @ P / (P @ P)),
                      rms_map=float(np.sqrt(np.mean(a1**2 + b1**2))),
                      rms_pred=float(np.sqrt(np.mean(a2**2 + b2**2))),
                      rms_diff=float(np.sqrt(np.mean((a1 - a2)**2 + (b1 - b2)**2))))
            # same for the LOO estimator
            lx, ly = MP[f][f"loofull_{d}_dx"], MP[f][f"loofull_{d}_dy"]
            qx, qy = MP[f][f"rec_gdc_loo_shift_{d}_dx"], MP[f][f"rec_gdc_loo_shift_{d}_dy"]
            good = np.isfinite(lx) & np.isfinite(qx)
            _, c1, d1 = sim_rm(np.where(good, lx, np.nan), np.where(good, ly, np.nan))
            _, c2, d2 = sim_rm(np.where(good, qx, np.nan), np.where(good, qy, np.nan))
            M2 = np.r_[c1, d1]; P2 = np.r_[c2, d2]
            pr["loo"] = dict(r=float(np.corrcoef(M2, P2)[0, 1]), slope=float(M2 @ P2 / (P2 @ P2)),
                             rms_map=float(np.sqrt(np.mean(c1**2 + d1**2))),
                             rms_pred=float(np.sqrt(np.mean(c2**2 + d2**2))),
                             rms_diff=float(np.sqrt(np.mean((c1 - c2)**2 + (d1 - d2)**2))))
        row.update(stdgdc_sim=g, stdgdc_aff=ga, v2v1=v21, dq_sim=q, dq_aff=qa, dq_vs_delta=x,
                   dq_vs_pred=pr)
        S["model"][d] = row
        L.append("| " + " | ".join([
            d,
            f"{f2(g['rms_mas'])} / {f2(g['max_mas'])}" if g else "–",
            f"{f2(ga['rms_mas'])} / {f2(ga['max_mas'])}" if ga else "–",
            f2(g["max_interior_mas"]) if g else "–",
            f2(v21["rms_mas"]) if v21 else "–",
            f"{f2(q['rms_mas'])} / {f2(q['max_mas'])}" if q else "–",
            f2(qa["rms_mas"]) if qa else "–",
            f2(x["pearson_r"]) if x else "–", f2(x["slope_map_on_pred"]) if x else "–",
            f2(pr["r"]) if pr else "–", f2(pr["slope"]) if pr else "–"]) + " |")
    L.append("")

    # ---- (b) empirical
    def emp_table(key, title):
        L.append(f"### {title}\n")
        L.append("| det | " + " | ".join(SOLS) + " | null |")
        L.append("|---|" + "---|" * (len(SOLS) + 1))
        out = {}
        for d in dets:
            f = filt_of[d]
            blk = E[f]["consensus"][key]
            vals = [blk.get(s, {}).get(d, {}).get("sysrms_debiased_mas") for s in SOLS]
            null = blk["crds"].get(d, {}).get("null_rms_mas")
            out[d] = dict(zip(SOLS, vals), null=null)
            L.append("| " + " | ".join([d] + [f2(v) for v in vals] + [f2(null)]) + " |")
        L.append("")
        return out

    S["consensus"] = {}
    S["consensus"]["loo_similarity"] = emp_table(
        "loo_similarity", "(b1) Frame − leave-one-out tile consensus; per-frame similarity removed; "
        "debiased binned RMS (16x16 cells), mas")
    S["consensus"]["loo_shift"] = emp_table(
        "loo_shift", "(b2) Frame − leave-one-out tile consensus; per-frame shift removed")
    S["consensus"]["dq_shift"] = emp_table(
        "dq_shift", "(b3) Frame − official tile consensus (data-qa estimator); per-frame shift removed")
    S["consensus"]["dq_similarity"] = emp_table(
        "dq_similarity", "(b4) Frame − official tile consensus (data-qa estimator); per-frame "
        "similarity removed")

    L.append("### (b5) Per-tile consistency (8x8 cells per tile; loo_similarity): median change "
             "of the binned RMS relative to CRDS [mas] / fraction of tiles where it decreased\n")
    sols2 = ["gdc_v1", "gdc_v2", "dqmap_cv", "loomap_cv"]
    L.append("| det | CRDS tile debiased median | " + " | ".join(sols2) + " |")
    L.append("|---|---|" + "---|" * len(sols2))
    S["per_tile"] = {}
    for key in ("loo_similarity", "loo_shift", "dq_shift"):
        S["per_tile"][key] = {}
        for d in dets:
            f = filt_of[d]
            pt = E[f]["per_tile"][key]
            row = {s: pt.get(s, {}).get(d) for s in sols2}
            S["per_tile"][key][d] = row
            if key == "loo_similarity":
                base = next((v["crds_tile_debiased_median_mas"] for v in row.values() if v), None)
                L.append("| " + " | ".join(
                    [d, f2(base)] + [f"{f2(v['median_delta_rms_mas'])} / {v['frac_tiles_reduced']:.2f}"
                                     if v else "–" for v in row.values()]) + " |")
    L.append("")

    # refcat
    L.append("### (b6) Frame − Gaia DR3/VIRAC2 refcat (PM-propagated); per-frame similarity removed; "
             "debiased binned RMS, 8x8 and 4x4 cells [mas]\n")
    rs = ["crds", "gdc_v1", "gdc_v2", "dqmap_cv", "loomap_cv"]
    L.append("| det | n | per-star RMS | " + " | ".join(f"{s} 8x8" for s in rs) + " | null 8x8 | "
             + " | ".join(f"{s} 4x4" for s in rs) + " | null 4x4 |")
    L.append("|---|---|---|" + "---|" * (2 * len(rs) + 2))
    S["refcat"] = {}
    for mode in ("similarity", "shift"):
        S["refcat"][mode] = {}
        for d in dets:
            f = filt_of[d]
            R = E[f].get("refcat", {}).get(mode, {})
            r8 = {s: R.get("nb8", {}).get(s, {}).get(d, {}).get("sysrms_debiased_mas") for s in rs}
            r4 = {s: R.get("nb4", {}).get(s, {}).get(d, {}).get("sysrms_debiased_mas") for s in rs}
            c8 = R.get("nb8", {}).get("crds", {}).get(d, {})
            c4 = R.get("nb4", {}).get("crds", {}).get(d, {})
            S["refcat"][mode][d] = dict(nb8=r8, nb4=r4, null8=c8.get("null_rms_mas"),
                                        null4=c4.get("null_rms_mas"), n=c8.get("n"),
                                        perstar=c8.get("perstar_rms_mas"))
            if mode == "similarity":
                L.append("| " + " | ".join(
                    [d, str(c8.get("n")), f2(c8.get("perstar_rms_mas"))] + [f2(v) for v in r8.values()]
                    + [f2(c8.get("null_rms_mas"))] + [f2(v) for v in r4.values()]
                    + [f2(c4.get("null_rms_mas"))]) + " |")
    gp = E["F212N"].get("refcat", {}).get("gdc_truth_prediction", {})
    S["refcat"]["gdc_truth_prediction"] = gp
    L.append("")
    L.append("Predicted refcat map if STDGDC v1 were the truth (8x8, similarity removed): "
             + ", ".join(f"{d} {f2(v['similarity']['pred_rms_mas'])} mas (r with CRDS map "
                         f"{f2(v['similarity'].get('corr_crdsmap_vs_pred'))})" for d, v in gp.items()))
    L.append("")

    # recovery
    L.append("### Dilution: recovery slope (measured / injected, 16x16 cells) of a noise-free "
             "injected field\n")
    L.append("| det | inject CRDS−STDGDC: dq_shift | loo_shift | loo_similarity | inject measured map "
             "shape: dq_shift | loo_shift | loo_similarity |")
    L.append("|---|---|---|---|---|---|---|")
    S["recovery"] = {}
    for d in dets:
        f = filt_of[d]
        rc = E[f]["recovery"]
        g = rc.get("gdc", {}); m = rc.get("dqmap", {})
        row = {f"gdc_{k}": g.get(k, {}).get(d, {}).get("slope") for k in ("dq_shift", "loo_shift", "loo_similarity")}
        row.update({f"map_{k}": m.get(k, {}).get(d, {}).get("slope") for k in ("dq_shift", "loo_shift", "loo_similarity")})
        row.update({f"corr_gdc_{k}": g.get(k, {}).get(d, {}).get("corr") for k in ("dq_shift", "loo_shift")})
        S["recovery"][d] = row
        L.append("| " + " | ".join([d] + [f2(row[k]) for k in (
            "gdc_dq_shift", "gdc_loo_shift", "gdc_loo_similarity", "map_dq_shift", "map_loo_shift",
            "map_loo_similarity")]) + " |")
    L.append("")

    # split half
    L.append("### Split-half repeatability of the data-qa-estimator map (fold 0 vs fold 1 tiles)\n")
    L.append("| det | r | RMS fold 0 | RMS fold 1 | RMS(half difference) |")
    L.append("|---|---|---|---|---|")
    S["splithalf"] = {}
    for d in dets:
        v = E[filt_of[d]]["dqmap_splithalf"].get(d)
        S["splithalf"][d] = v
        if v:
            L.append(f"| {d} | {f2(v['r_splithalf'])} | {f2(v['rms_fold0'])} | {f2(v['rms_fold1'])} | "
                     f"{f2(v['rms_halfdiff'])} |")
    L.append("")
    if CI:
        L.append("### Catalogue-level impact (consensus-star shift, per-tile mean removed) [mas]\n")
        L.append("| filter | correction | stars | RMS | p99 | max | per-detection RMS |")
        L.append("|---|---|---|---|---|---|---|")
        for f, v in CI.items():
            for k in ("gdc_v1", "dqmap", "dqmap_dec"):
                if k in v:
                    x = v[k]
                    L.append(f"| {f} | {k} | {x['n_stars']} | {f2(x['star_shift_rms_mas'])} | "
                             f"{f2(x['star_shift_p99_mas'])} | {f2(x['star_shift_max_mas'])} | "
                             f"{f2(x['detection_rms_mas'])} |")
        S["catalog_impact"] = CI
    rp_path = os.path.join(dc, "empirical", "refcat_paired.json")
    if os.path.exists(rp_path):
        RP = json.load(open(rp_path))
        L.append("### (b7) Paired regression against Gaia DR3/VIRAC2: binned CRDS residual map (frame − refcat, "
                 "similarity removed) regressed on the residual each correction predicts; beta ± tile-bootstrap "
                 "error (1 = full amplitude, 0 = no relation)\n")
        for nb in (8, 4):
            keys = [k for k in ("gdc_v1", "gdc_v2", "dqmap", "loomap")]
            L.append(f"{nb}x{nb} cells:\n")
            L.append("| det | " + " | ".join(f"{k} beta" for k in keys) + " | " +
                     " | ".join(f"{k} r" for k in keys) + " |")
            L.append("|---|" + "---|" * (2 * len(keys)))
            for f in RP:
                for d, v in RP[f].items():
                    cells = []
                    for k in keys:
                        x = v.get(f"{k}_nb{nb}")
                        cells.append("–" if x is None else f"{x['beta']:.2f} ± {x['beta_err']:.2f}")
                    for k in keys:
                        x = v.get(f"{k}_nb{nb}")
                        cells.append("–" if x is None or "corr" not in x else f"{x['corr']:.2f}")
                    L.append(f"| {f} {d} | " + " | ".join(cells) + " |")
            L.append("")
        S["refcat_paired"] = RP
    with open(os.path.join(dc, "tables.md"), "w") as fh:
        fh.write("\n".join(L) + "\n")
    with open(os.path.join(dc, "summary.json"), "w") as fh:
        json.dump(S, fh, indent=1, default=float)
    print("\n".join(L))


if __name__ == "__main__":
    main()
