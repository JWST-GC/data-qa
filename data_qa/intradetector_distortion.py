"""Intra-detector residual distortion stacked over every pointing of a program.

Stage 8's intra-detector map (``data_qa.intradetector``) stacks one tile's ~6 exposures per
detector.  This tool runs the same per-frame measurement (each per-frame catalogue minus its tile's
consensus catalogue, bulk offset removed by the pair-histogram peak, mutual-NN, flux-vetted,
residual in detector axes) on every pointing found in a catalogue directory and stacks the
residuals per detector and per filter, so a residual distortion that the WCS distortion model
leaves in every exposure averages up while per-tile noise and tile-specific structure average down.

Outputs, per filter, in ``--outdir``:

* ``intradet_<filt>_allpointings.png`` -- stacked per-detector quiver maps + median-residual
  profiles (``intradetector.plot_stacks``);
* ``intradet_<filt>_consistency.png`` -- per pointing and detector, the debiased map RMS and the
  correlation of that pointing's binned map with the stack of all OTHER pointings (a pattern that
  repeats from tile to tile gives r > 0; tile-specific structure or noise gives r ~ 0);
* ``intradet_<filt>.json`` -- per-detector stacked numbers and binned maps, per-pointing numbers.

Usage::

    python -m data_qa.intradetector_distortion \\
        --catdir /orange/adamginsburg/jwst/gc-treasury/catalogs_rollcorr/v1_dataqa346_perframe \\
        --filters F212N F480M --outdir <dir> --nproc 16
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

import numpy as np

from . import intradetector as I

DEFAULT_CATDIR = "/orange/adamginsburg/jwst/gc-treasury/catalogs_rollcorr/v1_dataqa346_perframe"
_OBS_RE = re.compile(r"_merged_o(\d{3})_indivexp_merged_m3_dao_basic_vetted\.fits$")


def pointings(catdir, filt):
    """{obs: (consensus path, [frame paths])} for every obs with a vetted consensus catalogue."""
    fl = filt.lower()
    out = {}
    for c in sorted(glob.glob(os.path.join(catdir, f"{fl}_merged_o*_indivexp_merged_m3_dao_basic_vetted.fits"))):
        m = _OBS_RE.search(os.path.basename(c))
        if not m:
            continue
        obs = m.group(1)
        frames = sorted(p for p in glob.glob(os.path.join(catdir, f"{fl}_nrc*_o{obs}_visit*_m3_daophot_basic.fits")))
        if frames:
            out[obs] = (c, frames)
    return out


def measure_pointing(args):
    """Per-detector concatenated residuals of one pointing: (obs, {det: dict(x,y,dx,dy)}, info)."""
    obs, cons, frames = args
    ref = I.load_consensus(cons)
    per_det, failed, bulks = {}, {}, []
    for p in frames:
        fr = I.load_frame(p)
        r = I.frame_residuals(fr, ref)
        per_det.setdefault(fr["detector"] or "unknown", []).append(r)
        if "dx" in r:
            bulks.append(float(np.hypot(r["bulk_dra"], r["bulk_ddec"])))
        else:
            failed[r["reason"]] = failed.get(r["reason"], 0) + 1
    by_det = {d: I.concat(rs) for d, rs in per_det.items()}
    info = dict(n_frames=len(frames), n_frames_used=len(bulks), frames_failed=failed,
                frame_bulk_median_mas=float(np.median(bulks)) if bulks else None)
    return obs, by_det, info


def _corr(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 10:
        return float("nan")
    a = a[ok] - a[ok].mean(); b = b[ok] - b[ok].mean()
    den = np.sqrt((a ** 2).sum() * (b ** 2).sum())
    return float((a * b).sum() / den) if den > 0 else float("nan")


def run_filter(catdir, filt, outdir, nb, nproc, cache=True):
    pts = pointings(catdir, filt)
    if not pts:
        print(f"[{filt}] no pointings in {catdir}")
        return None
    os.makedirs(outdir, exist_ok=True)
    cpath = os.path.join(outdir, f"intradet_{filt.lower()}_residuals.npz")
    results = {}
    if cache and os.path.exists(cpath):
        z = np.load(cpath, allow_pickle=True)
        results = z["results"].item()
        print(f"[{filt}] loaded {len(results)} cached pointings from {cpath}")
    todo = [(o, c, f) for o, (c, f) in pts.items() if o not in results]
    if todo:
        print(f"[{filt}] measuring {len(todo)} pointings with {nproc} processes", flush=True)
        if nproc > 1:
            from multiprocessing import Pool
            with Pool(nproc) as pool:
                for obs, by_det, info in pool.imap_unordered(measure_pointing, todo):
                    results[obs] = (by_det, info)
                    print(f"  o{obs}: {info['n_frames_used']}/{info['n_frames']} frames", flush=True)
        else:
            for t in todo:
                obs, by_det, info = measure_pointing(t)
                results[obs] = (by_det, info)
        np.savez(cpath, results=np.array(results, dtype=object))
    # all-pointing stack per detector
    dets = sorted({d for by_det, _ in results.values() for d in by_det if d != "unknown"})
    allres = {d: I.concat([by_det[d] for by_det, _ in results.values() if d in by_det])
              for d in dets}
    stacks = I.stack_maps(allres, nb=nb)
    # per pointing: its own tile-level map (stage-8 grid) and its correlation with the stack of
    # all OTHER pointings on the aggregate grid
    per_pt = {}
    for obs, (by_det, info) in sorted(results.items()):
        row = dict(info)
        for d in dets:
            r = by_det.get(d)
            if r is None or len(r["dx"]) < I.MIN_PER_CELL * 4:
                continue
            tile = I.stack_detector(r["x"], r["y"], r["dx"], r["dy"], n_perm=5)
            others = I.concat([b[d] for o2, (b, _) in results.items() if o2 != obs and d in b])
            lx, ly, _ = I.binned_map(others["x"], others["y"], others["dx"], others["dy"], nb=nb)
            px, py, _ = I.binned_map(r["x"], r["y"], r["dx"], r["dy"], nb=nb, minn=5)
            row[d] = dict(n=tile["n"], sysrms_mas=round(tile["sysrms_mas"], 3),
                          null_rms_mas=round(tile["null_rms_mas"], 3),
                          sysrms_debiased_mas=round(tile["sysrms_debiased_mas"], 3),
                          r_vs_others=round(_corr(np.r_[px.ravel(), py.ravel()],
                                                  np.r_[lx.ravel(), ly.ravel()]), 3))
        per_pt[obs] = row
    summ = I.summary_json(stacks)
    for d in dets:
        tiles = [per_pt[o][d] for o in per_pt if d in per_pt[o]]
        summ[d]["n_pointings"] = len(tiles)
        summ[d]["tile_sysrms_debiased_median_mas"] = (
            round(float(np.median([t["sysrms_debiased_mas"] for t in tiles])), 3) if tiles else None)
        rs = [t["r_vs_others"] for t in tiles if np.isfinite(t["r_vs_others"])]
        summ[d]["tile_r_vs_others_median"] = round(float(np.median(rs)), 3) if rs else None
    out = dict(filter=filt, catdir=catdir, nbin=nb, cell_px=I.DET_NPIX / nb,
               n_pointings=len(results), method=dict(
                   match_arcsec=I.MATCH_ARCSEC, min_frame_snr=I.MIN_FRAME_SNR,
                   min_ref_snr=I.MIN_REF_SNR, min_ref_nmatch=I.MIN_REF_NMATCH,
                   flux_tol_mag=I.FLUX_TOL_MAG, frame_flags="== 0"),
               detectors=summ, pointings=per_pt)
    jpath = os.path.join(outdir, f"intradet_{filt.lower()}.json")
    with open(jpath, "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    nfr = sum(i["n_frames_used"] for _, i in results.values())
    png = I.plot_stacks(stacks, allres,
                        f"GC Treasury 10678 — intra-detector residual, {filt}: each frame − its "
                        f"tile consensus (bulk removed), stacked over {len(results)} pointings / "
                        f"{nfr} frames", os.path.join(outdir, f"intradet_{filt.lower()}_allpointings.png"),
                        nb=nb)
    cpng = plot_consistency(per_pt, dets, summ, filt, os.path.join(outdir, f"intradet_{filt.lower()}_consistency.png"))
    print(f"[{filt}] wrote {jpath}, {png}, {cpng}")
    return out


def plot_consistency(per_pt, dets, summ, filt, out_path):
    """Left: per-pointing debiased tile-map RMS per detector, with the all-pointing stack's value.
    Right: per-pointing correlation of the tile map with the stack of the other pointings."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    xs = np.arange(len(dets))
    rng = np.random.default_rng(0)
    for k, d in enumerate(dets):
        tiles = [per_pt[o][d] for o in per_pt if d in per_pt[o]]
        j = k + rng.uniform(-0.25, 0.25, len(tiles))
        ax[0].scatter(j, [t["sysrms_debiased_mas"] for t in tiles], s=8, alpha=0.6, color=f"C{k % 10}")
        ax[0].plot([k - 0.35, k + 0.35], [summ[d]["sysrms_debiased_mas"]] * 2, color="k", lw=2)
        ax[1].scatter(j, [t["r_vs_others"] for t in tiles], s=8, alpha=0.6, color=f"C{k % 10}")
        med = summ[d].get("tile_r_vs_others_median")
        if med is not None:
            ax[1].plot([k - 0.35, k + 0.35], [med] * 2, color="k", lw=2)
    ax[0].set_ylabel("debiased binned-map RMS [mas]")
    ax[0].set_title("single tile (points) vs all-pointing stack (black bar)", fontsize=10)
    ax[1].axhline(0, color="0.5", lw=0.7)
    ax[1].set_ylabel("Pearson r, tile map vs stack of other pointings")
    ax[1].set_title("tile-to-tile repeatability of the pattern (black = median)", fontsize=10)
    for a in ax:
        a.set_xticks(xs); a.set_xticklabels(dets, rotation=30, fontsize=8)
    fig.suptitle(f"GC Treasury 10678 — intra-detector residual consistency, {filt}", fontsize=11)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--catdir", default=DEFAULT_CATDIR)
    ap.add_argument("--filters", nargs="+", default=["F212N", "F480M"])
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--nbin", type=int, default=16, help="detector grid per axis (default 16)")
    ap.add_argument("--nproc", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    ap.add_argument("--no-cache", action="store_true", help="re-measure every pointing")
    a = ap.parse_args(argv)
    os.makedirs(a.outdir, exist_ok=True)
    for f in a.filters:
        run_filter(a.catdir, f, a.outdir, a.nbin, a.nproc, cache=not a.no_cache)


if __name__ == "__main__":
    main()
