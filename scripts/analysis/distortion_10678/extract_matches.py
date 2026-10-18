"""Extract per-detection matched pairs (frame star <-> tile consensus star) for 10678.

Same per-frame selection as data_qa.intradetector.frame_residuals (S/N, flags, qfit cuts;
pair-histogram bulk shift; mutual-NN within 0.06"; 0.2-mag flux vet; 5-sigma clip), but it keeps
the consensus-star index and each frame's Jacobian so that the residuals can be re-computed under
a different distortion correction and against a leave-one-out consensus (empirical_test.py).

Output: <outdir>/matches_<filt>.npz with one row per kept detection:
  obs, frame (global frame index), det (index into DETS), x, y, star (consensus row index within
  the obs; unique key = (obs, star)), sxi, seta  (frame - consensus on the tangent plane after the
  frame's bulk shift, arcsec), and per frame: frame_name, frame_det, J (pix/arcsec, 2x2),
  pixscale (arcsec/pix, header PIXSCALE), bulk_dra/bulk_ddec (mas).

Usage:
  python extract_matches.py --filters F212N F480M --outdir <distcmp>/matches --nproc 16
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
from data_qa import intradetector as I                 # noqa: E402
from data_qa import intradetector_distortion as ID     # noqa: E402

DETS = ["nrca1", "nrca2", "nrca3", "nrca4", "nrcb1", "nrcb2", "nrcb3", "nrcb4",
        "nrcalong", "nrcblong"]


def frame_pairs(frame, ref):
    """Replica of I.frame_residuals that also returns the consensus index, Jacobian and the
    tangent-plane residual.  Returns None when the frame cannot be measured."""
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    from data_qa import astrometry_audit as aa
    if len(frame["x"]) < I.MIN_PAIRS or len(ref["ra"]) < I.MIN_PAIRS:
        return None
    ra0 = float(np.median(frame["ra"])); dec0 = float(np.median(frame["dec"]))
    cosd = float(np.cos(np.radians(dec0)))
    pad = 2.0 / 3600.0
    near = np.nonzero((ref["ra"] > frame["ra"].min() - pad / cosd)
                      & (ref["ra"] < frame["ra"].max() + pad / cosd)
                      & (ref["dec"] > frame["dec"].min() - pad)
                      & (ref["dec"] < frame["dec"].max() + pad))[0]
    rra, rde, rfl = ref["ra"][near], ref["dec"][near], ref["flux"][near]
    if len(rra) < I.MIN_PAIRS:
        return None
    fsc = SkyCoord(frame["ra"] * u.deg, frame["dec"] * u.deg)
    rsc = SkyCoord(rra * u.deg, rde * u.deg)
    xc = aa.xcorr(fsc, rsc, maxsep=I.XCORR_MAXSEP_ARCSEC * u.arcsec, binarc=0.02)
    if not xc or xc.get("peak_ratio", 0) < I.MIN_PEAK_RATIO:
        return None
    sra = frame["ra"] + xc["dra"] / 3.6e6 / cosd
    sde = frame["dec"] + xc["ddec"] / 3.6e6
    ia, ib = I._mutual_nn(SkyCoord(sra * u.deg, sde * u.deg), rsc, I.MATCH_ARCSEC)
    if len(ia) < I.MIN_PAIRS:
        return None
    with np.errstate(invalid="ignore", divide="ignore"):
        dm = -2.5 * np.log10(frame["flux"][ia] / rfl[ib])
    good = np.isfinite(dm)
    if good.sum() < I.MIN_PAIRS:
        return None
    good &= np.abs(dm - np.median(dm[good])) < I.FLUX_TOL_MAG
    ia, ib = ia[good], ib[good]
    if len(ia) < I.MIN_PAIRS:
        return None
    dxi = (sra[ia] - rra[ib]) * cosd * 3600.0
    deta = (sde[ia] - rde[ib]) * 3600.0
    xi = (rra[ib] - ra0) * cosd * 3600.0; eta = (rde[ib] - dec0) * 3600.0
    A = np.column_stack([np.ones_like(xi), xi, eta])
    cx, *_ = np.linalg.lstsq(A, frame["x"][ia], rcond=None)
    cy, *_ = np.linalg.lstsq(A, frame["y"][ia], rcond=None)
    J = np.array([[cx[1], cx[2]], [cy[1], cy[2]]])
    pix = J @ np.vstack([dxi, deta])
    ps = frame.get("pixscale")
    if not (ps and np.isfinite(ps)):
        ps = 1.0 / np.sqrt(abs(np.linalg.det(J)))
    dx = pix[0] * ps * 1000.0; dy = pix[1] * ps * 1000.0
    r = np.hypot(dx - np.median(dx), dy - np.median(dy))
    s = 1.4826 * np.median(r) / 1.1774 if len(r) else 0.0
    keep = r < max(5.0 * s, 1.0)
    return dict(x=frame["x"][ia][keep], y=frame["y"][ia][keep], star=near[ib][keep],
                sxi=dxi[keep], seta=deta[keep], J=J, pixscale=float(ps),
                bulk_dra=float(xc["dra"]), bulk_ddec=float(xc["ddec"]))


def measure_obs(args):
    obs, cons, frames = args
    ref = I.load_consensus(cons)
    out = []
    for p in frames:
        fr = I.load_frame(p)
        r = frame_pairs(fr, ref)
        if r is None:
            continue
        r["name"] = os.path.basename(p)
        r["det"] = fr["detector"]
        out.append(r)
    return obs, out


def run(filt, catdir, outdir, nproc):
    pts = ID.pointings(catdir, filt)
    todo = [(o, c, f) for o, (c, f) in sorted(pts.items())]
    print(f"[{filt}] {len(todo)} pointings", flush=True)
    res = {}
    if nproc > 1:
        from multiprocessing import Pool
        with Pool(nproc) as pool:
            for obs, out in pool.imap_unordered(measure_obs, todo):
                res[obs] = out
                print(f"  o{obs}: {len(out)} frames", flush=True)
    else:
        for t in todo:
            obs, out = measure_obs(t)
            res[obs] = out
    cols = {k: [] for k in ("obs", "frame", "det", "x", "y", "star", "sxi", "seta")}
    fnames, fdet, fJ, fps, fbulk = [], [], [], [], []
    k = 0
    for obs in sorted(res):
        for r in res[obs]:
            n = len(r["x"])
            cols["obs"].append(np.full(n, int(obs), np.int16))
            cols["frame"].append(np.full(n, k, np.int32))
            cols["det"].append(np.full(n, DETS.index(r["det"]), np.int8))
            cols["x"].append(r["x"].astype(np.float32)); cols["y"].append(r["y"].astype(np.float32))
            cols["star"].append(r["star"].astype(np.int32))
            cols["sxi"].append(r["sxi"]); cols["seta"].append(r["seta"])
            fnames.append(r["name"]); fdet.append(DETS.index(r["det"])); fJ.append(r["J"])
            fps.append(r["pixscale"]); fbulk.append((r["bulk_dra"], r["bulk_ddec"]))
            k += 1
    arr = {c: np.concatenate(v) for c, v in cols.items()}
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"matches_{filt.lower()}.npz")
    np.savez(path, **arr, frame_name=np.array(fnames), frame_det=np.array(fdet, np.int8),
             frame_J=np.array(fJ), frame_pixscale=np.array(fps), frame_bulk=np.array(fbulk),
             catdir=catdir)
    print(f"[{filt}] wrote {path}: {len(arr['x'])} detections in {k} frames", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--catdir", default=ID.DEFAULT_CATDIR)
    ap.add_argument("--filters", nargs="+", default=["F212N", "F480M"])
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--nproc", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    a = ap.parse_args()
    for f in a.filters:
        run(f, a.catdir, a.outdir, a.nproc)


if __name__ == "__main__":
    main()
