"""Per-frame matches against the absolute reference catalogue (Gaia DR3 + VIRAC2 fill,
PM-propagated to the observation epoch: catalogs/gaia_virac2_refcat_epoch*_o<obs>.fits).

Frame stars use the same cuts as data_qa.intradetector.load_frame (flags == 0, S/N > 30,
qfit <= 0.2).  Per frame: pair-histogram bulk shift (astrometry_audit.xcorr, 0.5" search, 0.02"
bins), mutual-NN within MATCH_ARCSEC, 5-sigma clip.  No flux vet (refmag is VIRAC2 J or Gaia G,
which do not predict F212N/F480M fluxes in the extincted GC field).

Output: <outdir>/refmatches_<filt>.npz with per-detection obs, frame, det, x, y, sxi, seta
(frame - reference on the tangent plane after the bulk shift, arcsec), refsrc (0 = Gaia,
1 = VIRAC2, 2 = other) and per-frame frame_name, frame_J (pix/arcsec), frame_pixscale.

Usage:
  python extract_refcat.py --filters F212N F480M --outdir <distcmp>/matches --nproc 16
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".."))
from data_qa import intradetector as I                 # noqa: E402
from data_qa import intradetector_distortion as ID     # noqa: E402
from extract_matches import DETS                       # noqa: E402

REFDIR = "/orange/adamginsburg/jwst/gc-treasury/catalogs"
MATCH_ARCSEC = 0.10


def load_ref(obs):
    from astropy.table import Table
    f = sorted(glob.glob(f"{REFDIR}/gaia_virac2_refcat_epoch*_o{obs}.fits"))
    if len(f) != 1:
        raise FileNotFoundError(f"refcat for o{obs}: {f}")
    t = Table.read(f[0])
    src = np.array([str(s).lower() for s in t["source"]])
    code = np.where(np.char.find(src, "gaia") >= 0, 0, np.where(np.char.find(src, "virac") >= 0, 1, 2))
    return dict(ra=np.asarray(t["RA"], float), dec=np.asarray(t["DEC"], float), src=code,
                path=f[0])


def frame_ref_pairs(frame, ref):
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    from data_qa import astrometry_audit as aa
    if len(frame["x"]) < I.MIN_PAIRS:
        return None
    ra0 = float(np.median(frame["ra"])); dec0 = float(np.median(frame["dec"]))
    cosd = float(np.cos(np.radians(dec0)))
    pad = 2.0 / 3600.0
    near = np.nonzero((ref["ra"] > frame["ra"].min() - pad / cosd)
                      & (ref["ra"] < frame["ra"].max() + pad / cosd)
                      & (ref["dec"] > frame["dec"].min() - pad)
                      & (ref["dec"] < frame["dec"].max() + pad))[0]
    if len(near) < I.MIN_PAIRS:
        return None
    rra, rde = ref["ra"][near], ref["dec"][near]
    fsc = SkyCoord(frame["ra"] * u.deg, frame["dec"] * u.deg)
    rsc = SkyCoord(rra * u.deg, rde * u.deg)
    xc = aa.xcorr(fsc, rsc, maxsep=I.XCORR_MAXSEP_ARCSEC * u.arcsec, binarc=0.02)
    if not xc or xc.get("peak_ratio", 0) < I.MIN_PEAK_RATIO:
        return None
    sra = frame["ra"] + xc["dra"] / 3.6e6 / cosd
    sde = frame["dec"] + xc["ddec"] / 3.6e6
    ia, ib = I._mutual_nn(SkyCoord(sra * u.deg, sde * u.deg), rsc, MATCH_ARCSEC)
    if len(ia) < I.MIN_PAIRS:
        return None
    dxi = (sra[ia] - rra[ib]) * cosd * 3600.0
    deta = (sde[ia] - rde[ib]) * 3600.0
    xi = (rra[ib] - ra0) * cosd * 3600.0; eta = (rde[ib] - dec0) * 3600.0
    A = np.column_stack([np.ones_like(xi), xi, eta])
    cx, *_ = np.linalg.lstsq(A, frame["x"][ia], rcond=None)
    cy, *_ = np.linalg.lstsq(A, frame["y"][ia], rcond=None)
    J = np.array([[cx[1], cx[2]], [cy[1], cy[2]]])
    ps = frame.get("pixscale")
    if not (ps and np.isfinite(ps)):
        ps = 1.0 / np.sqrt(abs(np.linalg.det(J)))
    pix = J @ np.vstack([dxi, deta])
    dx = pix[0] * ps * 1000.0; dy = pix[1] * ps * 1000.0
    r = np.hypot(dx - np.median(dx), dy - np.median(dy))
    s = 1.4826 * np.median(r) / 1.1774
    keep = r < max(5.0 * s, 1.0)
    return dict(x=frame["x"][ia][keep], y=frame["y"][ia][keep], sxi=dxi[keep], seta=deta[keep],
                refsrc=ref["src"][near][ib][keep], J=J, pixscale=float(ps))


def measure_obs(args):
    obs, frames = args
    ref = load_ref(obs)
    out = []
    for p in frames:
        fr = I.load_frame(p)
        r = frame_ref_pairs(fr, ref)
        if r is None:
            continue
        r["name"] = os.path.basename(p); r["det"] = fr["detector"]
        out.append(r)
    return obs, out


def run(filt, catdir, outdir, nproc):
    pts = ID.pointings(catdir, filt)
    todo = [(o, f) for o, (c, f) in sorted(pts.items())]
    res = {}
    from multiprocessing import Pool
    with Pool(nproc) as pool:
        for obs, out in pool.imap_unordered(measure_obs, todo):
            res[obs] = out
            print(f"  o{obs}: {len(out)} frames", flush=True)
    cols = {k: [] for k in ("obs", "frame", "det", "x", "y", "sxi", "seta", "refsrc")}
    fnames, fJ, fps = [], [], []
    k = 0
    for obs in sorted(res):
        for r in res[obs]:
            n = len(r["x"])
            cols["obs"].append(np.full(n, int(obs), np.int16))
            cols["frame"].append(np.full(n, k, np.int32))
            cols["det"].append(np.full(n, DETS.index(r["det"]), np.int8))
            cols["x"].append(r["x"].astype(np.float32)); cols["y"].append(r["y"].astype(np.float32))
            cols["sxi"].append(r["sxi"]); cols["seta"].append(r["seta"])
            cols["refsrc"].append(r["refsrc"].astype(np.int8))
            fnames.append(r["name"]); fJ.append(r["J"]); fps.append(r["pixscale"])
            k += 1
    arr = {c: np.concatenate(v) for c, v in cols.items()}
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"refmatches_{filt.lower()}.npz")
    np.savez(path, **arr, frame_name=np.array(fnames), frame_J=np.array(fJ),
             frame_pixscale=np.array(fps))
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
