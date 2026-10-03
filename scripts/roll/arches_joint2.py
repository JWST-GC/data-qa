"""Arches 2045 o001 F212N m7 nrca+nrcb rotation vs Gaia DR3 / VIRAC2.

For each applied pre-rotation (0, 8.3, 13.0 arcsec, +N->E, about the tangent point), refit with
  (a) single-shift similarity (common rotation, scale, one shift), and
  (b) joint-module model (common rotation and scale, free per-module shift),
bootstrapping over unique reference stars.  Pair set: swept bulk -> iterate_fit (mutual-NN 80 mas,
de-blend, flux-vet, 3-sigma clip) from measure_rotation.py; the joint model is fitted on the pairs
kept by the single-shift clip.
"""
import sys, os, json
import numpy as np
from astropy.table import Table
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import measure_rotation as M

EPOCH = 2023.6339
RA0, DEC0 = 266.4677709391126, -28.85953243375873
base = "/orange/adamginsburg/jwst/arches/catalogs"
NBOOT = 300


def load(fn):
    t = Table.read(fn)
    sc = t["skycoord"]
    f, fe = np.asarray(t["flux"], float), np.asarray(t["flux_err"], float)
    g = np.isfinite(f) & (f > 0) & (f / fe > 20)
    x, y = M.tangent(sc.ra.deg[g], sc.dec.deg[g], RA0, DEC0)
    return x, y, -2.5 * np.log10(f[g])


def rot(theta_arcsec):
    """Similarity params that rotate positions by +theta (N->E) in the measure_rotation convention."""
    a = np.radians(theta_arcsec / 3600)
    return np.array([np.cos(a), -np.sin(a), 0.0, 0.0])


raw = {}
for mod in ("nrca", "nrcb"):
    raw[mod] = load(f"{base}/f212n_{mod}_indivexp_merged_resbgsub_m7_dao_basic_vetted.fits")
refs = {}
for kind in ("gaia", "virac2"):
    R = M.ref_at_epoch(M.query_ref(kind, "2045_001", RA0, DEC0, 0.0), kind, EPOCH)
    R["x"], R["y"] = M.tangent(R["ra"], R["dec"], RA0, DEC0)
    refs[kind] = R

out = {}
for pre in (0.0, 8.3, 13.0):
    cats = []
    for mod, (x, y, m) in raw.items():
        xr, yr = M.apply_sim(rot(pre), x, y)
        cats.append(dict(x=xr, y=yr, mag=m, epoch=EPOCH, det=mod, exp="all"))
    allx = np.concatenate([c["x"] for c in cats]); ally = np.concatenate([c["y"] for c in cats])
    for kind, R in refs.items():
        b, _ = M.swept_bulk(allx, ally, R["x"], R["y"])
        fit, P, keep, info, hist = M.iterate_fit(cats, {round(EPOCH, 3): R},
                                                 np.array([1, 0, b["dx"], b["dy"]]), 0.08, True, 20, nboot=NBOOT)
        mod = np.array([d[:4] for d in P["det"]])
        J = M.fit_joint_modules(P["x"], P["y"], P["X"], P["Y"], mod, keep)
        u_sid, inv = np.unique(P["sid"][keep], return_inverse=True)
        idx_k = np.where(keep)[0]
        groups = [idx_k[inv == k] for k in range(len(u_sid))]
        bt = []
        for _ in range(NBOOT):
            pick = np.concatenate([groups[k] for k in M.RNG.integers(0, len(groups), len(groups))])
            kk = np.zeros(len(keep), bool); kk[pick] = True
            bt.append(M.fit_joint_modules(P["x"], P["y"], P["X"], P["Y"], mod, kk)["theta_arcsec"])
        r = dict(pre_rotation=pre, ref=kind, n_pairs_after_fluxvet=int(len(keep)), n_kept=int(keep.sum()),
                 clip_fraction=float(1 - keep.mean()), n_ref_stars=int(len(u_sid)),
                 n_nrca=int(((mod == "nrca") & keep).sum()), n_nrcb=int(((mod == "nrcb") & keep).sum()),
                 single_theta=fit["theta_arcsec"], single_err=fit["theta_arcsec_err"],
                 joint_theta=J["theta_arcsec"], joint_err=float(np.std(bt)), joint_scale_ppm=J["scale_ppm"],
                 ab_dx_mas=J["dx_nrca_mas"] - J["dx_nrcb_mas"], ab_dy_mas=J["dy_nrca_mas"] - J["dy_nrcb_mas"],
                 rms_sim_mas=fit["rms_sim_mas"])
        out[f"{pre}/{kind}"] = r
        print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()}), flush=True)
json.dump(out, open(os.path.join(M.HERE, "arches_joint2.json"), "w"), indent=1)
