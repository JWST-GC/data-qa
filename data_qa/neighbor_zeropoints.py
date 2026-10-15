"""Photometric zero points of every tile (and every NIRCam module) of a mosaic program, solved as a
network from the stage-13 neighbour overlaps.

Stage 13 measures, for each pair of overlapping tiles, the median F212N magnitude difference of the
stars they share.  Each pairwise number mixes two tiles' zero points; this tool collects every pair
and solves for one zero point per tile so the field-to-field calibration can be read off directly.
It reuses stage 13's footprints, sample cuts and matching (``diagnostics._overlapping_neighbors``,
``_load_nb_sample``, ``_neighbor_agreement``), and runs them for every filter asked for (default
F212N and F480M) on three catalogue flavours: the merged m8 catalogue and the two per-module (NRCA,
NRCB) m8 catalogues.

Four weighted least-squares models are solved per filter.  Every edge is one pair measurement
``d = median(m_a - m_b)`` (Vega) with weight ``1/sigma^2``, ``sigma = 1.2533 * 1.4826 MAD / sqrt(N)``
(the standard error of a median):

* ``tile``         -- merged catalogues; ``d = z_i - z_j``; one zero point per tile;
* ``tile+AB``      -- per-module catalogues; ``d = z_i + s_a Delta/2 - z_j - s_b Delta/2`` with
  ``s = +1`` for NRCA and -1 for NRCB, so ``Delta`` is one global NRCA - NRCB offset per filter;
* ``tile×module``  -- per-module catalogues; one free zero point per (tile, module); the per-tile
  ``z_A - z_B`` and its covariance-propagated error come from this model.

* ``tile+G``       -- merged catalogues; ``d = z_i - z_j + G . (c_i - c_j)``, where ``c`` is the
  (east, north) arcmin offset of the edge's matched stars from each tile's centre: the tile zero
  points plus one linear within-tile gradient ``G`` shared by every tile (#362).  With every tile at
  the same roll, a gradient and a ramp of tile zero points along the grid predict the same overlap
  differences, so the overlaps alone cannot separate them.  ``G`` is anchored externally: per-star
  F<filt> - VIRAC2 Ks over each tile's whole footprint (Ks 14.5-16.5, PM-propagated to the epoch),
  fitted as ``t_tile + G . c + k (Ks - 15.5) + a (H - Ks - 1.5)`` (stars with VIRAC2 H) with a bootstrap over tiles for the covariance, enters
  the network as a Gaussian prior on ``G``.  The reference fit's tile offsets ``t`` are reported next
  to the ``tile+G`` zero points as an external check.  The H - Ks term absorbs reddening that varies
  across a tile; without it the F480M - Ks anchor took up the extinction gradient.  ``--no-reference`` skips it.

Module edges are every cross-tile module pair that overlaps (NRCA_i-NRCA_j, NRCA_i-NRCB_j, ...) and,
within one tile, NRCA against NRCB over the whole footprint (the modules overlap only where the
dither pattern puts them on the same sky).

Zero points are relative: the gauge is **mean zero over the tiles of each connected component** of
the overlap graph (per (tile, module) node for ``tile×module``), applied with a Lagrange constraint,
so a tile's ``z`` is its offset from the program mean.  Positive ``z`` = that tile's catalogue
magnitudes are fainter than the mean.

Outputs, per filter, in ``--outdir``: ``s13zp_<filt>_edges.json`` (cached pair measurements; a rerun
re-solves without re-matching while the cache key -- catdir, filter, tile set, input
catalogue and footprint size+mtime -- is unchanged, or unless ``--no-cache``), ``s13zp_<filt>.json`` (solutions, residuals,
chi^2/dof, the error floor that brings chi^2/dof to 1, per-tile metadata), and figures:
``s13zp_<filt>_skymap.png``, ``s13zp_<filt>_residuals.png``, ``s13zp_<filt>_modules.png``,
``s13zp_<filt>_vs_time_roll.png``, ``s13zp_<filt>_gradient.png`` (with the reference; its matches
are cached in ``s13zp_<filt>_reference.json``); across filters ``s13zp_summary.png``.

Usage::

    python -m data_qa.neighbor_zeropoints --outdir <dir> --filters F212N F480M --nproc 16
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from multiprocessing import Pool

import numpy as np

from . import diagnostics as D
from .observations import Observation

DEFAULT_CATDIR = "/orange/adamginsburg/jwst/gc-treasury/catalogs"
_CAT_FMT = "basic_{mod}_indivexp_photometry_tables_merged_resbgsub_m8_dedup_o{obs}.fits"
_OBS_RE = re.compile(r"_m8_dedup_o(\d{3})\.fits$")
MODULES = ("nrca", "nrcb")
_MED_SE = 1.2533                # standard error of a median = 1.2533 sigma / sqrt(N) (Gaussian)
_SIGN = {"nrca": 1.0, "nrcb": -1.0}
EDGE_VERSION = 2                # 2: edges carry the sky position (ra, dec) of their matched stars
REF_KS_RANGE = (14.5, 16.5)     # VIRAC2 Ks window for the gradient anchor (F212N saturates ~14.5)
REF_MATCH_AS = 0.1
REF_MIN_STARS = 30
REF_BOOTSTRAP = 300


# ----------------------------------------------------------------------------------- solver
def _components(nodes, edges):
    """Connected components of the edge graph, as lists of node indices."""
    parent = list(range(len(nodes)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    idx = {n: k for k, n in enumerate(nodes)}
    for e in edges:
        ra, rb = find(idx[e["a"]]), find(idx[e["b"]])
        if ra != rb:
            parent[ra] = rb
    comp = {}
    for k in range(len(nodes)):
        comp.setdefault(find(k), []).append(k)
    return sorted(comp.values(), key=len, reverse=True)


def solve_network(edges, nodes=None, node_of=None, sign_of=None, extra_of=None, prior=None):
    """Weighted least squares for relative zero points on an overlap graph.

    ``edges``: dicts with ``a``, ``b`` (node keys), ``d`` (measured ``m_a - m_b``) and ``err``.
    ``node_of(key)`` maps an edge endpoint to the zero-point node it constrains (default: itself);
    ``sign_of(key)`` returns +1/-1/0 for a global term ``Delta`` that enters as ``+s*Delta/2``
    (``None`` = no global term).  ``extra_of(edge)`` returns the design-row entries of further
    global parameters ``g`` (e.g. a within-tile gradient), and ``prior=(mean, cov)`` adds a Gaussian
    prior on them, counted as ``len(g)`` extra observations in chi^2 and dof (the error floor applies
    to the edges only).  Gauge: mean zero over the nodes of each connected component.

    Returns dict(nodes, z, z_err, cov, delta, delta_err, resid, chi2, dof, chi2_dof, components,
    sys_floor), plus g, g_err, g_cov with ``extra_of``, where ``resid`` is ``d - model`` per edge and
    ``sys_floor`` is the error added in quadrature to every edge that brings chi^2/dof to 1 (0 when
    it is already <= 1)."""
    node_of = node_of or (lambda k: k)
    if nodes is None:
        nodes = sorted({node_of(e[s]) for e in edges for s in ("a", "b")})
    nodes = list(nodes)
    idx = {n: k for k, n in enumerate(nodes)}
    nn = len(nodes)
    glob_term = sign_of is not None
    ng = nn + (1 if glob_term else 0)            # first column of the extra global parameters
    nx = len(np.atleast_1d(extra_of(edges[0]))) if extra_of is not None else 0
    npar = ng + nx
    A = np.zeros((len(edges), npar))
    d = np.array([e["d"] for e in edges], float)
    err = np.array([e["err"] for e in edges], float)
    for r, e in enumerate(edges):
        A[r, idx[node_of(e["a"])]] += 1.0
        A[r, idx[node_of(e["b"])]] -= 1.0
        if glob_term:
            A[r, nn] = 0.5 * (sign_of(e["a"]) - sign_of(e["b"]))
        if nx:
            A[r, ng:] = extra_of(e)
    comps = _components(nodes, [dict(a=node_of(e["a"]), b=node_of(e["b"])) for e in edges])
    C = np.zeros((len(comps), npar))
    for k, c in enumerate(comps):
        C[k, c] = 1.0 / len(c)

    if prior is not None:
        pmu = np.asarray(prior[0], float)
        pinv = np.linalg.inv(np.asarray(prior[1], float))

    def _solve(err_eff):
        w = 1.0 / err_eff ** 2
        N = A.T @ (A * w[:, None])
        rhs = A.T @ (w * d)
        if prior is not None:
            N[ng:, ng:] += pinv
            rhs[ng:] += pinv @ pmu
        K = np.block([[N, C.T], [C, np.zeros((len(comps), len(comps)))]])
        Kinv = np.linalg.pinv(K)
        p = (Kinv @ np.r_[rhs, np.zeros(len(comps))])[:npar]
        cov = Kinv[:npar, :npar]
        res = d - A @ p
        chi2 = float(np.sum(w * res ** 2))
        if prior is not None:
            chi2 += float((p[ng:] - pmu) @ pinv @ (p[ng:] - pmu))
        return p, cov, res, chi2

    p, cov, res, chi2 = _solve(err)
    dof = len(edges) - npar + len(comps) + (nx if prior is not None else 0)
    floor = 0.0
    if dof > 0 and chi2 / dof > 1.0:
        lo, hi = 0.0, max(1.0, float(np.max(np.abs(res))))
        for _ in range(60):                     # bisection on chi2/dof(floor) = 1
            mid = 0.5 * (lo + hi)
            if _solve(np.hypot(err, mid))[3] / dof > 1.0:
                lo = mid
            else:
                hi = mid
        floor = hi
    # report uncertainties that include the floor (the formal ones are too small when chi2/dof >> 1)
    pf, covf, _, _ = _solve(np.hypot(err, floor)) if floor > 0 else (p, cov, res, chi2)
    out = dict(nodes=nodes, z=p[:nn], z_err=np.sqrt(np.clip(np.diag(covf)[:nn], 0, None)),
               z_err_formal=np.sqrt(np.clip(np.diag(cov)[:nn], 0, None)),
               cov=covf[:nn, :nn], resid=res, chi2=chi2, dof=int(dof),
               chi2_dof=float(chi2 / dof) if dof > 0 else float("nan"),
               components=[[nodes[i] for i in c] for c in comps], sys_floor=float(floor))
    if glob_term:
        out.update(delta=float(p[nn]), delta_err=float(np.sqrt(max(covf[nn, nn], 0.0))),
                   delta_err_formal=float(np.sqrt(max(cov[nn, nn], 0.0))))
    if nx:
        out.update(g=p[ng:], g_err=np.sqrt(np.clip(np.diag(covf)[ng:], 0, None)),
                   g_cov=covf[ng:, ng:])
    return out


def _mad(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(1.4826 * np.median(np.abs(x - np.median(x)))) if len(x) else float("nan")


# ---------------------------------------------------------------------------- measurements
def _obs(obs):
    return Observation(program="10678", obs=obs, target="GC Treasury", release_field="gc-treasury",
                       instrument="NIRCam", filters=["F212N", "F480M"])


def tiles(catdir):
    """Obs ids with a merged, an NRCA and an NRCB m8_dedup catalogue."""
    out = []
    for p in sorted(glob.glob(os.path.join(catdir, _CAT_FMT.format(mod="merged", obs="*")))):
        m = _OBS_RE.search(os.path.basename(p))
        if m and all(os.path.exists(os.path.join(catdir, _CAT_FMT.format(mod=k, obs=m.group(1))))
                     for k in MODULES):
            out.append(m.group(1))
    return out


def _cat(catdir, mod, obs):
    return os.path.join(catdir, _CAT_FMT.format(mod=mod, obs=obs))


def _edge(res, a, b, kind, sample):
    """One edge; ``ra``/``dec`` is the mean position of the matched stars of ``sample`` (side a),
    where the gradient model evaluates each tile's within-tile offset."""
    sig = _MED_SE * res["dmag_scatter"] / np.sqrt(res["n_pairs"])
    ia = res["ia"]
    return dict(a=a, b=b, kind=kind, d=res["dmag_bulk"], err=float(max(sig, 1e-4)),
                n_pairs=res["n_pairs"], dmag_scatter=res["dmag_scatter"],
                offset_mas=res["offset_mas"], ra=float(np.mean(sample["ra"][ia])),
                dec=float(np.mean(sample["dec"][ia])))


def measure_pair(args):
    """All edges of one tile pair (or of one tile with itself for the intra-tile NRCA-NRCB edge)."""
    catdir, filt, i, j, poly, frame = args
    poly = np.asarray(poly)
    out = []
    if i == j:
        sa = D._load_nb_sample(_cat(catdir, "nrca", i), filt, poly, frame)
        sb = D._load_nb_sample(_cat(catdir, "nrcb", i), filt, poly, frame)
        r = D._neighbor_agreement(sa, sb)
        if r:
            out.append(_edge(r, f"{i}:nrca", f"{i}:nrcb", "intra", sa))
        return out
    s = {}
    for mod in ("merged",) + MODULES:
        for t in (i, j):
            s[(t, mod)] = D._load_nb_sample(_cat(catdir, mod, t), filt, poly, frame)
    r = D._neighbor_agreement(s[(i, "merged")], s[(j, "merged")])
    if r:
        out.append(_edge(r, i, j, "merged", s[(i, "merged")]))
    for ma in MODULES:
        for mb in MODULES:
            r = D._neighbor_agreement(s[(i, ma)], s[(j, mb)])
            if r:
                out.append(_edge(r, f"{i}:{ma}", f"{j}:{mb}", "module", s[(i, ma)]))
    return out


def tile_meta(obs, filt):
    """Footprint vertices, observation date (MJD), visit and PA_APER of one tile's mosaic."""
    from astropy.io import fits
    path = D._footprint_path(_obs(obs), filt)
    meta = dict(obs=obs, vertices=None, mjd=None, date=None, visit=None, pa_aper=None, pa_v3=None)
    if not path:
        return meta
    v = D._sregion_radec(path)
    meta["vertices"] = v.tolist() if v is not None else None
    h0 = fits.getheader(path, 0)
    h1 = fits.getheader(path, "SCI")
    meta.update(mjd=h0.get("EXPSTART"), date=h0.get("DATE-OBS"), visit=h0.get("VISIT_ID"),
                pa_aper=h1.get("PA_APER"), pa_v3=h1.get("PA_V3"))
    return meta


def measure_filter(catdir, filt, nproc=1, obs_list=None):
    """Every pair (and intra-tile) edge for one filter, plus per-tile metadata."""
    obs_list = obs_list or tiles(catdir)
    have = set(obs_list)
    jobs = []
    for i in obs_list:
        mine, nbs = D._overlapping_neighbors(_obs(i), filt)
        if mine is None:
            continue
        ra0, dec0 = float(np.mean(mine[:, 0])), float(np.mean(mine[:, 1]))
        jobs.append((catdir, filt, i, i, D._radec_to_plane(mine, ra0, dec0).tolist(), (ra0, dec0)))
        for nb in nbs:
            if nb["obs"] in have and nb["obs"] > i:
                jobs.append((catdir, filt, i, nb["obs"], nb["overlap_xy"].tolist(), nb["frame"]))
    print(f"[s13zp] {filt}: {len(obs_list)} tiles, {len(jobs)} pair/intra jobs", flush=True)
    if nproc > 1:
        with Pool(nproc) as pool:
            res = pool.map(measure_pair, jobs, chunksize=1)
    else:
        res = [measure_pair(j) for j in jobs]
    edges = [e for r in res for e in r]
    meta = {o: tile_meta(o, filt) for o in obs_list}
    return dict(filter=filt, tiles=obs_list, edges=edges, meta=meta,
                n_pair_jobs=len(jobs))


def tile_centre(meta):
    """(ra, dec) of a tile's footprint centre (vertex mean, as stage 13 uses), or None."""
    v = meta.get("vertices") if meta else None
    if not v:
        return None
    v = np.asarray(v, float)
    return float(np.mean(v[:, 0])), float(np.mean(v[:, 1]))


def _offset_arcmin(ra, dec, centre):
    """(east, north) offset in arcmin of sky positions from a tile centre (tangent-plane approx.)."""
    ra, dec = np.asarray(ra, float), np.asarray(dec, float)
    return ((ra - centre[0]) * np.cos(np.radians(centre[1])) * 60.0, (dec - centre[1]) * 60.0)


def _reference_with_h(o, ep):
    """(SkyCoord, Ks, H) of the stage-3 VIRAC2 reference, PM-propagated to ``ep``; H is NaN where
    VIRAC2 has none.  Same source and propagation as ``diagnostics._stage3_reference``."""
    from astropy.table import Table
    from . import astrometry_audit as aa
    path = (D._viraccache_path(o) or D._fetch_virac2_ks(o)) if ep else None
    if not path:
        return None, None, None
    t = Table.read(path)
    cols = {c.lower(): c for c in t.colnames}
    if "ksmag" not in cols or "hmag" not in cols:
        return None, None, None
    sc, ks = aa.load_reference(path, ep)

    def col(n):
        return np.asarray(np.ma.filled(np.ma.asarray(t[cols[n]], dtype=float), np.nan), float)
    # load_reference keeps the rows with finite coordinates, in order
    good = np.isfinite(col("raj2000" if "raj2000" in cols else "ra")) & \
        np.isfinite(col("dej2000" if "dej2000" in cols else "dec"))
    return sc, ks, col("hmag")[good]


def measure_reference_tile(args):
    """F<filt> − VIRAC2 Ks per matched star over one tile's whole footprint (merged m8 catalogue).

    Same clean-source cuts as the overlaps (``_load_nb_sample``), VIRAC2 PM-propagated to the
    mosaic epoch and paired one-to-one within ``REF_MATCH_AS`` after the stage-3 bulk alignment;
    Ks restricted to ``REF_KS_RANGE``.  Positions are (east, north) arcmin from the tile centre.
    Returns a dict of lists, or None when the tile has no footprint, reference or enough stars."""
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    catdir, filt, obs = args
    o = _obs(obs)
    fp = D._footprint_path(o, filt)
    v = D._sregion_radec(fp) if fp else None
    if v is None:
        return None
    centre = (float(np.mean(v[:, 0])), float(np.mean(v[:, 1])))
    ref_sc, ks, hmag = _reference_with_h(o, D._obs_epoch(o, fp))
    if ref_sc is None or ks is None:
        return None
    smp = D._load_nb_sample(_cat(catdir, "merged", obs), filt, D._radec_to_plane(v, *centre), centre)
    if smp is None or len(smp["ra"]) < REF_MIN_STARS:
        return None
    jsc = SkyCoord(smp["ra"] * u.deg, smp["dec"] * u.deg)
    jal, _, _ = D._stage3_align(jsc, ref_sc)
    ir, ij = D._one_to_one_match(ref_sc, jal, REF_MATCH_AS)
    k, h = np.asarray(ks, float)[ir], hmag[ir]
    keep = np.isfinite(k) & np.isfinite(h) & (k > REF_KS_RANGE[0]) & (k < REF_KS_RANGE[1])
    if keep.sum() < REF_MIN_STARS:
        return None
    ir, ij, k, h = ir[keep], ij[keep], k[keep], h[keep]
    x, y = _offset_arcmin(smp["ra"][ij], smp["dec"][ij], centre)
    return dict(obs=obs, x=x.tolist(), y=y.tolist(), d=(smp["mag"][ij] - k).tolist(), ks=k.tolist(),
                hk=(h - k).tolist())


def measure_reference(catdir, filt, nproc=1, obs_list=None):
    obs_list = obs_list or tiles(catdir)
    jobs = [(catdir, filt, o) for o in obs_list]
    if nproc > 1:
        with Pool(nproc) as pool:
            res = pool.map(measure_reference_tile, jobs, chunksize=1)
    else:
        res = [measure_reference_tile(j) for j in jobs]
    return dict(filter=filt, ks_range=list(REF_KS_RANGE), tiles=[r for r in res if r])


def _within_fit(tidx, X, d, nt, clip=4.0, niter=6):
    """Robust fixed-effects fit ``d = t_tile + X @ beta``: per-tile means are removed (over the kept
    stars) and ``beta`` solved by least squares, clipping at ``clip`` x 1.4826 MAD."""
    keep = np.ones(len(d), bool)
    for _ in range(niter):
        cnt = np.bincount(tidx[keep], minlength=nt).astype(float)
        cnt[cnt == 0] = np.nan
        mx = np.stack([np.bincount(tidx[keep], X[keep, j], nt) for j in range(X.shape[1])], 1) / cnt[:, None]
        md = np.bincount(tidx[keep], d[keep], nt) / cnt
        Xw, dw = X - mx[tidx], d - md[tidx]
        beta, *_ = np.linalg.lstsq(Xw[keep], dw[keep], rcond=None)
        r = dw - Xw @ beta
        sig = 1.4826 * np.median(np.abs(r[keep]))
        new = np.isfinite(r) & (np.abs(r) < clip * sig)
        if np.array_equal(new, keep):
            break
        keep = new
    t = md - mx @ beta
    return beta, t, keep, float(sig)


def fit_reference(ref, rng=None):
    """Within-tile gradient against the external reference: ``d = t_tile + Gx x + Gy y +
    c (Ks − 15.5) + a (H − Ks − 1.5)`` with x east, y north (arcmin from the tile centre).  The
    colour term absorbs reddening that varies across a tile (extinction gradients, which would
    otherwise enter G, most strongly for F480M − Ks).  ``g_cov`` comes from a
    bootstrap over tiles (the gradient varies a little from tile to tile, so per-star errors
    understate it); ``g_cov_formal`` from the per-star scatter.  ``t`` are the tile offsets
    against the reference (mean removed), the external check on the network zero points."""
    tl = [r for r in ref["tiles"] if len(r["d"]) >= REF_MIN_STARS]
    if len(tl) < 3:
        return None
    obs = [r["obs"] for r in tl]
    tidx = np.concatenate([np.full(len(r["d"]), k) for k, r in enumerate(tl)])
    X = np.c_[np.concatenate([r["x"] for r in tl]), np.concatenate([r["y"] for r in tl]),
              np.concatenate([r["ks"] for r in tl]) - 15.5,
              np.concatenate([r["hk"] for r in tl]) - 1.5]
    d = np.concatenate([r["d"] for r in tl])
    beta, t, keep, sig = _within_fit(tidx, X, d, len(tl))
    Xk = X[keep] - np.array([np.mean(X[keep & (tidx == k)], 0) for k in range(len(tl))])[tidx[keep]]
    cov_formal = np.linalg.inv(Xk.T @ Xk) * sig ** 2
    rng = rng or np.random.default_rng(0)
    boots = []
    for _ in range(REF_BOOTSTRAP):
        pick = rng.integers(0, len(tl), len(tl))
        rows = np.concatenate([np.flatnonzero(keep & (tidx == k)) for k in pick])
        newt = np.concatenate([np.full(int(np.sum(keep & (tidx == k))), n) for n, k in enumerate(pick)])
        b, *_ = _within_fit(newt, X[rows], d[rows], len(pick), niter=1)
        boots.append(b)
    cov = np.cov(np.array(boots).T)
    nk = np.bincount(tidx[keep], minlength=len(tl))
    t_err = sig / np.sqrt(np.maximum(nk, 1))
    good = nk > 0
    t = t - np.mean(t[good])
    return dict(g=beta[:2].tolist(), g_cov=cov[:2, :2].tolist(), g_err=np.sqrt(np.diag(cov)[:2]).tolist(),
                g_err_formal=np.sqrt(np.diag(cov_formal)[:2]).tolist(), ks_slope=float(beta[2]),
                hk_slope=float(beta[3]),
                n_stars=int(keep.sum()), n_tiles=int(good.sum()), sigma=sig,
                t={o: (float(t[k]), float(t_err[k])) for k, o in enumerate(obs) if good[k]})


# ------------------------------------------------------------------------------- solutions
def _gradient_columns(meta):
    """``extra_of`` for the within-tile gradient: an edge ``d = m_a − m_b`` measured at sky
    position p gains ``G · (c_a − c_b)``, with c the (east, north) arcmin offset of p from each
    tile's centre."""
    cen = {o: tile_centre(mm) for o, mm in meta.items()}

    def cols(e):
        xa, ya = _offset_arcmin(e["ra"], e["dec"], cen[e["a"]])
        xb, yb = _offset_arcmin(e["ra"], e["dec"], cen[e["b"]])
        return np.array([xa - xb, ya - yb])
    return cols, cen


def solve_filter(m, ref=None):
    """The models of the module docstring for one filter's measurements; with ``ref`` (the
    ``fit_reference`` result) also ``tile+G``."""
    edges = m["edges"]
    merged = [e for e in edges if e["kind"] == "merged"]
    modular = [e for e in edges if e["kind"] in ("module", "intra")]
    out = dict(filter=m["filter"])
    if merged:
        out["tile"] = solve_network(merged)
    if merged and ref:
        cols, cen = _gradient_columns(m["meta"])
        gm = [e for e in merged if "ra" in e and cen.get(e["a"]) and cen.get(e["b"])]
        if gm:
            out["tile+G"] = solve_network(gm, extra_of=cols, prior=(ref["g"], ref["g_cov"]))
            out["reference"] = ref
    if modular:
        out["tile+AB"] = solve_network(modular, node_of=lambda k: k.split(":")[0],
                                       sign_of=lambda k: _SIGN[k.split(":")[1]])
        tm = solve_network(modular)
        out["tile×module"] = tm
        idx = {n: k for k, n in enumerate(tm["nodes"])}
        ab = {}
        for o in m["tiles"]:
            ka, kb = idx.get(f"{o}:nrca"), idx.get(f"{o}:nrcb")
            if ka is None or kb is None:
                continue
            same = any(ka in c and kb in c for c in
                       [[idx[n] for n in comp] for comp in tm["components"]])
            if not same:
                continue
            var = tm["cov"][ka, ka] + tm["cov"][kb, kb] - 2 * tm["cov"][ka, kb]
            ab[o] = (float(tm["z"][ka] - tm["z"][kb]), float(np.sqrt(max(var, 0.0))))
        out["ab_per_tile"] = ab
    return out


def _jsonable(sol):
    def conv(v):
        if isinstance(v, np.ndarray):
            return v.tolist()
        if isinstance(v, dict):
            return {k: conv(x) for k, x in v.items() if k != "cov"}
        if isinstance(v, (list, tuple)):
            return [conv(x) for x in v]
        if isinstance(v, (np.floating, np.integer)):
            return v.item()
        return v
    return conv(sol)


def summarise(m, sol):
    """Headline numbers for one filter."""
    s = dict(filter=m["filter"], n_tiles=len(m["tiles"]),
             n_edges={k: sum(e["kind"] == k for e in m["edges"]) for k in ("merged", "module", "intra")})
    t = sol.get("tile")
    if t:
        z = np.asarray(t["z"])
        merged_d = np.array([e["d"] for e in m["edges"] if e["kind"] == "merged"])
        zd = dict(zip(t["nodes"], z))
        comps = [dict(n=len(c), obs=list(c), rms=float(np.std([zd[o] for o in c])),
                      range=float(np.ptp([zd[o] for o in c]))) for c in t["components"]]
        # each component has its own mean-zero gauge, so only per-component spreads are physical;
        # the pooled numbers mix gauges and are kept (labelled) for reference only
        s.update(tile_zp_rms_components=comps, tile_zp_rms_largest=comps[0]["rms"],
                 tile_zp_range_largest=comps[0]["range"],
                 tile_zp_rms_pooled=float(np.std(z)), tile_zp_mad_pooled=_mad(z),
                 tile_chi2_dof=t["chi2_dof"], tile_sys_floor=t["sys_floor"],
                 tile_n_components=len(t["components"]),
                 pair_dmag_mad_before=_mad(merged_d), pair_resid_mad_after=_mad(t["resid"]),
                 worst_tiles=[dict(obs=t["nodes"][k], z=float(z[k]), err=float(t["z_err"][k]))
                              for k in np.argsort(-np.abs(z))[:5]])
    ab = sol.get("tile+AB")
    if ab:
        s.update(ab_global=ab["delta"], ab_global_err=ab["delta_err"],
                 ab_global_err_formal=ab["delta_err_formal"], ab_chi2_dof=ab["chi2_dof"],
                 ab_sys_floor=ab["sys_floor"], ab_resid_mad=_mad(ab["resid"]))
    tg, ref = sol.get("tile+G"), sol.get("reference")
    if tg:
        zg = dict(zip(tg["nodes"], np.asarray(tg["z"])))
        comps = [dict(n=len(c), obs=list(c), rms=float(np.std([zg[o] for o in c])),
                      range=float(np.ptp([zg[o] for o in c]))) for c in tg["components"]]
        s.update(grad_g=[float(v) for v in tg["g"]], grad_g_err=[float(v) for v in tg["g_err"]],
                 grad_tile_zp_rms_components=comps, grad_tile_zp_rms_largest=comps[0]["rms"],
                 grad_chi2_dof=tg["chi2_dof"], grad_sys_floor=tg["sys_floor"],
                 grad_resid_mad=_mad(tg["resid"]))
        com = [o for o in zg if o in ref["t"]]
        if len(com) > 2:
            # each component has its own gauge: compare after removing the component means
            zc, tc = [], []
            for c in tg["components"]:
                cc = [o for o in c if o in ref["t"]]
                if len(cc) > 1:
                    zc += list(np.array([zg[o] for o in cc]) - np.mean([zg[o] for o in cc]))
                    tc += list(np.array([ref["t"][o][0] for o in cc]) - np.mean([ref["t"][o][0] for o in cc]))
            if len(zc) > 2:
                s.update(grad_vs_ref_n=len(zc), grad_vs_ref_r=float(np.corrcoef(zc, tc)[0, 1]),
                         grad_vs_ref_rms_diff=float(np.std(np.array(zc) - np.array(tc))))
    if ref:
        s.update(ref_g=ref["g"], ref_g_err=ref["g_err"], ref_g_err_formal=ref["g_err_formal"],
                 ref_ks_slope=ref["ks_slope"], ref_hk_slope=ref["hk_slope"],
                 ref_n_stars=ref["n_stars"], ref_n_tiles=ref["n_tiles"], ref_sigma=ref["sigma"],
                 ref_tile_offset_rms=float(np.std([v[0] for v in ref["t"].values()])))
    pas = [mm.get("pa_v3") for mm in (m.get("meta") or {}).values() if mm and mm.get("pa_v3") is not None]
    if pas:
        # the gradient is fitted in sky (east, north) axes, which equal the detector axes only
        # while every tile shares one roll
        s.update(pa_v3_range=[float(min(pas)), float(max(pas))])
    per = sol.get("ab_per_tile") or {}
    if per:
        v = np.array([x[0] for x in per.values()])
        e = np.array([x[1] for x in per.values()])
        w = 1 / e ** 2
        s.update(ab_tile_n=len(v), ab_tile_median=float(np.median(v)), ab_tile_std=float(np.std(v)),
                 ab_tile_wmean=float(np.sum(w * v) / np.sum(w)),
                 ab_tile_median_err=float(np.median(e)),
                 ab_tile_chi2_dof_about_wmean=float(np.sum(w * (v - np.sum(w * v) / np.sum(w)) ** 2)
                                                    / max(len(v) - 1, 1)),
                 ab_worst_tiles=[dict(obs=o, ab=per[o][0], err=per[o][1]) for o in
                                 sorted(per, key=lambda o: -abs(per[o][0] - np.median(v)))[:5]])
    return s


# --------------------------------------------------------------------------------- figures
def _galactic(vertices):
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    v = np.asarray(vertices)
    g = SkyCoord(v[:, 0] * u.deg, v[:, 1] * u.deg).galactic
    lon = g.l.wrap_at(180 * u.deg).deg
    return np.c_[lon, g.b.deg]


_COMP_COLOURS = ("k", "tab:green", "tab:purple", "tab:orange", "tab:cyan", "tab:olive")


def _skypanel(ax, meta, values, label, vmax=None, cmap="RdBu_r", components=None):
    """Tile footprints coloured by ``values``.  With ``components`` (lists of obs ids, largest
    first), each component's tiles are outlined in their own colour and listed in a legend, since
    each component carries its own mean-zero gauge."""
    import matplotlib as mpl
    from matplotlib.patches import Polygon as MPoly
    vals = [v for v in values.values() if np.isfinite(v)]
    vmax = vmax or (np.nanpercentile(np.abs(vals), 98) if vals else 0.05) or 0.05
    norm = mpl.colors.Normalize(-vmax, vmax)
    cm = mpl.colormaps[cmap]
    allxy = []
    for o, mt in meta.items():
        if not mt.get("vertices"):
            continue
        xy = _galactic(mt["vertices"])
        allxy.append(xy)
        v = values.get(o, np.nan)
        fc = cm(norm(v)) if np.isfinite(v) else (0.85, 0.85, 0.85, 1)
        ec, lw = "k", 0.4
        for k, c in enumerate(components or []):
            if o in c:
                ec, lw = _COMP_COLOURS[k % len(_COMP_COLOURS)], 1.6
        ax.add_patch(MPoly(xy, closed=True, fc=fc, ec=ec, lw=lw))
        cx, cy = xy.mean(0)
        ax.text(cx, cy, o, ha="center", va="center", fontsize=5)
    if allxy:
        a = np.vstack(allxy)
        ax.set_xlim(a[:, 0].max() + 0.01, a[:, 0].min() - 0.01)   # l increases to the left
        ax.set_ylim(a[:, 1].min() - 0.01, a[:, 1].max() + 0.01)
    ax.set_aspect("equal")
    ax.set_xlabel("Galactic l (deg)")
    ax.set_ylabel("Galactic b (deg)")
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cm)
    ax.figure.colorbar(sm, ax=ax, label=label, shrink=0.8)
    if components:
        from matplotlib.lines import Line2D
        hand = [Line2D([], [], color=_COMP_COLOURS[k % len(_COMP_COLOURS)], lw=2,
                       label=f"component {k + 1}: {len(c)} tiles"
                       + (f", RMS {np.std([values[o] for o in c if o in values]):.3f}"
                          if any(o in values for o in c) else ""))
                for k, c in enumerate(components)]
        ax.legend(handles=hand, fontsize=7, loc="upper left", title="separate gauges (mean 0 each)",
                  title_fontsize=7)


def plot_skymap(m, sol, path):
    import matplotlib.pyplot as plt
    f = m["filter"]
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.6))
    t = sol.get("tile")
    z = {n: float(v) for n, v in zip(t["nodes"], t["z"])} if t else {}
    comps = t["components"] if t else None
    _skypanel(ax[0], m["meta"], z, f"{f} tile zero point z (Vega mag; + = fainter than its component's mean)",
              components=comps)
    ax[0].set_title(f"{f}: per-tile zero point (merged m8, mean 0 per connected component)\n"
                    f"largest component RMS {np.std([z[o] for o in comps[0]]):.3f} mag, "
                    f"χ²/dof {t['chi2_dof']:.1f}" if t else f"{f}: no merged edges")
    ab = {o: v[0] for o, v in (sol.get("ab_per_tile") or {}).items()}
    g = sol.get("tile+AB")
    _skypanel(ax[1], m["meta"], {o: v - (g["delta"] if g else 0) for o, v in ab.items()},
              f"{f} (z_A − z_B) − global Δ (Vega mag)", cmap="PuOr_r")
    ax[1].set_title(f"{f}: per-tile NRCA − NRCB about the global Δ = "
                    f"{g['delta']:+.4f} ± {g['delta_err']:.4f} mag" if g else f"{f}: no module edges")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_residuals(m, sol, path):
    import matplotlib.pyplot as plt
    f = m["filter"]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    me = [e for e in m["edges"] if e["kind"] == "merged"]
    mo = [e for e in m["edges"] if e["kind"] in ("module", "intra")]
    t, g, tm = sol.get("tile"), sol.get("tile+AB"), sol.get("tile×module")
    bins = np.linspace(-0.15, 0.15, 61)
    if t:
        d = np.array([e["d"] for e in me])
        ax[0].hist(np.clip(d, bins[0], bins[-1]), bins, histtype="step", lw=1.5, color="0.4",
                   label=f"before: pair Δmag (MAD {_mad(d):.3f})")
        ax[0].hist(np.clip(t["resid"], bins[0], bins[-1]), bins, histtype="stepfilled", alpha=0.5,
                   color="C0", label=f"after tile solve (MAD {_mad(t['resid']):.3f})")
        ax[0].set_title(f"{f} merged catalogues: {len(me)} tile pairs")
    if g:
        d = np.array([e["d"] for e in mo])
        ax[1].hist(np.clip(d, bins[0], bins[-1]), bins, histtype="step", lw=1.5, color="0.4",
                   label=f"before: module-pair Δmag (MAD {_mad(d):.3f})")
        ax[1].hist(np.clip(g["resid"], bins[0], bins[-1]), bins, histtype="stepfilled", alpha=0.5,
                   color="C1", label=f"after tile + global A−B (MAD {_mad(g['resid']):.3f})")
        ax[1].hist(np.clip(tm["resid"], bins[0], bins[-1]), bins, histtype="step", lw=1.2,
                   color="C3", label=f"after tile×module (MAD {_mad(tm['resid']):.3f})")
        ax[1].set_title(f"{f} per-module catalogues: {len(mo)} module pairs")
    for a in ax[:2]:
        a.set_xlabel(f"{f} Δmag, Vega (mag; clipped to ±0.15)")
        a.set_ylabel("pairs")
        a.legend(fontsize=7)
    if t:
        r = np.asarray(t["resid"]) / np.array([e["err"] for e in me])
        ax[2].hist(np.clip(r, -20, 20), np.linspace(-20, 20, 81), histtype="stepfilled", alpha=0.5,
                   color="C0", label=f"tile model, χ²/dof {t['chi2_dof']:.1f}")
    if g:
        r = np.asarray(g["resid"]) / np.array([e["err"] for e in mo])
        ax[2].hist(np.clip(r, -20, 20), np.linspace(-20, 20, 81), histtype="step", lw=1.5,
                   color="C1", label=f"tile+A−B model, χ²/dof {g['chi2_dof']:.1f}")
    ax[2].set_xlabel("normalised residual (resid / formal σ; clipped to ±20)")
    ax[2].set_ylabel("pairs")
    ax[2].set_title("residual / formal median error")
    ax[2].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_modules(m, sol, path):
    import matplotlib.pyplot as plt
    f = m["filter"]
    per = sol.get("ab_per_tile") or {}
    g = sol.get("tile+AB")
    fig, ax = plt.subplots(1, 2, figsize=(16, 4.6), gridspec_kw=dict(width_ratios=[3, 1]))
    obs = sorted(per)
    x = np.arange(len(obs))
    v = np.array([per[o][0] for o in obs])
    e = np.array([per[o][1] for o in obs])
    intra = {e_["a"].split(":")[0] for e_ in m["edges"] if e_["kind"] == "intra"}
    col = ["C0" if o in intra else "C2" for o in obs]
    ax[0].errorbar(x, v, e, fmt="none", ecolor="0.6", lw=0.8)
    ax[0].scatter(x, v, c=col, s=14, zorder=3)
    if g:
        ax[0].axhspan(g["delta"] - g["delta_err"], g["delta"] + g["delta_err"], color="C1", alpha=0.3)
        ax[0].axhline(g["delta"], color="C1", lw=1, label=f"global Δ = {g['delta']:+.4f} ± {g['delta_err']:.4f}")
    ax[0].axhline(0, color="k", lw=0.5)
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(obs, rotation=90, fontsize=6)
    ax[0].set_xlabel("observation (blue: tile has an intra-tile NRCA–NRCB overlap edge; green: cross-tile only)")
    ax[0].set_ylabel(f"{f} z_NRCA − z_NRCB (Vega mag)")
    ax[0].set_title(f"{f}: per-tile module offset from the tile×module solve (±1σ incl. error floor)")
    ax[0].legend(fontsize=8, loc="upper right")
    if len(v):
        lo, hi = np.nanpercentile(v, [1, 99])
        pad = max(0.02, 0.2 * (hi - lo))
        ax[0].set_ylim(lo - pad, hi + pad)
        ax[1].hist(v, np.linspace(lo - pad, hi + pad, 30), orientation="horizontal", color="0.5")
        ax[1].set_ylim(lo - pad, hi + pad)
    ax[1].set_xlabel("tiles")
    ax[1].set_ylabel(f"{f} z_NRCA − z_NRCB (Vega mag)")
    ax[1].set_title(f"median {np.median(v):+.4f}, std {np.std(v):.4f}" if len(v) else "")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_time_roll(m, sol, path):
    import matplotlib.pyplot as plt
    f = m["filter"]
    t = sol.get("tile")
    per = sol.get("ab_per_tile") or {}
    fig, ax = plt.subplots(2, 2, figsize=(12, 7.5), sharex="col")
    if t:
        z = dict(zip(t["nodes"], t["z"]))
        ze = dict(zip(t["nodes"], t["z_err"]))
        for k, (key, lab) in enumerate((("mjd", "EXPSTART (MJD)"), ("pa_aper", "PA_APER (deg)"))):
            obs = [o for o in z if m["meta"][o].get(key) is not None]
            xx = np.array([m["meta"][o][key] for o in obs], float)
            ax[0][k].errorbar(xx, [z[o] for o in obs], [ze[o] for o in obs], fmt="o", ms=3, lw=0.6)
            for o, xv in zip(obs, xx):
                ax[0][k].annotate(o, (xv, z[o]), fontsize=5, xytext=(2, 2), textcoords="offset points")
            ax[0][k].axhline(0, color="k", lw=0.5)
            ax[0][k].set_ylabel(f"{f} tile z (Vega mag)")
            obs = [o for o in per if m["meta"][o].get(key) is not None]
            xx = np.array([m["meta"][o][key] for o in obs], float)
            ax[1][k].errorbar(xx, [per[o][0] for o in obs], [per[o][1] for o in obs], fmt="o",
                              ms=3, lw=0.6, color="C2")
            ax[1][k].set_ylabel(f"{f} z_NRCA − z_NRCB (Vega mag)")
            ax[1][k].set_xlabel(lab)
    ax[0][0].set_title(f"{f}: tile zero point vs observation date")
    ax[0][1].set_title(f"{f}: tile zero point vs aperture position angle")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_gradient(m, sol, rm, path):
    """Gradient model: (1) the tile-only z against the ramp one within-tile gradient predicts,
    (2) tile+G z against the tile offsets measured against VIRAC2 Ks, (3) the stacked
    F<filt> − Ks profile along east and north."""
    import matplotlib.pyplot as plt
    filt, tg, ref = m["filter"], sol["tile+G"], sol["reference"]
    g = np.asarray(tg["g"])
    cen = {o: tile_centre(mm) for o, mm in m["meta"].items()}
    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    t = sol.get("tile")
    if t:
        for k, comp in enumerate(t["components"]):
            cc = [o for o in comp if cen.get(o)]
            if len(cc) < 2:
                continue
            zt = dict(zip(t["nodes"], t["z"]))
            c0 = np.mean([cen[o] for o in cc], 0)
            x, y = _offset_arcmin([cen[o][0] for o in cc], [cen[o][1] for o in cc], c0)
            pred = -(g[0] * x + g[1] * y)
            ax[0].scatter(pred - pred.mean(), [zt[o] for o in cc], s=12, label=f"component {k + 1} ({len(cc)})")
        lim = ax[0].get_xlim()
        ax[0].plot(lim, lim, "k:", lw=0.8)
        ax[0].set_xlabel("ramp predicted by the within-tile gradient, −G·c (mag)")
        ax[0].set_ylabel(f"tile-only network z ({filt}, Vega mag)")
        ax[0].set_title(f"G = ({g[0]:+.4f}, {g[1]:+.4f}) mag/arcmin (east, north)")
        ax[0].legend(fontsize=7)
    zg = dict(zip(tg["nodes"], tg["z"]))
    for comp in tg["components"]:
        cc = [o for o in comp if o in ref["t"]]
        if len(cc) < 2:
            continue
        zz = np.array([zg[o] for o in cc]); tt = np.array([ref["t"][o][0] for o in cc])
        ax[1].errorbar(tt - tt.mean(), zz - zz.mean(), xerr=[ref["t"][o][1] for o in cc], fmt="o", ms=3, lw=0.5)
    lim = ax[1].get_xlim()
    ax[1].plot(lim, lim, "k:", lw=0.8)
    ax[1].set_xlabel("tile offset vs VIRAC2 Ks, gradient fitted (Vega mag)")
    ax[1].set_ylabel(f"tile+G network z ({filt}, Vega mag)")
    ax[1].set_title("network zero points after the gradient vs the external reference")
    x = np.concatenate([r["x"] for r in rm["tiles"]]); y = np.concatenate([r["y"] for r in rm["tiles"]])
    d = np.concatenate([r["d"] for r in rm["tiles"]])
    tid = np.concatenate([[r["obs"]] * len(r["d"]) for r in rm["tiles"]])
    d = d - np.array([ref["t"].get(o, (np.nan,))[0] for o in tid]) - ref["ks_slope"] * (
        np.concatenate([r["ks"] for r in rm["tiles"]]) - 15.5) - ref["hk_slope"] * (
        np.concatenate([r["hk"] for r in rm["tiles"]]) - 1.5)
    d -= np.nanmedian(d)
    bins = np.linspace(-1.5, 1.5, 16)
    for v, lab in ((x, "east"), (y, "north")):
        k = np.digitize(v, bins)
        med = [np.nanmedian(d[k == i]) if np.sum(k == i) > 20 else np.nan for i in range(1, len(bins))]
        ax[2].plot(0.5 * (bins[1:] + bins[:-1]), med, "o-", ms=3, label=lab)
    ax[2].plot(bins, ref["g"][0] * bins, "C0:", lw=0.8)
    ax[2].plot(bins, ref["g"][1] * bins, "C1:", lw=0.8)
    ax[2].axhline(0, color="k", lw=0.5)
    ax[2].set_xlabel("offset from tile centre (arcmin)")
    ax[2].set_ylabel(f"{filt} − Ks, tile offset removed (Vega mag)")
    ax[2].set_title(f"stacked profile vs VIRAC2 Ks ({ref['n_stars']} stars, {ref['n_tiles']} tiles)")
    ax[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_summary(results, path):
    """Across filters: per-tile z and per-tile A−B of one filter against the other."""
    import matplotlib.pyplot as plt
    fs = [r for r in results if r["sol"].get("tile")]
    if len(fs) < 2:
        return None
    a, b = fs[0], fs[1]
    fa, fb = a["m"]["filter"], b["m"]["filter"]
    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    za = dict(zip(a["sol"]["tile"]["nodes"], a["sol"]["tile"]["z"]))
    zb = dict(zip(b["sol"]["tile"]["nodes"], b["sol"]["tile"]["z"]))
    com = sorted(set(za) & set(zb))
    xa, xb = np.array([za[o] for o in com]), np.array([zb[o] for o in com])
    ax[0].scatter(xa, xb, s=12)
    for o, p, q in zip(com, xa, xb):
        ax[0].annotate(o, (p, q), fontsize=5, xytext=(2, 2), textcoords="offset points")
    lim = max(np.abs(np.r_[xa, xb]).max(), 0.01) * 1.1
    ax[0].plot([-lim, lim], [-lim, lim], "k:", lw=0.8)
    ax[0].set_xlim(-lim, lim); ax[0].set_ylim(-lim, lim)
    ax[0].set_xlabel(f"{fa} tile z (Vega mag)"); ax[0].set_ylabel(f"{fb} tile z (Vega mag)")
    r = np.corrcoef(xa, xb)[0, 1] if len(com) > 2 else np.nan
    ax[0].set_title(f"tile zero points, {fa} vs {fb} (Pearson r = {r:.2f}, N = {len(com)})")
    pa, pb = a["sol"].get("ab_per_tile") or {}, b["sol"].get("ab_per_tile") or {}
    com = sorted(set(pa) & set(pb))
    if com:
        ax[1].errorbar([pa[o][0] for o in com], [pb[o][0] for o in com],
                       xerr=[pa[o][1] for o in com], yerr=[pb[o][1] for o in com],
                       fmt="o", ms=3, lw=0.5, color="C2")
        r = np.corrcoef([pa[o][0] for o in com], [pb[o][0] for o in com])[0, 1] if len(com) > 2 else np.nan
        ax[1].set_title(f"per-tile NRCA − NRCB, {fa} vs {fb} (r = {r:.2f}, N = {len(com)})")
    ax[1].axhline(0, color="k", lw=0.5); ax[1].axvline(0, color="k", lw=0.5)
    ax[1].set_xlabel(f"{fa} z_NRCA − z_NRCB (Vega mag)")
    ax[1].set_ylabel(f"{fb} z_NRCA − z_NRCB (Vega mag)")
    labs, rms, ab, abe = [], [], [], []
    for res in results:
        s = res["summary"]
        labs.append(s["filter"])
        rms.append(s.get("tile_zp_rms_largest", np.nan))
        ab.append(s.get("ab_global", np.nan))
        abe.append(s.get("ab_global_err", np.nan))
    xx = np.arange(len(labs))
    ax[2].bar(xx - 0.2, rms, 0.4, label="tile zero-point RMS (merged, largest component)", color="C0")
    ax[2].bar(xx + 0.2, ab, 0.4, yerr=abe, label="global NRCA − NRCB Δ", color="C1", capsize=4)
    ax[2].axhline(0, color="k", lw=0.5)
    ax[2].set_xticks(xx); ax[2].set_xticklabels(labs)
    ax[2].set_ylabel("Vega mag")
    ax[2].set_title("per-filter summary")
    ax[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# ------------------------------------------------------------------------------------ main
def cache_key(catdir, filt, obs_list):
    """What the edges depend on: the catalogue directory, the filter, the tile set and every input
    catalogue's (path, size, mtime), plus each tile's footprint mosaic.  A regenerated m8 or a
    changed ``--obs``/``--catdir`` changes the key, so the cached edges are re-measured."""
    files = {}
    for o in obs_list:
        for mod in ("merged",) + MODULES:
            p = _cat(catdir, mod, o)
            st = os.stat(p) if os.path.exists(p) else None
            files[p] = [st.st_size, st.st_mtime] if st else None
        fp = D._footprint_path(_obs(o), filt)
        if fp:
            # a mosaic can be renamed (quarantined) between the glob and here
            st = os.stat(fp) if os.path.exists(fp) else None
            files[fp] = [st.st_size, st.st_mtime] if st else None
    return dict(catdir=os.path.abspath(catdir), filter=filt, tiles=list(obs_list), files=files,
                edge_version=EDGE_VERSION)


def _cached(path, key, build, label, use_cache=True):
    """JSON cache keyed on ``key``: reuse ``path`` while its key matches, else rebuild."""
    if use_cache and os.path.exists(path):
        with open(path) as fh:
            m = json.load(fh)
        if m.get("cache_key") == key:
            print(f"[s13zp] {label}: cached from {path}", flush=True)
            return m
        print(f"[s13zp] {label}: cache {path} is stale (inputs or arguments changed); re-measuring",
              flush=True)
    m = build()
    m["cache_key"] = key
    with open(path, "w") as fh:
        json.dump(m, fh, indent=1)
    return m


def run_filter(catdir, filt, outdir, nproc=1, cache=True, obs_list=None, reference=True):
    fl = filt.lower()
    obs_list = sorted(obs_list) if obs_list else tiles(catdir)
    key = cache_key(catdir, filt, obs_list)
    m = _cached(os.path.join(outdir, f"s13zp_{fl}_edges.json"), key,
                lambda: measure_filter(catdir, filt, nproc=nproc, obs_list=obs_list), f"{filt} edges",
                cache)
    ref = None
    if reference:
        rm = _cached(os.path.join(outdir, f"s13zp_{fl}_reference.json"),
                     dict(key, ks_range=list(REF_KS_RANGE), match_as=REF_MATCH_AS, ref_version=2),
                     lambda: measure_reference(catdir, filt, nproc=nproc, obs_list=obs_list),
                     f"{filt} VIRAC2 Ks reference", cache)
        ref = fit_reference(rm)
    sol = solve_filter(m, ref)
    summ = summarise(m, sol)
    with open(os.path.join(outdir, f"s13zp_{fl}.json"), "w") as fh:
        json.dump(dict(summary=summ, solutions=_jsonable(sol), meta=m["meta"],
                       gauge="mean zero over the tiles (nodes) of each connected component"),
                  fh, indent=1)
    plot_skymap(m, sol, os.path.join(outdir, f"s13zp_{fl}_skymap.png"))
    plot_residuals(m, sol, os.path.join(outdir, f"s13zp_{fl}_residuals.png"))
    plot_modules(m, sol, os.path.join(outdir, f"s13zp_{fl}_modules.png"))
    plot_time_roll(m, sol, os.path.join(outdir, f"s13zp_{fl}_vs_time_roll.png"))
    if sol.get("tile+G"):
        plot_gradient(m, sol, rm, os.path.join(outdir, f"s13zp_{fl}_gradient.png"))
    print(json.dumps(summ, indent=1, default=float), flush=True)
    return dict(m=m, sol=sol, summary=summ)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--catdir", default=DEFAULT_CATDIR)
    ap.add_argument("--filters", nargs="+", default=["F212N", "F480M"])
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--obs", nargs="+", default=None, help="restrict to these obs ids (testing)")
    ap.add_argument("--nproc", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    ap.add_argument("--no-cache", action="store_true", help="re-measure every pair")
    ap.add_argument("--no-reference", action="store_true",
                    help="skip the VIRAC2 Ks anchor and the tile+G gradient model")
    a = ap.parse_args(argv)
    os.makedirs(a.outdir, exist_ok=True)
    results = [run_filter(a.catdir, f, a.outdir, a.nproc, cache=not a.no_cache, obs_list=a.obs,
                          reference=not a.no_reference)
               for f in a.filters]
    plot_summary(results, os.path.join(a.outdir, "s13zp_summary.png"))


if __name__ == "__main__":
    main()
