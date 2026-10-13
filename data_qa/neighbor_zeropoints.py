"""Photometric zero points of every tile (and every NIRCam module) of a mosaic program, solved as a
network from the stage-13 neighbour overlaps.

Stage 13 measures, for each pair of overlapping tiles, the median F212N magnitude difference of the
stars they share.  Each pairwise number mixes two tiles' zero points; this tool collects every pair
and solves for one zero point per tile so the field-to-field calibration can be read off directly.
It reuses stage 13's footprints, sample cuts and matching (``diagnostics._overlapping_neighbors``,
``_load_nb_sample``, ``_neighbor_agreement``), and runs them for every filter asked for (default
F212N and F480M) on three catalogue flavours: the merged m8 catalogue and the two per-module (NRCA,
NRCB) m8 catalogues.

Three weighted least-squares models are solved per filter.  Every edge is one pair measurement
``d = median(m_a - m_b)`` (Vega) with weight ``1/sigma^2``, ``sigma = 1.2533 * 1.4826 MAD / sqrt(N)``
(the standard error of a median):

* ``tile``         -- merged catalogues; ``d = z_i - z_j``; one zero point per tile;
* ``tile+AB``      -- per-module catalogues; ``d = z_i + s_a Delta/2 - z_j - s_b Delta/2`` with
  ``s = +1`` for NRCA and -1 for NRCB, so ``Delta`` is one global NRCA - NRCB offset per filter;
* ``tile×module``  -- per-module catalogues; one free zero point per (tile, module); the per-tile
  ``z_A - z_B`` and its covariance-propagated error come from this model.

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
``s13zp_<filt>_vs_time_roll.png``; across filters ``s13zp_summary.png``.

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


def solve_network(edges, nodes=None, node_of=None, sign_of=None):
    """Weighted least squares for relative zero points on an overlap graph.

    ``edges``: dicts with ``a``, ``b`` (node keys), ``d`` (measured ``m_a - m_b``) and ``err``.
    ``node_of(key)`` maps an edge endpoint to the zero-point node it constrains (default: itself);
    ``sign_of(key)`` returns +1/-1/0 for a global term ``Delta`` that enters as ``+s*Delta/2``
    (``None`` = no global term).  Gauge: mean zero over the nodes of each connected component.

    Returns dict(nodes, z, z_err, cov, delta, delta_err, resid, chi2, dof, chi2_dof, components,
    sys_floor) where ``resid`` is ``d - model`` per edge and ``sys_floor`` is the error added in
    quadrature to every edge that brings chi^2/dof to 1 (0 when it is already <= 1)."""
    node_of = node_of or (lambda k: k)
    if nodes is None:
        nodes = sorted({node_of(e[s]) for e in edges for s in ("a", "b")})
    nodes = list(nodes)
    idx = {n: k for k, n in enumerate(nodes)}
    nn = len(nodes)
    glob_term = sign_of is not None
    npar = nn + (1 if glob_term else 0)
    A = np.zeros((len(edges), npar))
    d = np.array([e["d"] for e in edges], float)
    err = np.array([e["err"] for e in edges], float)
    for r, e in enumerate(edges):
        A[r, idx[node_of(e["a"])]] += 1.0
        A[r, idx[node_of(e["b"])]] -= 1.0
        if glob_term:
            A[r, nn] = 0.5 * (sign_of(e["a"]) - sign_of(e["b"]))
    comps = _components(nodes, [dict(a=node_of(e["a"]), b=node_of(e["b"])) for e in edges])
    C = np.zeros((len(comps), npar))
    for k, c in enumerate(comps):
        C[k, c] = 1.0 / len(c)

    def _solve(err_eff):
        w = 1.0 / err_eff ** 2
        N = A.T @ (A * w[:, None])
        K = np.block([[N, C.T], [C, np.zeros((len(comps), len(comps)))]])
        rhs = np.r_[A.T @ (w * d), np.zeros(len(comps))]
        Kinv = np.linalg.pinv(K)
        p = (Kinv @ rhs)[:npar]
        cov = Kinv[:npar, :npar]
        res = d - A @ p
        return p, cov, res, float(np.sum(w * res ** 2))

    p, cov, res, chi2 = _solve(err)
    dof = len(edges) - npar + len(comps)
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


def _edge(res, a, b, kind):
    sig = _MED_SE * res["dmag_scatter"] / np.sqrt(res["n_pairs"])
    return dict(a=a, b=b, kind=kind, d=res["dmag_bulk"], err=float(max(sig, 1e-4)),
                n_pairs=res["n_pairs"], dmag_scatter=res["dmag_scatter"],
                offset_mas=res["offset_mas"])


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
            out.append(_edge(r, f"{i}:nrca", f"{i}:nrcb", "intra"))
        return out
    s = {}
    for mod in ("merged",) + MODULES:
        for t in (i, j):
            s[(t, mod)] = D._load_nb_sample(_cat(catdir, mod, t), filt, poly, frame)
    r = D._neighbor_agreement(s[(i, "merged")], s[(j, "merged")])
    if r:
        out.append(_edge(r, i, j, "merged"))
    for ma in MODULES:
        for mb in MODULES:
            r = D._neighbor_agreement(s[(i, ma)], s[(j, mb)])
            if r:
                out.append(_edge(r, f"{i}:{ma}", f"{j}:{mb}", "module"))
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


# ------------------------------------------------------------------------------- solutions
def solve_filter(m):
    """The three models of the module docstring for one filter's measurements."""
    edges = m["edges"]
    merged = [e for e in edges if e["kind"] == "merged"]
    modular = [e for e in edges if e["kind"] in ("module", "intra")]
    out = dict(filter=m["filter"])
    if merged:
        out["tile"] = solve_network(merged)
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
    return dict(catdir=os.path.abspath(catdir), filter=filt, tiles=list(obs_list), files=files)


def run_filter(catdir, filt, outdir, nproc=1, cache=True, obs_list=None):
    ep = os.path.join(outdir, f"s13zp_{filt.lower()}_edges.json")
    obs_list = sorted(obs_list) if obs_list else tiles(catdir)
    key = cache_key(catdir, filt, obs_list)
    m = None
    if cache and os.path.exists(ep):
        with open(ep) as fh:
            m = json.load(fh)
        if m.get("cache_key") == key:
            print(f"[s13zp] {filt}: {len(m['edges'])} cached edges from {ep}", flush=True)
        else:
            print(f"[s13zp] {filt}: cached edges in {ep} are stale (inputs or arguments changed); "
                  "re-measuring", flush=True)
            m = None
    if m is None:
        m = measure_filter(catdir, filt, nproc=nproc, obs_list=obs_list)
        m["cache_key"] = key
        with open(ep, "w") as fh:
            json.dump(m, fh, indent=1)
    sol = solve_filter(m)
    summ = summarise(m, sol)
    fl = filt.lower()
    with open(os.path.join(outdir, f"s13zp_{fl}.json"), "w") as fh:
        json.dump(dict(summary=summ, solutions=_jsonable(sol), meta=m["meta"],
                       gauge="mean zero over the tiles (nodes) of each connected component"),
                  fh, indent=1)
    plot_skymap(m, sol, os.path.join(outdir, f"s13zp_{fl}_skymap.png"))
    plot_residuals(m, sol, os.path.join(outdir, f"s13zp_{fl}_residuals.png"))
    plot_modules(m, sol, os.path.join(outdir, f"s13zp_{fl}_modules.png"))
    plot_time_roll(m, sol, os.path.join(outdir, f"s13zp_{fl}_vs_time_roll.png"))
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
    a = ap.parse_args(argv)
    os.makedirs(a.outdir, exist_ok=True)
    results = [run_filter(a.catdir, f, a.outdir, a.nproc, cache=not a.no_cache, obs_list=a.obs)
               for f in a.filters]
    plot_summary(results, os.path.join(a.outdir, "s13zp_summary.png"))


if __name__ == "__main__":
    main()
