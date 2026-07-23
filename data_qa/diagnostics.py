"""Progressive QA diagnostic figures, posted as replies (comments) to a per-observation
tracking issue.

Four stages, each emitted as the corresponding data product becomes available while the
cataloging pipeline runs.  Each stage returns ``(png_path, metrics)``; the metrics drive
the checkbox state in the issue body (see ``make_issues.render_body``), and the PNG is
posted as an idempotent comment (one comment per stage, keyed on a hidden marker).

    Stage 1  first i2d       one SW + one LW grayscale mosaic       "delivered", "mosaics present"
    Stage 2  CMD             LW vs SW-LW colour-magnitude + LF      "catalog vetted", "depth"
    Stage 3  calibration     JWST (F212N-like) vs VIRAC Ks         "photometry zeropoints"
    Stage 4  offsets         JWST-VIRAC dRA/dDec + inter-module    "absolute frame", "inter-module"

Images LIVE IN THE ISSUE (posted to the GitHub CDN as release assets on a single
``qa-assets`` bucket release, then embedded in the comment) -- NOT committed to the repo
source tree.  Reuses the reference-free / crowding-proof machinery in
``astrometry_audit`` (detect / xcorr / direct_intermodule / load_reference).

Usage:
    python -m data_qa.diagnostics --program 5365 --obs 001 --stage 1 2 3 4          # build only
    python -m data_qa.diagnostics --program 5365 --obs 001 --stage 1 --post         # build + post
    python -m data_qa.diagnostics --program 2221 --obs 001 --sw F212N --lw F410M --post
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from dataclasses import replace

import numpy as np

from . import astrometry_audit as aa
from .observations import Observation, registry

BASE = os.environ.get("QA_BASE", "/orange/adamginsburg/jwst")
OUTDIR = os.environ.get("QA_OUTDIR", "/tmp/data_qa_figures")

# Comment idempotency markers (one comment per stage per issue).
DIAG_MARKER = "<!-- data-qa:diag:stage{n} -->"

# Filter selection: prefer the requested filters; else nearest SW / LW.
_SW_PREF = ["F212N", "F200W", "F187N", "F182M", "F162M", "F150W", "F115W"]
_LW_PREF = ["F480M", "F470N", "F466N", "F444W", "F410M", "F405N", "F360M", "F356W", "F323N", "F300M", "F277W"]


def _channel(filt):
    return "SW" if int(filt[1:4]) <= 212 else "LW"


def pick_filters(available, sw=None, lw=None):
    """Choose one SW + one LW filter from those available for the obs."""
    up = {f.upper() for f in available}
    sw = sw.upper() if sw else next((f for f in _SW_PREF if f in up), None)
    lw = lw.upper() if lw else next((f for f in _LW_PREF if f in up), None)
    return sw, lw


# --------------------------------------------------------------------------- product lookup
def _mosaic_path(o: Observation, filt):
    """Released merged i2d for this obs+filter, or None."""
    if not filt:                     # single-filter obs (e.g. gc2211 o028 = F150W only) has no LW
        return None
    pats = [
        f"{BASE}/{o.field}/{filt}/pipeline/{o.obsid}_t001_nircam_clear-{filt.lower()}-merged_i2d.fits",
        f"{BASE}/{o.field}/*/pipeline/{o.obsid}_t001_nircam_clear-{filt.lower()}-merged_i2d.fits",
    ]
    for pat in pats:
        hits = sorted(glob.glob(pat))
        if hits:
            return hits[-1]
    return None


_MLEVEL_RE = re.compile(r"_m([1-8])(?:_|\b)", re.I)


def _catalog_priority(basename):
    """Rising-priority rank of a catalog by pipeline stage (higher = preferred):
    MAST-shipped defaults (lowest) < m1 < m2 < ... < m8.  Returns (tier, kind_label)."""
    low = basename.lower()
    if "m8_dedup" in low:
        return 8, "m8_dedup"
    m = _MLEVEL_RE.search(low)
    if m:
        return int(m.group(1)), f"m{m.group(1)}"
    # raw MAST products (pipeline source catalogs) sit below every m-stage merge
    if low.endswith("_cat.fits") or "source_catalog" in low or "_segm" in low:
        return 0, "mast"
    return 0, "merged"       # un-tagged field merge: lowest tier, size breaks the tie


def _catalog_candidates(o: Observation):
    """All catalogs for the field, each tagged with its (priority-tier, kind).  Glob EVERY
    catalog (naming is inconsistent across fields); the caller filters by column presence
    and picks the highest tier, largest.  Skip residual/model/region sidecars."""
    out = []
    for p in sorted(glob.glob(f"{BASE}/{o.field}/catalogs/*.fits")):
        low = os.path.basename(p).lower()
        if any(s in low for s in ("_residual", "_model", "_reproject", "region")):
            continue
        tier, kind = _catalog_priority(low)
        out.append((p, kind, tier))
    return out


def _catalog_for(o: Observation, sw, lw):
    """Catalog that contains VEGA mags for both requested filters, chosen by RISING pipeline
    priority (MAST default < m1 < ... < m8), size breaking ties within a tier.  A cheap
    FITS-header probe (TTYPE/NAXIS2) avoids reading catalog data.
    Returns (path, kind, sw_col, lw_col) or (None,...)."""
    from astropy.io import fits
    best = (None, None, None, None, (-1, -1))
    for p, kind, tier in _catalog_candidates(o):
        try:
            hdr = fits.getheader(p, ext=1)             # header only -- cheap, no data read
        except (OSError, IndexError):
            continue
        ncol = hdr.get("TFIELDS", 0)
        low = {str(hdr.get(f"TTYPE{i}", "")).lower(): hdr[f"TTYPE{i}"]
               for i in range(1, ncol + 1) if hdr.get(f"TTYPE{i}")}
        csw = next((low[k] for k in (f"mag_vega_{sw.lower()}", f"mag_{sw.lower()}",
                                     f"mag_ab_{sw.lower()}") if k in low), None)
        clw = None if lw is None else next(
            (low[k] for k in (f"mag_vega_{lw.lower()}", f"mag_{lw.lower()}",
                              f"mag_ab_{lw.lower()}") if k in low), None)
        # single-filter obs (lw is None) needs only the SW column
        if not csw or (lw is not None and not clw):
            continue
        rank = (tier, hdr.get("NAXIS2", 0))
        if rank > best[-1]:
            best = (p, kind, csw, clw, rank)
    return best[:4]


def _refcat_path(o: Observation):
    """VIRAC2-Gaia refcat (newest epoch) for the absolute-frame (position-only) check."""
    hits = sorted(glob.glob(f"{BASE}/{o.field}/catalogs/gaia_virac2_refcat_epoch*.fits"))
    return hits[-1] if hits else None


def _viraccache_path(o: Observation):
    """Raw VIRAC2 cache (has a real Ksmag column) for the photometric-calibration check.
    The gaia_virac2 refcat carries only a blended 'refmag', unusable for a Ks zeropoint."""
    p = f"{BASE}/{o.field}/astrometry_diag/refcache/virac2.fits"
    return p if os.path.exists(p) else None


# --------------------------------------------------------------------------- figure helpers
def _fig(nrows=1, ncols=1, w=5.0, h=5.0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt.subplots(nrows, ncols, figsize=(w * ncols, h * nrows), squeeze=False)


def _grayscale(ax, path, title):
    from astropy.io import fits
    from astropy.visualization import ZScaleInterval, ImageNormalize, AsinhStretch
    from astropy.wcs import WCS
    with fits.open(path) as hdul:
        sci = hdul["SCI"] if "SCI" in hdul else hdul[1]
        data = sci.data.astype("float32")
    norm = ImageNormalize(data, interval=ZScaleInterval(), stretch=AsinhStretch())
    ax.imshow(data, origin="lower", cmap="gray", norm=norm)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    frac = float(np.isfinite(data).mean())
    return frac


def _save(fig, name):
    os.makedirs(OUTDIR, exist_ok=True)
    out = os.path.join(OUTDIR, name)
    fig.savefig(out, dpi=110, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- STAGE 1
def stage1_mosaics(o: Observation, sw, lw):
    """Grayscale SW + LW mosaics -- confirms the data arrived and looks sane."""
    psw, plw = _mosaic_path(o, sw), _mosaic_path(o, lw)
    fig, ax = _fig(1, 2, 5.2, 5.2)
    fracs = {}
    for a, filt, p in ((ax[0][0], sw, psw), (ax[0][1], lw, plw)):
        if p:
            fracs[filt] = _grayscale(a, p, f"{o.obsid}  {filt}")
        else:
            a.text(0.5, 0.5, f"{filt}\n(no i2d)", ha="center", va="center")
            a.set_xticks([]); a.set_yticks([])
    fig.suptitle(f"{o.target} {o.obsid} — first mosaics ({sw} / {lw})", fontsize=11)
    png = _save(fig, f"{o.obsid}_stage1.png")
    metrics = dict(stage=1, sw=sw, lw=lw,
                   sw_present=bool(psw), lw_present=bool(plw),
                   finite_fraction=fracs,
                   passed=bool(psw and (plw or lw is None)))   # single-filter obs: LW legitimately absent
    return png, metrics


# --------------------------------------------------------------------------- STAGE 2
def _mag_cols(t, sw, lw):
    """Locate SW/LW magnitude columns in a merged catalog (several naming schemes).
    ALWAYS prefer VEGA magnitudes over AB (survey convention)."""
    cols = {c.lower(): c for c in t.colnames}
    def find(filt):
        f = filt.lower()
        for pat in (f"mag_vega_{f}", f"mag_{f}", f"{f}_mag", f"mag_ab_{f}", f"{f}"):
            if pat in cols:
                return cols[pat]
        return None
    return find(sw), find(lw)


def stage2_cmd(o: Observation, sw, lw):
    """Colour-magnitude diagram (LW vs SW-LW) + luminosity-function inset."""
    from astropy.table import Table
    cat, kind, csw, clw = _catalog_for(o, sw, lw)
    fig, ax = _fig(1, 1, 5.5, 6.0)
    metrics = dict(stage=2, catalog=os.path.basename(cat) if cat else None, kind=kind)
    want = f"{sw}+{lw}" if lw else f"{sw}"
    if not cat:
        ax[0][0].text(0.5, 0.5, f"no catalog with {want} mags yet", ha="center", va="center")
        metrics["passed"] = False
        return _save(fig, f"{o.obsid}_stage2.png"), metrics
    t = Table.read(cat)
    a = ax[0][0]
    if lw is None and csw:
        # single-filter obs: no colour -> luminosity function only (still tracks depth)
        m = np.asarray(t[csw], float); g = np.isfinite(m)
        hh, edges = np.histogram(m[g], bins=60)
        ctr = 0.5 * (edges[1:] + edges[:-1])
        a.step(ctr, hh, where="mid", color="k", lw=1.0)
        peak = ctr[int(np.argmax(hh))]
        a.axvline(peak, color="r", lw=0.8, label=f"turnover≈{peak:.1f}")
        a.set_xlabel(sw); a.set_ylabel("N stars"); a.legend(fontsize=8)
        a.set_title(f"{sw} luminosity function (single filter — no colour)", fontsize=9)
        metrics.update(n_stars=int(g.sum()), lf_turnover=float(peak),
                       sw_col=csw, lw_col=None, single_filter=True,
                       passed=int(g.sum()) > 500)
        fig.suptitle(f"{o.target} {o.obsid} — LF ({kind})", fontsize=11)
        return _save(fig, f"{o.obsid}_stage2.png"), metrics
    if csw and clw:
        msw = np.asarray(t[csw], float); mlw = np.asarray(t[clw], float)
        g = np.isfinite(msw) & np.isfinite(mlw)
        color = msw[g] - mlw[g]
        hb = a.hexbin(color, mlw[g], gridsize=120, bins="log", cmap="viridis", mincnt=1)
        fig.colorbar(hb, ax=a, label="log N stars", shrink=0.85)
        a.set_xlabel(f"{sw} - {lw}"); a.set_ylabel(lw)
        a.invert_yaxis()
        a.set_xlim(np.nanpercentile(color, [1, 99]))
        # LF inset (depth): where do counts turn over
        ins = a.inset_axes([0.62, 0.62, 0.36, 0.36])
        hh, edges = np.histogram(mlw[g], bins=40)
        ctr = 0.5 * (edges[1:] + edges[:-1])
        ins.step(ctr, hh, where="mid", color="k", lw=0.8)
        peak = ctr[int(np.argmax(hh))]
        ins.axvline(peak, color="r", lw=0.8)
        ins.set_title(f"LF {lw}\nturnover≈{peak:.1f}", fontsize=7)
        ins.tick_params(labelsize=6)
        metrics.update(n_stars=int(g.sum()), lf_turnover=float(peak),
                       sw_col=csw, lw_col=clw, passed=int(g.sum()) > 500)
    else:
        a.text(0.5, 0.5, f"no {sw}/{lw} mag cols\nin {os.path.basename(cat)}",
               ha="center", va="center", fontsize=8)
        metrics["passed"] = False
    fig.suptitle(f"{o.target} {o.obsid} — CMD ({kind})", fontsize=11)
    return _save(fig, f"{o.obsid}_stage2.png"), metrics


# --------------------------------------------------------------------------- STAGE 3
def stage3_calibration(o: Observation, sw):
    """JWST (SW ~ F212N) instrumental mag vs VIRAC Ks for matched stars: a tight linear
    locus proves the RIGHT stars were matched and the photometric zeropoint is sane."""
    import astropy.units as u
    from astropy.coordinates import search_around_sky
    fig, ax = _fig(1, 1, 5.5, 5.5)
    metrics = dict(stage=3, sw=sw)
    path = _mosaic_path(o, sw)
    ref = _viraccache_path(o) or _refcat_path(o)   # cache has real Ksmag
    ep = aa.epoch_of(path) if path else None
    ref_sc, ref_mag = aa.load_reference(ref, ep) if (ref and ep) else (None, None)
    jsc, jmag = aa.detect(path) if path else (None, None)
    a = ax[0][0]
    if jsc is None or ref_sc is None or ref_mag is None:
        a.text(0.5, 0.5, "need mosaic + VIRAC refcat", ha="center", va="center")
        metrics["passed"] = False
        return _save(fig, f"{o.obsid}_stage3.png"), metrics
    ia, ib, sep, _ = search_around_sky(jsc, ref_sc, 0.2 * u.arcsec)
    if len(ia) < 30:
        a.text(0.5, 0.5, f"only {len(ia)} matches", ha="center", va="center")
        metrics["passed"] = False
        return _save(fig, f"{o.obsid}_stage3.png"), metrics
    x = ref_mag[ib]; y = jmag[ia]
    g = np.isfinite(x) & np.isfinite(y)
    x, y = x[g], y[g]
    # robust linear fit y = slope*x + zp
    slope, zp = np.polyfit(x, y, 1)
    resid = y - (slope * x + zp)
    scat = float(aa.mad_std(resid))
    hb = a.hexbin(x, y, gridsize=80, bins="log", cmap="magma", mincnt=1)
    fig.colorbar(hb, ax=a, label="log N stars", shrink=0.85)
    xs = np.array([np.nanmin(x), np.nanmax(x)])
    a.plot(xs, slope * xs + zp, "c-", lw=1, label=f"slope={slope:.2f} zp={zp:.2f}")
    a.set_xlabel("VIRAC Ks [mag]"); a.set_ylabel(f"JWST {sw} instr mag")
    a.legend(fontsize=8, loc="upper left")
    a.set_title(f"{o.obsid} calibration  n={g.sum()} scatter={scat:.2f}", fontsize=10)
    metrics.update(n_matched=int(g.sum()), slope=float(slope), zeropoint=float(zp),
                   scatter=scat, passed=(0.7 < slope < 1.3 and scat < 0.5))
    return _save(fig, f"{o.obsid}_stage3.png"), metrics


# --------------------------------------------------------------------------- STAGE 4
def stage4_offsets(o: Observation, sw):
    """JWST-VIRAC per-star dRA/dDec across the field (frame tie / PM precursor) + the
    reference-free inter-module (NRCA vs NRCB) offset."""
    import astropy.units as u
    from astropy.coordinates import search_around_sky
    metrics = dict(stage=4, sw=sw)
    path = _mosaic_path(o, sw)
    ref = _refcat_path(o)
    ep = aa.epoch_of(path) if path else None
    ref_sc, _ = aa.load_reference(ref, ep) if (ref and ep) else (None, None)
    jsc, _ = aa.detect(path) if path else (None, None)

    # inter-module offset -- ONLY when the release actually ships per-module (NRCA/NRCB)
    # mosaics for some filter.  Most releases are 'merged' only, so we simply omit the panel
    # rather than drawing an empty "no per-module mosaics" frame.
    mos = aa.find_mosaics(o.field)
    im = None
    for filt in (sw, o.filters[0] if o.filters else sw):
        mods = mos.get(filt.upper(), {})
        A = aa.detect(mods["nrca"])[0] if "nrca" in mods else (aa.detect(mods["nrcalong"])[0] if "nrcalong" in mods else None)
        B = aa.detect(mods["nrcb"])[0] if "nrcb" in mods else (aa.detect(mods["nrcblong"])[0] if "nrcblong" in mods else None)
        if A is not None and B is not None:
            im = aa.direct_intermodule(A, B)
            if im:
                metrics.update(intermodule_off=float(im["off"]), intermodule_filt=filt)
            break

    fig, ax = _fig(1, 2 if im else 1, 5.4, 5.0)
    a0 = ax[0][0]
    if jsc is not None and ref_sc is not None:
        bulk = aa.xcorr(jsc, ref_sc)
        ia, ib, sep, _ = search_around_sky(jsc, ref_sc, 0.3 * u.arcsec)
        if len(ia) >= 30:
            dra = (ref_sc[ib].ra - jsc[ia].ra).to(u.arcsec).value * np.cos(np.radians(jsc[ia].dec.value)) * 1000
            dde = (ref_sc[ib].dec - jsc[ia].dec).to(u.arcsec).value * 1000
            hb = a0.hexbin(dra, dde, gridsize=60, bins="log", cmap="cividis", mincnt=1)
            fig.colorbar(hb, ax=a0, label="log N pairs", shrink=0.85)
            a0.axhline(0, color="w", lw=0.5); a0.axvline(0, color="w", lw=0.5)
            a0.set_xlabel("dRA [mas]"); a0.set_ylabel("dDec [mas]")
            lim = max(100.0, 1.4 * (abs(bulk["off"]) if bulk else 0.0))
            a0.set_xlim(-lim, lim); a0.set_ylim(-lim, lim)
            a0.set_title(f"JWST-VIRAC  bulk={bulk['off']:.0f} mas" if bulk else "JWST-VIRAC", fontsize=9)
            metrics.update(bulk_off=float(bulk["off"]) if bulk else None,
                           bulk_dra=float(bulk["dra"]) if bulk else None,
                           bulk_ddec=float(bulk["ddec"]) if bulk else None,
                           n_matched=int(len(ia)))
    else:
        a0.text(0.5, 0.5, "need mosaic + VIRAC", ha="center", va="center")
    if im:
        a1 = ax[0][1]
        a1.bar(["dRA", "dDec"], [im["dra"], im["ddec"]], color=["#4477aa", "#ee6677"])
        a1.axhline(0, color="k", lw=0.5)
        a1.axhline(aa.THRESH["intermodule"], color="r", ls=":", lw=0.8)
        a1.axhline(-aa.THRESH["intermodule"], color="r", ls=":", lw=0.8)
        a1.set_ylabel("NRCA-NRCB [mas]")
        a1.set_title(f"inter-module {metrics['intermodule_filt']}  off={im['off']:.0f} mas", fontsize=9)
    bo = metrics.get("bulk_off"); io = metrics.get("intermodule_off")
    metrics["passed"] = bool((bo is not None and bo < aa.THRESH["absolute"]) and
                             (io is None or io < aa.THRESH["intermodule"]))
    fig.suptitle(f"{o.target} {o.obsid} — positional offsets", fontsize=11)
    return _save(fig, f"{o.obsid}_stage4.png"), metrics


# --------------------------------------------------------------------------- STAGE 5
_SW_DETS = ["nrca1", "nrca2", "nrca3", "nrca4", "nrcb1", "nrcb2", "nrcb3", "nrcb4"]


def _per_detector_offsets(o, filt, ref_sc):
    """Per-detector median residual vs a common frame, from the per-exposure daophot cats
    (pooled), for the 8 SW detectors.  Returns {det: dict(ra,dec,dra,dde,mad,n)} in mas.
    Uses the catalogs' skycoord_centroid (same WCS generation as the current mosaic)."""
    import astropy.units as u
    from astropy.table import vstack
    from astropy.coordinates import search_around_sky
    from astropy.table import Table
    out = {}
    for d in _SW_DETS:
        cats = glob.glob(f"{BASE}/{o.field}/{filt}/{filt.lower()}_{d}_visit*_*_m3_daophot_basic.fits")
        if not cats:
            continue
        try:
            T = vstack([Table.read(c) for c in cats], metadata_conflicts='silent')
        except (OSError, ValueError):
            continue
        if "skycoord_centroid" not in T.colnames:
            continue
        sc = T["skycoord_centroid"]
        ia, ib, sep, _ = search_around_sky(sc, ref_sc, 0.15 * u.arcsec)
        if len(ia) < 50:
            continue
        dra = (ref_sc[ib].ra - sc[ia].ra).to(u.mas).value * np.cos(np.radians(sc[ia].dec.deg))
        dde = (ref_sc[ib].dec - sc[ia].dec).to(u.mas).value
        out[d] = dict(ra=float(np.median(sc[ia].ra.deg)), dec=float(np.median(sc[ia].dec.deg)),
                      dra=float(np.median(dra)), dde=float(np.median(dde)),
                      mad=float(np.hypot(aa.mad_std(dra), aa.mad_std(dde))), n=int(len(ia)))
    return out


def _cutout_mosaic(o, filt):
    """Best full drizzled mosaic for the overlap-zone cutout gallery.  Prefer the all-detector
    'merged'; else a single-module mosaic ('nrcb'/'nrca' -- sickle is NRCB-only and names its
    mosaic 'nrcb', not 'merged')."""
    if not filt:
        return None
    d = f"{BASE}/{o.field}/{filt}/pipeline"
    def pick(tag):
        hits = [p for p in glob.glob(f"{d}/{o.obsid}_t001_nircam_clear-{filt.lower()}-{tag}_i2d.fits")
                if not any(s in p.lower() for s in ("residual", "model", "resbgsub", "bg_i2d"))]
        return hits[0] if hits else None
    return pick("merged") or pick("nrcb") or pick("nrca") or _mosaic_path(o, filt)


def _module_positions(o, filt):
    """(NRCA, NRCB) SkyCoords for the A/B tie, pooled from the per-detector daophot cats.
    The PIPELINE emits no merged-per-module mosaics (any on disk are stale, out-of-date
    artifacts), so the per-detector cats are the PRIMARY and only source.  Either module may
    be None for a single-module observation (e.g. sickle = NRCB only)."""
    from astropy.table import vstack, Table

    def pool(dets):
        cats = []
        for d in dets:
            cats += glob.glob(f"{BASE}/{o.field}/{filt}/{filt.lower()}_{d}_visit*_*_m3_daophot_basic.fits")
        if not cats:
            return None
        try:
            T = vstack([Table.read(c) for c in cats], metadata_conflicts="silent")
        except (OSError, ValueError):
            return None
        return T["skycoord_centroid"] if "skycoord_centroid" in T.colnames else None

    return pool(["nrca1", "nrca2", "nrca3", "nrca4"]), pool(["nrcb1", "nrcb2", "nrcb3", "nrcb4"])


def stage5_intermodule(o: Observation, sw):
    """Inter-detector / inter-module tie quality:
    (1) per-detector residual quiver vs VIRAC, bulk-subtracted (relative ties);
    (2) reference-free NRCA-vs-NRCB overlap: median offset + RMS of the SAME stars;
    (3) doubled-star cutout gallery on the module-overlap zone (mis-tie -> split PSF)."""
    import astropy.units as u
    from astropy.coordinates import search_around_sky, SkyCoord
    from astropy.io import fits
    from astropy.wcs import WCS
    from astropy.nddata import Cutout2D
    filt = sw
    metrics = dict(stage=5, filt=filt)
    ref = _viraccache_path(o) or _refcat_path(o)
    mpath = _cutout_mosaic(o, filt)                       # full mosaic for the cutout gallery
    ep = aa.epoch_of(mpath) if mpath else None
    ref_sc, _ = aa.load_reference(ref, ep) if (ref and ep) else (None, None)

    # (2) reference-free A vs B overlap from the per-detector cats (primary source).
    # CROWDING-ROBUST: the bulk A-B offset is the peak of the pair-separation histogram
    # (aa.xcorr) -- a direct search_around_sky+median fabricates pairs in a dense field (400k
    # chance coincidences within 0.3", RMS blown to ~100 mas). The RMS (tie precision) is the
    # residual scatter of the SAME stars: align A onto B by the peak, keep the tight matches.
    ov = None
    single_module = None
    a_sc, b_sc = _module_positions(o, filt)
    if (a_sc is None) ^ (b_sc is None):
        single_module = "NRCA" if a_sc is not None else "NRCB"
    if a_sc is not None and b_sc is not None and len(a_sc) >= 50 and len(b_sc) >= 50:
        xc = aa.xcorr(a_sc, b_sc, maxsep=1.5 * u.arcsec)
        if xc and xc["peak_ratio"] >= aa.MIN_PEAK_RATIO and xc["npairs"] >= 100:
            cosd = float(np.cos(np.radians(np.median(a_sc.dec.deg))))
            a_al = SkyCoord((a_sc.ra.deg + xc["dra"] / 1000.0 / 3600.0 / cosd) * u.deg,
                            (a_sc.dec.deg + xc["ddec"] / 1000.0 / 3600.0) * u.deg)
            ia, ib, sep, _ = search_around_sky(a_al, b_sc, 0.08 * u.arcsec)  # same star after align
            if len(ia) >= 20:
                dra = (a_al[ia].ra - b_sc[ib].ra).to(u.mas).value * cosd
                dde = (a_al[ia].dec - b_sc[ib].dec).to(u.mas).value
                ov = dict(dra=float(xc["dra"]), dde=float(xc["ddec"]), off=float(xc["off"]),
                          rms=float(np.hypot(aa.mad_std(dra), aa.mad_std(dde))),
                          n=int(len(ia)), peak_ratio=float(xc["peak_ratio"]),
                          pos=[(b_sc[i].ra.deg, b_sc[i].dec.deg) for i in ib[:200]])
                metrics.update(intermodule_off=ov["off"], intermodule_rms=ov["rms"],
                               n_overlap=ov["n"])

    # (1) per-detector residuals vs VIRAC, bulk-subtracted
    det = _per_detector_offsets(o, filt, ref_sc) if ref_sc is not None else {}
    if det:
        gdra = np.median([v["dra"] for v in det.values()])
        gdde = np.median([v["dde"] for v in det.values()])
        for v in det.values():
            v["rdra"], v["rdde"] = v["dra"] - gdra, v["dde"] - gdde
        mA = np.array([[det[d]["rdra"], det[d]["rdde"]] for d in det if d.startswith("nrca")])
        mB = np.array([[det[d]["rdra"], det[d]["rdde"]] for d in det if d.startswith("nrcb")])
        if len(mA) and len(mB):
            metrics["intermodule_diff"] = float(np.hypot(*(mA.mean(0) - mB.mean(0))))
            metrics["worst_detector"] = max(det, key=lambda d: np.hypot(det[d]["rdra"], det[d]["rdde"]))

    # ---- figure: 2 rows (quiver+overlap ; cutout gallery)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(11, 8.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 0.85])
    axq, axo = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])

    if det:
        xs = [v["ra"] for v in det.values()]; ys = [v["dec"] for v in det.values()]
        us = [v["rdra"] for v in det.values()]; vs = [v["rdde"] for v in det.values()]
        cols = ["#4477aa" if d.startswith("nrca") else "#ee6677" for d in det]
        q = axq.quiver(xs, ys, us, vs, color=cols, angles="xy", scale_units="xy",
                       scale=2000, width=0.007)
        axq.quiverkey(q, 0.12, 1.03, 5, "5 mas", labelpos="E", fontproperties={"size": 8})
        for d, v in det.items():
            axq.annotate(d, (v["ra"], v["dec"]), fontsize=6.5, ha="center", va="bottom")
        axq.invert_xaxis(); axq.set_xlabel("RA"); axq.set_ylabel("Dec")
        axq.set_title(f"per-detector residual (bulk-removed) — {filt}\n"
                      f"A-B diff = {metrics.get('intermodule_diff', float('nan')):.1f} mas", fontsize=9)
    else:
        axq.text(0.5, 0.5, "per-detector cats unavailable", ha="center", va="center", fontsize=8)

    if ov:
        # dra/dde are the same-star residuals after aligning A onto B by the histogram peak;
        # they scatter about 0 (RMS = tie precision). The bulk offset is the title number.
        axo.hexbin(dra, dde, gridsize=40, bins="log", cmap="cividis", mincnt=1)
        axo.axhline(0, color="w", lw=0.5); axo.axvline(0, color="w", lw=0.5)
        axo.set_xlabel("NRCA-NRCB residual dRA [mas]"); axo.set_ylabel("residual dDec [mas]")
        lim = max(50, 4 * ov["rms"])
        axo.set_xlim(-lim, lim); axo.set_ylim(-lim, lim)
        axo.set_title(f"A-vs-B overlap ({ov['n']} matched stars)\n"
                      f"offset={ov['off']:.1f} mas  RMS={ov['rms']:.1f} mas", fontsize=9)
    elif single_module:
        axo.text(0.5, 0.5, f"single module ({single_module} only)\nno A/B tie to check",
                 ha="center", va="center", fontsize=9)
        axo.set_xticks([]); axo.set_yticks([])
    else:
        axo.text(0.5, 0.5, "A/B overlap not measurable\n(need per-detector cats both modules)",
                 ha="center", va="center", fontsize=8)
        axo.set_xticks([]); axo.set_yticks([])

    # (3) doubled-star cutout gallery from the merged mosaic at overlap-star positions
    ncut = 6
    if ov and mpath and os.path.exists(mpath):
        with fits.open(mpath) as hdul:
            sci = hdul["SCI"] if "SCI" in hdul else hdul[1]
            data = sci.data.astype("float32"); w = WCS(sci.header)
        from astropy.coordinates import SkyCoord
        from astropy.visualization import ZScaleInterval, ImageNormalize, AsinhStretch
        picks = ov["pos"][:200]
        strip = fig.add_subplot(gs[1, :]); strip.axis("off")
        cut_axes = [strip.inset_axes([i / ncut + 0.01, 0.05, 0.92 / ncut, 0.85])
                    for i in range(ncut)]
        shown = 0
        for ra, dec in picks:
            if shown >= ncut:
                break
            try:
                x, y = w.world_to_pixel(SkyCoord(ra * u.deg, dec * u.deg))
                cut = Cutout2D(data, (float(x), float(y)), 25, wcs=w)
            except (ValueError, IndexError):
                continue
            if not np.isfinite(cut.data).any() or np.nanmax(cut.data) <= 0:
                continue
            a = cut_axes[shown]
            norm = ImageNormalize(cut.data, interval=ZScaleInterval(), stretch=AsinhStretch())
            a.imshow(cut.data, origin="lower", cmap="gray", norm=norm)
            a.set_xticks([]); a.set_yticks([])
            a.set_title(f"{shown + 1}", fontsize=7)
            shown += 1
        fig.text(0.5, 0.02, f"overlap-zone star cutouts from the merged mosaic "
                 f"(a mis-tie doubles/elongates these)", ha="center", fontsize=8)
    # single-module obs (sickle = NRCB only) has no A/B tie to fail -> N/A passes.
    if single_module:
        metrics["single_module"] = single_module
    metrics["passed"] = bool(single_module or (ov and ov["off"] < aa.THRESH["intermodule"]))
    fig.suptitle(f"{o.target} {o.obsid} — inter-detector / inter-module tie ({filt})", fontsize=11)
    return _save(fig, f"{o.obsid}_stage5.png"), metrics


STAGES = {1: stage1_mosaics, 2: stage2_cmd, 3: stage3_calibration, 4: stage4_offsets,
          5: stage5_intermodule}


def _build_stage5(o, sw, lw):
    return stage5_intermodule(o, sw)


def build_stage(o, n, sw, lw):
    if n == 1:
        return stage1_mosaics(o, sw, lw)
    if n == 2:
        return stage2_cmd(o, sw, lw)
    if n == 3:
        return stage3_calibration(o, sw)
    if n == 4:
        return stage4_offsets(o, sw)
    if n == 5:
        return stage5_intermodule(o, sw)
    raise ValueError(n)


CAPTIONS = {
    1: "**Stage 1 — first mosaics.** Grayscale {sw} (SW) and {lw} (LW) `i2d`. Confirms the "
       "observation was delivered and the mosaics are present and not obviously corrupt.",
    2: "**Stage 2 — colour-magnitude diagram** from the `{kind}` catalog ({n_stars} stars). "
       "LF-inset turnover ≈ {lf_turnover:.1f} tracks depth; regenerated as the catalog deepens.",
    3: "**Stage 3 — photometric calibration.** JWST {sw} vs VIRAC Ks for {n_matched} matched "
       "stars: slope {slope:.2f}, zp {zeropoint:.2f}, scatter {scatter:.2f} mag. A tight locus "
       "means the right stars were matched.",
    4: "**Stage 4 — positional offsets.** JWST−VIRAC ΔRA/ΔDec (bulk {bulk_off:.0f} mas) and the "
       "reference-free inter-module offset. First-order frame-match / proper-motion precursor.",
    5: "**Stage 5 — inter-detector / inter-module tie.** Per-detector residual quiver "
       "(bulk-removed; A–B diff {intermodule_diff:.1f} mas), the reference-free NRCA–NRCB overlap "
       "(offset {intermodule_off:.1f} mas, RMS {intermodule_rms:.1f} mas over {n_overlap} shared "
       "stars), and overlap-zone star cutouts (a mis-tie doubles them).",
}


def caption_for(n, metrics):
    try:
        return CAPTIONS[n].format(**{k: (v if v is not None else float("nan"))
                                     for k, v in metrics.items()})
    except (KeyError, ValueError):
        return CAPTIONS[n].split(".")[0] + "."


def _obs_from_disk(program, obs, base=BASE):
    """Fallback registry for on-cluster runs where the release portal is unreachable:
    find the field dir on disk holding this obs's mosaics and read its NIRCam filters."""
    from .observations import CURATED, FIELDS
    for d in sorted(glob.glob(f"{base}/*/")):
        fld = os.path.basename(d.rstrip("/"))
        hits = glob.glob(f"{base}/{fld}/*/pipeline/jw{int(program):05d}-o{obs}_t001_nircam_clear-*-merged_i2d.fits")
        if not hits:
            continue
        filts = sorted({m.group(1).upper() for h in hits
                        if (m := re.search(r"clear-(f\d{3}[wnm])-merged", os.path.basename(h).lower()))})
        cur = CURATED.get(f"jw{int(program):05d}-o{obs}", {})
        return Observation(program=str(int(program)), obs=obs,
                           target=FIELDS.get(fld, fld.title()),   # display name -> matches issue title
                           release_field=fld, instrument="NIRCam", filters=filts,
                           visits=cur.get("visits", []), epoch=cur.get("epoch", ""),
                           notes=cur.get("notes", ""))
    return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--program", required=True)
    ap.add_argument("--obs", required=True)
    ap.add_argument("--stage", nargs="+", type=int, default=[1, 2, 3, 4])
    ap.add_argument("--sw", default=None); ap.add_argument("--lw", default=None)
    ap.add_argument("--target", default=None, help="override display target (issue-title match)")
    ap.add_argument("--post", action="store_true", help="post/update the issue comments")
    ap.add_argument("--repo", default=os.environ.get("QA_REPO", "JWST-GC/data-qa"))
    args = ap.parse_args(argv)

    # diagnostics are NIRCam-only; when the portal registry is reachable it returns BOTH the
    # NIRCam and MIRI observation for a shared obsid (e.g. cloudc 2221-o002), so filter to
    # NIRCam explicitly -- else obs[0] can be the MIRI one (F2550W etc).
    obs = [o for o in registry(programs=[args.program])
           if o.obs == args.obs and o.instrument == "NIRCam"]
    o = obs[0] if obs else _obs_from_disk(args.program, args.obs)
    if o is None:
        print(f"no obs for program {args.program} obs {args.obs} (portal + on-disk both empty)",
              file=sys.stderr)
        return 1
    if args.target:
        o = replace(o, target=args.target)
    sw, lw = pick_filters(o.filters, args.sw, args.lw)
    print(f"{o.obsid}: SW={sw} LW={lw} filters={o.filters}")
    # metrics json where make_issues.render_body reads checkbox state; write INCREMENTALLY
    # and isolate each stage so a corrupt FITS / photutils failure / GitHub 5xx on one stage
    # doesn't drop the metrics of the stages that succeeded or stop later stages.
    mdir = os.path.join(os.path.dirname(__file__), "metrics")
    os.makedirs(mdir, exist_ok=True)
    mpath = os.path.join(mdir, f"{o.obsid}.json")
    all_metrics = {}
    if os.path.exists(mpath):
        try:
            with open(mpath) as fh:
                all_metrics = json.load(fh)
        except (OSError, ValueError):
            all_metrics = {}
    for n in args.stage:
        try:
            png, metrics = build_stage(o, n, sw, lw)
        except (OSError, ValueError, IndexError, KeyError, RuntimeError, AttributeError) as e:
            print(f"  stage {n}: FAILED to build: {type(e).__name__}: {e}", file=sys.stderr)
            all_metrics[f"stage{n}"] = dict(stage=n, error=f"{type(e).__name__}: {e}", passed=False)
            with open(mpath, "w") as fh:
                json.dump(all_metrics, fh, indent=2)
            continue
        all_metrics[f"stage{n}"] = metrics
        print(f"  stage {n}: {png}  passed={metrics.get('passed')}")
        with open(mpath, "w") as fh:          # persist before the (fallible) network post
            json.dump(all_metrics, fh, indent=2)
        if args.post:
            try:
                from .post_diagnostics import post_stage, PostError
                post_stage(o, n, png, caption_for(n, metrics), args.repo)
            except (PostError, OSError) as e:
                print(f"  stage {n}: post FAILED (figure built OK): {e}", file=sys.stderr)
    print(f"metrics -> {mpath}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
