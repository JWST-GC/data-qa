"""Progressive QA diagnostic figures, posted as replies (comments) to a per-observation
tracking issue.

Six stages, each emitted as the corresponding data product becomes available while the
cataloging pipeline runs.  Each stage returns ``(png_path, metrics)``; the metrics drive
the checkbox state in the issue body (see ``make_issues.render_body``, which reads stages
1-5; stage 6 is display-only), and the PNG is posted as an idempotent comment (one comment
per stage, keyed on a hidden marker).

    Stage 1  first i2d       one SW + one LW grayscale mosaic       "delivered", "mosaics present"
    Stage 2  CMD             LW vs SW-LW colour-magnitude + LF      "catalog vetted", "depth"
    Stage 3  calibration     JWST catalog mag vs VIRAC Ks          "photometry zeropoints"
    Stage 4  offsets         JWST-VIRAC dRA/dDec + significance + inter-module tie
    Stage 5  inter-detector  per-detector quiver + A/B overlap + overlap-star cutouts
    Stage 6  astrometry      per-star position sigma vs Vega mag (display-only)

This is a QA of RELEASEABLE PRODUCTS: every stage READS the catalog (the release merged
catalog, else the MAST-delivered per-i2d source catalog) and only bakes the crossmatch --
it does NOT re-detect sources.  Images LIVE IN THE ISSUE (posted to the GitHub CDN as
release assets on a single ``qa-assets`` bucket release, then embedded in the comment) --
NOT committed to the repo source tree.  Crossmatch / frame machinery from ``astrometry_audit``
(xcorr / load_reference).

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
_SW_PREF = ["F212N", "F210M", "F200W", "F187N", "F182M", "F164N", "F162M", "F150W", "F140M", "F115W"]
_LW_PREF = ["F480M", "F470N", "F466N", "F444W", "F410M", "F405N", "F360M", "F356W", "F335M",
            "F323N", "F300M", "F277W", "F250M"]

# VIRAC2 per-star PM-error floor (mas/yr): used for the offset-significance denominator when a
# field's refcache lacks e_pmRA/e_pmDE (median of the columns where they are present).
PM_ERR_FLOOR = 2.0

# A module with more than this fraction of NaN daophot centroids is an astrometry DEFECT to
# surface, not a few diverged fits to silently drop (real fields sit at ~0.01%).
_NAN_FRAC_FLAG = 0.05


def _channel(filt):
    return "SW" if int(filt[1:4]) <= 212 else "LW"


def _has_lw(o: "Observation"):
    """Does this obs actually carry an LW-channel filter?  Distinguishes a genuine
    single-channel obs (legitimately no colour) from a PREF-list gap (an LW mosaic exists but
    wasn't recognised) -- the two must NOT be conflated, or the latter silently PASSes with
    its LW data unexamined."""
    return any(_channel(f) == "LW" for f in getattr(o, "filters", []) if f)


def pick_filters(available, sw=None, lw=None):
    """Choose one SW + one LW filter from those available for the obs."""
    up = {f.upper() for f in available}
    sw = sw.upper() if sw else next((f for f in _SW_PREF if f in up), None)
    lw = lw.upper() if lw else next((f for f in _LW_PREF if f in up), None)
    return sw, lw


# --------------------------------------------------------------------------- product lookup
def _mosaic_path(o: Observation, filt):
    """Released merged i2d for this obs+filter, or None."""
    if not filt:                     # obs with no filter for this channel (e.g. a single-band obs)
        return None
    stem = f"{o.obsid}_t001_nircam_clear-{filt.lower()}-merged_i2d.fits"
    pats = [
        f"{BASE}/{o.field}/{filt}/pipeline/{stem}",
        f"{BASE}/{o.field}/*/pipeline/{stem}",
        f"{BASE}/{o.field}/images-merged/{stem}",   # not-yet-released fields (e.g. gc2211) land mosaics here
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


_OBS_TOK_RE = re.compile(r"_o(\d{3})\b")
_DATED_RE = re.compile(r"_(20\d{6})\b")          # _YYYYMMDD snapshot stamp


def _catalog_candidates(o: Observation):
    """Catalogs for THIS OBSERVATION, each tagged (path, kind, tier, mtime).  A field dir holds
    catalogs for every obs in the field, so we must NOT hand another observation's catalog to
    this obs's QA (that posts byte-identical, wrongly-green figures on multi-obs fields like
    gc2211 / cloudef).  Rules:
      * if any catalog is tokened for THIS obs (``_o<obs>``) -> use only those;
      * else drop catalogs tokened for a DIFFERENT obs (keep only field-level / untokened);
      * always drop ``_YYYYMMDD`` dated snapshots when a non-dated catalog remains (a later
        dedup pass makes the live catalog SMALLER, so size-based tie-breaks would otherwise
        prefer a stale pre-dedup snapshot -- a provenance violation).
    Skips residual/model/region sidecars."""
    cand = []
    for p in sorted(glob.glob(f"{BASE}/{o.field}/catalogs/*.fits")):
        low = os.path.basename(p).lower()
        if any(s in low for s in ("_residual", "_model", "_reproject", "region")):
            continue
        try:
            mtime = os.path.getmtime(p)
        except OSError:
            mtime = 0.0
        tier, kind = _catalog_priority(low)
        cand.append((p, kind, tier, mtime, low))
    this = [c for c in cand
            if (m := _OBS_TOK_RE.search(c[4])) and m.group(1) == o.obs]
    if this:
        cand = this
    else:
        cand = [c for c in cand if not _OBS_TOK_RE.search(c[4])]     # drop other-obs catalogs
    nondated = [c for c in cand if not _DATED_RE.search(c[4])]
    if nondated:
        cand = nondated
    return [(p, kind, tier, mtime) for (p, kind, tier, mtime, _low) in cand]


def _catalog_for(o: Observation, sw, lw):
    """Catalog that contains VEGA mags for both requested filters, chosen by RISING pipeline
    priority (MAST default < m1 < ... < m8), size breaking ties within a tier.  A cheap
    FITS-header probe (TTYPE/NAXIS2) avoids reading catalog data.
    Returns (path, kind, sw_col, lw_col) or (None,...)."""
    from astropy.io import fits
    best = (None, None, None, None, (-1.0, -1.0))
    for p, kind, tier, mtime in _catalog_candidates(o):
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
        rank = (tier, mtime)                       # highest tier, then NEWEST (not largest: dedup shrinks)
        if rank > best[-1]:
            best = (p, kind, csw, clw, rank)
    return best[:4]


def _catalog_with_vega(o: Observation, filt):
    """Highest-priority catalog carrying ``mag_vega_<filt>`` (+ its skycoord col), or
    (None, None, None).  Used to Vega-calibrate the per-exposure instrumental mags."""
    from astropy.io import fits
    want_mag = f"mag_vega_{filt.lower()}"
    want_sc = f"skycoord_{filt.lower()}"
    best = (None, None, None, (-1.0, -1.0))
    for p, kind, tier, mtime in _catalog_candidates(o):
        try:
            hdr = fits.getheader(p, ext=1)
        except (OSError, IndexError):
            continue
        ncol = hdr.get("TFIELDS", 0)
        low = {str(hdr.get(f"TTYPE{i}", "")).lower() for i in range(1, ncol + 1)}
        # skycoord mixin serializes as "<name>.ra"/".dec" in TTYPE
        rank = (tier, mtime)                       # highest tier, then newest (skip stale snapshots)
        if want_mag in low and f"{want_sc}.ra" in low and rank > best[-1]:
            best = (p, want_mag, want_sc, rank)
    return best[:3]


def _vega_zeropoint(o: Observation, filt, sc, instr):
    """Robust instrumental->Vega zeropoint ZP (vega = instr + ZP) for one filter, from matching
    the pooled per-exposure detections to the merged catalog's ``mag_vega_<filt>``.  Uses the
    bright 40% (where the instrumental mag is well-measured) for the median.  Returns ZP or
    None when there is no Vega catalog / too few matches."""
    import astropy.units as u
    from astropy.table import Table
    cat, magcol, sccol = _catalog_with_vega(o, filt)
    if not cat:
        return None
    m = Table.read(cat)
    if sccol not in m.colnames or magcol not in m.colnames:
        return None
    vg = np.asarray(m[magcol], float)
    idx, sep, _ = sc.match_to_catalog_sky(m[sccol])
    good = (sep < 0.05 * u.arcsec) & np.isfinite(instr) & np.isfinite(vg[idx])
    if good.sum() < 50:
        return None
    ii = instr[good]
    zz = vg[idx][good] - ii
    bright = ii <= np.percentile(ii, 40)     # bright end: cleanest instrumental mags
    return float(np.median(zz[bright])) if bright.sum() >= 20 else float(np.median(zz))


def _mast_source_catalog(o: Observation, filt):
    """MAST-delivered per-i2d source catalog for one filter (single-band), or None.  Named
    ``<obsid>_t001_nircam_clear-<filt>_cat.fits`` next to the i2d -- NOT the per-detector
    ``*_nrcaN_destreak_cat.fits`` intermediates."""
    for d in (f"{BASE}/{o.field}/{filt}/pipeline", f"{BASE}/{o.field}/*/pipeline",
              f"{BASE}/{o.field}/images-merged"):
        hits = [p for p in glob.glob(f"{d}/{o.obsid}_t001_nircam_*{filt.lower()}*_cat.fits")
                if not any(t in os.path.basename(p).lower() for t in ("nrca", "nrcb", "destreak"))]
        if hits:
            return sorted(hits)[-1]
    return None


def _jwst_sources(o: Observation, filt):
    """JWST source positions + magnitude for one filter, READ FROM THE CATALOG -- never
    re-detected.  This is a QA of releaseable products: show what the catalog contains, don't
    bake our own detection.  Priority: the RELEASE merged catalog (mag_vega_<filt> +
    skycoord_<filt>); else the MAST-delivered per-i2d source catalog (sky_centroid +
    aper abmag).  Returns (SkyCoord, mag, source_label) or (None, None, None) -> caller red-flags.
    The only thing baked downstream is the crossmatch (to VIRAC / across filters)."""
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    from astropy.table import Table
    # 1) release merged catalog
    cat, magcol, sccol = _catalog_with_vega(o, filt)
    if cat:
        m = Table.read(cat)
        if sccol in m.colnames and magcol in m.colnames:
            sc = m[sccol]
            mag = np.asarray(m[magcol], float)
            g = np.isfinite(sc.ra.deg) & np.isfinite(sc.dec.deg) & np.isfinite(mag)
            if g.sum() >= 30:
                return sc[g], mag[g], f"release:{os.path.basename(cat)}"
    # 2) MAST-delivered per-i2d source catalog
    mp = _mast_source_catalog(o, filt)
    if mp:
        m = Table.read(mp)
        magc = next((c for c in ("aper_total_abmag", "aper50_abmag", "aper70_abmag",
                                 "aper30_abmag") if c in m.colnames), None)
        if "sky_centroid" in m.colnames and magc:
            sc = m["sky_centroid"]
            mag = np.asarray(m[magc], float)
            g = np.isfinite(sc.ra.deg) & np.isfinite(sc.dec.deg) & np.isfinite(mag)
            if g.sum() >= 30:
                return sc[g], mag[g], f"MAST:{os.path.basename(mp)}"
        else:
            # sky_centroid stored as split float columns rather than a mixin; find the actual
            # (case-preserving) column names, and require a magnitude -- an all-NaN mag would
            # blank the CMD / fail stage 3 instead of cleanly red-flagging.
            low = {c.lower(): c for c in m.colnames}
            if "sky_centroid.ra" in low and "sky_centroid.dec" in low and magc:
                ra = np.asarray(m[low["sky_centroid.ra"]], float)
                dec = np.asarray(m[low["sky_centroid.dec"]], float)
                mag = np.asarray(m[magc], float)
                g = np.isfinite(ra) & np.isfinite(dec) & np.isfinite(mag)
                if g.sum() >= 30:
                    return SkyCoord(ra[g] * u.deg, dec[g] * u.deg), mag[g], f"MAST:{os.path.basename(mp)}"
    return None, None, None


def _crossmatch_cmd_arrays(o: Observation, sw, lw):
    """Build CMD (color, mag) by crossmatching the two SINGLE-BAND source catalogs -- MAST
    per-i2d, or per-filter RELEASE catalogs -- when no single MERGED catalog carries both bands.
    The color is the only baking allowed.  Uses a MUTUAL nearest match at a tight radius (so no
    LW source is reused by many SW sources -> the CMD width is the catalog's, not the crossmatch's)
    and REFUSES to mix magnitude systems (release=Vega, MAST=AB).  Returns
    (color, mag, n, label, magsys) or None."""
    import astropy.units as u
    if not lw:
        return None
    sc_sw, m_sw, lab_sw = _jwst_sources(o, sw)
    sc_lw, m_lw, lab_lw = _jwst_sources(o, lw)
    if sc_sw is None or sc_lw is None:
        return None
    sys_sw = "Vega" if str(lab_sw).startswith("release") else "AB"
    sys_lw = "Vega" if str(lab_lw).startswith("release") else "AB"
    if sys_sw != sys_lw:
        return None                      # never plot a Vega-minus-AB colour
    i_swlw, sep1, _ = sc_sw.match_to_catalog_sky(sc_lw)
    i_lwsw, _, _ = sc_lw.match_to_catalog_sky(sc_sw)
    mutual = (sep1 < 0.05 * u.arcsec) & (i_lwsw[i_swlw] == np.arange(len(sc_sw)))
    if mutual.sum() < 100:
        return None
    color = m_sw[mutual] - m_lw[i_swlw[mutual]]
    mag = m_lw[i_swlw[mutual]]
    kind = "MAST" if sys_sw == "AB" else "release"
    return color, mag, int(mutual.sum()), f"{kind} positional crossmatch", sys_sw


def _refcat_path(o: Observation):
    """VIRAC2-Gaia refcat (newest epoch) for the absolute-frame (position-only) check."""
    hits = sorted(glob.glob(f"{BASE}/{o.field}/catalogs/gaia_virac2_refcat_epoch*.fits"))
    return hits[-1] if hits else None


def _obs_epoch(o: Observation, mosaic_path):
    """Observation epoch (jyear) for PM-propagating the VIRAC reference.  Prefer the mosaic
    DATE-OBS, but fall back to the epoch baked into the refcat filename
    (gaia_virac2_refcat_epoch<YYYY.dd>.fits) so the catalog-based stages don't hard-depend on a
    mosaic being on disk."""
    if mosaic_path:
        ep = aa.epoch_of(mosaic_path)
        if ep:
            return ep
    rp = _refcat_path(o)
    if rp:
        m = re.search(r"epoch(\d{4}\.\d+)", os.path.basename(rp))
        if m:
            return float(m.group(1))
    return None


def _viraccache_path(o: Observation):
    """Raw VIRAC2 cache (has a real Ksmag column) for the photometric-calibration check.
    The gaia_virac2 refcat carries only a blended 'refmag', unusable for a Ks zeropoint."""
    p = f"{BASE}/{o.field}/astrometry_diag/refcache/virac2.fits"
    return p if os.path.exists(p) else None


_DAO_OBS_RE = re.compile(r"_o(\d{3})_")     # per-exposure token is underscore-bounded: _o023_visit


def _daophot_glob(o: Observation, filt, det="*"):
    """Obs-scoped per-exposure daophot cats for filt (+ detector).  Fields name these either
    untokened (brick: ``f212n_nrca1_visit*``) or per-obs (gc2211: ``f200w_nrca1_o023_visit*``,
    with an older untokened generation possibly alongside).  Never hand one observation's
    per-exposure cats to another's QA (stage 5 / 6 / significance):
      * prefer THIS obs's tokened files;
      * if a per-obs generation exists but not for this obs -> return [] (don't fall back to a
        different obs or a stale untokened generation);
      * else use the untokened files (single-obs-per-field layout)."""
    base = f"{BASE}/{o.field}/{filt}/{filt.lower()}_{det}"
    tok = sorted(glob.glob(f"{base}_o{o.obs}_visit*_*_m3_daophot_basic.fits"))
    if tok:
        return tok
    if glob.glob(f"{base}_o[0-9][0-9][0-9]_visit*_*_m3_daophot_basic.fits"):
        return []
    return [c for c in sorted(glob.glob(f"{base}_visit*_*_m3_daophot_basic.fits"))
            if not _DAO_OBS_RE.search(os.path.basename(c))]


def _virac_with_errors(o: Observation, epoch):
    """VIRAC2 cache PM-propagated to ``epoch`` WITH per-star position sigma at that epoch:
    sigma = hypot(base position error, |baseline| * PM error).  The gaia_virac2 refcat used
    elsewhere carries no per-star error, so offset SIGNIFICANCE needs the raw cache
    (e_RAJ2000/e_pmRA...).  Returns (SkyCoord, sig_ra_mas, sig_de_mas) or None.  At an ~8.7 yr
    baseline the PM-error term (~2 mas/yr) dominates -- so it is the right denominator for
    'is the measured offset significant?', not the JWST single-exposure sigma alone.

    Prefer ``virac2_full.fits`` (carries per-star e_pmRA/e_pmDE) over ``virac2.fits`` -- several
    fields' virac2.fits lacks the PM-error columns, and using the real per-star PM errors beats
    a constant floor (which collapses the significance to a fixed unit conversion)."""
    full = f"{BASE}/{o.field}/astrometry_diag/refcache/virac2_full.fits"
    p = full if os.path.exists(full) else _viraccache_path(o)
    if not p:
        return None
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    from astropy.table import Table
    t = Table.read(p)
    need = {"RAJ2000", "DEJ2000", "e_RAJ2000", "e_DEJ2000", "pmRA", "pmDE"}
    if not need.issubset(set(t.colnames)):
        return None
    ra = np.asarray(t["RAJ2000"], float); dec = np.asarray(t["DEJ2000"], float)
    pmra = np.nan_to_num(np.asarray(t["pmRA"], float))
    pmdec = np.nan_to_num(np.asarray(t["pmDE"], float))
    dt = epoch - 2014.0                                 # VIRAC2 base epoch
    ra = ra + (pmra * dt / 3.6e6) / np.cos(np.radians(dec))
    dec = dec + pmdec * dt / 3.6e6
    # Some field caches carry position errors but not per-star PM errors; the PM-error term
    # dominates at an 8.7 yr baseline, so fall back to VIRAC2's ~2 mas/yr median rather than
    # dropping it (which would spuriously inflate the offset significance).
    e_pmra = np.asarray(t["e_pmRA"], float) if "e_pmRA" in t.colnames else np.full(len(t), PM_ERR_FLOOR)
    e_pmde = np.asarray(t["e_pmDE"], float) if "e_pmDE" in t.colnames else np.full(len(t), PM_ERR_FLOOR)
    e_pmra = np.where(np.isfinite(e_pmra), e_pmra, PM_ERR_FLOOR)
    e_pmde = np.where(np.isfinite(e_pmde), e_pmde, PM_ERR_FLOOR)
    sra = np.hypot(np.asarray(t["e_RAJ2000"], float), abs(dt) * e_pmra)
    sde = np.hypot(np.asarray(t["e_DEJ2000"], float), abs(dt) * e_pmde)
    g = (np.isfinite(ra) & np.isfinite(dec) & np.isfinite(sra) & np.isfinite(sde) &
         (sra > 0) & (sde > 0))
    if g.sum() < 30:
        return None
    return SkyCoord(ra[g] * u.deg, dec[g] * u.deg), sra[g], sde[g]


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


def plt_circle(r, color):
    """Dashed origin-centred circle patch (used to mark n-sigma contours)."""
    from matplotlib.patches import Circle
    return Circle((0, 0), r, fill=False, ec=color, lw=1.0, ls="--")


def _red_flag_figure(o, stage_name, title, reason):
    """A literal RED FLAG.  When a stage plot would be EMPTY (no data to show), an empty
    scatter reads as 'fine, nothing wrong' -- the opposite of the truth.  Draw an
    unmistakable red panel instead so the missing measurement stands out at a glance."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(7.6, 3.4))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_axis_off()
    ax.add_patch(plt.Rectangle((0, 0), 1, 1, color="#b30000"))
    ax.text(0.5, 0.72, "⚑ RED FLAG", color="white", fontsize=30, fontweight="bold",
            ha="center", va="center")
    ax.text(0.5, 0.45, title, color="white", fontsize=15, fontweight="bold",
            ha="center", va="center")
    ax.text(0.5, 0.22, reason, color="#ffe0e0", fontsize=10, ha="center", va="center",
            wrap=True)
    ax.text(0.5, 0.05, f"{o.target} {o.obsid}", color="#ffd0d0", fontsize=8,
            ha="center", va="center")
    return _save(fig, f"{o.obsid}_{stage_name}.png")


# --------------------------------------------------------------------------- STAGE 1
def stage1_mosaics(o: Observation, sw, lw):
    """One full-width grayscale panel PER FILTER, stacked vertically.  NIRCam mosaics are
    wide-aspect strips; a side-by-side SW/LW layout squishes them and leaves huge whitespace,
    so each filter gets its own row sized to the image's native aspect."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from astropy.io import fits

    psw, plw = _mosaic_path(o, sw), _mosaic_path(o, lw)
    panels = []                                   # (filt, path, aspect = ny/nx)
    for filt, p in ((sw, psw), (lw, plw)):
        if not filt:
            continue                              # single-filter obs: no LW row at all
        asp = 0.45
        if p:
            with fits.open(p) as h:
                s = h["SCI"] if "SCI" in h else h[1]
                ny, nx = s.data.shape
                asp = ny / nx if nx else 0.45
        panels.append((filt, p, asp))

    W = 11.0
    # per-row height from native aspect (clamp so a near-square or a razor-thin strip stays
    # legible); a missing i2d gets a short placeholder row.
    heights = [(0.25 if p is None else max(0.18, min(1.0, asp))) * W for _, p, asp in panels]
    fig = plt.figure(figsize=(W, sum(heights) + 0.35 * len(panels)))
    gs = fig.add_gridspec(len(panels), 1, height_ratios=heights, hspace=0.18)
    fracs = {}
    for i, (filt, p, _asp) in enumerate(panels):
        a = fig.add_subplot(gs[i, 0])
        if p:
            fracs[filt] = _grayscale(a, p, f"{o.obsid}  {filt}")
        else:
            a.text(0.5, 0.5, f"{filt}\n(no i2d)", ha="center", va="center")
            a.set_xticks([]); a.set_yticks([])
    png = _save(fig, f"{o.obsid}_stage1.png")
    metrics = dict(stage=1, sw=sw, lw=lw,
                   sw_present=bool(psw), lw_present=bool(plw),
                   finite_fraction=fracs,
                   # pass on SW alone ONLY when the obs genuinely has no LW-channel filter;
                   # if an LW filter exists but its mosaic/pick is missing, that is NOT a pass.
                   passed=bool(psw and (plw or not _has_lw(o))))
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
    """Colour-magnitude diagram (LW vs SW-LW) with the luminosity function as a RIGHT-SIDE
    marginal whose y-axis (magnitude) is locked to the CMD -- the LF reads straight across
    from the CMD instead of an inset floating on top of the data."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from astropy.table import Table
    cat, kind, csw, clw = _catalog_for(o, sw, lw)
    metrics = dict(stage=2, catalog=os.path.basename(cat) if cat else None, kind=kind)
    want = f"{sw}+{lw}" if lw else f"{sw}"
    if not cat:
        # No single merged catalog with both bands -> build the CMD by crossmatching the two
        # single-band catalogs (release or MAST; the only baking allowed).  Absent too -> red flag.
        mc = _crossmatch_cmd_arrays(o, sw, lw)
        if mc is not None:
            color, mag, nkeep, mlabel, magsys = mc
            fig, ax = _fig(1, 1, 6.2, 6.0)
            a = ax[0][0]
            hb = a.hexbin(color, mag, gridsize=100, bins="log", cmap="viridis", mincnt=1)
            fig.colorbar(hb, ax=a, label="log N stars")
            a.set_xlim(np.nanpercentile(color, [1, 99]))
            ylo, yhi = np.nanpercentile(mag, [0.5, 99.5]); a.set_ylim(yhi, ylo)
            a.set_xlabel(f"{sw} - {lw} [{magsys}]"); a.set_ylabel(f"{lw} [{magsys}]")
            a.set_title(f"{o.target} {o.obsid} — CMD ({mlabel}, n={nkeep})\n"
                        f"width is positional-crossmatch limited, not the catalog's colour precision",
                        fontsize=8)
            metrics.update(n_stars=nkeep, kind="crossmatch", mag_system=magsys, passed=nkeep > 500)
            return _save(fig, f"{o.obsid}_stage2.png"), metrics
        png = _red_flag_figure(o, "stage2", "NO CATALOG FOR CMD",
                               f"No release catalog and no MAST source catalog for {want} yet.")
        metrics.update(red_flag=True, red_flag_reason=f"no catalog for {want}", passed=False)
        return png, metrics
    t = Table.read(cat)

    # LF-only fallback ONLY for a genuine single-channel obs (no LW filter at all).  If lw is
    # None but the obs DOES carry an LW-channel filter, that is a PREF-gap / missing-mosaic
    # problem, not a single-band obs -- fall through so it fails visibly instead of quietly
    # degrading to a single-band LF with the LW data unexamined.
    if lw is None and csw and not _has_lw(o):
        fig, ax = _fig(1, 1, 6.5, 5.0)
        a = ax[0][0]
        m = np.asarray(t[csw], float); g = np.isfinite(m)
        hh, edges = np.histogram(m[g], bins=60)
        ctr = 0.5 * (edges[1:] + edges[:-1])
        a.step(ctr, hh, where="mid", color="k", lw=1.0)
        peak = ctr[int(np.argmax(hh))]
        a.axvline(peak, color="r", lw=0.8, label=f"turnover≈{peak:.1f}")
        a.set_xlabel(sw); a.set_ylabel("N stars"); a.legend(fontsize=8)
        a.set_title(f"{o.target} {o.obsid} — {sw} luminosity function "
                    f"(single filter — no colour)", fontsize=10)
        metrics.update(n_stars=int(g.sum()), lf_turnover=float(peak),
                       sw_col=csw, lw_col=None, single_filter=True,
                       passed=int(g.sum()) > 500)
        return _save(fig, f"{o.obsid}_stage2.png"), metrics

    if not (csw and clw):
        fig, ax = _fig(1, 1, 5.5, 6.0)
        ax[0][0].text(0.5, 0.5, f"no {sw}/{lw} mag cols\nin {os.path.basename(cat)}",
                      ha="center", va="center", fontsize=8)
        metrics["passed"] = False
        return _save(fig, f"{o.obsid}_stage2.png"), metrics

    # CMD + shared-y marginal LF
    msw = np.asarray(t[csw], float); mlw = np.asarray(t[clw], float)
    g = np.isfinite(msw) & np.isfinite(mlw)
    color = msw[g] - mlw[g]; mag = mlw[g]
    fig = plt.figure(figsize=(8.2, 6.2))
    # main CMD | marginal LF | dedicated colorbar column (so the bar doesn't steal marginal width)
    gs = fig.add_gridspec(1, 3, width_ratios=[4.0, 1.15, 0.16], wspace=0.05)
    a = fig.add_subplot(gs[0, 0])
    amarg = fig.add_subplot(gs[0, 1], sharey=a)          # y-axis (mag) LOCKED to the CMD
    cax = fig.add_subplot(gs[0, 2])
    hb = a.hexbin(color, mag, gridsize=120, bins="log", cmap="viridis", mincnt=1)
    a.set_xlabel(f"{sw} - {lw}"); a.set_ylabel(lw)
    a.set_xlim(np.nanpercentile(color, [1, 99]))
    # y-range from the mag percentiles (LF outliers otherwise leave ~1/3 of the panel empty);
    # inverted so brighter is up.
    ylo, yhi = np.nanpercentile(mag, [0.5, 99.5])
    a.set_ylim(yhi, ylo)                                  # marginal follows via sharey
    fig.colorbar(hb, cax=cax, label="log N stars (CMD)")
    # marginal LF: counts vs magnitude, bars run horizontally so mag lines up with the CMD
    hh, edges = np.histogram(mag, bins=50)
    ctr = 0.5 * (edges[1:] + edges[:-1])
    amarg.step(hh, ctr, where="mid", color="k", lw=0.9)
    peak = ctr[int(np.argmax(hh))]
    amarg.axhline(peak, color="r", lw=0.8)
    amarg.set_xlabel(f"N stars\nturnover≈{peak:.1f}", fontsize=8)
    amarg.tick_params(labelleft=False, labelsize=7)
    amarg.margins(y=0)
    metrics.update(n_stars=int(g.sum()), lf_turnover=float(peak),
                   sw_col=csw, lw_col=clw, passed=int(g.sum()) > 500)
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
    ep = _obs_epoch(o, path)
    ref_sc, ref_mag = aa.load_reference(ref, ep) if (ref and ep) else (None, None)
    # Read the JWST catalog (release -> MAST) -- do NOT re-detect on the mosaic.
    jsc, jmag, src = _jwst_sources(o, sw)
    metrics["source"] = src
    a = ax[0][0]
    if jsc is None:
        import matplotlib.pyplot as plt
        plt.close(fig)          # close the empty fig before the red-flag builds its own
        png = _red_flag_figure(o, "stage3", "NO CATALOG TO CALIBRATE",
                               f"No release or MAST source catalog for {sw} yet — nothing to QA.")
        metrics.update(red_flag=True, red_flag_reason=f"no catalog for {sw}", passed=False)
        return png, metrics
    if ref_sc is None or ref_mag is None:
        a.text(0.5, 0.5, "need VIRAC refcat", ha="center", va="center")
        metrics["passed"] = False
        return _save(fig, f"{o.obsid}_stage3.png"), metrics
    # Anchor on VIRAC (sparse, Ks-bright) -> nearest JWST catalog source.  The release catalog
    # goes far deeper than VIRAC, so an all-pairs match would pair faint JWST sources with the
    # wrong VIRAC star and blow up the scatter; nearest-from-VIRAC keeps the locus clean.
    idx, sep, _ = ref_sc.match_to_catalog_sky(jsc)
    keep = sep < 0.1 * u.arcsec
    if keep.sum() < 30:
        a.text(0.5, 0.5, f"only {int(keep.sum())} matches", ha="center", va="center")
        metrics["passed"] = False
        return _save(fig, f"{o.obsid}_stage3.png"), metrics
    x = ref_mag[keep]; y = jmag[idx[keep]]
    g = np.isfinite(x) & np.isfinite(y)
    x, y = x[g], y[g]
    # robust linear fit y = slope*x + zp, sigma-clipped to the locus.  The deep release catalog
    # matched to VIRAC has a red/mismatch cloud above the locus (Ks-bright, F212N-faint stars);
    # one clip pass measures the CALIBRATION scatter (is the zeropoint sane) rather than the
    # astrophysical colour spread.
    # Iterate the 3-sigma locus clip to CONVERGENCE (a single pass is not converged -- the
    # reported slope/scatter otherwise depend on stopping after one step at k=3).
    xf, yf = x, y
    slope, zp = np.polyfit(xf, yf, 1)
    for _ in range(5):
        resid = yf - (slope * xf + zp)
        loc = np.abs(resid) < 3 * aa.mad_std(resid)
        if loc.all() or loc.sum() < 30:
            break
        xf, yf = xf[loc], yf[loc]
        slope, zp = np.polyfit(xf, yf, 1)
    scat = float(aa.mad_std(yf - (slope * xf + zp)))
    n_locus = int(len(xf))
    hb = a.hexbin(x, y, gridsize=80, bins="log", cmap="magma", mincnt=1)
    fig.colorbar(hb, ax=a, label="log N stars", shrink=0.85)
    xs = np.array([np.nanmin(x), np.nanmax(x)])
    a.plot(xs, slope * xs + zp, "c-", lw=1, label=f"slope={slope:.2f} zp={zp:.2f}")
    a.set_xlabel("VIRAC Ks [mag]"); a.set_ylabel(f"JWST {sw} catalog mag")
    a.legend(fontsize=8, loc="upper left")
    a.set_title(f"{o.obsid} calibration  n={int(g.sum())} (locus {n_locus})  "
                f"scatter={scat:.2f}", fontsize=10)
    # Split gate: keep the SLOPE window tight (a zeropoint check must falsify on slope), widen
    # only the SCATTER for the real narrow-vs-broad (F212N vs Ks) colour/extinction spread.
    metrics.update(n_matched=int(g.sum()), n_locus=n_locus, slope=float(slope),
                   zeropoint=float(zp), scatter=scat,
                   passed=(0.8 < slope < 1.2 and scat < 0.8))
    return _save(fig, f"{o.obsid}_stage3.png"), metrics


# --------------------------------------------------------------------------- STAGE 4
def _pooled_daophot(o: Observation, filt, max_files=64):
    """Pool the per-exposure DAOPHOT cats for one filter into (position, per-star astrometric
    sigma, instrumental mag, flux).  Unlike the merged science catalog, the per-exposure cats
    carry the formal PSF-fit position uncertainty: ``dra``/``ddec`` are the RA/Dec 1-sigma
    errors in arcsec (== x_err/y_err * pixel scale, so no pixel-scale assumption is needed).
    Returns (SkyCoord, sig_ra_mas, sig_de_mas, instr_mag, flux) or None."""
    import astropy.units as u
    from astropy.table import vstack, Table
    cats = _daophot_glob(o, filt)          # obs-scoped
    if not cats:
        return None
    if len(cats) > max_files:
        # Cap the pool, but sample ROUND-ROBIN across detectors: a plain alphabetical head takes
        # nrca1..nrca3 and drops nrca4 + all of NRCB, so the significance panel / stage-6 curve
        # would describe module A only -- exactly the module the inter-module tie compares.
        by_det = {}
        for c in cats:
            m = re.search(r"_(nrc[ab](?:[1-4]|long))_", os.path.basename(c))
            by_det.setdefault(m.group(1) if m else "z", []).append(c)
        dets, picked = sorted(by_det), []
        while len(picked) < max_files and any(by_det.values()):
            for d in dets:
                if by_det[d]:
                    picked.append(by_det[d].pop(0))
                    if len(picked) >= max_files:
                        break
        cats = picked
    need = {"skycoord_centroid", "dra", "ddec", "flux_fit"}
    tabs = []
    for c in cats:
        try:
            t = Table.read(c)
        except (OSError, ValueError):
            continue
        if need.issubset(set(t.colnames)):
            tabs.append(t["skycoord_centroid", "dra", "ddec", "flux_fit"])
    if not tabs:
        return None
    T = vstack(tabs, metadata_conflicts="silent")
    sc = T["skycoord_centroid"]
    sig_ra = np.asarray(T["dra"], float) * 1000.0     # arcsec -> mas
    sig_de = np.asarray(T["ddec"], float) * 1000.0
    flux = np.asarray(T["flux_fit"], float)
    with np.errstate(invalid="ignore", divide="ignore"):
        mag = -2.5 * np.log10(flux)
    good = (np.isfinite(sc.ra.deg) & np.isfinite(sc.dec.deg) & np.isfinite(sig_ra) &
            np.isfinite(sig_de) & (sig_ra > 0) & (sig_de > 0) & np.isfinite(mag) & (flux > 0))
    if good.sum() < 50:
        return None
    return sc[good], sig_ra[good], sig_de[good], mag[good], flux[good]


def _binned_stat(x, y, width=0.5, minn=15):
    """Median + 16/84 percentile band of ``y`` in fixed-width bins of ``x``.  Bins with fewer
    than ``minn`` points are dropped.  Returns (med, p16, p84, centre) arrays or (None,)*4."""
    g = np.isfinite(x) & np.isfinite(y) & (y > 0)
    x, y = x[g], y[g]
    if x.size < minn:
        return None, None, None, None
    edges = np.arange(np.floor(x.min() / width) * width,
                      np.ceil(x.max() / width) * width + width, width)
    idx = np.digitize(x, edges)
    ctr, med, p16, p84 = [], [], [], []
    for b in range(1, len(edges)):
        m = idx == b
        if m.sum() < minn:
            continue
        ctr.append(0.5 * (edges[b - 1] + edges[b]))
        med.append(np.median(y[m]))
        p16.append(np.percentile(y[m], 16)); p84.append(np.percentile(y[m], 84))
    if len(ctr) < 3:
        return None, None, None, None
    return np.array(med), np.array(p16), np.array(p84), np.array(ctr)


def stage4_offsets(o: Observation, sw):
    """JWST-VIRAC per-star dRA/dDec across the field (frame tie / PM precursor), the per-star
    offset SIGNIFICANCE (measured offset / its uncertainty), and the reference-free
    inter-module (NRCA vs NRCB) offset."""
    import astropy.units as u
    from astropy.coordinates import search_around_sky
    metrics = dict(stage=4, sw=sw, offset_signif_med=None, offset_med_mas=None)
    path = _mosaic_path(o, sw)
    ref = _refcat_path(o)
    ep = _obs_epoch(o, path)
    ref_sc, _ = aa.load_reference(ref, ep) if (ref and ep) else (None, None)
    # JWST positions from the catalog (release -> MAST), NOT re-detected on the mosaic.
    jsc, _jmag, src = _jwst_sources(o, sw)
    metrics["source"] = src

    # inter-module offset from the per-detector daophot cats (module-split), NOT by re-detecting
    # on per-module mosaics.  Use the xcorr histogram PEAK (recovers offsets up to 1.5"), not
    # direct_intermodule's 0.1" nearest-match: a badly-tied field (>~100 mas) has no real pairs
    # inside 0.1", so its median -> ~0 and the gate would FALSELY pass the exact case it exists
    # to catch.  Omitted when the per-detector cats for both modules aren't present.
    a_sc, b_sc, _minfo = _module_positions(o, sw)
    im = None
    if a_sc is not None and b_sc is not None and len(a_sc) >= 50 and len(b_sc) >= 50:
        xc = aa.xcorr(a_sc, b_sc, maxsep=1.5 * u.arcsec)
        if xc and xc.get("peak_ratio", 0) >= aa.MIN_PEAK_RATIO and xc.get("npairs", 0) >= 100:
            im = dict(dra=xc["dra"], ddec=xc["ddec"], off=xc["off"])
    if im:
        metrics.update(intermodule_off=float(im["off"]), intermodule_filt=sw)

    # Measure the JWST-VIRAC pairs BEFORE building the figure so an empty result becomes a
    # red flag, not an empty scatter that reads as "nothing wrong".
    bulk = None; dra = dde = None; nmatch = 0
    if jsc is not None and ref_sc is not None:
        bulk = aa.xcorr(jsc, ref_sc)
        # VIRAC-anchored nearest match (release catalog is deeper than VIRAC -> all-pairs would
        # add wrong-star crowd noise to the offset cloud).
        idx, sep, _ = ref_sc.match_to_catalog_sky(jsc)
        keep = sep < 0.3 * u.arcsec
        nmatch = int(keep.sum())
        if nmatch >= 30:
            rk = ref_sc[keep]; jk = jsc[idx[keep]]
            dra = (rk.ra - jk.ra).to(u.arcsec).value * np.cos(np.radians(jk.dec.value)) * 1000
            dde = (rk.dec - jk.dec).to(u.arcsec).value * 1000

    if dra is None:
        # nothing to plot -> RED FLAG (not an empty panel).  https://github.com/JWST-GC/data-qa/issues/27#issuecomment-5055329892
        reason = ("no release or MAST JWST catalog for this filter yet" if jsc is None else
                  "no VIRAC reference for this field/epoch" if ref_sc is None else
                  f"only {nmatch} JWST-VIRAC matches (<30) — frame likely far off-tie")
        png = _red_flag_figure(o, "stage4", "JWST↔VIRAC OFFSET UNMEASURABLE",
                               f"The positional-offset plot is empty: {reason}.")
        metrics.update(red_flag=True, red_flag_reason=reason, n_matched=int(nmatch), passed=False)
        return png, metrics

    # Per-star offset SIGNIFICANCE (measured offset / its uncertainty).  The mosaic detection
    # used above carries no per-star error, so re-match using the per-exposure DAOPHOT cats,
    # which do: sig = (JWST-VIRAC offset) / (per-star position sigma).  |sig|~1 means the
    # residual is consistent with measurement noise; |sig|>>1 means a real, resolved offset.
    sig_panel = None
    pooled = _pooled_daophot(o, sw)
    vwe = _virac_with_errors(o, ep) if ep else None
    if pooled is not None and vwe is not None and jsc is not None:
        psc, sig_ra, sig_de, pmag, _pf = pooled
        vsc, vsr, vsd = vwe
        # OFFSET from the MERGED catalog (one position per star -> unbiased), same as the main
        # panel.  Taking the nearest of a star's many per-exposure pool detections would pick the
        # noise realization closest to VIRAC and bias the offset toward zero.  The per-star
        # position SIGMA (only the per-exposure cats carry it) is then borrowed from the nearest
        # daophot detection to that catalog star.  Uncertainty COMBINES both catalogs:
        # hypot(JWST per-exposure sigma, VIRAC position+PM error).
        jidx, jsep, _ = vsc.match_to_catalog_sky(jsc)          # VIRAC -> merged catalog
        keep = jsep < 0.15 * u.arcsec
        if keep.sum() >= 30:
            vk = vsc[keep]; jk = jsc[jidx[keep]]
            oda = (vk.ra - jk.ra).to(u.arcsec).value * np.cos(np.radians(jk.dec.deg)) * 1000
            odd = (vk.dec - jk.dec).to(u.arcsec).value * 1000
            didx, dsep, _ = jk.match_to_catalog_sky(psc)        # catalog star -> its daophot sigma
            near = dsep.arcsec < 0.05
            sjr = np.where(near, sig_ra[didx], np.nan)
            sjd = np.where(near, sig_de[didx], np.nan)
            stot_ra = np.hypot(sjr, vsr[keep])
            stot_de = np.hypot(sjd, vsd[keep])
            zra = oda / stot_ra; zde = odd / stot_de
            gz = np.isfinite(zra) & np.isfinite(zde)
            if gz.sum() >= 30:
                # the OFFSET is the bulk tie = hypot of the per-axis MEDIANS.  (median(|d|) is a
                # scatter statistic bounded below by the per-star spread -- it never approaches
                # zero even for a perfect tie, so it must NOT be labelled "offset".)
                off = float(np.hypot(np.median(oda[gz]), np.median(odd[gz])))
                scat = float(np.hypot(aa.mad_std(oda[gz]), aa.mad_std(odd[gz])))
                sig_panel = dict(zra=zra[gz], zde=zde[gz], off_med=off, scatter_mas=scat,
                                 sig_med=float(np.median(np.hypot(zra[gz], zde[gz]))),
                                 n=int(gz.sum()))
                metrics.update(offset_med_mas=off, offset_scatter_mas=scat,
                               offset_signif_med=sig_panel["sig_med"], n_signif=sig_panel["n"])

    ncols = 1 + (1 if sig_panel else 0) + (1 if im else 0)
    fig, ax = _fig(1, ncols, 5.4, 5.0)
    col = 0
    a0 = ax[0][col]; col += 1
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
                   n_matched=int(nmatch))
    if sig_panel:
        asig = ax[0][col]; col += 1
        hs = asig.hexbin(sig_panel["zra"], sig_panel["zde"], gridsize=60, bins="log",
                         cmap="magma", mincnt=1)
        fig.colorbar(hs, ax=asig, label="log N pairs", shrink=0.85)
        for r, c in [(1, "#33cc66"), (3, "#ffcc00"), (5, "#ff5555")]:
            asig.add_patch(plt_circle(r, c))
        asig.axhline(0, color="w", lw=0.4); asig.axvline(0, color="w", lw=0.4)
        asig.set_aspect("equal")
        slim = max(6.0, 1.3 * np.nanpercentile(np.hypot(sig_panel["zra"], sig_panel["zde"]), 98))
        asig.set_xlim(-slim, slim); asig.set_ylim(-slim, slim)
        asig.set_xlabel(r"dRA / $\sigma_{RA}$"); asig.set_ylabel(r"dDec / $\sigma_{Dec}$")
        asig.set_title(f"offset significance  median={sig_panel['sig_med']:.1f}$\\sigma$\n"
                       f"(tie {sig_panel['off_med']:.0f} mas, scatter {sig_panel['scatter_mas']:.0f} "
                       f"mas, n={sig_panel['n']}; $\\sigma$ from per-exposure cats)", fontsize=8)
    if im:
        a1 = ax[0][col]; col += 1
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
        cats = _daophot_glob(o, filt, d)          # obs-scoped
        if not cats:
            continue
        try:
            T = vstack([Table.read(c) for c in cats], metadata_conflicts='silent')
        except (OSError, ValueError):
            continue
        if "skycoord_centroid" not in T.colnames:
            continue
        sc = T["skycoord_centroid"]
        sc = sc[np.isfinite(sc.ra.deg) & np.isfinite(sc.dec.deg)]      # NaN centroids crash the match
        if not len(sc):
            continue
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
    dirs = [f"{BASE}/{o.field}/{filt}/pipeline", f"{BASE}/{o.field}/images-merged"]
    def pick(tag):
        for d in dirs:
            hits = [p for p in glob.glob(f"{d}/{o.obsid}_t001_nircam_clear-{filt.lower()}-{tag}_i2d.fits")
                    if not any(s in p.lower() for s in ("residual", "model", "resbgsub", "bg_i2d"))]
            if hits:
                return hits[0]
        return None
    return pick("merged") or pick("nrcb") or pick("nrca") or _mosaic_path(o, filt)


def _finite_sc(sc):
    """SkyCoord subset with finite RA/Dec.  NaN centroids crash astropy's KDTree matchers
    (xcorr / search_around_sky / match_to_catalog_sky reject ANY NaN), so every per-detector
    daophot position list must pass through this before a match."""
    return sc[np.isfinite(sc.ra.deg) & np.isfinite(sc.dec.deg)]


def _module_positions(o, filt):
    """(NRCA, NRCB) finite SkyCoords for the A/B tie, pooled from the per-detector daophot cats,
    plus per-module ``meta``.  The per-detector cats are the PRIMARY source.  A module is None
    either because it is genuinely ABSENT (single-module obs, e.g. sickle = NRCB only) or because
    its centroids are unusable -- the caller MUST distinguish these (an all-NaN module is an
    astrometry FAILURE to red-flag, not a legitimate single-module pass).

    Returns ``(a_sc, b_sc, meta)`` where meta[module] = dict(present, n_raw, n_nan, nan_frac,
    dead): ``present`` = cats exist on disk; ``dead`` = cats exist but too few finite centroids
    to tie (astrometry failed); ``nan_frac`` = dropped fraction (flagged when high even if usable)."""
    from astropy.table import vstack, Table

    def pool(dets):
        cats = []
        for d in dets:
            cats += _daophot_glob(o, filt, d)          # obs-scoped
        if not cats:
            return None, dict(present=False, n_raw=0, n_nan=0, nan_frac=0.0, dead=False)
        try:
            T = vstack([Table.read(c) for c in cats], metadata_conflicts="silent")
        except (OSError, ValueError):
            return None, dict(present=True, n_raw=0, n_nan=0, nan_frac=0.0, dead=True)
        if "skycoord_centroid" not in T.colnames:
            return None, dict(present=True, n_raw=0, n_nan=0, nan_frac=0.0, dead=True)
        sc = T["skycoord_centroid"]
        scf = _finite_sc(sc)
        n_raw, n_kept = len(sc), len(scf)
        n_nan = n_raw - n_kept
        info = dict(present=True, n_raw=n_raw, n_nan=n_nan,
                    nan_frac=(n_nan / n_raw if n_raw else 0.0), dead=(n_kept < 50))
        return (scf if n_kept else None), info

    a_sc, a_info = pool(["nrca1", "nrca2", "nrca3", "nrca4"])
    b_sc, b_info = pool(["nrcb1", "nrcb2", "nrcb3", "nrcb4"])
    return a_sc, b_sc, dict(a=a_info, b=b_info)


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
    a_sc, b_sc, minfo = _module_positions(o, filt)
    metrics["nan_frac"] = round(max(minfo["a"]["nan_frac"], minfo["b"]["nan_frac"]), 4)
    # A module that is None because its cats exist on disk but have too few finite centroids is
    # an astrometry FAILURE (dead) -- NOT a legitimate single-module obs.  Surface it loudly
    # instead of letting it masquerade as a single-module pass.
    dead_module = next((mod for mod, sc, k in (("NRCA", a_sc, "a"), ("NRCB", b_sc, "b"))
                        if sc is None and minfo[k]["present"] and minfo[k]["dead"]), None)
    if dead_module:
        k = "a" if dead_module == "NRCA" else "b"
        png = _red_flag_figure(o, "stage5", f"{dead_module} ASTROMETRY FAILED",
                               f"{dead_module} has per-exposure catalogs but "
                               f"{minfo[k]['nan_frac'] * 100:.0f}% NaN centroids (<50 usable) — the "
                               f"A/B tie cannot be measured. This is an astrometry failure, not a "
                               f"single-module observation.")
        metrics.update(red_flag=True, red_flag_reason=f"{dead_module} centroids unusable (NaN)",
                       dead_module=dead_module, passed=False)
        return png, metrics
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

    # ---- figure layout depends on whether there is an A/B overlap to show.  With NO overlap
    # (single module, or not measurable) the A/B half + cutout gallery are OMITTED entirely --
    # an empty panel is noise -- and the figure is just the per-detector quiver.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def _draw_quiver(axq):
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
        fig = plt.figure(figsize=(11, 8.5))
        gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 0.85])
        axq, axo = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
        _draw_quiver(axq)
        # dra/dde are the same-star residuals after aligning A onto B by the histogram peak;
        # they scatter about 0 (RMS = tie precision). The bulk offset is the title number.
        axo.hexbin(dra, dde, gridsize=40, bins="log", cmap="cividis", mincnt=1)
        axo.axhline(0, color="w", lw=0.5); axo.axvline(0, color="w", lw=0.5)
        axo.set_xlabel("NRCA-NRCB residual dRA [mas]"); axo.set_ylabel("residual dDec [mas]")
        lim = max(50, 4 * ov["rms"])
        axo.set_xlim(-lim, lim); axo.set_ylim(-lim, lim)
        axo.set_title(f"A-vs-B overlap ({ov['n']} matched stars)\n"
                      f"offset={ov['off']:.1f} mas  RMS={ov['rms']:.1f} mas", fontsize=9)

        # (3) doubled-star cutout gallery from the merged mosaic at overlap-star positions
        ncut = 6
        if mpath and os.path.exists(mpath):
            with fits.open(mpath) as hdul:
                sci = hdul["SCI"] if "SCI" in hdul else hdul[1]
                data = sci.data.astype("float32"); w = WCS(sci.header)
            from astropy.coordinates import SkyCoord
            from astropy.visualization import ZScaleInterval, ImageNormalize, AsinhStretch
            # De-duplicate: search_around_sky returns MANY pairs per bright overlap star, so the
            # raw list repeats the same few stars (e.g. one star shown 4x).  Greedily keep only
            # spatially DISTINCT stars (>0.5") so the gallery is 6 DIFFERENT stars.
            picks = []
            for ra, dec in ov["pos"][:2000]:
                if picks:
                    prev = SkyCoord([p[0] for p in picks] * u.deg, [p[1] for p in picks] * u.deg)
                    if SkyCoord(ra * u.deg, dec * u.deg).separation(prev).arcsec.min() < 0.5:
                        continue
                picks.append((ra, dec))
                if len(picks) >= ncut * 3:      # gather extras; some cutouts fail the finite check
                    break
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
            from astropy.wcs.utils import proj_plane_pixel_scales
            pscale = float(np.mean(proj_plane_pixel_scales(w))) * 3600.0     # arcsec/pix from WCS
            fig.text(0.5, 0.02,
                     f"{shown} star{'s' if shown != 1 else ''} from the NRCA∩NRCB overlap of the "
                     f"{filt} merged mosaic (each detected in BOTH modules; 25 px ≈ "
                     f"{25 * pscale:.1f}\").  A good A↔B tie = one round PSF; a mis-tie doubles or "
                     f"elongates the star.",
                     ha="center", fontsize=8)
        title_extra = ""
        suptitle_y = 0.98
    else:
        # no A/B overlap -> quiver only, at a compact size (no empty right half / cutout row).
        # Reserve top room so the suptitle clears the axes title + quiverkey (all top-anchored).
        fig = plt.figure(figsize=(6.4, 5.9))
        _draw_quiver(fig.add_subplot(1, 1, 1))
        fig.subplots_adjust(top=0.80)
        suptitle_y = 0.995
        title_extra = (f"  ·  single module ({single_module})" if single_module
                       else "  ·  A/B overlap not measurable")

    # single-module obs (sickle = NRCB only) has no A/B tie to fail -> N/A passes.
    if single_module:
        metrics["single_module"] = single_module
    nan_frac = metrics.get("nan_frac", 0.0)
    high_nan = nan_frac > _NAN_FRAC_FLAG        # usable but degraded -> surface + don't pass
    if high_nan:
        title_extra += f"  ·  ⚠ {nan_frac * 100:.0f}% NaN centroids"
    metrics["passed"] = bool((single_module or (ov and ov["off"] < aa.THRESH["intermodule"]))
                             and not high_nan)
    fig.suptitle(f"{o.target} {o.obsid} — inter-detector / inter-module tie ({filt}){title_extra}",
                 fontsize=11, y=suptitle_y)
    return _save(fig, f"{o.obsid}_stage5.png"), metrics


STAGES = {1: stage1_mosaics, 2: stage2_cmd, 3: stage3_calibration, 4: stage4_offsets,
          5: stage5_intermodule}


def _build_stage5(o, sw, lw):
    return stage5_intermodule(o, sw)


# --------------------------------------------------------------------------- STAGE 6
def stage6_astrom_error(o: Observation, sw, lw):
    """Astrometric precision curve: per-star position sigma (mas) vs Vega magnitude,
    from the per-exposure PSF fits.  The bright-end floor is the astrometric systematic limit;
    the faint-end rise tracks S/N.  One curve per available channel (SW / LW)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    metrics = dict(stage=6, sw=sw, lw=lw)
    fig, ax = _fig(1, 1, 6.8, 5.2)
    a = ax[0][0]
    any_data = False
    all_vega = True
    for filt, color in [(sw, "#3366cc"), (lw, "#cc3311")]:
        if not filt:
            continue
        pooled = _pooled_daophot(o, filt)
        if pooled is None:
            continue
        _sc, sig_ra, sig_de, mag, _flux = pooled
        # Vega-calibrate the instrumental mag (vega = instr + ZP) via the merged catalog; fall
        # back to instrumental if there is no Vega catalog for this filter.
        zp = _vega_zeropoint(o, filt, _sc, mag)
        if zp is not None:
            mag = mag + zp
            metrics[f"vega_zp_{filt.lower()}"] = float(zp)
        else:
            all_vega = False
        sig = np.hypot(sig_ra, sig_de) / np.sqrt(2.0)     # per-axis-equivalent astrometric error
        # Drop failed fits: a formal sigma of hundreds-to-thousands of mas is a diverged PSF
        # fit on a noise peak, not astrometry -- keeping it compresses the informative regime.
        ok = sig < 500.0
        med, lo, hi, ctr = _binned_stat(mag[ok], sig[ok])
        if med is None:
            continue
        any_data = True
        lbl = f"{filt}  (n={int(ok.sum())}" + ("" if zp is not None else ", instr") + ")"
        a.plot(ctr, med, "-", color=color, lw=1.7, label=lbl)
        a.fill_between(ctr, lo, hi, color=color, alpha=0.20)
        metrics[f"floor_mas_{filt.lower()}"] = float(np.nanmin(med))
        metrics[f"nstars_{filt.lower()}"] = int(ok.sum())
    metrics["mag_kind"] = "vega" if all_vega else "mixed"
    if not any_data:
        plt.close(fig)          # close the empty curve fig before the red-flag builds its own
        reason = "no per-exposure DAOPHOT catalogs on disk for this obs/filter"
        png = _red_flag_figure(o, "stage6", "ASTROMETRIC-ERROR CURVE UNAVAILABLE",
                               f"Cannot build the precision-vs-magnitude curve: {reason}.")
        metrics.update(red_flag=True, red_flag_reason=reason, passed=False)
        return png, metrics
    a.set_yscale("log")
    a.set_ylim(0.03, 100.0)          # 0.03-100 mas: floor through the S/N rise; junk lives above
    xlbl = ("Vega magnitude" if all_vega else
            "magnitude  (Vega where calibrated, else instrumental)")
    a.set_xlabel(xlbl)
    a.set_ylabel(r"astrometric error  $\sigma_{\rm pos}$ (mas)")
    a.legend(fontsize=9, loc="upper left")
    a.grid(alpha=0.25, which="both")
    a.set_title(f"{o.target} {o.obsid} — astrometric precision vs magnitude\n"
                r"($\sigma$ from per-exposure daophot cats, not the release catalog)", fontsize=9)
    metrics["passed"] = True
    return _save(fig, f"{o.obsid}_stage6.png"), metrics


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
    if n == 6:
        return stage6_astrom_error(o, sw, lw)
    raise ValueError(n)


CAPTIONS = {
    1: "**Stage 1 — first mosaics.** Grayscale {sw} (SW) and {lw} (LW) `i2d`. Confirms the "
       "observation was delivered and the mosaics are present and not obviously corrupt.",
    2: "**Stage 2 — colour-magnitude diagram** from the `{kind}` catalog ({n_stars} stars). "
       "LF-inset turnover ≈ {lf_turnover:.1f} tracks depth; regenerated as the catalog deepens.",
    3: "**Stage 3 — photometric calibration.** JWST {sw} vs VIRAC Ks for {n_matched} matched "
       "stars: slope {slope:.2f}, zp {zeropoint:.2f}, scatter {scatter:.2f} mag. A tight locus "
       "means the right stars were matched.",
    4: "**Stage 4 — positional offsets.** JWST−VIRAC ΔRA/ΔDec (bulk {bulk_off:.0f} mas), the "
       "per-star offset significance (median {offset_signif_med:.1f}σ = measured offset ÷ its "
       "uncertainty), and the reference-free inter-module offset. Frame-match / PM precursor.",
    5: "**Stage 5 — inter-detector / inter-module tie.** Per-detector residual quiver "
       "(bulk-removed; A–B diff {intermodule_diff:.1f} mas), the reference-free NRCA–NRCB overlap "
       "(offset {intermodule_off:.1f} mas, RMS {intermodule_rms:.1f} mas over {n_overlap} shared "
       "stars), and a cutout gallery of 6 stars in the NRCA∩NRCB overlap — each cut from the SW "
       "merged `i2d` mosaic and detected in BOTH modules. A good tie shows one round PSF; a "
       "mis-tie doubles/elongates the star (the same source drizzled twice at offset positions).",
    6: "**Stage 6 — astrometric precision.** Per-star position error σ_pos (mas) vs Vega "
       "magnitude from the per-exposure PSF fits (instrumental mag Vega-calibrated against the "
       "merged catalog), one curve per channel. The bright-end floor is the astrometric "
       "systematic limit; the faint-end rise tracks S/N. Shaded band = 16–84th percentile.",
}


def caption_for(n, metrics):
    if metrics.get("red_flag"):
        return (f"🚩 **Stage {n} — RED FLAG.** The plot is empty: "
                f"{metrics.get('red_flag_reason', 'no data to show')}. "
                f"An empty result here means the measurement could not be made — investigate.")
    if n == 4 and metrics.get("offset_signif_med") is None:
        # significance panel omitted (no per-exposure cats / VIRAC errors) -> don't assert a
        # third panel exists or print "nanσ".
        return ("**Stage 4 — positional offsets.** JWST−VIRAC ΔRA/ΔDec (bulk "
                f"{metrics.get('bulk_off') or float('nan'):.0f} mas) and the reference-free "
                "inter-module offset (the per-star significance panel is omitted — no "
                "per-exposure catalogs or VIRAC per-star errors for this field). "
                "Frame-match / PM precursor.")
    if n == 5 and metrics.get("intermodule_off") is None:
        # No reference-free NRCA-NRCB overlap measurement -> the full template's overlap clause
        # ({intermodule_off}/{intermodule_rms}/{n_overlap}) would KeyError and drop the whole
        # caption to a bare fragment.  Say WHICH case this is instead (>half the fields hit it):
        # a legitimate single-module obs, or two modules with no shared stars to tie them.
        diff = metrics.get("intermodule_diff")
        diff_clause = (f" Per-detector residual quiver shows an A–B diff of {diff:.1f} mas."
                       if diff is not None else "")
        if metrics.get("single_module"):
            return (f"**Stage 5 — inter-detector tie.** Single module "
                    f"({metrics['single_module']}) for this observation, so there is no "
                    f"NRCA–NRCB tie to check — the reference-free overlap panel is omitted."
                    f"{diff_clause}")
        return ("**Stage 5 — inter-detector / inter-module tie.** The reference-free NRCA–NRCB "
                "overlap could not be measured (no shared stars in the NRCA∩NRCB dither overlap "
                "after alignment), so that panel and the cutout gallery are omitted."
                f"{diff_clause} The inter-module tie is unverified for this observation.")
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
        stem = f"jw{int(program):05d}-o{obs}_t001_nircam_clear-*-merged_i2d.fits"
        hits = (glob.glob(f"{base}/{fld}/*/pipeline/{stem}")
                + glob.glob(f"{base}/{fld}/images-merged/{stem}"))  # not-yet-released layout
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


def _json_default(o):
    """json.dump default: coerce numpy scalars/arrays (a stage metric like ``passed`` or
    ``red_flag`` can be a ``np.bool_`` from an ndarray comparison, which stdlib json cannot
    serialize -> the whole metrics write crashes AFTER every stage already posted)."""
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--program", required=True)
    ap.add_argument("--obs", required=True)
    ap.add_argument("--stage", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6])
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
        # NB: AttributeError is deliberately NOT caught -- it almost always means a typo in a
        # stage, not a data problem (the real None-attr data cases are guarded at the source).
        except (OSError, ValueError, IndexError, KeyError, RuntimeError) as e:
            print(f"  stage {n}: FAILED to build: {type(e).__name__}: {e}", file=sys.stderr)
            all_metrics[f"stage{n}"] = dict(stage=n, error=f"{type(e).__name__}: {e}", passed=False)
            with open(mpath, "w") as fh:
                json.dump(all_metrics, fh, indent=2, default=_json_default)
            continue
        all_metrics[f"stage{n}"] = metrics
        print(f"  stage {n}: {png}  passed={metrics.get('passed')}")
        with open(mpath, "w") as fh:          # persist before the (fallible) network post
            json.dump(all_metrics, fh, indent=2, default=_json_default)
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
