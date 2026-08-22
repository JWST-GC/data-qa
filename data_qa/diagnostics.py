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
    Stage 4  offsets         JWST-VIRAC dRA/dDec per spatial cell + NRCA-NRCB offset
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
import contextlib
import glob
import json
import os
import re
import sys
from dataclasses import replace

import numpy as np

from . import astrometry_audit as aa
from .observations import Observation, registry

# The sanctioned, validated GC offset estimator: offset-histogram stacking with window sweep +
# contrast gate + window-edge-alias rejection (memory: dataqa-astrometry-offset-method).  A light,
# import-clean module of jwst_gc_pipeline (no jwst/crds/stpipe pulled -- verified).  Reused rather
# than re-implemented so data-qa measures offsets the same way the pipeline does.  Graceful None if
# the package is absent so the rest of data-qa still imports; callers fall back to aa.xcorr.
try:
    from jwst_gc_pipeline.photometry.astrometry_offsets import measure_offset as _pipe_measure_offset
except ImportError:
    _pipe_measure_offset = None

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


# --------------------------------------------------------------------------- input provenance
# Every stage records the FULL PATH of each file it reads, so the posted comment and the metrics
# JSON both say exactly which data produced the numbers.  A stage that quietly read a stale
# generation, a different observation's catalog, or the wrong module used to be indistinguishable
# from one that read the right thing.
#
# ``_used`` is called at the point of the READ, not where a path is resolved.  Those differ: the
# catalog pickers header-peek every candidate before choosing one, ``_mosaic_path`` is called for
# filters whose panel is then skipped, and several resolvers return a path that a later existence
# check rejects.  Recording at resolve time would list files the analysis never opened, which is
# the failure this exists to prevent.
_INPUTS = []


@contextlib.contextmanager
def _recording_inputs():
    """Collect the files read inside the block, as a list of ``(role, absolute path)`` in read
    order.  Restores the previous collector on exit, so a nested build cannot lose the outer
    stage's record."""
    global _INPUTS
    prev = _INPUTS
    _INPUTS = []
    try:
        yield _INPUTS
    finally:
        _INPUTS = prev


def _used(path, role):
    """Record ``path`` as a file this stage read, and return it unchanged so it can wrap the read:

        with fits.open(_used(p, "SW mosaic")) as hdul:

    Absolutised, de-duplicated, and skipped when the path is falsy or absent from disk.  Recording
    a file that was never opened would be a false provenance claim, which is worse than none."""
    if not path:
        return path
    ap = os.path.abspath(path)
    if os.path.exists(ap) and (role, ap) not in _INPUTS:
        _INPUTS.append((role, ap))
    return path


def _used_many(paths, role):
    """``_used`` over an iterable, returning the list unchanged -- for the ``vstack`` reads that
    pool hundreds of per-exposure catalogs."""
    paths = list(paths)
    for p in paths:
        _used(p, role)
    return paths


def _channel(filt):
    return "SW" if int(filt[1:4]) <= 212 else "LW"


def _has_lw(o: "Observation"):
    """Does this obs actually carry an LW-channel filter?  Distinguishes a genuine
    single-channel obs (legitimately no colour) from a PREF-list gap (an LW mosaic exists but
    wasn't recognised) -- the two must NOT be conflated, or the latter silently PASSes with
    its LW data unexamined."""
    return any(_channel(f) == "LW" for f in getattr(o, "filters", []) if f)


def pick_filters(available, sw=None, lw=None, prefer=None):
    """Choose one SW + one LW representative filter from those available for the obs.

    ``prefer`` (optional) is the subset of filters that have a REDUCED science mosaic on disk.
    Most stages (1 mosaics, 4 frame registration, 5 inter-module, 7 MAST-vs-pipeline, 9
    PSF-vs-aper) key off the representative filter's mosaic, so a filter that has only a catalog /
    raw MAST product but no reduced mosaic must not win the pick just because it ranks higher in the
    preference order.  cloudef jw02092-o005 delivered F162M+F360M reduced but only raw MAST i2d for
    F210M/F480M; F210M/F480M outrank F162M/F360M in the preference lists, so the old pick chose the
    UNREDUCED pair and every mosaic-keyed stage blanked with 'no i2d' (issue #38).  Preferring a
    mosaic-backed filter per channel fixes stages 1/4/5/7/9; it falls back to any available filter
    when the channel has no reduced mosaic at all, preserving the prior behaviour."""
    up = {f.upper() for f in available}
    pref = {f.upper() for f in (prefer or ())} & up      # mosaic-backed AND actually available

    def choose(explicit, order):
        if explicit:
            return explicit.upper()
        # first preference-ranked filter with a reduced mosaic; else first available (unchanged)
        return (next((f for f in order if f in pref), None)
                or next((f for f in order if f in up), None))
    return choose(sw, _SW_PREF), choose(lw, _LW_PREF)


def _filters_with_mosaic(o: Observation):
    """Subset of the obs's filters that have a reduced science mosaic on disk (``_mosaic_path``).
    Feeds ``pick_filters(prefer=...)`` so the representative SW/LW filters are ones the
    mosaic-keyed stages can actually render."""
    return [f for f in o.filters if _mosaic_path(o, f)]


def _available_filters(o: Observation):
    """The obs's filters that actually have a product on disk (mosaic, or a catalog/DAO catalog
    with usable positions) -- as opposed to the program's nominal filter list from the portal.
    Restricting pick_filters to these avoids choosing a filter (e.g. F444W) that was proposed but
    whose data are not present, which would spuriously red-flag the CMD/calibration (issue #39).

    A single unreadable catalog header must not sink the whole obs's QA, so a per-filter FITS error
    is treated as 'no usable catalog for that filter' rather than propagating."""
    out = []
    for f in o.filters:
        try:
            has = bool(_mosaic_path(o, f) or _catalog_with_vega(o, f)[0] or _dao_position_catalog(o, f))
        except (OSError, ValueError):
            has = bool(_mosaic_path(o, f))     # header unreadable -> fall back to mosaic presence
        if has:
            out.append(f)
    return out


# --------------------------------------------------------------------------- product lookup
def _mosaic_path(o: Observation, filt):
    """Released science mosaic for this obs+filter, or None.  Prefers the all-detector 'merged'
    drizzle.  A genuinely single-module observation (e.g. sickle jw03958-o007, NRCB-only) names its
    mosaic '-nrcb', never '-merged', so a merged-only glob would wrongly report 'no mosaic' and
    blank stage 1 / the stage-7 pipeline panel (issue #13).

    But a single-module mosaic is accepted ONLY when the OTHER module has no mosaic for this
    obs+filter: a two-module observation that simply has not been merged yet (e.g. cloudc o002
    F212N, which has both -nrca and -nrcb over DIFFERENT sky) must NOT silently return one half as
    'the mosaic' -- that flips 'delivered' green and hides that NRCA/the merge is missing.  In that
    case return None so the deliverable reads incomplete (#13 review)."""
    if not filt:                     # obs with no filter for this channel (e.g. a single-band obs)
        return None
    dir_pats = [
        f"{BASE}/{o.field}/{filt}/pipeline",
        f"{BASE}/{o.field}/*/pipeline",
        f"{BASE}/{o.field}/images-merged",   # not-yet-released fields (e.g. gc2211) land mosaics here
    ]

    def find(tag):
        stem = f"{o.obsid}_t001_nircam_clear-{filt.lower()}-{tag}_i2d.fits"
        for d in dir_pats:
            hits = sorted(glob.glob(f"{d}/{stem}"))
            if hits:
                return hits[-1]
        return None

    merged = find("merged")
    if merged:
        return merged
    nrcb, nrca = find("nrcb"), find("nrca")
    present = [p for p in (nrcb, nrca) if p]
    if len(present) == 1:                     # only one module has a mosaic for THIS filter
        # OBSERVATION-level completeness: a lone nrcX is a genuine single-module deliverable only if
        # the OBSERVATION is single-module.  If a SIBLING filter of the same obsid has a merged
        # mosaic or both modules, the obs has two modules and this filter is a half-delivered filter
        # (e.g. cloudef jw02092-o002 F360M has only NRCA while F162M/F210M/F480M have both + merged)
        # -- return None so it reads incomplete rather than flipping 'delivered' green (#13 review).
        if _obs_is_two_module(o, dir_pats):
            return None
        return present[0]
    return None                               # both modules but no merged -> incomplete, not half


def _obs_is_two_module(o: Observation, dir_pats):
    """True if ANY filter of this obsid has a merged mosaic or BOTH NRCA and NRCB mosaics on disk --
    evidence the observation is two-module, so a lone single-module mosaic for some other filter is
    an incomplete half, not a single-module deliverable."""
    a_filts, b_filts = set(), set()
    for d in dir_pats:
        for p in glob.glob(f"{d}/{o.obsid}_t001_nircam_clear-*_i2d.fits"):
            m = re.search(r"clear-(f\d{3}[wnm])-(merged|nrca|nrcb)_i2d\.fits$", os.path.basename(p).lower())
            if not m:
                continue
            filt, tag = m.group(1), m.group(2)
            if tag == "merged":
                return True
            (a_filts if tag == "nrca" else b_filts).add(filt)
    return bool(a_filts & b_filts)            # some filter has BOTH modules


def _mosaic_module(path):
    """'NRCA' / 'NRCB' if ``path`` is a single-module mosaic, else '' (merged / all-detector).  Used
    to tag the stage-1 title + metrics so a single-module panel is never mistaken for the full obs."""
    low = os.path.basename(path or "").lower()
    if "-nrca_i2d" in low:
        return "NRCA"
    if "-nrcb_i2d" in low:
        return "NRCB"
    return ""


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
    m = Table.read(_used(cat, f"Vega zeropoint catalogue ({filt})"))
    if sccol not in m.colnames or magcol not in m.colnames:
        return None
    vg = np.asarray(m[magcol], float)
    # DROP catalogue rows with a NaN sky position before matching: match_to_catalog_sky builds a
    # kd-tree of the catalogue and raises "Catalog coordinates cannot contain NaN entries" on any
    # NaN (quintuplet o003, sickle o007 carry NaN skycoord_<filt> rows).  Filter coords AND the
    # vega mag together so the indices stay aligned.
    csc = m[sccol]
    cfin = np.isfinite(csc.ra.deg) & np.isfinite(csc.dec.deg) & np.isfinite(vg)
    if int(cfin.sum()) < 50:
        return None
    csc, vg = csc[cfin], vg[cfin]
    idx, sep, _ = sc.match_to_catalog_sky(csc)
    good = (sep < 0.05 * u.arcsec) & np.isfinite(instr) & np.isfinite(vg[idx])
    if good.sum() < 50:
        return None
    ii = instr[good]
    zz = vg[idx][good] - ii
    bright = ii <= np.percentile(ii, 40)     # bright end: cleanest instrumental mags
    return float(np.median(zz[bright])) if bright.sum() >= 20 else float(np.median(zz))


def _mast_source_catalog(o: Observation, filt):
    """MAST-delivered L3 source catalog for one filter (single-band), or None.  Two homes, in
    priority order:
      1. next to a locally-reduced i2d: ``<obsid>_t001_nircam_*<filt>*_cat.fits`` under the field's
         ``<filt>/pipeline`` / ``images-merged`` dirs;
      2. the STScI archive delivery under ``mastDownload/JWST/<product-dir>/..._cat.ecsv`` (or
         ``.fits``, or ``MAST_FITS/``).  This is where the pupil-filter deliveries live -- F162M
         ships as ``f150w2-f162m`` -- and it is the ONLY catalog for a field whose merged/DAO
         products are absent (e.g. cloud E/F o002), so it must be searched or stage 4 red-flags an
         obs that IS catalogued.  Any tile token (t001/t002); per-detector / segmentation files
         excluded."""
    fl = filt.lower()
    _bad = ("nrca", "nrcb", "destreak", "segm")
    for d in (f"{BASE}/{o.field}/{filt}/pipeline", f"{BASE}/{o.field}/*/pipeline",
              f"{BASE}/{o.field}/images-merged"):
        hits = [p for p in glob.glob(f"{d}/{o.obsid}_t*_nircam_*{fl}*_cat.fits")
                if not any(t in os.path.basename(p).lower() for t in _bad)]
        if hits:
            return sorted(hits)[-1]
    # MAST archive delivery.  TARGETED patterns (one product-dir deep -- NOT a recursive ``**`` walk
    # of the enormous mastDownload tree, which made stage 4 hang).  Prefer .ecsv (the L3 default).
    for ext in ("ecsv", "fits"):
        for pat in (f"{BASE}/{o.field}/mastDownload/JWST/{o.obsid}_t*_nircam_*{fl}*/"
                    f"{o.obsid}_t*_nircam_*{fl}*_cat.{ext}",
                    f"{BASE}/{o.field}/MAST_FITS/{o.obsid}_t*_nircam_*{fl}*_cat.{ext}"):
            hits = [p for p in glob.glob(pat)
                    if not any(t in os.path.basename(p).lower() for t in _bad)]
            if hits:
                return sorted(hits)[-1]
    # Nothing local: DOWNLOAD it rather than let the caller red-flag a catalogued obs (guarded;
    # no-ops for every current field, whose catalogues are already on disk above).
    return _download_mast_l3_catalog(o, filt)


def _jwst_sources(o: Observation, filt, position_valid=False):
    """JWST source positions + magnitude for one filter, READ FROM THE CATALOG -- never
    re-detected.  This is a QA of releaseable products: show what the catalog contains, don't
    bake our own detection.  Priority: the RELEASE merged catalog (mag_vega_<filt> +
    skycoord_<filt>); else the MAST-delivered per-i2d source catalog (sky_centroid +
    aper abmag).  Returns (SkyCoord, mag, source_label) or (None, None, None) -> caller red-flags.
    The only thing baked downstream is the crossmatch (to VIRAC / across filters).

    ``position_valid`` additionally drops rows whose ``skycoord_<filt>`` is not that row's own
    measured position (see ``_position_valid``).  Callers that need POSITIONS pass True; the
    photometric stages leave it False, because a row's flux is its own even when the cross-filter
    position match was loose."""
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    from astropy.table import Table
    # 1) release merged catalog
    cat, magcol, sccol = _catalog_with_vega(o, filt)
    if cat:
        m = Table.read(_used(cat, f"JWST catalogue, release merged ({filt})"))
        if sccol in m.colnames and magcol in m.colnames:
            sc = m[sccol]
            mag = np.asarray(m[magcol], float)
            g = np.isfinite(sc.ra.deg) & np.isfinite(sc.dec.deg) & np.isfinite(mag)
            note = None
            if position_valid:
                g, note = _position_valid(m, filt, g)
            if g.sum() >= 30:
                lbl = f"release:{os.path.basename(cat)}"
                return sc[g], mag[g], (f"{lbl} [{note}]" if note else lbl)
    # 2) MAST-delivered per-i2d source catalog
    mp = _mast_source_catalog(o, filt)
    if mp:
        m = Table.read(_used(mp, f"JWST catalogue, MAST per-i2d ({filt})"))
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


def _dao_position_catalog(o: Observation, filt):
    """Per-filter RELEASE DAO catalog (positions only, no calibrated magnitude), obs-scoped, as a
    LAST-RESORT position source for the stage-4 offset.  A field that has been DETECTED but not
    yet merged/calibrated (gc2211 o046: per-filter ``f200w_..._dao_basic_o046_vetted.fits`` exist,
    but no merged photometry table and no MAST ``_cat.fits``) still has real positions, so the
    offset the reader cares about stays measurable and the obs need not be red-flagged.  Prefers the
    vetted science catalog at the highest pipeline stage; excludes carta/seed helper files."""
    pats = [p for p in glob.glob(f"{BASE}/{o.field}/catalogs/{filt.lower()}_*dao_basic*_o{o.obs}_vetted.fits")
            if "carta" not in p and "seed" not in p]
    if not pats:
        return None

    def _mtime(p):
        try:
            return os.path.getmtime(p)
        except OSError:                       # racing deletion / stale glob entry -> deprioritise
            return 0.0
    return max(pats, key=lambda p: (_catalog_priority(os.path.basename(p))[0], _mtime(p)))


# A merged catalog's ``skycoord_<filt>`` is only that ROW's own position when the cross-filter
# match was tight.  jicama's merge accepts anything inside max_offset=0.10", which at GC density
# (JWST NN spacing ~0.1-0.2") also admits the NEIGHBOUR of an undetected star, so the row carries
# a position ~one neighbour-spacing away.  Measured on brick 2221-o001 F212N: rows with
# sep_f212n in 0.05-0.10" are 43% of the second lobe in the JWST-VIRAC offset cloud and only 2%
# of its core, at the SAME magnitude and the SAME saturated fraction, which makes it a match-quality
# effect.  Cutting at 0.05" (~1.6 SW pix) leaves an offset that is flat with magnitude (-0.6 to
# +0.2 mas from Vega 8 to 18.5) and moves the cloud's median from -11.2 to -0.9 mas.  Astrometry
# therefore uses a TIGHTER radius than the merge's photometric tolerance.
# See JWST-GC/data-qa#1.
POSITION_VALID_SEP_ARCSEC = 0.05


def _position_valid(tbl, filt, finite):
    """Rows of ``tbl`` whose ``skycoord_<filt>`` is that row's own measured position.

    Requires ``sep_<filt>`` within POSITION_VALID_SEP_ARCSEC.  Returns ``finite`` unchanged when
    the catalog has no ``sep_<filt>`` column (a MAST or per-filter DAO catalog, where the position
    is the detection's own by construction), so this never silently empties a valid source."""
    import astropy.units as u
    col = f"sep_{filt.lower()}"
    if col not in tbl.colnames:
        return finite, None
    sep = tbl[col]
    # merge_catalogs writes sep_<filt> as a Quantity/Column in DEGREES.  A bare Column still has
    # a .to() method, so dispatch on the UNIT, not on hasattr, and assume degrees when absent.
    unit = getattr(sep, "unit", None)
    sep = (u.Quantity(np.asarray(sep, float), unit).to(u.arcsec).value if unit is not None
           else np.asarray(sep, float) * 3600.0)
    ok = finite & np.isfinite(sep) & (sep <= POSITION_VALID_SEP_ARCSEC)
    # never turn a measurable field into a red flag: if the cut leaves too little, keep the
    # uncut selection and say so, rather than reporting "offset unmeasurable".
    if ok.sum() < 30:
        return finite, "sep-cut-skipped(too-few)"
    return ok, f"sep<={POSITION_VALID_SEP_ARCSEC}\""


def _jwst_positions(o: Observation, filt):
    """Positions (+source label) for the stage-4 offset ONLY (stage 4 needs no magnitude).
    Prefers a catalog WITH photometry (``_jwst_sources``: release merged -> MAST); falls back to a
    per-filter release DAO catalog (positions only) so a detected-but-not-yet-merged obs still gets
    a real offset measurement.  Applies the positional-validity cut above: stage 4 measures where
    the frame sits, and a row holding a neighbour's position carries no evidence about that.
    Returns (SkyCoord, label) or (None, None)."""
    from astropy.table import Table
    sc, _mag, src = _jwst_sources(o, filt, position_valid=True)
    if sc is not None:
        return sc, src
    dp = _dao_position_catalog(o, filt)
    if dp:
        m = Table.read(_used(dp, f"JWST positions, per-filter DAO ({filt})"))
        if "skycoord" in m.colnames:
            sc = _finite_sc(m["skycoord"])
            if len(sc) >= 30:
                return sc, f"release-dao(positions):{os.path.basename(dp)}"
    return None, None


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
    """VIRAC2-Gaia refcat for the absolute-frame (position-only) check, OBS-SCOPED.  A field can
    carry both an untokened full-field refcat (``..._epoch2023.71.fits``) and per-obs subsets
    (``..._epoch2023.71_o028.fits``) that cover only ONE observation's footprint.  A plain
    ``sorted(...)[-1]`` picks the lexically-last file, which for gc2211-o023 handed back o028's
    refcat -- a DISJOINT patch of sky (measured min separation 281") -> 0 JWST-VIRAC matches -> a
    false "frame grossly offset" red flag (issues #7/#8/#28).  Prefer this obs's own tokened refcat;
    else the untokened full-field refcat; NEVER a different obs's tokened refcat (wrong footprint).

    NOTE (epoch-blindness): when several EPOCHS coexist this still takes the lexically-newest, not
    the one nearest the observation.  Only ngc6334 has multiple epochs today so nothing moves, but
    the rule is epoch-blind and should be revisited if per-epoch refcats proliferate.  Related:
    the untokened gc2211 refcat carries no pmRA/pmDE, so aa.load_reference does no PM propagation
    and _obs_epoch has no effect on the reference here (the ~128 mas offset is flat in dt anyway)."""
    hits = sorted(glob.glob(f"{BASE}/{o.field}/catalogs/gaia_virac2_refcat_epoch*.fits"))
    if not hits:
        return None
    tok = [h for h in hits if (m := _OBS_TOK_RE.search(os.path.basename(h))) and m.group(1) == o.obs]
    if tok:
        return sorted(tok)[-1]
    unt = [h for h in hits if not _OBS_TOK_RE.search(os.path.basename(h))]
    if unt:
        return sorted(unt)[-1]
    return None                                  # only other-obs tokened refcats exist -> refuse


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
    t = Table.read(_used(p, "VIRAC2 refcache (per-star errors)"))
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


def _grayscale(ax, path, title, role="mosaic (displayed)"):
    from astropy.io import fits
    from astropy.visualization import ZScaleInterval, ImageNormalize, AsinhStretch
    from astropy.wcs import WCS
    with fits.open(_used(path, role)) as hdul:
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

    # A panel image is our reduced science mosaic when present, else the MAST-delivered STScI L3
    # i2d.  MAST ALWAYS ships an i2d for a delivered filter, so a filter with data is NEVER blank;
    # only a nominal filter with nothing on disk (dropped) has no image.  (The "passed" gate below
    # still keys on the REDUCED mosaic, so a MAST-only filter reads not-yet-complete, not delivered.)
    def _panel_image(filt):
        p = _mosaic_path(o, filt)
        if p:
            return p, "reduced"
        m = _mast_i2d(o, filt)
        return (m, "mast") if m else (None, None)

    psw, plw = _mosaic_path(o, sw), _mosaic_path(o, lw)      # reduced-only, for the passed gate
    panels = []                                   # (filt, path, source, aspect = ny/nx)
    for filt in (sw, lw):
        if not filt:
            continue                              # single-filter obs: no LW row at all
        p, src = _panel_image(filt)
        asp = 0.45
        if p:
            kind = "reduced mosaic" if src == "reduced" else "MAST i2d"
            with fits.open(_used(p, f"{filt} {kind}")) as h:
                s = h["SCI"] if "SCI" in h else h[1]
                ny, nx = s.data.shape
                asp = ny / nx if nx else 0.45
        panels.append((filt, p, src, asp))

    W = 11.0
    # per-row height from native aspect (clamp so a near-square or a razor-thin strip stays
    # legible); a truly imageless filter (no MAST i2d either) gets a short placeholder row.
    heights = [(0.25 if p is None else max(0.18, min(1.0, asp))) * W for _, p, _s, asp in panels]
    fig = plt.figure(figsize=(W, sum(heights) + 0.35 * len(panels)))
    gs = fig.add_gridspec(len(panels), 1, height_ratios=heights, hspace=0.18)
    fracs = {}
    modules = {}                                   # filt -> 'NRCA'/'NRCB' when a single-module mosaic
    mast_shown = []                                # filters rendered from the MAST i2d (not reduced)
    for i, (filt, p, src, _asp) in enumerate(panels):
        a = fig.add_subplot(gs[i, 0])
        if p and src == "reduced":
            mod = _mosaic_module(p)
            if mod:
                modules[filt] = mod
            title = f"{o.obsid}  {filt}" + (f"  [{mod} only]" if mod else "")
            fracs[filt] = _grayscale(a, p, title, role=f"{filt} reduced mosaic")
        elif p:                                    # MAST-delivered i2d fallback (not yet reduced)
            mast_shown.append(filt)
            fracs[filt] = _grayscale(a, p, f"{o.obsid}  {filt}  [MAST i2d — not yet pipeline-reduced]",
                                     role=f"{filt} MAST i2d")
        else:
            a.text(0.5, 0.5, f"{filt}\n(no i2d on disk)", ha="center", va="center")
            a.set_xticks([]); a.set_yticks([])
    # Record which NOMINAL (portal) filters have NO product on disk: pick_filters selects only
    # from filters that are present, so a filter that WAS observed but is not yet reduced would
    # otherwise vanish from QA with no trace (issue: F444W/F322W2 on Sgr A*).  Leaving the list
    # here keeps that visible.
    avail = _available_filters(o)
    dropped = [f for f in o.filters if f not in avail]
    # Filters with a product on disk (catalog or raw MAST i2d) but NO reduced science mosaic yet:
    # 'awaiting reduction', distinct from 'dropped' (nothing on disk at all).  A filter here reads
    # 'no i2d' in its panel while its raw MAST product still exists -- surface it so that is not
    # mistaken for missing data (issue #38: cloudef o005 F210M/F480M reduced later than F162M/F360M).
    with_mosaic = _filters_with_mosaic(o)
    awaiting_reduction = [f for f in avail if f not in with_mosaic]
    png = _save(fig, f"{o.obsid}_stage1.png")
    metrics = dict(stage=1, sw=sw, lw=lw,
                   sw_present=bool(psw), lw_present=bool(plw),
                   finite_fraction=fracs, single_module_filters=modules,
                   nominal_filters=list(o.filters), available_filters=avail,
                   mosaic_filters=with_mosaic, awaiting_reduction=awaiting_reduction,
                   mast_fallback_filters=mast_shown, dropped_filters=dropped,
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
        # Distinguish "not catalogued at all" from "detected but not yet calibrated": a CMD needs
        # magnitudes, and the per-filter DAO catalogs carry positions only.  Say which it is.
        dao_only = bool(_dao_position_catalog(o, sw) and (not lw or _dao_position_catalog(o, lw)))
        reason = (f"per-filter DAO catalogues exist for {want} but carry positions only (no "
                  f"calibrated photometry) — a CMD needs magnitudes; the merged/MAST catalogue is "
                  f"not built yet" if dao_only else
                  f"no release catalogue and no MAST source catalogue for {want} yet")
        png = _red_flag_figure(o, "stage2", "NO CATALOG FOR CMD",
                               f"The CMD is empty: {reason}.")
        metrics.update(red_flag=True, red_flag_reason=reason, passed=False)
        return png, metrics
    t = Table.read(_used(cat, f"CMD catalogue ({kind})"))

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

    # CMD + shared-y marginal LF.  TWO versions: all stars, and (when per-band flux errors exist)
    # one limited to S/N > 10 in BOTH bands -- the cleaner locus.
    msw = np.asarray(t[csw], float); mlw = np.asarray(t[clw], float)
    g = np.isfinite(msw) & np.isfinite(mlw)

    def _sn(band):
        fc, ec = f"flux_{band.lower()}", f"flux_err_{band.lower()}"
        if fc in t.colnames and ec in t.colnames:
            with np.errstate(invalid="ignore", divide="ignore"):
                return np.asarray(t[fc], float) / np.asarray(t[ec], float)
        return None
    snsw, snlw = _sn(sw), _sn(lw)
    have_sn = snsw is not None and snlw is not None
    hi = (g & np.isfinite(snsw) & np.isfinite(snlw) & (snsw > 10) & (snlw > 10)) if have_sn else None

    def _draw_cmd(gs, r, sel, tag):
        a = fig.add_subplot(gs[r, 0]); amarg = fig.add_subplot(gs[r, 1], sharey=a)
        cax = fig.add_subplot(gs[r, 2])
        col = msw[sel] - mlw[sel]; mg = mlw[sel]
        xlo, xhi = np.nanpercentile(col, [1, 99])
        ylo, yhi = np.nanpercentile(mg, [0.5, 99.5])
        a.set_xlim(xlo, xhi)
        a.set_ylim(yhi, ylo)   # brighter up

        fig.canvas.draw()
        bbox = a.get_window_extent()
        nx = 100
        ny = max(1, int(round(nx * bbox.height / (bbox.width * np.sqrt(3)))))
        hb = a.hexbin(col, mg, gridsize=(nx, ny), extent=(xlo, xhi, ylo, yhi), bins="log", cmap="viridis", mincnt=1)
        a.set_xlabel(f"{sw} - {lw}"); a.set_ylabel(lw)

        fig.colorbar(hb, cax=cax, label="log N")
        hh, edges = np.histogram(mg, bins=50); ctr = 0.5 * (edges[1:] + edges[:-1])
        amarg.step(hh, ctr, where="mid", color="k", lw=0.9)
        pk = ctr[int(np.argmax(hh))]
        amarg.axhline(pk, color="r", lw=0.8)
        amarg.set_xlabel(f"N\nturnover≈{pk:.1f}", fontsize=8)
        amarg.tick_params(labelleft=False, labelsize=7); amarg.margins(y=0)
        a.set_title(f"{tag} (n={int(np.sum(sel))})", fontsize=9)
        return float(pk)

    # only draw the S/N>10 row when it has enough stars -- an empty selection would make
    # nanpercentile return NaN and crash set_xlim.
    have_hi = bool(have_sn and int(np.sum(hi)) >= 50)
    nrows = 2 if have_hi else 1
    fig = plt.figure(figsize=(8.2, 5.6 * nrows))
    gs = fig.add_gridspec(nrows, 3, width_ratios=[4.0, 1.15, 0.16], wspace=0.05, hspace=0.32)
    peak = _draw_cmd(gs, 0, g, "all stars")
    metrics.update(n_stars=int(g.sum()), lf_turnover=peak, sw_col=csw, lw_col=clw,
                   passed=int(g.sum()) > 500)
    if have_hi:
        peak_hi = _draw_cmd(gs, 1, hi, "S/N > 10 in both bands")
        metrics.update(n_stars_hi_sn=int(np.sum(hi)), lf_turnover_hi_sn=peak_hi)
    fig.suptitle(f"{o.target} {o.obsid} — CMD ({kind.replace('_dedup', '')})", fontsize=11)
    return _save(fig, f"{o.obsid}_stage2.png"), metrics


# --------------------------------------------------------------------------- STAGE 3
def stage3_calibration(o: Observation, sw):
    """JWST (SW ~ F212N) catalogue mag vs VIRAC Ks for matched stars.  The cyan 1:1 line is the
    ideal unit-slope relation (not a fit); the measured free slope and the scatter about the
    locus gate whether the photometric zeropoint is sane."""
    import astropy.units as u
    from astropy.coordinates import search_around_sky
    fig, ax = _fig(1, 1, 5.5, 5.5)
    metrics = dict(stage=3, sw=sw)
    path = _mosaic_path(o, sw)
    ref = _viraccache_path(o) or _refcat_path(o)   # cache has real Ksmag
    ep = _obs_epoch(o, path)
    ref_sc, ref_mag = (aa.load_reference(_used(ref, "VIRAC2/Gaia reference catalogue"), ep)
                       if (ref and ep) else (None, None))
    # Read the JWST catalog (release -> MAST) -- do NOT re-detect on the mosaic.
    jsc, jmag, src = _jwst_sources(o, sw)
    metrics["source"] = src
    a = ax[0][0]
    if jsc is None:
        import matplotlib.pyplot as plt
        plt.close(fig)          # close the empty fig before the red-flag builds its own
        # Calibration needs magnitudes; a positions-only DAO catalogue can't be calibrated.  Say
        # so, and point at stage 4 (which CAN measure the frame offset from those positions).
        dao_only = bool(_dao_position_catalog(o, sw))
        reason = (f"per-filter DAO catalogue exists for {sw} but carries positions only (no "
                  f"calibrated photometry) — calibration needs magnitudes; see stage 4 for the "
                  f"frame offset" if dao_only else
                  f"no release or MAST source catalogue for {sw} yet")
        png = _red_flag_figure(o, "stage3", "NO PHOTOMETRY TO CALIBRATE", reason + ".")
        metrics.update(red_flag=True, red_flag_reason=reason, passed=False)
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
    # Iterate the 3-sigma locus clip to CONVERGENCE: stopping after one step at k=3 leaves the
    # reported slope/scatter dependent on where the iteration happened to halt.
    #
    # NOTE the loop has TWO exits, and the second one changes what gets reported.  ``loc.all()``
    # is convergence.  ``loc.sum() < 30`` is a floor on the surviving sample, and it breaks BEFORE
    # ``xf, yf`` are reassigned, so a field sparse enough to hit it reports the slope, scatter and
    # n_locus of the UNCLIPPED set.  The upstream gate admits a field at 30 matches, so a ~34-star
    # locus carrying a red mismatch cloud takes that exit on the first pass and reports
    # n_locus == n_matched -- which is indistinguishable from a locus so clean the clip rejected
    # nothing.  See JWST-GC/data-qa#97.
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
    # Draw ONLY the ideal UNIT-SLOPE (1:1) reference line -- the relation a well-calibrated
    # zeropoint should follow.  Anchor it on the DENSE stellar locus via the MODE of (y-x): the
    # sigma-clipped fit does not cleanly separate the bright locus from the red mismatch cloud, so
    # a median of the clipped set lands between the two populations and the line misses the locus.
    # The mode picks the densest ridge.  The free-slope fit is NOT drawn (its slope wanders with
    # the cloud and reads as a bad fit); the slope is still measured and gated below.
    dy = y - x
    hcnt, hedge = np.histogram(dy, bins=60)
    zp1 = float(0.5 * (hedge[int(np.argmax(hcnt))] + hedge[int(np.argmax(hcnt)) + 1]))   # locus zp
    xs = np.array([np.nanmin(x), np.nanmax(x)])
    a.plot(xs, xs + zp1, "c-", lw=1.4, label="1:1 line")
    a.set_xlabel("VIRAC Ks [mag]"); a.set_ylabel(f"JWST {sw} catalog mag")
    a.legend(fontsize=8, loc="upper left")
    a.set_title(f"{o.obsid} calibration  n={int(g.sum())} (locus {n_locus})  "
                f"slope={slope:.2f}  scatter={scat:.2f}  locus zp={zp1:.2f}", fontsize=9)
    # Split gate: keep the SLOPE window tight (a zeropoint check must falsify on slope), widen
    # only the SCATTER for the real narrow-vs-broad (F212N vs Ks) colour/extinction spread.
    metrics.update(n_matched=int(g.sum()), n_locus=n_locus, slope=float(slope),
                   zeropoint=float(zp), scatter=scat,
                   passed=(0.8 < slope < 1.2 and scat < 0.8))
    return _save(fig, f"{o.obsid}_stage3.png"), metrics


# --------------------------------------------------------------------------- STAGE 4
def _pooled_daophot(o: Observation, filt, max_files=64, exclude=None):
    """Pool the per-exposure DAOPHOT cats for one filter into (position, per-star astrometric
    sigma, instrumental mag, flux).  Unlike the merged science catalog, the per-exposure cats
    carry the formal PSF-fit position uncertainty: ``dra``/``ddec`` are the RA/Dec 1-sigma
    errors in arcsec (== x_err/y_err * pixel scale, so no pixel-scale assumption is needed).
    ``exclude`` is a set of peppar exposure tokens (``jw…_<vgroup>_<exp>``) whose cats are dropped
    -- used by the stage-6 clean recompute to leave out bad-PSF exposures.
    Returns (SkyCoord, sig_ra_mas, sig_de_mas, instr_mag, flux) or None."""
    import astropy.units as u
    from astropy.table import vstack, Table
    cats = _daophot_glob(o, filt)          # obs-scoped
    if exclude:                            # drop the flagged exposures (vgroup<VG>_exp<EE> in name)
        keys = {_daophot_key_for_token(t) for t in exclude}
        cats = [c for c in cats if not any(k in os.path.basename(c) for k in keys)]
    if not cats:
        return None
    if len(cats) > max_files:
        # Cap the pool, but sample ROUND-ROBIN across detectors: a plain alphabetical head takes
        # nrca1..nrca3 and drops nrca4 + all of NRCB, so the stage-6 curve would describe module A
        # only -- one of the two modules stage 5 sets out to compare.
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
            t = Table.read(_used(c, f"per-exposure daophot ({filt})"))
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


def _binned_rms(x, r, width=0.5, minn=15):
    """RMS of ``r`` (per-star residual magnitude, mas) in fixed-width bins of ``x``.
    Returns (rms, centre) arrays or (None, None)."""
    g = np.isfinite(x) & np.isfinite(r)
    x, r = x[g], r[g]
    if x.size < minn:
        return None, None
    edges = np.arange(np.floor(x.min() / width) * width,
                      np.ceil(x.max() / width) * width + width, width)
    idx = np.digitize(x, edges)
    ctr, rms = [], []
    for b in range(1, len(edges)):
        m = idx == b
        if m.sum() < minn:
            continue
        ctr.append(0.5 * (edges[b - 1] + edges[b]))
        rms.append(float(np.sqrt(np.mean(r[m] ** 2))))
    if len(ctr) < 3:
        return None, None
    return np.array(rms), np.array(ctr)


def _offset_failure_reason(o: Observation, filt, jsc, ref_sc, bulk):
    """Explain WHY the JWST<->VIRAC offset is unmeasurable, as specifically as the data allow --
    the user's ask: not a bare "FAILURE" but the actual cause (issue #7).  Order: no catalog, no
    reference, disjoint footprints, then (both overlap, still no peak) report what IS known and say
    the cause is undetermined.  It does NOT assert a magnitude-range mismatch it never measured."""
    if jsc is None:
        # _jwst_positions already tried release-merged -> MAST -> per-filter DAO, so nothing usable
        # exists on disk for this obs+filter.
        return (f"no release, MAST, or per-filter DAO catalog with usable positions for {filt} "
                f"yet — this observation is not catalogued")
    if ref_sc is None:
        return "no VIRAC reference catalogue for this field/epoch"
    # Both catalogues exist but produced no matches: is it a footprint or a content problem?  The
    # footprint boxes below assume no RA wrap; every GC/registry field is far from RA=0, but a
    # wrap-straddling field would give a bogus box -- guard it rather than emit a wrong verdict.
    jra = jsc.ra.deg; rra = ref_sc.ra.deg
    ra_wrap = (float(np.nanmax(jra)) - float(np.nanmin(jra)) > 180.0 or
               float(np.nanmax(rra)) - float(np.nanmin(rra)) > 180.0)
    jd = (float(np.nanmin(jsc.dec.deg)), float(np.nanmax(jsc.dec.deg)))
    rd = (float(np.nanmin(ref_sc.dec.deg)), float(np.nanmax(ref_sc.dec.deg)))
    if not ra_wrap:
        jr = (float(np.nanmin(jra)), float(np.nanmax(jra)))
        rr = (float(np.nanmin(rra)), float(np.nanmax(rra)))
        if (min(jd[1], rd[1]) - max(jd[0], rd[0]) <= 0) or (min(jr[1], rr[1]) - max(jr[0], rr[0]) <= 0):
            return (f"the JWST footprint (RA {jr[0]:.3f}–{jr[1]:.3f}, Dec {jd[0]:.3f}–{jd[1]:.3f}) and the "
                    f"VIRAC reference (RA {rr[0]:.3f}–{rr[1]:.3f}, Dec {rd[0]:.3f}–{rd[1]:.3f}) do not "
                    f"overlap on the sky — the reference covers a different region")
    pr = (bulk or {}).get("peak_ratio", 0.0)
    npairs = (bulk or {}).get("npairs")
    # Report the MEASURED counts (JWST sources, VIRAC reference stars, matched pairs) rather than
    # naming one of several possible causes -- the failure could be too few reference stars, too few
    # JWST sources, or a genuine no-peak, and this function did not distinguish them.
    counts = (f"{len(jsc)} JWST sources vs {len(ref_sc)} VIRAC reference stars in the footprint"
              + (f", {npairs} matched pairs at the best peak" if npairs is not None else ""))
    # Report the JWST magnitude range if we have it; do NOT claim a VIRAC comparison that was not run.
    jmag = _jwst_sources(o, filt)[1]
    mag_note = ""
    if jmag is not None and np.isfinite(jmag).any():
        mag_note = (f"  (JWST {filt} spans {np.nanpercentile(jmag, 1):.1f}–{np.nanpercentile(jmag, 99):.1f} "
                    f"mag; no VIRAC magnitude comparison was made.)")
    if pr >= aa.MIN_PEAK_RATIO:
        # A confident whole-field peak exists while the per-cell fallback found none.  Report that;
        # printing "peak_ratio {>=4} < 4" here would contradict itself.
        return (f"a whole-field cross-correlation peak exists (peak_ratio {pr:.2f} ≥ "
                f"{aa.MIN_PEAK_RATIO}) but no cell yielded a peak of its own ({counts}).{mag_note}")
    return (f"footprints overlap but no common-star histogram peak (peak_ratio {pr:.2f} < "
            f"{aa.MIN_PEAK_RATIO}); {counts}. The cause was not determined here.{mag_note}")


# A cell whose offset differs from the field value by more than this (mas) is "deviating".  Set at
# the 15-30 mas release tolerance, well below the 75 mas absolute gate: a field whose offset varies
# by more than this across the mosaic falls outside survey noise however small its field value.
_CELL_SPREAD_MAX = 30.0
# A coherent (adjacency-confirmed) deviating region holding more than this FRACTION of the measured
# sources fails the consistency gate.  Small, because a real ~100 mas sub-region offset ruins the
# proper motions of the stars in it even at a few percent of the catalog.
_CELL_BAD_FRAC = 0.02
# Require at least this fraction of the field's sources to sit in cells with a measurable peak,
# else the field is too sparsely sampled to pass.
_CELL_MIN_COVERAGE = 0.5
# Low peak floor: a cell qualifies once it has SOME peak above chance.  aa.MIN_PEAK_RATIO (4.0)
# anti-correlates with source count -- the background it divides by, median(H[H>0]), grows with the
# chance-pair count -- so a 4.0 cut keeps the SPARSE cells and drops the dense ones, which makes the
# verdict depend on which side of a defect happens to be sparse (o046 vs o049; PR #54 review).  This
# floor accepts any real peak; the cells are then weighted by SOURCE COUNT and judged for
# consistency by adjacency (below), neither of which depends on the density-biased ratio.
_CELL_PR_FLOOR = 1.5
# A cell offset farther than this from the source-weighted field consensus is not a believable tie
# (a real tie, even with distortion or a sub-region discontinuity, does not jump ~arcsec between
# cells).  Used ONLY together with the weight floor below to drop spurious low-occupancy xcorr peaks;
# a genuine, well-populated discontinuity is kept and judged by the adjacency test.  4x the 75 mas gate.
_CELL_SPURIOUS_MAX = 300.0
# ...and only when the far cell holds less than this fraction of the measured sources, so a real,
# well-populated deviating region is flagged (adjacency test), never silently discarded.
# Separate constant from the pass-gate tolerance (_CELL_BAD_FRAC): this is the pre-statistics
# discard threshold for a low-occupancy spurious cell, a different decision that happens to share a
# value today (PR #101 review -- do not re-collapse them onto one name).
_CELL_SPURIOUS_WT = 0.02
# When the SURVIVING cells (after spurious rejection) still scatter by more than this, the per-cell
# histogram-peak method has broken down wholesale -- not a real spatial discontinuity (which spans a
# few cells at ~tens of mas) but noise, ~arcsec of it (cloudef o005: 16 cells scatter 1783 mas while
# clean isolated stars tie the field to 12 mas).  A field that scatters this far between cells is not
# measuring a tie; defer to the clean isolated-star bulk instead of reporting the garbage and failing
# on a phantom discontinuity.  Well above any real mis-registration + distortion (gc2211 o050's real
# discontinuity is ~58 mas), well below the breakdown regime.
_CELL_SPREAD_ABSURD = 300.0
# Trust the isolated-star bulk as the override only when it rests on a solid clean sample.
_ISO_OVERRIDE_MIN_N = 50


def _cell_grid(jsc, ref_sc, ncell, min_src, pr_floor=_CELL_PR_FLOOR, min_pairs=None):
    """One ``ncell`` x ``ncell`` pass of the per-cell xcorr offset (see ``_cell_offsets``).

    Three DISTINCT thresholds, each on its own quantity.  Collapsing them into one constant is what
    produced the _ab_overlap 9.6x inflation, by counting stars where pairs were meant:
      * ``min_src`` -- minimum STAR occupancy required of BOTH the JWST cell and the (2"-margin)
        reference crop before xcorr is attempted.  At GC density the VIRAC *reference* crop is the
        binding one, since VIRAC is far sparser than JWST: a cell can hold 1000+ JWST sources and
        still fall short of ``min_src`` reference stars, in which case xcorr is never called.  That
        is why sickle's 4x4 cells drop with peak_ratio=None -- the REFERENCE stars ran out.
      * ``pr_floor`` -- minimum xcorr peak_ratio (peak height / chance background) to trust a peak.
      * ``min_pairs`` -- minimum number of matched PAIRS in the accepted peak (a pair count, not a
        star count); defaults to ``min_src``.

    Returns (cells, dropped)."""
    if min_pairs is None:
        min_pairs = min_src
    ra = jsc.ra.deg; dec = jsc.dec.deg
    rra = ref_sc.ra.deg; rde = ref_sc.dec.deg
    # RA-wrap guard: a footprint straddling RA=0 would give bogus linear bins (no GC field does).
    if float(np.nanmax(ra) - np.nanmin(ra)) > 180.0:
        ncell = 1
    re_ = np.linspace(ra.min(), ra.max(), ncell + 1)
    de_ = np.linspace(dec.min(), dec.max(), ncell + 1)
    mrg = 2.0 / 3600.0
    cells, dropped = [], []
    for i in range(ncell):
        for j in range(ncell):
            m = (ra >= re_[i]) & (ra <= re_[i + 1]) & (dec >= de_[j]) & (dec <= de_[j + 1])
            n = int(m.sum())
            if n < min_src:                          # too few JWST sources in the cell
                continue
            cra, cdec = 0.5 * (re_[i] + re_[i + 1]), 0.5 * (de_[j] + de_[j + 1])
            rm = ((rra >= re_[i] - mrg) & (rra <= re_[i + 1] + mrg) &
                  (rde >= de_[j] - mrg) & (rde <= de_[j + 1] + mrg))
            n_ref = int(rm.sum())
            xc = aa.xcorr(jsc[m], ref_sc[rm]) if n_ref >= min_src else None
            if xc and xc.get("peak_ratio", 0) >= pr_floor and xc.get("npairs", 0) >= min_pairs:
                cells.append(dict(i=i, j=j, ra=cra, dec=cdec, dra=float(xc["dra"]),
                                  dde=float(xc["ddec"]), off=float(xc["off"]),
                                  peak_ratio=float(xc["peak_ratio"]), n=n, n_ref=n_ref,
                                  npairs=int(xc["npairs"])))
            else:
                # record WHY it dropped: no reference stars to correlate against, vs a real no-peak.
                dropped.append(dict(i=i, j=j, ra=cra, dec=cdec, n=n, n_ref=n_ref,
                                    reason=("too few reference stars" if n_ref < min_src
                                            else "no clear peak")))
    return cells, dropped


def _cell_offsets(jsc, ref_sc, ncell=4, min_per_cell=300):
    """Measure the JWST-minus-VIRAC offset in an ``ncell`` x ``ncell`` spatial grid over the JWST
    footprint, each cell by the xcorr HISTOGRAM PEAK against the local reference (cropped to the
    cell + a 2" margin).  A field-wide nearest-neighbour median reads SMALLER the further the frame
    is displaced (~1.8 mas at a 2" shift; PR #54 review), so it is unusable here.

    ADAPTIVE: a small or sparse field (e.g. sickle jw03958-o007, a sub640 subarray) can have a
    clean WHOLE-FIELD peak and still hold too few *reference* stars per 4x4 cell to attempt xcorr.
    Gating on the fine grid alone then red-flags a perfectly measurable field (issue #13; sickle's
    4x4 cells hold 1000+ JWST sources but only ~200 VIRAC reference stars, below the 300 crop
    floor).  So when the requested grid yields no cell, fall back to progressively coarser grids,
    each cell spanning more reference stars, down to a single whole-field cell measured under the
    confident-peak gate (``aa.MIN_PEAK_RATIO``).  A handful of common stars measures a field
    offset; the fine grid adds the spatial information on top of it.

    Returns (cells, dropped, grid_used).  ``grid_used`` is the ncell of the grid that produced the
    measurement (1 = whole-field fallback, carrying no per-cell spatial information), so the caller
    can set the coverage/consistency gates from it directly."""
    attempts = [(ncell, min_per_cell, _CELL_PR_FLOOR)]
    if ncell > 2:
        attempts.append((2, 150, _CELL_PR_FLOOR))
    attempts.append((1, 100, aa.MIN_PEAK_RATIO))     # whole-field offset, confident peak required
    last_dropped = []
    for nc, mpc, prf in attempts:
        cells, dropped = _cell_grid(jsc, ref_sc, nc, mpc, prf)
        last_dropped = dropped
        if cells:
            return cells, dropped, nc
    return [], last_dropped, 0


def _cell_consistency(cells, dropped):
    """Combine the per-cell JWST-minus-reference offsets into one field offset plus a
    spatial-consistency verdict.

    * ``off_dra``/``off_dde`` are each a WEIGHTED MEDIAN over cells, weighted by the cell's source
      count: sort the cells by that component and take the value at which the cumulative source
      count crosses half the total.  Weighting by source count makes a cell holding a tenth of the
      catalog count for a tenth of the catalog.  The two components are taken SEPARATELY and then
      combined as ``off_med = hypot``, so ``off_med`` is the length of the component-wise median
      vector.  Four cells at (+50, 0), (-50, 0), (0, +50), (0, -50) mas therefore give
      ``off_med = 0`` while every cell sits 50 mas from the field value; the adjacency test below
      is what catches that arrangement.
    * a cell is DEVIATING when its offset differs from the field value by more than
      ``_CELL_SPREAD_MAX``, and CONFIRMED only when an orthogonally-adjacent cell also deviates: a
      real sub-region discontinuity spans several cells, while a lone mis-peaked cell (one 544 mas
      cell amid 9 mas neighbours) is a measurement failure in that cell and leaves the frame usable.
    * consistent = the confirmed-deviating cells hold < ``_CELL_BAD_FRAC`` of the measured sources
      AND enough of the field was measurable (coverage >= ``_CELL_MIN_COVERAGE``).

    Returns a dict of the numbers plus per-cell ``deviating``/``confirmed`` flags for plotting."""
    if not cells:
        return dict(n_cells=0, consistent=False)
    def _wmed(v, w):
        o = np.argsort(v); vs, ws = v[o], w[o]; cw = np.cumsum(ws)
        return float(vs[np.searchsorted(cw, 0.5 * cw[-1])])

    # Reject SPURIOUS cells before any stat.  A frame tie -- even with distortion or a real
    # sub-region discontinuity -- does not jump hundreds of mas between cells; per-cell dRA/dDec that
    # differ that much are not a tie.  A cell far from the SOURCE-WEIGHTED consensus while holding
    # only a tiny fraction of the sources is a spurious per-cell xcorr peak (few JWST sources against
    # a dense reference -> an accidental histogram bump that clears the low pr floor), not a
    # measurement.  Drop it into ``dropped`` so it neither pollutes the map nor inflates the spread;
    # the consensus is source-weighted, so the dense trustworthy cells define it and the spurious
    # low-occupancy cells cannot move it.  A real, high-weight discontinuity is NOT rejected (weight
    # floor) -- it stays and is judged by the adjacency test.  (issue #37: cloudef o002 had 4 edge
    # cells at 0.6-2.3" amid 3 dense cells consistent at ~150 mas.)
    cells, dropped = list(cells), list(dropped)
    dra0 = np.array([c["dra"] for c in cells]); dde0 = np.array([c["dde"] for c in cells])
    ns0 = np.array([c["n"] for c in cells], float)
    cdra, cdde = _wmed(dra0, ns0), _wmed(dde0, ns0)
    resid0 = np.hypot(dra0 - cdra, dde0 - cdde)
    spurious = (resid0 > _CELL_SPURIOUS_MAX) & (ns0 / ns0.sum() < _CELL_SPURIOUS_WT)
    n_spurious = int(spurious.sum())
    if n_spurious and not spurious.all():         # never empty the whole set (pathological)
        for k in np.where(spurious)[0]:
            c = cells[k]
            dropped.append(dict(i=c["i"], j=c["j"], ra=c["ra"], dec=c["dec"], n=c["n"],
                                n_ref=c.get("n_ref"),
                                reason="spurious peak (offset inconsistent with field consensus)"))
        cells = [c for k, c in enumerate(cells) if not spurious[k]]
    else:
        n_spurious = 0

    dra = np.array([c["dra"] for c in cells]); dde = np.array([c["dde"] for c in cells])
    ns = np.array([c["n"] for c in cells], float)
    mdra, mdde = _wmed(dra, ns), _wmed(dde, ns)
    off_med = float(np.hypot(mdra, mdde))
    dev = np.hypot(dra - mdra, dde - mdde)
    deviating = dev > _CELL_SPREAD_MAX
    ij = {(c["i"], c["j"]): k for k, c in enumerate(cells)}
    confirmed = np.zeros(len(cells), bool)
    for k, c in enumerate(cells):
        if not deviating[k]:
            continue
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nb = ij.get((c["i"] + di, c["j"] + dj))
            if nb is not None and deviating[nb]:
                confirmed[k] = True
                break
    meas_n = float(ns.sum())
    drop_n = float(sum(d["n"] for d in dropped))
    coverage = meas_n / (meas_n + drop_n) if (meas_n + drop_n) else 0.0
    bad_frac = float(ns[confirmed].sum() / meas_n) if meas_n else 0.0
    # How much the cells disagree with each other.  This is a SPREAD and is reported as one.  The
    # consistency verdict comes from the minority-sensitive adjacency test above, since a robust
    # spread is by construction blind to a small number of deviating cells.
    #
    # It stays a spread.  ``off_med / (spread / sqrt(n_cells))`` was reported here as "Nsigma from
    # zero"; that ratio sits at ~1.1 (n>=9) or ~1.5 (n=4) whenever the true offset is zero, whatever
    # the scale of the scatter, because ``off_med`` is a positive-definite length built from the
    # same two medians whose sampling error is the denominator.  It therefore reported "1 sigma"
    # for a 780 mas offset -- a number that carried no information about the offset.  It also
    # shrinks as sqrt(n_cells) while the error that dominates here (the several-mas pull of the
    # dense reference on every cell's histogram peak) is common to all cells and holds steady as
    # cells are added.  Stage 8 measures its significance against
    # a shuffled-position null for the same reason.
    spread = float(np.hypot(aa.mad_std(dra), aa.mad_std(dde))) if len(cells) >= 2 else None
    consistent = bool(len(cells) >= 4 and bad_frac < _CELL_BAD_FRAC and coverage >= _CELL_MIN_COVERAGE)
    return dict(off_med=off_med, off_dra=mdra, off_dde=mdde, spread=spread,
                n_cells=len(cells), n_dropped=len(dropped), n_deviating=int(deviating.sum()),
                n_confirmed=int(confirmed.sum()), n_spurious=n_spurious, bad_src_frac=bad_frac,
                coverage=coverage, consistent=consistent, deviating=deviating, confirmed=confirmed,
                cells=cells, dropped=dropped)     # spurious-filtered; caller plots THESE


def _dataset_label(metrics):
    """Short label naming WHICH catalogue a stage-4/5 plot is built from (jicama mN / DAO / MAST),
    so the reader is never left guessing the data source."""
    s = str(metrics.get("source", ""))
    if "dao" in s.lower():
        return "per-filter DAO positions"
    if s.startswith("release"):
        m = _MLEVEL_RE.search(s)
        return f"jicama m{m.group(1)}" if m else ("jicama m8" if "m8" in s.lower() else "jicama")
    if "mast" in s.lower():
        return "MAST catalogue"
    return "jicama catalogue"


def _add_marginals(ax, x, y, color="#4477aa", bins=40, weights=None):
    """Attach top (x) and right (y) marginal histograms to ``ax`` as inset axes locked to its
    data limits, so the 1-D ΔRA / ΔDec distributions are shown alongside the 2-D scatter/hexbin
    on the positional-offset panels.  The insets sit just outside the axes; the caller must leave
    top/right room (wspace/hspace) so they do not collide with a neighbour or the panel title."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    w = np.asarray(weights, float) if weights is not None else None
    gx = np.isfinite(x); gy = np.isfinite(y)
    axt = ax.inset_axes([0.0, 1.02, 1.0, 0.16])
    axr = ax.inset_axes([1.02, 0.0, 0.16, 1.0])
    if gx.any():
        axt.hist(x[gx], bins=bins, color=color, weights=(w[gx] if w is not None else None))
    axt.set_xlim(ax.get_xlim()); axt.axis("off")
    if gy.any():
        axr.hist(y[gy], bins=bins, orientation="horizontal", color=color,
                 weights=(w[gy] if w is not None else None))
    axr.set_ylim(ax.get_ylim()); axr.axis("off")
    return axt, axr


# A confident histogram-peak field offset that sits farther than this from the clean isolated-star
# offset is not the true bulk: at GC density against a SPARSE reference the pairwise-offset histogram
# grows spurious lobes, and the peak can land on one (cloudef o002: peak 153 mas at (-142,+57) while
# unambiguous isolated matches sit at (+68,-39), ~78 mas -- opposite direction).  Flagged low-confidence.
_BULK_DISAGREE_MAX = 50.0


def _mast_catalog_positions(o: Observation, filt):
    """(SkyCoord, flux) from the MAST-delivered STScI L3 source catalogue (``*_<filt>_cat.ecsv`` or
    ``.fits``), the pipeline-independent reference.  Handles the pupil-filter naming (F162M ships as
    ``f150w2-f162m``) and any tile token (t001/t002).  None if not on disk (download separately)."""
    import astropy.units as u
    from astropy.table import Table
    from astropy.coordinates import SkyCoord
    if not filt:
        return None
    fl = filt.lower()
    hits = []
    for ext in ("ecsv", "fits"):
        # TARGETED patterns (NOT a recursive ``**`` walk of the whole mastDownload tree, which is
        # enormous and made stage 4 hang): the MAST L3 catalogue sits one product-dir deep.
        pats = (f"{BASE}/{o.field}/mastDownload/JWST/{o.obsid}_t*_nircam_*{fl}*/"
                f"{o.obsid}_t*_nircam_*{fl}*_cat.{ext}",
                f"{BASE}/{o.field}/MAST_FITS/{o.obsid}_t*_nircam_*{fl}*_cat.{ext}")
        for pat in pats:
            hits += [p for p in glob.glob(pat)
                     if not any(s in os.path.basename(p).lower()
                                for s in ("nrca", "nrcb", "destreak", "segm"))]
        if hits:
            break
    # Nothing local: DOWNLOAD it (guarded) rather than return None -> a missing MAST catalogue is
    # fetched, not treated as absent.  No-ops for every current field (catalogues already on disk).
    best = sorted(hits)[-1] if hits else _download_mast_l3_catalog(o, filt)
    if not best:
        return None
    try:
        t = Table.read(best)
    except (OSError, ValueError):
        return None
    if "sky_centroid" in t.colnames:
        sc = SkyCoord(t["sky_centroid"])
    elif "sky_centroid.ra" in t.colnames:
        sc = SkyCoord(np.asarray(t["sky_centroid.ra"], float) * u.deg,
                      np.asarray(t["sky_centroid.dec"], float) * u.deg)
    else:
        return None
    fcol = next((c for c in ("aper_total_flux", "aper70_flux", "aper50_flux", "aper30_flux")
                 if c in t.colnames), None)
    flux = np.asarray(t[fcol], float) if fcol else np.ones(len(sc))
    g = np.isfinite(sc.ra.deg) & np.isfinite(sc.dec.deg)
    return (sc[g], flux[g]) if int(g.sum()) >= 50 else None


def _crossmatch_offset(jsc, ref_sc, restrict_footprint=False):
    """Bulk JWST−VIRAC offset via the pipeline's validated histogram-stacking `measure_offset`
    (sweep + contrast + edge-alias rejection).  Falls back to `aa.xcorr` when the pipeline is not
    installed.  When ``restrict_footprint``, VIRAC is cropped to the JWST footprint first (a small
    single-tile MAST catalogue against the full VIRAC tile otherwise aliases on the footprint-slide
    ridge).  Returns dict(off, dra, dde, contrast, ok, edge, n, source) or None.  Sign convention:
    (JWST − VIRAC), matching stage 4's per-cell offsets."""
    if jsc is None or ref_sc is None or len(jsc) < 50:
        return None
    ref = ref_sc
    if restrict_footprint:
        mrg = 2.0 / 3600.0
        ra, dec = jsc.ra.deg, jsc.dec.deg
        box = ((ref_sc.ra.deg >= ra.min() - mrg) & (ref_sc.ra.deg <= ra.max() + mrg) &
               (ref_sc.dec.deg >= dec.min() - mrg) & (ref_sc.dec.deg <= dec.max() + mrg))
        ref = ref_sc[box]
        if len(ref) < 50:
            return None
    if _pipe_measure_offset is not None:
        import astropy.units as u
        r = _pipe_measure_offset(jsc, ref, confirm_windows=True)   # (JWST − VIRAC) at the peak
        if r is None:
            return None
        edge = float(r.get("window_edge_fraction", 0.0))
        # a peak riding the search-window edge is a footprint-slide alias, not a tie (issue #158);
        # treat it as not-ok even if measure_offset's own gate passed it.
        ok = bool(r["ok"]) and edge < 0.5
        return dict(off=float(r["off"]), dra=float(r["dra"]), dde=float(r["ddec"]),
                    contrast=float(r["contrast"]), ok=ok, edge=edge,
                    window=float(r["window_arcsec"]), n=int(r["npairs"]), source="measure_offset")
    xc = aa.xcorr(jsc, ref)                                        # fallback: single-window histogram
    if xc is None:
        return None
    return dict(off=float(xc["off"]), dra=float(xc["dra"]), dde=float(xc["ddec"]),
                contrast=float(xc["peak_ratio"]), ok=bool(xc["peak_ratio"] >= aa.MIN_PEAK_RATIO),
                edge=0.0, window=float(aa.XMAXSEP.to("arcsec").value), n=int(xc["npairs"]),
                source="xcorr")


def _cell_map_broke_down(spread, iso_off, iso_n):
    """True when the per-cell offset map has broken down wholesale (surviving cells scatter by more
    than ``_CELL_SPREAD_ABSURD``) AND a solid clean isolated-star bulk exists to fall back on.  In
    that state the per-cell histogram peaks are noise, not a tie, and the isolated bulk is the
    trustworthy field offset (cloudef o005: 1783 mas cell scatter vs a 12 mas isolated tie)."""
    return bool(spread is not None and spread > _CELL_SPREAD_ABSURD
                and iso_off is not None and iso_n >= _ISO_OVERRIDE_MIN_N)


def _isolated_bulk(jsc, ref_sc, iso_arcsec=0.5, match_arcsec=0.15, ambig_arcsec=0.4):
    """Bulk JWST−VIRAC offset from CLEAN matches only, as an independent check on the histogram peak:
    isolated JWST stars (nearest other JWST > ``iso_arcsec``) with an unambiguous VIRAC match (nearest
    < ``match_arcsec``, second-nearest > ``ambig_arcsec``).  These do not suffer the nearest-neighbour
    collapse (they are isolated) nor the spurious-peak failure (they are matched one-to-one), so their
    median offset is the true bulk where enough of them exist.  Returns (median_dRA, median_dDec, n) in
    mas, or None if too few clean matches (VIRAC too sparse to check)."""
    import astropy.units as u
    if jsc is None or ref_sc is None or len(jsc) < 100 or len(ref_sc) < 50:
        return None
    _i, ss, _d = jsc.match_to_catalog_sky(jsc, nthneighbor=2)
    j = jsc[ss > iso_arcsec * u.arcsec]
    if len(j) < 20:
        return None
    i1, s1, _ = j.match_to_catalog_sky(ref_sc, nthneighbor=1)
    _i2, s2, _ = j.match_to_catalog_sky(ref_sc, nthneighbor=2)
    keep = (s1 < match_arcsec * u.arcsec) & (s2 > ambig_arcsec * u.arcsec)
    n = int(keep.sum())
    if n < 15:
        return None
    jm, rm = j[keep], ref_sc[i1[keep]]
    cosd = float(np.cos(np.radians(float(np.median(jm.dec.deg)))))
    dra = (jm.ra - rm.ra).to(u.mas).value * cosd
    dde = (jm.dec - rm.dec).to(u.mas).value
    return float(np.median(dra)), float(np.median(dde)), n


def _catalog_vs_alignment_age(o: Observation, src):
    """(catalog_date, alignment_date, catalog_name) ISO strings for the staleness check, or Nones.

    Stage 4 reads a merged/release catalog whose ``skycoord_ref`` bakes in whatever frame tie was
    current WHEN THE MERGE RAN.  If that catalog is older than the field's current alignment (the
    VIRAC2-locked offsets table the reduction now uses), the merge predates the re-tie, so the
    absolute offset it shows is the OLD tie -- not the data on disk (cloudef: the read merged catalog
    is 2026-07-01, the offsets table 2026-08; the per-filter August products are ~16 mas off VIRAC
    while this stale merge reads ~150).  Reduction fix: re-run the cross-band merge."""
    import datetime
    if not src or "release:" not in src:
        return None, None, None
    name = src.split("release:", 1)[1].split(" [", 1)[0].strip()
    cpath = os.path.join(BASE, o.field, "catalogs", name)
    # Compare against the OPERATIVE alignment table only -- not the newest of every CSV in the dir,
    # since an older per-filter/VVV table would otherwise set the bar (PR #101 review).
    #
    # Which table is operative depends on the field.  `alignment_config` dispatches each field to
    # either a VIRAC2-locked table (TABLE_LOCKED) or a checkpoint-written consensus table
    # (TABLE_CONSENSUS).  Globbing only for VIRAC2locked made this check a no-op for every
    # consensus-source field -- arches and w51 have no locked table at all, so `offs` was empty and
    # the function returned "not stale" for anything.
    #
    # arches is what this cost: its release catalogue is 2026-06-30 and its consensus table
    # 2026-08-16, so stage 4 reported a 14.8 mas offset and stage 7 an `astrom_improved: False`
    # against a catalogue built seven weeks before the field had any alignment at all.
    #
    # Preference, not union: where a locked table exists it is the operative one and a stale
    # consensus table beside it must not set the bar, which is the PR #101 finding.  The consensus
    # table is consulted only when no locked table exists.
    offs = glob.glob(os.path.join(BASE, o.field, "offsets", "Offsets_*VIRAC2locked.csv"))
    if not offs:
        offs = glob.glob(os.path.join(BASE, o.field, "offsets", "Offsets_*_consensus.csv"))
    if not (os.path.exists(cpath) and offs):
        return None, None, None
    cm = os.path.getmtime(cpath)
    am = max(os.path.getmtime(p) for p in offs)
    if cm >= am - 86400.0:                        # within a day -> not meaningfully stale
        return None, None, None
    iso = lambda t: datetime.datetime.fromtimestamp(t).strftime("%Y-%m-%d")
    return iso(cm), iso(am), name


def stage4_offsets(o: Observation, sw):
    """How far the JWST catalog positions sit from the same stars' VIRAC positions, measured
    separately in each of up to 16 spatial cells by the ``xcorr`` histogram peak, plus the
    NRCA-minus-NRCB offset measured without any external catalog.

    Reported as the field offset (the component-wise source-weighted median over cells) with the
    cell-to-cell spread.  A PASS needs a small field offset AND cells that agree with each other:
    one scalar cannot describe a frame whose two halves are registered differently, which is the
    failure this stage exists to catch (PR #54)."""
    import astropy.units as u
    metrics = dict(stage=4, sw=sw, offset_med_mas=None)
    path = _mosaic_path(o, sw)
    ref = _refcat_path(o)
    ep = _obs_epoch(o, path)
    ref_sc, _ = (aa.load_reference(_used(ref, "VIRAC2/Gaia reference catalogue"), ep)
                 if (ref and ep) else (None, None))
    # Positions come from the catalog (release -> MAST -> per-filter DAO).  Stage 4 needs only
    # positions, so an obs that has been detected but not yet merged (gc2211 o046) falls back to
    # its per-filter DAO catalog and is still measurable.
    jsc, src = _jwst_positions(o, sw)
    metrics["source"] = src

    # NRCA-minus-NRCB offset, from the per-detector daophot catalogs (module-split).  Measured by
    # the xcorr histogram peak, which recovers offsets out to the 1.5" search radius.  A 0.1"
    # nearest-match would run out of real pairs once the two modules sit more than ~100 mas apart,
    # and its median would collapse toward zero exactly when this gate needs to fire.  Omitted when
    # the per-detector catalogs for both modules are absent.
    a_sc, b_sc, _minfo = _module_positions(o, sw)
    im = None
    if a_sc is not None and b_sc is not None and len(a_sc) >= 50 and len(b_sc) >= 50:
        xc = aa.xcorr(a_sc, b_sc, maxsep=1.5 * u.arcsec)
        if xc and xc.get("peak_ratio", 0) >= aa.MIN_PEAK_RATIO and xc.get("npairs", 0) >= 100:
            im = dict(dra=xc["dra"], ddec=xc["ddec"], off=xc["off"])
    if im:
        metrics.update(intermodule_off=float(im["off"]), intermodule_filt=sw)

    # Measure the offset PER SPATIAL CELL (robust at density) before building the figure, so an
    # unmeasurable result becomes a red flag rather than an empty panel.
    if jsc is not None and ref_sc is not None:
        cells, dropped, grid_used = _cell_offsets(jsc, ref_sc)
    else:
        cells, dropped, grid_used = [], [], 0

    if not cells:
        bulk = aa.xcorr(jsc, ref_sc) if (jsc is not None and ref_sc is not None) else None
        reason = _offset_failure_reason(o, sw, jsc, ref_sc, bulk)
        png = _red_flag_figure(o, "stage4", "JWST↔VIRAC OFFSET UNMEASURABLE",
                               f"The positional-offset plot is empty: {reason}.")
        metrics.update(red_flag=True, red_flag_reason=reason, n_cells=0, passed=False)
        return png, metrics

    cc = _cell_consistency(cells, dropped)
    cells, dropped = cc["cells"], cc["dropped"]      # spurious cells moved to `dropped`; plot the rest
    cell_off_med, spread = cc["off_med"], cc["spread"]
    io = metrics.get("intermodule_off")
    # Each cell's value is a histogram peak against a DENSE reference, which carries a several-mas
    # bias: two catalogs tracing the same clustered field make a correlated wrong-pair background
    # that pulls the peak (brick 2221-o001 reads 9-17 mas by histogram where the same stars,
    # matched one to one, are 0.4-1.6 mas apart).  So once the cells have established that the
    # field offset is small, re-measure the reported value from the SAME STARS.  The per-cell map
    # goes on doing the job it exists for, which is finding a spatial discontinuity.
    # Clean isolated-star bulk (independent of the per-cell peaks) and the pipeline's swept,
    # contrast-gated, edge-alias-rejecting WHOLE-FIELD offset -- both needed to decide what to do
    # when the per-cell map breaks down.
    ib = _isolated_bulk(jsc, ref_sc)
    iso_off = float(np.hypot(ib[0], ib[1])) if ib is not None else None
    iso_n = int(ib[2]) if ib is not None else 0
    wf = _crossmatch_offset(jsc, ref_sc, restrict_footprint=True)     # swept measure_offset
    wf_ok = bool(wf is not None and wf.get("ok"))
    # Cell-method breakdown: the surviving cells scatter by ~arcsec, so the per-cell peaks are noise,
    # not a tie or a real discontinuity (cloudef o005: 1783 mas).  In that state DO NOT trust the
    # per-cell map -- and DO NOT trust the isolated-star median either, which collapses toward 0 when
    # the true offset approaches its 0.15" match window (Solved-Problem rule).  Use the swept
    # whole-field offset when it is CONFIDENT; otherwise the offset is genuinely unmeasurable here
    # (sparse VIRAC, no contrast) and must be reported as indicative, not as a clean tie.
    cell_map_unreliable = _cell_map_broke_down(spread, iso_off, iso_n)
    offset_unmeasurable = bool(cell_map_unreliable and not wf_ok)

    ss = aa.same_star_tie(jsc, ref_sc)
    if cell_map_unreliable and wf_ok:
        off_med = float(wf["off"])
        bulk_source = "whole-field xcorr (cell map unreliable)"
    elif cell_map_unreliable:
        off_med = iso_off                        # indicative only; flagged unmeasurable below
        bulk_source = "isolated (indicative; offset unmeasurable)"
    else:
        off_med = ss["off"] if ss else cell_off_med
        bulk_source = "same-star" if ss else "histogram"
    # The magnitude gate tests the per-cell histogram median (``cell_off_med``), which tracks a real
    # mis-registration out to XMAXSEP.  The same-star value is a refinement reported once a small
    # offset is confirmed: every same-star pair is matched inside 0.05", so that median is always
    # < 50 mas < THRESH["absolute"] and can never fail -- gating on it would pass a real 90 mas
    # offset (cell_off_med ~= 87, same-star ~= 13).  Gate on the LARGER of the two so a genuine
    # mis-registration the histogram sees survives the refinement.  When the cell map is unreliable,
    # gate on the confident whole-field offset; when even that is absent the offset is unmeasurable.
    if cell_map_unreliable:
        gate_off = off_med if wf_ok else None
    else:
        gate_off = max(cell_off_med, off_med)
    # PASS needs a small offset, enough COVERAGE, spatially CONSISTENT cells (no adjacency-confirmed
    # sub-region off by >30 mas holding >2% of sources; catches a minority a robust spread cannot),
    # and a small inter-module offset.  The adjacency test needs a real grid, so the WHOLE-FIELD
    # fallback (grid_used == 1) is the one case that skips it.  A low cell count still faces it: a
    # large field with 3/16 measurable cells is 21% coverage and must fail (issue #13 review).
    # Coverage is enforced in EVERY branch.
    whole_field = (grid_used == 1)
    spatial_assessed = not whole_field and not cell_map_unreliable
    coverage_ok = cc["coverage"] >= _CELL_MIN_COVERAGE
    # A garbage cell map cannot pronounce a spatial discontinuity: its adjacency-confirmed "deviating"
    # cells are noise agreeing with noise, so do not fail the field on them.
    spatial_ok = True if (whole_field or cell_map_unreliable) else cc["consistent"]
    # An unmeasurable offset cannot PASS: we could not confirm the frame is tied, so leave it for a
    # human rather than green-tick it on a collapse-biased number.
    passed = bool(gate_off is not None and gate_off < aa.THRESH["absolute"] and spatial_ok
                  and coverage_ok and (io is None or io < aa.THRESH["intermodule"]))
    metrics.update(offset_med_mas=off_med, offset_scatter_mas=spread,
                   bulk_off=off_med,                        # reported offset (same-star when available)
                   gate_off_mas=gate_off,                   # value the magnitude gate tests (cell histogram)
                   spatial_assessed=spatial_assessed,       # False -> whole-field fallback, no per-cell map
                   grid_used=grid_used, cell_map_unreliable=cell_map_unreliable,
                   offset_unmeasurable=offset_unmeasurable,
                   bulk_source=bulk_source, cell_off_med=cell_off_med,
                   same_star_off=(ss["off"] if ss else None),
                   same_star_npairs=(ss["npairs"] if ss else None),
                   same_star_scatter_mas=(ss["scatter"] if ss else None),
                   n_cells=cc["n_cells"], n_cells_dropped=cc["n_dropped"],
                   n_cells_spurious=cc.get("n_spurious", 0),
                   n_cells_deviating=cc["n_deviating"], n_cells_confirmed=cc["n_confirmed"],
                   bad_src_frac=cc["bad_src_frac"], cell_coverage=cc["coverage"],
                   cells_consistent=cc["consistent"], passed=passed)

    # Guard A -- cross-check the histogram-peak bulk against the CLEAN isolated matches (computed
    # above), which suffer neither the nearest-neighbour collapse nor the spurious-peak failure.  A
    # large disagreement means the reported peak is not the true bulk (sparse reference / spurious
    # lobe), so the number is low-confidence even when its peak_ratio looks strong.  When the cell map
    # is already declared unreliable the isolated bulk IS the reported value, so no disagreement.
    if ib is not None:
        mdra, mdde, nclean = ib
        disagree = 0.0 if cell_map_unreliable else float(np.hypot(cc["off_dra"] - mdra, cc["off_dde"] - mdde))
        metrics.update(isolated_bulk_off_mas=float(np.hypot(mdra, mdde)),
                       isolated_bulk_n=nclean, bulk_vs_isolated_disagree_mas=disagree,
                       bulk_low_confidence=bool(not cell_map_unreliable and disagree > _BULK_DISAGREE_MAX))
    # Guard B -- did we measure a catalog OLDER than the field's current alignment?  Then its
    # absolute offset is the pre-re-tie frame, not the reduction on disk.
    cdate, adate, cname = _catalog_vs_alignment_age(o, src)
    if cdate:
        metrics.update(catalog_stale=True, catalog_date=cdate,
                       alignment_date=adate, stale_catalog_name=cname)

    # Whole-field crossmatch offset from BOTH catalogues, via the pipeline's validated histogram-
    # stacking (sweep + contrast + edge-alias rejection): the jicama product AND the pipeline-
    # independent raw MAST L3 catalogue.  Shown side by side so a jicama-reduction offset the raw
    # MAST does not share is visible, and each carries its own confidence (contrast/ok).
    jic_x = _crossmatch_offset(jsc, ref_sc)
    if jic_x is not None:
        metrics.update(jicama_xoff_mas=jic_x["off"], jicama_xoff_contrast=jic_x["contrast"],
                       jicama_xoff_ok=jic_x["ok"])
    mastpos = _mast_catalog_positions(o, sw)
    mast_x = _crossmatch_offset(mastpos[0], ref_sc, restrict_footprint=True) if mastpos else None
    if mast_x is not None:
        metrics.update(mast_xoff_mas=mast_x["off"], mast_xoff_dra=mast_x["dra"],
                       mast_xoff_dde=mast_x["dde"], mast_xoff_contrast=mast_x["contrast"],
                       mast_xoff_ok=mast_x["ok"], mast_xoff_n=mast_x["n"])

    from matplotlib.patches import Circle
    cdra = np.array([c["dra"] for c in cells]); cdde = np.array([c["dde"] for c in cells])
    cra = np.array([c["ra"] for c in cells]); cdec = np.array([c["dec"] for c in cells])
    coff = np.array([c["off"] for c in cells])
    confirmed = cc["confirmed"]; deviating = cc["deviating"]
    ncols = 2 + (1 if im else 0)
    fig, ax = _fig(1, ncols, 6.2, 6.1)
    # extra column gap + top room so the middle panel's marginal histograms (which hang above the
    # axes) AND their multi-line title clear the suptitle: the top marginal reaches ~1.18x the axes
    # height and carries a 4-line title above that, so top must sit well below the suptitle at 0.99
    # (top=0.80 was too high -- the title collided with the suptitle, issue #35).  Taller figure
    # keeps the panels from shrinking.
    fig.subplots_adjust(wspace=0.62, top=0.70, bottom=0.10)
    col = 0
    # panel 1: contiguous 4x4 map of the per-cell offset (colour = offset).  Confirmed-deviating
    # cells get a RED outline; DROPPED cells (sources present, no clear peak) render grey, so a
    # discontinuity or missing coverage is visible on the map itself.
    a0 = ax[0][col]; col += 1
    # Scale the colour to the DATA.  A floor of 2*THRESH (=150 mas) drew a well-registered field
    # whose cells span 0-20 mas against a 0-140 ramp: every cell rendered near-black and the real
    # structure was invisible.  Stretch to the largest measured cell (+10% headroom), floored so an
    # essentially-perfect field still gets a usable ramp and capped so one wild cell cannot flatten
    # the rest; the 75 mas gate is drawn ON the colourbar, where it is legible for free.
    # CONTIGUOUS grid (imshow), so the cells tile with no whitespace and a coherent patch is
    # obvious.  Use the TRUE grid edges (the same linspace as _cell_offsets): edges reconstructed
    # from the surviving cell centres would mis-size the extent when an interior cell is skipped.
    from matplotlib.patches import Rectangle
    # Draw on the grid that ACTUALLY measured the offsets (4x4 / 2x2 / 1x1 whole-field).  A
    # hardcoded 4x4 paints a 2x2-fallback field (sickle) into one quarter with 12/16 cells grey
    # "never measured", and shows a 1x1 field as 94% unmeasured when 100% was measured (#13).
    NCELL = grid_used or 4
    ra_all = jsc.ra.deg; dec_all = jsc.dec.deg
    if float(np.nanmax(ra_all) - np.nanmin(ra_all)) > 180.0:        # RA-wrap guard (matches source)
        NCELL = 1
    re_ = np.linspace(np.nanmin(ra_all), np.nanmax(ra_all), NCELL + 1)
    de_ = np.linspace(np.nanmin(dec_all), np.nanmax(dec_all), NCELL + 1)
    grid = np.full((NCELL, NCELL), np.nan)
    for c in cells:
        grid[c["i"], c["j"]] = c["off"]
    ext = [re_[0], re_[-1], de_[0], de_[-1]]
    import matplotlib as mpl
    # Two DIFFERENT empty states, drawn differently so the map is not a sea of grey:
    #  * a cell with no JWST sources (outside the mosaic footprint -- most of a diagonal strip's
    #    bounding box) is NOT a measurement failure and stays WHITE;
    #  * a cell that HAD sources but produced no usable peak (in `dropped`, incl. spurious-rejected)
    #    is a genuine "unmeasured" square and is drawn GREY below.  These are rare when they occur.
    cmap = mpl.colormaps["inferno"].copy(); cmap.set_bad("white")
    gmax = float(np.nanmax(grid)) if np.isfinite(grid).any() else 0.0
    # Scale the colour ramp to the DATA (+10% headroom), so a uniform field near 150 mas is not
    # clipped flat against a fixed cap and pegged to the top of the bar; keep a high sanity ceiling.
    vmax = min(3.0 * aa.THRESH["absolute"], max(10.0, 1.1 * gmax))
    im0 = a0.imshow(grid.T, origin="lower", extent=ext, aspect="auto", cmap=cmap,
                    vmin=0, vmax=vmax)
    for d in dropped:                               # grey ONLY the had-sources-but-no-peak cells
        di, dj = d["i"], d["j"]
        if not (0 <= di < NCELL and 0 <= dj < NCELL):
            continue                                # a coarser-grid drop cannot be placed on this grid
        a0.add_patch(Rectangle((re_[di], de_[dj]), re_[di + 1] - re_[di], de_[dj + 1] - de_[dj],
                               facecolor="0.7", edgecolor="none", zorder=1))
    cb0 = fig.colorbar(im0, ax=a0, label="JWST−VIRAC offset in cell [mas]", shrink=0.85)
    if aa.THRESH["absolute"] <= vmax:
        cb0.ax.axhline(aa.THRESH["absolute"], color="#39ff14", lw=1.6)
        cb0.ax.text(1.6, aa.THRESH["absolute"], " 75 mas gate", color="#2a7d0f", fontsize=6,
                    va="center", transform=cb0.ax.get_yaxis_transform())
    else:
        cb0.ax.set_title(f"gate 75\n({aa.THRESH['absolute'] / max(vmax, 1e-9):.0f}× top)",
                         fontsize=5.5, color="0.35", pad=3)
    # RED outline on confirmed-deviating cells (deviating = bad), on the true grid edges
    for k, c in enumerate(cells):
        if confirmed[k]:
            a0.add_patch(Rectangle((re_[c["i"]], de_[c["j"]]),
                                   re_[c["i"] + 1] - re_[c["i"]], de_[c["j"] + 1] - de_[c["j"]],
                                   fill=False, ec="#e41a1c", lw=2.0, zorder=3))
    a0.set_xlabel("RA [deg]"); a0.set_ylabel("Dec [deg]"); a0.invert_xaxis()
    _nsp = cc.get("n_spurious", 0)
    _dropnote = f"{cc['n_dropped']} unmeasured" + (f" ({_nsp} spurious)" if _nsp else "")
    a0.set_title(f"JWST−VIRAC offset per cell ({_dataset_label(metrics)}): {cc['n_cells']} measured, "
                 f"{_dropnote}\nfield {cell_off_med:.0f} mas; {cc['n_confirmed']} cells "
                 f"({100 * cc['bad_src_frac']:.0f}% of sources) deviate", fontsize=8)
    # panel 2: the same per-cell offsets as (dRA, dDec) points sized by source count (big = more
    # sources = more weight in the median), the field value, and the 75 mas gate.
    a1 = ax[0][col]; col += 1
    # sized by source count; HOLLOW circles so overlapping cells at similar (ΔRA,ΔDec) are both
    # visible instead of one hiding the other.
    sz = 30 + 130 * np.array([c["n"] for c in cells]) / max(c["n"] for c in cells)
    # Colour each point by WHERE ON THE SKY its cell sits, so a coherent sub-region reads as a
    # colour clump here instead of being an anonymous dot: the question this panel has to answer
    # is "is the spread random, or is one part of the mosaic pulled?".  Quadrant of the 4x4 grid,
    # named on sky (RA above the field centre = East, Dec above = North) and oriented to match
    # panel 1, which has RA inverted.
    QCOL = {"NE": "#4477aa", "NW": "#228833", "SE": "#ccbb44", "SW": "#aa3377"}
    ci = np.array([c["i"] for c in cells]); cj = np.array([c["j"] for c in cells])
    quad = np.array([("N" if j >= NCELL / 2 else "S") + ("E" if i >= NCELL / 2 else "W")
                     for i, j in zip(ci, cj)])
    for q in ["NE", "NW", "SE", "SW"]:
        m = (quad == q) & ~deviating
        if m.any():
            a1.scatter(cdra[m], cdde[m], s=sz[m], facecolors="none", edgecolors=QCOL[q],
                       linewidths=1.3, label=q)
    if deviating.any():
        # deviating keeps the red ring (deviating = bad) but stays quadrant-filled, so it is
        # still readable which part of the sky is the one that deviates.
        for q in ["NE", "NW", "SE", "SW"]:
            m = (quad == q) & deviating
            if m.any():
                a1.scatter(cdra[m], cdde[m], s=sz[m], facecolors=QCOL[q], alpha=0.55,
                           edgecolors="#e41a1c", linewidths=1.8,
                           label=f"{q} (deviating)")
    a1.plot(cc["off_dra"], cc["off_dde"], "k+", ms=15, mew=2, label="per-cell field offset")
    # Whole-field crossmatch offsets (validated histogram-stacking): jicama catalogue and the raw
    # MAST L3 catalogue.  A hollow marker = low confidence (contrast/edge failed).  When both land
    # near each other the field is well-registered; a jicama marker far from the MAST one is a
    # jicama-reduction offset the raw data do not share.
    if jic_x is not None:
        a1.plot(jic_x["dra"], jic_x["dde"], "o", mfc=("#3366cc" if jic_x["ok"] else "none"),
                mec="#3366cc", ms=10, mew=1.6,
                label=f"jicama xmatch {jic_x['off']:.0f} mas (C={jic_x['contrast']:.0f})")
    if mast_x is not None:
        a1.plot(mast_x["dra"], mast_x["dde"], "*", mfc=("#ee7733" if mast_x["ok"] else "none"),
                mec="#ee7733", ms=15, mew=1.4,
                label=f"raw MAST xmatch {mast_x['off']:.0f} mas (C={mast_x['contrast']:.0f})")
    a1.axhline(0, color="k", lw=0.4); a1.axvline(0, color="k", lw=0.4); a1.set_aspect("equal")
    # Frame the DATA.  A limit floored at 1.2*THRESH (=90 mas) drew a field whose cells all sit
    # inside 20 mas as a few dots lost inside a 75 mas gate circle that dominated the panel.  Zoom
    # to the cells; draw the gate ring when it is near them, and otherwise say how far outside
    # the view it falls.
    dmax = float(np.max(np.hypot(cdra, cdde))) if len(cells) else 0.0
    # include the crossmatch markers so they are in view, but cap each at 400 mas so an edge-alias
    # (drawn hollow) cannot blow the panel scale.
    for _x in (jic_x, mast_x):
        if _x is not None:
            dmax = max(dmax, min(400.0, float(np.hypot(_x["dra"], _x["dde"]))))
    lim = max(5.0, 1.5 * dmax)
    if aa.THRESH["absolute"] <= 1.05 * lim:
        a1.add_patch(Circle((0, 0), aa.THRESH["absolute"], fill=False, ec="r", ls=":", lw=0.9))
        gate_note = "dotted = 75 mas gate"
    else:
        gate_note = f"75 mas gate is {aa.THRESH['absolute'] / max(lim, 1e-9):.0f}× outside this view"
    # a circle at the cell-to-cell spread IS at the data's scale, so it gives the eye something
    # to judge the scatter against now that the gate ring is usually off-view.
    if spread is not None and spread > 0 and spread <= lim:
        a1.add_patch(Circle((cc["off_dra"], cc["off_dde"]), spread, fill=False, ec="0.45",
                            ls="--", lw=0.8))
    a1.set_xlim(-lim, lim); a1.set_ylim(-lim, lim)
    a1.set_xlabel("ΔRA [mas]"); a1.set_ylabel("ΔDec [mas]")
    a1.legend(fontsize=6, loc="upper right", ncols=2, handletextpad=0.3, columnspacing=0.6)
    # marginal ΔRA / ΔDec histograms of the per-cell offsets, each cell weighted by its source
    # count so the marginals show the same distribution the field median is taken from.  Title
    # goes on the top marginal so it clears the inset histograms.
    a1t, _a1r = _add_marginals(a1, cdra, cdde, color="#4477aa", bins=12,
                               weights=np.array([c["n"] for c in cells], float))
    sp_str = f", cells scatter by {spread:.0f} mas" if spread is not None else ""
    # This panel shows the per-CELL histogram peaks, so its headline is the median over cells.  The
    # same-star measurement of the same quantity is reported beside it.
    ss_str = (f"\nsame stars, matched one to one: {ss['off']:.1f} mas (n={ss['npairs']})" if ss else
              "\nsame-star measurement unavailable; the histogram value is the one reported")
    a1t.set_title(f"JWST−VIRAC offset {cell_off_med:.0f} mas over {cc['n_cells']} cells"
                  f"{sp_str}{ss_str}\n"
                  f"(colour = sky quadrant; point size ∝ sources; dashed circle = cell-to-cell "
                  f"spread; {gate_note}; marginals weighted by source count)",
                  fontsize=7)
    if im:
        # The inter-module offset is two numbers; print them.
        a2 = ax[0][col]; col += 1
        a2.axis("off")
        ok = im["off"] < aa.THRESH["intermodule"]
        a2.text(0.5, 0.62,
                f"NRCA − NRCB\ninter-module offset\n({metrics['intermodule_filt']})",
                ha="center", va="center", fontsize=10, transform=a2.transAxes)
        a2.text(0.5, 0.36,
                f"ΔRA = {im['dra']:+.1f} mas\nΔDec = {im['ddec']:+.1f} mas\n"
                f"|offset| = {im['off']:.1f} mas",
                ha="center", va="center", fontsize=12, transform=a2.transAxes,
                family="monospace")
        a2.text(0.5, 0.14, f"({'≤' if ok else '>'} {aa.THRESH['intermodule']:.0f} mas gate)",
                ha="center", va="center", fontsize=9,
                color=("#2a7" if ok else "#c33"), transform=a2.transAxes)
    fig.suptitle(f"{o.target} {o.obsid} — positional offsets (JWST catalog − VIRAC)",
                 fontsize=11, y=0.99)
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
            T = vstack([Table.read(c) for c in
                        _used_many(cats, f"per-detector daophot, {d} ({filt})")],
                       metadata_conflicts='silent')
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


def _mosaic_covering(o, filt, ra, dec):
    """From the cutout-mosaic candidates (merged / per-module tiles), the path whose footprint
    contains the MOST of the given sky positions, plus that count.  A fixed 'prefer merged' pick can
    land on a tile the overlap stars are not on: cloudef o005's -merged tile covers RA 266.44-266.49
    while the module catalogs (and the A-B overlap zone) sit at 266.65-266.69, so every cutout fell
    off the mosaic and the gallery was empty (issue #38).  Returns (path, n_covered) or (None, 0)."""
    from astropy.io import fits
    from astropy.wcs import WCS
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    if not filt or ra is None or len(ra) == 0:
        return None, 0
    cands = []
    for tag in ("merged", "nrcb", "nrca"):
        for d in (f"{BASE}/{o.field}/{filt}/pipeline", f"{BASE}/{o.field}/images-merged"):
            cands += [p for p in glob.glob(f"{d}/{o.obsid}_t001_nircam_clear-{filt.lower()}-{tag}_i2d.fits")
                      if not any(s in p.lower() for s in ("residual", "model", "resbgsub", "bg_i2d"))]
    seen = set(); cands = [c for c in cands if not (c in seen or seen.add(c))]
    sc = SkyCoord(np.asarray(ra) * u.deg, np.asarray(dec) * u.deg)
    best, best_n = None, 0
    for p in cands:
        try:
            with fits.open(p) as h:
                sci = h["SCI"] if "SCI" in h else h[1]
                wc = WCS(sci.header); H, W = int(sci.header["NAXIS2"]), int(sci.header["NAXIS1"])
        except (OSError, ValueError, KeyError):
            continue
        x, y = wc.world_to_pixel(sc)
        inb = int(np.sum(np.isfinite(x) & np.isfinite(y) & (x >= 0) & (x < W) & (y >= 0) & (y < H)))
        if inb > best_n:
            best, best_n = p, inb
    return best, best_n


def _finite_sc(sc):
    """SkyCoord subset with finite RA/Dec.  NaN centroids crash astropy's KDTree matchers
    (xcorr / search_around_sky / match_to_catalog_sky reject ANY NaN), so every per-detector
    daophot position list must pass through this before a match."""
    return sc[np.isfinite(sc.ra.deg) & np.isfinite(sc.dec.deg)]


def _module_positions(o, filt):
    """(NRCA, NRCB) finite SkyCoords for the A-vs-B comparison, pooled from the per-detector
    daophot cats, plus per-module ``meta``.  The per-detector cats are the PRIMARY source.  A module
    comes back None for one of two reasons -- it is genuinely ABSENT (a single-module obs, e.g.
    sickle = NRCB only), or its centroids are unusable -- and the caller MUST tell them apart: an
    all-NaN module is an astrometry FAILURE to red-flag, and it would otherwise pass as a
    single-module observation.

    Returns ``(a_sc, b_sc, meta)`` where meta[module] = dict(present, n_raw, n_nan, nan_frac,
    dead): ``present`` = cats exist on disk; ``dead`` = cats exist but hold too few finite centroids
    to measure with (astrometry failed); ``nan_frac`` = dropped fraction (flagged when high even
    where the module is still usable)."""
    from astropy.table import vstack, Table

    def pool(dets):
        cats = []
        for d in dets:
            cats += _daophot_glob(o, filt, d)          # obs-scoped
        if not cats:
            return None, dict(present=False, n_raw=0, n_nan=0, nan_frac=0.0, dead=False)
        try:
            T = vstack([Table.read(c) for c in
                        _used_many(cats, f"per-detector daophot, module positions ({filt})")],
                       metadata_conflicts="silent")
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


def _ab_overlap(a_sc, b_sc):
    """How far a star seen in NRCA sits from the same star seen in NRCB, from two module position
    lists and no external catalogue.

    ``off`` is the bulk A->B shift, taken as the ``aa.xcorr`` histogram PEAK.  A
    search_around_sky median would fabricate pairs at this density.  ``rms`` is the scatter left in
    the SAME stars after A is aligned onto B by that peak: ``hypot`` of the two axes' ``mad_std``.

    Two factors separate that from a stage-6 curve, and both have to be applied to compare them:
    ``hypot`` COMBINES the axes where stage 6 divides by sqrt(2) to stay per-axis, and each
    residual is a DIFFERENCE A - B of two independent measurements of one star, so its per-axis
    scatter is already sqrt(2) above a single module's.  All three stage-6 curves are per-axis
    single-measurement quantities, so this number runs 2x above each of them for isotropic scatter
    (measured: a 10.00 mas per-axis single-module error gives 20.00 mas here and 10.00 mas there).
    Aligning A onto B removes two numbers over ~1000 stars, which does not touch that.

    ``rms(offset)`` sits higher than ``sig_pos`` on a real field for a reason that is NOT a
    convention: it is measured against VIRAC and carries VIRAC's ~20 mas per-star error.  Its
    estimator is ``sqrt(mean(r**2))`` (``_binned_rms``) over a residual stage 6 has already divided
    by sqrt(2), which reads 1.00x a per-axis sigma, the same as ``sig_pos``.

    Returns a dict with the offset/scatter/count, the per-star residual arrays (for the
    hexbin + marginals), and a list of overlap-star positions (for the cutout gallery), or None
    when it cannot be measured."""
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    if a_sc is None or b_sc is None or len(a_sc) < 50 or len(b_sc) < 50:
        return None
    xc = aa.xcorr(a_sc, b_sc, maxsep=1.5 * u.arcsec)
    if not (xc and xc["peak_ratio"] >= aa.MIN_PEAK_RATIO and xc["npairs"] >= 100):
        return None
    cosd = float(np.cos(np.radians(np.median(a_sc.dec.deg))))
    a_al = SkyCoord((a_sc.ra.deg + xc["dra"] / 1000.0 / 3600.0 / cosd) * u.deg,
                    (a_sc.dec.deg + xc["ddec"] / 1000.0 / 3600.0) * u.deg)
    # One-to-one match.  In a crowded GC field an 80-mas ball match is many-to-many -- one bright
    # B star pairs with every nearby A star -- so search_around_sky counts PAIRS where the reader
    # expects distinct overlap stars, which is where the ~34k count came from.  Take the nearest B
    # for each A, keep those within 80 mas, then drop duplicate B (keeping the closest A), so every
    # physical star is counted once and fabricated pairs stay out of the scatter.
    idx, sep2d, _ = a_al.match_to_catalog_sky(b_sc)
    keep = sep2d < 0.08 * u.arcsec
    ia = np.where(keep)[0]
    ib = np.asarray(idx)[keep]
    if len(ia) < 20:
        return None
    order = np.argsort(sep2d[ia].arcsec)                 # smallest separation first
    ia, ib = ia[order], ib[order]
    _, first = np.unique(ib, return_index=True)          # one A per B: the closest
    ia, ib = ia[first], ib[first]
    if len(ia) < 20:
        return None
    dra = (a_al[ia].ra - b_sc[ib].ra).to(u.mas).value * cosd
    dde = (a_al[ia].dec - b_sc[ib].dec).to(u.mas).value
    return dict(dra=float(xc["dra"]), dde=float(xc["ddec"]), off=float(xc["off"]),
                rms=float(np.hypot(aa.mad_std(dra), aa.mad_std(dde))),
                n=int(len(ia)), peak_ratio=float(xc["peak_ratio"]),
                dra_arr=dra, dde_arr=dde,
                # per-star matched positions (for the spatial overlap-footprint map)
                ra_arr=b_sc[ib].ra.deg, dec_arr=b_sc[ib].dec.deg,
                pos=[(b_sc[i].ra.deg, b_sc[i].dec.deg) for i in ib[:200]])


def _draw_ab_panel(ax, ovd, title):
    """Draw one A↔B residual panel: a 2-D hexbin of the per-star ΔRA/ΔDec residuals (the same
    stars, aligned by the histogram peak) with ΔRA/ΔDec marginal histograms.  The numeric summary
    goes on the TOP marginal's title, clear of the marginals themselves."""
    dra_a = np.asarray(ovd["dra_arr"], float); dde_a = np.asarray(ovd["dde_arr"], float)
    ax.hexbin(dra_a, dde_a, gridsize=40, bins="log", cmap="cividis", mincnt=1)
    ax.axhline(0, color="w", lw=0.5); ax.axvline(0, color="w", lw=0.5)
    ax.set_xlabel("NRCA−NRCB residual ΔRA [mas]"); ax.set_ylabel("residual ΔDec [mas]")
    lim = max(50.0, 4.0 * ovd["rms"])
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    axt, _axr = _add_marginals(ax, dra_a, dde_a, color="#5566aa", bins=40)
    axt.set_title(f"{title}  ({ovd['n']} stars)\n"
                  f"offset = {ovd['off']:.1f} mas   scatter = {ovd['rms']:.1f} mas (ΔRA,ΔDec "
                  f"combined)", fontsize=8)


def _draw_ab_footprint(ax, ovd, label):
    """Sky scatter of the A↔B overlap stars, coloured by per-star |A−B| offset.  Shows WHERE the
    genuinely-shared stars are -- they trace the thin NRCA∩NRCB dither-overlap strip -- and whether
    the two modules agree less well in some part of it."""
    import matplotlib.pyplot as plt
    ra = np.asarray(ovd["ra_arr"], float); dec = np.asarray(ovd["dec_arr"], float)
    diff = np.hypot(np.asarray(ovd["dra_arr"], float), np.asarray(ovd["dde_arr"], float))
    vmax = float(np.nanpercentile(diff, 95)) if len(diff) else 1.0
    sc = ax.scatter(ra, dec, c=diff, s=6, cmap="viridis", vmin=0, vmax=max(vmax, 1.0),
                    linewidths=0)
    # Data-driven aspect: the overlap is a thin, long strip, and a full-width row with 'auto'
    # aspect fills the panel so the per-star colour is readable ('equal' rendered it as a sliver).
    ax.invert_xaxis(); ax.set_aspect("auto"); ax.margins(0.02)
    ax.set_xlabel("RA [deg]"); ax.set_ylabel("Dec [deg]")
    plt.colorbar(sc, ax=ax, label="per-star residual |A−B| [mas]", shrink=0.85)
    ax.set_title(f"A↔B overlap footprint — {label} ({len(ra)} stars)\n"
                 f"colour = per-star |A−B|; these stars trace the module-overlap strip",
                 fontsize=8)


def _module_hi_sn(o, filt, snmin=10.0):
    """(NRCA, NRCB) finite SkyCoords restricted to flux S/N > ``snmin`` (S/N = flux_fit/flux_err
    from the per-exposure PSF fit), pooled from the per-detector daophot cats.  Used for the
    stage-5 high-S/N overlap panel, where the residual scatter measures how well the two modules
    agree rather than how well a faint star's centroid is known.  Either module is None if its cats
    lack flux errors or hold too few high-S/N stars."""
    from astropy.table import vstack, Table

    def pool(dets):
        cats = []
        for d in dets:
            cats += _daophot_glob(o, filt, d)          # obs-scoped
        tabs = []
        for c in cats:
            try:
                t = Table.read(_used(c, f"per-detector daophot, S/N cut ({filt})"))
            except (OSError, ValueError):
                continue
            if {"skycoord_centroid", "flux_fit", "flux_err"}.issubset(set(t.colnames)):
                tabs.append(t["skycoord_centroid", "flux_fit", "flux_err"])
        if not tabs:
            return None
        T = vstack(tabs, metadata_conflicts="silent")
        sc = T["skycoord_centroid"]
        f = np.asarray(T["flux_fit"], float); e = np.asarray(T["flux_err"], float)
        with np.errstate(invalid="ignore", divide="ignore"):
            sn = f / e
        good = (np.isfinite(sc.ra.deg) & np.isfinite(sc.dec.deg) & np.isfinite(sn) & (sn > snmin))
        return sc[good] if int(good.sum()) >= 50 else None

    return pool(["nrca1", "nrca2", "nrca3", "nrca4"]), pool(["nrcb1", "nrcb2", "nrcb3", "nrcb4"])


def stage5_intermodule(o: Observation, sw):
    """How well the detectors and the two modules agree on where a star is:
    (1) per-detector residual quiver vs VIRAC, field offset removed, so the arrows are relative;
    (2) NRCA vs NRCB in their overlap, using no external catalog: offset + scatter of the SAME
        stars;
    (3) cutout gallery on the module-overlap zone, where disagreement shows as a split PSF."""
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    from astropy.io import fits
    from astropy.wcs import WCS
    from astropy.nddata import Cutout2D
    filt = sw
    metrics = dict(stage=5, filt=filt)
    ref = _viraccache_path(o) or _refcat_path(o)
    mpath = _cutout_mosaic(o, filt)                       # full mosaic for the cutout gallery
    ep = aa.epoch_of(_used(mpath, f"{filt} mosaic (cutout gallery)")) if mpath else None
    ref_sc, _ = (aa.load_reference(_used(ref, "VIRAC2/Gaia reference catalogue"), ep)
                 if (ref and ep) else (None, None))

    # (2) A vs B overlap from the per-detector cats (primary source), using no external catalog.
    # The bulk A-B offset is the peak of the pair-separation histogram (aa.xcorr).  A direct
    # search_around_sky+median fabricates pairs at this density -- 400k chance coincidences within
    # 0.3", which blows the scatter up to ~100 mas.  The scatter reported is what is left in the
    # SAME stars: align A onto B by the peak, keep the tight matches, measure their spread.
    ov = None
    single_module = None
    a_sc, b_sc, minfo = _module_positions(o, filt)
    metrics["nan_frac"] = round(max(minfo["a"]["nan_frac"], minfo["b"]["nan_frac"]), 4)
    # A module whose cats exist on disk but hold too few finite centroids is an astrometry
    # FAILURE (dead).  Surface it loudly; left alone it reads as a single-module obs and passes.
    dead_module = next((mod for mod, sc, k in (("NRCA", a_sc, "a"), ("NRCB", b_sc, "b"))
                        if sc is None and minfo[k]["present"] and minfo[k]["dead"]), None)
    if dead_module:
        k = "a" if dead_module == "NRCA" else "b"
        png = _red_flag_figure(o, "stage5", f"{dead_module} ASTROMETRY FAILED",
                               f"{dead_module} has per-exposure catalogs but "
                               f"{minfo[k]['nan_frac'] * 100:.0f}% NaN centroids (<50 usable), so "
                               f"NRCA cannot be compared with NRCB. This is an astrometry "
                               f"failure in {dead_module}.")
        metrics.update(red_flag=True, red_flag_reason=f"{dead_module} centroids unusable (NaN)",
                       dead_module=dead_module, passed=False)
        return png, metrics
    if (a_sc is None) ^ (b_sc is None):
        single_module = "NRCA" if a_sc is not None else "NRCB"
    ov = _ab_overlap(a_sc, b_sc)
    if ov:
        metrics.update(intermodule_off=ov["off"], intermodule_rms=ov["rms"], n_overlap=ov["n"])
    # The same comparison restricted to flux S/N > 10 (the best-measured stars), where the
    # residual scatter measures how well the two modules agree rather than how well a faint star's
    # centroid is known.  This adds a panel; the all-stars panel above stays.
    ov_hi = None
    if ov:
        a_hi, b_hi = _module_hi_sn(o, filt, snmin=10.0)
        ov_hi = _ab_overlap(a_hi, b_hi)
        if ov_hi:
            metrics.update(intermodule_off_hi=ov_hi["off"], intermodule_rms_hi=ov_hi["rms"],
                           n_overlap_hi=ov_hi["n"], sn_cut=10.0)

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
    # (single module, or unmeasurable) the A/B half + cutout gallery are OMITTED and the figure is
    # the per-detector quiver alone; an empty panel would only add noise.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def _draw_quiver(axq):
        if det:
            xs = np.array([v["ra"] for v in det.values()], float)
            ys = np.array([v["dec"] for v in det.values()], float)
            us = np.array([v["rdra"] for v in det.values()], float)
            vs = np.array([v["rdde"] for v in det.values()], float)
            cols = ["#4477aa" if d.startswith("nrca") else "#ee6677" for d in det]
            # ADAPTIVE scale (mas per data-degree): size the largest arrow to ~25% of the field
            # span, instead of a fixed scale=2000 tuned for a full-mosaic FOV.  On a small subarray
            # field (sickle sub640 ~0.008 deg) that fixed scale drew ~9" arrows from the corner
            # detectors that ran clean off the axes (issue #13, "all vectors out of frame").
            mag = np.hypot(us, vs); maxmag = float(np.nanmax(mag)) if mag.size else 1.0
            spanx = float(np.ptp(xs)) or 1e-3; spany = float(np.ptp(ys)) or 1e-3
            span = max(spanx, spany)
            # Scale (mas per data-degree).  A PURELY adaptive scale (largest arrow = 25% of the span)
            # would render a 0.5 mas and a 500 mas residual identically, so floor the scale at a
            # FIXED 15 mas (= THRESH['intermodule']) reference: residuals below the gate stay
            # visibly small, and only larger ones grow (and are then capped to 25% of the span so
            # they cannot run off the axes, the sub640 failure in #13).
            ref = float(aa.THRESH["intermodule"]) / (0.25 * span)     # 15 mas -> ~25% of the span
            scale = max(maxmag / (0.25 * span), ref, 1e-6)
            q = axq.quiver(xs, ys, us, vs, color=cols, angles="xy", scale_units="xy",
                           scale=scale, width=0.007)
            # key is the fixed 15 mas ruler, so arrow length is comparable across observations
            axq.quiverkey(q, 0.5, 0.06, float(aa.THRESH["intermodule"]),
                          f"{aa.THRESH['intermodule']:g} mas (gate)", labelpos="E",
                          coordinates="axes", fontproperties={"size": 8})
            for d, v in det.items():
                # annotate each arrow with the number of VIRAC-matched stars behind it
                axq.annotate(f"{d} (n={v['n']})", (v["ra"], v["dec"]), fontsize=5.8,
                             ha="center", va="bottom")
            # EXPAND the axes to contain the arrow TIPS (+ margin) so no vector leaves the frame.
            tipx = xs + us / scale; tipy = ys + vs / scale
            allx = np.concatenate([xs, tipx]); ally = np.concatenate([ys, tipy])
            mx = 0.18 * max(float(np.ptp(allx)), 1e-3); my = 0.18 * max(float(np.ptp(ally)), 1e-3)
            axq.set_xlim(allx.min() - mx, allx.max() + mx)
            axq.set_ylim(ally.min() - my, ally.max() + my)
            axq.invert_xaxis(); axq.set_ylabel("Dec"); axq.set_xlabel("RA")
            # 'A−B diff' only means something with BOTH modules; a single-module field has no NRCA
            # to difference, so drop the 'A−B diff = nan' clause there.
            two_module = (any(d.startswith("nrca") for d in det) and
                          any(d.startswith("nrcb") for d in det))
            abdiff = (f"A−B diff = {metrics['intermodule_diff']:.1f} mas  ·  "
                      if two_module and metrics.get("intermodule_diff") is not None else "")
            axq.set_title(f"per-detector residual vs VIRAC (bulk-removed) — {filt}\n"
                          f"{abdiff}one arrow/detector (n = VIRAC matches)", fontsize=9, pad=12)
        else:
            axq.text(0.5, 0.5, "per-detector cats unavailable", ha="center", va="center", fontsize=8)

    if ov:
        # Top row: per-detector quiver | A↔B overlap (all stars) | A↔B (S/N>10, if measurable) |
        # A↔B overlap FOOTPRINT (sky map coloured by |A−B|, high-S/N).  Bottom row: cutout gallery
        # spanning the width.
        # Row 0: per-detector quiver | A↔B overlap (all stars) | A↔B (S/N>10, if measurable).
        # Row 1: the A↔B overlap FOOTPRINT spanning the full width (the overlap is a thin, long
        # strip, so a full-width row with data-driven aspect makes the per-star colour readable).
        # Row 2: cutout gallery spanning the full width.
        ncols = 3 if ov_hi else 2
        fig = plt.figure(figsize=(5.0 * ncols + 1.0, 11.5))
        gs = fig.add_gridspec(3, ncols, height_ratios=[1.25, 0.55, 0.75], hspace=0.7, wspace=0.62)
        fig.subplots_adjust(top=0.88, bottom=0.05, left=0.06, right=0.97)
        axq = fig.add_subplot(gs[0, 0]); _draw_quiver(axq)
        axo = fig.add_subplot(gs[0, 1]); _draw_ab_panel(axo, ov, "A↔B overlap — all stars")
        if ov_hi:
            axh = fig.add_subplot(gs[0, 2])
            _draw_ab_panel(axh, ov_hi, "A↔B overlap — flux S/N > 10")
        # footprint map (full-width row): prefer the high-S/N set (cleaner), else all stars
        axfp = fig.add_subplot(gs[1, :])
        fp_ov = ov_hi if ov_hi else ov
        _draw_ab_footprint(axfp, fp_ov, "S/N > 10" if ov_hi else "all stars")
        metrics["n_overlap_footprint"] = int(len(fp_ov["ra_arr"]))

        # (3) doubled-star cutout gallery.  Pick the mosaic that actually COVERS the overlap-star
        # positions (a -merged tile can cover a different sub-region than the module catalogues), not
        # a fixed 'prefer merged' that may leave every cutout off-frame (issue #38 cloudef o005).
        ncut = 6
        _ovpos = np.asarray(ov["pos"]) if ov.get("pos") is not None and len(ov["pos"]) else None
        cut_path, _ncov = (_mosaic_covering(o, filt, _ovpos[:, 0], _ovpos[:, 1])
                           if _ovpos is not None else (None, 0))
        cut_path = cut_path or mpath                       # fall back to the default pick
        if cut_path and os.path.exists(cut_path):
            mpath = cut_path
            with fits.open(_used(mpath, f"{filt} mosaic (cutout gallery)")) as hdul:
                sci = hdul["SCI"] if "SCI" in hdul else hdul[1]
                data = sci.data.astype("float32"); w = WCS(sci.header)
            from astropy.coordinates import SkyCoord
            from astropy.visualization import ZScaleInterval, ImageNormalize, AsinhStretch
            # Space the gallery out: pos is already one entry per distinct overlap star, but greedily
            # keep only stars >0.5" apart so the 6 cutouts sample different parts of the strip rather
            # than clustering on one bright clump.
            picks = []
            for ra, dec in ov["pos"][:2000]:
                if picks:
                    prev = SkyCoord([p[0] for p in picks] * u.deg, [p[1] for p in picks] * u.deg)
                    if SkyCoord(ra * u.deg, dec * u.deg).separation(prev).arcsec.min() < 0.5:
                        continue
                picks.append((ra, dec))
                if len(picks) >= ncut * 3:      # gather extras; some cutouts fail the finite check
                    break
            # Collect the VALID cutouts FIRST, then lay out exactly as many axes as we can draw.
            # (Pre-creating ncut inset axes and filling only some left blank white boxes on any
            # field with fewer than ncut usable cutouts -- a real bug seen on issue #38.)
            cuts = []
            for ra, dec in picks:
                if len(cuts) >= ncut:
                    break
                try:
                    x, y = w.world_to_pixel(SkyCoord(ra * u.deg, dec * u.deg))
                    cut = Cutout2D(data, (float(x), float(y)), 25, wcs=w)
                except (ValueError, IndexError):
                    continue
                if not np.isfinite(cut.data).any() or np.nanmax(cut.data) <= 0:
                    continue
                cuts.append(cut.data)
            shown = len(cuts)
            strip = fig.add_subplot(gs[2, :]); strip.axis("off")
            if shown:
                n = shown
                for i, cdata in enumerate(cuts):
                    a = strip.inset_axes([i / n + 0.01, 0.05, 0.92 / n, 0.85])
                    norm = ImageNormalize(cdata, interval=ZScaleInterval(), stretch=AsinhStretch())
                    a.imshow(cdata, origin="lower", cmap="gray", norm=norm)
                    a.set_xticks([]); a.set_yticks([]); a.set_title(f"{i + 1}", fontsize=7)
            else:
                # Distinguish "no drizzled mosaic covers the overlap zone" (a footprint mismatch --
                # the module catalogues and the mosaics are on disjoint sky, a real reduction problem)
                # from a generic empty result, so the message is a diagnostic, not a shrug.
                msg = ("the drizzled mosaic does not cover the module-overlap zone — the catalogue "
                       "and the mosaic are on disjoint footprints (reduction mismatch)"
                       if _ncov == 0 else "no usable overlap-star cutouts on the mosaic")
                strip.text(0.5, 0.5, msg, ha="center", va="center", fontsize=9, style="italic")
                metrics["cutout_footprint_mismatch"] = bool(_ncov == 0)
            from astropy.wcs.utils import proj_plane_pixel_scales
            pscale = float(np.mean(proj_plane_pixel_scales(w))) * 3600.0     # arcsec/pix from WCS
            fig.text(0.5, 0.02,
                     f"{shown} star{'s' if shown != 1 else ''} from the NRCA∩NRCB overlap of the "
                     f"{filt} merged mosaic (each detected in BOTH modules; 25 px ≈ "
                     f"{25 * pscale:.1f}\").  Where the two modules agree the star is one round "
                     f"PSF; where they disagree it doubles or elongates.",
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

    # a single-module obs (sickle = NRCB only) has no A/B comparison to fail -> N/A passes.
    if single_module:
        metrics["single_module"] = single_module
    nan_frac = metrics.get("nan_frac", 0.0)
    high_nan = nan_frac > _NAN_FRAC_FLAG        # usable but degraded -> surface + don't pass
    if high_nan:
        title_extra += f"  ·  ⚠ {nan_frac * 100:.0f}% NaN centroids"
    metrics["passed"] = bool((single_module or (ov and ov["off"] < aa.THRESH["intermodule"]))
                             and not high_nan)
    fig.suptitle(f"{o.target} {o.obsid} — inter-detector / inter-module agreement ({filt}){title_extra}",
                 fontsize=11, y=suptitle_y)
    return _save(fig, f"{o.obsid}_stage5.png"), metrics


# --------------------------------------------------------------------------- STAGE 9 (PSF vs aper)
def _psf_flux_positions(o, filt):
    """(SkyCoord, PSF flux_fit, catalog basename) from the highest-tier jicama catalog carrying
    ``flux_<filt>`` + ``skycoord_<filt>``, or (None, None, None)."""
    from astropy.io import fits
    from astropy.table import Table
    fcol = f"flux_{filt.lower()}"; sccol = f"skycoord_{filt.lower()}"
    best_path = None; best_rank = (-1.0, -1.0)
    for p, kind, tier, mtime in _catalog_candidates(o):
        if (tier, mtime) <= best_rank:
            continue
        # cheap TTYPE header probe before the (large) full read -- match _catalog_for's pattern
        try:
            hdr = fits.getheader(p, ext=1)
        except (OSError, IndexError):
            continue
        low = {str(hdr.get(f"TTYPE{i}", "")).lower() for i in range(1, hdr.get("TFIELDS", 0) + 1)}
        if fcol in low and f"{sccol}.ra" in low:      # skycoord mixin serializes as "<name>.ra"
            best_path = p; best_rank = (tier, mtime)
    if best_path is None:
        return None, None, None
    try:
        t = Table.read(_used(best_path, f"PSF-flux catalogue ({filt})"))
    except (OSError, ValueError):
        return None, None, None
    sc = t[sccol]; flux = np.asarray(t[fcol], float); name = os.path.basename(best_path)
    g = np.isfinite(sc.ra.deg) & np.isfinite(sc.dec.deg) & np.isfinite(flux) & (flux > 0)
    return (sc[g], flux[g], name) if int(g.sum()) >= 50 else (None, None, None)


# Minimum aperture SNR for a star to enter the PSF-vs-aperture comparison.  Set high: the point is
# a CLEAN locus of well-measured stars, not completeness, and in a crowded field a low-SNR aperture
# is background-dominated and scatters by magnitudes.
_APER_SNR_MIN = 30.0


def _recentroid_com(data, x, y, box=11):
    """Snap each (x, y) to the local intensity centroid within a ``box``x``box`` stamp (background
    removed as the stamp minimum).  The catalog positions can sit several pixels off the star on the
    mosaic when the catalog and the mosaic are on DIFFERENT frames (a stale catalogue vs a re-tied
    mosaic: cloudef's ~150 mas ~= 5 px), which would place a small aperture on blank sky.  Recentring
    within a box larger than the plausible shift but smaller than the isolation radius puts the
    aperture back on the star.  Returns (x_new, y_new, moved_px) with the per-star shift; stars whose
    stamp runs off the image or has no positive flux keep their original position (moved = 0)."""
    half = box // 2
    x = np.asarray(x, float); y = np.asarray(y, float)
    xn, yn = x.copy(), y.copy()
    ny, nx = data.shape
    xi = np.round(x).astype(int); yi = np.round(y).astype(int)
    ok = (xi - half >= 0) & (xi + half < nx) & (yi - half >= 0) & (yi + half < ny)
    if ok.any():
        oy, ox = np.mgrid[-half:half + 1, -half:half + 1]
        stamps = data[yi[ok, None, None] + oy[None], xi[ok, None, None] + ox[None]]
        stamps = np.where(np.isfinite(stamps), stamps, 0.0).astype(float)
        stamps = stamps - stamps.min(axis=(1, 2), keepdims=True)
        w = stamps.sum(axis=(1, 2))
        wpos = w > 0
        cx = np.zeros(int(ok.sum())); cy = np.zeros(int(ok.sum()))
        cx[wpos] = (stamps * ox[None]).sum(axis=(1, 2))[wpos] / w[wpos]
        cy[wpos] = (stamps * oy[None]).sum(axis=(1, 2))[wpos] / w[wpos]
        xn[ok] = np.where(wpos, xi[ok] + cx, x[ok])
        yn[ok] = np.where(wpos, yi[ok] + cy, y[ok])
    moved = np.hypot(xn - x, yn - y)
    return xn, yn, moved


def stage9_psf_vs_aper(o: Observation, sw, r_ap=3.0, r_in=6.0, r_out=9.0, iso_px=12.0, maxn=20000):
    """PSF vs aperture photometry check.  The jicama catalog reports PSF-fit fluxes; here we
    RE-MEASURE simple aperture photometry (local-annulus background) on the mosaic at the catalog
    positions and compare.  Restricted to ISOLATED stars (nearest catalog neighbour > ``iso_px``
    px) so a neighbour's light doesn't contaminate the aperture -- a cheap stand-in until the
    pipeline emits aperture catalogs.  A tight (aper−psf) locus at a constant offset (the aperture
    correction) means the two photometries agree; curvature/scatter flags PSF-model or
    crowding problems."""
    import matplotlib
    matplotlib.use("Agg")
    from astropy.io import fits
    from astropy.wcs import WCS
    from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry, ApertureStats
    metrics = dict(stage=9, sw=sw, r_ap_px=r_ap, iso_px=iso_px)
    sc, psf_flux, src = _psf_flux_positions(o, sw)
    mpath = _mosaic_path(o, sw)
    if sc is None or not mpath:
        reason = ("no jicama catalog with a PSF flux column for this filter" if sc is None
                  else "no mosaic on disk to measure aperture photometry on")
        png = _red_flag_figure(o, "stage9", "PSF-vs-APER UNMEASURABLE",
                               f"Cannot compare PSF vs aperture photometry: {reason}.")
        metrics.update(red_flag=True, red_flag_reason=reason, passed=False)
        return png, metrics
    metrics["catalog"] = src
    try:
        with fits.open(_used(mpath, f"{sw} mosaic (aperture photometry)")) as h:
            sci = h["SCI"] if "SCI" in h else h[1]
            data = sci.data.astype("float32"); w = WCS(sci.header)
    except (OSError, ValueError, KeyError):
        png = _red_flag_figure(o, "stage9", "MOSAIC UNREADABLE", f"Could not read {mpath}.")
        metrics.update(red_flag=True, red_flag_reason="mosaic unreadable", passed=False)
        return png, metrics
    x, y = w.world_to_pixel(sc)
    ny, nx = data.shape
    inb = (x > r_out + 1) & (x < nx - r_out - 1) & (y > r_out + 1) & (y < ny - r_out - 1)
    # ISOLATED: nearest catalog neighbour farther than iso_px (exclude blended stars)
    from scipy.spatial import cKDTree
    xy = np.c_[x, y]
    nn = cKDTree(xy).query(xy, k=2)[0][:, 1]
    keep = inb & (nn > iso_px)
    if int(keep.sum()) < 50:
        png = _red_flag_figure(o, "stage9", "TOO FEW ISOLATED STARS",
                               f"Only {int(keep.sum())} isolated (>{iso_px:.0f} px) in-bounds stars "
                               f"— not enough for a PSF-vs-aperture comparison.")
        metrics.update(red_flag=True, red_flag_reason="too few isolated stars", passed=False)
        return png, metrics
    idx = np.where(keep)[0]
    if idx.size > maxn:                         # cap cost: brightest maxn isolated stars
        idx = idx[np.argsort(psf_flux[idx])[::-1][:maxn]]
    # Snap to the star on the MOSAIC before measuring: the catalog can be on a different frame than
    # the mosaic (stale catalogue vs re-tied mosaic), which would otherwise put the aperture on blank
    # sky and manufacture a huge, scattered aperture-minus-PSF "offset" (cloudef o005).  The box
    # (11 px) exceeds a plausible frame shift and stays inside the isolation radius, so no CATALOGUED
    # neighbour sits in it -- but uncatalogued flux can still pull the centroid several px in a
    # crowded field, which the ``moved < 3`` quality gate below keeps out of the comparison.
    xr, yr, moved = _recentroid_com(data, x[idx], y[idx], box=11)
    # med_shift is measured over ALL isolated stars (including the far-moved ones the gate will drop),
    # so a genuine whole-frame shift still shows; in a crowded field the uncatalogued-flux drag makes
    # it an UPPER BOUND on the true catalog-vs-mosaic registration offset, not an exact value.
    med_shift = float(np.median(moved)) if len(moved) else 0.0
    metrics["recentroid_shift_px_med"] = med_shift
    metrics["catalog_mosaic_mismatch"] = bool(med_shift > 2.0)   # ~half the F162M PSF FWHM
    pos = np.c_[xr, yr]
    ap = CircularAperture(pos, r=r_ap)
    ann = CircularAnnulus(pos, r_in=r_in, r_out=r_out)
    # ApertureStats is NaN-aware -- feed it the RAW data so a coverage-gap NaN in the annulus is
    # ignored (nan_to_num(0) would bias the median low).  The aperture SUM needs finite pixels, so
    # zero-fill only there; the `good` mask below drops any star whose aperture still went bad.
    apstats = ApertureStats(data, ann)                       # NaN-aware annulus stats
    bkg = apstats.median                                     # local background per pixel
    bkg_std = np.asarray(apstats.std, float)                 # its scatter (inflated by any neighbour)
    raw = aperture_photometry(np.nan_to_num(data, nan=0.0), ap)["aperture_sum"]
    aper_flux = np.asarray(raw, float) - np.asarray(bkg, float) * ap.area
    # DROP apertures that contain a NaN pixel (usually a saturated core or coverage gap): zero-fill
    # would read them too faint by ~0.15 mag with no flag.  Better excluded than silently biased.
    nanmask = (~np.isfinite(data)).astype("float32")
    nan_in_ap = np.asarray(aperture_photometry(nanmask, ap)["aperture_sum"], float) > 0.5
    metrics["n_aper_with_nan"] = int(nan_in_ap.sum())
    pf = psf_flux[idx]
    # STAR-QUALITY GATE.  A simple aperture is only meaningful on a star that dominates its aperture:
    # in a crowded field the annulus fills with neighbour light (large bkg_std), the background
    # subtraction goes wrong, and aper-minus-PSF scatters by magnitudes even for nominally "isolated"
    # stars (cloudef o005: 2 mag).  Keep only stars whose aperture flux is well above the local
    # background noise (SNR) and whose recentroid landed confidently (a large move = blended/ambiguous).
    aper_noise = bkg_std * np.sqrt(float(ap.area))
    with np.errstate(invalid="ignore", divide="ignore"):
        snr = aper_flux / aper_noise
    metrics["aper_snr_min"] = float(_APER_SNR_MIN)
    good = (np.isfinite(aper_flux) & (aper_flux > 0) & np.isfinite(pf) & (pf > 0) & (~nan_in_ap)
            & np.isfinite(snr) & (snr > _APER_SNR_MIN) & (moved < 3.0))
    metrics["n_after_quality_gate"] = int(good.sum())
    if int(good.sum()) < 50:                                 # gate removed too many -> not measurable
        png = _red_flag_figure(o, "stage9", "TOO FEW CLEAN STARS",
                               f"Only {int(good.sum())} isolated stars pass the aperture SNR/"
                               f"recentroid quality gate (crowding or a catalog-mosaic frame "
                               f"mismatch) — not enough for a PSF-vs-aperture comparison.")
        metrics.update(red_flag=True, red_flag_reason="too few clean stars after quality gate",
                       passed=False)
        return png, metrics
    m_psf = -2.5 * np.log10(pf[good]); m_aper = -2.5 * np.log10(aper_flux[good])
    dmag = m_aper - m_psf
    apcorr = float(np.median(dmag)); scat = float(aa.mad_std(dmag))
    # tail fraction: the disagreement population mad_std is blind to.  Gate on BOTH the core
    # scatter and this tail so a PSF-model failure in a minority can't read green.
    tail_frac = float(np.mean(np.abs(dmag - apcorr) > 0.3)) if len(dmag) else 1.0
    metrics.update(n_isolated=int(good.sum()), aper_corr_med=apcorr, aper_psf_scatter=scat,
                   frac_gt_0p3mag=tail_frac, n_capped=bool(idx.size >= maxn),
                   passed=bool(scat < 0.15 and tail_frac < 0.05))
    import matplotlib.pyplot as plt
    fig, ax = _fig(1, 2, 5.6, 5.0)
    fig.subplots_adjust(wspace=0.32, top=0.86, bottom=0.12)
    a0 = ax[0][0]
    hb = a0.hexbin(m_psf, m_aper, gridsize=60, bins="log", cmap="viridis", mincnt=1)
    fig.colorbar(hb, ax=a0, label="log N", shrink=0.85)
    lo, hi = np.nanpercentile(np.r_[m_psf, m_aper], [1, 99])
    a0.plot([lo, hi], [lo + apcorr, hi + apcorr], "r-", lw=1.2,
            label=f"1:1 + aper.corr ({apcorr:+.2f})")
    a0.set_xlim(lo, hi); a0.set_ylim(lo, hi); a0.set_aspect("equal")
    a0.set_xlabel("PSF instrumental mag"); a0.set_ylabel("aperture instrumental mag")
    a0.legend(fontsize=8, loc="upper left")
    a0.set_title(f"PSF vs aperture ({_dataset_label({'source': 'release:' + src})})", fontsize=9)
    a1 = ax[0][1]
    hb2 = a1.hexbin(m_psf, dmag, gridsize=60, bins="log", cmap="magma", mincnt=1)
    fig.colorbar(hb2, ax=a1, label="log N", shrink=0.85)
    a1.axhline(apcorr, color="c", lw=1.2, label=f"median {apcorr:+.2f}")
    # show the FULL residual range (don't crop the disagreement tail out of the plot that exists to
    # surface it); pad the data range a little.
    dlo, dhi = np.nanpercentile(dmag, [0.2, 99.8])
    a1.set_ylim(min(dlo, apcorr - 0.3), max(dhi, apcorr + 0.3))
    a1.set_xlabel("PSF instrumental mag"); a1.set_ylabel("aperture − PSF [mag]")
    a1.legend(fontsize=8, loc="upper right")
    a1.set_title(f"{int(good.sum())} isolated stars (>{iso_px:.0f} px)  "
                 f"scatter {scat:.3f} mag\n{100 * tail_frac:.1f}% beyond ±0.3 mag "
                 f"(r_ap={r_ap:.0f}px)", fontsize=9)
    fig.suptitle(f"{o.target} {o.obsid} — PSF vs aperture photometry ({sw})", fontsize=11, y=0.98)
    return _save(fig, f"{o.obsid}_stage9.png"), metrics


# --------------------------------------------------------------------------- STAGE 10
# Jay Anderson's JWST1PASS across-exposure consistency product.  jwst1pass fits every _cal frame
# with an empirical library PSF (STDPSF) + geometric-distortion correction (STDGDC), the per-chip
# .xympqsuvw catalogues are combined into a per-exposure META frame, matched across exposures
# (xym2mat) and collated (xym2bar) into MATCHUP.XYMEEE: one row per star carrying the mean position
# and instrumental magnitude in the reference frame and the RMS of each across the exposures it was
# found in.  Those RMS columns are the "is everything in family?" photometric+astrometric check --
# reproduced here as the four panels of Jay's show_matchup.sm (XRMS / YRMS / MAG-RMS / QFIT vs mag).
# Jay's META reference grid is 32 mas/pixel for the F182M (SW) products in hand -- it is the grid
# the map2avg transforms resample every chip onto, the SW native scale.  An LW MATCHUP would be
# built on a coarser grid, so this SW constant would make its position RMS ~2x too small; the stage
# flags that case (meta_scale_assumed_sw) rather than silently converting, since no LW product
# exists yet to read the true scale from.
_META_PIX_MAS = 32.0
# Products live under the per-field reduction tree; brick and cloudc are on /blue, the rest on
# /orange (the same split the peppar products use).
_JWST1PASS_ROOTS = {"brick": "/blue/adamginsburg/adamginsburg/jwst",
                    "cloudc": "/blue/adamginsburg/adamginsburg/jwst"}
_JWST1PASS_DEFAULT_ROOT = "/orange/adamginsburg/jwst"
# MATCHUP.XYMEEE columns (1-indexed in the header): xbar ybar mbar xsig ysig msig qbar Nf Ng Nm ...
_XYMEEE_COLS = dict(x=0, y=1, m=2, ex=3, ey=4, em=5, q=6, Nf=7, Ng=8, Nm=9)
# jwst1pass "no valid measurement" sentinels: 9.0 on the position/mag RMS columns and 9.999 on
# qfit (a different value -- they must be filtered separately), and mbar == 0 marks an unmeasured
# star.  All pile up as a false floor/ceiling if kept: on the real F182M file 197 rows carry
# q == 9.999, which without the qfit cut draw a hard line at the top of the QFIT panel three orders
# of magnitude from the data (its 99.9th percentile is 0.35).
_XYMEEE_SENTINEL = 9.0
_XYMEEE_QFIT_SENTINEL = 9.999


def _read_matchup_xymeee(path):
    """Parse a JWST1PASS ``MATCHUP.XYMEEE`` into a column dict (numpy arrays), keeping only rows a
    star was actually measured in: found in >= 2 exposures (``Ng``), a real instrumental magnitude
    (``mbar`` < 0), and no RMS/qfit sentinel.  Returns None if the file has no usable rows."""
    cols = tuple(_XYMEEE_COLS[k] for k in ("x", "y", "m", "ex", "ey", "em", "q", "Nf", "Ng", "Nm"))
    arr = np.loadtxt(_used(path, "JWST1PASS MATCHUP.XYMEEE"), usecols=cols, comments="#", ndmin=2)
    if arr.size == 0:
        return None
    x, y, m, ex, ey, em, q, Nf, Ng, Nm = arr.T
    good = (np.isfinite(m) & (m < 0) & (Ng >= 2)
            & (ex < _XYMEEE_SENTINEL) & (ey < _XYMEEE_SENTINEL) & (em < _XYMEEE_SENTINEL)
            & (q < _XYMEEE_QFIT_SENTINEL))
    if int(good.sum()) < 50:
        return None
    keys = ("x", "y", "m", "ex", "ey", "em", "q", "Nf", "Ng", "Nm")
    return {k: v[good] for k, v in zip(keys, (x, y, m, ex, ey, em, q, Nf, Ng, Nm))}


def _jwst1pass_matchup(o: Observation, filt):
    """Path to the ``MATCHUP.XYMEEE`` for this obs's field+filter, or None if jwst1pass has not been
    run for it.  Convention ``{root}/{field}/jwst1pass/{FILT}/MATCHUP.XYMEEE`` (shallow globs only --
    no recursive ``**`` over the products tree).  ``QA_JWST1PASS_DIR`` overrides the lookup wholesale
    (used to point the stage at a one-off product directory)."""
    if not filt:
        return None
    override = os.environ.get("QA_JWST1PASS_DIR")
    roots = [override] if override else [
        f"{_JWST1PASS_ROOTS.get(o.field, _JWST1PASS_DEFAULT_ROOT)}/{o.field}/jwst1pass/{filt}"]
    for base in roots:
        for cand in (f"{base}/MATCHUP.XYMEEE", f"{base}/03.MATCHUP/MATCHUP.XYMEEE"):
            if os.path.isfile(cand):
                return cand
        hits = sorted(glob.glob(f"{base}/*/MATCHUP.XYMEEE"))
        if hits:
            return hits[0]
    return None


def _saturation_turnover(m, sig, floor):
    """Brightest magnitude at which the binned RMS has climbed to 2x the faint-side floor -- the
    onset of the saturation/bright-star degradation Jay notes.  None if it never does."""
    med, _, _, ctr = _binned_stat(m, sig)
    if med is None:
        return None
    over = ctr[(med > 2.0 * floor) & (ctr < np.median(ctr))]     # bright half only
    return float(over.min()) if over.size else None


def stage10_photometric_consistency(o: Observation, sw, lw):
    """JWST1PASS across-exposure consistency (Jay Anderson's ``MATCHUP.XYMEEE``): for every star the
    RMS of its position (X, Y) and instrumental magnitude across the exposures it was found in, plus
    the mean quality-of-fit, all versus instrumental magnitude.  The four panels reproduce Jay's
    ``show_matchup.sm``.  A tight, flat bright-end floor that rises only at the faint (S/N) and
    saturated (bright) ends is "in family"; a raised or structured floor flags a photometric or
    distortion problem in that filter's frames."""
    import matplotlib
    matplotlib.use("Agg")
    metrics = dict(stage=10, sw=sw, lw=lw)
    # jwst1pass is run per filter; prefer the SW channel (Jay's F182M example), fall back to LW.
    filt = next((f for f in (sw, lw) if f and _jwst1pass_matchup(o, f)), None)
    if filt is None:
        reason = "no JWST1PASS MATCHUP.XYMEEE product on disk for this obs/filter"
        png = _red_flag_figure(o, "stage10", "JWST1PASS CONSISTENCY UNAVAILABLE",
                               f"Cannot build the across-exposure consistency panels: {reason}.")
        metrics.update(red_flag=True, red_flag_reason=reason, passed=False)
        return png, metrics
    metrics["filter"] = filt
    d = _read_matchup_xymeee(_jwst1pass_matchup(o, filt))
    if d is None:
        reason = "MATCHUP.XYMEEE has too few multiply-measured stars (Ng>=2) to assess consistency"
        png = _red_flag_figure(o, "stage10", "JWST1PASS CONSISTENCY UNAVAILABLE", reason + ".")
        metrics.update(red_flag=True, red_flag_reason=reason, passed=False)
        return png, metrics

    m = d["m"]
    exm = d["ex"] * _META_PIX_MAS         # position RMS: META pixels -> mas (SW grid, see constant)
    eym = d["ey"] * _META_PIX_MAS
    metrics["n_stars"] = int(m.size)
    metrics["n_exposures"] = int(np.nanmax(d["Nf"]))
    metrics["meta_pix_mas"] = _META_PIX_MAS
    # _META_PIX_MAS is the SW grid scale; an LW MATCHUP would sit on a coarser grid, so flag the
    # position-RMS-in-mas as unconfirmed rather than converting it silently 2x too small.
    if filt.upper() in _LW_PREF:
        metrics["meta_scale_assumed_sw"] = True

    fig, ax = _fig(2, 2, 5.4, 4.3)
    panels = [
        (ax[0][0], exm, "X RMS (mas)", "x_rms_floor_mas", 3.0 * _META_PIX_MAS),
        (ax[0][1], eym, "Y RMS (mas)", "y_rms_floor_mas", 3.0 * _META_PIX_MAS),
        (ax[1][0], d["em"], "magnitude RMS (mag)", "mag_rms_floor", 0.25),
        (ax[1][1], d["q"], "quality of fit", "qfit_floor", 0.25),
    ]
    for a, y, ylab, mkey, ytop in panels:
        a.plot(m, y, ".", ms=1.4, color="#666666", alpha=0.35, rasterized=True)
        med, lo, hi, ctr = _binned_stat(m, y)
        if med is not None:
            a.plot(ctr, med, "-", color="#cc3311", lw=1.8)
            a.fill_between(ctr, lo, hi, color="#cc3311", alpha=0.20)
            metrics[mkey] = float(np.nanmin(med))            # faint/bright-safe bright-end floor
        a.set_xlim(-16, -3)
        a.set_ylim(0, ytop)
        a.set_xlabel("instrumental magnitude")
        a.set_ylabel(ylab)
        a.grid(alpha=0.25)
    # onset of the bright-star (saturation) degradation Jay flags, from the mag-RMS curve
    if metrics.get("mag_rms_floor") is not None:
        turn = _saturation_turnover(m, d["em"], metrics["mag_rms_floor"])
        if turn is not None:
            metrics["saturation_turnover_mag"] = turn
            ax[1][0].axvline(turn, color="#3366cc", ls="--", lw=1.0)
    fig.suptitle(f"{o.target} {o.obsid} — JWST1PASS across-exposure consistency ({filt}, "
                 f"n={metrics['n_stars']}, {metrics['n_exposures']} exp)", fontsize=11, y=0.99)
    metrics["passed"] = True
    return _save(fig, f"{o.obsid}_stage10.png"), metrics


STAGES = {1: stage1_mosaics, 2: stage2_cmd, 3: stage3_calibration, 4: stage4_offsets,
          5: stage5_intermodule}


def _build_stage5(o, sw, lw):
    return stage5_intermodule(o, sw)


# --------------------------------------------------------------------------- STAGE 7
def _mast_i2d(o, filt):
    """The MAST-delivered STScI level-3 mosaic (``mastDownload/…_<filt>_i2d.fits``), the "raw"
    product the pipeline improves on.  None if not downloaded for this obs/filter."""
    if not filt:
        return None
    fl = filt.lower()

    def _raw(p):
        # keep only the MAST-delivered raw i2d: drop per-detector products and any of our own
        # reprocessed mosaics (merged/residual/destreak/reproject/segm) the recursive+cross-field
        # globs can otherwise reach under a mastDownload tree.
        b = os.path.basename(p).lower()
        return not any(s in b for s in ("nrca", "nrcb", "segm", "merged", "residual",
                                        "destreak", "reproject", "_data_"))
    # t* wildcard (the target index is not always t001) and, as a fallback, a cross-field search:
    # a (program,obs) filed under one field's registry can have its MAST i2d staged in a sibling
    # field's tree (e.g. jw02221-o002 belongs to cloudc but its i2d sits in brick/mastDownload).
    for pat in (f"{BASE}/{o.field}/mastDownload/**/{o.obsid}_t*_nircam_clear-{fl}_i2d.fits",
                f"{BASE}/{o.field}/mastDownload/**/{o.obsid}_t*_nircam_*{fl}*_i2d.fits",
                f"{BASE}/*/mastDownload/**/{o.obsid}_t*_nircam_clear-{fl}_i2d.fits",
                f"{BASE}/*/mastDownload/**/{o.obsid}_t*_nircam_*{fl}*_i2d.fits"):
        hits = [p for p in glob.glob(pat, recursive=True) if _raw(p)]
        if hits:
            return sorted(hits)[0]
    return None


def _detect_on_mosaic(path, crop=5000, fwhm_pix=2.5, nsigma=5.0, maxn=500000):
    """Detect point sources on a mosaic's central ``crop`` x ``crop`` window and return
    (SkyCoord, instrumental_mag, wcs, cutout_data).  This APPROXIMATES a MAST L3 source list by
    running DAOStarFinder at ``nsigma``σ over a photutils Background2D -- MAST does not archive the
    merged catalogue locally, only the i2d.  It is an approximation of, not a match to, the STScI
    ``SourceCatalogStep`` (which uses image segmentation with deblending), so the two source lists
    differ; the count is a depth indicator, not a reproduction of the STScI catalogue.  Central crop
    bounds the cost and gives a common sky region for the MAST-vs-pipeline comparison.  Returns None
    on failure."""
    from astropy.io import fits
    from astropy.wcs import WCS
    from astropy.nddata import Cutout2D
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    from photutils.background import Background2D, MMMBackground, MADStdBackgroundRMS
    from photutils.detection import DAOStarFinder
    from astropy.stats import SigmaClip
    try:
        with fits.open(_used(path, "MAST i2d (DAOStarFinder fallback detection)")) as hdul:
            sci = hdul["SCI"] if "SCI" in hdul else hdul[1]
            data = sci.data.astype("float32"); w = WCS(sci.header)
    except (OSError, ValueError, KeyError):
        return None
    ny, nx = data.shape
    cy, cx = ny // 2, nx // 2
    size = (min(crop, ny), min(crop, nx))
    try:
        cut = Cutout2D(data, (cx, cy), size, wcs=w)
    except (ValueError, IndexError):
        return None
    img = np.nan_to_num(cut.data, nan=np.nanmedian(cut.data))
    try:
        bkg = Background2D(img, box_size=128, filter_size=3,
                           sigma_clip=SigmaClip(sigma=3, maxiters=5),
                           bkgrms_estimator=MADStdBackgroundRMS(), bkg_estimator=MMMBackground(),
                           exclude_percentile=90)
        sub = img - bkg.background
        rms = float(np.median(bkg.background_rms))
        found = DAOStarFinder(threshold=nsigma * rms, fwhm=fwhm_pix, peakmax=None)(sub)
    except (ValueError, RuntimeError):
        return None
    if found is None or not len(found):
        return None
    if len(found) > maxn:                                  # brightest maxn (bound the crossmatch)
        found = found[np.argsort(np.asarray(found["flux"]))[::-1][:maxn]]
    x = np.asarray(found["xcentroid"], float); y = np.asarray(found["ycentroid"], float)
    flux = np.asarray(found["flux"], float)
    sc = cut.wcs.pixel_to_world(x, y)
    sc = SkyCoord(sc.ra, sc.dec)                           # ensure plain ICRS SkyCoord
    with np.errstate(invalid="ignore", divide="ignore"):
        mag = -2.5 * np.log10(flux)
    good = np.isfinite(sc.ra.deg) & np.isfinite(sc.dec.deg) & np.isfinite(mag)
    return sc[good], mag[good], cut.wcs, cut.data


# The STScI L3 source catalogue is delivered as ``_cat.ecsv`` (the default) or ``_cat.fits``; the
# per-detector, destreak and segmentation products share the ``_cat`` stem and must be excluded.
_MAST_CAT_EXCLUDE = ("nrca", "nrcb", "destreak", "segm")


def _download_mast_l3_catalog(o, filt):
    """Download this obs+filter's MAST L3 source catalogue (``_cat.ecsv`` or ``_cat.fits``) into the
    scratch OUTDIR and return the local path, or None.  Best-effort and guarded: missing astroquery,
    no network, a hang, or no such product on MAST all fall back to None.  The download NEVER writes
    into the read-only product tree.  Callers reach this only when no local copy was found."""
    # A network HANG is not an exception (the prior "Pipeline MAST hang" was exactly this), so cap
    # every socket with a default timeout and SCOPE the query to this obsid so get_product_list
    # cannot pull the whole programme.  QA_MAST_DOWNLOAD=0 disables the network reach entirely.
    if os.environ.get("QA_MAST_DOWNLOAD", "1") == "0":
        return None
    excs = [ImportError, OSError, ValueError, KeyError, TimeoutError, ConnectionError]
    try:
        from astroquery.exceptions import RemoteServiceError
        excs.append(RemoteServiceError)
    except ImportError:
        pass
    try:
        import requests
        excs.append(requests.exceptions.RequestException)
    except ImportError:
        pass
    import socket
    prev_to = socket.getdefaulttimeout()
    socket.setdefaulttimeout(float(os.environ.get("QA_MAST_TIMEOUT", "60")))
    try:
        from astroquery.mast import Observations
        obs_t = Observations.query_criteria(
            obs_collection="JWST", instrument_name="NIRCAM*",
            obs_id=f"jw{int(o.program):05d}-o{o.obs}*", filters=filt.upper())
        if obs_t is None or not len(obs_t):
            return None
        prod = Observations.get_product_list(obs_t)
        want = np.array([(("_cat.ecsv" in str(fn).lower() or "_cat.fits" in str(fn).lower())
                          and o.obsid in str(fn) and filt.lower() in str(fn).lower()
                          and not any(s in str(fn).lower() for s in _MAST_CAT_EXCLUDE))
                         for fn in prod["productFilename"]])
        if not want.any():
            return None
        dl = Observations.download_products(
            prod[want],
            download_dir=os.path.join(OUTDIR, "mastDownload"))   # scratch, never the product tree
        good = sorted(p for p in (dl["Local Path"] if dl is not None and len(dl) else [])
                      if p and os.path.exists(p))
        # prefer .ecsv (the L3 default) if both landed
        return next((p for p in good if p.lower().endswith(".ecsv")), good[0] if good else None)
    except tuple(excs):
        return None
    finally:
        socket.setdefaulttimeout(prev_to)


def _mast_l3_catalog(o, filt, allow_download=None):
    """The MAST-delivered STScI level-3 source catalogue (``_cat.ecsv``/``_cat.fits``) for this obs.
    Prefer a local copy; if absent, download it from MAST when it exists there (guarded).  When
    still None, the caller reconstructs the list by detecting on the i2d.  Returns a path or None."""
    for ext in ("ecsv", "fits"):
        for d in (f"{BASE}/{o.field}/mastDownload", f"{BASE}/{o.field}/mastDownload/**",
                  f"{BASE}/{o.field}/MAST_FITS", f"{BASE}/{o.field}/{filt}/pipeline",
                  f"{BASE}/{o.field}/images-merged"):
            hits = [p for p in glob.glob(f"{d}/{o.obsid}_t*_nircam_*{filt.lower()}*_cat.{ext}",
                                         recursive=True)
                    if not any(s in os.path.basename(p).lower() for s in _MAST_CAT_EXCLUDE)]
            if hits:
                return sorted(hits)[-1]
    # allow_download kept for the explicit callers/tests; default now attempts (guarded) so a
    # genuinely-missing product is DOWNLOADED rather than red-flagged.
    if allow_download is False:
        return None
    return _download_mast_l3_catalog(o, filt)


def _load_mast_catalog(path):
    """(SkyCoord, abmag) from a MAST-delivered L3 source catalogue, or (None, None)."""
    from astropy.table import Table
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    try:
        t = Table.read(_used(path, "MAST L3 source catalogue"))
    except (OSError, ValueError):
        return None, None
    if "sky_centroid" in t.colnames:
        sc = t["sky_centroid"]
    elif {"ra", "dec"}.issubset(set(t.colnames)):
        sc = SkyCoord(np.asarray(t["ra"], float) * u.deg, np.asarray(t["dec"], float) * u.deg)
    else:
        return None, None
    magc = next((c for c in ("aper_total_abmag", "aper50_abmag", "aper70_abmag", "aper30_abmag",
                             "abmag") if c in t.colnames), None)
    mag = np.asarray(t[magc], float) if magc else np.full(len(sc), np.nan)
    good = np.isfinite(sc.ra.deg) & np.isfinite(sc.dec.deg)
    if int(good.sum()) < 20:
        return None, None
    return sc[good], mag[good]


def _offset_cloud(jsc, ref_sc):
    """The per-star (JWST − VIRAC) offset VECTORS for genuinely-matched stars.

    VIRAC is sparse, so the distance from a random JWST source to its nearest VIRAC neighbour is
    about the reference's own source spacing (~250 mas at GC density) for ANY frame, and it swamps
    the offset being looked for.  So: coarse-align by the ``aa.xcorr`` histogram peak, which finds
    the bulk shift out to 1.5", keep the pairs that then fall within 0.1" (the real matches), and
    return their (ΔRA, ΔDec) in mas.  The centre of that cloud is the field offset from VIRAC.
    Returns (dra_arr, dde_arr, bulk_off_mas) or None."""
    import astropy.units as u
    from astropy.coordinates import search_around_sky, SkyCoord
    if jsc is None or ref_sc is None or len(jsc) < 50:
        return None
    xc = aa.xcorr(jsc, ref_sc, maxsep=1.5 * u.arcsec)
    if not (xc and xc.get("peak_ratio", 0) >= aa.MIN_PEAK_RATIO and xc.get("npairs", 0) >= 100):
        return None
    cosd = float(np.cos(np.radians(np.median(jsc.dec.deg))))
    j_al = SkyCoord((jsc.ra.deg + xc["dra"] / 1000.0 / 3600.0 / cosd) * u.deg,
                    (jsc.dec.deg + xc["ddec"] / 1000.0 / 3600.0) * u.deg)
    ia, ib, sep, _ = search_around_sky(j_al, ref_sc, 0.1 * u.arcsec)
    if len(ia) < 20:
        return None
    dra = (jsc[ia].ra - ref_sc[ib].ra).to(u.mas).value * cosd
    dde = (jsc[ia].dec - ref_sc[ib].dec).to(u.mas).value
    return dra, dde, float(np.hypot(np.median(dra), np.median(dde)))


def _stage7_astrom_title(mast_off, jic_off):
    """Astrometry sub-panel title, worded from the SIGN of (jicama offset − MAST offset).  It calls
    the pipeline 'tighter' only when both offsets are measured AND jicama is STRICTLY smaller; in
    every other case it reports the number(s) it has and claims no improvement.  ``mast_off``/
    ``jic_off`` are ``_offset_cloud`` results (…, bulk_mas) or None.

    This is a STRICTER rule than the pass gate: ``_stage7_verdict`` counts a field as improved at
    ``jic <= mast + 5``, so a frame where jicama is up to 5 mas worse passes while this title
    claims nothing.  The two are meant to differ -- the verdict tolerates measurement noise, the
    title asserts an improvement -- and a caption that says nothing next to a green tick is that
    5 mas band, not a contradiction."""
    both = mast_off is not None and jic_off is not None
    if both:
        head = (f"offset from VIRAC — jicama {jic_off[2]:.0f} mas vs MAST "
                f"{mast_off[2]:.0f} mas")
        return head + (" (pipeline tighter)" if jic_off[2] < mast_off[2] else "")
    if jic_off is not None:
        return f"offset from VIRAC — jicama {jic_off[2]:.0f} mas (MAST offset not measured)"
    if mast_off is not None:
        return f"offset from VIRAC — MAST {mast_off[2]:.0f} mas (pipeline offset not measured)"
    return "astrometry vs VIRAC (bulk offset)"


def _stage7_verdict(our_path, mast_off, jic_off, jic_unmeas):
    """PASS / red-flag decision for stage 7, factored out so it can be pinned by tests.

    Failing to measure our OWN (jicama) offset while VIRAC is present is our product failing, so it
    fails AND red-flags.  A missing VIRAC reference, or only the MAST offset being unmeasurable,
    leaves the comparison 'not measured' / 'unavailable' and says so.

    ``improved`` carries a 5 mas tolerance (``jic <= mast + 5``) that the panel title does not:
    ``_stage7_astrom_title`` says 'tighter' only on a strict ``jic < mast``.  So a field sitting in
    that 5 mas band passes here while the title claims no improvement.

    Returns (passed: bool, red_flag: bool, red_flag_reason: str|None)."""
    improved = (jic_off is not None and mast_off is not None and jic_off[2] <= mast_off[2] + 5)
    passed = bool(our_path and not jic_unmeas
                  and (improved or mast_off is None or jic_off is None))
    if jic_unmeas:
        return False, True, ("the pipeline (jicama) offset from VIRAC is unmeasurable — no xcorr "
                             "peak within 1.5″, which can mean our product is grossly "
                             "mis-registered")
    return passed, False, None


def _depth_hist(ax, mags, label, color):
    """Draw one step magnitude-histogram of the FINITE entries of ``mags`` and return that count.
    Guards the two ways ``np.histogram`` raises "autodetected range of [nan, nan] is not finite":
    an EMPTY selection (no sources in the common window) and an all-NaN column (MAST abmag carries
    NaN rows).  A count of 0 draws nothing -- the panel simply omits that series."""
    m = np.asarray(mags, float)
    m = m[np.isfinite(m)]
    n = int(m.size)
    if n:
        ax.hist(m, bins=50, histtype="step", color=color, lw=1.4, label=f"{label} (n={n})")
    return n


def stage7_mast_vs_pipeline(o: Observation, sw):
    """Compare the pipeline against the raw MAST-delivered products:
    (top) the MAST L3 i2d mosaic beside our pipeline mosaic over the SAME sky region;
    (bottom-left) source-count / brightness histogram, MAST-i2d detections vs the jicama catalogue;
    (bottom-right, MAIN) each catalogue's per-star offset from VIRAC, which is where the pipeline's
    astrometric gain shows up."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from astropy.io import fits
    from astropy.wcs import WCS
    from astropy.nddata import Cutout2D
    from astropy.visualization import ZScaleInterval, ImageNormalize, AsinhStretch
    metrics = dict(stage=7, sw=sw)
    mast_path = _mast_i2d(o, sw)
    our_path = _mosaic_path(o, sw)
    if not mast_path:
        png = _red_flag_figure(o, "stage7", "NO MAST i2d ON DISK",
                               f"No MAST-delivered {sw} i2d in mastDownload/ for this obs, so the "
                               f"before/after comparison can't be made. (Not a data defect — the "
                               f"raw MAST product just isn't staged locally.)")
        metrics.update(red_flag=True, red_flag_reason=f"no MAST {sw} i2d on disk", passed=False)
        return png, metrics

    # MAST source list: prefer the MAST-delivered L3 catalogue (download if it exists on MAST),
    # else RECONSTRUCT it by detecting on the MAST i2d (what the STScI L3 step does).
    mast_sc = mast_mag = None
    mast_kind = None
    mcat = _mast_l3_catalog(o, sw)
    if mcat:
        mast_sc, mast_mag = _load_mast_catalog(mcat)
        if mast_sc is not None:
            mast_kind = "MAST L3 catalogue"
    if mast_sc is None:
        det = _detect_on_mosaic(mast_path)
        if det is not None:
            mast_sc, mast_mag, _cutwcs, _cutdata = det
            mast_kind = "MAST i2d (re-detected)"
    if mast_sc is not None:
        metrics["n_mast"] = int(len(mast_sc)); metrics["mast_kind"] = mast_kind

    # jicama catalogue positions + mag (the pipeline product).  _jwst_sources falls back to the
    # MAST per-i2d _cat.fits when no release/merged catalogue exists yet -- in that case the
    # "pipeline" side is NOT actually the jicama product, so label it as such (else the panel
    # silently becomes MAST-vs-MAST).
    # Use the INSTRUMENTAL magnitude from flux_<filt>, which is populated for the FULL catalogue, so
    # the depth histogram reaches the catalogue's real faint limit.  mag_vega_<filt> (what
    # _jwst_sources returns) is filled only for the bright stars that cross-matched to VIRAC for the
    # zeropoint (~1700 for cloudef), which truncated this histogram at ~16.5 mag instead of ~24 and
    # made the pipeline product look orders of magnitude shallower than it is (issue #38).  Fall back
    # to _jwst_sources only when no flux catalogue exists (then it is the MAST per-i2d cat, labelled).
    jsc, jflux, jname = _psf_flux_positions(o, sw)
    if jsc is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            jmag = -2.5 * np.log10(jflux)
        fin = np.isfinite(jmag)
        jsc, jmag = jsc[fin], jmag[fin]
        jsrc = f"release:{jname}"; jic_is_release = True
    else:
        jsc, jmag, jsrc = _jwst_sources(o, sw)
        jic_is_release = str(jsrc).startswith("release")
    metrics["jicama_source"] = jsrc
    metrics["jicama_is_release"] = bool(jic_is_release)
    jic_label = "jicama catalogue" if jic_is_release else "pipeline (no merged catalogue yet)"
    if jsc is not None:
        metrics["n_jicama"] = int(len(jsc))

    # display + count window: central crop of the MAST i2d (so both image panels show the SAME sky,
    # and the source-count comparison is over one common region, not full-field vs a crop).
    try:
        with fits.open(_used(mast_path, f"{sw} MAST i2d (before panel)")) as h:
            mh = (h["SCI"] if "SCI" in h else h[1]).header
        mwcs = WCS(mh); mny, mnx = int(mh["NAXIS2"]), int(mh["NAXIS1"])
    except (OSError, ValueError, KeyError):
        mwcs = None; mny = mnx = 0
    crop = min(5000, mny, mnx) if mwcs is not None else 0
    cx, cy = (mnx / 2.0, mny / 2.0)
    # Center the common window where the PIPELINE catalogue actually has sources, not on the
    # geometric centre of the MAST single-visit i2d.  The MAST i2d is one visit's wide strip; its
    # centre can fall in a shallow-coverage edge of the deep merged mosaic, so a centred crop shows
    # the pipeline as nearly empty while MAST re-detects thousands there (cloudef o005: 0 vs ~1e5).
    # Use the jicama median position, clipped so the crop stays inside the MAST frame.
    if mwcs is not None and jsc is not None and crop > 0:
        try:
            jx, jy = mwcs.world_to_pixel(jsc)
            infoot = np.isfinite(jx) & np.isfinite(jy) & (jx > 0) & (jx < mnx) & (jy > 0) & (jy < mny)
            if int(infoot.sum()) >= 100:
                cx = float(np.clip(np.median(jx[infoot]), crop / 2, mnx - crop / 2))
                cy = float(np.clip(np.median(jy[infoot]), crop / 2, mny - crop / 2))
        except (ValueError, IndexError):
            pass
    cen = mwcs.pixel_to_world(cx, cy) if mwcs is not None else None
    if mwcs is not None:
        cw = mwcs.pixel_to_world([cx - crop / 2, cx + crop / 2], [cy - crop / 2, cy + crop / 2])
        ra_lo, ra_hi = sorted(cw.ra.deg); de_lo, de_hi = sorted(cw.dec.deg)

        def _inbox(sc):
            return ((sc.ra.deg >= ra_lo) & (sc.ra.deg <= ra_hi) &
                    (sc.dec.deg >= de_lo) & (sc.dec.deg <= de_hi))
    else:
        def _inbox(sc):
            return np.ones(len(sc), bool)

    # VIRAC reference (PM-propagated) for the crowding-robust bulk-offset comparison
    ref = _viraccache_path(o) or _refcat_path(o)
    ep = aa.epoch_of(mast_path) if mast_path else None
    ref_sc, _ = (aa.load_reference(_used(ref, "VIRAC2/Gaia reference catalogue"), ep)
                 if (ref and ep) else (None, None))
    mast_off = _offset_cloud(mast_sc, ref_sc) if mast_sc is not None else None
    jic_off = _offset_cloud(jsc, ref_sc) if jsc is not None else None
    if mast_off is not None:
        metrics["mast_offset_med_mas"] = mast_off[2]
    if jic_off is not None:
        metrics["jicama_offset_med_mas"] = jic_off[2]

    # ---- figure: two image panels on top, two comparison panels below
    fig = plt.figure(figsize=(12.5, 10.8))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 0.95], hspace=0.3, wspace=0.28)

    def _show(ax, path, title, role="mosaic (displayed)"):
        try:
            with fits.open(_used(path, role)) as hdul:
                sci = hdul["SCI"] if "SCI" in hdul else hdul[1]
                d = sci.data.astype("float32"); wv = WCS(sci.header)
        except (OSError, ValueError, KeyError):
            ax.text(0.5, 0.5, "mosaic unreadable", ha="center", va="center"); ax.axis("off"); return
        try:
            if cen is not None:
                px, py = wv.world_to_pixel(cen)
                d = Cutout2D(d, (float(px), float(py)), min(5000, *d.shape), wcs=wv).data
        except (ValueError, IndexError):
            pass
        norm = ImageNormalize(d, interval=ZScaleInterval(), stretch=AsinhStretch())
        ax.imshow(d, origin="lower", cmap="gray", norm=norm)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title, fontsize=10)

    ax_m = fig.add_subplot(gs[0, 0])
    _show(ax_m, mast_path, f"MAST L3 i2d — {sw} (before)", role=f"{sw} MAST i2d (before panel)")
    ax_o = fig.add_subplot(gs[0, 1])
    if our_path:
        _show(ax_o, our_path, f"pipeline i2d — {sw} (after)", role=f"{sw} pipeline mosaic (after panel)")
    else:
        ax_o.text(0.5, 0.5, "no pipeline mosaic yet", ha="center", va="center"); ax_o.axis("off")

    # bottom-left: source counts / brightness histogram in the common window (brief -- jicama is
    # deeper by construction; the point is the count + depth, not the zeropoint).
    axh = fig.add_subplot(gs[1, 0])
    n_mast_win = n_jic_win = None
    # put the two catalogues on the SAME zeropoint (jicama's) so the depth is directly comparable:
    # shift the MAST mags by the median (jicama − MAST) of cross-matched stars.
    mast_zp = None
    if (mast_sc is not None and mast_mag is not None and jsc is not None and jmag is not None):
        import astropy.units as u
        idx, sep, _ = mast_sc.match_to_catalog_sky(jsc)
        mm2 = sep < 0.2 * u.arcsec
        if int(mm2.sum()) >= 30:
            mast_zp = float(np.median(jmag[idx[mm2]] - mast_mag[mm2]))
            metrics["mast_to_jicama_zp"] = mast_zp
    if mast_sc is not None and mast_mag is not None:
        mm = mast_mag[_inbox(mast_sc)] + (mast_zp or 0.0)      # on jicama ZP when calibrated
        zlab = "" if mast_zp is not None else " (own ZP)"
        n_mast_win = _depth_hist(axh, mm, f"{mast_kind}{zlab}", "#ee6677")
    if jsc is not None and jmag is not None:
        n_jic_win = _depth_hist(axh, jmag[_inbox(jsc)], jic_label, "#4477aa")
    if n_mast_win is not None:
        metrics["n_mast_window"] = n_mast_win
    if n_jic_win is not None:
        metrics["n_jicama_window"] = n_jic_win
    axh.set_yscale("log")
    axh.set_xlabel("magnitude" + (" (jicama zeropoint)" if mast_zp is not None
                                  else " (each in its own zeropoint)"))
    axh.set_ylabel("N sources"); axh.legend(fontsize=7.5, loc="upper left")
    axh.set_title("source counts in the common window — MAST vs pipeline", fontsize=9)

    # bottom-right (MAIN): each catalogue's offset from VIRAC, as a 2-D (ΔRA, ΔDec) cloud with
    # marginals.  VIRAC's own ~250 mas source spacing swamps a nearest-neighbour distance, so this
    # coarse-aligns by the xcorr histogram peak first; the cloud CENTRE is then the field offset
    # (MAST far from 0, jicama near 0).
    axo = fig.add_subplot(gs[1, 1])
    # Wording is DERIVED from the sign of (jicama offset − MAST offset): claim the pipeline
    # "tightens" only when BOTH are measured AND jicama is the smaller.  When jicama is wider or
    # equal, or one side is unmeasurable, report the numbers and assert no improvement.
    both_measured = mast_off is not None and jic_off is not None
    jic_tighter = bool(both_measured and jic_off[2] < mast_off[2])
    metrics["astrom_improved"] = jic_tighter
    lim = 60.0
    for cloud, col, lab in ((mast_off, "#ee6677", mast_kind or "MAST"),
                            (jic_off, "#4477aa", jic_label)):
        if cloud is not None:
            dra, dde, bulk = cloud
            axo.scatter(dra, dde, s=3, alpha=0.25, color=col,
                        label=f"{lab}  (bulk {bulk:.0f} mas)")
            lim = max(lim, 1.3 * float(np.nanpercentile(np.hypot(dra, dde), 95)))
    if mast_off is not None or jic_off is not None:
        axo.axhline(0, color="k", lw=0.4); axo.axvline(0, color="k", lw=0.4)
        axo.plot(0, 0, "k+", ms=12, mew=2, label="VIRAC (target)")
        axo.set_xlim(-lim, lim); axo.set_ylim(-lim, lim); axo.set_aspect("equal")
        axo.set_xlabel("ΔRA to VIRAC [mas]"); axo.set_ylabel("ΔDec to VIRAC [mas]")
        axo.legend(fontsize=7.5, loc="upper right")
        # combined marginals (both catalogues in ONE pair of inset axes, so they don't overplot);
        # the panel title goes on the TOP marginal so it can't collide with the histograms.
        axt = axo.inset_axes([0.0, 1.02, 1.0, 0.16]); axr = axo.inset_axes([1.02, 0.0, 0.16, 1.0])
        for cloud, col in ((mast_off, "#ee6677"), (jic_off, "#4477aa")):
            if cloud is not None:
                axt.hist(cloud[0], bins=40, range=(-lim, lim), histtype="step", color=col, lw=1.1)
                axr.hist(cloud[1], bins=40, range=(-lim, lim), orientation="horizontal",
                         histtype="step", color=col, lw=1.1)
        axt.set_xlim(-lim, lim); axt.axis("off"); axr.set_ylim(-lim, lim); axr.axis("off")
        axt.set_title(_stage7_astrom_title(mast_off, jic_off), fontsize=8)
    # UNMEASURABLE (data + VIRAC present but no xcorr peak within 1.5") must be distinguishable
    # from measured-and-fine: left undistinguished, a grossly mis-registered product (the ~20"
    # jw01182-v001 class) renders as a clean one-cloud pass.
    mast_unmeas = mast_sc is not None and ref_sc is not None and mast_off is None
    jic_unmeas = jsc is not None and ref_sc is not None and jic_off is None
    metrics.update(mast_offset_unmeasurable=bool(mast_unmeas), jicama_offset_unmeasurable=bool(jic_unmeas))
    if mast_off is None and jic_off is None:
        axo.axis("off")
        if ref_sc is None:
            axo.text(0.5, 0.5, "no VIRAC reference for the offset comparison",
                     ha="center", va="center", fontsize=9)
        else:
            axo.text(0.5, 0.5, "NEITHER offset measurable: no xcorr peak within 1.5″\n"
                     "→ likely gross mis-registration; investigate",
                     ha="center", va="center", fontsize=9, color="#c33", weight="bold")
    elif jic_unmeas:
        # OUR product cannot be placed against VIRAC -> a defect on our side (passed/red_flag).
        axo.text(0.5, -0.16, "⚠ pipeline (jicama) offset unmeasurable (>1.5″ or no xcorr peak) — "
                 "possible gross mis-registration of our product; not a clean pass",
                 transform=axo.transAxes, ha="center", va="top", fontsize=8, color="#c33")
    elif mast_unmeas:
        # Only the MAST offset is unmeasurable, so the COMPARISON is what is unavailable.  That
        # says nothing about our product, so the wording stays neutral and the stage does not fail
        # on it alone (below).
        axo.text(0.5, -0.16, "⚠ MAST comparison unavailable (MAST offset >1.5″ or no xcorr peak) — "
                 "possible MAST mis-registration; the pipeline offset is measured",
                 transform=axo.transAxes, ha="center", va="top", fontsize=8, color="#c33")

    # PASS / red-flag decision (factored into _stage7_verdict so tests can pin it): our offset must
    # be measurable, a mosaic must exist, and where MAST is also measurable our offset must be no
    # worse.  A missing VIRAC ref, or only the MAST offset being unmeasurable, leaves the comparison
    # "not measured" / "unavailable"; failing to measure our OWN offset fails AND red-flags.
    passed, red_flag, red_flag_reason = _stage7_verdict(our_path, mast_off, jic_off, jic_unmeas)
    metrics["passed"] = passed
    if red_flag:
        metrics["red_flag"] = True
        metrics["red_flag_reason"] = red_flag_reason
    fig.suptitle(f"{o.target} {o.obsid} — MAST vs pipeline ({sw})", fontsize=12, y=0.98)
    return _save(fig, f"{o.obsid}_stage7.png"), metrics


# --------------------------------------------------------------------------- STAGE 8 (distortion)
_SKYCOORD_COL_RE = re.compile(r"^skycoord_(f\d{3}[wnm])\.ra$")
# Fixed absolute amplitude (mas) above which the inter-filter residual is flagged as a GROSS
# per-filter WCS break rather than the expected ~1 mas distortion term.  Fixed on purpose -- it is
# NOT scaled by the per-star scatter being tested, so injected noise cannot loosen it.
_STAGE8_GROSS_OFFSET_MAS = 15.0


def _interfilter_residuals(o, f1):
    """Per-star (RA, Dec, ΔRA, ΔDec) position difference between filter ``f1`` and a second JWST
    filter of the SAME field, from the merged catalogue's per-band positions of the SAME source
    rows -- bulk-removed.

    This is the distortion reference the way #60's review asks for: two filters share the frames,
    the offsets table, the DVA correction and the registration onto VIRAC, which leaves a per-filter
    WCS term as the ONLY thing that differs -- the position-dependent distortion residual this is
    for, with no external catalogue in the measurement (VIRAC's ~20 mas per-star PM error would
    swamp a few-mas residual).
    The rows were paired across bands upstream, by a mutual-nearest-neighbour match at ~100 mas per
    band in ``merge_catalogs.py`` (visible as a hard truncation of the kept separations near
    100 mas), so a blind spot survives at ~100 mas.  That is ~100x the ~1 mas signal, well above
    where the VIRAC-referenced version would cut.  It is still a cut, and it leaves a tail of
    nearest-neighbour-ambiguous rows (see ``frac_gt_20mas`` in the stage metrics) that inflates the
    per-cell sampling noise.

    The partner filter is the nearest in wavelength that has a ``skycoord_<f>`` column in the same
    catalogue.  Returns (ra, dec, dra_mas, dde_mas, f2, catname) or None."""
    from astropy.io import fits
    from astropy.table import Table
    import astropy.units as u
    sc1col = f"skycoord_{f1.lower()}"
    best = None; rank = (-1.0, -1.0)
    for p, kind, tier, mtime in _catalog_candidates(o):
        if (tier, mtime) <= rank:
            continue
        try:
            hdr = fits.getheader(p, ext=1)
        except (OSError, IndexError):
            continue
        ttypes = [str(hdr.get(f"TTYPE{i}", "")).lower() for i in range(1, hdr.get("TFIELDS", 0) + 1)]
        if f"{sc1col}.ra" not in ttypes:
            continue
        others = sorted({m.group(1) for t in ttypes if (m := _SKYCOORD_COL_RE.match(t))
                         and m.group(1) != f1.lower()})
        if others:
            best = (p, others); rank = (tier, mtime)
    if best is None:
        return None
    p, others = best

    def _wl(name):                                          # "f212n" -> 212
        m = re.search(r"f(\d{3})", name); return int(m.group(1)) if m else 9999
    f2 = min(others, key=lambda c: abs(_wl(c) - _wl(f1.lower())))
    try:
        t = Table.read(_used(p, f"cross-band catalogue ({f1} + {f2})"))
    except (OSError, ValueError):
        return None
    sc1 = t[sc1col]; sc2 = t[f"skycoord_{f2}"]
    g = (np.isfinite(sc1.ra.deg) & np.isfinite(sc1.dec.deg)
         & np.isfinite(sc2.ra.deg) & np.isfinite(sc2.dec.deg))
    # keep only WELL-MEASURED stars in BOTH bands (flux S/N > 10 where the columns exist), so the
    # per-star scatter is the real centroid precision, not blends/faint junk -- this is what makes
    # a coherent sub-mas distortion recoverable above a shuffled-position null (see stage8).
    for fb in (f1.lower(), f2):
        fc, ec = f"flux_{fb}", f"flux_err_{fb}"
        if fc in t.colnames and ec in t.colnames:
            with np.errstate(invalid="ignore", divide="ignore"):
                sn = np.asarray(t[fc], float) / np.asarray(t[ec], float)
            g &= np.isfinite(sn) & (sn > 10)
    if int(g.sum()) < 200:
        return None
    sc1 = sc1[g]; sc2 = sc2[g]
    cosd = float(np.cos(np.radians(np.median(sc1.dec.deg))))
    dra = (sc1.ra - sc2.ra).to(u.mas).value * cosd
    dde = (sc1.dec - sc2.dec).to(u.mas).value
    dra -= np.median(dra); dde -= np.median(dde)           # bulk-removed -> residual = distortion
    return sc1.ra.deg, sc1.dec.deg, dra, dde, f2.upper(), os.path.basename(p)


def _binned_median_2d(x, y, vals, nb, minn=3, cosd=1.0):
    """Median of ``vals`` on an ``nb`` x ``nb`` grid over (x, y), plus the per-cell count.
    Numpy-only (avoids a scipy dependency that the CI env lacks).  Returns (med, xe, ye, cnt)
    with med/cnt shaped ``[nb (x), nb (y)]`` and NaN in cells below ``minn`` points.

    When ``x`` is RA in degrees, pass ``cosd = cos(dec)`` so the x bins span equal on-sky
    distance (a degree of RA is ``cosd`` of a degree of Dec on the sky, ~1.14x at dec=-28.9);
    the returned edges ``xe`` are converted back to native RA degrees for plotting."""
    xs = np.asarray(x, float) * cosd
    xse = np.linspace(np.min(xs), np.max(xs), nb + 1)
    ye = np.linspace(np.min(y), np.max(y), nb + 1)
    ix = np.clip(np.digitize(xs, xse) - 1, 0, nb - 1)
    iy = np.clip(np.digitize(y, ye) - 1, 0, nb - 1)
    xe = xse / cosd if cosd else xse
    med = np.full((nb, nb), np.nan); cnt = np.zeros((nb, nb), int)
    for i in range(nb):
        xi = ix == i
        for j in range(nb):
            m = xi & (iy == j)
            c = int(m.sum())
            if c >= minn:
                med[i, j] = float(np.median(vals[m])); cnt[i, j] = c
    return med, xe, ye, cnt


def _amp90(mx, my, cnt):
    """90th-percentile per-cell residual amplitude (mas) over POPULATED cells; 0 if none."""
    cell_amp = np.hypot(np.nan_to_num(mx), np.nan_to_num(my))[cnt > 0]
    return float(np.nanpercentile(cell_amp, 90)) if cell_amp.size else 0.0


def _shuffled_null_amp90(ra, dec, dra, dde, nb, cosd, minn=3, n_perm=20, seed=0):
    """Null distribution of the 90th-percentile cell amplitude under permuting the
    (residual-vector) -> (position) association.  Positions and cell membership are held fixed and
    the (ΔRA, ΔDec) pairs are shuffled together, so each permutation is the cell amplitude expected
    from finite-sample noise with NO spatial coherence.  This carries the nearest-neighbour tail
    (``frac_gt_20mas``) and the median's efficiency penalty, both of which the per-cell SEM omits,
    so it is a less optimistic significance floor.  Returns (median null amp90, all null amp90s)."""
    rng = np.random.default_rng(seed)
    dra = np.asarray(dra); dde = np.asarray(dde)
    nulls = np.empty(int(n_perm))
    for k in range(int(n_perm)):
        perm = rng.permutation(len(dra))
        mxp, _, _, cp = _binned_median_2d(ra, dec, dra[perm], nb, minn=minn, cosd=cosd)
        myp, _, _, _ = _binned_median_2d(ra, dec, dde[perm], nb, minn=minn, cosd=cosd)
        nulls[k] = _amp90(mxp, myp, cp)
    return float(np.median(nulls)), nulls


def stage8_distortion(o: Observation, sw):
    """Distortion diagnostic: the INTER-FILTER position residual (bulk-removed) as a function of
    position across the field -- ``sw`` minus a second JWST filter of the same field, same source
    rows.  Two filters share the frames, the offsets table, the DVA correction and the
    registration onto VIRAC, so the residual is a per-filter WCS (distortion) term measured with no
    external catalogue in it (rows are paired across bands upstream at ~100 mas per band, far above
    the ~1 mas signal; see ``_interfilter_residuals``).
    LEFT/MIDDLE are binned-median ΔRA/ΔDec maps; RIGHT is a per-cell quiver.  A flat map = no
    differential distortion; a coherent gradient/swirl = a distortion residual between the two
    filters' solutions, whose significance is quoted against a shuffled-position null."""
    import matplotlib
    matplotlib.use("Agg")
    metrics = dict(stage=8, sw=sw)
    res = _interfilter_residuals(o, sw)
    if res is None:
        # NOT APPLICABLE, not a defect: a single-filter or not-yet-merged obs simply has no second
        # band to difference.  Use a distinct "measurement unavailable" state -- do NOT red_flag it
        # and do NOT mark it failed (either would post a non-defect as a defect).
        png = _red_flag_figure(o, "stage8", "DISTORTION MAP NOT APPLICABLE",
                               f"No inter-filter distortion map for {sw}: need a merged catalogue "
                               f"carrying {sw} positions plus a second filter's positions for the "
                               f"same sources. (Not a defect — a single-filter or not-yet-merged "
                               f"obs simply has no second band to difference.)")
        metrics.update(measurable=False, passed=None,
                       na_reason="no second-filter positions for an inter-filter distortion map")
        return png, metrics
    ra, dec, dra, dde, f2, catname = res
    cosd = float(np.cos(np.radians(np.median(dec))))
    rad = np.hypot(dra, dde)                                # bulk already removed upstream
    metrics.update(f2=f2, catalog=catname, n_stars=int(len(ra)),
                   resid_rms_mas=float(np.hypot(aa.mad_std(dra), aa.mad_std(dde))),
                   frac_gt_20mas=float(np.mean(rad > 20.0)))

    nb = 12; minn = 3
    fig, ax = _fig(1, 3, 5.6, 5.2)
    fig.subplots_adjust(wspace=0.34, top=0.85, bottom=0.12)
    # binned-median maps.  med is [x(RA), y(Dec)] -> med.T is [Dec, RA] with column 0 = LOWEST RA,
    # so extent runs ra.min..ra.max then invert_xaxis() -> RA-increases-left, matching the quiver.
    mx, xe, ye, cnt = _binned_median_2d(ra, dec, dra, nb, minn=minn, cosd=cosd)
    my, _, _, _ = _binned_median_2d(ra, dec, dde, nb, minn=minn, cosd=cosd)
    extent = [xe[0], xe[-1], ye[0], ye[-1]]
    # aspect = 1/cosd draws a degree of RA and a degree of Dec at their true on-sky ratio (RA is
    # compressed by cosd on the sky), so a separation read off the map is correct.
    aspect = (1.0 / cosd) if cosd else "auto"
    vlim = max(2.0, float(np.nanpercentile(rad, 90)))
    for col, (med, lab) in enumerate(((mx, "ΔRA"), (my, "ΔDec"))):
        a = ax[0][col]
        im = a.imshow(med.T, origin="lower", extent=extent, aspect=aspect,
                      cmap="RdBu_r", vmin=-vlim, vmax=vlim)
        fig.colorbar(im, ax=a, label=f"median {lab} ({sw}−{f2}) [mas]", shrink=0.85)
        a.set_xlabel("RA [deg]"); a.set_ylabel("Dec [deg]"); a.invert_xaxis()
        a.set_title(f"{lab} residual vs position ({sw}−{f2})", fontsize=9)
    # per-cell quiver (same binned medians)
    aq = ax[0][2]
    xc = 0.5 * (xe[1:] + xe[:-1]); yc = 0.5 * (ye[1:] + ye[:-1])
    XX, YY = np.meshgrid(xc, yc)
    q = aq.quiver(XX, YY, mx.T, my.T, np.hypot(mx.T, my.T), angles="xy", cmap="viridis")
    aq.quiverkey(q, 0.82, 1.08, vlim, f"{vlim:.1f} mas", labelpos="E", coordinates="axes",
                 fontproperties={"size": 8})
    aq.invert_xaxis(); aq.set_xlabel("RA [deg]"); aq.set_ylabel("Dec [deg]")
    if aspect != "auto":
        aq.set_aspect(aspect)

    cells_total = nb * nb
    cells_used = int(np.count_nonzero(cnt >= minn))
    spb = float(np.median(cnt[cnt > 0])) if np.any(cnt > 0) else 0.0
    per_cell_sem = metrics["resid_rms_mas"] / np.sqrt(max(spb, 1.0))
    amp = _amp90(mx, my, cnt)
    # significance from a shuffled-position NULL, not the per-cell SEM.  The SEM (scatter/sqrt n) is
    # ~2x optimistic because the nearest-neighbour-ambiguous tail (frac_gt_20mas) inflates the
    # sampling noise of a cell median; the null carries that tail, so it is the reported floor.
    null_amp, nulls = _shuffled_null_amp90(ra, dec, dra, dde, nb, cosd, minn=minn)
    signif = float(amp / null_amp) if null_amp > 0 else float("inf")
    p_value = float((int(np.count_nonzero(nulls >= amp)) + 1) / (len(nulls) + 1))
    metrics.update(binned_amp90_mas=amp, per_cell_sem_mas=float(per_cell_sem),
                   null_amp90_mas=float(null_amp), amp90_significance=signif,
                   amp90_p_value=p_value, stars_per_cell=spb,
                   cells_used=cells_used, cells_total=cells_total)
    aq.set_title(f"per-cell residual quiver\n({len(ra)} stars, {metrics['resid_rms_mas']:.2f} mas "
                 f"per-star; amp90 {amp:.2f} vs null {null_amp:.2f} mas = {signif:.1f}×)",
                 fontsize=9, pad=16)
    # A real ~1 mas inter-filter distortion term is an EXPECTED measurement, not a defect, so
    # pass/fail reflects only whether the MEASUREMENT SUCCEEDED (enough populated cells) -- it is
    # NOT gated on the amplitude vs a self-derived noise level, so injecting noise cannot flip it
    # (noise leaves the cells populated).  Amplitude + null significance are reported for reading.
    metrics["passed"] = bool(cells_used >= minn)
    # red_flag ONLY on a GROSS absolute inter-filter offset -- a fixed threshold a normal ~1 mas
    # distortion residual never reaches, so it fires only on a genuine per-filter WCS break.
    if amp > _STAGE8_GROSS_OFFSET_MAS:
        metrics.update(red_flag=True,
                       red_flag_reason=(f"gross inter-filter offset: amp90 {amp:.1f} mas "
                                        f"(> {_STAGE8_GROSS_OFFSET_MAS:.0f} mas) between {sw} and "
                                        f"{f2} — likely a per-filter WCS break"))
    fig.suptitle(f"{o.target} {o.obsid} — inter-filter distortion residual ({sw} − {f2})",
                 fontsize=11, y=0.98)
    return _save(fig, f"{o.obsid}_stage8.png"), metrics


# --------------------------------------------------------------------------- MIRI overview
# Spitzer comparison mosaics (GC-wide; overridable via env).  IRAC ch4 = 8 um (GLIMPSE),
# MIPS = 24 um (MIPSGAL).  Cropped to the MIRI footprint at plot time.
SPITZER_IRAC4 = os.environ.get(
    "QA_IRAC4", "/orange/adamginsburg/cmz/glimpse_data/GLM_00000+0000_mosaic_I4.fits")
SPITZER_MIPS24 = os.environ.get(
    "QA_MIPS24", "/orange/adamginsburg/cmz/mipsgal_24micron_data/gc_mosaic_MIPSGAL.fits")
# central wavelength (um) of each MIRI imaging filter, to pick the nearest Spitzer band
_MIRI_WAVE = {"F560W": 5.6, "F770W": 7.7, "F1000W": 10.0, "F1130W": 11.3, "F1280W": 12.8,
              "F1500W": 15.0, "F1800W": 18.0, "F2100W": 21.0, "F2550W": 25.5}


def _miri_i2d(o, filt):
    """MAST-delivered MIRI level-3 mosaic for one filter, or None.

    MAST stages L3 products under ``mastDownload/JWST/<product>/`` subdirs (so the search must
    recurse), the tile token varies (``_t001_``/``_t002_``/``_t003_``), and some obs have no
    field-named mastDownload dir of their own -- e.g. cloudc's ``jw02221-o002`` mosaic lives in
    the sibling ``brick/mastDownload`` tree.  A cross-field wildcard therefore backs up the
    field-scoped hit; the field-scoped hit is PREFERRED (tried first) so the wildcard cannot
    grab a wrong field's file when a scoped one exists."""
    filt = filt.lower()
    pats = (f"{BASE}/{o.field}/mastDownload/**/{o.obsid}_t*_miri_*{filt}*_i2d.fits",
            f"{BASE}/*/mastDownload/**/{o.obsid}_t*_miri_*{filt}*_i2d.fits")
    for pat in pats:
        hits = glob.glob(pat, recursive=True)
        if hits:
            return sorted(hits)[0]
    return None


def _spitzer_for_miri(filt):
    """(label, mosaic path) of the nearer Spitzer band for this MIRI filter: IRAC 8 um below
    ~14 um, MIPS 24 um above.  None if the mosaic isn't on disk.

    The split sits at the geometric midpoint of the two Spitzer bands (sqrt(8*24) = 13.9 um) so
    F1280W (12.8 um, 4.8 um from IRAC 8 vs 11.2 um from MIPS 24) maps to the closer IRAC band."""
    w = _MIRI_WAVE.get(filt.upper())
    if w is None:
        return None
    path, lbl = ((SPITZER_IRAC4, "Spitzer IRAC 8 µm (GLIMPSE)") if w < 14
                 else (SPITZER_MIPS24, "Spitzer MIPS 24 µm (MIPSGAL)"))
    return (lbl, path) if os.path.exists(path) else None


def _saturation_mask(o):
    """Saturation summary from the SATURATED DQ bit (=2) across the per-exposure MAST products
    of this obs, or None.  The i2d carries no DQ, so saturation comes from the per-exposure
    frames.  A single frame is one arbitrary exposure of many, so this scans every readable
    ``_cal`` (falling back to ``_rate`` when no ``_cal`` opens -- brick's 72 ``_cal`` are empty
    FITS) and reports the spread.

    Returns a dict: ``sat_median`` / ``sat_max`` (saturated pixel FRACTION, over readable frames),
    ``n_frames`` (how many were read), ``kind`` ("_cal"/"_rate"), ``mask`` (the DQ mask of the
    worst frame), and ``source`` (that frame's filename)."""
    from astropy.io import fits
    # per-exposure MAST filenames are jw<prog><obs><visit>_..._mirimage_cal.fits (e.g.
    # jw02221001001_..._mirimage_cal.fits), NOT the o.obsid form jw02221-o001 -- so scope on the
    # jw<prog><obs> prefix.  An unscoped fallback would silently show a DIFFERENT obs's DQ.
    stem = f"jw{int(o.program):05d}{o.obs}"
    for kind in ("_cal", "_rate"):
        pat = f"{BASE}/{o.field}/mastDownload/**/{stem}*mirimage{kind}.fits"
        fracs = []      # (frac, mask, name) per readable frame of this product type
        for p in sorted(glob.glob(pat, recursive=True)):
            try:
                with fits.open(_used(p, f"MIRI {kind.lstrip('_')} frame (DQ)")) as h:
                    if "DQ" not in h:
                        continue
                    dq = h["DQ"].data
                    if dq is None:
                        continue
                    sat = (np.asarray(dq).astype(int) & 2) > 0     # SATURATED = bit 1 (value 2)
                    fracs.append((float(sat.mean()), sat, os.path.basename(p)))
            except (OSError, ValueError, KeyError, IndexError):
                continue
        if not fracs:
            continue                                              # try the next product type
        vals = np.array([f for f, _, _ in fracs])
        worst = max(fracs, key=lambda t: t[0])                    # display the most-saturated frame
        return dict(sat_median=float(np.median(vals)), sat_max=float(vals.max()),
                    n_frames=len(fracs), kind=kind, mask=worst[1], source=worst[2])
    return None


def miri_overview(o: Observation, filt=None):
    """MIRI basics for a MIRI observation: the MAST i2d image, a Spitzer side-by-side at the
    matching wavelength (IRAC 8 um / MIPS 24 um, reprojected onto the MIRI grid), and a saturation
    mask from the MAST DQ.  Picks the first MIRI filter with an i2d on disk if ``filt`` is None."""
    import matplotlib
    matplotlib.use("Agg")
    from astropy.io import fits
    from astropy.wcs import WCS
    from astropy.nddata import Cutout2D
    from astropy.visualization import ZScaleInterval, ImageNormalize, AsinhStretch
    filts = [filt] if filt else list(o.filters)
    mpath = None
    for f in filts:
        mpath = _miri_i2d(o, f)
        if mpath:
            filt = f; break
    metrics = dict(stage="miri", filt=filt)
    if not mpath:
        png = _red_flag_figure(o, "miri", "NO MIRI i2d ON DISK",
                               f"No MAST MIRI i2d in mastDownload/ for {o.obsid} "
                               f"(filters tried: {', '.join(filts)}).")
        metrics.update(red_flag=True, red_flag_reason="no MIRI i2d on disk", passed=False)
        return png, metrics

    with fits.open(_used(mpath, f"MIRI {filt} mosaic")) as h:
        sci = h["SCI"] if "SCI" in h else h[1]
        mdata = sci.data.astype("float32"); mwcs = WCS(sci.header)
    spz = _spitzer_for_miri(filt)
    sat = _saturation_mask(o)
    ncol = 1 + (1 if spz else 0) + (1 if sat else 0)
    fig, ax = _fig(1, ncol, 5.4, 5.2); fig.subplots_adjust(top=0.86, wspace=0.28)
    col = 0

    def _gray(a, d, title):
        # scale from finite, NON-ZERO pixels only: a MIRI mosaic can carry a wide exact-zero
        # border (brick o001 F2550W is ~19% zeros) that drags ZScale limits so far the real
        # ~90-count signal flattens to invisible.
        sample = d[np.isfinite(d) & (d != 0)]
        ref = sample if sample.size else np.nan_to_num(d)
        norm = ImageNormalize(ref, interval=ZScaleInterval(), stretch=AsinhStretch())
        a.imshow(d, origin="lower", cmap="gray", norm=norm)
        a.set_xticks([]); a.set_yticks([]); a.set_title(title, fontsize=9)

    a0 = ax[0][col]; col += 1
    _gray(a0, mdata, f"MIRI {filt} MAST i2d")

    if spz:                                              # Spitzer, reprojected onto the MIRI grid
        lbl, spath = spz
        a1 = ax[0][col]; col += 1
        try:
            with fits.open(_used(spath, "Spitzer comparison image")) as h:
                hd = h[0] if h[0].data is not None else h[1]
                sd = hd.data.astype("float32"); swcs = WCS(hd.header)
            cen = mwcs.pixel_to_world(mdata.shape[1] / 2, mdata.shape[0] / 2)
            # cut a generous window around the MIRI centre (with slack for the ~266 deg PA
            # rotation) so reproject has enough coverage, then resample onto the MIRI WCS.
            from astropy.wcs.utils import proj_plane_pixel_scales
            mscale = float(np.mean(proj_plane_pixel_scales(mwcs)))
            sscale = float(np.mean(proj_plane_pixel_scales(swcs)))
            size = int(max(mdata.shape) * mscale / sscale * 1.6)
            px, py = swcs.world_to_pixel(cen)
            cut = Cutout2D(sd, (float(px), float(py)), max(size, 20), wcs=swcs, mode="trim")
            # GLIMPSE (GLON-CAR) / MIPSGAL (RA-TAN) are ~quarter-turn from the MIRI PA; reproject
            # onto the MIRI WCS+shape so the two panels share orientation and pixel grid.
            reprojected = False
            panel = cut.data
            from reproject import reproject_interp
            try:
                rep, _ = reproject_interp((cut.data, cut.wcs), mwcs, shape_out=mdata.shape)
                panel = rep; reprojected = True
            except (ValueError, MemoryError, RuntimeError, TypeError):
                panel = cut.data                             # fall back to the un-reprojected cut
            # coverage is measured on the panel ACTUALLY DRAWN: an archival mosaic that does not
            # overlap the field (e.g. sickle o001 vs MIPS24), or that only partly overlaps after the
            # reprojection onto the MIRI grid, leaves the drawn panel mostly NaN -- do NOT then claim
            # a matched footprint.  (Measuring the raw cutout instead over-counts coverage, since the
            # generous pre-reproject window can be half-full while the reprojected panel is far less.)
            covered = float(np.isfinite(panel).mean()) >= 0.5
            metrics["spitzer_panel_finite_frac"] = float(np.isfinite(panel).mean())
            matched = bool(reprojected and covered)
            note = ("\n(same footprint)" if matched
                    else "\n(not covered)" if not covered
                    else "\n(not reprojected)")
            _gray(a1, panel, f"{lbl}{note}")
            metrics["spitzer"] = os.path.basename(spath)
            metrics["spitzer_footprint_matched"] = matched
        except (ValueError, IndexError, OSError):
            a1.text(0.5, 0.5, f"{lbl}\nfootprint not covered", ha="center", va="center", fontsize=8)
            a1.axis("off")

    if sat:                                              # saturation mask from MAST DQ
        a2 = ax[0][col]; col += 1
        a2.imshow(sat["mask"], origin="lower", cmap="Reds", vmin=0, vmax=1)
        a2.set_xticks([]); a2.set_yticks([])
        a2.set_title(f"saturated pixels (MAST DQ)\nmedian {100 * sat['sat_median']:.2f}%, "
                     f"max {100 * sat['sat_max']:.2f}% over {sat['n_frames']} {sat['kind']} frames",
                     fontsize=8)
        metrics.update(sat_median=sat["sat_median"], sat_max=sat["sat_max"],
                       sat_n_frames=sat["n_frames"], sat_kind=sat["kind"], sat_source=sat["source"])
    # MIRI basics is a display panel, not a numeric gate; the one condition that carries information
    # is whether the primary product -- the MIRI i2d -- actually rendered usable data.  A blank or
    # mostly-empty mosaic (finite fraction below 20%) does NOT pass, so a degenerate i2d is
    # distinguishable from a real one rather than both reading passed=True.  Record the component
    # states so a reader can tell a complete panel from a partial one (Spitzer present / footprint
    # matched / saturation product present).
    miri_finite_frac = float(np.isfinite(mdata).mean())
    metrics.update(miri_finite_frac=miri_finite_frac,
                   spitzer_present=bool(spz), sat_present=bool(sat), red_flag=False)
    metrics["passed"] = bool(miri_finite_frac > 0.2)
    fig.suptitle(f"{o.target} {o.obsid} — MIRI {filt} basics", fontsize=11, y=0.98)
    return _save(fig, f"{o.obsid}_miri.png"), metrics


# --------------------------------------------------------------------------- STAGE 6
def _internal_pos_rms(o, filt):
    """(mag_vega, internal-position-rms in mas) per star from the highest-tier catalog carrying
    ``std_ra_<filt>``/``std_dec_<filt>`` (the empirical scatter of a star's position ACROSS
    exposures, in deg) + ``mag_vega_<filt>``.  This is rms(jwst) -- JWST internal repeatability.
    Returns None if unavailable."""
    from astropy.io import fits
    from astropy.table import Table
    sr = f"std_ra_{filt.lower()}"; sd = f"std_dec_{filt.lower()}"
    mv = f"mag_vega_{filt.lower()}"; sccol = f"skycoord_{filt.lower()}"
    best = None; rank = (-1.0, -1.0)
    for p, kind, tier, mtime in _catalog_candidates(o):
        if (tier, mtime) <= rank:
            continue
        try:
            hdr = fits.getheader(p, ext=1)
        except (OSError, IndexError):
            continue
        low = {str(hdr.get(f"TTYPE{i}", "")).lower() for i in range(1, hdr.get("TFIELDS", 0) + 1)}
        if sr in low and sd in low and mv in low:
            best = p; rank = (tier, mtime)
    if best is None:
        return None
    try:
        t = Table.read(_used(best, f"merged catalogue, per-star position scatter ({filt})"))
    except (OSError, ValueError):
        return None
    ra_std = np.asarray(t[sr], float); de_std = np.asarray(t[sd], float)
    m = np.asarray(t[mv], float)
    cosd = (float(np.cos(np.radians(np.nanmedian(t[sccol].dec.deg))))
            if sccol in t.colnames else 1.0)
    rms = np.hypot(ra_std * cosd, de_std) * 3.6e6 / np.sqrt(2.0)   # deg -> mas, PER-AXIS (match σ_pos)
    # a position scatter is only meaningful with several exposures; 1-2 detections give a
    # degenerate std (0 or near-0).  Require >=3 detections AND drop unphysically-tiny values
    # (< 0.1 mas, well below the real ~1 mas internal floor) so a degenerate tail can't drag the
    # binned median toward zero at the faint end.
    g = np.isfinite(rms) & np.isfinite(m) & (rms > 0.1)
    nmcol = f"nmatch_{filt.lower()}"
    if nmcol in t.colnames:
        g &= np.asarray(t[nmcol], float) >= 3
    return (m[g], rms[g]) if int(g.sum()) >= 50 else None


# peppar (Matt Hosek's WebbPSF PSF photometry/astrometry) products live under a DIFFERENT root than
# BASE for brick/cloudc (/blue), everything else on /orange -- mirrors fields.yaml.
_PEPPAR_ROOTS = {"brick": "/blue/adamginsburg/adamginsburg/jwst",
                 "cloudc": "/blue/adamginsburg/adamginsburg/jwst"}
_PEPPAR_DEFAULT_ROOT = "/orange/adamginsburg/jwst"


def _peppar_cal_for_cat(catpath):
    """The per-frame cal.fits that a peppar ``*_iter1_cat.fits`` was fit on, or None.  Cat lives at
    ``<field>/peppar/<FILT>/<DET>/<exp>_<det>_cal_<field>_iter1_cat.fits``; the cal sits at
    ``<field>/<FILT>/pipeline/<exp>_<det>_cal.fits``."""
    base = os.path.basename(catpath)
    m = re.match(r"(.+_cal)_.*_iter1_cat\.fits$", base)          # strip the _<field>_iter1_cat tail
    if not m:
        return None
    calname = m.group(1) + ".fits"
    filt_dir = os.path.dirname(os.path.dirname(os.path.dirname(catpath)))   # .../peppar
    field_dir = os.path.dirname(filt_dir)
    filt = os.path.basename(os.path.dirname(os.path.dirname(catpath)))      # <FILT>
    for cand in (f"{field_dir}/{filt}/pipeline/{calname}",
                 f"{field_dir}/images-merged/{calname}"):
        if os.path.isfile(cand):
            return cand
    return None


def _peppar_frame_std(pdir, pixscale, max_frames=12, nbright=2000, tol_mas=40.0, min_frames=3,
                      exclude=None):
    """Frame-to-frame position scatter from the per-frame peppar catalogues, for fields with no
    combined starlist (every current field): the standard deviation of each star's SKY position
    across the exposures it appears in.  The exposures are dithered AND mosaicked, so a star sits at
    different detector pixels -- and appears on different chips -- in different frames; matching must
    be in SKY coordinates, using each frame's cal WCS.  Detections within ``tol_mas`` are grouped and
    a group seen in >= ``min_frames`` distinct exposures yields one scatter point.  Returns
    (mag, framestd_mas) or None.  Bounded (brightest ``nbright``/frame, ``max_frames`` exposures)."""
    from astropy.table import Table
    from astropy.io import fits
    from astropy.wcs import WCS
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        return None
    cats = _exclude_frames(sorted(glob.glob(f"{pdir}/*/*_iter1_cat.fits")), exclude)
    # cap the number of exposures (by exposure token), not raw files: keep all detectors of the
    # chosen exposures so a star's chip-to-chip appearances are all available.
    exps = []
    for c in cats:
        mo = re.search(r"(jw\d+_\d+_\d+)_nrc", os.path.basename(c))
        if mo and mo.group(1) not in exps:
            exps.append(mo.group(1))
    keep_exps = set(exps[:max_frames])
    ra, dec, mag, fid = [], [], [], []
    for c in cats:
        mo = re.search(r"(jw\d+_\d+_\d+)_nrc", os.path.basename(c))
        if not mo or mo.group(1) not in keep_exps:
            continue
        cal = _peppar_cal_for_cat(c)
        if cal is None:
            continue
        try:
            t = Table.read(c)
            with fits.open(cal) as hd:
                w = WCS(hd["SCI"].header if "SCI" in hd else hd[1].header)
        except (OSError, ValueError, KeyError):
            continue
        if not {"x_fit", "y_fit", "m"} <= set(t.colnames):
            continue
        m = np.asarray(t["m"], float); x = np.asarray(t["x_fit"], float); y = np.asarray(t["y_fit"], float)
        ok = np.isfinite(m) & np.isfinite(x) & np.isfinite(y)
        idx = np.where(ok)[0]
        idx = idx[np.argsort(m[idx])][:nbright]                 # brightest per frame
        if idx.size < 20:
            continue
        r, d = w.all_pix2world(x[idx], y[idx], 0)               # peppar x/y are 0-based pixels
        ra.append(np.asarray(r, float)); dec.append(np.asarray(d, float))
        mag.append(m[idx]); fid.append(np.full(idx.size, keep_exps and exps.index(mo.group(1))))
    if len(ra) < min_frames:
        return None
    ra = np.concatenate(ra); dec = np.concatenate(dec)
    mag = np.concatenate(mag); fid = np.concatenate(fid).astype(int)
    good = np.isfinite(ra) & np.isfinite(dec)
    ra, dec, mag, fid = ra[good], dec[good], mag[good], fid[good]
    if ra.size < 50:
        return None
    # group detections by sky position (single-link within tol_mas on the unit sphere)
    xyz = SkyCoord(ra * u.deg, dec * u.deg).cartesian.xyz.value.T
    chord = 2.0 * np.sin(np.radians(tol_mas / 3.6e6) / 2.0)     # tol in mas -> deg is /3.6e6
    tree = cKDTree(xyz)
    grp = np.full(ra.size, -1, int)
    gid = 0
    cosd = np.cos(np.radians(np.median(dec)))
    mags_out, std_out = [], []
    for i in range(ra.size):
        if grp[i] >= 0:
            continue
        members = [j for j in tree.query_ball_point(xyz[i], chord) if grp[j] < 0]
        for j in members:
            grp[j] = gid
        gid += 1
        mj = np.asarray(members)
        if np.unique(fid[mj]).size < min_frames:               # seen in >= min_frames exposures
            continue
        dra = (ra[mj] - ra[mj].mean()) * cosd * 3.6e6           # deg -> mas
        dde = (dec[mj] - dec[mj].mean()) * 3.6e6
        prec = float(np.hypot(np.std(dra), np.std(dde)) / np.sqrt(2.0))   # per-axis
        if prec > 0:
            mags_out.append(float(np.median(mag[mj]))); std_out.append(prec)
    if len(std_out) < 50:
        return None
    return np.array(mags_out), np.array(std_out)


def _exclude_frames(cats, exclude):
    """Drop per-frame catalogues whose exposure token (``jw…_<vgroup>_<exp>``) is in ``exclude``."""
    if not exclude:
        return cats
    def _tok(c):
        mo = re.search(r"(jw\d+_\d+_\d+)_nrc", os.path.basename(c))
        return mo.group(1) if mo else None
    return [c for c in cats if _tok(c) not in exclude]


def _peppar_precision(o: Observation, filt, max_frames=48, exclude=None):
    """Independent peppar astrometric-precision series vs instrumental magnitude, as a dict with the
    two quantities peppar carries (either key may be absent, whichever products exist):
      ``frame_std`` : (mag, prec_mas) -- the combined starlist's ACROSS-FRAME position scatter
                      (``x_wcs_std``/``y_wcs_std``, arcsec->mas): the **frame-to-frame standard
                      deviation**, i.e. the achieved repeatability.
      ``formal``    : (mag, prec_mas) -- the per-frame FORMAL PSF-fit position error
                      (``x_err``/``y_err``, px->mas): the predicted, noise-limited precision.
    Peppar mags are instrumental (no Vega zero-point).  None if no peppar products for this
    obs's field/filter."""
    from astropy.table import Table
    if not filt:
        return None
    pdir = f"{_PEPPAR_ROOTS.get(o.field, _PEPPAR_DEFAULT_ROOT)}/{o.field}/peppar/{filt}"
    if not os.path.isdir(pdir):
        return None
    pixscale = 63.0 if filt.upper() in _LW_PREF else 31.0        # NIRCam LW / SW mas per pixel

    def _pool(paths, mcol, xcol, ycol, scale):
        rows = []
        for p in paths:
            try:
                t = Table.read(p)
            except (OSError, ValueError):
                continue
            if all(k in t.colnames for k in (mcol, xcol, ycol)):
                prec = np.hypot(np.asarray(t[xcol], float), np.asarray(t[ycol], float)) * scale / np.sqrt(2.0)
                rows.append((np.asarray(t[mcol], float), prec))
        if not rows:
            return None
        m = np.concatenate([r[0] for r in rows]); pr = np.concatenate([r[1] for r in rows])
        g = np.isfinite(m) & np.isfinite(pr) & (pr > 0)
        return (m[g], pr[g]) if int(g.sum()) >= 50 else None

    out = {}
    # A combined starlist bakes in ALL exposures, so it cannot honour ``exclude``; when excluding,
    # skip it and take the per-frame computed scatter (which can drop the bad frames) instead.
    combos = [] if exclude else glob.glob(f"{pdir}/combo_starlist_{filt}_*.fits")
    if combos:
        # x_wcs/y_wcs are tangent-plane offsets in ARCSEC, not degrees: x_wcs spans 70.5" over a
        # 2227 px detector = 0.0317"/px = the SW pixel scale, so x_wcs_std is arcsec and the factor
        # to mas is 1e3 (a median x_wcs_std of 0.004" -> 4.0 mas, matching xe's 0.074 px -> 2.3 mas).
        fs = _pool(combos, "m", "x_wcs_std", "y_wcs_std", 1.0e3)    # arcsec -> mas
        if fs is not None:
            out["frame_std"] = fs
    if "frame_std" not in out:
        # no combined starlist (e.g. cloud E/F): COMPUTE the frame-to-frame scatter from the
        # per-frame catalogues by cross-matching each star across the dithered exposures.
        cf = _peppar_frame_std(pdir, pixscale, exclude=exclude)
        if cf is not None:
            out["frame_std"] = cf
    cats = _exclude_frames(sorted(glob.glob(f"{pdir}/*/*_iter1_cat.fits")), exclude)[:max_frames]
    fm = _pool(cats, "m", "x_err", "y_err", pixscale)              # px -> mas
    if fm is not None:
        out["formal"] = fm
    return out or None


def _stage6_figure(o: Observation, sw, lw, exclude=None, png_suffix=""):
    """Astrometric precision vs Vega magnitude, one set of curves per channel (SW / LW):
      - solid  formal sigma_fit -- the PSF fitter's formal 1-sigma position error per detection.
               A formal error bar carries no systematic, so its ~0.06 mas bright-end floor is NOT
               the achieved precision; it is the noise-limited fit uncertainty.
      - dotted rms(jwst) internal -- the EMPIRICAL scatter of a star's position across exposures
               (merged std_ra/std_dec).  This is the achieved repeatability (a sub-mas floor, well
               above the formal sigma); the headline `floor_mas` is this number when available,
               else the formal floor (floor_is_empirical flag).
      - dashed rms(offset-VIRAC) -- external scatter against the reference frame (VIRAC floor incl.)
    The faint-end rise of all three tracks S/N.  A parallel lower panel histograms the source counts
    per Vega-mag bin (the sample behind each curve point)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    metrics = dict(stage=6, sw=sw, lw=lw)
    if exclude:
        metrics["excluded_exposures"] = sorted(e.split("_")[-1] for e in exclude)
    # LEFT column: two stacked panels sharing the magnitude axis -- the jicama precision curve on top
    # (a) and a parallel source-count histogram (sources per Vega-mag bin) below (ah), so the number
    # of stars behind each curve point -- and where the sample runs out at the faint end -- is visible.
    # RIGHT column: the same precision-vs-magnitude from the INDEPENDENT peppar catalogues (a2).
    fig = plt.figure(figsize=(11.2, 6.4))
    gs = fig.add_gridspec(2, 2, height_ratios=[3.0, 1.0], width_ratios=[1.0, 1.0],
                          hspace=0.08, wspace=0.28)
    a = fig.add_subplot(gs[0, 0])
    ah = fig.add_subplot(gs[1, 0], sharex=a)
    a2 = fig.add_subplot(gs[0, 1])                    # peppar curves (top-right)
    a2h = fig.add_subplot(gs[1, 1], sharex=a2)        # peppar source-count histogram (bottom-right)
    hist_series = []                 # (mag_used, colour, filt) for the count histogram below
    any_data = False
    all_vega = True
    for filt, color in [(sw, "#3366cc"), (lw, "#cc3311")]:
        if not filt:
            continue
        pooled = _pooled_daophot(o, filt, exclude=exclude)
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
        a.plot(ctr, med, "-", color=color, lw=1.7,
               label=lbl + r"  formal $\sigma_{\rm fit}$")
        a.fill_between(ctr, lo, hi, color=color, alpha=0.20)
        # This solid curve is the fitter's FORMAL 1-sigma position error, per detection -- it has no
        # systematic in it by construction, so it is NOT the achieved astrometric precision (that is
        # the empirical rms(jwst) dotted curve below, ~20x larger).  Record it under an explicit key
        # and make the headline `floor_mas` the EMPIRICAL floor (set in the rms(jwst) block), so a
        # reader of the metric gets the achieved precision, not the fit error (issue #1 review).
        metrics[f"formal_sigma_floor_mas_{filt.lower()}"] = float(np.nanmin(med))
        metrics[f"floor_mas_{filt.lower()}"] = float(np.nanmin(med))   # provisional; empirical overrides
        metrics[f"floor_is_empirical_{filt.lower()}"] = False
        metrics[f"nstars_{filt.lower()}"] = int(ok.sum())
        hist_series.append((mag[ok], color, filt))    # same sample as the curve, for the count panel
        # rms(offset): the EXTERNAL scatter vs VIRAC (includes the VIRAC error floor), dashed, same
        # colour -- shown alongside sigma_pos so "how precisely measured" vs "how well it agrees
        # with the external frame" are both visible.
        import astropy.units as u
        ref = _viraccache_path(o) or _refcat_path(o)
        ep = _obs_epoch(o, _mosaic_path(o, filt))
        ref_sc, _ = (aa.load_reference(_used(ref, "VIRAC2/Gaia reference catalogue"), ep)
                     if (ref and ep) else (None, None))
        if ref_sc is not None:
            idx, sep, _ = ref_sc.match_to_catalog_sky(_sc)      # anchor on sparse VIRAC
            keep = (sep < 0.15 * u.arcsec).nonzero()[0]
            if keep.size >= 50:
                cosd = float(np.cos(np.radians(np.median(_sc[idx[keep]].dec.deg))))
                dra = (_sc[idx[keep]].ra - ref_sc[keep].ra).to(u.mas).value * cosd
                dde = (_sc[idx[keep]].dec - ref_sc[keep].dec).to(u.mas).value
                # PER-AXIS to match sigma_pos (which is hypot(sig_ra,sig_de)/sqrt2): the radial
                # residual hypot(dra',dde') is divided by sqrt(2) so all three curves share one
                # per-axis convention on the same axis.
                resid = np.hypot(dra - np.median(dra), dde - np.median(dde)) / np.sqrt(2.0)
                rms, rctr = _binned_rms(mag[idx[keep]], resid)
                if rms is not None:
                    a.plot(rctr, rms, "--", color=color, lw=1.5, alpha=0.9,
                           label=f"{filt}  rms(offset−VIRAC)")
                    metrics[f"rms_offset_floor_mas_{filt.lower()}"] = float(np.nanmin(rms))
        # rms(jwst): the INTERNAL per-star position scatter across exposures (merged-catalog
        # std_ra/std_dec, deg -> mas), median vs mag -- the empirical JWST repeatability, distinct
        # from the formal sigma_pos and from the external rms(offset-VIRAC).  SKIPPED in the
        # exposure-excluded recompute: the merged std bakes in ALL exposures and cannot be re-derived
        # here without re-matching, so the peppar frame-to-frame σ carries the achieved-repeatability
        # story in the clean figure instead.
        jr = None if exclude else _internal_pos_rms(o, filt)
        if jr is not None:
            jmag_v, jrms = jr
            med_j, _, _, ctr_j = _binned_stat(jmag_v, jrms)
            if med_j is not None:
                a.plot(ctr_j, med_j, ":", color=color, lw=2.2, alpha=0.95,
                       label=f"{filt}  rms(jwst) internal — achieved repeatability")
                metrics[f"rms_jwst_floor_mas_{filt.lower()}"] = float(np.nanmin(med_j))
                # the ACHIEVED precision: promote the empirical floor to the headline metric.
                metrics[f"floor_mas_{filt.lower()}"] = float(np.nanmin(med_j))
                metrics[f"floor_is_empirical_{filt.lower()}"] = True
    metrics["mag_kind"] = "vega" if all_vega else "mixed"
    if not any_data:
        plt.close(fig)          # close the empty curve fig before the red-flag builds its own
        reason = "no per-exposure DAOPHOT catalogs on disk for this obs/filter"
        png = _red_flag_figure(o, "stage6" + png_suffix, "ASTROMETRIC-ERROR CURVE UNAVAILABLE",
                               f"Cannot build the precision-vs-magnitude curve: {reason}.")
        metrics.update(red_flag=True, red_flag_reason=reason, passed=False)
        return png, metrics
    a.set_yscale("log")
    a.set_ylim(0.03, 300.0)          # 0.03-300 mas: floor through S/N rise incl. faint rms(offset)
    xlbl = ("Vega magnitude" if all_vega else
            "magnitude  (Vega where calibrated, else instrumental)")
    a.set_ylabel("astrometric error (mas)")
    a.legend(fontsize=9, loc="upper left")
    a.grid(alpha=0.25, which="both")
    a.set_title(f"{o.target} {o.obsid} — jicama astrometric precision", fontsize=10)
    # LEFT-BOTTOM panel: number of sources per Vega-mag bin (same 0.5-mag binning as the curve, same
    # sample), one step histogram per channel.  Shows how many stars each curve point rests on and
    # where the sample dies out at the faint end.
    allmag = np.concatenate([m[np.isfinite(m)] for m, _c, _f in hist_series]) if hist_series else np.array([])
    if allmag.size:
        bins = np.arange(np.floor(allmag.min()), np.ceil(allmag.max()) + 0.5, 0.5)
        mids = 0.5 * (bins[:-1] + bins[1:])
        for m, color, filt in hist_series:
            cnt, _ = np.histogram(m[np.isfinite(m)], bins=bins)
            ah.step(mids, cnt, where="mid", color=color, lw=1.6)
            if cnt.max() > 0:
                metrics[f"lf_peak_mag_{filt.lower()}"] = float(mids[int(np.argmax(cnt))])
        ah.set_yscale("log")
        ah.set_ylabel("sources / 0.5 mag")
        ah.grid(alpha=0.25, which="both")
    ah.set_xlabel(xlbl)

    # RIGHT column: the same precision-vs-magnitude from the INDEPENDENT peppar (Hosek WebbPSF)
    # catalogues -- a cross-check that does not share jicama's detection/fit choices.  Two curves per
    # channel: the per-frame FORMAL PSF-fit error (dashed, predicted precision) and the combined
    # starlist's FRAME-TO-FRAME standard deviation (solid, achieved repeatability -- the analogue of
    # the jicama rms(jwst) curve).  Peppar mags are instrumental (no Vega ZP).  A parallel
    # source-count histogram (bottom-right) mirrors the jicama one.
    pep_any = False
    pep_kinds = set()
    pep_hist = []                    # (mag, colour, filt) behind the primary (frame_std) curve
    for filt, color in [(sw, "#3366cc"), (lw, "#cc3311")]:
        if not filt:
            continue
        pp = _peppar_precision(o, filt, exclude=exclude)
        if not pp:
            continue
        fl = filt.lower()
        fs = pp.get("frame_std")
        hist_mag = None                          # the sample to histogram for this channel
        if fs is not None:
            fmag, fprec = fs
            ok = fprec < 500.0
            med, lo, hi, ctr = _binned_stat(fmag[ok], fprec[ok])
            if med is not None:
                pep_any = True; pep_kinds.add("frame-to-frame σ")
                a2.plot(ctr, med, "-", color=color, lw=1.9,
                        label=f"{filt} frame-to-frame σ (n={int(ok.sum())})")
                a2.fill_between(ctr, lo, hi, color=color, alpha=0.20)
                metrics[f"peppar_framestd_floor_mas_{fl}"] = float(np.nanmin(med))
                metrics[f"peppar_floor_mas_{fl}"] = float(np.nanmin(med))   # headline = achieved
                hist_mag = fmag[ok]
        fm = pp.get("formal")
        if fm is not None:
            gmag, gprec = fm
            ok = gprec < 500.0
            med, lo, hi, ctr = _binned_stat(gmag[ok], gprec[ok])
            if med is not None:
                pep_any = True; pep_kinds.add("per-frame formal σ_fit")
                a2.plot(ctr, med, "--", color=color, lw=1.4, alpha=0.9,
                        label=f"{filt} per-frame formal σ_fit")
                metrics[f"peppar_formal_floor_mas_{fl}"] = float(np.nanmin(med))
                metrics.setdefault(f"peppar_floor_mas_{fl}", float(np.nanmin(med)))
                hist_mag = gmag[ok]              # full per-frame detection sample (mirrors jicama)
        if hist_mag is not None:
            pep_hist.append((hist_mag, color, filt))
    if pep_any:
        a2.set_yscale("log"); a2.set_ylim(0.03, 300.0)
        a2.set_ylabel("astrometric precision (mas)")
        a2.legend(fontsize=8.5, loc="upper left"); a2.grid(alpha=0.25, which="both")
        a2.set_title("peppar (independent) — " + " + ".join(sorted(pep_kinds)), fontsize=9.5)
        metrics["peppar_kind"] = ", ".join(sorted(pep_kinds))
        # BOTTOM-RIGHT: peppar source counts per instrumental-mag bin, mirroring the jicama panel.
        pallmag = (np.concatenate([m[np.isfinite(m)] for m, _c, _f in pep_hist])
                   if pep_hist else np.array([]))
        if pallmag.size:
            pbins = np.arange(np.floor(pallmag.min()), np.ceil(pallmag.max()) + 0.5, 0.5)
            pmids = 0.5 * (pbins[:-1] + pbins[1:])
            for m, color, filt in pep_hist:
                cnt, _ = np.histogram(m[np.isfinite(m)], bins=pbins)
                a2h.step(pmids, cnt, where="mid", color=color, lw=1.6)
                if cnt.max() > 0:
                    metrics[f"peppar_lf_peak_mag_{filt.lower()}"] = float(pmids[int(np.argmax(cnt))])
            a2h.set_yscale("log"); a2h.set_ylabel("sources / 0.5 mag")
            a2h.grid(alpha=0.25, which="both")
        a2h.set_xlabel("instrumental magnitude (peppar; no ZP)")
    else:
        a2.text(0.5, 0.5, "no peppar catalogues\nfor this obs/filter", ha="center", va="center",
                fontsize=10, style="italic", transform=a2.transAxes)
        a2.set_xticks([]); a2.set_yticks([])
        a2h.set_xticks([]); a2h.set_yticks([])
    if exclude:
        fig.suptitle(f"Stage 6 recomputed — EXCLUDING {len(exclude)} bad-PSF exposure(s) "
                     f"({', '.join(metrics['excluded_exposures'])}); rms(jwst) omitted",
                     fontsize=11, y=1.0)
    metrics["passed"] = True
    return _save(fig, f"{o.obsid}_stage6{png_suffix}.png"), metrics


def stage6_astrom_error(o: Observation, sw, lw):
    """Stage 6 = the astrometric-precision figure.  When stage 11 has flagged bad-PSF (streaked)
    exposures for this obs, ALSO build a SECOND, separate figure recomputed with those exposures
    excluded, so the reader sees how much the flagged exposures cost the precision.  The clean figure
    is returned in ``metrics['clean_png']`` and posted under its own marker by ``main``."""
    png, metrics = _stage6_figure(o, sw, lw)
    if metrics.get("red_flag"):
        return png, metrics
    bad = set()
    for f in (sw, lw):
        if f:
            bad |= _streaked_exposures(o, f)
    if bad:
        cpng, cm = _stage6_figure(o, sw, lw, exclude=bad, png_suffix="clean")
        if not cm.get("red_flag"):
            metrics["clean_png"] = cpng
            metrics["excluded_exposures"] = cm.get("excluded_exposures")
            # carry the clean floors for the caption / metrics JSON
            for k, v in cm.items():
                if k.endswith("_floor_mas") or k.startswith(("floor_mas_", "peppar_framestd_floor",
                                                             "peppar_formal_floor", "rms_offset_floor",
                                                             "formal_sigma_floor")):
                    metrics[f"clean_{k}"] = v
    return png, metrics


# --------------------------------------------------------------------------- STAGE 11
# A streaked/broadened PSF (e.g. arches jw02045-o001 exposure 4, "tracking failed for a second")
# shows two ways in the peppar products: the per-frame PSF-fit quality-of-fit (qfit) jumps for that
# exposure's bright stars (the empirical PSF no longer fits), and the effective PSF built from that
# exposure's own stars is visibly fatter/washed-out.  Stage 11 shows BOTH, from OUR data (peppar
# per-frame catalogues + the cal images) -- no JWST1PASS run required.
_EPSF_HALF = 11                      # ePSF stamp half-size (px) -> 23x23
_EPSF_QFIT_STREAK_FACTOR = 2.0       # exposure qfit > this x the median-of-exposures = streak flag
_EPSF_LOG_VMIN = 0.003               # log-stretch floor (fraction of the peak) for the ePSF stamps


def _exposure_qfit(o: Observation, filt):
    """Median PSF-fit quality-of-fit of the bright stars in each exposure, pooled over detectors, from
    the peppar per-frame catalogues.  A streaked/broadened exposure fits the empirical PSF far worse,
    so its qfit spikes above the run's baseline.  Returns {exposure_token: (median_qfit, n_bright)}
    or {} if no peppar catalogues."""
    from astropy.table import Table
    pdir = f"{_PEPPAR_ROOTS.get(o.field, _PEPPAR_DEFAULT_ROOT)}/{o.field}/peppar/{filt}"
    if not os.path.isdir(pdir):
        return {}
    # SCOPE to this observation: a peppar filter dir can hold the frames of several observations
    # (gc2211 o023 sits beside o046/o049; cloudef o002 beside o005), so filter the exposure token to
    # this obs -- an unscoped glob would grid a DIFFERENT obs's exposures under this one's issue.
    pref = f"jw{int(o.program):05d}{o.obs}"
    per_exp = {}
    for c in sorted(glob.glob(f"{pdir}/*/*_iter1_cat.fits")):
        if not os.path.basename(c).startswith(pref):
            continue
        mo = re.search(r"(jw\d+_\d+_\d+)_nrc", os.path.basename(c))
        if not mo:
            continue
        try:
            t = Table.read(_used(c, f"peppar per-frame catalogue ({filt})"))
        except (OSError, ValueError):
            continue
        if not {"m", "qfit"} <= set(t.colnames):
            continue
        m = np.asarray(t["m"], float); q = np.asarray(t["qfit"], float)
        g = np.isfinite(m) & np.isfinite(q)
        if g.sum() < 50:
            continue
        bright = g & (m < np.nanpercentile(m[g], 20))     # brightest 20% -- cleanest fit regime
        per_exp.setdefault(mo.group(1), []).append(q[bright])
    out = {}
    for exp, arrs in per_exp.items():
        qq = np.concatenate(arrs)
        qq = qq[np.isfinite(qq)]
        if qq.size >= 30:
            out[exp] = (float(np.median(qq)), int(qq.size))
    return out


def _streaked_exposures(o: Observation, filt):
    """The set of exposure tokens (``jw…_<vgroup>_<exp>``) stage 11 flags as bad-PSF for this
    obs/filter: qfit above ``_EPSF_QFIT_STREAK_FACTOR`` x the median qfit across the run's exposures.
    Empty set when there is no peppar data or nothing is flagged."""
    qf = _exposure_qfit(o, filt)
    if not qf:
        return set()
    base = float(np.median([v[0] for v in qf.values()]))
    thr = _EPSF_QFIT_STREAK_FACTOR * base
    return {tok for tok, (q, _n) in qf.items() if q > thr}


def _daophot_key_for_token(token):
    """Map a peppar exposure token ``jw<prog><obs><visit>_<vgroup>_<exp>`` to the substring the
    per-exposure DAOPHOT catalogues carry for that same exposure, ``vgroup<vgroup>_exp<exp>``."""
    parts = token.split("_")
    return f"vgroup{parts[-2]}_exp{parts[-1]}" if len(parts) >= 3 else token


_DQ_SATURATED = 2                    # JWST DQ SATURATED flag (bit 1)


def _effective_psf(cal_path, half=_EPSF_HALF, nbright=40, thresh_sigma=30.0):
    """The effective PSF of one exposure/detector: the MEAN of the peak-normalised cutouts of its
    bright, isolated, UNSATURATED stars, detected directly on the cal image (independent of any fit).
    A streaked exposure yields a stretched/washed-out stamp.  Saturated stars are excluded (their
    flat-topped cores hide the trail and are the brightest, so they would otherwise dominate); the
    mean (not median) keeps the asymmetric trail a streak leaves.  Returns (stamp, n_stars) or
    (None, n_found)."""
    from astropy.io import fits
    from astropy.stats import sigma_clipped_stats
    try:
        from photutils.detection import DAOStarFinder
        from scipy.spatial import cKDTree
    except ImportError:
        return None, 0
    try:
        with fits.open(cal_path) as hd:
            img = np.asarray((hd["SCI"] if "SCI" in hd else hd[1]).data, float)
            dq = np.asarray(hd["DQ"].data, int) if "DQ" in hd else np.zeros(img.shape, int)
    except (OSError, ValueError, KeyError):
        return None, 0
    fin = np.isfinite(img)
    if fin.sum() < 1000:
        return None, 0
    _mean, med, std = sigma_clipped_stats(img[fin], sigma=3.0)
    tb = DAOStarFinder(fwhm=2.5, threshold=thresh_sigma * std)(img - med)
    if tb is None or len(tb) < 20:
        return None, 0
    x = np.asarray(tb["xcentroid"]); y = np.asarray(tb["ycentroid"]); flux = np.asarray(tb["flux"])
    xy = np.column_stack([x, y])
    d, _ = cKDTree(xy).query(xy, k=2)
    iso = d[:, 1] > 2.3 * half                          # isolated: no neighbour inside the stamp
    stamps = []
    for i in np.where(iso)[0][np.argsort(-flux[np.where(iso)[0]])]:   # brightest isolated first
        xi, yi = int(round(x[i])), int(round(y[i]))
        if xi - half < 0 or yi - half < 0 or xi + half + 1 > img.shape[1] or yi + half + 1 > img.shape[0]:
            continue
        if (dq[yi - 2:yi + 3, xi - 2:xi + 3] & _DQ_SATURATED).any():  # drop saturated cores
            continue
        c = (img - med)[yi - half:yi + half + 1, xi - half:xi + half + 1]
        if not (np.isfinite(c).all() and c.max() > 0):
            continue
        # peak must sit at the stamp centre: a cosmic ray in the corner would otherwise contribute a
        # full-amplitude off-centre pixel to the peak-normalised mean (the median used to absorb it).
        py, px = np.unravel_index(int(np.argmax(c)), c.shape)
        if abs(py - half) > 1 or abs(px - half) > 1:
            continue
        stamps.append(c / c.max())
        if len(stamps) >= nbright:
            break
    if len(stamps) < 10:
        return None, len(stamps)
    return np.mean(stamps, axis=0), len(stamps)


def _epsf_rms_radius(stamp):
    """Flux-weighted rms radius (px) of an ePSF stamp -- a size/breadth measure that separates a
    broadened (streaked) exposure from the sharp ones even when the axis ratio does not."""
    s = np.clip(np.asarray(stamp, float), 0, None)
    tot = s.sum()
    if tot <= 0:
        return None
    yy, xx = np.mgrid[0:s.shape[0], 0:s.shape[1]]
    cx = (xx * s).sum() / tot; cy = (yy * s).sum() / tot
    return float(np.sqrt((((xx - cx) ** 2 + (yy - cy) ** 2) * s).sum() / tot))


def stage11_effective_psf(o: Observation, sw, lw):
    """Effective PSF per exposure, to catch a streaked/broadened PSF (e.g. a momentary tracking
    failure).  For each exposure of a representative detector the empirical PSF is built by stacking
    its bright isolated stars (from the cal image), shown as a stamp; the exposure's peppar
    quality-of-fit -- which spikes when the PSF no longer fits -- labels each stamp and flags the bad
    exposures.  Independent of JWST1PASS (uses our peppar catalogues + cal images)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    metrics = dict(stage=11, sw=sw, lw=lw)
    filt = next((f for f in (sw, lw)
                 if f and os.path.isdir(f"{_PEPPAR_ROOTS.get(o.field, _PEPPAR_DEFAULT_ROOT)}"
                                        f"/{o.field}/peppar/{f}")), None)
    if filt is None:
        reason = "no peppar per-frame catalogues on disk for this obs/filter"
        png = _red_flag_figure(o, "stage11", "EFFECTIVE-PSF CHECK UNAVAILABLE",
                               f"Cannot build the per-exposure effective PSF: {reason}.")
        metrics.update(red_flag=True, red_flag_reason=reason, passed=False)
        return png, metrics
    metrics["filter"] = filt
    qf = _exposure_qfit(o, filt)          # {exp_token: (median_qfit, n)} pooled over detectors
    # a representative detector to show the ePSF stamps for (NRCA1 if present, else the first)
    pdir = f"{_PEPPAR_ROOTS.get(o.field, _PEPPAR_DEFAULT_ROOT)}/{o.field}/peppar/{filt}"
    dets = sorted(os.path.basename(d) for d in glob.glob(f"{pdir}/NRC*"))
    det = "NRCA1" if "NRCA1" in dets else (dets[0] if dets else None)
    pref = f"jw{int(o.program):05d}{o.obs}"          # scope to THIS obs (see _exposure_qfit)
    cats = (sorted(c for c in glob.glob(f"{pdir}/{det}/*_iter1_cat.fits")
                   if os.path.basename(c).startswith(pref)) if det else [])
    exps = []
    for c in cats:
        mo = re.search(r"(jw\d+_\d+_\d+)_nrc", os.path.basename(c))
        if mo and mo.group(1) not in exps:
            exps.append(mo.group(1))
    if not exps:
        reason = f"no exposures found under peppar {filt}/{det}"
        png = _red_flag_figure(o, "stage11", "EFFECTIVE-PSF CHECK UNAVAILABLE", reason + ".")
        metrics.update(red_flag=True, red_flag_reason=reason, passed=False)
        return png, metrics

    # The streak flag lives in ONE place: _streaked_exposures (shared with the stage-6 clean
    # recompute), so the two stages can never disagree on which exposures are bad.  qbase is kept
    # only for the display/metric baseline.
    qvals = np.array([qf[e][0] for e in exps if e in qf], float)
    qbase = float(np.median(qvals)) if qvals.size else None
    flagged = _streaked_exposures(o, filt)
    streaked = []

    from astropy.visualization import LogStretch, ImageNormalize
    # LOG stretch: compress the bright core hardest so the faint WINGS -- the six diffraction spikes,
    # and the broadened halo a streaked exposure grows -- are visible; asinh/sqrt still buried them.
    # Floor at _EPSF_LOG_VMIN of the peak (stamps are peak-normalised to 1).
    norm = ImageNormalize(vmin=_EPSF_LOG_VMIN, vmax=1.0, stretch=LogStretch())
    n = len(exps)
    ncol = min(6, n); nrow = int(np.ceil(n / ncol))
    fig, ax = plt.subplots(nrow, ncol, figsize=(2.35 * ncol, 2.9 * nrow), squeeze=False)
    for a in ax.flat:
        a.set_xticks([]); a.set_yticks([]); a.set_axis_off()
    nstars_by_exp = {}; rms_by_exp = {}
    for k, exp in enumerate(exps):
        a = ax.flat[k]; a.set_axis_on(); a.set_xticks([]); a.set_yticks([])
        cal = _peppar_cal_for_cat(f"{pdir}/{det}/{exp}_{det.lower()}_cal_{o.field}_iter1_cat.fits")
        e, ns = (_effective_psf(cal) if cal else (None, 0))
        nstars_by_exp[exp] = int(ns)
        rr = _epsf_rms_radius(e) if e is not None else None
        rms_by_exp[exp] = rr
        if e is not None:
            a.imshow(np.clip(e, 0, None), origin="lower", cmap="inferno", norm=norm)
        else:
            a.text(0.5, 0.5, "no ePSF", ha="center", va="center", fontsize=8,
                   style="italic", transform=a.transAxes)
        q = qf.get(exp, (None, 0))[0]
        bad = exp in flagged
        if bad:
            streaked.append(exp)
        lbl = exp.split("_")[-1]                       # the exposure number
        qtxt = f"qfit={q:.1f}" if q is not None else "qfit —"
        rtxt = f"  r={rr:.1f}px" if rr is not None else ""
        a.set_title(f"exp {lbl}  {qtxt}\n{ns} stars{rtxt}" + ("  ⚠STREAK" if bad else ""),
                    fontsize=8.5, color=("#c33" if bad else "black"))
    ntot = sum(nstars_by_exp.values())
    metrics.update(n_exposures=n, detector_shown=det,
                   qfit_baseline=qbase, qfit_by_exposure={e: qf[e][0] for e in exps if e in qf},
                   epsf_nstars_by_exposure={e.split("_")[-1]: nstars_by_exp[e] for e in exps},
                   epsf_nstars_total=ntot, epsf_nstars_median=int(np.median(list(nstars_by_exp.values()))),
                   epsf_rms_radius_by_exposure={e.split("_")[-1]: rms_by_exp[e] for e in exps
                                                if rms_by_exp[e] is not None},
                   streaked_exposures=[e.split("_")[-1] for e in streaked],
                   n_streaked=len(streaked))
    metrics["passed"] = (len(streaked) == 0)
    ttl = (f"{o.target} {o.obsid} — effective PSF per exposure ({filt}, {det}); each stamp is the "
           f"mean of bright, isolated, unsaturated stars (log stretch)")
    if streaked:
        ttl += f"  —  {len(streaked)} flagged exposure(s) (high qfit / broadened PSF)"
    fig.suptitle(ttl, fontsize=10, y=0.995)
    return _save(fig, f"{o.obsid}_stage11.png"), metrics


def _dispatch_stage(o, n, sw, lw):
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
    if n == 7:
        return stage7_mast_vs_pipeline(o, sw)
    if n == 8:
        return stage8_distortion(o, sw)
    if n == 9:
        return stage9_psf_vs_aper(o, sw)
    if n == 10:
        return stage10_photometric_consistency(o, sw, lw)
    if n == 11:
        return stage11_effective_psf(o, sw, lw)
    raise ValueError(n)


def build_stage(o, n, sw, lw):
    """Build stage ``n`` and attach the full path of every file it read as ``metrics['inputs']``.

    The single choke point for all nine stages, so no stage can be added without its provenance
    coming with it.  ``record_inputs`` runs even when the stage red-flags or raises partway: what
    it read up to that point is exactly what a reader needs to see."""
    with _recording_inputs() as rec:
        try:
            png, metrics = _dispatch_stage(o, n, sw, lw)
        except BaseException:
            # the caller turns this into a stage error record; the paths read before it failed are
            # the most useful thing about that record, so do not lose them
            _LAST_FAILED_INPUTS[:] = [dict(role=r, path=q) for r, q in rec]
            raise
    metrics["inputs"] = [dict(role=r, path=q) for r, q in rec]
    return png, metrics


# Paths read by the most recent stage that raised, so the error record in ``main`` can carry them.
_LAST_FAILED_INPUTS = []


# Every posted caption links its shorthand to docs/qa_methods.md so no term is left undefined.
# Templates carry the sentinel ``DOCROOT`` (kept out of ``str.format`` so the metric braces stay
# clean); ``_linkify`` swaps it for the live blob URL at the single return choke-point.
_DOC_REPO = os.environ.get("QA_REPO", "JWST-GC/data-qa")
_DOC_URL = f"https://github.com/{_DOC_REPO}/blob/main/docs/qa_methods.md"


def _linkify(s):
    return s.replace("DOCROOT", _DOC_URL)


# Link-safe headline fallback (the generic .split('.') fallback would truncate a URL at '.md').
_HEADLINE = {
    1: "**Stage 1 — first mosaics.**",
    2: "**Stage 2 — colour–magnitude diagram.**",
    3: "**Stage 3 — photometric calibration.**",
    4: "**Stage 4 — positional offsets (JWST catalogue − VIRAC).**",
    5: "**Stage 5 — inter-detector / inter-module agreement.**",
    6: "**Stage 6 — astrometric precision.**",
    7: "**Stage 7 — MAST vs pipeline.**",
    8: "**Stage 8 — distortion residual map.**",
    9: "**Stage 9 — PSF vs aperture photometry.**",
    10: "**Stage 10 — JWST1PASS across-exposure consistency.**",
    11: "**Stage 11 — effective PSF per exposure.**",
}

# Templates reached via the generic `CAPTIONS[n].format(...)` fallback in _caption_for_impl.  Only
# stages whose caption is NOT built in code live here (1, 3).  Stages 2/4/5/6/7 build their caption
# in code (variant- or availability-dependent), so no template exists for them -- avoids a dead
# duplicate that drifts.
CAPTIONS = {
    1: "**Stage 1 — first mosaics.** Grayscale {sw} (SW) and {lw} (LW) `i2d`. Confirms the "
       "observation was delivered and the mosaics are present and not obviously corrupt. "
       "([how this is made](DOCROOT#stage1))",
    3: "**Stage 3 — photometric calibration (zeropoint).** 2-D histogram (colour = star counts) of "
       "JWST {sw} catalogue magnitude vs [VIRAC Ks](DOCROOT#glossary-virac) for {n_matched} "
       "[cross-matched](DOCROOT#glossary-crossmatch) stars. The **cyan 1:1 line** (in the legend) "
       "is anchored on the densest stellar ridge; a well-calibrated catalogue lies along it. The "
       "measured slope is {slope:.2f} and the scatter about the locus is {scatter:.2f} mag. "
       "([how this is made](DOCROOT#stage3))",
}


# A directory holding more filenames than this is rendered as its path, its count and a sample.
# The per-exposure daophot sets are the reason: gc2211 F200W pools 592 catalogs, and at ~101
# characters a path that one role alone is ~60 kB against GitHub's 65 kB comment limit.  The
# COMPLETE list is always in the metrics JSON, which is where a script should read it from.
_INPUTS_SAMPLE_AT = 12
_INPUTS_SAMPLE_HEAD = 3
# Hard ceiling on the whole block.  The per-directory sampling above bounds each directory, but the
# number of ROLES is not bounded (stage 5 already emits 11), so a many-filter or many-detector field
# could in principle still grow past what GitHub accepts.  Past this size the block drops to one
# line per role.  A comment that fails to post carries no provenance at all.
_INPUTS_BLOCK_MAX = 20000


def _inputs_block_compact(by_role, distinct):
    """One line per role: its count and the directories it read from.  The fallback when the full
    listing would be too large to post."""
    lines = []
    for role, paths in by_role.items():
        dirs = sorted({os.path.dirname(q) for q in paths})
        shown = ", ".join(f"`{d}/`" for d in dirs[:3]) + (f" +{len(dirs) - 3} more" if len(dirs) > 3 else "")
        lines.append(f"- **{role}** — {len(paths)} file{'s' if len(paths) != 1 else ''} in {shown}")
    return (f"\n\n<details><summary>Files read for this stage ({len(distinct)})</summary>\n\n"
            + "\n".join(lines)
            + "\n\nToo many to list individually here. The complete list of full paths is the "
              "`inputs` key of this stage in `data_qa/metrics/<obsid>.json`.\n\n</details>")


def _inputs_block(metrics):
    """A collapsed ``<details>`` listing the full path of every file the stage read.

    Grouped by role, then by directory, so every path is reconstructible as
    ``<directory>/<filename>``.  A directory holding more than ``_INPUTS_SAMPLE_AT`` files is
    summarised (count + first few + last), and the block SAYS it summarised and points at the
    metrics JSON holding the complete list -- a silent truncation would read as "these are all the
    files the stage used", which is the claim this exists to make honestly."""
    inputs = metrics.get("inputs") or []
    if not inputs:
        return ""
    # A file read for two purposes is two entries (one per role), so the per-role counts below can
    # sum to more than the number of distinct files -- stage 5 reads the same 192 per-exposure
    # catalogs for the module positions, the S/N cut and the per-detector quiver.  The headline is
    # the DISTINCT count, since that is what "which files did this use" means, and the block says
    # so when the two differ.
    distinct = {d.get("path", "") for d in inputs}
    by_role = {}
    for d in inputs:
        by_role.setdefault(d.get("role", "input"), []).append(d.get("path", ""))
    lines, summarised = [], False
    for role, paths in by_role.items():
        lines.append(f"**{role}** — {len(paths)} file{'s' if len(paths) != 1 else ''}")
        by_dir = {}
        for q in paths:
            by_dir.setdefault(os.path.dirname(q), []).append(os.path.basename(q))
        for d, names in by_dir.items():
            lines.append(f"- `{d}/`")
            if len(names) > _INPUTS_SAMPLE_AT:
                summarised = True
                lines += [f"  - `{x}`" for x in names[:_INPUTS_SAMPLE_HEAD]]
                lines.append(f"  - … {len(names) - _INPUTS_SAMPLE_HEAD - 1} more not listed here …")
                lines += [f"  - `{names[-1]}`"]
            else:
                lines += [f"  - `{x}`" for x in names]
        lines.append("")
    tail = ""
    if len(inputs) > len(distinct):
        tail += (f"\n{len(inputs) - len(distinct)} of the entries above are the same file read for "
                 f"a second purpose, so the per-role counts sum to more than {len(distinct)}.")
    if summarised:
        tail += ("\nThe complete list, with nothing summarised, is the `inputs` key of this stage "
                 "in `data_qa/metrics/<obsid>.json`.")
    block = (f"\n\n<details><summary>Files read for this stage ({len(distinct)})</summary>\n\n"
             + "\n".join(lines) + tail + "\n\n</details>")
    return block if len(block) <= _INPUTS_BLOCK_MAX else _inputs_block_compact(by_role, distinct)


def caption_for(n, metrics):
    """Public entry: build the stage caption, swap the DOCROOT sentinel for the live doc URL, and
    append the full path of every file the stage read."""
    return _linkify(_caption_for_impl(n, metrics)) + _inputs_block(metrics)


def _caption_stage8(metrics):
    """Stage-8 caption.  Handles three states: not-applicable (no second band, so no pass/fail is
    set), a normal measured residual (amplitude + null-based significance), and a gross
    inter-filter offset (red-flagged).  Kept out of the generic red-flag branch, whose wording
    describes an empty plot; a red-flagged gross offset still renders a full map."""
    sw = metrics.get("sw", "SW")
    if metrics.get("measurable") is False:
        return ("**Stage 8 — inter-filter distortion residual: not applicable.** No second-filter "
                f"positions for {sw} in a merged catalogue, so there is no band to difference (a "
                "single-filter or not-yet-merged obs). No pass/fail is set. "
                "([how this is made](DOCROOT#stage8))")
    f2 = metrics.get("f2", "a 2nd filter")
    nS = metrics.get("n_stars"); rms = metrics.get("resid_rms_mas")
    amp = metrics.get("binned_amp90_mas"); null = metrics.get("null_amp90_mas")
    signif = metrics.get("amp90_significance"); frac = metrics.get("frac_gt_20mas")
    base = (f"**Stage 8 — inter-filter distortion residual ({sw} − {f2}).** The per-star position "
            f"difference between two JWST filters of the same field (the SAME source rows, "
            f"[S/N > 10](DOCROOT#glossary-snr) in both, field [bulk](DOCROOT#glossary-bulk) "
            f"removed) as a function of position. The two filters share the frames, the offsets "
            f"table, the DVA correction and the registration onto VIRAC, which leaves a per-filter "
            f"WCS (distortion) term measured **with no external catalogue in it**. Sources are "
            f"paired across bands upstream, by mutual nearest neighbour within ~100 mas in "
            f"`merge_catalogs` (visible as truncation near 100 mas) — ~100× the ~1 mas signal, so "
            f"the radius does little to shape this map. LEFT/MIDDLE: binned-median ΔRA/ΔDec maps; "
            f"RIGHT: per-cell quiver. A flat map = the two solutions agree; a coherent "
            f"gradient/swirl = a differential distortion residual. ")
    if nS is not None and rms is not None:
        base += f"Here: {nS} stars, {rms:.2f} mas per-star"
        if amp is not None and null is not None and signif is not None:
            base += (f"; the map's 90th-percentile cell amplitude is {amp:.2f} mas against a "
                     f"shuffled-position null of {null:.2f} mas, so {signif:.1f}× the null. That "
                     f"ratio is the significance quoted here; a per-cell standard error reads "
                     f"about 2× more optimistic and is not used")
        if frac is not None:
            base += (f"; {100 * frac:.1f}% of kept rows have |Δ| > 20 mas, from nearest-neighbour "
                     f"ambiguity inside the match radius, which inflates a standard error")
        base += ". "
    if metrics.get("red_flag"):
        base += f"🚩 {metrics.get('red_flag_reason', 'gross inter-filter offset')}. "
    return base + "([how this is made](DOCROOT#stage8))"


def _caption_for_impl(n, metrics):
    if n == "6clean":
        exps = ", ".join(metrics.get("excluded_exposures") or [])
        sw = metrics.get("sw"); lw = metrics.get("lw")
        base = (f"**Stage 6 (recomputed) — astrometric precision EXCLUDING bad-PSF exposures.** "
                f"The same curves as the stage-6 figure above, but with the exposure(s) "
                f"[stage 11](DOCROOT#stage11) flagged as streaked/broadened (**{exps}**) left out of "
                f"the per-exposure pools, so a momentary tracking failure no longer inflates the "
                f"precision. The **peppar frame-to-frame σ** carries the achieved-repeatability "
                f"story here (the jicama rms(jwst) curve is omitted — it comes from the merged "
                f"catalogue's all-exposure std, which cannot be re-derived per-exposure). ")
        for f in (sw, lw):
            if not f:
                continue
            fl = f.lower()
            full = metrics.get(f"peppar_framestd_floor_mas_{fl}")
            clean = metrics.get(f"clean_peppar_framestd_floor_mas_{fl}")
            if full is not None and clean is not None:
                base += (f"{f} peppar frame-to-frame σ floor: {full:.2f} → **{clean:.2f} mas** "
                         f"with the bad exposure(s) excluded. ")
        return base + "([how this is made](DOCROOT#stage6))"
    if n == 8:
        return _caption_stage8(metrics)
    # Stage 7 builds its own red-flag caption below (its red-flag cases still render a full figure,
    # so the generic "the plot is empty" wording would not fit).
    if metrics.get("red_flag") and n != 7:
        return (f"🚩 **Stage {n} — RED FLAG.** The plot is empty: "
                f"{metrics.get('red_flag_reason', 'no data to show')}. "
                f"An empty result here means the measurement could not be made — investigate. "
                f"([how this stage works](DOCROOT#stage{n}))")
    if n == 1 and (metrics.get("dropped_filters") or metrics.get("awaiting_reduction")):
        base = CAPTIONS[1].format(**{k: (v if v is not None else float("nan"))
                                     for k, v in metrics.items() if k in ("sw", "lw")})
        notes = []
        if metrics.get("awaiting_reduction"):
            # a filter with a raw MAST i2d / catalogue but no reduced science mosaic yet: its panel
            # reads 'no i2d' while the MAST product still exists -- say so, don't imply data loss
            ar = ", ".join(metrics["awaiting_reduction"])
            mf = ", ".join(metrics.get("mosaic_filters") or []) or "none"
            notes.append(f"filter(s) with MAST/raw data but NO reduced science mosaic yet "
                         f"(awaiting reduction): {ar} — the representative SW/LW panels use a "
                         f"reduced filter ({mf}) instead")
        if metrics.get("dropped_filters"):
            # a nominal (proposed) filter with no product on disk at all must leave a trace
            notes.append(f"nominal filter(s) with no mosaic/catalogue on disk (observed but not "
                         f"reduced, or not delivered): {', '.join(metrics['dropped_filters'])}")
        return base + " NOTE: " + "; ".join(notes) + "."
    if n == 2:
        # Built in code so the three CMD variants (full CMD / single-filter LF / crossmatch) each
        # read correctly and a missing lf_turnover never drops the caption to a bare fragment.
        kind = str(metrics.get("kind", "catalog")).replace("_dedup", ""); ns = metrics.get("n_stars")
        nstr = f"{ns} stars" if ns is not None else "the catalogue"
        lf = metrics.get("lf_turnover")
        cat = f"the `{kind}` [catalog](DOCROOT#glossary-mtier)"
        if metrics.get("single_filter"):
            body = (f"**Stage 2 — {metrics.get('sw','SW')} luminosity function** from {cat} "
                    f"({nstr}): star counts vs magnitude (no colour — single filter). ")
        else:
            body = (f"**Stage 2 — colour–magnitude diagram (CMD)** from {cat} ({nstr}): "
                    f"LW magnitude vs (SW−LW) colour, with the "
                    f"[luminosity function](DOCROOT#glossary-lf) (star counts vs magnitude) as the "
                    f"right-side marginal. ")
        if lf is not None:
            body += (f"Turnover ≈ {lf:.1f} mag is a rough depth indicator "
                     f"(fainter turnover = deeper catalogue). ")
        if metrics.get("n_stars_hi_sn") is not None:
            body += (f"A second CMD below is limited to [S/N > 10](DOCROOT#glossary-snr) in both "
                     f"bands ({metrics['n_stars_hi_sn']} stars, turnover ≈ "
                     f"{metrics.get('lf_turnover_hi_sn', float('nan')):.1f} mag). ")
        if kind == "crossmatch":
            body += ("The colour width here is set by the positional cross-match tolerance. ")
        return body + "([how this is made](DOCROOT#stage2))"
    if n == 4:
        # Built in code so it renders cleanly whatever was measured, and gated on the CELL COUNT:
        # a field whose cells all agree has zero spread and must still pass.
        om = metrics.get("offset_med_mas"); nc = metrics.get("n_cells") or 0
        if om is None or nc == 0:
            return ("**Stage 4 — positional offsets.** The JWST−VIRAC offset could not be measured "
                    "(no usable spatial cells). ([how this is made](DOCROOT#stage4))")
        sp = metrics.get("offset_scatter_mas")
        nd = metrics.get("n_cells_dropped") or 0; ncf = metrics.get("n_cells_confirmed") or 0
        badf = metrics.get("bad_src_frac")
        unc = ("" if metrics.get("cell_map_unreliable") or sp is None
               else f", and the cells scatter by {sp:.0f} mas about it")
        om_str = f"{om:.1f}" if abs(om) < 10 else f"{om:.0f}"
        base = (f"**Stage 4 — positional offsets (JWST catalogue − VIRAC).** How far a star in the "
                f"JWST catalogue sits from the same star in [VIRAC](DOCROOT#glossary-virac), which "
                f"is in the Gaia frame — the **absolute** tie to the external reference. This is a "
                f"different quantity from the *internal* ties in stages [5](DOCROOT#stage5)/"
                f"[6](DOCROOT#stage6) (module-to-module, exposure-to-exposure), which can be small "
                f"even when this is large: a large value here means the frame needs re-tying to the "
                f"reference, not that the data are internally bad. The [**field** "
                f"offset](DOCROOT#glossary-bulk) is measured separately in each spatial cell, from "
                f"the peak of the histogram of [all JWST−VIRAC pair "
                f"separations](DOCROOT#glossary-xcorr) in that cell.\n\n"
                f"LEFT maps that offset across the mosaic; outlined cells "
                f"[deviate together](DOCROOT#glossary-adjacency) from the field value, grey cells "
                f"were not measurable.\n\n"
                f"MIDDLE plots the **per-cell** offsets as (ΔRA, ΔDec) points sized by source "
                f"count, with the field value in the title, a circle at the 75 mas gate, and "
                f"ΔRA/ΔDec marginal histograms.\n\n"
                f"The field offset is {om_str} mas over {nc} measured cells ({nd} without a "
                f"peak){unc}. ")
        # Cell-method breakdown (issue #38, cloudef o005): the cells scatter by ~arcsec, so the
        # per-cell histogram peaks are noise, not a tie or a real discontinuity.  Say the map failed;
        # then either quote the confident swept whole-field offset, or -- if even that has no peak --
        # say the offset is UNMEASURABLE (do NOT pass off the collapse-biased isolated median as a tie).
        if metrics.get("cell_map_unreliable"):
            base += (f"⚠️ **The per-cell map is unreliable here.** The cells scatter by "
                     f"{(sp or 0):.0f} mas (≈{(sp or 0) / 1000:.1f}″) — far more than any real tie or "
                     f"sub-region discontinuity — so the per-cell histogram peaks are spurious (VIRAC "
                     f"is too sparse over this field to peak reliably per cell), and **no "
                     f"spatial-discontinuity verdict is drawn**. ")
            if metrics.get("offset_unmeasurable"):
                ibn = metrics.get("isolated_bulk_n")
                base += (f"The whole-field swept cross-correlation also finds no confident peak, so "
                         f"the JWST−VIRAC offset is **not reliably measurable** over this field. The "
                         f"only clean-match estimate is the isolated-star bulk ({om_str} mas, "
                         f"n={ibn}), but with VIRAC this sparse a near-zero value can be a "
                         f"nearest-neighbour collapse rather than a true tie — read it as indicative "
                         f"only, and this observation does not auto-pass. ")
            else:
                base += (f"The offset quoted above ({om_str} mas) is instead the confident **swept "
                         f"whole-field cross-correlation**, which is density-immune. ")
        # Reliability + provenance guards (issue #37/#38): say plainly when the number is measured
        # on a pre-re-tie catalogue, or when the histogram peak disagrees with clean isolated matches.
        if metrics.get("catalog_stale"):
            base += (f"⚠️ **Measured on a stale catalogue.** The merged catalogue read "
                     f"(`{metrics.get('stale_catalog_name', '?')}`, dated {metrics.get('catalog_date')}) "
                     f"predates this field's current alignment ({metrics.get('alignment_date')}), so "
                     f"the offset shown is the **pre-re-tie** frame, not the reduction now on disk — "
                     f"re-run the cross-band merge and re-measure. ")
        if metrics.get("bulk_low_confidence"):
            ib = metrics.get("isolated_bulk_off_mas"); ibn = metrics.get("isolated_bulk_n")
            dg = metrics.get("bulk_vs_isolated_disagree_mas")
            base += (f"⚠️ **Low confidence.** The histogram-peak offset disagrees with clean, "
                     f"unambiguous isolated-star matches ({ib:.0f} mas, n={ibn}) by {dg:.0f} mas — "
                     f"VIRAC is sparse over this field and the peak may sit on a spurious lobe rather "
                     f"than the true bulk, so read the quoted value as indicative only. ")
        # Which of the two measurements the quoted number came from.  They differ: see the comment
        # in stage4_offsets on the dense-reference pull on the histogram peak.
        if metrics.get("bulk_source") == "same-star":
            ssn = metrics.get("same_star_npairs"); sss = metrics.get("same_star_scatter_mas")
            cellm = metrics.get("cell_off_med")
            base += (f"That offset is measured **star by star**, from {ssn} pairs that are each "
                     f"other's nearest neighbour in the two catalogues"
                     + (f", scattering {sss:.0f} mas per star" if sss is not None else "")
                     + (f". The per-cell histogram reads {cellm:.0f} mas for the same field: it "
                        f"works on every pair at once, and against a reference as dense as VIRAC "
                        f"the wrong pairs pile up in a way that pulls its peak by several mas. The "
                        f"pass gate tests the larger of the two" if cellm is not None
                        else "") + ". ")
        if ncf and not metrics.get("cell_map_unreliable"):
            base += (f"{ncf} adjacent cell(s) holding {100 * (badf or 0):.0f}% of the sources sit "
                     f"at a different offset from the rest of the field — an internal "
                     f"discontinuity, so this observation does NOT pass. ")
        base += ("The RIGHT panel, when present, is the NRCA-minus-NRCB offset, measured "
                 "[without any external catalogue](DOCROOT#glossary-reffree). ")
        if metrics.get("cell_map_unreliable"):
            # cells exist but are noise; the per-cell spatial-consistency check is void (not skipped
            # for lack of cells), and an unmeasurable offset is left for a human rather than passed.
            base += ("The per-cell spatial-consistency check is void because the cell map broke "
                     "down, so a subtle sub-region discontinuity could go unseen. "
                     "([how this is made](DOCROOT#stage4))")
        elif metrics.get("spatial_assessed", True):
            base += ("A pass needs a small field offset AND cells that agree with each other. "
                     "([how this is made](DOCROOT#stage4))")
        else:
            # whole-field fallback: only one cell measured, so the per-cell spatial check did not
            # run.  Say so, since a caption claiming it ran would be false.
            base += ("NOTE: too few stars in common to sub-divide the field, so the offset was "
                     "measured WHOLE-FIELD (one cell). A small value passes the magnitude gate; "
                     "the per-cell spatial-consistency check did not run, and a sub-region "
                     "discontinuity would go unseen here. ([how this is made](DOCROOT#stage4))")
        if str(metrics.get("source", "")).startswith("release-dao"):
            base += (" (Positions here come from a per-filter DAO catalogue. This observation has "
                     "yet to be photometrically catalogued, which is what stage 3 red-flags.)")
        return base
    if n == 5:
        # Built entirely in code, which lets it (a) gate the S/N>10 clause on the panel actually
        # being present (ov_hi), (b) get the panel POSITION right ("to its right", it is gs[0,2]),
        # and (c) survive a missing intermodule_diff without a KeyError.
        diff = metrics.get("intermodule_diff")
        diff_clause = (f" The [per-detector quiver](DOCROOT#glossary-quiver) shows an A–B diff of "
                       f"{diff:.1f} mas." if diff is not None else "")
        if metrics.get("intermodule_off") is None:
            # a legitimate single-module obs, or two modules sharing no stars to compare
            if metrics.get("single_module"):
                return (f"**Stage 5 — inter-detector agreement.** Single module "
                        f"({metrics['single_module']}) for this observation, so there is no "
                        f"NRCA–NRCB comparison to make and the "
                        f"[JWST-against-itself](DOCROOT#glossary-reffree) overlap panel is "
                        f"omitted.{diff_clause} ([how this is made](DOCROOT#stage5))")
            return ("**Stage 5 — inter-detector / inter-module agreement.** The NRCA–NRCB overlap "
                    "could not be measured (no shared stars in the NRCA∩NRCB dither overlap after "
                    "alignment), so that panel and the cutout gallery are omitted."
                    f"{diff_clause} How well the two modules agree is unverified for this "
                    "observation. ([how this is made](DOCROOT#stage5))")
        # overlap measured -> full caption; the S/N>10 panel is only present when ov_hi succeeded
        off = metrics.get("intermodule_off"); rms = metrics.get("intermodule_rms")
        no = metrics.get("n_overlap")
        # top row is quiver + all-stars + optional S/N>10; the all-stars panel is TOP-MIDDLE when
        # the S/N panel is present (3 cols), else TOP-RIGHT (2 cols).  The footprint is its own
        # full-width row below.
        ov_pos = "TOP-MIDDLE" if metrics.get("n_overlap_hi") else "TOP-RIGHT"
        base = ("**Stage 5 — inter-detector / inter-module agreement.** How far a star seen in one "
                "detector sits from the same star seen in another. The TOP-LEFT "
                "[per-detector quiver](DOCROOT#glossary-quiver) shows each detector's median "
                "residual **against VIRAC** (field offset removed), each arrow annotated with its "
                "matched-star count. VIRAC is the shared reference, so every detector gets a "
                "vector, including NRCB2, which shares no sky with NRCA at all; the NRCA−NRCB "
                f"difference is {(diff if diff is not None else float('nan')):.1f} mas. The "
                f"{ov_pos} panel compares [JWST against itself](DOCROOT#glossary-reffree) in the "
                f"NRCA∩NRCB overlap — {off:.1f} mas offset, {rms:.1f} mas scatter (ΔRA and ΔDec "
                f"combined) over {no} shared stars — with ΔRA/ΔDec marginal histograms.")
        if metrics.get("n_overlap_hi"):
            base += (f" The panel to its right repeats that comparison for "
                     f"[S/N > 10](DOCROOT#glossary-snr) stars ({metrics['n_overlap_hi']} stars, "
                     f"{metrics.get('intermodule_rms_hi', float('nan')):.1f} mas scatter), where "
                     f"the scatter measures how well the modules agree, with the centroid noise of "
                     f"faint stars taken out of it.")
        if metrics.get("n_overlap_footprint"):
            base += (" The full-width row below the panels maps the overlap stars on the sky, "
                     "coloured by per-star |A−B|. It shows the shared stars tracing the thin "
                     "NRCA∩NRCB dither-overlap strip, and flags any part of that strip where the "
                     "two modules agree less well.")
        if metrics.get("cutout_footprint_mismatch"):
            base += (" ⚠️ The BOTTOM cutout strip is empty because **no drizzled mosaic covers the "
                     "module-overlap zone — the catalogue and the mosaic are on disjoint footprints** "
                     "(a reduction mismatch, not a QA gap). ([how this is made](DOCROOT#stage5))")
        else:
            base += (" The BOTTOM strip shows overlap-star cutouts from the SW merged `i2d`. Where "
                     "the two modules agree, each star is one round PSF; where they disagree, it "
                     "doubles or elongates. ([how this is made](DOCROOT#stage5))")
        return base
    if n == 9:
        ni = metrics.get("n_isolated"); ac = metrics.get("aper_corr_med")
        sct = metrics.get("aper_psf_scatter")
        base = ("**Stage 9 — PSF vs aperture photometry.** The jicama catalogue reports PSF-fit "
                "fluxes; QA **re-measures** simple aperture photometry (local-annulus background) on "
                "the mosaic at the catalogue positions and compares them, restricted to **isolated** "
                "stars (nearest catalogue neighbour beyond the sky annulus) so a neighbour's light "
                "doesn't contaminate the aperture or its background. LEFT: aperture vs PSF "
                "instrumental mag with the 1:1 + aperture-correction line. RIGHT: (aperture − PSF) "
                "vs PSF mag, showing the full range. ")
        if ni is not None and ac is not None and sct is not None:
            base += (f"Here: {ni} isolated stars"
                     + (" (capped)" if metrics.get("n_capped") else "")
                     + f", aperture correction {ac:+.2f} mag, scatter {sct:.3f} mag")
            tf = metrics.get("frac_gt_0p3mag")
            if tf is not None:
                base += f", {100 * tf:.1f}% beyond ±0.3 mag"
            base += ". "
        base += ("A tight locus at a constant offset with a small tail means the two photometries "
                 "agree; large scatter or a heavy tail flags PSF-model or crowding problems. "
                 "([how this is made](DOCROOT#stage9))")
        return base
    if n == 7:
        if metrics.get("red_flag"):
            return (f"🚩 **Stage 7 — RED FLAG.** "
                    f"{metrics.get('red_flag_reason', 'MAST-vs-pipeline comparison unavailable')}. "
                    f"([how this is made](DOCROOT#stage7))")
        # counts in the COMMON WINDOW (fair), not the full-field totals
        nj = metrics.get("n_jicama_window"); nm = metrics.get("n_mast_window")
        mo = metrics.get("mast_offset_med_mas"); jo = metrics.get("jicama_offset_med_mas")
        base = ("**Stage 7 — MAST vs pipeline.** A comparison of the pipeline against the raw "
                "[MAST-delivered](DOCROOT#glossary-mtier) products. The TOP row shows the "
                "MAST level-3 `i2d` mosaic (before) next to our pipeline mosaic (after) over the "
                "same sky region. The BOTTOM-LEFT panel compares source counts — the "
                "[jicama](DOCROOT#glossary-jicama) catalogue vs the MAST catalogue "
                "(the MAST-delivered `_cat.fits` when archived, else approximated by running "
                "DAOStarFinder at 5σ on the MAST i2d — an approximation of the STScI L3 catalogue "
                "step, which uses segmentation with deblending). ")
        if nj is not None and nm is not None:
            base += f"jicama holds {nj} vs {nm} (MAST) in the compared window. "
        if metrics.get("jicama_is_release") is False:
            base += ("(NOTE: no merged/release jicama catalogue exists yet for this obs, so the "
                     "pipeline side falls back to the per-i2d MAST catalogue — both sides of this "
                     "comparison are MAST.) ")
        base += ("The BOTTOM-RIGHT panel (the main result) is each catalogue's "
                 "[offset from VIRAC](DOCROOT#glossary-bulk), found by coarse-aligning on the "
                 "[xcorr histogram peak](DOCROOT#glossary-xcorr) and taking the centre of the "
                 "per-star cloud")
        # Improvement clause is CONDITIONAL: assert tightening only when both offsets are measured
        # AND jicama is the smaller.  Otherwise report the numbers, or state that the comparison is
        # unavailable, and claim no improvement.
        if mo is not None and jo is not None:
            base += f" — {jo:.0f} mas (jicama) vs {mo:.0f} mas (MAST)"
            if jo < mo:
                base += ", the astrometric tightening the pipeline delivers over MAST. "
            else:
                base += " (MAST is as close to VIRAC as the pipeline here). "
        elif jo is not None:
            base += (f" — {jo:.0f} mas (jicama); the MAST comparison is unavailable "
                     f"(the MAST offset is unmeasurable, possibly MAST mis-registration). ")
        elif mo is not None:
            base += f" — {mo:.0f} mas (MAST); the pipeline offset is unmeasurable here. "
        else:
            base += ". "
        base += ("The cloud's WIDTH is bounded by the 0.1″ cross-match radius; "
                 "[stage 5](DOCROOT#stage5) and [stage 6](DOCROOT#stage6) are where the per-star "
                 "scatter is measured. ([how this is made](DOCROOT#stage7))")
        return base
    if n == 6:
        # Built in code so the rms(jwst) sentence is only stated when that curve is actually drawn:
        # without per-exposure catalogs (e.g. w51 o002, gc2211 o046) `floor_mas` falls back to the
        # FORMAL floor (floor_is_empirical false), and the generic template's unconditional
        # "floor_mas is the empirical rms(jwst)" would name a curve that is absent and misattribute
        # the metric -- the exact formal-sold-as-achieved conflation this stage was fixed to avoid.
        sw, lw = metrics.get("sw"), metrics.get("lw")
        emp = any(metrics.get(f"floor_is_empirical_{f.lower()}") for f in (sw, lw) if f)
        base = ("**Stage 6 — astrometric precision.** Error curves vs Vega magnitude per channel. "
                "**formal σ_fit** (solid) is the PSF fitter's formal per-detection position error; a "
                "formal error bar has no systematic in it, so its bright-end floor is the "
                "noise-limited fit uncertainty, **not** the achieved precision. **rms(offset)** "
                "(dashed) is the RMS of the per-star JWST−[VIRAC](DOCROOT#glossary-virac) offset "
                "(external scatter, incl. the VIRAC floor). ")
        if emp:
            base += ("**rms(jwst)** (dotted) is the empirical scatter of a star across exposures — "
                     "the **achieved internal repeatability** (sub-mas, well above the formal σ_fit), "
                     "and the number the headline `floor_mas` reports (`floor_is_empirical` true). ")
        else:
            base += ("The empirical **rms(jwst)** repeatability curve needs per-exposure catalogs, "
                     "absent for this obs, so it is not drawn and `floor_mas` falls back to the "
                     "formal σ_fit floor (`floor_is_empirical` false). ")
        base += ("All curves rise at the faint end with S/N; shaded band = 16–84th percentile. The "
                 "**lower-left panel** histograms the source counts per Vega-mag bin — the sample "
                 "behind each curve point. The **RIGHT column** repeats the precision-vs-magnitude "
                 "from the INDEPENDENT peppar (Hosek WebbPSF) catalogues (mags instrumental, no "
                 "zero-point): **per-frame formal σ_fit** (dashed, the noise-limited fit error) and "
                 "the **frame-to-frame σ** (solid, the standard deviation of each star's position "
                 "across the exposures — the achieved repeatability, ~20× the formal error), from "
                 "the combined starlist's across-exposure scatter where present, else computed by "
                 "cross-matching the per-frame catalogues. Its own source-count histogram sits below "
                 "it. ([how this is made](DOCROOT#stage6))")
        return base
    if n == 10:
        # Built in code so the floors are only quoted when a curve was actually drawn.  The
        # no-product case is handled by the generic red-flag branch above (n != 7).
        filt = metrics.get("filter", "?")
        xf = metrics.get("x_rms_floor_mas"); mf = metrics.get("mag_rms_floor")
        base = (f"**Stage 10 — JWST1PASS across-exposure consistency ({filt}).** From Jay Anderson's "
                f"`MATCHUP.XYMEEE`: for each of {metrics.get('n_stars', 0)} stars found in ≥2 of the "
                f"{metrics.get('n_exposures', 0)} exposures, the RMS of its position (X, Y) and "
                f"instrumental magnitude across those exposures, plus mean quality-of-fit, all vs "
                f"instrumental magnitude — the four panels of Jay's `show_matchup.sm`. A tight, flat "
                f"bright-end floor that rises only at the faint (S/N) and saturated (bright) ends is "
                f"\"in family\"; a raised or structured floor flags a photometric or distortion "
                f"problem in that filter's frames. Position RMS is in mas "
                f"({metrics.get('meta_pix_mas', _META_PIX_MAS):.0f} mas/META-pixel). ")
        if xf is not None and mf is not None:
            base += (f"Bright-end floors: {xf:.2f} mas in X, {mf:.3f} mag. ")
        if metrics.get("meta_scale_assumed_sw"):
            base += ("⚠️ Position RMS assumes the 32 mas/pixel **SW** META grid; this is an LW "
                     "filter, whose META grid is coarser — read the X/Y-RMS-in-mas as unconfirmed "
                     "until the LW grid scale is applied. ")
        if metrics.get("saturation_turnover_mag") is not None:
            base += (f"The mag-RMS doubles above its floor brighter than "
                     f"{metrics['saturation_turnover_mag']:.1f} mag (blue dashed — saturation onset). ")
        return base + "([how this is made](DOCROOT#stage10))"
    if n == 11:
        # Built in code so the streak-flag sentence is only stated when an exposure is actually
        # flagged.  The no-product case is the generic red-flag branch above.
        filt = metrics.get("filter", "?"); det = metrics.get("detector_shown", "?")
        ne = metrics.get("n_exposures", 0)
        base = (f"**Stage 11 — effective PSF per exposure ({filt}, {det}).** Each panel is the "
                f"empirical PSF of one exposure — the mean of its bright, isolated, **unsaturated** "
                f"stars stacked from the cal image (log stretch, so the wings show). A momentary "
                f"tracking failure or guide-star glitch **broadens** that exposure's stars, so its "
                f"stamp is fatter and more washed-out — a larger halo and a lower, less-peaked core — "
                f"than the sharp, six-spike NIRCam PSF of the good exposures. (The broadening is "
                f"roughly symmetric, not a clean elongation, so the flag is not an axis-ratio.) Each "
                f"stamp is labelled with its star count, its ePSF rms radius (`r`, the breadth), and "
                f"the exposure's peppar PSF-fit **quality-of-fit** (`qfit`), which spikes when the "
                f"empirical PSF no longer fits — the objective flag. Built from our own data (peppar "
                f"catalogues + cal images), independent of JWST1PASS. ")
        ns = metrics.get("n_streaked") or 0
        if ns:
            exps = ", ".join(metrics.get("streaked_exposures") or [])
            qb = metrics.get("qfit_baseline")
            base += (f"⚠️ **{ns} exposure(s) flagged:** {exps} — qfit above "
                     f"{_EPSF_QFIT_STREAK_FACTOR:.0f}× the run baseline"
                     + (f" ({qb:.1f})" if qb is not None else "") + ". Those exposures degrade the "
                     f"PSF-fit astrometry/photometry and are candidates to down-weight or drop. ")
        else:
            base += (f"All {ne} exposures have a consistent qfit (none flagged). ")
        return base + "([how this is made](DOCROOT#stage11))"
    try:
        return CAPTIONS[n].format(**{k: (v if v is not None else float("nan"))
                                     for k, v in metrics.items()})
    except (KeyError, ValueError):
        return _HEADLINE.get(n, f"**Stage {n}.**") + f" ([how this is made](DOCROOT#stage{n}))"


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


def _miri_obs_from_disk(program, obs, base=BASE):
    """Construct a MIRI Observation from the MAST MIRI i2d(s) on disk (portal-independent)."""
    from .observations import FIELDS, CURATED
    obsid = f"jw{int(program):05d}-o{obs}"
    for d in sorted(glob.glob(f"{base}/*/")):
        fld = os.path.basename(d.rstrip("/"))
        # recurse into mastDownload/JWST/<product>/ and accept any tile token (_t001_../_t003_)
        hits = glob.glob(f"{base}/{fld}/mastDownload/**/{obsid}_t*_miri_*_i2d.fits",
                         recursive=True)
        if not hits:
            continue
        filts = sorted({m.group(1).upper() for h in hits
                        if (m := re.search(r"miri_([a-z0-9]+)_i2d", os.path.basename(h).lower()))})
        cur = CURATED.get(obsid, {})
        return Observation(program=str(int(program)), obs=obs, target=FIELDS.get(fld, fld.title()),
                           release_field=fld, instrument="MIRI", filters=filts,
                           visits=cur.get("visits", []), epoch=cur.get("epoch", ""),
                           notes=cur.get("notes", ""))
    return None


def _miri_caption(metrics, repo):
    doc = f"https://github.com/{repo}/blob/main/docs/qa_methods.md#stagemiri"
    parts = [f"**MIRI {metrics.get('filt','')} basics.** MAST i2d image"]
    if metrics.get("spitzer"):
        # only claim a shared footprint when the Spitzer cutout was reprojected onto the MIRI grid
        # AND actually covers the field; otherwise the panel is only wavelength-matched.
        foot = (", reprojected onto the MIRI grid (same footprint)"
                if metrics.get("spitzer_footprint_matched") else " (footprint not matched)")
        parts.append("a Spitzer side-by-side at the matching wavelength (IRAC 8 µm below ~14 µm, "
                     "MIPS 24 µm above)" + foot)
    if metrics.get("sat_median") is not None:
        parts.append(f"a per-exposure saturation mask from the MAST DQ "
                     f"(median {100 * metrics['sat_median']:.2f}%, max {100 * metrics['sat_max']:.2f}% "
                     f"saturated over {metrics['sat_n_frames']} `{metrics['sat_kind']}` frames)")
    body = ", plus ".join(parts) if len(parts) > 1 else parts[0]
    if metrics.get("red_flag"):
        return (f"🚩 **MIRI basics — {metrics.get('red_flag_reason','no data')}.** "
                f"([how this is made]({doc}))")
    return f"{body}. ([how this is made]({doc}))"


def _run_miri(args):
    """Build + optionally post the MIRI overview for one observation."""
    mo = [o for o in registry(programs=[args.program])
          if o.obs == args.obs and o.instrument == "MIRI"]
    o = mo[0] if mo else _miri_obs_from_disk(args.program, args.obs)
    if o is None:
        print(f"no MIRI obs for program {args.program} obs {args.obs} (portal + on-disk empty)",
              file=sys.stderr)
        return 1
    if args.target:
        o = replace(o, target=args.target)
    with _recording_inputs() as rec:
        png, metrics = miri_overview(o)
    metrics["inputs"] = [dict(role=r, path=q) for r, q in rec]
    print(f"{o.obsid}: MIRI overview -> {png}  passed={metrics.get('passed')}")
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
    all_metrics["miri"] = metrics
    with open(mpath, "w") as fh:
        json.dump(all_metrics, fh, indent=2, default=_json_default)
    if args.post:
        try:
            from .post_diagnostics import post_stage, PostError
            post_stage(o, "miri", png, _miri_caption(metrics, args.repo), args.repo)
        except (PostError, OSError) as e:
            print(f"  MIRI: post FAILED (figure built OK): {e}", file=sys.stderr)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--program", required=True)
    ap.add_argument("--obs", required=True)
    ap.add_argument("--stage", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])
    ap.add_argument("--sw", default=None); ap.add_argument("--lw", default=None)
    ap.add_argument("--target", default=None, help="override display target (issue-title match)")
    ap.add_argument("--post", action="store_true", help="post/update the issue comments")
    ap.add_argument("--repo", default=os.environ.get("QA_REPO", "JWST-GC/data-qa"))
    ap.add_argument("--miri", action="store_true",
                    help="build the MIRI overview (i2d + Spitzer + saturation) instead of NIRCam stages")
    args = ap.parse_args(argv)

    if args.miri:
        return _run_miri(args)

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
    # Pick the CMD/QA filters from those that actually HAVE data on disk, not the program's nominal
    # filter list.  The portal lists all six Sgr A* filters, but only F212N+F405N have mosaics and
    # catalogs; picking blindly gave F212N+F444W -> a false "no catalog" red flag (issue #39).
    avail = _available_filters(o) or o.filters
    with_mosaic = _filters_with_mosaic(o)
    sw, lw = pick_filters(avail, args.sw, args.lw, prefer=with_mosaic)
    print(f"{o.obsid}: SW={sw} LW={lw} filters={o.filters} (with data: {avail}; "
          f"reduced mosaic: {with_mosaic})")
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
            all_metrics[f"stage{n}"] = dict(stage=n, error=f"{type(e).__name__}: {e}",
                                            passed=False, inputs=list(_LAST_FAILED_INPUTS))
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
                # Stage 6 emits a SECOND figure recomputed excluding stage-11-flagged bad-PSF
                # exposures; post it under its own marker so it sits beside, not over, the main one.
                if n == 6 and metrics.get("clean_png"):
                    post_stage(o, "6clean", metrics["clean_png"],
                               caption_for("6clean", metrics), args.repo)
                    print(f"  stage 6clean: {metrics['clean_png']}  "
                          f"excluding {metrics.get('excluded_exposures')}")
            except (PostError, OSError) as e:
                print(f"  stage {n}: post FAILED (figure built OK): {e}", file=sys.stderr)
    print(f"metrics -> {mpath}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
