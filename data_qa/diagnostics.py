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


def _dao_position_catalog(o: Observation, filt):
    """Per-filter RELEASE DAO catalog (positions only, no calibrated magnitude), obs-scoped, as a
    LAST-RESORT position source for the frame-tie check.  A field that has been DETECTED but not
    yet merged/calibrated (gc2211 o046: per-filter ``f200w_..._dao_basic_o046_vetted.fits`` exist,
    but no merged photometry table and no MAST ``_cat.fits``) still has real positions -> we can
    still measure the frame offset the user cares about, rather than red-flagging.  Prefers the
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


def _jwst_positions(o: Observation, filt):
    """Positions (+source label) for the frame-tie check ONLY (stage 4 needs no magnitude).
    Prefers a catalog WITH photometry (``_jwst_sources``: release merged -> MAST); falls back to a
    per-filter release DAO catalog (positions only) so a detected-but-not-yet-merged obs still gets
    a real offset measurement.  Returns (SkyCoord, label) or (None, None)."""
    from astropy.table import Table
    sc, _mag, src = _jwst_sources(o, filt)
    if sc is not None:
        return sc, src
    dp = _dao_position_catalog(o, filt)
    if dp:
        m = Table.read(dp)
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
    false "frame far off-tie" red flag (issues #7/#8/#28).  Prefer this obs's own tokened refcat;
    else the untokened full-field refcat; NEVER a different obs's tokened refcat (wrong footprint).

    NOTE (epoch-blindness): when several EPOCHS coexist this still takes the lexically-newest, not
    the one nearest the observation.  Only ngc6334 has multiple epochs today so nothing moves, but
    the rule is epoch-blind and should be revisited if per-epoch refcats proliferate.  Related:
    the untokened gc2211 refcat carries no pmRA/pmDE, so aa.load_reference does no PM propagation
    and _obs_epoch has no effect on the reference here (the ~128 mas tie is flat in dt regardless)."""
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
    # Record which NOMINAL (portal) filters have NO product on disk: pick_filters selects only
    # from filters that are present, so a filter that WAS observed but is not yet reduced would
    # otherwise vanish from QA with no trace (issue: F444W/F322W2 on Sgr A*).  Leaving the list
    # here keeps that visible.
    avail = _available_filters(o)
    dropped = [f for f in o.filters if f not in avail]
    png = _save(fig, f"{o.obsid}_stage1.png")
    metrics = dict(stage=1, sw=sw, lw=lw,
                   sw_present=bool(psw), lw_present=bool(plw),
                   finite_fraction=fracs,
                   nominal_filters=list(o.filters), available_filters=avail,
                   dropped_filters=dropped,
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
        hb = a.hexbin(col, mg, gridsize=120, bins="log", cmap="viridis", mincnt=1)
        a.set_xlabel(f"{sw} - {lw}"); a.set_ylabel(lw)
        a.set_xlim(np.nanpercentile(col, [1, 99]))
        ylo, yhi = np.nanpercentile(mg, [0.5, 99.5]); a.set_ylim(yhi, ylo)   # brighter up
        fig.colorbar(hb, cax=cax, label="log N")
        hh, edges = np.histogram(mg, bins=50); ctr = 0.5 * (edges[1:] + edges[:-1])
        amarg.step(hh, ctr, where="mid", color="k", lw=0.9)
        pk = ctr[int(np.argmax(hh))]
        amarg.axhline(pk, color="r", lw=0.8)
        amarg.set_xlabel(f"N\nturnover≈{pk:.1f}", fontsize=8)
        amarg.tick_params(labelleft=False, labelsize=7); amarg.margins(y=0)
        a.set_title(f"{tag} (n={int(np.sum(sel))})", fontsize=9)
        return float(pk)

    nrows = 2 if have_sn else 1
    fig = plt.figure(figsize=(8.2, 5.6 * nrows))
    gs = fig.add_gridspec(nrows, 3, width_ratios=[4.0, 1.15, 0.16], wspace=0.05, hspace=0.32)
    peak = _draw_cmd(gs, 0, g, "all stars")
    metrics.update(n_stars=int(g.sum()), lf_turnover=peak, sw_col=csw, lw_col=clw,
                   passed=int(g.sum()) > 500)
    if have_sn:
        peak_hi = _draw_cmd(gs, 1, hi, "S/N > 10 in both bands")
        metrics.update(n_stars_hi_sn=int(hi.sum()), lf_turnover_hi_sn=peak_hi)
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
    ref_sc, ref_mag = aa.load_reference(ref, ep) if (ref and ep) else (None, None)
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
                f"fitted slope={slope:.2f}  scatter={scat:.2f}", fontsize=10)
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
    # Report the JWST magnitude range if we have it; do NOT claim a VIRAC comparison that was not run.
    jmag = _jwst_sources(o, filt)[1]
    mag_note = ""
    if jmag is not None and np.isfinite(jmag).any():
        mag_note = (f"  (JWST {filt} spans {np.nanpercentile(jmag, 1):.1f}–{np.nanpercentile(jmag, 99):.1f} "
                    f"mag; no VIRAC magnitude comparison was made.)")
    return (f"footprints overlap but no common-star histogram peak (peak_ratio {pr:.2f} < "
            f"{aa.MIN_PEAK_RATIO}); cause not determined — plausibly a magnitude-range mismatch "
            f"(too few VIRAC-bright stars) or crowding, but that was not measured here.{mag_note}")


# Max cell-to-cell spread (robust) for a frame to count as internally consistent enough to PASS.
# Tied to the ~15-30 mas release tolerances, not the looser 75 mas absolute-offset gate: a field
# whose tie varies by more than this across the mosaic is not "within survey noise" no matter how
# small its median.
# A cell whose tie differs from the field median by more than this (mas) is "deviating".  Tied to
# the 15-30 mas release tolerance, well below the 75 mas absolute gate.
_CELL_SPREAD_MAX = 30.0
# A coherent (adjacency-confirmed) deviating region holding more than this FRACTION of the measured
# sources fails the consistency gate.  Small: a real ~100 mas sub-region tie is a proper-motion
# killer even at a few percent.
_CELL_BAD_FRAC = 0.02
# Require at least this fraction of the field's sources to sit in cells with a measurable peak,
# else the tie is not adequately sampled to pass.
_CELL_MIN_COVERAGE = 0.5
# Low peak floor: enough that a cell has SOME peak above chance, not aa.MIN_PEAK_RATIO -- that
# constant anti-correlates with source count (bg = median(H[H>0]) grows with chance pairs), so a
# 4.0 cut keeps the SPARSE population and drops the dense one, making the verdict depend on which
# side of a defect happens to be sparse (o046 vs o049; PR #54 review).  We instead accept any real
# peak, weight cells by SOURCE COUNT, and judge consistency by adjacency (below), which does not
# depend on the density-biased ratio.
_CELL_PR_FLOOR = 1.5


def _cell_offsets(jsc, ref_sc, ncell=4, min_per_cell=300):
    """Measure the JWST<->VIRAC offset in an ``ncell`` x ``ncell`` spatial grid over the JWST
    footprint, each cell by the xcorr HISTOGRAM PEAK against the local reference (cropped to the
    cell + a 2" margin).  Replaces the field-wide nearest-neighbour median, which at GC density
    reads SMALLER the further the frame is displaced (~1.8 mas at a 2" shift; PR #54 review).

    Returns (cells, dropped): ``cells`` is a list of dicts (i, j grid index; ra, dec centre; dra,
    dde, off in mas; peak_ratio; n sources) for cells with a peak above ``_CELL_PR_FLOOR``, and
    ``dropped`` is (i, j, ra, dec, n) for cells with enough sources but NO clear peak -- recorded
    so coverage is accounted and the map can show them, rather than silently vanishing."""
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
            if n < min_per_cell:
                continue
            cra, cdec = 0.5 * (re_[i] + re_[i + 1]), 0.5 * (de_[j] + de_[j + 1])
            rm = ((rra >= re_[i] - mrg) & (rra <= re_[i + 1] + mrg) &
                  (rde >= de_[j] - mrg) & (rde <= de_[j + 1] + mrg))
            xc = aa.xcorr(jsc[m], ref_sc[rm]) if int(rm.sum()) >= min_per_cell else None
            if xc and xc.get("peak_ratio", 0) >= _CELL_PR_FLOOR and xc.get("npairs", 0) >= min_per_cell:
                cells.append(dict(i=i, j=j, ra=cra, dec=cdec, dra=float(xc["dra"]),
                                  dde=float(xc["ddec"]), off=float(xc["off"]),
                                  peak_ratio=float(xc["peak_ratio"]), n=n))
            else:
                dropped.append(dict(i=i, j=j, ra=cra, dec=cdec, n=n))
    return cells, dropped


def _cell_consistency(cells, dropped):
    """Aggregate per-cell ties into a SOURCE-WEIGHTED field tie plus a spatial-consistency verdict.

    * offset = source-count-weighted median of the per-cell ties -> the offset the CATALOG actually
      experiences, not the offset of whichever cells had the sharpest (density-biased) peaks.
    * a cell is DEVIATING if its tie is >``_CELL_SPREAD_MAX`` from that median, and CONFIRMED only
      if an orthogonally-adjacent cell also deviates -- a real sub-region discontinuity spans
      several cells, whereas a lone mis-peaked cell (e.g. one 544 mas cell amid 9 mas neighbours)
      does not and must not fail the frame.
    * consistent = the confirmed-deviating cells hold < ``_CELL_BAD_FRAC`` of the measured sources
      AND enough of the field was measurable (coverage >= ``_CELL_MIN_COVERAGE``).

    Returns a dict of the numbers plus per-cell ``deviating``/``confirmed`` flags for plotting."""
    if not cells:
        return dict(n_cells=0, consistent=False)
    dra = np.array([c["dra"] for c in cells]); dde = np.array([c["dde"] for c in cells])
    ns = np.array([c["n"] for c in cells], float)

    def _wmed(v, w):
        o = np.argsort(v); vs, ws = v[o], w[o]; cw = np.cumsum(ws)
        return float(vs[np.searchsorted(cw, 0.5 * cw[-1])])
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
    # uncertainty on the tie (kept as mad_std of the cell ties -- fine as a spread, NOT used for the
    # consistency verdict, which needs the minority-sensitive test above)
    spread = float(np.hypot(aa.mad_std(dra), aa.mad_std(dde))) if len(cells) >= 2 else None
    se = (spread / np.sqrt(len(cells))) if (spread is not None and spread > 0) else None
    signif = float(off_med / se) if se else None
    consistent = bool(len(cells) >= 4 and bad_frac < _CELL_BAD_FRAC and coverage >= _CELL_MIN_COVERAGE)
    return dict(off_med=off_med, off_dra=mdra, off_dde=mdde, spread=spread, se=se, signif=signif,
                n_cells=len(cells), n_dropped=len(dropped), n_deviating=int(deviating.sum()),
                n_confirmed=int(confirmed.sum()), bad_src_frac=bad_frac, coverage=coverage,
                consistent=consistent, deviating=deviating, confirmed=confirmed)


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


def stage4_offsets(o: Observation, sw):
    """JWST-VIRAC frame tie measured PER SPATIAL CELL (xcorr histogram peak), reported as the
    median cell offset with its cell-to-cell spread (the offset's uncertainty), plus the
    reference-free inter-module (NRCA vs NRCB) offset.  A PASS requires both a small median offset
    AND cells that agree -- a single scalar cannot represent a frame with an internal
    discontinuity, and a nearest-neighbour median collapses toward zero at GC density (PR #54)."""
    import astropy.units as u
    metrics = dict(stage=4, sw=sw, offset_signif_med=None, offset_med_mas=None)
    path = _mosaic_path(o, sw)
    ref = _refcat_path(o)
    ep = _obs_epoch(o, path)
    ref_sc, _ = aa.load_reference(ref, ep) if (ref and ep) else (None, None)
    # JWST positions from the catalog (release -> MAST -> per-filter DAO), NOT re-detected on the
    # mosaic.  Stage 4 needs only positions, so a detected-but-not-yet-merged obs (gc2211 o046)
    # falls back to its per-filter DAO catalog and is still measurable.
    jsc, src = _jwst_positions(o, sw)
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

    # Measure the offset PER SPATIAL CELL (robust at density) before building the figure, so an
    # unmeasurable result becomes a red flag rather than an empty panel.
    cells, dropped = (_cell_offsets(jsc, ref_sc) if (jsc is not None and ref_sc is not None)
                      else ([], []))

    if not cells:
        bulk = aa.xcorr(jsc, ref_sc) if (jsc is not None and ref_sc is not None) else None
        reason = _offset_failure_reason(o, sw, jsc, ref_sc, bulk)
        png = _red_flag_figure(o, "stage4", "JWST↔VIRAC OFFSET UNMEASURABLE",
                               f"The positional-offset plot is empty: {reason}.")
        metrics.update(red_flag=True, red_flag_reason=reason, n_cells=0, passed=False)
        return png, metrics

    cc = _cell_consistency(cells, dropped)
    off_med, spread, se, signif = cc["off_med"], cc["spread"], cc["se"], cc["signif"]
    io = metrics.get("intermodule_off")
    # PASS needs a small SOURCE-WEIGHTED median tie, spatially CONSISTENT cells (no
    # adjacency-confirmed sub-region off by >30 mas holding >2% of sources; catches a minority a
    # mad_std cannot), enough coverage, and no inter-module offset.
    passed = bool(off_med < aa.THRESH["absolute"] and cc["consistent"] and
                  (io is None or io < aa.THRESH["intermodule"]))
    metrics.update(offset_med_mas=off_med, offset_scatter_mas=spread, offset_signif_med=signif,
                   bulk_off=off_med,                        # primary (source-weighted) offset
                   n_cells=cc["n_cells"], n_cells_dropped=cc["n_dropped"],
                   n_cells_deviating=cc["n_deviating"], n_cells_confirmed=cc["n_confirmed"],
                   bad_src_frac=cc["bad_src_frac"], cell_coverage=cc["coverage"],
                   cells_consistent=cc["consistent"], passed=passed)

    from matplotlib.patches import Circle
    cdra = np.array([c["dra"] for c in cells]); cdde = np.array([c["dde"] for c in cells])
    cra = np.array([c["ra"] for c in cells]); cdec = np.array([c["dec"] for c in cells])
    coff = np.array([c["off"] for c in cells])
    confirmed = cc["confirmed"]; deviating = cc["deviating"]
    ncols = 2 + (1 if im else 0)
    fig, ax = _fig(1, ncols, 6.2, 5.4)
    # extra column gap + top room so the middle panel's marginal histograms (which hang above and
    # to the right of the axes) don't overlap the neighbouring panel or the suptitle.
    fig.subplots_adjust(wspace=0.62, top=0.80, bottom=0.12)
    col = 0
    # panel 1: contiguous 4x4 map of the per-cell tie (colour = offset).  Confirmed-deviating cells
    # get a RED outline (deviating = bad); DROPPED cells (sources present, no clear peak) render
    # grey -- so a discontinuity or missing coverage is visible, not inferable.
    a0 = ax[0][col]; col += 1
    # cap the colour scale near the field tie (a few wild cells would otherwise wash out the
    # 30-vs-130 structure); over-scale cells saturate but are already flagged by the green outline.
    vmax = max(2.0 * aa.THRESH["absolute"], 2.0 * off_med)
    # CONTIGUOUS 4x4 grid (imshow), so the cells tile with no whitespace and a coherent patch is
    # obvious.  Build an [i(RA), j(Dec)] grid of offsets; dropped/absent cells are NaN -> grey.
    from matplotlib.patches import Rectangle
    ncell = 1 + max([c["i"] for c in cells] + [c["j"] for c in cells]
                    + [d["i"] for d in dropped] + [d["j"] for d in dropped] + [0])
    grid = np.full((ncell, ncell), np.nan)
    for c in cells:
        grid[c["i"], c["j"]] = c["off"]
    # RA/Dec edges from the cell centres (uniform grid) so the axes read in sky coords
    raC = sorted({round(c["ra"], 8) for c in cells} | {round(d["ra"], 8) for d in dropped})
    deC = sorted({round(c["dec"], 8) for c in cells} | {round(d["dec"], 8) for d in dropped})
    dra_c = (raC[1] - raC[0]) if len(raC) > 1 else 1e-3
    dde_c = (deC[1] - deC[0]) if len(deC) > 1 else 1e-3
    ext = [raC[0] - dra_c / 2, raC[-1] + dra_c / 2, deC[0] - dde_c / 2, deC[-1] + dde_c / 2]
    import matplotlib as mpl
    cmap = mpl.colormaps["inferno"].copy(); cmap.set_bad("0.7")     # dropped/absent cells -> grey
    im0 = a0.imshow(grid.T, origin="lower", extent=ext, aspect="auto", cmap=cmap,
                    vmin=0, vmax=vmax)
    fig.colorbar(im0, ax=a0, label="cell tie [mas]", shrink=0.85)
    # green outline on confirmed-deviating cells (drawn as rectangles on the same grid)
    for k, c in enumerate(cells):
        if confirmed[k]:
            a0.add_patch(Rectangle((c["ra"] - dra_c / 2, c["dec"] - dde_c / 2), dra_c, dde_c,
                                   fill=False, ec="#e41a1c", lw=2.0, zorder=3))   # red = deviating
    a0.set_xlabel("RA [deg]"); a0.set_ylabel("Dec [deg]"); a0.invert_xaxis()
    a0.set_title(f"per-cell tie ({_dataset_label(metrics)}): {cc['n_cells']} measured, "
                 f"{cc['n_dropped']} no-peak\nmedian {off_med:.0f} mas; {cc['n_confirmed']} cells "
                 f"({100 * cc['bad_src_frac']:.0f}% of sources) deviate", fontsize=8)
    # panel 2: per-cell offsets in (dRA, dDec) sized by source count (big = more sources = more
    # weight), the source-weighted median tie, and the 75 mas gate.
    a1 = ax[0][col]; col += 1
    # sized by source count; semi-transparent so overlapping cells at similar (ΔRA,ΔDec) are both
    # visible instead of one hiding the other.
    sz = 30 + 130 * np.array([c["n"] for c in cells]) / max(c["n"] for c in cells)
    a1.scatter(cdra[~deviating], cdde[~deviating], s=sz[~deviating], c="#4477aa",
               edgecolor="k", linewidth=0.3, alpha=0.6, label="consistent")
    if deviating.any():
        a1.scatter(cdra[deviating], cdde[deviating], s=sz[deviating], c="#e41a1c",
                   edgecolor="k", linewidth=0.3, alpha=0.6, label="deviating")
    a1.plot(cc["off_dra"], cc["off_dde"], "k+", ms=15, mew=2)
    a1.add_patch(Circle((0, 0), aa.THRESH["absolute"], fill=False, ec="r", ls=":", lw=0.9))
    a1.axhline(0, color="k", lw=0.4); a1.axvline(0, color="k", lw=0.4); a1.set_aspect("equal")
    lim = max(aa.THRESH["absolute"] * 1.2, 1.4 * float(np.max(np.hypot(cdra, cdde))))
    a1.set_xlim(-lim, lim); a1.set_ylim(-lim, lim)
    a1.set_xlabel("ΔRA [mas]"); a1.set_ylabel("ΔDec [mas]")
    a1.legend(fontsize=7, loc="upper right")
    # marginal ΔRA / ΔDec histograms of the per-cell ties, weighted by source count (so the
    # marginals reflect the source-weighted tie, not raw cell counts).  Title goes on the top
    # marginal so it clears the inset histograms.
    a1t, _a1r = _add_marginals(a1, cdra, cdde, color="#4477aa", bins=12,
                               weights=np.array([c["n"] for c in cells], float))
    se_str = f"±{se:.0f}" if se is not None else ""
    sig_str = f", {signif:.0f}σ" if signif is not None else ""
    a1t.set_title(f"source-weighted tie {off_med:.0f}{se_str} mas{sig_str}\n"
                  f"(point size ∝ sources; dotted = 75 mas gate; marginals source-weighted)",
                  fontsize=8)
    if im:
        # inter-module offset is just two numbers -- print them, don't histogram/bar them.
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
    fig.suptitle(f"{o.target} {o.obsid} — positional offsets (JWST ↔ VIRAC frame tie)",
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


def _ab_overlap(a_sc, b_sc):
    """Reference-free NRCA↔NRCB tie from two module position lists (no external catalogue).
    Crowding-robust: the bulk A→B shift is the ``aa.xcorr`` histogram PEAK (a direct
    search_around_sky median fabricates pairs in a dense field); the RMS (tie precision) is the
    residual scatter of the SAME stars after aligning A onto B by that peak.  Returns a dict with
    the offset/RMS/count, the per-star residual arrays (for the hexbin + marginals), and a list of
    overlap-star positions (for the cutout gallery), or None if unmeasurable."""
    import astropy.units as u
    from astropy.coordinates import search_around_sky, SkyCoord
    if a_sc is None or b_sc is None or len(a_sc) < 50 or len(b_sc) < 50:
        return None
    xc = aa.xcorr(a_sc, b_sc, maxsep=1.5 * u.arcsec)
    if not (xc and xc["peak_ratio"] >= aa.MIN_PEAK_RATIO and xc["npairs"] >= 100):
        return None
    cosd = float(np.cos(np.radians(np.median(a_sc.dec.deg))))
    a_al = SkyCoord((a_sc.ra.deg + xc["dra"] / 1000.0 / 3600.0 / cosd) * u.deg,
                    (a_sc.dec.deg + xc["ddec"] / 1000.0 / 3600.0) * u.deg)
    ia, ib, sep, _ = search_around_sky(a_al, b_sc, 0.08 * u.arcsec)   # same star after align
    if len(ia) < 20:
        return None
    dra = (a_al[ia].ra - b_sc[ib].ra).to(u.mas).value * cosd
    dde = (a_al[ia].dec - b_sc[ib].dec).to(u.mas).value
    return dict(dra=float(xc["dra"]), dde=float(xc["ddec"]), off=float(xc["off"]),
                rms=float(np.hypot(aa.mad_std(dra), aa.mad_std(dde))),
                n=int(len(ia)), peak_ratio=float(xc["peak_ratio"]),
                dra_arr=dra, dde_arr=dde,
                pos=[(b_sc[i].ra.deg, b_sc[i].dec.deg) for i in ib[:200]])


def _draw_ab_panel(ax, ovd, title):
    """Draw one reference-free A↔B residual panel: a 2-D hexbin of the per-star ΔRA/ΔDec
    residuals (same stars, aligned by the histogram peak) with ΔRA/ΔDec marginal histograms.
    The numeric summary goes on the TOP marginal's title so it can't collide with the marginals."""
    dra_a = np.asarray(ovd["dra_arr"], float); dde_a = np.asarray(ovd["dde_arr"], float)
    ax.hexbin(dra_a, dde_a, gridsize=40, bins="log", cmap="cividis", mincnt=1)
    ax.axhline(0, color="w", lw=0.5); ax.axvline(0, color="w", lw=0.5)
    ax.set_xlabel("NRCA−NRCB residual ΔRA [mas]"); ax.set_ylabel("residual ΔDec [mas]")
    lim = max(50.0, 4.0 * ovd["rms"])
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    axt, _axr = _add_marginals(ax, dra_a, dde_a, color="#5566aa", bins=40)
    axt.set_title(f"{title}  ({ovd['n']} stars)\n"
                  f"offset = {ovd['off']:.1f} mas   RMS = {ovd['rms']:.1f} mas", fontsize=8)


def _module_hi_sn(o, filt, snmin=10.0):
    """(NRCA, NRCB) finite SkyCoords restricted to flux S/N > ``snmin`` (S/N = flux_fit/flux_err
    from the per-exposure PSF fit), pooled from the per-detector daophot cats.  Used for the
    stage-5 high-S/N overlap panel so the A↔B residual reflects the tie, not faint-source
    centroiding.  Either module is None if its cats lack flux errors or have too few high-S/N
    stars."""
    from astropy.table import vstack, Table

    def pool(dets):
        cats = []
        for d in dets:
            cats += _daophot_glob(o, filt, d)          # obs-scoped
        tabs = []
        for c in cats:
            try:
                t = Table.read(c)
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
    """Inter-detector / inter-module tie quality:
    (1) per-detector residual quiver vs VIRAC, bulk-subtracted (relative ties);
    (2) reference-free NRCA-vs-NRCB overlap: median offset + RMS of the SAME stars;
    (3) doubled-star cutout gallery on the module-overlap zone (mis-tie -> split PSF)."""
    import astropy.units as u
    from astropy.coordinates import SkyCoord
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
    ov = _ab_overlap(a_sc, b_sc)
    if ov:
        metrics.update(intermodule_off=ov["off"], intermodule_rms=ov["rms"], n_overlap=ov["n"])
    # Same tie restricted to flux S/N > 10 (best-measured stars): the residual scatter then
    # reflects the A↔B tie rather than faint-source centroiding.  An ADDITIONAL panel, not a
    # replacement -- the all-stars panel above stays.
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
            axq.quiverkey(q, 0.5, 0.10, 5, "5 mas", labelpos="E", coordinates="axes",
                          fontproperties={"size": 8})
            for d, v in det.items():
                # annotate each arrow with the number of VIRAC-matched stars behind it
                axq.annotate(f"{d} (n={v['n']})", (v["ra"], v["dec"]), fontsize=5.8,
                             ha="center", va="bottom")
            axq.invert_xaxis(); axq.set_ylabel("Dec"); axq.set_xlabel("RA")
            axq.set_title(f"per-detector residual vs VIRAC (bulk-removed) — {filt}\n"
                          f"A−B diff = {metrics.get('intermodule_diff', float('nan')):.1f} mas  ·  "
                          f"one arrow/detector (n = VIRAC matches)", fontsize=9, pad=12)
        else:
            axq.text(0.5, 0.5, "per-detector cats unavailable", ha="center", va="center", fontsize=8)

    if ov:
        # Top row: per-detector quiver | A↔B overlap (all stars, with marginals) | A↔B (S/N>10).
        # Bottom row: overlap-star cutout gallery spanning the width.  The S/N>10 column is added
        # only when measurable, so a field without flux errors just shows the two-panel top row.
        ncols = 3 if ov_hi else 2
        fig = plt.figure(figsize=(5.0 * ncols + 1.0, 9.2))
        gs = fig.add_gridspec(2, ncols, height_ratios=[1.3, 0.75], hspace=0.62, wspace=0.62)
        # top reserve so the A↔B panels' top-marginal titles clear the suptitle
        fig.subplots_adjust(top=0.82, bottom=0.06, left=0.06, right=0.97)
        axq = fig.add_subplot(gs[0, 0]); _draw_quiver(axq)
        axo = fig.add_subplot(gs[0, 1]); _draw_ab_panel(axo, ov, "A↔B overlap — all stars")
        if ov_hi:
            axh = fig.add_subplot(gs[0, 2])
            _draw_ab_panel(axh, ov_hi, "A↔B overlap — flux S/N > 10")

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
            strip = fig.add_subplot(gs[1, :]); strip.axis("off")
            if shown:
                n = shown
                for i, cdata in enumerate(cuts):
                    a = strip.inset_axes([i / n + 0.01, 0.05, 0.92 / n, 0.85])
                    norm = ImageNormalize(cdata, interval=ZScaleInterval(), stretch=AsinhStretch())
                    a.imshow(cdata, origin="lower", cmap="gray", norm=norm)
                    a.set_xticks([]); a.set_yticks([]); a.set_title(f"{i + 1}", fontsize=7)
            else:
                strip.text(0.5, 0.5, "no usable overlap-star cutouts on the mosaic",
                           ha="center", va="center", fontsize=9, style="italic")
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
        t = Table.read(best)
    except (OSError, ValueError):
        return None
    ra_std = np.asarray(t[sr], float); de_std = np.asarray(t[sd], float)
    m = np.asarray(t[mv], float)
    cosd = (float(np.cos(np.radians(np.nanmedian(t[sccol].dec.deg))))
            if sccol in t.colnames else 1.0)
    rms = np.hypot(ra_std * cosd, de_std) * 3.6e6           # deg -> mas
    # a position scatter is only meaningful with several exposures; 1-2 detections give a
    # degenerate std (0 or near-0).  Require >=3 detections AND drop unphysically-tiny values
    # (< 0.1 mas, well below the real ~1 mas internal floor) so a degenerate tail can't drag the
    # binned median toward zero at the faint end.
    g = np.isfinite(rms) & np.isfinite(m) & (rms > 0.1)
    nmcol = f"nmatch_{filt.lower()}"
    if nmcol in t.colnames:
        g &= np.asarray(t[nmcol], float) >= 3
    return (m[g], rms[g]) if int(g.sum()) >= 50 else None


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
        a.plot(ctr, med, "-", color=color, lw=1.7, label=lbl + r"  $\sigma_{\rm pos}$")
        a.fill_between(ctr, lo, hi, color=color, alpha=0.20)
        metrics[f"floor_mas_{filt.lower()}"] = float(np.nanmin(med))
        metrics[f"nstars_{filt.lower()}"] = int(ok.sum())
        # rms(offset): the EXTERNAL scatter vs VIRAC (includes the VIRAC error floor), dashed, same
        # colour -- shown alongside sigma_pos so "how precisely measured" vs "how well it agrees
        # with the external frame" are both visible.
        import astropy.units as u
        ref = _viraccache_path(o) or _refcat_path(o)
        ep = _obs_epoch(o, _mosaic_path(o, filt))
        ref_sc, _ = aa.load_reference(ref, ep) if (ref and ep) else (None, None)
        if ref_sc is not None:
            idx, sep, _ = ref_sc.match_to_catalog_sky(_sc)      # anchor on sparse VIRAC
            keep = (sep < 0.15 * u.arcsec).nonzero()[0]
            if keep.size >= 50:
                cosd = float(np.cos(np.radians(np.median(_sc[idx[keep]].dec.deg))))
                dra = (_sc[idx[keep]].ra - ref_sc[keep].ra).to(u.mas).value * cosd
                dde = (_sc[idx[keep]].dec - ref_sc[keep].dec).to(u.mas).value
                resid = np.hypot(dra - np.median(dra), dde - np.median(dde))   # bulk-removed
                rms, rctr = _binned_rms(mag[idx[keep]], resid)
                if rms is not None:
                    a.plot(rctr, rms, "--", color=color, lw=1.5, alpha=0.9,
                           label=f"{filt}  rms(offset−VIRAC)")
                    metrics[f"rms_offset_floor_mas_{filt.lower()}"] = float(np.nanmin(rms))
        # rms(jwst): the INTERNAL per-star position scatter across exposures (merged-catalog
        # std_ra/std_dec, deg -> mas), median vs mag -- the empirical JWST repeatability, distinct
        # from the formal sigma_pos and from the external rms(offset-VIRAC).
        jr = _internal_pos_rms(o, filt)
        if jr is not None:
            jmag_v, jrms = jr
            med_j, _, _, ctr_j = _binned_stat(jmag_v, jrms)
            if med_j is not None:
                a.plot(ctr_j, med_j, ":", color=color, lw=1.8, alpha=0.9,
                       label=f"{filt}  rms(jwst) internal")
                metrics[f"rms_jwst_floor_mas_{filt.lower()}"] = float(np.nanmin(med_j))
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
    a.set_title(f"{o.target} {o.obsid} — astrometric precision vs magnitude", fontsize=10)
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
    4: "**Stage 4 — positional offsets (JWST ↔ VIRAC frame tie).**",
    5: "**Stage 5 — inter-detector / inter-module tie.**",
    6: "**Stage 6 — astrometric precision.**",
    7: "**Stage 7 — MAST vs pipeline.**",
}

# Templates reached via the generic `CAPTIONS[n].format(...)` fallback in _caption_for_impl.  Only
# stages whose caption is NOT built in code live here (1, 3, 6).  Stages 2/4/5/7 build their caption
# in code (variant-dependent), so no template exists for them -- avoids a dead duplicate that drifts.
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
    6: "**Stage 6 — astrometric precision.** Three per-star error curves vs Vega magnitude per "
       "channel: σ_pos (solid — the per-exposure PSF-fit position error `dra`/`ddec`, the predicted "
       "precision), rms(jwst) (dotted — the empirical position scatter across exposures, the "
       "internal repeatability), and rms(offset) (dashed — the RMS of the per-star "
       "JWST−[VIRAC](DOCROOT#glossary-virac) offset, the external scatter incl. the VIRAC floor). "
       "The σ_pos bright-end floor is the astrometric systematic limit; the faint-end rise tracks "
       "S/N. Shaded band = 16–84th percentile. ([how this is made](DOCROOT#stage6))",
}


def caption_for(n, metrics):
    """Public entry: build the stage caption and swap the DOCROOT sentinel for the live doc URL."""
    return _linkify(_caption_for_impl(n, metrics))


def _caption_for_impl(n, metrics):
    if metrics.get("red_flag"):
        return (f"🚩 **Stage {n} — RED FLAG.** The plot is empty: "
                f"{metrics.get('red_flag_reason', 'no data to show')}. "
                f"An empty result here means the measurement could not be made — investigate. "
                f"([how this stage works](DOCROOT#stage{n}))")
    if n == 1 and metrics.get("dropped_filters"):
        # a nominal (proposed) filter with no product on disk must leave a trace, not vanish
        df = ", ".join(metrics["dropped_filters"])
        return (CAPTIONS[1].format(**{k: (v if v is not None else float("nan"))
                                      for k, v in metrics.items() if k in ("sw", "lw")})
                + f" NOTE: nominal filter(s) with no mosaic/catalogue on disk (observed but not "
                  f"reduced, or not delivered): {df}.")
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
        if kind == "crossmatch":
            body += ("The colour width is set by the positional cross-match tolerance, not the "
                     "catalogue's colour precision. ")
        return body + "([how this is made](DOCROOT#stage2))"
    if n == 4:
        # Built in code (not a template) so it renders cleanly whatever is/ isn't measured -- never
        # "nanσ", and gated on the CELL COUNT, not on significance being None (a zero-spread field
        # can be consistent and pass).
        om = metrics.get("offset_med_mas"); nc = metrics.get("n_cells") or 0
        if om is None or nc == 0:
            return ("**Stage 4 — positional offsets.** JWST↔VIRAC frame tie unmeasured "
                    "(no usable spatial cells). ([how this is made](DOCROOT#stage4))")
        sig = metrics.get("offset_signif_med"); sp = metrics.get("offset_scatter_mas")
        nd = metrics.get("n_cells_dropped") or 0; ncf = metrics.get("n_cells_confirmed") or 0
        badf = metrics.get("bad_src_frac")
        unc = (f", cell-to-cell spread {sp:.0f} mas"
               + (f" → {sig:.0f}σ from zero" if sig is not None else "")) if sp is not None else ""
        base = (f"**Stage 4 — positional offsets (JWST ↔ VIRAC frame tie).** The "
                f"[**bulk** offset](DOCROOT#glossary-bulk) is the JWST catalogue "
                f"[cross-matched to VIRAC](DOCROOT#glossary-crossmatch) and registered onto the "
                f"Gaia/VIRAC frame, measured per spatial cell by the "
                f"[xcorr histogram peak](DOCROOT#glossary-xcorr) (crowding-robust; a plain "
                f"nearest-neighbour median collapses toward zero at GC density). The LEFT panel "
                f"maps the source-weighted tie across the mosaic — filled squares are measured "
                f"cells (colour = offset), grey squares are cells with sources but no clear peak, "
                f"and a green outline marks cells that coherently deviate. The MIDDLE panel plots "
                f"the per-cell offsets as (ΔRA, ΔDec) points sized by source count, with the "
                f"source-weighted median tie, the 75 mas gate, and ΔRA/ΔDec marginal histograms. "
                f"Here the tie is {om:.0f} mas over {nc} measured cells ({nd} without a peak){unc}; "
                f"the [uncertainty](DOCROOT#glossary-tie-uncertainty) is the cell-to-cell standard "
                f"error (not a per-star RMS or per-star error). ")
        if ncf:
            base += (f"{ncf} adjacent cell(s) holding {100 * (badf or 0):.0f}% of the sources are "
                     f"tied differently — an internal discontinuity, so it does NOT pass. ")
        base += ("The RIGHT panel, when present, is the "
                 "[reference-free](DOCROOT#glossary-reffree) NRCA-vs-NRCB inter-module offset. A "
                 "pass needs a small source-weighted median AND spatially consistent cells. "
                 "([how this is made](DOCROOT#stage4))")
        if str(metrics.get("source", "")).startswith("release-dao"):
            base += (" (Positions here come from a per-filter DAO catalogue, not a merged/calibrated "
                     "one — this obs is not yet photometrically catalogued, so stage 3 red-flags it.)")
        return base
    if n == 5:
        # Built entirely in code (not the CAPTIONS[5] template) so: (a) the S/N>10 clause is gated
        # on the panel actually being present (ov_hi), (b) the panel POSITION wording is correct
        # ("to its right", it is gs[0,2]), and (c) a missing intermodule_diff can never KeyError.
        diff = metrics.get("intermodule_diff")
        diff_clause = (f" The [per-detector quiver](DOCROOT#glossary-quiver) shows an A–B diff of "
                       f"{diff:.1f} mas." if diff is not None else "")
        if metrics.get("intermodule_off") is None:
            # a legitimate single-module obs, or two modules with no shared stars to tie them
            if metrics.get("single_module"):
                return (f"**Stage 5 — inter-detector tie.** Single module "
                        f"({metrics['single_module']}) for this observation, so there is no "
                        f"NRCA–NRCB tie to check — the [reference-free](DOCROOT#glossary-reffree) "
                        f"overlap panel is omitted.{diff_clause} "
                        f"([how this is made](DOCROOT#stage5))")
            return ("**Stage 5 — inter-detector / inter-module tie.** The "
                    "[reference-free](DOCROOT#glossary-reffree) NRCA–NRCB overlap could not be "
                    "measured (no shared stars in the NRCA∩NRCB dither overlap after alignment), so "
                    "that panel and the cutout gallery are omitted."
                    f"{diff_clause} The inter-module tie is unverified for this observation. "
                    "([how this is made](DOCROOT#stage5))")
        # overlap measured -> full caption; the S/N>10 panel is only present when ov_hi succeeded
        off = metrics.get("intermodule_off"); rms = metrics.get("intermodule_rms")
        no = metrics.get("n_overlap")
        # the all-stars overlap panel is TOP-MIDDLE when the S/N>10 panel is also drawn (3 cols),
        # otherwise TOP-RIGHT (2 cols)
        ov_pos = "TOP-MIDDLE" if metrics.get("n_overlap_hi") else "TOP-RIGHT"
        base = ("**Stage 5 — inter-detector / inter-module tie.** "
                "[\"Reference-free\"](DOCROOT#glossary-reffree) means JWST is matched against itself "
                "(NRCA vs NRCB), using no external catalogue. The TOP-LEFT "
                "[per-detector quiver](DOCROOT#glossary-quiver) shows each detector's median "
                "residual **against VIRAC** (field bulk offset removed), each arrow annotated with "
                "its matched-star count — every detector gets a vector because the shared reference "
                "is VIRAC, not NRCA, so e.g. NRCB2 (which never overlaps NRCA on the sky) is still "
                f"measured; the NRCA−NRCB difference is {(diff if diff is not None else float('nan')):.1f} "
                f"mas. The {ov_pos} panel is the reference-free NRCA∩NRCB overlap tie — {off:.1f} "
                f"mas offset, {rms:.1f} mas RMS over {no} shared stars — with ΔRA/ΔDec marginal "
                f"histograms.")
        if metrics.get("n_overlap_hi"):
            base += (f" The panel to its right repeats the tie for [S/N > 10](DOCROOT#glossary-snr) "
                     f"stars ({metrics['n_overlap_hi']} stars, "
                     f"{metrics.get('intermodule_rms_hi', float('nan')):.1f} mas RMS), where the "
                     f"scatter reflects the tie rather than faint-source centroiding.")
        base += (" The BOTTOM strip shows overlap-star cutouts from the SW merged `i2d`: a good tie "
                 "shows one round PSF, a mis-tie doubles or elongates the star. "
                 "([how this is made](DOCROOT#stage5))")
        return base
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
    # Pick the CMD/QA filters from those that actually HAVE data on disk, not the program's nominal
    # filter list.  The portal lists all six Sgr A* filters, but only F212N+F405N have mosaics and
    # catalogs; picking blindly gave F212N+F444W -> a false "no catalog" red flag (issue #39).
    avail = _available_filters(o) or o.filters
    sw, lw = pick_filters(avail, args.sw, args.lw)
    print(f"{o.obsid}: SW={sw} LW={lw} filters={o.filters} (with data: {avail})")
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
