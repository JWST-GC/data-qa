"""Two-band -> three-color RGB treasury images with embedded AVM.

Composes the CMZ house two-color scheme (``jwst_gc_pipeline.cmz.hips``):
``B = asinh(F212N)``, ``R = asinh(long band)`` (F480M preferred, F405N legacy),
``G = 0.5*(R + B)``, with a GLOBAL asinh stretch per band (per-tile/per-region
stretches would seam).  The long band is reprojected onto the F212N pixel grid.

Outputs, per ``--out`` basename:

* ``<out>.png``  -- RGBA; alpha=0 only where BOTH bands are NaN; AVM (XMP) sidecar
  metadata embedded so HiPS builders / Aladin / WWT can place it on the sky.
* ``<out>.jpg``  -- progressive JPEG (no alpha) for web preview.
* ``<out>.validation.json`` -- machine-readable validation verdict (see below).

AVM convention: the PNG pixel rows are written top-down (``flipud`` of the FITS
array, i.e. normal display orientation) and the embedded AVM carries the
UNFLIPPED FITS-convention WCS -- exactly what ``reproject``'s PNG+AVM reader
expects (it flips the PNG back to a bottom-up array before applying the WCS).
The WCS is stored as a flat ``Spatial.CDMatrix`` (ported from
``jwst_rgb.save_rgb.faithful_avm``): pyavm's Scale+Rotation representation is
DEGENERATE near position angle 90 deg -- exactly where JWST GC fields sit -- and
reconstructs a mirrored rotation there; the CD matrix is honored verbatim by
``pyavm.AVM.to_wcs`` and is correct at every roll angle.

Validation (``--validate``, and ALWAYS run automatically after a write):
re-reads the AVM from the PNG, reconstructs the WCS, and compares it against the
F212N FITS WCS at the reference pixel + 4 corners (PASS: max offset < 0.1");
checks alpha/NaN consistency and the nonzero finite fraction; writes
``<out>.validation.json``.  Exits nonzero on FAIL.  The verdict records
``outputs.{png,jpg}_sha256`` -- the hashes of the exact files it validated --
which ``publish.py``'s avm gate re-checks at push time (stale-verdict
protection).

Star-position check (the SECOND avm-publish gate, user decision 4): whenever a
reference catalog is available -- ``--ref-catalog``, or found by convention
from ``--field`` (``/orange/adamginsburg/jwst/<field>/catalogs/``,
``gaia_virac2_refcat*.fits`` preferred, ``gaia_refcat*.fits`` fallback) -- up
to ~200 bright catalog stars are projected through the PNG's embedded AVM WCS
and matched to local luminance peaks (+-4 px box).  PASS: median offset <= 2
px, matched fraction >= 0.5, >= 20 usable stars.  The result is recorded under
``checks.star_positions`` in the validation JSON; with no catalog it records
``{skipped: true, reason}`` and ``publish.py`` requires an explicit
``--no-star-check`` acknowledgment to push.  ``--validate-stars`` makes a
missing catalog an error instead of a skip.

Usage::

    python -m data_qa.rgb_treasury --f212n F212N_i2d.fits --long F480M_i2d.fits \
        --long-band F480M --out /path/to/sgrb2_rgb
    python -m data_qa.rgb_treasury --fields-spec fields.json [--dry-run]
    python -m data_qa.rgb_treasury --f212n ... --long ... --out BASE --validate

``--fields-spec`` JSON: ``{"fields": [{"name": ..., "f212n_i2d": ...,
"long_i2d": ..., "long_band": "F480M", "out": ...}, ...]}`` (``out`` optional if
``out_dir`` given at top level; then ``out = <out_dir>/<name>_rgb``).
"""
from __future__ import annotations

import argparse
import datetime
import glob
import hashlib
import json
import os
import sys

import numpy as np

DEFAULT_PIPE_ROOT = "/blue/adamginsburg/adamginsburg/repos/jwst-gc-pipeline"
LONG_BANDS = ("F480M", "F405N")
WCS_PASS_ARCSEC = 0.1
PERCENTILES = (1.0, 99.5)   # same limits the cmz.hips global stretch uses

# Star-position check (the SECOND avm gate, user decision 4 2026-07-22):
# reference-catalog stars must land on luminance peaks in the written PNG.
REFCAT_ROOT = "/orange/adamginsburg/jwst"
STAR_MAX_STARS = 200        # brightest finite-mag stars inside the footprint
STAR_BOX_PX = 4             # +-box around the predicted pixel to search
STAR_PASS_MEDIAN_PX = 2.0   # PASS: median |predicted - peak| <= this
STAR_MIN_MATCH_FRACTION = 0.5
STAR_MIN_USED = 20          # refuse to conclude from fewer usable stars
STAR_PLATEAU_MIN_PX = 3     # >= this many px at the box max = a plateau
STAR_SATURATED_LUM = 255.0  # uint8 luminance ceiling: star top is clipped


# --------------------------------------------------------------------------- helpers
def _import_hips(pipe_root=None):
    """Import ``jwst_gc_pipeline.cmz.hips`` (source of the house stretch/compose),
    with a ``--pipe-root`` sys.path fallback.  Returns the module or None."""
    try:
        from jwst_gc_pipeline.cmz import hips
        return hips
    except ImportError:
        pass
    root = pipe_root or DEFAULT_PIPE_ROOT
    if root and os.path.isdir(root) and root not in sys.path:
        sys.path.insert(0, root)
        try:
            from jwst_gc_pipeline.cmz import hips
            return hips
        except ImportError:
            return None
    return None


def _asinh_norm_local(arr, vmin, vmax):
    """Asinh stretch to [0,1].  Identical to
    ``jwst_gc_pipeline.cmz.hips._asinh_norm`` (the jwst_rgb house stretch);
    duplicated here only as a fallback when the pipeline is not importable."""
    a = 0.1
    x = (np.asarray(arr, float) - vmin) / max(vmax - vmin, 1e-30)
    x = np.clip(x, 0, 1)
    out = np.arcsinh(x / a) / np.arcsinh(1.0 / a)
    return np.clip(out, 0, 1)


def _two_color_local(blue_arr, red_arr, blue_lims, red_lims):
    """RGBA compose: R=long, B=F212N, G=0.5*(R+B); alpha=0 where BOTH NaN.
    Identical to ``jwst_gc_pipeline.cmz.hips.two_color_tile`` (fallback copy)."""
    b = _asinh_norm_local(blue_arr, *blue_lims)
    r = _asinh_norm_local(red_arr, *red_lims)
    g = 0.5 * (r + b)
    finite = np.isfinite(blue_arr) | np.isfinite(red_arr)
    r = np.nan_to_num(r, nan=0.0)
    g = np.nan_to_num(g, nan=0.0)
    b = np.nan_to_num(b, nan=0.0)
    rgba = np.zeros(r.shape + (4,), dtype=np.uint8)
    rgba[..., 0] = (r * 255).astype(np.uint8)
    rgba[..., 1] = (g * 255).astype(np.uint8)
    rgba[..., 2] = (b * 255).astype(np.uint8)
    rgba[..., 3] = np.where(finite, 255, 0).astype(np.uint8)
    return rgba


def compose_rgba(blue_arr, red_arr, blue_lims, red_lims, pipe_root=None):
    """Two-color RGBA compose, preferring the pipeline's ``two_color_tile``."""
    hips = _import_hips(pipe_root)
    fn = hips.two_color_tile if hips is not None else _two_color_local
    return fn(blue_arr, red_arr, blue_lims, red_lims)


def band_limits(arr, percentiles=PERCENTILES):
    """Global (vmin, vmax) for one band: percentiles over all finite pixels
    (the ``cmz.hips.global_limits`` approach, applied to a full array instead of
    coarse HiPS tiles)."""
    v = np.asarray(arr, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        raise ValueError("no finite pixels; cannot derive stretch limits")
    lo, hi = np.percentile(v, percentiles)
    return float(lo), float(hi)


def load_sci(path):
    """Read (data, celestial WCS, header) from the SCI extension (or first HDU
    with data).  memmap so cutout-sized reads of huge mosaics stay cheap."""
    from astropy.io import fits
    from astropy.wcs import WCS
    with fits.open(path, memmap=True) as hdul:
        names = [h.name for h in hdul]
        hdu = hdul["SCI"] if "SCI" in names else next(
            h for h in hdul if getattr(h, "data", None) is not None)
        data = np.asarray(hdu.data, dtype=float)
        header = hdu.header.copy()
    return data, WCS(header).celestial, header


def reproject_long(long_path, target_wcs, shape_out, exact=False):
    """Reproject the long band onto the F212N grid (interp default, exact for
    flux conservation)."""
    if exact:
        from reproject import reproject_exact as _reproject
    else:
        from reproject import reproject_interp as _reproject
    data, wcs, _ = load_sci(long_path)
    out, _foot = _reproject((data, wcs), target_wcs, shape_out=shape_out)
    return out


def faithful_avm(wcs, shape):
    """Faithful AVM as a flat ``Spatial.CDMatrix`` (port of
    ``jwst_rgb.save_rgb.faithful_avm``): pyavm's Scale+Rotation form is
    degenerate near PA~90 deg (mirrors the rotation); CDMatrix round-trips
    exactly through ``AVM.to_wcs`` at any roll angle."""
    import pyavm
    wcs = wcs.celestial.deepcopy()
    ny, nx = shape
    wcs.pixel_shape = (nx, ny)
    cd = wcs.pixel_scale_matrix
    avm = pyavm.AVM.from_wcs(wcs, shape=(ny, nx))
    avm.Spatial.CDMatrix = [cd[0, 0], cd[0, 1], cd[1, 0], cd[1, 1]]
    avm.Spatial.Scale = None
    avm.Spatial.Rotation = None
    return avm


def _sha256(path, blocksize=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(blocksize), b""):
            h.update(block)
    return h.hexdigest()


# --------------------------------------------------------------------------- build
def write_outputs(rgba, out_base, wcs, jpg_quality=95):
    """Write ``<out>.png`` (RGBA + embedded AVM) and ``<out>.jpg`` (progressive).

    PNG rows are flipud'ed (display orientation, top-down); the AVM keeps the
    unflipped FITS-convention WCS -- the reproject PNG+AVM convention (see
    module docstring)."""
    from PIL import Image
    png = out_base + ".png"
    jpg = out_base + ".jpg"
    os.makedirs(os.path.dirname(os.path.abspath(png)), exist_ok=True)
    disp = np.flipud(rgba)
    Image.fromarray(disp, mode="RGBA").save(png)
    avm = faithful_avm(wcs, rgba.shape[:2])
    tagged = os.path.join(os.path.dirname(os.path.abspath(png)),
                          "avm_" + os.path.basename(png))
    avm.embed(png, tagged)
    os.replace(tagged, png)
    Image.fromarray(disp[..., :3], mode="RGB").save(
        jpg, format="JPEG", quality=jpg_quality, progressive=True)
    return png, jpg


def build_rgb(f212n_path, long_path, out_base, long_band="F480M", exact=False,
              percentiles=PERCENTILES, pipe_root=None):
    """Full build: load, reproject long->F212N grid, global-stretch compose,
    write PNG (+AVM) and JPG.  Returns (png, jpg, red_reprojected, blue, wcs)."""
    blue, wcs, header = load_sci(f212n_path)
    red = reproject_long(long_path, wcs, blue.shape, exact=exact)
    blue_lims = band_limits(blue, percentiles)
    red_lims = band_limits(red, percentiles)
    rgba = compose_rgba(blue, red, blue_lims, red_lims, pipe_root=pipe_root)
    png, jpg = write_outputs(rgba, out_base, wcs)
    return png, jpg, red, blue, wcs


# ---------------------------------------------------------------- star positions gate
def find_ref_catalog(field, root=REFCAT_ROOT):
    """Reference catalog by field convention:
    ``<root>/<field>/catalogs/gaia_virac2_refcat*.fits`` preferred,
    ``gaia_refcat*.fits`` fallback.  None when the field has neither."""
    if not field:
        return None
    for pattern in ("gaia_virac2_refcat*.fits", "gaia_refcat*.fits"):
        hits = sorted(glob.glob(os.path.join(root, field, "catalogs", pattern)))
        if hits:
            return hits[0]
    return None


def _refcat_radec_mag(tbl):
    """(ra, dec, mag) float arrays from a reference catalog table, tolerant of
    the column-name conventions in use (RA/DEC/refmag is the
    gaia_virac2_refcat form)."""
    cols = {c.lower(): c for c in tbl.colnames}
    if "ra" not in cols or "dec" not in cols:
        raise ValueError(f"reference catalog lacks RA/DEC columns "
                         f"(has {tbl.colnames})")
    ra = np.asarray(tbl[cols["ra"]], float)
    dec = np.asarray(tbl[cols["dec"]], float)
    for name in ("refmag", "mag", "ks", "phot_g_mean_mag"):
        if name in cols:
            mag = np.asarray(tbl[cols[name]], float)
            break
    else:
        raise ValueError(f"reference catalog lacks a magnitude column "
                         f"(has {tbl.colnames})")
    return ra, dec, mag


def _box_peak(box, box_px):
    """Peak position ``(py, px)`` (floats) in a ``(2*box_px+1)``-square
    luminance box, plateau-aware; ``None`` when there is no interior peak.

    ``nanargmax`` returns the FIRST index of the maximum, which for a
    flat-topped (clipped / near-saturated) star biases the "peak" toward the
    low-index corner of the plateau by up to the plateau radius -- enough to
    push a perfectly registered star past the 2 px gate.  So when the
    max-value region is a plateau (>= STAR_PLATEAU_MIN_PX pixels at the max)
    the plateau CENTROID is used instead.  A max region touching the box
    border is a gradient toward something outside the box, not a local peak
    -> ``None`` (unmatched)."""
    if not np.any(np.isfinite(box)):
        return None
    mx = np.nanmax(box)
    pys, pxs = np.nonzero(box == mx)
    edge = 2 * box_px
    if np.any((pys == 0) | (pys == edge) | (pxs == 0) | (pxs == edge)):
        return None                          # max touches the border: no peak
    if len(pys) >= STAR_PLATEAU_MIN_PX:
        return float(np.mean(pys)), float(np.mean(pxs))
    py, px = np.unravel_index(np.nanargmax(box), box.shape)
    return float(py), float(px)


def validate_star_positions(out_base, ref_catalog, max_stars=STAR_MAX_STARS,
                            box_px=STAR_BOX_PX):
    """Star-position check against ``ref_catalog`` (the second avm gate).

    Projects up to ``max_stars`` bright finite-mag catalog stars through the
    PNG's EMBEDDED AVM WCS (exactly as a consumer would) to pixel coordinates
    and measures the offset to the local luminance peak within a +-``box_px``
    box (plateau-aware centroid, see ``_box_peak``).  A star whose box
    maximum sits on the box border has no local peak there (it is a gradient
    toward something else) and is excluded as unmatched.  Saturated stars
    (box max at the uint8 ceiling, 255) carry no reliable centroid and are
    SKIPPED whenever >= STAR_MIN_USED unsaturated stars remain.  PASS
    requires median offset <= STAR_PASS_MEDIAN_PX px, matched fraction >=
    STAR_MIN_MATCH_FRACTION, and >= STAR_MIN_USED usable stars.

    Returns the ``checks.star_positions`` dict (``pass`` is the verdict)."""
    import pyavm
    from astropy.table import Table
    from PIL import Image

    png = out_base + ".png"
    avm_wcs = pyavm.AVM.from_image(png).to_wcs()
    with Image.open(png) as im:
        arr = np.asarray(im.convert("RGBA"), dtype=float)
    arr = np.flipud(arr)                     # back to FITS (bottom-up) rows
    lum = arr[..., :3].mean(axis=2)
    lum[arr[..., 3] == 0] = np.nan           # transparent = no data
    ny, nx = lum.shape

    ra, dec, mag = _refcat_radec_mag(Table.read(ref_catalog))
    finite = np.isfinite(ra) & np.isfinite(dec) & np.isfinite(mag)
    ra, dec, mag = ra[finite], dec[finite], mag[finite]
    xs, ys = avm_wcs.wcs_world2pix(ra, dec, 0)
    # a star far off the projection can come back non-finite: not in footprint
    proj = np.isfinite(xs) & np.isfinite(ys)
    xi = np.full(xs.shape, -1, dtype=int)
    yi = np.full(ys.shape, -1, dtype=int)
    xi[proj] = np.round(xs[proj]).astype(int)
    yi[proj] = np.round(ys[proj]).astype(int)
    # footprint: full box inside the image, on an opaque (data) pixel
    inside = (proj & (xi >= box_px) & (xi < nx - box_px)
              & (yi >= box_px) & (yi < ny - box_px))
    on_data = np.zeros(len(xs), dtype=bool)
    on_data[inside] = np.isfinite(lum[yi[inside], xi[inside]])
    order = np.argsort(mag)                  # brightest (smallest mag) first
    keep = order[on_data[order]][:max_stars]

    usable = []                              # (offset_px, saturated) per star
    n_selected = int(len(keep))
    for k in keep:
        box = lum[yi[k] - box_px: yi[k] + box_px + 1,
                  xi[k] - box_px: xi[k] + box_px + 1]
        peak = _box_peak(box, box_px)
        if peak is None:
            continue                         # no interior local peak
        py, px = peak
        off = float(np.hypot(xi[k] - box_px + px - xs[k],
                             yi[k] - box_px + py - ys[k]))
        usable.append((off, bool(np.nanmax(box) >= STAR_SATURATED_LUM)))

    # saturated (uint8-clipped) stars have no reliable peak position; skip
    # them whenever enough unsaturated stars remain to conclude from
    unsaturated = [off for off, sat in usable if not sat]
    if len(unsaturated) >= STAR_MIN_USED:
        offsets = unsaturated
        n_saturated_skipped = len(usable) - len(unsaturated)
    else:
        offsets = [off for off, _sat in usable]
        n_saturated_skipped = 0

    n_used = len(offsets)
    matched_fraction = (n_used / n_selected) if n_selected else 0.0
    median = float(np.median(offsets)) if offsets else None
    ok = (n_used >= STAR_MIN_USED
          and matched_fraction >= STAR_MIN_MATCH_FRACTION
          and median is not None and median <= STAR_PASS_MEDIAN_PX)
    result = {
        "pass": bool(ok),
        "median_offset_px": median,
        "n_used": n_used,
        "n_selected": n_selected,
        "n_saturated_skipped": n_saturated_skipped,
        "matched_fraction": float(matched_fraction),
        "ref_catalog": os.path.abspath(ref_catalog),
    }
    if n_selected == 0:
        # distinct from "stars matched poorly": nothing to check at all
        result["reason"] = "catalog does not overlap image footprint"
    return result


# --------------------------------------------------------------------------- validate
def _wcs_points(wcs, shape):
    """Reference pixel + 4 corners, as (x, y) 0-based pixel coordinates."""
    ny, nx = shape
    crx, cry = (wcs.wcs.crpix[0] - 1.0, wcs.wcs.crpix[1] - 1.0)
    crx = min(max(crx, 0.0), nx - 1.0)
    cry = min(max(cry, 0.0), ny - 1.0)
    return [(crx, cry), (0.0, 0.0), (nx - 1.0, 0.0), (0.0, ny - 1.0),
            (nx - 1.0, ny - 1.0)]


def validate(out_base, f212n_path, long_path, long_band="F480M",
             red_reproj=None, write_json=True, field=None, ref_catalog=None):
    """Validate ``<out>.png`` against the F212N FITS WCS + alpha/NaN rules,
    plus the star-position check whenever a reference catalog is available.

    ``red_reproj`` (the reprojected long array) enables the EXACT two-sided
    alpha check; standalone validation (no reprojection in hand) falls back to
    the one-sided check "finite F212N pixel => opaque", which every correctly
    written image also satisfies (alpha=0 only where BOTH bands are NaN).

    The star check runs AUTOMATICALLY when ``ref_catalog`` is given or the
    ``field`` convention finds one (``find_ref_catalog``); with no catalog it
    is recorded as ``checks.star_positions = {skipped: true, reason}`` --
    ``publish.py``'s avm gate then demands an explicit ``--no-star-check``
    acknowledgment.  A star check that RUNS and FAILS fails the whole verdict.
    Returns the verdict dict (``verdict['pass']`` is the overall result)."""
    import pyavm
    from PIL import Image

    png = out_base + ".png"
    checks = {}

    blue, fits_wcs, _ = load_sci(f212n_path)
    ny, nx = blue.shape

    # -- WCS round-trip: AVM-reconstructed WCS vs FITS WCS at 5 points
    avm = pyavm.AVM.from_image(png)
    avm_wcs = avm.to_wcs()
    pts = _wcs_points(fits_wcs, blue.shape)
    xs = np.array([p[0] for p in pts])
    ys = np.array([p[1] for p in pts])
    ref = fits_wcs.pixel_to_world(xs, ys)
    got = avm_wcs.pixel_to_world(xs, ys)
    seps = ref.separation(got).arcsec
    checks["wcs_max_offset_arcsec"] = float(np.max(seps))
    checks["wcs_pass"] = bool(np.max(seps) < WCS_PASS_ARCSEC)

    # -- alpha vs NaN + finite fraction
    with Image.open(png) as im:
        arr = np.asarray(im.convert("RGBA"))
    alpha = np.flipud(arr[..., 3])          # back to FITS (bottom-up) rows
    checks["shape_pass"] = bool(alpha.shape == blue.shape)
    if checks["shape_pass"]:
        if red_reproj is not None:
            finite = np.isfinite(blue) | np.isfinite(red_reproj)
            mism = np.mean((alpha > 0) != finite)
            checks["alpha_mode"] = "exact (both bands)"
        else:
            # one-sided: any finite F212N pixel must be opaque
            bad = np.isfinite(blue) & (alpha == 0)
            mism = np.mean(bad)
            checks["alpha_mode"] = "one-sided (F212N only; standalone)"
        checks["alpha_mismatch_fraction"] = float(mism)
        checks["alpha_pass"] = bool(mism == 0.0)
        frac = float(np.mean(alpha > 0))
    else:
        checks["alpha_mode"] = "skipped (shape mismatch)"
        checks["alpha_pass"] = False
        frac = 0.0
    checks["finite_fraction"] = frac
    checks["finite_pass"] = bool(frac > 0.0)

    # -- star positions (second gate): automatic whenever a catalog is known
    cat = ref_catalog or find_ref_catalog(field)
    if cat:
        checks["star_positions"] = validate_star_positions(out_base, cat)
    else:
        reason = (f"no reference catalog for field {field!r} under "
                  f"{REFCAT_ROOT}/{field}/catalogs/" if field
                  else "no --field/--ref-catalog given")
        checks["star_positions"] = {"skipped": True, "reason": reason}

    ok = all(checks[k] for k in ("wcs_pass", "shape_pass", "alpha_pass",
                                 "finite_pass"))
    if not checks["star_positions"].get("skipped"):
        ok = ok and checks["star_positions"]["pass"]
    verdict = {
        "pass": bool(ok),
        "checks": checks,
        "timestamp": datetime.datetime.now().astimezone().isoformat(),
        "inputs": {
            "f212n": {"path": os.path.abspath(f212n_path),
                      "sha256": _sha256(f212n_path)},
            "long": {"path": os.path.abspath(long_path),
                     "sha256": _sha256(long_path)},
            "long_band": long_band,
            "ref_catalog": os.path.abspath(cat) if cat else None,
        },
        # sha256 of the EXACT files this verdict vouches for: publish.py
        # re-hashes at push time and refuses on mismatch, so a PNG/JPG
        # regenerated after validation can never ride a stale verdict.
        "outputs": {"png": os.path.abspath(png),
                    "png_sha256": _sha256(png),
                    "jpg": os.path.abspath(out_base + ".jpg"),
                    "jpg_sha256": (_sha256(out_base + ".jpg")
                                   if os.path.exists(out_base + ".jpg")
                                   else None)},
    }
    if write_json:
        with open(out_base + ".validation.json", "w") as fh:
            json.dump(verdict, fh, indent=2)
    return verdict


# --------------------------------------------------------------------------- CLI
def _one_field(f212n, long_path, long_band, out_base, exact=False,
               pipe_root=None, validate_only=False, percentiles=PERCENTILES,
               field=None, ref_catalog=None, require_stars=False):
    cat = ref_catalog or find_ref_catalog(field)
    if require_stars and not cat:
        print(f"[rgb_treasury] ERROR: --validate-stars but no reference "
              f"catalog (field={field!r}; looked for "
              f"gaia_virac2_refcat*/gaia_refcat* under "
              f"{REFCAT_ROOT}/<field>/catalogs/; or pass --ref-catalog)",
              file=sys.stderr)
        return False
    if validate_only:
        red = None
    else:
        png, jpg, red, _blue, _wcs = build_rgb(
            f212n, long_path, out_base, long_band=long_band, exact=exact,
            percentiles=percentiles, pipe_root=pipe_root)
        print(f"[rgb_treasury] wrote {png} + {jpg}")
    verdict = validate(out_base, f212n, long_path, long_band=long_band,
                       red_reproj=red, field=field, ref_catalog=cat)
    status = "PASS" if verdict["pass"] else "FAIL"
    stars = verdict["checks"]["star_positions"]
    star_txt = ("stars skipped"
                if stars.get("skipped") else
                f"stars {'PASS' if stars['pass'] else 'FAIL'} "
                f"(median {stars['median_offset_px']} px, "
                f"n={stars['n_used']}/{stars['n_selected']})")
    print(f"[rgb_treasury] validation {status}: "
          f"wcs max offset {verdict['checks'].get('wcs_max_offset_arcsec', -1):.4g}\" "
          f"({verdict['checks'].get('alpha_mode')}), "
          f"finite fraction {verdict['checks'].get('finite_fraction', 0):.3f}, "
          f"{star_txt} "
          f"-> {out_base}.validation.json")
    return verdict["pass"]


def build_parser():
    p = argparse.ArgumentParser(
        prog="python -m data_qa.rgb_treasury", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--f212n", help="F212N (blue) merged _i2d.fits")
    p.add_argument("--long", dest="long_path", help="long-band (red) _i2d.fits")
    p.add_argument("--long-band", choices=LONG_BANDS, default="F480M")
    p.add_argument("--out", help="output basename (writes <out>.png/.jpg/"
                                 ".validation.json)")
    p.add_argument("--fields-spec", help="JSON batch spec (see module docstring)")
    p.add_argument("--exact", action="store_true",
                   help="flux-conserving reproject_exact (slow); default interp")
    p.add_argument("--validate", action="store_true",
                   help="skip the build; validate existing <out>.png only")
    p.add_argument("--field", help="field name (e.g. sgrb2): finds the "
                                   "reference catalog by convention and runs "
                                   "the star-position check automatically")
    p.add_argument("--ref-catalog", help="explicit reference catalog FITS "
                                         "(overrides the --field convention)")
    p.add_argument("--validate-stars", action="store_true",
                   help="REQUIRE the star-position check (error if no "
                        "reference catalog can be found)")
    p.add_argument("--percentiles", nargs=2, type=float, default=list(PERCENTILES),
                   metavar=("LO", "HI"), help="global stretch percentiles")
    p.add_argument("--pipe-root", default=DEFAULT_PIPE_ROOT,
                   help="jwst-gc-pipeline checkout for the cmz.hips import "
                        "fallback")
    p.add_argument("--dry-run", action="store_true",
                   help="batch mode: print the plan, build nothing")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    pct = tuple(args.percentiles)
    if args.fields_spec:
        with open(args.fields_spec) as fh:
            spec = json.load(fh)
        out_dir = spec.get("out_dir", ".")
        ok = True
        for f in spec["fields"]:
            out = f.get("out") or os.path.join(out_dir, f["name"] + "_rgb")
            if args.dry_run:
                print(f"[rgb_treasury] would build {f['name']}: "
                      f"B={f['f212n_i2d']} R={f['long_i2d']} "
                      f"({f.get('long_band', 'F480M')}) -> {out}.png/.jpg")
                continue
            ok &= _one_field(f["f212n_i2d"], f["long_i2d"],
                             f.get("long_band", "F480M"), out,
                             exact=args.exact, pipe_root=args.pipe_root,
                             validate_only=args.validate, percentiles=pct,
                             field=f.get("field", f["name"]),
                             ref_catalog=f.get("ref_catalog"),
                             require_stars=args.validate_stars)
        return 0 if ok else 1
    if not (args.f212n and args.long_path and args.out):
        build_parser().error("--f212n, --long and --out are required "
                             "(or use --fields-spec)")
    ok = _one_field(args.f212n, args.long_path, args.long_band, args.out,
                    exact=args.exact, pipe_root=args.pipe_root,
                    validate_only=args.validate, percentiles=pct,
                    field=args.field, ref_catalog=args.ref_catalog,
                    require_stars=args.validate_stars)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
