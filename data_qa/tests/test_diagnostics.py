"""Unit tests for the QA diagnostics helpers that previously had ZERO coverage.

Covers the pure numeric helpers (`_binned_stat`), the caption fallback that once printed
"nanσ" when the spread was absent (`caption_for`), and the observation-scoping of the
per-exposure daophot glob (`_daophot_glob`) that keeps one observation's cats out of another's
QA on multi-obs fields.  No I/O beyond touching empty files under a temporary QA_BASE.
"""
import glob
import os

import numpy as np
import pytest

from data_qa import diagnostics as D
from data_qa.observations import Observation


def _obs(field="gc2211", obs="023", filt="F200W"):
    return Observation(program="2211", obs=obs, target="T", release_field=field,
                       instrument="NIRCam", filters=[filt], visits=[], epoch="", notes="")


def test_base_field_and_viraccache_fallback(tmp_path, monkeypatch):
    # a per-obs split field falls back to its base field's VIRAC Ks refcache (issue #119)
    assert D._base_field("gc2211_o023") == "gc2211"
    assert D._base_field("gc2211") == "gc2211"          # no suffix -> unchanged
    assert D._base_field("cloudef_controlfield") == "cloudef_controlfield"  # not an _o<obs> split
    monkeypatch.setattr(D, "BASE", str(tmp_path))
    o = _obs(field="gc2211_o023")
    assert D._viraccache_path(o) is None                # nothing on disk
    # cache only under the BASE field -> the per-obs split field still finds it
    base = tmp_path / "gc2211" / "astrometry_diag" / "refcache"; base.mkdir(parents=True)
    (base / "virac2.fits").write_text("x")
    assert D._viraccache_path(o) == str(base / "virac2.fits")
    # a per-obs cache, when present, is preferred over the base one
    own = tmp_path / "gc2211_o023" / "astrometry_diag" / "refcache"; own.mkdir(parents=True)
    (own / "virac2.fits").write_text("x")
    assert D._viraccache_path(o) == str(own / "virac2.fits")


# --------------------------------------------------------------------------- _binned_stat
def test_binned_stat_basic():
    x = np.repeat(np.arange(10.0), 20)          # 10 bins, 20 pts each
    med, lo, hi, ctr = D._binned_stat(x, np.ones_like(x), width=1.0, minn=15)
    assert med is not None and len(ctr) >= 3
    assert np.allclose(med, 1.0)


def test_binned_stat_too_few_points():
    assert D._binned_stat(np.arange(5.0), np.arange(5.0), minn=15)[0] is None


def test_binned_stat_drops_sparse_bins_to_none():
    # one dense bin + two singletons -> fewer than 3 qualifying bins -> None
    x = np.concatenate([np.zeros(50), np.array([5.0, 9.0])])
    assert D._binned_stat(x, np.ones_like(x), width=1.0, minn=15)[0] is None


# --------------------------------------------------------------------------- caption_for
def test_caption_stage4_no_spread_no_nan():
    # no cell-to-cell spread -> still report the field offset over its cells, never "nan"
    cap = D.caption_for(4, dict(stage=4, sw="F212N",
                                offset_med_mas=7.1, n_cells=6, offset_scatter_mas=None))
    assert "nan" not in cap.lower()
    # a small offset keeps a decimal: re-measured from the same stars it is routinely sub-mas,
    # and "0 mas" hid the difference between 0.4 and 4.
    assert "7.1 mas" in cap and "6 measured cells" in cap


def test_caption_stage4_scatter_panel_position():
    # 2-column figure (no NRCA-NRCB panel): the (ΔRA,ΔDec) scatter is the RIGHT panel, not MIDDLE,
    # and there is no intermodule sentence.
    cap = D.caption_for(4, dict(stage=4, sw="F212N", offset_med_mas=7.1, n_cells=6))
    assert "RIGHT plots the" in cap and "MIDDLE plots" not in cap
    assert "NRCA-minus-NRCB" not in cap
    # 3-column figure (intermodule_off set -> NRCA-NRCB drawn as the RIGHT column): scatter is MIDDLE
    cap = D.caption_for(4, dict(stage=4, sw="F212N", offset_med_mas=7.1, n_cells=6,
                                intermodule_off=4.0))
    assert "MIDDLE plots the" in cap and "NRCA-minus-NRCB" in cap


def test_caption_stage4_flags_discontinuity():
    # an adjacency-confirmed deviating region is called out in the caption
    cap = D.caption_for(4, dict(stage=4, offset_med_mas=31, n_cells=12, offset_scatter_mas=8.0,
                                n_cells_confirmed=3, bad_src_frac=0.08,
                                n_cells_dropped=0))
    assert "internal discontinuity" in cap and "8%" in cap


def test_cell_map_broke_down_decision():
    # arcsec-scale cell scatter + a solid clean isolated bulk -> cell map unreliable (cloudef o005)
    assert D._cell_map_broke_down(1783.0, 12.0, 402) is True
    # a real, tight cell map (cloudef o002: 16 mas scatter) is trusted, not overridden
    assert D._cell_map_broke_down(16.0, 116.0, 3464) is False
    # a real ~58 mas sub-region discontinuity (gc2211 o050) is NOT swallowed as "unreliable"
    assert D._cell_map_broke_down(58.0, 30.0, 500) is False
    # absurd scatter but too few clean isolated stars to trust the fallback -> stay with the cells
    assert D._cell_map_broke_down(1783.0, 12.0, 10) is False
    assert D._cell_map_broke_down(1783.0, None, 0) is False


def test_caption_stage4_cell_map_unreliable_and_unmeasurable():
    # arcsec cell scatter AND no confident whole-field peak (issue #38 o005): suppress the phantom
    # "internal discontinuity", call the offset UNMEASURABLE, and flag the isolated median as a
    # possible NN-collapse -- do NOT sell it as a clean tie.
    cap = D.caption_for(4, dict(stage=4, sw="F162M", offset_med_mas=12, n_cells=16,
                                offset_scatter_mas=1783, n_cells_confirmed=16, bad_src_frac=1.0,
                                n_cells_dropped=0, cell_map_unreliable=True,
                                offset_unmeasurable=True, isolated_bulk_n=402))
    assert "per-cell map is unreliable" in cap
    assert "internal discontinuity" not in cap
    assert "not measured" in cap and "collapse" in cap and "does not auto-pass" in cap
    # the cause is a frame offset beyond the window; do not (falsely) blame a sparse reference
    assert "too sparse" not in cap and "sparse over this field" not in cap
    assert "n=402" in cap
    assert cap.count("1783 mas") == 1        # scatter stated once, in the warning


def test_caption_stage4_cell_map_unreliable_confident_wholefield():
    # cells garbage but the swept whole-field xcorr IS confident -> quote it, not "unmeasurable"
    cap = D.caption_for(4, dict(stage=4, sw="F162M", offset_med_mas=150, n_cells=16,
                                offset_scatter_mas=900, n_cells_dropped=0,
                                cell_map_unreliable=True, offset_unmeasurable=False))
    assert "swept whole-field cross-correlation" in cap
    assert "internal discontinuity" not in cap and "not measured" not in cap
    assert "150 mas" in cap


def test_caption_stage1_dropped_filters_noted():
    cap = D.caption_for(1, dict(stage=1, sw="F212N", lw="F405N", dropped_filters=["F444W", "F322W2"]))
    assert "F444W" in cap and "F322W2" in cap and "not reduced" in cap


def test_caption_redflag():
    cap = D.caption_for(3, dict(red_flag=True, red_flag_reason="no catalog for F212N"))
    assert "RED FLAG" in cap and "no catalog" in cap


def test_caption_stage6_names_formal_vs_empirical():
    # the formal PSF-fit sigma must NOT be sold as the achieved precision (issue #1 review): the
    # caption names it "formal", drops the old "systematic limit" claim, documents the source-count
    # histogram, and -- crucially -- only credits floor_mas to rms(jwst) WHEN that curve is drawn.
    emp = D.caption_for(6, dict(stage=6, sw="F212N", lw="F466N",
                                floor_is_empirical_f212n=True, floor_is_empirical_f466n=True))
    assert "systematic limit" not in emp
    # pin the POSITIVE branch on the phrase UNIQUE to it (not the "achieved" that also appears in the
    # shared "not the achieved precision" base text), so collapsing the emp condition is caught (#99).
    assert "formal" in emp and "rms(jwst)" in emp and "achieved internal repeatability" in emp
    assert "source counts per" in emp
    # no per-exposure catalogs -> rms(jwst) not drawn: the caption must NOT claim floor_mas is it,
    # and must say the fallback to the formal floor (the conflation #99 review caught).
    noemp = D.caption_for(6, dict(stage=6, sw="F115W", lw=None, floor_is_empirical_f115w=False))
    assert "falls back to the" in noemp and "formal" in noemp
    assert "achieved internal repeatability" not in noemp


def test_caption_stage5_no_overlap_two_modules():
    # both modules present but no shared stars -> overlap keys absent; keeps the measured A-B diff
    # and does not editorialize about the omitted panel (an omitted panel is simply not expected).
    cap = D.caption_for(5, dict(stage=5, intermodule_diff=3.0, passed=False))
    assert "nan" not in cap.lower()
    assert "could not be measured" not in cap and "unverified" not in cap
    assert "A–B diff" in cap and "3.0 mas" in cap


def test_caption_stage5_single_module():
    cap = D.caption_for(5, dict(stage=5, single_module="NRCA", passed=True))
    assert "Single module" in cap and "NRCA" in cap
    assert "nan" not in cap.lower()


def test_caption_stage5_full_when_overlap_present():
    # overlap + a high-S/N panel present -> the S/N clause appears and the all-stars panel is
    # TOP-MIDDLE (3-column figure)
    cap = D.caption_for(5, dict(stage=5, intermodule_diff=3.0, intermodule_off=4.1,
                                intermodule_rms=6.2, n_overlap=137,
                                intermodule_off_hi=4.0, intermodule_rms_hi=4.4, n_overlap_hi=34355))
    assert "137 shared stars" in cap and "4.1 mas" in cap
    assert "TOP-LEFT" in cap and "glossary-reffree" in cap  # panels labelled + term linked
    assert "S/N > 10" in cap and "marginal" in cap          # the new S/N panel + marginals noted
    assert "TOP-MIDDLE" in cap and "to its right" in cap    # correct panel positions
    # the NRCB2/no-overlap question is answered inline
    assert "NRCB2" in cap and "shares no sky with NRCA" in cap


def test_caption_stage5_surfaces_cutout_footprint_mismatch():
    base = dict(stage=5, intermodule_diff=3.0, intermodule_off=4.1, intermodule_rms=6.2, n_overlap=137)
    # no drizzled mosaic covers the overlap zone -> the caption says so, not the normal cutout blurb
    cap = D.caption_for(5, dict(base, cutout_footprint_mismatch=True))
    assert "disjoint footprints" in cap and "overlap-star cutouts" not in cap
    # normal case keeps the cutout description
    assert "overlap-star cutouts" in D.caption_for(5, dict(base, cutout_footprint_mismatch=False))


def test_caption_stage5_overlap_without_hi_sn_panel():
    # overlap present but no S/N>10 panel (field lacks flux errors) -> no S/N promise; the all-stars
    # panel is TOP-RIGHT (2-column top row), and the footprint is its own full-width row below
    cap = D.caption_for(5, dict(stage=5, intermodule_diff=3.0, intermodule_off=4.1,
                                intermodule_rms=6.2, n_overlap=137, n_overlap_footprint=137))
    assert "S/N > 10" not in cap and "to its right" not in cap
    assert "TOP-RIGHT" in cap and "137 shared stars" in cap
    assert "full-width row" in cap and "|A−B|" in cap   # footprint described


# --------------------------------------------------------------------------- doc links / clarity
def test_caption_linkifies_docroot():
    # no caption may leave the DOCROOT sentinel unresolved, and every one carries a doc link
    for m in (dict(stage=3, sw="F212N", n_matched=100, slope=1.0, scatter=0.2),
              dict(stage=5, single_module="NRCA", passed=True),
              dict(red_flag=True, red_flag_reason="x")):
        n = m.get("stage", 3)
        cap = D.caption_for(n, m)
        assert "DOCROOT" not in cap
        assert "qa_methods.md#" in cap


def test_caption_stage3_grades_on_our_catalog_and_names_mast():
    cap = D.caption_for(3, dict(stage=3, sw="F212N", source="jicama-m8", n_matched=2603,
                                slope=1.0, scatter=0.28, our_slope=1.0,
                                mast_slope=0.99, mast_scatter=0.30))
    assert "right stars were matched" not in cap and "NOT a fit" not in cap
    # the graded panel is named (our catalogue) and MAST is called out as in the dropdown
    assert "jicama-m8" in cap and "MAST" in cap and "dropdown" in cap
    assert "slope" in cap and "scatter" in cap


def test_caption_stage3_informational_when_mast_only():
    # no pipeline catalogue yet -> MAST shown for information, stage not graded (not a red flag)
    cap = D.caption_for(3, dict(stage=3, sw="F212N", passed=None,
                                primary_source="MAST catalogue", mast_slope=0.02, mast_scatter=1.2,
                                na_reason="pipeline catalogue not yet available"))
    assert "information" in cap and "not graded" in cap
    assert "MAST" in cap


def test_caption_stage2_spells_out_lf_and_drops_meaningless_clause():
    cap = D.caption_for(2, dict(stage=2, kind="m7", n_stars=161196, lf_turnover=18.3,
                                sw="F212N", lw="F405N"))
    assert "luminosity function" in cap and "regenerated as the catalog deepens" not in cap
    assert "m7" in cap and "qa_methods.md#glossary-mtier" in cap    # catalog term links out


def test_caption_stage2_crossmatch_and_single_filter_variants():
    xm = D.caption_for(2, dict(stage=2, kind="crossmatch", n_stars=800, sw="F212N", lw="F405N"))
    assert "cross-match tolerance" in xm and "DOCROOT" not in xm    # no lf_turnover -> no crash
    sf = D.caption_for(2, dict(stage=2, kind="m8_dedup", n_stars=500, lf_turnover=17.0,
                               sw="F212N", single_filter=True))
    assert "luminosity function" in sf and "single filter" in sf


def test_caption_anchors_exist_in_docs():
    # every qa_methods.md#<anchor> a caption emits must exist as an <a id="..."> in the doc
    import re
    docs = os.path.join(os.path.dirname(__file__), "..", "..", "docs", "qa_methods.md")
    with open(docs) as fh:
        ids = set(re.findall(r'<a id="([^"]+)"', fh.read()))
    samples = {
        1: dict(stage=1, sw="F212N", lw="F405N"),
        2: dict(stage=2, kind="m8", n_stars=1000, lf_turnover=18.0, sw="F212N", lw="F405N"),
        3: dict(stage=3, sw="F212N", n_matched=500, slope=1.0, scatter=0.3),
        4: dict(stage=4, offset_med_mas=12.0, n_cells=14, offset_scatter_mas=6.0,
                n_cells_dropped=2),
        5: dict(stage=5, intermodule_diff=3.0, intermodule_off=4.1, intermodule_rms=6.0,
                n_overlap=100, n_overlap_hi=50, intermodule_rms_hi=4.0),
        6: dict(red_flag=True, red_flag_reason="x"),
        9: dict(stage=9, n_isolated=19812, aper_corr_med=0.45, aper_psf_scatter=0.07),
        12: dict(stage=12, sw="F212N", primary_filter="F212N", filters_measured=["F212N"],
                 per_filter={"F212N": dict(slope=0.006, slope_err=0.004, turnover_mag=None,
                                           aper_corr=0.42, n=1800, flagged=False)},
                 slope=0.006, slope_err=0.004, turnover_mag=None, aper_corr=0.42, n_flagged=0),
    }
    for n, m in samples.items():
        cap = D.caption_for(n, m)
        for anc in re.findall(r"qa_methods\.md#([A-Za-z0-9\-]+)", cap):
            assert anc in ids, f"stage {n} caption links #{anc} but no <a id> exists in the doc"


def test_recentroid_com_snaps_to_offset_star():
    # a star ~3 px from the catalog position (a stale catalogue vs a re-tied mosaic) must be
    # recovered so the aperture lands on it, not on blank sky (issue #38 stage 9).
    ny = nx = 41
    yy, xx = np.mgrid[0:ny, 0:nx]
    cx0, cy0 = 25.0, 18.0                                   # true star centre
    img = 100.0 * np.exp(-(((xx - cx0) ** 2 + (yy - cy0) ** 2) / (2 * 1.5 ** 2))) + 1.0
    xn, yn, moved = D._recentroid_com(img, np.array([22.0]), np.array([16.0]), box=11)
    assert abs(xn[0] - cx0) < 0.5 and abs(yn[0] - cy0) < 0.5   # snapped onto the star
    assert moved[0] > 2.0                                   # and it reports the shift it made


def test_recentroid_com_keeps_edge_star_put():
    # a catalog position whose stamp runs off the image keeps its original position (moved = 0)
    img = np.ones((41, 41)) + 0.0
    xn, yn, moved = D._recentroid_com(img, np.array([2.0]), np.array([2.0]), box=11)
    assert xn[0] == 2.0 and yn[0] == 2.0 and moved[0] == 0.0


def test_caption_stage9_psf_vs_aper():
    cap = D.caption_for(9, dict(stage=9, n_isolated=19812, aper_corr_med=0.45,
                                aper_psf_scatter=0.073, frac_gt_0p3mag=0.01))
    assert "DOCROOT" not in cap and "qa_methods.md#stage9" in cap
    assert "PSF vs aperture" in cap and "isolated" in cap
    assert "19812 isolated stars" in cap and "+0.45 mag" in cap


def test_stage9_end_to_end_synthetic(tmp_path, monkeypatch):
    pytest.importorskip("photutils"); pytest.importorskip("scipy")
    from astropy.io import fits
    from astropy.wcs import WCS
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    # a 520x520 frame with a 10x10 grid (100) of well-separated Gaussians of KNOWN total flux
    ny = nx = 520
    yy, xx = np.mgrid[0:ny, 0:nx]
    gx = np.linspace(40, nx - 40, 10); gy = np.linspace(40, ny - 40, 10)
    XX, YY = np.meshgrid(gx, gy)
    xs = XX.ravel(); ys = YY.ravel()             # 100 stars, ~48 px apart -> all isolated
    flux = np.full(len(xs), 1.0e4); sig = 1.5
    img = np.zeros((ny, nx), "float32")
    for xi, yi, f in zip(xs, ys, flux):
        img += (f / (2 * np.pi * sig ** 2)) * np.exp(-((xx - xi) ** 2 + (yy - yi) ** 2) / (2 * sig ** 2))
    w = WCS(naxis=2)
    w.wcs.crpix = [nx / 2, ny / 2]; w.wcs.cdelt = [-1 / 3600.0, 1 / 3600.0]
    w.wcs.crval = [266.4, -28.7]; w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    mp = str(tmp_path / "m.fits")
    fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(img, header=w.to_header(), name="SCI")]).writeto(mp)
    sc = w.pixel_to_world(xs, ys); sc = SkyCoord(sc.ra, sc.dec)
    monkeypatch.setattr(D, "_psf_flux_positions", lambda o, f: (sc, flux.copy(), "synth.fits"))
    monkeypatch.setattr(D, "_mosaic_path", lambda o, f: mp)
    o = Observation(program="2221", obs="001", target="Brick", release_field="brick",
                    instrument="NIRCam", filters=["F212N"], visits=[], epoch="", notes="")
    png, m = D.stage9_psf_vs_aper(o, "F212N")
    assert not m.get("red_flag") and m["n_isolated"] >= 30
    # PSF flux is the TOTAL; a 3px aperture misses the wings -> aperture fainter -> apcorr > 0
    assert m["aper_corr_med"] > 0 and m["aper_psf_scatter"] < 0.1


def test_provenance_footer_has_doc_and_source():
    from data_qa import post_diagnostics as P
    foot = P._provenance_footer("JWST-GC/data-qa", 4)
    assert "docs/qa_methods.md#stage4" in foot
    assert "stage4_offsets()" in foot and "data_qa/diagnostics.py" in foot


# --------------------------------------------------------------------------- _refcat_path obs-scope
def _touch(d, name):
    (d / name).write_text("")


def test_refcat_path_prefers_this_obs_tokened(tmp_path, monkeypatch):
    monkeypatch.setattr(D, "BASE", str(tmp_path))
    d = tmp_path / "gc2211" / "catalogs"; d.mkdir(parents=True)
    _touch(d, "gaia_virac2_refcat_epoch2023.71.fits")            # untokened full-field
    _touch(d, "gaia_virac2_refcat_epoch2023.71_o028.fits")       # o028-only footprint
    assert D._refcat_path(_obs(obs="028")).endswith("_o028.fits")


def test_refcat_path_falls_back_to_untokened_not_other_obs(tmp_path, monkeypatch):
    # the o023/o050/o028 bug (#7/#8/#28): a plain sorted()[-1] handed o023 the o028 refcat (a
    # disjoint patch of sky).  o023 has no tokened refcat -> must use the untokened full one.
    monkeypatch.setattr(D, "BASE", str(tmp_path))
    d = tmp_path / "gc2211" / "catalogs"; d.mkdir(parents=True)
    _touch(d, "gaia_virac2_refcat_epoch2023.71.fits")
    _touch(d, "gaia_virac2_refcat_epoch2023.71_o028.fits")
    got = D._refcat_path(_obs(obs="023"))
    assert got.endswith("epoch2023.71.fits") and "_o0" not in os.path.basename(got)


def test_refcat_path_refuses_only_other_obs_tokened(tmp_path, monkeypatch):
    monkeypatch.setattr(D, "BASE", str(tmp_path))
    d = tmp_path / "gc2211" / "catalogs"; d.mkdir(parents=True)
    _touch(d, "gaia_virac2_refcat_epoch2023.71_o028.fits")       # ONLY a foreign-obs footprint
    assert D._refcat_path(_obs(obs="023")) is None


# --------------------------------------------------------------------------- _offset_failure_reason
def test_offset_reason_no_catalog():
    r = D._offset_failure_reason(_obs(), "F200W", None, object(), None)
    assert "not catalogued" in r and "F200W" in r


def test_offset_reason_no_reference():
    r = D._offset_failure_reason(_obs(), "F200W", object(), None, None)
    assert "no virac reference" in r.lower()


def test_offset_reason_disjoint_footprint():
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    j = SkyCoord([266.40, 266.41] * u.deg, [-28.90, -28.89] * u.deg)   # north patch
    r = SkyCoord([266.40, 266.41] * u.deg, [-29.20, -29.19] * u.deg)   # south patch, disjoint
    msg = D._offset_failure_reason(_obs(), "F200W", j, r, {"peak_ratio": 0.0})
    assert "do not" in msg and "overlap" in msg


def test_offset_reason_overlap_but_no_peak(monkeypatch):
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    # no I/O: the reason may quote the JWST mag range, so stub the catalog read out
    monkeypatch.setattr(D, "_jwst_sources", lambda o, f: (None, None, None))
    sc = SkyCoord([266.40, 266.41, 266.42] * u.deg, [-28.90, -28.89, -28.88] * u.deg)
    msg = D._offset_failure_reason(_obs(), "F200W", sc, sc, {"peak_ratio": 0.3, "npairs": 2})
    # must NOT assert an unmeasured cause -- reports the measured counts and says it is undetermined
    assert "was not determined" in msg
    assert "3 JWST sources vs 3 VIRAC reference stars" in msg


# --------------------------------------------------------------------------- cell-based stage-4
def _grid_cells(spec):
    """Build cells from {(i,j): (dra, dde, n)} for _cell_consistency tests."""
    return [dict(i=i, j=j, ra=0.0, dec=0.0, dra=d[0], dde=d[1], off=float(np.hypot(*d[:2])),
                 peak_ratio=5.0, n=d[2]) for (i, j), d in spec.items()]


def test_cell_consistency_uniform_passes():
    cells = _grid_cells({(i, j): (9.0 + i, 3.0 + j, 40000) for i in range(4) for j in range(4)})
    cc = D._cell_consistency(cells, [])
    assert cc["off_med"] < 20 and cc["consistent"] and cc["n_confirmed"] == 0


def test_cell_consistency_source_weighted_offset():
    # 99% of sources sit in cells offset by ~130 mas and 1% in a cell offset by ~28, so the offset
    # the CATALOG carries is ~130.  (The density-biased peak-ratio cut used to keep the sparse
    # 28 mas side and report that instead; #54 review 🔴1.)
    cells = _grid_cells({**{(i, j): (130.0, 0.0, 40000) for i in range(4) for j in range(4) if not (i == 0 and j == 0)},
                         (0, 0): (28.0, 0.0, 400)})
    cc = D._cell_consistency(cells, [])
    assert cc["off_med"] > 100            # weighted by source count, so the sparse cell loses


def test_cell_consistency_adjacent_deviation_fails():
    # a coherent block of adjacent cells 130 mas off (holding real sources) -> inconsistent
    base = {(i, j): (9.0, 0.0, 40000) for i in range(4) for j in range(4)}
    for ij in [(3, 2), (3, 3), (2, 3)]:       # an adjacent corner block, ~130 mas off
        base[ij] = (130.0, 0.0, 40000)
    cc = D._cell_consistency(_grid_cells(base), [])
    assert cc["off_med"] < 75 and not cc["consistent"] and cc["n_confirmed"] >= 3


def test_cell_consistency_isolated_outlier_ignored():
    # one lone 544 mas cell amid 9 mas neighbours (no adjacent deviator) -> NOT failed (#54 review 🔴2)
    base = {(i, j): (9.0, 0.0, 20000) for i in range(4) for j in range(4)}
    base[(1, 2)] = (544.0, 0.0, 19000)        # isolated
    cc = D._cell_consistency(_grid_cells(base), [])
    assert cc["n_deviating"] == 1 and cc["n_confirmed"] == 0 and cc["consistent"]


def test_cell_consistency_rejects_spurious_low_occupancy_cells():
    # cloudef o002 (#37): 3 dense cells consistent at ~150 mas + 4 low-occupancy edge cells with
    # wild, mutually-inconsistent offsets (>300 mas from consensus, each <2% of the sources) =
    # spurious per-cell xcorr peaks.  A tie cannot differ by ~arcsec between cells, so they are
    # dropped, not shown as "measured".
    cells = _grid_cells({
        (3, 0): (-145.0, 38.0, 105000), (3, 1): (-143.0, 58.0, 88000), (3, 2): (-139.0, 69.0, 88000),
        (0, 1): (-1771.0, 1482.0, 1545), (0, 2): (-529.0, -270.0, 1460),
        (0, 3): (-787.0, -1260.0, 1133), (3, 3): (-575.0, 775.0, 558),
    })
    cc = D._cell_consistency(cells, [])
    assert cc["n_spurious"] == 4 and cc["n_cells"] == 3 and cc["n_dropped"] == 4
    assert cc["spread"] < 30                       # survivors agree; no 782 mas inflation
    assert 140 < cc["off_med"] < 170               # uniform ~150 mas field offset preserved


def test_cell_consistency_keeps_high_weight_far_cell():
    # a WELL-POPULATED cell far from consensus is a real discontinuity, not a spurious peak: it must
    # be kept (and flagged by adjacency), never dropped by the spurious filter.
    base = {(i, j): (9.0, 0.0, 40000) for i in range(4) for j in range(4)}
    base[(0, 0)] = (409.0, 0.0, 40000)             # 400 mas off, full cell's worth of sources
    cc = D._cell_consistency(_grid_cells(base), [])
    assert cc["n_spurious"] == 0 and cc["n_cells"] == 16 and cc["n_deviating"] == 1


def test_isolated_bulk_recovers_known_offset():
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    # 120 well-separated (2") stars; VIRAC = same shifted +30 mas in RA, no decoys -> clean match set
    ra = 266.4 + np.arange(120) * (2.0 / 3600.0)
    dec = np.full(120, -28.5)
    jsc = SkyCoord(ra * u.deg, dec * u.deg)
    # VIRAC placed 30 mas east of JWST; the function returns JWST−VIRAC, so dRA should be −30
    ref = SkyCoord((ra + 30.0 / 3.6e6 / np.cos(np.radians(-28.5))) * u.deg, dec * u.deg)
    out = D._isolated_bulk(jsc, ref)
    assert out is not None
    mdra, mdde, n = out
    assert n >= 100 and abs(mdra - (-30.0)) < 3.0 and abs(mdde) < 3.0


def test_crossmatch_offset_normalises_and_rejects_edge_alias(monkeypatch):
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    a = SkyCoord(np.linspace(266.40, 266.42, 100) * u.deg, np.full(100, -28.9) * u.deg)
    # a clean peak passes through
    monkeypatch.setattr(D, "_pipe_measure_offset", lambda x, y, confirm_windows=True: dict(
        off=8.0, dra=6.0, ddec=5.0, contrast=40.0, ok=True, window_edge_fraction=0.02,
        window_arcsec=3.0, npairs=500))
    r = D._crossmatch_offset(a, a)
    assert r["off"] == 8.0 and r["ok"] and r["source"] == "measure_offset"
    # a window-edge alias is forced not-ok even though the pipeline gate passed it
    monkeypatch.setattr(D, "_pipe_measure_offset", lambda x, y, confirm_windows=True: dict(
        off=7000.0, dra=-1000.0, ddec=-6900.0, contrast=200.0, ok=True,
        window_edge_fraction=0.75, window_arcsec=10.0, npairs=9))
    r2 = D._crossmatch_offset(a, a)
    assert r2["ok"] is False and r2["edge"] == 0.75


def test_crossmatch_offset_falls_back_to_xcorr(monkeypatch):
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    monkeypatch.setattr(D, "_pipe_measure_offset", None)          # pipeline unavailable (CI path)
    ra = 266.40 + np.arange(400) * (0.5 / 3600.0); dec = np.full(400, -28.9)
    a = SkyCoord(ra * u.deg, dec * u.deg)
    b = SkyCoord((ra + 30.0 / 3.6e6 / np.cos(np.radians(-28.9))) * u.deg, dec * u.deg)
    r = D._crossmatch_offset(a, b)
    assert r is not None and r["source"] == "xcorr" and np.isfinite(r["off"])


def test_crossmatch_offset_restricts_virac_to_jwst_footprint(monkeypatch):
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    jwst = SkyCoord(np.linspace(266.40, 266.41, 200) * u.deg, np.full(200, -28.90) * u.deg)
    # VIRAC: 200 stars in the JWST region + 5000 far away (a full-tile refcat)
    ra = np.concatenate([np.linspace(266.40, 266.41, 200), np.linspace(200.0, 260.0, 5000)])
    dec = np.concatenate([np.full(200, -28.90), np.full(5000, 10.0)])
    ref = SkyCoord(ra * u.deg, dec * u.deg)
    seen = {}

    def spy(a, b, confirm_windows=True):
        seen["n"] = len(b)
        return dict(off=1.0, dra=1.0, ddec=0.0, contrast=50.0, ok=True,
                    window_edge_fraction=0.01, window_arcsec=3.0, npairs=100)
    monkeypatch.setattr(D, "_pipe_measure_offset", spy)
    D._crossmatch_offset(jwst, ref, restrict_footprint=True)
    assert seen["n"] < 300                          # far VIRAC cropped out
    D._crossmatch_offset(jwst, ref, restrict_footprint=False)
    assert seen["n"] > 5000                          # full refcat passed through


def test_catalog_staleness_only_uses_the_virac2locked_table(tmp_path, monkeypatch):
    # A catalogue NEWER than the operative VIRAC2locked table but OLDER than a newer legacy table
    # (VVV/consensus/per-filter) must NOT read as stale -- the check compares only against the
    # operative table.  Fails if the glob widens back to Offsets_*.csv (PR #101 review).
    monkeypatch.setattr(D, "BASE", str(tmp_path))
    cats = tmp_path / "brick" / "catalogs"; offs = tmp_path / "brick" / "offsets"
    cats.mkdir(parents=True); offs.mkdir(parents=True)
    name = "merged_cat.fits"
    _touch(cats, name)
    _touch(offs, "Offsets_JWST_Brick2221_VIRAC2locked.csv")
    _touch(offs, "Offsets_JWST_Brick2221_VVV_average.csv")
    t0 = 1_000_000_000.0
    os.utime(str(offs / "Offsets_JWST_Brick2221_VIRAC2locked.csv"), (t0, t0))          # operative
    os.utime(str(cats / name), (t0 + 5 * 86400, t0 + 5 * 86400))                       # catalogue newer
    os.utime(str(offs / "Offsets_JWST_Brick2221_VVV_average.csv"), (t0 + 10 * 86400, t0 + 10 * 86400))  # newer legacy
    o = Observation(program="2221", obs="001", target="Brick", release_field="brick",
                    instrument="NIRCam", filters=["F212N"], visits=[], epoch="", notes="")
    cdate, adate, cname = D._catalog_vs_alignment_age(o, f"release:{name}")
    assert cdate is None                              # NOT stale vs the VIRAC2locked table


def test_a_consensus_source_field_is_not_invisible_to_the_staleness_check(tmp_path,
                                                                          monkeypatch):
    """arches and w51 have no VIRAC2locked table -- `alignment_config` dispatches them to the
    checkpoint-written consensus table.  Globbing only for VIRAC2locked returned "not stale" for
    them whatever the dates, which is how arches came to be QA'd on a release catalogue seven
    weeks older than its own alignment (stage 4: 14.8 mas; stage 7: astrom_improved False)."""
    monkeypatch.setattr(D, "BASE", str(tmp_path))
    cats = tmp_path / "arches" / "catalogs"
    offs = tmp_path / "arches" / "offsets"
    cats.mkdir(parents=True); offs.mkdir(parents=True)
    name = "basic_merged_indivexp_photometry_tables_merged_resbgsub_m7.fits"
    _touch(cats, name)
    _touch(offs, "Offsets_JWST_Brick2045_consensus.csv")
    t0 = 1_000_000_000.0
    os.utime(str(cats / name), (t0, t0))                                     # catalogue first
    os.utime(str(offs / "Offsets_JWST_Brick2045_consensus.csv"),
             (t0 + 47 * 86400, t0 + 47 * 86400))                             # alignment 47 days later
    o = Observation(program="2045", obs="001", target="Arches", release_field="arches",
                    instrument="NIRCam", filters=["F212N"], visits=[], epoch="", notes="")
    cdate, adate, cname = D._catalog_vs_alignment_age(o, f"release:{name}")
    assert cdate is not None, (
        "a consensus-source field reported not-stale with its catalogue 47 days older "
        "than its only alignment table")
    assert cname == name


def test_a_stale_consensus_table_does_not_outrank_the_locked_one(tmp_path, monkeypatch):
    """The PR #101 finding, which the fallback must not undo: where a locked table exists it is
    the operative one, and a NEWER consensus table beside it must not set the bar."""
    monkeypatch.setattr(D, "BASE", str(tmp_path))
    cats = tmp_path / "cloudef" / "catalogs"
    offs = tmp_path / "cloudef" / "offsets"
    cats.mkdir(parents=True); offs.mkdir(parents=True)
    name = "basic_merged_indivexp_photometry_tables_merged_resbgsub_m8.fits"
    _touch(cats, name)
    _touch(offs, "Offsets_JWST_Brick2092_VIRAC2locked.csv")
    _touch(offs, "Offsets_JWST_Brick2092_consensus.csv")
    t0 = 1_000_000_000.0
    os.utime(str(offs / "Offsets_JWST_Brick2092_VIRAC2locked.csv"), (t0, t0))
    os.utime(str(cats / name), (t0 + 5 * 86400, t0 + 5 * 86400))             # newer than operative
    os.utime(str(offs / "Offsets_JWST_Brick2092_consensus.csv"),
             (t0 + 30 * 86400, t0 + 30 * 86400))                             # newer, but NOT operative
    o = Observation(program="2092", obs="002", target="CloudEF", release_field="cloudef",
                    instrument="NIRCam", filters=["F212N"], visits=[], epoch="", notes="")
    cdate, _adate, _cname = D._catalog_vs_alignment_age(o, f"release:{name}")
    assert cdate is None, "the consensus table set the bar while a locked table existed"


def test_isolated_bulk_none_when_too_sparse():
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    jsc = SkyCoord(np.array([266.4, 266.5]) * u.deg, np.array([-28.5, -28.5]) * u.deg)  # <100
    assert D._isolated_bulk(jsc, jsc) is None


def test_cell_consistency_componentwise_median_can_read_zero_on_a_bad_field():
    # The worked example in _cell_consistency's docstring and in docs/qa_methods.md.  off_dra and
    # off_dde are weighted medians taken SEPARATELY, so four cells at (+50,0), (-50,0), (0,+50),
    # (0,-50) mas cancel to a field offset of 0 while every cell sits 50 mas from it.  The
    # magnitude gate alone would pass this field; the adjacency test is what fails it.
    cells = _grid_cells({(0, 0): (50.0, 0.0, 40000), (1, 0): (-50.0, 0.0, 40000),
                         (0, 1): (0.0, 50.0, 40000), (1, 1): (0.0, -50.0, 40000)})
    cc = D._cell_consistency(cells, [])
    assert cc["off_med"] == 0.0                     # the field offset reads clean...
    assert all(abs(c["off"] - 50.0) < 1e-9 for c in cells)   # ...while no cell is
    assert cc["n_deviating"] == 4 and cc["n_confirmed"] == 4
    assert cc["consistent"] is False                # adjacency catches what the magnitude misses


def test_cell_consistency_low_coverage_not_consistent():
    # most sources sit in DROPPED (no-peak) cells -> not adequately sampled to pass
    cells = _grid_cells({(0, 0): (9.0, 0.0, 500), (0, 1): (9.0, 0.0, 500),
                         (1, 0): (9.0, 0.0, 500), (1, 1): (9.0, 0.0, 500)})
    dropped = [dict(i=2, j=2, ra=0.0, dec=0.0, n=100000)]
    cc = D._cell_consistency(cells, dropped)
    assert cc["coverage"] < 0.5 and not cc["consistent"]


def test_cell_consistency_reports_a_spread_and_no_significance():
    # Stage 4 reports how much the cells disagree.  It reports no significance; the next test is
    # the measurement of why.
    cells = _grid_cells({(i, j): (9.0 + i, 3.0 + j, 40000) for i in range(4) for j in range(4)})
    cc = D._cell_consistency(cells, [])
    assert cc["spread"] is not None and cc["spread"] > 0
    assert "signif" not in cc and "se" not in cc


@pytest.mark.parametrize("scale", [5.0, 500.0])
def test_retired_significance_sits_at_one_when_the_true_offset_is_zero(scale):
    # Why stage 4 quotes no sigma.  Draw 16 cells from a zero-mean scatter of `scale` mas, so the
    # true offset is zero by construction, and evaluate the retired statistic
    # off_med / (spread / sqrt(n_cells)) on them.  It lands near 1 at both scales: it is a length
    # divided by the sampling error of the two medians that length is built from, so it has a floor
    # near 1 and reads there whatever the offset is.  That is how a 780 mas offset was posted as
    # "1 sigma from zero".
    rng = np.random.default_rng(7)
    vals = []
    for _ in range(200):
        d = rng.normal(0.0, scale, (16, 2))
        cc = D._cell_consistency(
            _grid_cells({(i % 4, i // 4): (d[i, 0], d[i, 1], 40000) for i in range(16)}), [])
        vals.append(cc["off_med"] / (cc["spread"] / np.sqrt(cc["n_cells"])))
    assert 0.8 < float(np.median(vals)) < 1.6


def test_stage4_caption_reports_the_spread_and_quotes_no_sigma():
    cap = D.caption_for(4, dict(stage=4, offset_med_mas=780.0, n_cells=9,
                                offset_scatter_mas=640.0, bulk_source="histogram"))
    assert "cells scatter by 640 mas" in cap
    assert "σ" not in cap and "sigma" not in cap.lower()


def test_cell_offsets_recovers_uniform_shift():
    # synthetic field + reference shifted by a KNOWN 100 mas in RA; every cell must recover it
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    rng = np.random.RandomState(0)
    ra = 266.40 + rng.uniform(0, 0.02, 1200)
    dec = -28.90 + rng.uniform(0, 0.02, 1200)
    jsc = SkyCoord(ra * u.deg, dec * u.deg)
    cosd = np.cos(np.radians(-28.9))
    ref = SkyCoord((ra + 100.0 / 3.6e6 / cosd) * u.deg, dec * u.deg)   # ref is +100 mas E of jsc
    cells, dropped, grid = D._cell_offsets(jsc, ref, ncell=2, min_per_cell=50)
    assert grid == 2 and len(cells) >= 3
    dra = np.array([c["dra"] for c in cells])
    assert np.all(np.abs(dra - 100.0) < 15)      # each cell recovers ~+100 mas


def _uniform_shift(n, shift_mas, seed, span=0.02):
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    rng = np.random.RandomState(seed)
    ra = 266.40 + rng.uniform(0, span, n); dec = -28.90 + rng.uniform(0, span, n)
    cosd = np.cos(np.radians(-28.9))
    jsc = SkyCoord(ra * u.deg, dec * u.deg)
    ref = SkyCoord((ra + shift_mas / 3.6e6 / cosd) * u.deg, dec * u.deg)
    return jsc, ref


def test_cell_offsets_adaptive_grid_fired_2x2():
    # issue #13: a field too sparse to fill a 4x4 cell (min 300 stars) must FALL BACK to the 2x2
    # grid -- and _cell_offsets must report grid_used == 2, not silently succeed via the 1x1 rung.
    jsc, ref = _uniform_shift(700, 90.0, seed=7)   # ~44 per 4x4 cell (<300); ~175 per 2x2 cell
    assert D._cell_grid(jsc, ref, 4, 300)[0] == []            # fine grid measures nothing
    cells, _dropped, grid = D._cell_offsets(jsc, ref)
    assert grid == 2, "the 2x2 rung must fire here; deleting it must break this test"
    off = float(np.hypot(np.median([c["dra"] for c in cells]),
                         np.median([c["dde"] for c in cells])))
    assert abs(off - 90.0) < 20


def test_cell_offsets_adaptive_grid_fired_1x1():
    # even sparser: neither 4x4 nor 2x2 fills a cell, so the WHOLE-FIELD (1x1) fallback must fire.
    jsc, ref = _uniform_shift(200, 90.0, seed=11)  # ~50 per 2x2 cell (<150) -> only 1x1 works
    assert D._cell_grid(jsc, ref, 2, 150)[0] == []
    cells, _dropped, grid = D._cell_offsets(jsc, ref)
    assert grid == 1, "the 1x1 whole-field fallback must fire; deleting it must break this test"
    assert len(cells) == 1


def _one_cell_field(n=400):
    """A single dense patch: one 1x1 cell, so _cell_grid's floors are the only thing deciding."""
    return _uniform_shift(n, 0.0, seed=3)


def _fake_xcorr(npairs, npeak, peak_ratio=8.0):
    return lambda a, b, **kw: dict(dra=5.0, ddec=0.0, off=5.0, npairs=npairs, npeak=npeak,
                                   peak_ratio=peak_ratio)


def test_cell_grid_pair_floor_is_independent_of_the_star_floor(monkeypatch):
    # issue #65: min_pairs is a PAIR count, chosen for pairs.  With the star floor held fixed and
    # generous, moving the pair floor alone must decide the cell -- which is only possible because
    # the pair floor is no longer `min_pairs = min_src`.
    jsc, ref = _one_cell_field()
    monkeypatch.setattr(D.aa, "xcorr", _fake_xcorr(npairs=120, npeak=60))
    kept, _ = D._cell_grid(jsc, ref, 1, 50, min_pairs=100, min_peak_pairs=10)
    dropped_cells, dropped = D._cell_grid(jsc, ref, 1, 50, min_pairs=5000, min_peak_pairs=10)
    assert len(kept) == 1 and dropped_cells == []
    assert dropped[0]["reason"] == "too few pairs in the search radius"
    # ...and the star floor still decides on its own quantity, with the pair floor generous
    _c, star_dropped = D._cell_grid(jsc, ref, 1, 100000, min_pairs=1, min_peak_pairs=1)
    assert star_dropped == [] or star_dropped[0]["reason"] == "too few reference stars"


def test_cell_grid_drops_a_cell_with_plenty_of_stars_but_a_peak_no_stars_support(monkeypatch):
    # the case the star floor cannot see: 40 000 chance pairs inside the search radius (so the
    # total-pair floor is satisfied many times over) while the reported offset rests on 5 stars.
    jsc, ref = _one_cell_field()
    monkeypatch.setattr(D.aa, "xcorr", _fake_xcorr(npairs=40000, npeak=5))
    cells, dropped = D._cell_grid(jsc, ref, 1, 50)
    assert cells == [] and dropped[0]["reason"] == "too few pairs supporting the peak"
    monkeypatch.setattr(D.aa, "xcorr", _fake_xcorr(npairs=40000, npeak=50))
    cells, _ = D._cell_grid(jsc, ref, 1, 50)
    assert len(cells) == 1 and cells[0]["npeak"] == 50


def test_cell_offsets_passes_explicit_pair_floors_at_every_rung(monkeypatch):
    # the floors must arrive at _cell_grid as the deliberate constants, at every adaptive rung, and
    # must NOT track the rung's star floor (300 / 150 / 100).
    seen = []

    def _spy(jsc, ref_sc, ncell, min_src, pr_floor=None, min_pairs=None, min_peak_pairs=None):
        seen.append(dict(ncell=ncell, min_src=min_src, min_pairs=min_pairs,
                         min_peak_pairs=min_peak_pairs))
        return [], []

    monkeypatch.setattr(D, "_cell_grid", _spy)
    jsc, ref = _one_cell_field()
    D._cell_offsets(jsc, ref)
    assert [s["min_src"] for s in seen] == [300, 150, 100]        # star floors still step down
    assert {s["min_pairs"] for s in seen} == {D._CELL_MIN_PAIRS}
    assert {s["min_peak_pairs"] for s in seen} == {D._CELL_MIN_PEAK_PAIRS}
    assert all(s["min_pairs"] != s["min_src"] or s["min_peak_pairs"] != s["min_src"] for s in seen)


def test_xcorr_reports_the_pairs_that_support_the_peak():
    # npeak is the PEAK BIN's occupancy, so a synthetic field of N stars shifted rigidly reports
    # ~N supporting pairs while npairs counts every pair in the 2.5" radius.
    jsc, ref = _uniform_shift(300, 100.0, seed=5, span=0.004)
    xc = D.aa.xcorr(jsc, ref)
    assert xc["npeak"] <= xc["npairs"]
    assert xc["npeak"] >= 200                 # the 300 true pairs, minus bin-edge splitting
    assert xc["npairs"] > 2 * xc["npeak"]     # the rest of the radius is chance pairs


def test_offset_failure_reason_reports_counts_not_cause(monkeypatch, tmp_path):
    # the reason string must report measured counts (JWST / reference / pairs) and NOT assert a
    # single cause; and a confident peak must never print "peak_ratio >=4 < 4".
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    monkeypatch.setattr(D, "_jwst_sources", lambda o, f: (None, None, None))
    ra = 266.40 + np.linspace(0, 0.02, 300); dec = -28.90 + np.linspace(0, 0.02, 300)
    jsc = SkyCoord(ra * u.deg, dec * u.deg); ref = SkyCoord(ra * u.deg, dec * u.deg)
    hi = D._offset_failure_reason(_obs(), "F210M", jsc, ref,
                                  {"peak_ratio": 16.0, "npairs": 250, "npeak": 40})
    # the pair count must be named for what it is -- pairs inside the search radius, of which only
    # some support the peak (issue #65); "matched pairs at the best peak" described neither.
    assert "300 JWST sources" in hi and "250 pairs within" in hi and "40 land in the peak bin" in hi
    assert "16.0" in hi and "< 4" not in hi and "≥" in hi
    lo = D._offset_failure_reason(_obs(), "F210M", jsc, ref, {"peak_ratio": 1.2, "npairs": 5})
    assert "300 JWST sources" in lo and "not determined" in lo


def _stage4_seams(monkeypatch, cells, dropped, grid_used):
    """Monkeypatch stage-4's I/O seams so stage4_offsets runs on a synthetic cell result, exercising
    the REAL gate (not a re-implementation of it)."""
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    ra = 266.40 + np.linspace(0, 0.02, 300); dec = -28.90 + np.linspace(0, 0.02, 300)
    jsc = SkyCoord(ra * u.deg, dec * u.deg)
    monkeypatch.setattr(D, "_mosaic_path", lambda o, f: "/dev/null/x_i2d.fits")
    monkeypatch.setattr(D, "_refcat_path", lambda o: "/dev/null/ref")
    monkeypatch.setattr(D, "_obs_epoch", lambda o, p: 2024.0)
    monkeypatch.setattr(D.aa, "load_reference", lambda ref, ep: (jsc, None))
    # stage 4 now sources positions from the MAST L3 catalogue; feed the synthetic jsc there.
    monkeypatch.setattr(D, "_mast_catalog_positions", lambda o, f: (jsc, None))
    monkeypatch.setattr(D, "_jwst_positions", lambda o, sw: (jsc, "release-m8"))
    monkeypatch.setattr(D, "_module_positions", lambda o, sw: (None, None, None))
    monkeypatch.setattr(D, "_cell_offsets", lambda j, r: (cells, dropped, grid_used))
    monkeypatch.setattr(D.aa, "same_star_tie", lambda j, r: None)   # -> off_med = cell_off_med
    monkeypatch.setattr(D, "_crossmatch_offset", lambda j, r, restrict_footprint=False: None)
    monkeypatch.setattr(D, "_save", lambda fig, name: name)


def test_stage4_whole_field_passes_but_flags_spatial_unassessed(monkeypatch):
    # a genuine 1x1 whole-field tie (grid_used=1) with a SMALL offset PASSES: no per-cell spatial
    # check is possible, so it is bypassed -- but spatial_assessed is False so make_issues will not
    # auto-tick 'frame_ok'.  Reverting `spatial_ok = True if whole_field else cc["consistent"]`
    # fails this (one cell is never 'consistent').
    cells = [dict(i=0, j=0, ra=266.41, dec=-28.89, dra=5.0, dde=0.0, off=5.0, peak_ratio=20.0,
                  n=5000, n_ref=5000, npairs=5000)]
    _stage4_seams(monkeypatch, cells, [], grid_used=1)
    _png, m = D.stage4_offsets(_obs(), "F210M")
    assert m["passed"] is True and m["spatial_assessed"] is False and m["grid_used"] == 1


def test_stage4_low_coverage_grid_does_not_pass(monkeypatch):
    # A large field with only 3 of 16 cells measurable (grid_used=4) is 19% coverage -- NOT a small
    # field.  It must FAIL on coverage; keying on cell count alone (the reverted heuristic) would
    # wrongly pass it (#13 review).
    cells = [dict(i=i, j=0, ra=266.41 + 0.001 * i, dec=-28.89, dra=5.0, dde=0.0, off=5.0,
                  peak_ratio=8.0, n=400, n_ref=400, npairs=400) for i in range(3)]
    # the 13 grid positions of a 4x4 that are NOT the three measured (0..2, 0)
    dropped = [dict(i=i, j=j, ra=266.41, dec=-28.89, n=400, n_ref=100,
                    reason="too few reference stars")
               for i in range(4) for j in range(4) if not (i < 3 and j == 0)]
    _stage4_seams(monkeypatch, cells, dropped, grid_used=4)
    _png, m = D.stage4_offsets(_obs(), "F210M")
    assert m["cell_coverage"] < 0.5 and m["passed"] is False


def test_mosaic_path_single_module_nrcb(tmp_path, monkeypatch):
    # issue #13: a single-module (NRCB-only) obs names its mosaic '-nrcb', not '-merged'.
    # _mosaic_path must find it (else stage 1 blanks and stage 7 shows "no pipeline mosaic").
    monkeypatch.setattr(D, "BASE", str(tmp_path))
    d = tmp_path / "sickle" / "F210M" / "pipeline"; d.mkdir(parents=True)
    (d / "jw03958-o007_t001_nircam_clear-f210m-nrcb_i2d.fits").write_text("")
    # a residual sidecar with the same module tag must NOT be picked
    (d / "jw03958-o007_t001_nircam_clear-f210m-nrcb_m2_daophot_basic_mergedcat_residual_i2d.fits").write_text("")
    o = Observation(program="3958", obs="007", target="Sickle", release_field="sickle",
                    instrument="NIRCam", filters=["F210M"], visits=[], epoch="", notes="")
    hit = D._mosaic_path(o, "F210M")
    assert hit is not None and hit.endswith("clear-f210m-nrcb_i2d.fits")
    assert D._mosaic_module(hit) == "NRCB"
    # 'merged', when present, is PREFERRED over the single-module mosaic
    (d / "jw03958-o007_t001_nircam_clear-f210m-merged_i2d.fits").write_text("")
    assert D._mosaic_path(o, "F210M").endswith("clear-f210m-merged_i2d.fits")
    assert D._mosaic_module(D._mosaic_path(o, "F210M")) == ""


def _three_of_four(dra_by_cell=None, n_by_cell=None):
    """3 of the 4 cells of a 2x2 grid, plus the 1 dropped cell (coverage 0.75)."""
    ij3 = [(0, 0), (0, 1), (1, 0)]
    dra_by_cell = dra_by_cell or {}
    n_by_cell = n_by_cell or {}
    cells = [dict(i=i, j=j, ra=266.41 + 0.001 * i, dec=-28.89 + 0.001 * j,
                  dra=dra_by_cell.get((i, j), 5.0), dde=0.0,
                  off=abs(dra_by_cell.get((i, j), 5.0)), peak_ratio=8.0,
                  n=n_by_cell.get((i, j), 400), n_ref=400, npairs=400) for i, j in ij3]
    dropped = [dict(i=1, j=1, ra=266.41, dec=-28.89, n=400, n_ref=100,
                    reason="too few reference stars")]
    return cells, dropped


def test_stage4_2x2_three_of_four_agreeing_passes_but_is_not_auto_ticked(monkeypatch):
    # issue #66: a 2x2 grid with 3 of 4 cells measured (coverage 0.75, ABOVE the 0.5 floor) whose
    # cells AGREE must not be failed for the thinness of its own grid.  It passes the magnitude and
    # coverage gates, and `spatial_assessed` stays False so make_issues leaves 'frame_ok' unticked
    # -- the same treatment the 1x1 whole-field fallback gets.  Restoring `len(cells) >= 4` inside
    # `consistent` fails this.
    cells, dropped = _three_of_four()
    _stage4_seams(monkeypatch, cells, dropped, grid_used=2)
    _png, m = D.stage4_offsets(_obs(), "F210M")
    assert m["grid_used"] == 2 and 0.5 <= m["cell_coverage"] < 1.0
    assert m["passed"] is True and m["spatial_assessed"] is False
    assert m["cells_spatial_conclusive"] is False


def test_stage4_three_of_four_with_adjacent_deviating_cells_still_fails(monkeypatch):
    # ...and the adjacency test keeps its teeth on the same thin grid: two ORTHOGONALLY ADJACENT
    # cells sitting 120 mas from the field value confirm each other, so the field FAILS on evidence
    # rather than on cell count.  (0,0)-(0,1) are adjacent in j, and the third cell carries most of
    # the sources, so the source-weighted consensus is the quiet one and the pair is the minority
    # (13.8% of the sources -- above the 2% _CELL_BAD_FRAC).
    cells, dropped = _three_of_four({(0, 0): 120.0, (0, 1): 120.0}, {(1, 0): 5000})
    _stage4_seams(monkeypatch, cells, dropped, grid_used=2)
    _png, m = D.stage4_offsets(_obs(), "F210M")
    assert m["n_cells_confirmed"] >= 2 and m["cells_consistent"] is False
    assert m["passed"] is False


def test_stage4_verdict_is_monotonic_in_sampling(monkeypatch):
    # issue #66, the inversion itself: the SAME clean field measured with a 1x1 whole-field cell and
    # with a 2x2 grid that resolved 3 of 4 cells must not disagree.  Before the fix the better-sampled
    # measurement scored WORSE (2x2 3-of-4 -> passed False; 1x1 -> passed True).
    one = [dict(i=0, j=0, ra=266.41, dec=-28.89, dra=5.0, dde=0.0, off=5.0, peak_ratio=20.0,
                n=1200, n_ref=1200, npairs=1200)]
    _stage4_seams(monkeypatch, one, [], grid_used=1)
    _png, m1 = D.stage4_offsets(_obs(), "F210M")
    cells, dropped = _three_of_four()
    _stage4_seams(monkeypatch, cells, dropped, grid_used=2)
    _png, m2 = D.stage4_offsets(_obs(), "F210M")
    assert m1["passed"] is True and m2["passed"] is True
    assert m1["spatial_assessed"] is False and m2["spatial_assessed"] is False


def test_stage4_caption_states_when_spatial_check_skipped():
    # ask 2 consumer: the caption must claim the spatial-consistency check only when it ran.
    assessed = D.caption_for(4, dict(stage=4, offset_med_mas=5.0, n_cells=4,
                                     offset_scatter_mas=2.0, spatial_assessed=True,
                                     bulk_source="histogram"))
    assert "cells that agree with each other" in assessed
    whole = D.caption_for(4, dict(stage=4, offset_med_mas=5.0, n_cells=1,
                                  offset_scatter_mas=None, spatial_assessed=False,
                                  bulk_source="histogram"))
    assert "WHOLE-FIELD" in whole and "did not run" in whole
    assert "cells that agree with each other" not in whole


def test_make_issues_frame_ok_untficked_when_spatial_unassessed(monkeypatch):
    # ask 2 consumer: make_issues must NOT tick the astrometry box on a whole-field tie.
    from data_qa import make_issues as MI
    monkeypatch.setattr(MI, "_guidestar_json", lambda: {})
    o = Observation(program="3958", obs="007", target="Sickle", release_field="sickle",
                    instrument="NIRCam", filters=["F210M"], visits=[], epoch="", notes="")

    def _M(spatial):
        return {"stage1": {"passed": True}, "stage2": {"passed": True}, "stage3": {"passed": True},
                "stage4": {"passed": True, "spatial_assessed": spatial}, "stage5": {}}

    monkeypatch.setattr(MI, "_qa_metrics", lambda oo: _M(False))
    line = [l for l in MI.render_body(o).splitlines() if "Astrometry" in l][0]
    assert "[ ]" in line and "[x]" not in line
    monkeypatch.setattr(MI, "_qa_metrics", lambda oo: _M(True))
    line2 = [l for l in MI.render_body(o).splitlines() if "Astrometry" in l][0]
    assert "[x]" in line2


def test_make_issues_product_existence_checkboxes(monkeypatch):
    # jicama / JWST1PASS / peppar presence boxes are auto-set from the diagnostics: a stage whose
    # product is absent reports available=False (no red flag), so "not available" leaves the box
    # unticked.
    from data_qa import make_issues as MI
    monkeypatch.setattr(MI, "_guidestar_json", lambda: {})
    o = Observation(program="2045", obs="001", target="Arches", release_field="arches",
                    instrument="NIRCam", filters=["F212N"], visits=[], epoch="", notes="")
    monkeypatch.setattr(MI, "_qa_metrics", lambda oo: {
        "stage3": {"passed": True},                       # jicama present (Vega calib ran)
        "stage6": {"peppar_kind": "frame-to-frame σ"},    # peppar present
        "stage10": {"stage": 10, "available": False},     # JWST1PASS absent
        "stage11": {"stage": 11, "available": True}})
    lines = {l.split("**Products**: ")[1].split(" present")[0]: l
             for l in MI.render_body(o).splitlines() if "**Products**:" in l}
    assert "[x]" in lines["jicama (merged/release catalogue)"]
    assert "[ ]" in lines["JWST1PASS products"]           # stage 10 unavailable -> absent
    assert "[x]" in lines["peppar products"]
    # nothing present -> all three unticked
    monkeypatch.setattr(MI, "_qa_metrics", lambda oo: {"stage10": {"available": False}})
    lines2 = [l for l in MI.render_body(o).splitlines() if "**Products**:" in l]
    assert len(lines2) == 3 and all("[ ]" in l for l in lines2)


def test_mosaic_path_lone_module_incomplete_when_sibling_filter_two_module(tmp_path, monkeypatch):
    # issue #13 re-review: the two-module guard must be OBSERVATION-scoped.  cloudef jw02092-o002
    # F360M has only NRCA, but sibling filters have merged mosaics -> the obs is two-module, so the
    # lone F360M half must read incomplete (None), while a genuine single-module obs (sickle, all
    # NRCB, no merged/NRCA anywhere) still returns its nrcb mosaic.
    monkeypatch.setattr(D, "BASE", str(tmp_path))
    d = tmp_path / "cloudef" / "F210M" / "pipeline"; d.mkdir(parents=True)
    (d / "jw02092-o002_t001_nircam_clear-f210m-merged_i2d.fits").write_text("")   # sibling is complete
    d2 = tmp_path / "cloudef" / "F360M" / "pipeline"; d2.mkdir(parents=True)
    (d2 / "jw02092-o002_t001_nircam_clear-f360m-nrca_i2d.fits").write_text("")     # lone half
    o = Observation(program="2092", obs="002", target="Cloud E/F", release_field="cloudef",
                    instrument="NIRCam", filters=["F210M", "F360M"], visits=[], epoch="", notes="")
    assert D._mosaic_path(o, "F360M") is None                    # incomplete: obs is two-module
    assert D._mosaic_path(o, "F210M").endswith("f210m-merged_i2d.fits")
    # a genuine single-module obs (all NRCB, no merged, no NRCA) still returns its mosaic
    s = tmp_path / "sickle" / "F210M" / "pipeline"; s.mkdir(parents=True)
    (s / "jw03958-o007_t001_nircam_clear-f210m-nrcb_i2d.fits").write_text("")
    so = Observation(program="3958", obs="007", target="Sickle", release_field="sickle",
                     instrument="NIRCam", filters=["F210M"], visits=[], epoch="", notes="")
    assert D._mosaic_path(so, "F210M").endswith("f210m-nrcb_i2d.fits")


def test_mosaic_path_two_module_no_merged_returns_none(tmp_path, monkeypatch):
    # issue #13 review: a two-module obs that simply has not been merged (both -nrca and -nrcb over
    # DIFFERENT sky, e.g. cloudc o002 F212N) must NOT return one half as 'the mosaic' -- that would
    # flip 'delivered' green while NRCA/the merge is missing.  Return None (incomplete).
    monkeypatch.setattr(D, "BASE", str(tmp_path))
    d = tmp_path / "cloudc" / "F212N" / "pipeline"; d.mkdir(parents=True)
    (d / "jw02221-o002_t001_nircam_clear-f212n-nrca_i2d.fits").write_text("")
    (d / "jw02221-o002_t001_nircam_clear-f212n-nrcb_i2d.fits").write_text("")
    o = Observation(program="2221", obs="002", target="Cloud C", release_field="cloudc",
                    instrument="NIRCam", filters=["F212N"], visits=[], epoch="", notes="")
    assert D._mosaic_path(o, "F212N") is None      # both modules, no merged -> incomplete
    # once a merged exists, it is returned
    (d / "jw02221-o002_t001_nircam_clear-f212n-merged_i2d.fits").write_text("")
    assert D._mosaic_path(o, "F212N").endswith("clear-f212n-merged_i2d.fits")


def test_ab_overlap_returns_matched_positions():
    # _ab_overlap must return per-star matched sky positions (for the A↔B footprint map), aligned
    # in length with the residual arrays
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    rng = np.random.RandomState(3)
    ra = 266.40 + rng.uniform(0, 0.02, 1000); dec = -28.90 + rng.uniform(0, 0.02, 1000)
    b = SkyCoord(ra * u.deg, dec * u.deg)
    cosd = np.cos(np.radians(-28.9))
    a = SkyCoord((ra + 8.0 / 3.6e6 / cosd) * u.deg, dec * u.deg)   # A is 8 mas E of B
    ov = D._ab_overlap(a, b)
    assert ov is not None
    assert len(ov["ra_arr"]) == ov["n"] == len(ov["dra_arr"]) == len(ov["dec_arr"])
    assert np.all(np.isfinite(ov["ra_arr"])) and np.all(np.isfinite(ov["dec_arr"]))


def test_ab_overlap_rms_is_twice_the_per_axis_single_module_error():
    # The stage-5 scatter is NOT on the same footing as a stage-6 curve, and the docstring says by
    # how much.  Two factors: hypot combines the axes (stage 6 divides by sqrt(2) to stay
    # per-axis), and each residual is a difference A - B of two independent measurements of one
    # star.  Inject a known per-axis error into BOTH modules and check the returned rms lands at 2x
    # it, so the docstring's factor fails here if the estimator changes.
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    rng = np.random.RandomState(11)
    n, sig_mas = 4000, 6.0
    ra = 266.40 + rng.uniform(0, 0.02, n); dec = -28.90 + rng.uniform(0, 0.02, n)
    cosd = np.cos(np.radians(-28.9))

    def jitter(r, d):
        return SkyCoord((r + rng.normal(0, sig_mas, n) / 3.6e6 / cosd) * u.deg,
                        (d + rng.normal(0, sig_mas, n) / 3.6e6) * u.deg)
    ov = D._ab_overlap(jitter(ra, dec), jitter(ra, dec))
    assert ov is not None
    assert 1.8 * sig_mas < ov["rms"] < 2.2 * sig_mas


def test_binned_rms_reads_one_times_the_per_axis_sigma():
    # The other half of the 2x claim: it holds against ALL THREE stage-6 curves only because each
    # is per-axis.  rms(offset) builds hypot(dra', dde')/sqrt(2) and hands it to _binned_rms, whose
    # estimator is sqrt(mean(r**2)) -- so it reads 1.00x a per-axis sigma, the same as sig_pos.
    # (The MEDIAN of that same residual is 0.83x, which is the factor the docstring used to quote.)
    # Pinning the estimator here keeps the docs' factor from drifting if _binned_rms changes.
    rng = np.random.RandomState(17)
    n, sig_mas = 200000, 10.0
    d = rng.normal(0, sig_mas, (n, 2))
    resid = np.hypot(d[:, 0] - np.median(d[:, 0]), d[:, 1] - np.median(d[:, 1])) / np.sqrt(2.0)
    mag = rng.uniform(14.0, 18.0, n)                       # spread over several magnitude bins
    rms, ctr = D._binned_rms(mag, resid)
    assert rms is not None and len(ctr) >= 3
    assert 0.97 * sig_mas < float(np.median(rms)) < 1.03 * sig_mas


def test_ab_overlap_one_to_one_no_pair_inflation():
    # Guards the count fix: several A sources clustered inside 80 mas of ONE B source must collapse
    # to a SINGLE match (one-to-one), not one pair each -- the search_around_sky many-to-many ball
    # match counted PAIRS and inflated the star count ~10x.
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    rng = np.random.RandomState(3)
    ra = 266.40 + rng.uniform(0, 0.02, 1000); dec = -28.90 + rng.uniform(0, 0.02, 1000)
    b = SkyCoord(ra * u.deg, dec * u.deg)
    cosd = np.cos(np.radians(-28.9))
    a_ra = ra + 8.0 / 3.6e6 / cosd                                    # 1:1 base, A 8 mas E of B
    a_base = SkyCoord(a_ra * u.deg, dec * u.deg)
    ov_base = D._ab_overlap(a_base, b)
    assert ov_base is not None
    # add 4 extra A sources all within 80 mas of b[0]; the ball match would emit 4 more pairs on b[0]
    ex_ra = list(a_ra) + [ra[0] + off / 3.6e6 / cosd for off in (10.0, 20.0, 30.0, 40.0)]
    ex_dec = list(dec) + [dec[0]] * 4
    ov_plus = D._ab_overlap(SkyCoord(np.asarray(ex_ra) * u.deg, np.asarray(ex_dec) * u.deg), b)
    assert ov_plus is not None
    # the clustered extras add ZERO: b[0] is already matched, so one-to-one keeps the count the same
    assert ov_plus["n"] == ov_base["n"]
    # one-to-one: every matched B position is distinct (no B counted twice), and n cannot exceed |B|
    for ov in (ov_base, ov_plus):
        seen = set(zip(np.round(ov["ra_arr"], 10), np.round(ov["dec_arr"], 10)))
        assert len(seen) == ov["n"] <= len(b)


def test_available_filters_only_present(tmp_path, monkeypatch):
    monkeypatch.setattr(D, "BASE", str(tmp_path))
    d = tmp_path / "sgra" / "F212N" / "pipeline"; d.mkdir(parents=True)
    _touch(d, "jw01939-o001_t001_nircam_clear-f212n-merged_i2d.fits")   # only F212N has a mosaic
    o = Observation(program="1939", obs="001", target="Sgr A*", release_field="sgra",
                    instrument="NIRCam", filters=["F212N", "F444W"], visits=[], epoch="", notes="")
    assert D._available_filters(o) == ["F212N"]                          # F444W (no data) dropped


def test_peppar_precision_returns_formal_and_combo_framestd(tmp_path, monkeypatch):
    from astropy.table import Table
    monkeypatch.setitem(D._PEPPAR_ROOTS, "brick", str(tmp_path))
    pdir = tmp_path / "brick" / "peppar" / "F212N"
    det = pdir / "NRCA1"; det.mkdir(parents=True)
    # ONE per-frame cat -> 'formal' only (frame-to-frame needs >=3 frames, so no computed frame_std)
    Table({"m": np.linspace(-5.0, 5.0, 120), "x_fit": np.arange(120.0), "y_fit": np.arange(120.0),
           "x_err": np.full(120, 0.1), "y_err": np.full(120, 0.1)}
          ).write(str(det / "jw02221001001_00001_nrca1_cal_brick_iter1_cat.fits"), overwrite=True)
    o = Observation(program="2221", obs="001", target="Brick", release_field="brick",
                    instrument="NIRCam", filters=["F212N"], visits=[], epoch="", notes="")
    pp = D._peppar_precision(o, "F212N")
    assert set(pp) == {"formal"}                               # per-frame only, no frame_std yet
    assert abs(float(np.median(pp["formal"][1])) - 3.1) < 0.5  # hypot(.1,.1)/√2 * 31 mas = 3.1
    # a combined starlist supplies frame_std directly.  x_wcs_std is ARCSEC (tangent-plane), so
    # 0.004" -> ~4 mas pins the factor at 1e3 (the old deg->mas 3.6e6 would read it as 14416 mas).
    Table({"m": np.linspace(-5.0, 5.0, 120), "x_wcs_std": np.full(120, 0.004),
           "y_wcs_std": np.full(120, 0.004)}).write(
        str(pdir / "combo_starlist_F212N_NRCA1.fits"), overwrite=True)
    pp2 = D._peppar_precision(o, "F212N")
    assert "frame_std" in pp2 and "formal" in pp2
    assert abs(float(np.median(pp2["frame_std"][1])) - 4.0) < 0.2


def test_peppar_frame_std_computed_from_per_frame_catalogues(tmp_path, monkeypatch):
    # No combo starlist: frame-to-frame std is COMPUTED by SKY-matching per-frame detections across
    # the (dithered, mosaicked) exposures via each frame's cal WCS.  Build 5 exposures of the SAME
    # 200 stars at fixed sky positions with a known 3 mas per-axis positional jitter.
    from astropy.table import Table
    from astropy.io import fits
    from astropy.wcs import WCS
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    monkeypatch.setitem(D._PEPPAR_ROOTS, "brick", str(tmp_path))
    det = tmp_path / "brick" / "peppar" / "F212N" / "NRCA1"; det.mkdir(parents=True)
    pipe = tmp_path / "brick" / "F212N" / "pipeline"; pipe.mkdir(parents=True)

    def _wcs(cr):
        w = WCS(naxis=2); w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        w.wcs.crval = list(cr); w.wcs.crpix = [1024.5, 1024.5]
        w.wcs.cd = (0.031 / 3600.0) * np.array([[-1.0, 0.0], [0.0, 1.0]])   # 31 mas/px
        return w
    rng = np.random.default_rng(0)
    n = 200
    ra0 = 266.5 + rng.uniform(0, 0.01, n); dec0 = -28.9 + rng.uniform(0, 0.01, n)  # 36" spread
    mags = np.linspace(-9.0, -4.0, n)
    jit_deg = 3.0 / 3.6e6                                       # 3 mas per axis
    for k in range(5):
        cr = (266.5, -28.9)                                    # same pointing (dither irrelevant here)
        w = _wcs(cr)
        ra = ra0 + rng.normal(0, jit_deg, n) / np.cos(np.radians(dec0))
        dec = dec0 + rng.normal(0, jit_deg, n)
        x, y = w.world_to_pixel(SkyCoord(ra * u.deg, dec * u.deg))
        Table({"m": mags, "x_fit": np.asarray(x, float), "y_fit": np.asarray(y, float),
               "x_err": np.full(n, 0.02), "y_err": np.full(n, 0.02)}).write(
            str(det / f"jw02221001001_02101_0000{k+1}_nrca1_cal_brick_iter1_cat.fits"), overwrite=True)
        hdu = fits.ImageHDU(np.zeros((4, 4), "float32"), header=w.to_header(), name="SCI")
        fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(
            str(pipe / f"jw02221001001_02101_0000{k+1}_nrca1_cal.fits"), overwrite=True)
    res = D._peppar_frame_std(str(tmp_path / "brick" / "peppar" / "F212N"), pixscale=31.0)
    assert res is not None
    m, prec = res
    # per-axis sky jitter ~3 mas -> hypot(3,3)/√2 = 3 mas
    assert abs(float(np.median(prec)) - 3.0) < 1.2


def test_peppar_frame_std_removes_per_frame_pointing(tmp_path, monkeypatch):
    # Each frame's cal WCS carries an independent pointing solution: inject a large per-frame BULK
    # offset (~8 mas/axis) on top of a small intrinsic per-star jitter (0.4 mas).  The registration
    # step must remove the bulk offset so the reported scatter is the intrinsic ~0.4 mas, not the
    # ~8 mas pointing jitter (issue #129).
    from astropy.table import Table
    from astropy.io import fits
    from astropy.wcs import WCS
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    monkeypatch.setitem(D._PEPPAR_ROOTS, "brick", str(tmp_path))
    det = tmp_path / "brick" / "peppar" / "F212N" / "NRCA1"; det.mkdir(parents=True)
    pipe = tmp_path / "brick" / "F212N" / "pipeline"; pipe.mkdir(parents=True)

    def _wcs(cr):
        w = WCS(naxis=2); w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        w.wcs.crval = list(cr); w.wcs.crpix = [1024.5, 1024.5]
        w.wcs.cd = (0.031 / 3600.0) * np.array([[-1.0, 0.0], [0.0, 1.0]])
        return w
    rng = np.random.default_rng(3)
    n = 200
    ra0 = 266.5 + rng.uniform(0, 0.01, n); dec0 = -28.9 + rng.uniform(0, 0.01, n)
    mags = np.linspace(-9.0, -4.0, n)
    intrinsic = 0.4 / 3.6e6                                     # 0.4 mas/axis real repeatability
    for k in range(6):
        # a per-frame pointing error, encoded in the frame's WCS CRVAL (~8 mas/axis)
        off_ra = rng.normal(0, 8.0 / 3.6e6); off_de = rng.normal(0, 8.0 / 3.6e6)
        w = _wcs((266.5 + off_ra / np.cos(np.radians(-28.9)), -28.9 + off_de))
        ra = ra0 + rng.normal(0, intrinsic, n) / np.cos(np.radians(dec0))
        dec = dec0 + rng.normal(0, intrinsic, n)
        # place stars at their TRUE sky positions through this frame's offset WCS -> x/y carry the
        # pointing error, exactly as a real cal frame does
        x, y = w.world_to_pixel(SkyCoord(ra * u.deg, dec * u.deg))
        Table({"m": mags, "x_fit": np.asarray(x, float), "y_fit": np.asarray(y, float),
               "x_err": np.full(n, 0.02), "y_err": np.full(n, 0.02)}).write(
            str(det / f"jw02221001001_02101_0000{k+1}_nrca1_cal_brick_iter1_cat.fits"), overwrite=True)
        hdu = fits.ImageHDU(np.zeros((4, 4), "float32"), header=w.to_header(), name="SCI")
        fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(
            str(pipe / f"jw02221001001_02101_0000{k+1}_nrca1_cal.fits"), overwrite=True)
    res = D._peppar_frame_std(str(tmp_path / "brick" / "peppar" / "F212N"), pixscale=31.0)
    assert res is not None
    _, prec = res
    # registered residual is the intrinsic ~0.4 mas, well below the injected ~8 mas pointing jitter
    assert float(np.median(prec)) < 1.5


def test_peppar_precision_none_without_products(tmp_path, monkeypatch):
    monkeypatch.setitem(D._PEPPAR_ROOTS, "brick", str(tmp_path))
    o = Observation(program="2221", obs="001", target="Brick", release_field="brick",
                    instrument="NIRCam", filters=["F212N"], visits=[], epoch="", notes="")
    assert D._peppar_precision(o, "F212N") is None


def test_peppar_dir_prefers_per_obs_then_flat(tmp_path, monkeypatch):
    """gc-treasury's disjoint tiles moved to peppar/o<obs>/<FILT>/ (per-obs); the flat
    peppar/<FILT>/ layout (gc2211, cloud E/F) is the fallback.  Per-obs must win when both exist,
    and a missing directory returns None."""
    monkeypatch.setitem(D._PEPPAR_ROOTS, "gc-treasury", str(tmp_path))
    o = Observation(program="10678", obs="132", target="GC Treasury", release_field="gc-treasury",
                    instrument="NIRCam", filters=["F212N"], visits=[], epoch="", notes="")
    root = tmp_path / "gc-treasury" / "peppar"
    assert D._peppar_dir(o, "F212N") is None                     # neither layout on disk
    (root / "F212N").mkdir(parents=True)
    assert D._peppar_dir(o, "F212N") == str(root / "F212N")      # flat fallback
    (root / "o132" / "F212N").mkdir(parents=True)
    assert D._peppar_dir(o, "F212N") == str(root / "o132" / "F212N")   # per-obs wins


def test_peppar_cal_for_cat_resolves_under_per_obs_layout(tmp_path):
    """A per-obs cat (peppar/o<obs>/<FILT>/<DET>/) must still resolve its cal in
    <field>/<FILT>/pipeline/ — the extra o<obs> level would otherwise send dirname-counting to the
    wrong field_dir/filt and blank stage 11 + stage 6's peppar half."""
    field = tmp_path / "gc-treasury"
    catdir = field / "peppar" / "o132" / "F212N" / "NRCA1"
    catdir.mkdir(parents=True)
    cat = catdir / "jw10678132001_02101_00001_nrca1_cal_gc-treasury_iter1_cat.fits"
    cat.write_text("")
    caldir = field / "F212N" / "pipeline"
    caldir.mkdir(parents=True)
    cal = caldir / "jw10678132001_02101_00001_nrca1_cal.fits"
    cal.write_text("")
    assert D._peppar_cal_for_cat(str(cat)) == str(cal)


def test_peppar_cal_for_cat_no_peppar_ancestor_returns_none():
    """A valid cat name whose path has NO 'peppar' ancestor must return None, not spin forever at
    the filesystem root (os.path.dirname('/') == '/')."""
    assert D._peppar_cal_for_cat(
        "/tmp/nope/jw10678132001_02101_00001_nrca1_cal_gc-treasury_iter1_cat.fits") is None


def test_apply_vega_zp_adds_when_present_else_unchanged():
    """Pins the depth-histogram calibration arithmetic: with a ZP the mag is shifted by exactly it;
    with None it is unchanged.  Guards against the +ZP being dropped while the 'Vega' label stays."""
    m = np.array([-7.0, -5.0, -3.0])
    np.testing.assert_allclose(D._apply_vega_zp(m, 26.0), m + 26.0)
    np.testing.assert_array_equal(D._apply_vega_zp(m, None), m)


def test_mast_depth_zp_nan_tolerant():
    """The stage-7 depth histogram lost its whole MAST series when the MAST→jicama zeropoint came
    out NaN (one NaN among matched jicama mags -> plain median NaN -> `zp or 0.0` keeps NaN, since
    NaN is truthy -> every MAST mag becomes NaN).  The zeropoint must survive a stray NaN and be
    None (not NaN) when it cannot be measured."""
    j = np.arange(40, dtype=float)            # jicama mags
    mm = j - 1.8                              # MAST mags, constant 1.8 offset
    assert D._mast_depth_zp(j, mm) == pytest.approx(1.8)
    j2 = j.copy(); j2[5] = np.nan            # one stray NaN must not poison the median
    assert D._mast_depth_zp(j2, mm) == pytest.approx(1.8)
    assert D._mast_depth_zp(np.full(40, np.nan), mm) is None   # unmeasurable -> None, never NaN
    assert D._mast_depth_zp(j[:10], mm[:10]) is None           # too few pairs -> None


def test_clip_to_core_drops_far_outliers():
    """Stage 8 blanked most of its map because a few wild-coordinate rows (~1° off the ~0.1° mosaic)
    stretched the bin grid.  The core clip keeps the dense field and drops the strays."""
    ra = np.concatenate([266.60 + 0.02 * np.random.default_rng(0).random(500), [265.8, 267.9]])
    dec = np.concatenate([-28.50 + 0.02 * np.random.default_rng(1).random(500), [-29.9, -27.1]])
    keep = D._clip_to_core(ra, dec)
    assert not keep[-1] and not keep[-2]      # the two far strays are dropped
    assert keep[:500].mean() > 0.9            # nearly all of the core survives
    # a clean field with NO strays is returned untouched, so edge stars are not trimmed
    clean_ra = 266.60 + 0.02 * np.random.default_rng(2).random(500)
    clean_dec = -28.50 + 0.02 * np.random.default_rng(3).random(500)
    assert D._clip_to_core(clean_ra, clean_dec).all()


def test_psfperts_scale_floor_cap_and_percentile():
    """Stage-10 perturbation panels were flat-white because a ~0.002 residual was drawn on a fixed
    ±0.1 scale.  The scale is the 99th percentile of |flux|, floored at 0.01 and capped at 0.1."""
    assert D._psfperts_scale(np.full(1000, 0.001)) == pytest.approx(0.01)   # tiny -> floor
    assert D._psfperts_scale(np.full(1000, 0.5)) == pytest.approx(0.1)      # huge -> Jay's cap
    mid = D._psfperts_scale(np.full(1000, 0.03))
    assert mid == pytest.approx(0.03)                                       # in-range -> percentile
    assert D._psfperts_scale(np.array([])) == pytest.approx(D._PSFPERTS_VLIM)


def test_stage7_caption_flags_real_misregistration_when_jicama_far_worse():
    """When jicama is materially farther from VIRAC than raw MAST, the caption must call it a real
    mis-registration (re-tie), not the neutral 'MAST as close as pipeline' — o132 is jicama 70 vs
    MAST 14."""
    cap = D.caption_for(7, dict(stage=7, sw="F212N", jicama_offset_med_mas=70.0,
                                mast_offset_med_mas=14.0, n_jicama_window=75000, n_mast_window=0))
    assert "farther from VIRAC than raw MAST" in cap and "re-tie" in cap
    # comparable offsets -> neutral wording, no false alarm
    cap2 = D.caption_for(7, dict(stage=7, sw="F212N", jicama_offset_med_mas=13.0,
                                 mast_offset_med_mas=11.0, n_jicama_window=75000, n_mast_window=5000))
    assert "farther from VIRAC" not in cap2
    # pin the margin: just below -> neutral, just above -> mis-registration
    mo = 14.0; margin = D._STAGE7_MISREG_MARGIN_MAS
    lo = D.caption_for(7, dict(stage=7, sw="F212N", jicama_offset_med_mas=mo + margin - 2,
                               mast_offset_med_mas=mo, n_jicama_window=1, n_mast_window=1))
    hi = D.caption_for(7, dict(stage=7, sw="F212N", jicama_offset_med_mas=mo + margin + 2,
                               mast_offset_med_mas=mo, n_jicama_window=1, n_mast_window=1))
    assert "farther from VIRAC" not in lo and "farther from VIRAC" in hi


def test_pick_filters_prefers_mosaic_backed_over_higher_ranked():
    # cloudef jw02092-o005: all four available, but only F162M/F360M have a reduced mosaic.
    # F210M/F480M rank HIGHER in the preference lists, so the naive pick chose the unreduced pair
    # and every mosaic-keyed stage blanked (issue #38).  prefer= must flip the pick to the reduced
    # filters WITHOUT changing behaviour when the top-ranked filter already has a mosaic.
    avail = ["F162M", "F210M", "F360M", "F480M"]
    # no prefer: unchanged legacy behaviour -> highest-ranked available (F210M / F480M)
    assert D.pick_filters(avail) == ("F210M", "F480M")
    # prefer only the reduced pair -> pick flips to them
    assert D.pick_filters(avail, prefer=["F162M", "F360M"]) == ("F162M", "F360M")
    # explicit args always win over prefer
    assert D.pick_filters(avail, sw="F210M", lw="F480M", prefer=["F162M", "F360M"]) == ("F210M", "F480M")
    # a channel with NO mosaic-backed filter falls back to any available (does not return None)
    assert D.pick_filters(avail, prefer=["F162M"]) == ("F162M", "F480M")


def test_filters_with_mosaic(tmp_path, monkeypatch):
    monkeypatch.setattr(D, "BASE", str(tmp_path))
    for filt in ("F162M", "F360M"):
        d = tmp_path / "cloudef" / filt / "pipeline"; d.mkdir(parents=True)
        _touch(d, f"jw02092-o005_t001_nircam_clear-{filt.lower()}-merged_i2d.fits")
    o = Observation(program="2092", obs="005", target="Cloud E/F", release_field="cloudef",
                    instrument="NIRCam", filters=["F162M", "F210M", "F360M", "F480M"],
                    visits=[], epoch="", notes="")
    # only the two reduced filters are mosaic-backed; F210M/F480M (no mosaic) are excluded
    assert D._filters_with_mosaic(o) == ["F162M", "F360M"]


def _write_i2d(path, ny=8, nx=16):
    from astropy.io import fits
    os.makedirs(os.path.dirname(path), exist_ok=True)
    hdu = fits.ImageHDU(data=np.ones((ny, nx), "float32"), name="SCI")
    fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(path, overwrite=True)


def _write_i2d_wcs(path, ra0, dec0, npix=100, scale_arcsec=0.03):
    from astropy.io import fits
    os.makedirs(os.path.dirname(path), exist_ok=True)
    hdr = fits.Header()
    hdr["NAXIS"] = 2; hdr["NAXIS1"] = npix; hdr["NAXIS2"] = npix
    hdr["CTYPE1"] = "RA---TAN"; hdr["CTYPE2"] = "DEC--TAN"
    hdr["CRPIX1"] = npix / 2; hdr["CRPIX2"] = npix / 2
    hdr["CRVAL1"] = ra0; hdr["CRVAL2"] = dec0
    hdr["CD1_1"] = -scale_arcsec / 3600.0; hdr["CD2_2"] = scale_arcsec / 3600.0
    hdr["CD1_2"] = 0.0; hdr["CD2_1"] = 0.0
    hdu = fits.ImageHDU(data=np.ones((npix, npix), "float32"), header=hdr, name="SCI")
    fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(path, overwrite=True)


def test_mosaic_covering_picks_the_tile_that_contains_the_positions(tmp_path, monkeypatch):
    # two -merged tiles on DISJOINT sky (cloudef o005: the tile the 'prefer merged' rule picks does
    # NOT contain the overlap stars).  _mosaic_covering must return the one that actually covers them.
    monkeypatch.setattr(D, "BASE", str(tmp_path))
    d = tmp_path / "cloudef" / "F162M" / "pipeline"
    _write_i2d_wcs(str(d / "jw02092-o005_t001_nircam_clear-f162m-merged_i2d.fits"),
                   ra0=266.45, dec0=-28.50, npix=1000)
    _write_i2d_wcs(str(d / "jw02092-o005_t001_nircam_clear-f162m-nrca_i2d.fits"),
                   ra0=266.66, dec0=-28.51, npix=1000)
    o = Observation(program="2092", obs="005", target="Cloud E/F", release_field="cloudef",
                    instrument="NIRCam", filters=["F162M"], visits=[], epoch="", notes="")
    ra = np.array([266.6600, 266.6603, 266.6597]); dec = np.array([-28.5100, -28.5103, -28.5097])
    p, n = D._mosaic_covering(o, "F162M", ra, dec)
    assert p is not None and p.endswith("-nrca_i2d.fits") and n == 3      # the covering tile, not merged
    # positions on NEITHER tile -> (None, 0)
    p2, n2 = D._mosaic_covering(o, "F162M", np.array([10.0]), np.array([10.0]))
    assert p2 is None and n2 == 0


def test_stage1_falls_back_to_mast_i2d_never_blank(tmp_path, monkeypatch):
    # MAST always delivers an i2d, so a delivered filter must never render a blank panel. F210M has
    # only a MAST i2d (no reduced mosaic); stage 1 must show it from MAST, not "(no i2d)" (issue #38).
    monkeypatch.setattr(D, "BASE", str(tmp_path))
    red = tmp_path / "cloudef" / "F162M" / "pipeline"
    _write_i2d(str(red / "jw02092-o005_t001_nircam_clear-f162m-merged_i2d.fits"))
    mast = tmp_path / "cloudef" / "mastDownload" / "JWST" / "jw02092-o005_t002_nircam_clear-f210m"
    _write_i2d(str(mast / "jw02092-o005_t002_nircam_clear-f210m_i2d.fits"))
    o = Observation(program="2092", obs="005", target="Cloud E/F", release_field="cloudef",
                    instrument="NIRCam", filters=["F162M", "F210M"], visits=[], epoch="", notes="")
    png, m = D.stage1_mosaics(o, "F210M", "F162M")           # SW pick = the MAST-only filter
    assert os.path.exists(png)
    assert m["mast_fallback_filters"] == ["F210M"]           # rendered from MAST, not blank
    assert "F210M" in m["finite_fraction"] and "F162M" in m["finite_fraction"]
    assert m["sw_present"] is False                          # gate still keys on the REDUCED mosaic


def _write_skycoord_cat(path, ra, dec):
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    from astropy.table import Table
    t = Table()
    t["skycoord"] = SkyCoord(np.asarray(ra) * u.deg, np.asarray(dec) * u.deg)
    t.write(path, overwrite=True)


def test_jwst_positions_falls_back_to_dao(tmp_path, monkeypatch):
    # no merged/MAST catalog, only a per-filter DAO position catalog -> positions still returned
    monkeypatch.setattr(D, "BASE", str(tmp_path))
    monkeypatch.setattr(D, "_jwst_sources",
                        lambda o, f, position_valid=False: (None, None, None))  # no merged/MAST
    d = tmp_path / "gc2211" / "catalogs"; d.mkdir(parents=True)
    ra = np.linspace(266.4, 266.45, 60); dec = np.linspace(-29.0, -28.95, 60)
    _write_skycoord_cat(d / "f200w_merged_indivexp_merged_m6_dao_basic_o046_vetted.fits", ra, dec)
    sc, src = D._jwst_positions(_obs(field="gc2211", obs="046", filt="F200W"), "F200W")
    assert sc is not None and len(sc) == 60
    assert "release-dao(positions)" in src


# --------------------------------------------------------------------------- _daophot_glob


def test_daophot_glob_prefers_this_obs(tmp_path, monkeypatch):
    monkeypatch.setattr(D, "BASE", str(tmp_path))
    d = tmp_path / "gc2211" / "F200W"; d.mkdir(parents=True)
    for det in ("nrca1", "nrcb1"):
        _touch(d, f"f200w_{det}_o023_visit001_exp1_m3_daophot_basic.fits")
        _touch(d, f"f200w_{det}_o050_visit001_exp1_m3_daophot_basic.fits")
    got = D._daophot_glob(_obs(obs="023"), "F200W")
    assert got
    assert all("_o023_" in os.path.basename(g) for g in got)
    assert not any("_o050_" in os.path.basename(g) for g in got)


def test_daophot_glob_other_obs_only_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(D, "BASE", str(tmp_path))
    d = tmp_path / "gc2211" / "F200W"; d.mkdir(parents=True)
    _touch(d, "f200w_nrca1_o050_visit001_exp1_m3_daophot_basic.fits")   # only o050 present
    # a per-obs generation exists but not for o023 -> must NOT fall back to o050 or legacy
    assert D._daophot_glob(_obs(obs="023"), "F200W") == []


def test_daophot_glob_untokened_single_obs_field(tmp_path, monkeypatch):
    monkeypatch.setattr(D, "BASE", str(tmp_path))
    d = tmp_path / "brick" / "F212N"; d.mkdir(parents=True)
    _touch(d, "f212n_nrca1_visit001_exp1_m3_daophot_basic.fits")
    _touch(d, "f212n_nrcb1_visit001_exp1_m3_daophot_basic.fits")
    got = D._daophot_glob(_obs(field="brick", obs="001", filt="F212N"), "F212N")
    assert len(got) == 2


def test_daophot_glob_untokened_excludes_stray_tokened(tmp_path, monkeypatch):
    # if a tokened generation exists, an untokened field is never used for a non-matching obs
    monkeypatch.setattr(D, "BASE", str(tmp_path))
    d = tmp_path / "brick" / "F212N"; d.mkdir(parents=True)
    _touch(d, "f212n_nrca1_visit001_exp1_m3_daophot_basic.fits")           # legacy untokened
    _touch(d, "f212n_nrca1_o007_visit001_exp1_m3_daophot_basic.fits")      # a per-obs generation
    assert D._daophot_glob(_obs(field="brick", obs="001", filt="F212N"), "F212N") == []


# --------------------------------------------------------------------------- NaN centroid guards
def test_finite_sc_drops_nan():
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    sc = SkyCoord([1.0, 2.0, np.nan] * u.deg, [1.0, np.nan, 3.0] * u.deg)
    assert len(D._finite_sc(sc)) == 1     # only row 0 is finite in both axes


def _write_daophot(path, ras, decs):
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    from astropy.table import Table
    n = len(ras)
    t = Table()
    t["skycoord_centroid"] = SkyCoord(np.asarray(ras) * u.deg, np.asarray(decs) * u.deg)
    t["dra"] = np.full(n, 0.003); t["ddec"] = np.full(n, 0.003); t["flux_fit"] = np.full(n, 100.0)
    t.write(path, overwrite=True)


def test_module_positions_dead_vs_absent(tmp_path, monkeypatch):
    # NRCA present but ALL-NaN centroids (astrometry failure); NRCB genuinely absent.
    monkeypatch.setattr(D, "BASE", str(tmp_path))
    d = tmp_path / "gc2211" / "F200W"; d.mkdir(parents=True)
    nan = np.full(80, np.nan)
    for det in ("nrca1", "nrca2", "nrca3", "nrca4"):
        _write_daophot(d / f"f200w_{det}_o023_visit001_exp1_m3_daophot_basic.fits", nan, nan)
    a_sc, b_sc, meta = D._module_positions(_obs(obs="023"), "F200W")
    assert a_sc is None and meta["a"]["present"] and meta["a"]["dead"]      # dead, NOT absent
    assert b_sc is None and not meta["b"]["present"]                        # genuinely absent


def test_module_positions_normal(tmp_path, monkeypatch):
    monkeypatch.setattr(D, "BASE", str(tmp_path))
    d = tmp_path / "gc2211" / "F200W"; d.mkdir(parents=True)
    ra = np.linspace(266.4, 266.5, 200); dec = np.linspace(-28.9, -28.8, 200)
    for det in ("nrca1", "nrcb1"):
        _write_daophot(d / f"f200w_{det}_o023_visit001_exp1_m3_daophot_basic.fits", ra, dec)
    a_sc, b_sc, meta = D._module_positions(_obs(obs="023"), "F200W")
    assert a_sc is not None and b_sc is not None
    assert not meta["a"]["dead"] and not meta["b"]["dead"]
    assert meta["a"]["nan_frac"] == 0.0


# --------------------------------------------------------------------------- position validity
#
# jicama's merge accepts a cross-filter match anywhere inside max_offset=0.10", which at GC
# density also admits the NEIGHBOUR of a star undetected in that filter -- so skycoord_<filt>
# can be a position ~one neighbour-spacing away.  On brick 2221-o001 F212N those rows were 43%
# of the second lobe in the JWST-VIRAC offset cloud and 2% of its core (same magnitude, same
# saturated fraction: match quality, not a bright-star centroid bias).  JWST-GC/data-qa#1.


def _merged_table(n=200, sep_arcsec=None):
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    from astropy.table import Table
    ra = np.linspace(266.40, 266.45, n)
    dec = np.linspace(-29.00, -28.95, n)
    t = Table()
    t["skycoord_f200w"] = SkyCoord(ra * u.deg, dec * u.deg)
    t["mag_vega_f200w"] = np.linspace(14.0, 20.0, n)
    if sep_arcsec is not None:
        t["sep_f200w"] = (np.asarray(sep_arcsec, float) / 3600.0) * u.deg   # merge writes degrees
    return t


def test_position_valid_drops_loose_matches():
    n = 200
    sep = np.where(np.arange(n) < 60, 0.30, 0.004)      # 60 rows borrowed from a neighbour
    t = _merged_table(n, sep)
    finite = np.ones(n, dtype=bool)
    ok, note = D._position_valid(t, "F200W", finite)
    assert ok.sum() == n - 60
    assert not ok[:60].any() and ok[60:].all()
    assert "sep<=" in note


def test_position_valid_noop_without_sep_column():
    # MAST / per-filter DAO catalogs have no sep_<filt>: the position IS the detection's own,
    # so the cut must pass everything through rather than emptying the source.
    t = _merged_table(120, sep_arcsec=None)
    finite = np.ones(120, dtype=bool)
    ok, note = D._position_valid(t, "F200W", finite)
    assert ok.all() and note is None


def test_position_valid_keeps_uncut_when_too_few_survive():
    # a field where almost nothing passes must NOT become "offset unmeasurable"; the cut backs
    # off and labels itself instead.
    n = 200
    t = _merged_table(n, np.full(n, 0.40))
    finite = np.ones(n, dtype=bool)
    ok, note = D._position_valid(t, "F200W", finite)
    assert ok.sum() == n
    assert note == "sep-cut-skipped(too-few)"


def test_position_valid_unitless_sep_column_treated_as_degrees():
    # a Column with no unit must not raise (Column.to() exists but cannot convert) and must be
    # read as degrees, matching what merge_catalogs writes.
    from astropy.table import Table
    t = _merged_table(200, np.where(np.arange(200) < 50, 0.30, 0.004))
    t["sep_f200w"] = np.asarray(t["sep_f200w"], float)      # strip the unit
    ok, note = D._position_valid(t, "F200W", np.ones(200, dtype=bool))
    assert ok.sum() == 150 and "sep<=" in note


def _uniform_sc(n, seed, ra0=266.4, dec0=-29.0, span=0.02):
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    rng = np.random.default_rng(seed)
    ra = ra0 + rng.uniform(0, span, n); dec = dec0 + rng.uniform(0, span, n)
    return SkyCoord(ra * u.deg, dec * u.deg)


def test_same_star_tie_refuses_when_bulk_is_large():
    # the guard that keeps this from becoming a dense nearest-neighbour median: without a verified
    # SMALL global tie the nearest pair is not the right star, so it must refuse rather than
    # return a number that collapses toward zero.
    from data_qa import astrometry_audit as aa
    a = _uniform_sc(400, 0)
    assert aa.same_star_tie(a, a, bulk=dict(off=500.0)) is None
    # an explicitly-supplied bulk with no peak_ratio is treated as vetted by the caller
    out = aa.same_star_tie(a, a, bulk=dict(off=2.0))
    assert out is not None and out["off"] < 1e-6 and out["npairs"] == 400


def test_same_star_tie_refuses_ambiguous_peak_ratio():
    # an ambiguous xcorr (peak_ratio below MIN_PEAK_RATIO) with a small off must NOT admit the
    # same-star estimate -- otherwise a chance-small off silently fabricates agreement.
    from data_qa import astrometry_audit as aa
    a = _uniform_sc(400, 1)
    assert aa.same_star_tie(a, a, bulk=dict(off=2.0, peak_ratio=1.0)) is None
    assert aa.same_star_tie(a, a, bulk=dict(off=2.0, peak_ratio=aa.MIN_PEAK_RATIO)) is not None


def test_same_star_tie_real_path_bulk_none():
    # the path stage 4 actually uses: bulk=None, so xcorr is measured internally (incl. peak_ratio).
    # A small real tie is recovered; a >100 mas mis-registration is refused (nearest pair is wrong).
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    from data_qa import astrometry_audit as aa
    a = _uniform_sc(3000, 2)
    cosd = np.cos(np.radians(-29.0))
    b_small = SkyCoord((a.ra.deg + 8.0 / 3.6e6 / cosd) * u.deg, a.dec.deg * u.deg)   # +8 mas
    out = aa.same_star_tie(a, b_small)                       # bulk=None -> real xcorr path
    assert out is not None and abs(out["off"] - 8.0) < 4.0
    b_far = SkyCoord((a.ra.deg + 300.0 / 3.6e6 / cosd) * u.deg, a.dec.deg * u.deg)   # +300 mas
    assert aa.same_star_tie(a, b_far) is None


def test_xcorr_recentring_no_floor_at_zero():
    # on a uniform, UNCLUSTERED synthetic the recentred xcorr must not carry the ~half-bin (~5 mas)
    # quantization floor at truth 0, yet must still track a real 90 mas offset.
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    from data_qa import astrometry_audit as aa
    a = _uniform_sc(2000, 3)
    cosd = np.cos(np.radians(-29.0))
    z = aa.xcorr(a, a)
    assert z is not None and z["off"] < 1.5                  # was ~5 mas with a single refinement
    b90 = SkyCoord((a.ra.deg + 90.0 / 3.6e6 / cosd) * u.deg, a.dec.deg * u.deg)
    f = aa.xcorr(a, b90)
    assert f is not None and abs(f["off"] - 90.0) < 3.0


def _stage4_injection(monkeypatch, shift_mas):
    """Run stage4_offsets end-to-end on a dense uniform synthetic field whose JWST positions are
    shifted ``shift_mas`` in RA from the reference, mocking only the I/O seams.  Returns metrics."""
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    from data_qa import astrometry_audit as aa
    rng = np.random.default_rng(11)
    n = 12000
    ra = 266.40 + rng.uniform(0, 0.02, n); dec = -29.00 + rng.uniform(0, 0.02, n)
    ref_sc = SkyCoord(ra * u.deg, dec * u.deg)
    cosd = np.cos(np.radians(-29.0))
    jsc = SkyCoord((ra + shift_mas / 3.6e6 / cosd) * u.deg, dec * u.deg)   # JWST shifted vs ref
    monkeypatch.setattr(D, "_mosaic_path", lambda o, sw: "/dev/null/mosaic_i2d.fits")
    monkeypatch.setattr(D, "_refcat_path", lambda o: "/dev/null/refcat.fits")
    monkeypatch.setattr(D, "_obs_epoch", lambda o, path: 2022.5)
    monkeypatch.setattr(aa, "load_reference", lambda ref, ep: (ref_sc, None))
    # stage 4 sources positions from the MAST L3 catalogue; feed the shifted synthetic field there.
    monkeypatch.setattr(D, "_mast_catalog_positions", lambda o, sw: (jsc, None))
    monkeypatch.setattr(D, "_jwst_positions", lambda o, sw: (jsc, "release-m8"))
    monkeypatch.setattr(D, "_module_positions", lambda o, sw: (None, None, None))
    monkeypatch.setattr(D, "_crossmatch_offset", lambda j, r, restrict_footprint=False: None)
    _png, metrics = D.stage4_offsets(_obs(field="brick", obs="001", filt="F212N"), "F212N")
    return metrics


def test_stage4_passes_at_zero_offset(monkeypatch):
    # a correctly-registered frame (0 mas) must PASS end-to-end.
    m = _stage4_injection(monkeypatch, 0.0)
    assert m["passed"] is True
    assert m["cell_off_med"] < 10 and m["gate_off_mas"] < 10


def test_stage4_fails_on_90mas_misregistration(monkeypatch):
    # THE blocker: a 90 mas bulk mis-registration must FAIL, even though the same-star refinement
    # (mutual NN inside 0.05") would report a small collapsed value.  The gate reads the cell
    # histogram median, so the mis-registration cannot pass.
    m = _stage4_injection(monkeypatch, 90.0)
    assert m["passed"] is False
    assert m["cell_off_med"] > 75 and m["gate_off_mas"] > 75


def test_stage4_unavailable_when_no_mast_catalogue(monkeypatch):
    # No MAST L3 catalogue on disk is a not-yet-delivered state, not a defect: available=False,
    # passed=None, and NO red flag (the stage is left blank, not flagged).
    monkeypatch.setattr(D, "_mosaic_path", lambda o, sw: "/dev/null/m_i2d.fits")
    monkeypatch.setattr(D, "_refcat_path", lambda o: "/dev/null/ref.fits")
    monkeypatch.setattr(D, "_obs_epoch", lambda o, p: 2022.5)
    monkeypatch.setattr(D.aa, "load_reference", lambda ref, ep: (object(), None))   # ref present
    monkeypatch.setattr(D, "_mast_catalog_positions", lambda o, sw: None)           # no MAST cat
    monkeypatch.setattr(D, "_save", lambda fig, name: name)
    _png, m = D.stage4_offsets(_obs(field="brick", obs="001", filt="F212N"), "F212N")
    assert m.get("available") is False and m.get("passed") is None and not m.get("red_flag")
    assert m.get("source") == "MAST L3 catalogue"


def test_source_label_from_path_tokens():
    f = D._source_label_from_path
    assert f("/x/mastDownload/JWST/jw10678-o1_t1_nircam_clear-f212n/..._cat.ecsv") == "MAST L3"
    assert f("/x/mastDownload/JWST/..._i2d.fits") == "MAST i2d"
    assert f("jw10678-o1_t001_nircam_clear-f212n_m1_daophot_cat.fits") == "jicama-m1"
    assert f("jw10678-o1_t001_nircam_clear-f212n_m3_daophot_basic_mergedcat.fits") == "jicama-m3"
    assert f("jw10678-o1_t001_nircam_clear-f212n-merged_cat.ecsv") == "jicama-m3"
    assert f("gaia_virac2_refcat_epoch2026.7_o1.fits") == "VIRAC/Gaia ref"
    assert f("jw10678-o1_LOG.MATCHUP.XYMEEE") == "JWST1PASS"
    assert f("something_miri_f770w_i2d.fits") == "MIRI i2d"
    # a MIRI CATALOGUE is one of our products: keep the m-level with a MIRI tag, do not collapse
    # to a bare "MIRI" (which could not be told from a MAST/unknown-origin MIRI catalogue).
    assert f("jw10678-o40_t001_miri_f770w_m2_daophot_cat.ecsv") == "jicama-m2 MIRI"
    assert f("jw10678-o40_t001_miri_clear-f770w-merged_cat.ecsv") == "jicama-m3 MIRI"
    assert f("jw10678-o40_t001_miri_f770w_cat.ecsv") == "jicama MIRI"
    assert f("/x/mastDownload/JWST/jw10678-o40_t1_miri_f770w/..._cat.ecsv") == "MAST L3"


def test_save_annotates_data_source(tmp_path, monkeypatch):
    """`_save` leaves a 'Data source' footer built from the files the stage recorded via _used."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    monkeypatch.setattr(D, "OUTDIR", str(tmp_path))
    monkeypatch.setattr(D, "_INPUTS",
                        [("cat", "jw10678-o1_t001_nircam_clear-f212n-merged_cat.ecsv"),
                         ("ref", "gaia_virac2_refcat_epoch2026.7_o1.fits")])
    fig = plt.figure()
    D._save(fig, "src_footer_test.png")
    foot = [t.get_text() for t in fig.texts if "Data source" in t.get_text()]
    assert len(foot) == 1
    assert "jicama-m3" in foot[0] and "VIRAC/Gaia ref" in foot[0]
    plt.close(fig)


def test_save_does_not_overwrite_a_stage_own_source_footer(tmp_path, monkeypatch):
    """A stage that already wrote its own, more specific 'Data source' footer keeps it -- the
    central annotation must not stack a second one (a duplicated footer is ugly, a replaced one
    is wrong)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    monkeypatch.setattr(D, "OUTDIR", str(tmp_path))
    monkeypatch.setattr(D, "_INPUTS",
                        [("cat", "jw10678-o1_t001_nircam_clear-f212n-merged_cat.ecsv")])
    fig = plt.figure()
    fig.text(0.5, 0.005, "Data source: MAST L3", ha="center")   # the stage's own, specific footer
    D._save(fig, "src_footer_keep_test.png")
    foots = [t.get_text() for t in fig.texts if "Data source" in t.get_text()]
    assert foots == ["Data source: MAST L3"]                    # untouched, not stacked
    plt.close(fig)


def test_offset_summary_figure_measures_or_blank(monkeypatch):
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    jsc = SkyCoord(np.full(50, 266.4) * u.deg, np.full(50, -28.9) * u.deg)
    ref = jsc
    cells = [{"i": k % 4, "j": k // 4, "ra": 266.4, "dec": -28.9, "dra": 1.0, "dde": 0.0,
              "off": 1.0, "n": 40} for k in range(16)]
    monkeypatch.setattr(D, "_cell_offsets", lambda j, r: (cells, [], 4))
    monkeypatch.setattr(D, "_cell_consistency", lambda c, d: {"cells": c, "off_med": 1.0,
                        "spread": 0.5, "n_cells": len(c), "coverage": 1.0})
    monkeypatch.setattr(D.aa, "same_star_tie", lambda j, r: {"off": 1.2, "npairs": 40, "scatter": 0.3})
    monkeypatch.setattr(D, "_offset_cloud", lambda j, r: (np.array([1.0, 2.0]), np.array([0.0, 0.0]), 1.0))
    monkeypatch.setattr(D, "_save", lambda fig, name: name)
    monkeypatch.setattr(D, "_bulk_offset",
                        lambda j, r, *a, **k: (np.array([1.0, 2.0]), np.array([0.0, 0.0]), 1.2))
    png, sub = D._offset_summary_figure(_obs(), "F212N", jsc, ref, "release:x_m3_cat.ecsv", "out.png")
    assert png == "out.png" and sub["offset_med_mas"] == 1.2 and sub["offset_unmeasurable"] is False
    # no confident whole-field peak -> UNMEASURABLE; the noise-locked per-cell median (jw10678 o112:
    # 464 mas from cells at 600-2200 mas) must never stand in for the field offset
    monkeypatch.setattr(D, "_cell_consistency", lambda c, d: {"cells": c, "off_med": 464.0,
                        "spread": 900.0, "n_cells": len(c), "coverage": 1.0})
    monkeypatch.setattr(D, "_bulk_offset", lambda j, r, *a, **k: None)
    png, sub = D._offset_summary_figure(_obs(), "F212N", jsc, ref, "release:x_m3_cat.ecsv", "out.png")
    assert png == "out.png" and sub["offset_med_mas"] is None and sub["offset_unmeasurable"] is True
    # no catalogue -> blank (no figure, no metrics), never a red flag
    assert D._offset_summary_figure(_obs(), "F212N", None, ref, "x", "o.png") == (None, {})


# --------------------------------------------------------------------------- stage 7 (MAST vs pipeline)
def test_offset_cloud_recovers_bulk_shift():
    # jicama-like catalogue offset from VIRAC by a KNOWN 120 mas E; _offset_cloud must recover it
    # as the cloud centre, having coarse-aligned on the xcorr peak first
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    rng = np.random.RandomState(1)
    ra = 266.40 + rng.uniform(0, 0.03, 1500); dec = -28.90 + rng.uniform(0, 0.03, 1500)
    ref = SkyCoord(ra * u.deg, dec * u.deg)
    cosd = np.cos(np.radians(-28.9))
    jsc = SkyCoord((ra + 120.0 / 3.6e6 / cosd) * u.deg, dec * u.deg)   # jsc is +120 mas E of ref
    out = D._offset_cloud(jsc, ref)
    assert out is not None
    dra, dde, bulk = out
    assert abs(bulk - 120.0) < 15 and abs(np.median(dra) - 120.0) < 15


def test_offset_cloud_none_when_offset_exceeds_maxsep():
    # a gross offset (> the 1.5" xcorr window) must return None, so stage 7 flags a grossly
    # mis-registered product; a wrong small number here would let it pass
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    rng = np.random.RandomState(2)
    ra = 266.40 + rng.uniform(0, 0.03, 1500); dec = -28.90 + rng.uniform(0, 0.03, 1500)
    ref = SkyCoord(ra * u.deg, dec * u.deg)
    cosd = np.cos(np.radians(-28.9))
    jsc = SkyCoord((ra + 5000.0 / 3.6e6 / cosd) * u.deg, dec * u.deg)   # 5" E, way past 1.5"
    assert D._offset_cloud(jsc, ref) is None


def test_caption_stage7_full():
    cap = D.caption_for(7, dict(stage=7, n_jicama=294615, n_mast=39365,
                                jicama_offset_med_mas=14.5, mast_offset_med_mas=134.0))
    assert "DOCROOT" not in cap and "qa_methods.md#stage7" in cap
    assert "MAST vs pipeline" in cap and "offset from VIRAC" in cap
    assert "14 mas (jicama)" in cap and "134 mas (MAST)" in cap
    # jicama (14) < MAST (134): the improvement clause is present
    assert "so the pipeline sits closer to VIRAC" in cap
    # the cloud width is set by the match radius -> the caveat is present
    assert "0.1″ cross-match radius" in cap


def test_stage7_title_tightening_only_when_jicama_tighter():
    # (blocker A/C) the title says 'tighter' ONLY when both offsets are measured AND jicama < MAST
    tighter = D._stage7_astrom_title((134.0,) * 3, (14.5,) * 3)   # jicama 14 < MAST 134
    assert "tighter" in tighter and "jicama 14 mas vs MAST 134 mas" in tighter
    # Sgr C o012 case: jicama 19.56 is WORSE than MAST 17.62 -> no 'tighter', both numbers reported
    worse = D._stage7_astrom_title((17.62,) * 3, (19.56,) * 3)
    assert "tighter" not in worse
    assert "jicama 20 mas vs MAST 18 mas" in worse
    # only one side measured -> neutral wording, never 'tighter'
    assert "tighter" not in D._stage7_astrom_title(None, (14.5,) * 3)
    assert "tighter" not in D._stage7_astrom_title((17.6,) * 3, None)


def test_caption_stage7_neutral_when_jicama_not_tighter():
    # (blocker A) jicama (20) NOT tighter than MAST (18): caption reports both, no 'tightening' claim
    cap = D.caption_for(7, dict(stage=7, jicama_offset_med_mas=19.56, mast_offset_med_mas=17.62))
    assert "20 mas (jicama)" in cap and "18 mas (MAST)" in cap
    assert "so the pipeline sits closer to VIRAC" not in cap
    assert "farther from VIRAC" not in cap                  # small diff -> no mis-registration alarm
    assert "MAST is about as close to VIRAC as the pipeline here" in cap


def test_caption_stage7_drops_clause_when_mast_unavailable():
    # (blocker B) only the MAST offset is unmeasurable: the caption drops the improvement clause
    # and states that the comparison is unavailable, claiming no tightening.
    cap = D.caption_for(7, dict(stage=7, jicama_offset_med_mas=14.5))
    assert "so the pipeline sits closer to VIRAC" not in cap
    assert "MAST comparison is unavailable" in cap


def test_stage7_verdict_jicama_unmeasurable_fails_and_redflags():
    # (blocker B / test 3) our own offset unmeasurable -> passed False AND red_flag set
    passed, red_flag, reason = D._stage7_verdict("our.fits", (17.6,) * 3, None, jic_unmeas=True)
    assert passed is False and red_flag is True and reason and "jicama" in reason
    # and the caption then enters the red-flag branch (no 'tightening' claim)
    cap = D.caption_for(7, dict(stage=7, red_flag=True, red_flag_reason=reason))
    assert cap.startswith("🚩") and "RED FLAG" in cap
    assert "so the pipeline sits closer to VIRAC" not in cap


def test_stage7_verdict_mast_only_unmeasurable_passes_without_redflag():
    # (blocker B / test 2) only the MAST offset unmeasurable -> the comparison is what is
    # unavailable, so passed stays True with no red flag and the boolean agrees with the caption.
    passed, red_flag, reason = D._stage7_verdict("our.fits", None, (14.5,) * 3, jic_unmeas=False)
    assert passed is True and red_flag is False and reason is None


def test_depth_hist_guards_empty_and_all_nan():
    # REGRESSION: stage 7's depth histogram crashed with "autodetected range of [nan, nan] is not
    # finite" when the common-window selection was EMPTY or the MAST abmag column was all NaN.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    assert D._depth_hist(ax, np.array([]), "empty", "#000000") == 0          # empty: no crash
    assert D._depth_hist(ax, np.full(40, np.nan), "all-nan", "#000000") == 0  # all NaN: no crash
    assert D._depth_hist(ax, np.array([1.0, np.nan, 2.0, 3.0]), "mixed", "#000000") == 3
    plt.close(fig)


def test_vega_zeropoint_drops_nan_catalog_coords(tmp_path, monkeypatch):
    # REGRESSION: stage 6 crashed "Catalog coordinates cannot contain NaN entries" (quintuplet
    # o003, sickle o007) because _vega_zeropoint fed a merged catalogue's skycoord_<filt> straight
    # into match_to_catalog_sky, and some rows carry NaN positions.
    from astropy.table import Table
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    n = 80
    ra = np.append(266.40 + np.arange(n) * 1e-4, [np.nan, np.nan])     # 80 finite + 2 NaN rows
    dec = np.append(-28.90 + np.arange(n) * 1e-4, [np.nan, -28.9])
    vega = np.append(18.0 + np.arange(n) * 0.02, [20.0, 20.0])
    cat = tmp_path / "merged.fits"
    Table({"skycoord_f212n": SkyCoord(ra * u.deg, dec * u.deg),
           "mag_vega_f212n": vega}).write(str(cat), overwrite=True)
    monkeypatch.setattr(D, "_catalog_with_vega",
                        lambda o, f: (str(cat), "mag_vega_f212n", "skycoord_f212n"))
    o = _obs(field="quintuplet", filt="F212N")
    sc = SkyCoord(ra[:n] * u.deg, dec[:n] * u.deg)                     # detections at finite positions
    zp = D._vega_zeropoint(o, "F212N", sc, vega[:n] - 5.0)             # instr = vega - 5 -> ZP +5
    assert zp is not None and abs(zp - 5.0) < 0.01                     # no crash; correct ZP


def test_stage7_verdict_pass_requires_mosaic_and_no_worse_offset():
    # a mosaic must exist; and where both offsets are measured ours must be no worse
    assert D._stage7_verdict(None, (134.0,) * 3, (14.5,) * 3, jic_unmeas=False)[0] is False
    assert D._stage7_verdict("our.fits", (134.0,) * 3, (14.5,) * 3, jic_unmeas=False)[0] is True
    # jicama much worse than MAST -> not improved -> not a pass
    assert D._stage7_verdict("our.fits", (18.0,) * 3, (140.0,) * 3, jic_unmeas=False)[0] is False


def test_mast_i2d_cross_field_and_twildcard(tmp_path, monkeypatch):
    # (🟠) o002 belongs to cloudc but its MAST i2d is staged under brick/mastDownload; the finder
    # must reach it via the cross-field fallback, and it must not depend on t001 exactly.
    monkeypatch.setattr(D, "BASE", str(tmp_path))
    brick = tmp_path / "brick" / "mastDownload"; brick.mkdir(parents=True)
    (tmp_path / "cloudc" / "mastDownload").mkdir(parents=True)      # empty for this obs
    _touch(brick, "jw02221-o002_t004_nircam_clear-f212n_i2d.fits")  # non-t001 tag, sibling field
    _touch(brick, "jw02221-o002_t004_nircam_clear-f212n-merged_i2d.fits")   # reprocessed: excluded
    o = Observation(program="2221", obs="002", target="Cloud C", release_field="cloudc",
                    instrument="NIRCam", filters=["F212N"], visits=[], epoch="", notes="")
    got = D._mast_i2d(o, "F212N")
    assert got is not None and got.endswith("clear-f212n_i2d.fits")
    assert "merged" not in os.path.basename(got)


def test_mast_i2d_and_l3cat_pathing(tmp_path, monkeypatch):
    monkeypatch.setattr(D, "BASE", str(tmp_path))
    md = tmp_path / "brick" / "mastDownload"; md.mkdir(parents=True)
    _touch(md, "jw02221-o001_t001_nircam_clear-f212n_i2d.fits")
    _touch(md, "jw02221-o001_t001_nircam_clear-f212n_cat.fits")
    _touch(md, "jw02221001001_03101_00001_nrca4_destreak_cat.fits")   # a per-detector one to exclude
    o = Observation(program="2221", obs="001", target="Brick", release_field="brick",
                    instrument="NIRCam", filters=["F212N"], visits=[], epoch="", notes="")
    assert D._mast_i2d(o, "F212N").endswith("clear-f212n_i2d.fits")
    # local L3 cat is found (download not attempted); the per-detector destreak cat is excluded
    got = D._mast_l3_catalog(o, "F212N", allow_download=False)
    assert got.endswith("clear-f212n_cat.fits")


def test_mast_l3_catalog_never_returns_our_merged_product(tmp_path, monkeypatch):
    """The MAST catalogue search reaches ``<FILT>/pipeline``/``images-merged`` where OUR reduction
    writes ``..._t001_...-merged_cat.ecsv``.  That product is ours, not MAST: picking it swaps the
    stage-7 MAST/pipeline series and inverts the conclusion (JWST-GC/data-qa#192).  The genuine MAST
    per-i2d catalogue under mastDownload is chosen, and when only our merged product exists the search
    finds nothing MAST (download disabled)."""
    monkeypatch.setattr(D, "BASE", str(tmp_path))
    pipe = tmp_path / "gc-treasury" / "F212N" / "pipeline"; pipe.mkdir(parents=True)
    _touch(pipe, "jw10678-o137_t001_nircam_clear-f212n-merged_cat.ecsv")   # OUR product, not MAST
    o = Observation(program="10678", obs="137", target="GC Treasury",
                    release_field="gc-treasury", instrument="NIRCam", filters=["F212N"],
                    visits=[], epoch="", notes="")
    assert D._mast_l3_catalog(o, "F212N", allow_download=False) is None   # our merged is not "MAST"

    md = (tmp_path / "gc-treasury" / "mastDownload" / "JWST"
          / "jw10678-o137_t137_nircam_clear-f212n"); md.mkdir(parents=True)
    _touch(md, "jw10678-o137_t137_nircam_clear-f212n_cat.ecsv")           # genuine MAST per-i2d
    got = D._mast_l3_catalog(o, "F212N", allow_download=False)
    assert got is not None and got.endswith("clear-f212n_cat.ecsv")
    assert "merged" not in os.path.basename(got)


def test_load_mast_catalog_radec_and_mag(tmp_path):
    from astropy.table import Table
    p = str(tmp_path / "cat.fits")
    ra = np.append(266.40 + np.arange(25) * 1e-4, np.nan)     # 25 finite + 1 NaN
    dec = np.append(-28.90 + np.arange(25) * 1e-4, -28.9)
    mag = np.append(18.0 + np.arange(25) * 0.05, 25.0)
    Table({"ra": ra, "dec": dec, "aper50_abmag": mag}).write(p, overwrite=True)
    sc, m = D._load_mast_catalog(p)
    assert sc is not None and len(sc) == 25 and np.all(np.isfinite(sc.ra.deg))
    assert len(m) == 25


def test_mast_l3_catalog_none_when_absent_and_no_download(tmp_path, monkeypatch):
    monkeypatch.setattr(D, "BASE", str(tmp_path))
    (tmp_path / "brick" / "mastDownload").mkdir(parents=True)
    o = Observation(program="2221", obs="001", target="Brick", release_field="brick",
                    instrument="NIRCam", filters=["F212N"], visits=[], epoch="", notes="")
    assert D._mast_l3_catalog(o, "F212N", allow_download=False) is None


# ---- REGRESSION: the MAST source catalogue is delivered as .ecsv under mastDownload/ with a
# pupil-filter product name (F162M ships as f150w2-f162m).  A resolver that only looked at
# pipeline/*_cat.fits missed it and red-flagged a catalogued obs (cloud E/F o002) as "not
# catalogued".  These pin every home + naming so it cannot regress. ----
def _write_mast_ecsv(path, n=60):
    from astropy.table import Table
    ra = 266.40 + np.arange(n) * 1e-4
    dec = -28.90 + np.arange(n) * 1e-4
    Table({"sky_centroid.ra": ra, "sky_centroid.dec": dec,
           "aper_total_abmag": 18.0 + np.arange(n) * 0.02}).write(str(path), overwrite=True)


def test_mast_source_catalog_finds_mastdownload_ecsv_pupil_filter(tmp_path, monkeypatch):
    monkeypatch.setattr(D, "BASE", str(tmp_path))
    # F162M is delivered as the pupil pair f150w2-f162m, as an .ecsv, one product-dir deep.
    d = tmp_path / "cloudef" / "mastDownload" / "JWST" / "jw02092-o002_t001_nircam_f150w2-f162m"
    d.mkdir(parents=True)
    _write_mast_ecsv(d / "jw02092-o002_t001_nircam_f150w2-f162m_cat.ecsv")
    o = Observation(program="2092", obs="002", target="Cloud E/F", release_field="cloudef",
                    instrument="NIRCam", filters=["F162M"], visits=[], epoch="", notes="")
    got = D._mast_source_catalog(o, "F162M")
    assert got is not None and got.endswith("f150w2-f162m_cat.ecsv")
    # and it flows through to usable positions (the stage-4 red-flag was jsc is None)
    sc, mag, lbl = D._jwst_sources(o, "F162M", position_valid=True)
    assert sc is not None and len(sc) >= 30 and "MAST" in lbl


def test_mast_source_catalog_downloads_when_absent(tmp_path, monkeypatch):
    # No local product: the resolver DOWNLOADS instead of returning None (-> stage red-flag).
    monkeypatch.setattr(D, "BASE", str(tmp_path))
    (tmp_path / "cloudef").mkdir()
    o = Observation(program="2092", obs="002", target="Cloud E/F", release_field="cloudef",
                    instrument="NIRCam", filters=["F162M"], visits=[], epoch="", notes="")
    called = {}
    def _fake_dl(oo, ff):
        called["hit"] = (oo.obsid, ff)
        return "/scratch/dl/jw02092-o002_t001_nircam_f150w2-f162m_cat.ecsv"
    monkeypatch.setattr(D, "_download_mast_l3_catalog", _fake_dl)
    got = D._mast_source_catalog(o, "F162M")
    assert called["hit"] == ("jw02092-o002", "F162M")           # download was attempted
    assert got.endswith("f150w2-f162m_cat.ecsv")


def test_download_mast_l3_catalog_disabled_by_env(tmp_path, monkeypatch):
    # QA_MAST_DOWNLOAD=0 (set for the whole suite by conftest) fast-fails without any network.
    o = Observation(program="2092", obs="002", target="Cloud E/F", release_field="cloudef",
                    instrument="NIRCam", filters=["F162M"], visits=[], epoch="", notes="")
    assert D._download_mast_l3_catalog(o, "F162M") is None


def test_caption_stage8_distortion():
    cap = D.caption_for(8, dict(stage=8, sw="F212N", f2="F187N", n_stars=39025,
                                resid_rms_mas=1.63, binned_amp90_mas=1.09, per_cell_sem_mas=0.10,
                                null_amp90_mas=0.19, amp90_significance=5.6, frac_gt_20mas=0.076))
    assert "DOCROOT" not in cap and "qa_methods.md#stage8" in cap
    assert "inter-filter" in cap.lower() and "F212N − F187N" in cap and "S/N > 10" in cap
    assert "39025 stars" in cap
    # quotes the shuffled-position null, not "many-σ from SEM"; states the match radius is not zero
    assert "null" in cap.lower() and "5.6×" in cap
    assert "100 mas" in cap and "|Δ| > 20 mas" in cap
    assert "no match radius" not in cap.lower()


def test_caption_stage8_not_applicable_is_not_a_red_flag():
    cap = D.caption_for(8, dict(stage=8, sw="F212N", measurable=False, passed=None))
    assert "not applicable" in cap.lower()
    # a non-defect must NOT be rendered as a red flag / empty plot
    assert "🚩" not in cap and "RED FLAG" not in cap and "plot is empty" not in cap.lower()


def test_caption_stage8_gross_offset_flags_but_describes_map():
    cap = D.caption_for(8, dict(stage=8, sw="F212N", f2="F187N", n_stars=5000,
                                resid_rms_mas=2.0, binned_amp90_mas=25.0, null_amp90_mas=0.5,
                                amp90_significance=50.0, frac_gt_20mas=0.05, red_flag=True,
                                red_flag_reason="gross inter-filter offset: amp90 25.0 mas"))
    assert "🚩" in cap and "gross" in cap.lower()
    # even red-flagged, it describes the rendered map -- not the generic empty-plot caption
    assert "plot is empty" not in cap.lower()


def test_caption_stage8_provisional_describes_perfilter_and_not_flagged():
    cap = D.caption_for(8, dict(stage=8, sw="F212N", f2="F480M", n_stars=110000,
                                resid_rms_mas=14.3, binned_amp90_mas=34.0, amp90_significance=12.0,
                                provisional=True, passed=True))
    assert "provisional" in cap.lower() and "per-filter" in cap.lower()
    assert "not a data defect" in cap.lower()
    assert "🚩" not in cap                                  # a provisional offset is never flagged


def test_interfilter_residuals_bulk_removed_and_gradient(tmp_path, monkeypatch):
    from astropy.table import Table
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    rng = np.random.RandomState(5)
    ra = 266.40 + rng.uniform(0, 0.03, 1000); dec = -28.90 + rng.uniform(0, 0.03, 1000)
    cosd = np.cos(np.radians(-28.9))
    grad = (ra - ra.mean()) * 2000.0                       # RA-dependent ΔRA (distortion-like)
    sc1 = SkyCoord((ra + (60.0 + grad) / 3.6e6 / cosd) * u.deg, dec * u.deg)   # bulk 60 + gradient
    sc2 = SkyCoord(ra * u.deg, dec * u.deg)
    t = Table({"skycoord_f212n": sc1, "skycoord_f187n": sc2,
               "flux_f212n": np.full(len(ra), 1e4), "flux_err_f212n": np.full(len(ra), 1e2),
               "flux_f187n": np.full(len(ra), 1e4), "flux_err_f187n": np.full(len(ra), 1e2)})
    p = str(tmp_path / "cat.fits"); t.write(p)
    monkeypatch.setattr(D, "_catalog_candidates", lambda o: [(p, "m8", 8, 1.0)])
    out = D._interfilter_residuals(object(), "F212N")
    assert out is not None
    rr, dd, dra, dde, f2, name = out
    assert f2 == "F187N"
    assert abs(np.median(dra)) < 2 and abs(np.median(dde)) < 2          # bulk removed
    assert np.corrcoef(rr, dra)[0, 1] > 0.8                             # gradient recovered, no flip


def test_binned_median_2d_orientation():
    # a value that increases with x must land in higher-x cells (guards the imshow orientation)
    x = np.linspace(0, 1, 400); y = np.random.RandomState(7).uniform(0, 1, 400)
    med, xe, ye, cnt = D._binned_median_2d(x, x, y * 0 + x, nb=4)   # vals == x
    col_means = np.nanmean(med, axis=1)                              # mean over y per x-bin
    assert col_means[0] < col_means[-1]                             # increases with x-bin index


def test_binned_median_2d_respects_minn():
    # a cell with fewer than minn (=3 default) points must be EMPTY (NaN, cnt 0), not filled --
    # pins the minn threshold so a mutation minn 3->1 (or 3->2) is caught.  y is constant so all
    # points share one y-bin; x splits 2 into x-bin0 and 4 into x-bin1 (nb=2).
    x = np.array([0.1, 0.2, 0.6, 0.7, 0.8, 0.85])
    y = np.full(6, 0.2)
    med, xe, ye, cnt = D._binned_median_2d(x, y, x, nb=2)
    (i2,) = np.where(cnt.ravel() == 2); (i4,) = np.where(cnt.ravel() == 4)
    assert i2.size == 0                                    # the 2-point cell is dropped at minn=3
    assert i4.size == 1 and np.isfinite(med.ravel()[i4[0]])  # the 4-point cell is kept
    assert np.count_nonzero(cnt) == 1                     # exactly one populated cell


def _two_filter_cat(path, ra, dec, dra_mas, dde_mas, sn=100.0,
                    extra_partner=None, f1="f212n", f2="f187n"):
    """Write a merged-catalogue-like FITS: ``skycoord_<f1>`` is ``skycoord_<f2>`` shifted by
    (dra_mas, dde_mas) on the same rows, with matching flux / flux_err giving S/N ``sn``.
    ``extra_partner`` (e.g. "f480m") adds a farther-wavelength band to test partner selection."""
    from astropy.table import Table
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    ra = np.asarray(ra, float); dec = np.asarray(dec, float)
    dra_mas = np.broadcast_to(np.asarray(dra_mas, float), ra.shape)
    dde_mas = np.broadcast_to(np.asarray(dde_mas, float), ra.shape)
    cosd = float(np.cos(np.radians(np.median(dec))))
    sc2 = SkyCoord(ra * u.deg, dec * u.deg)
    sc1 = SkyCoord((ra + dra_mas / 3.6e6 / cosd) * u.deg, (dec + dde_mas / 3.6e6) * u.deg)
    n = len(ra); flux = np.full(n, 1e4); ferr = flux / np.broadcast_to(sn, (n,))
    cols = {f"skycoord_{f1}": sc1, f"skycoord_{f2}": sc2,
            f"flux_{f1}": flux, f"flux_err_{f1}": ferr.copy(),
            f"flux_{f2}": flux, f"flux_err_{f2}": ferr.copy()}
    if extra_partner:
        cols[f"skycoord_{extra_partner}"] = sc2
    Table(cols).write(path, overwrite=True)


def _grid_radec(n, seed, span=0.03):
    rng = np.random.RandomState(seed)
    return 266.40 + rng.uniform(0, span, n), -28.90 + rng.uniform(0, span, n)


def test_interfilter_residuals_applies_sn_cut(tmp_path, monkeypatch):
    # 1000 well-measured stars + 1000 low-S/N junk; only the 1000 high-S/N rows may survive.
    # A mutation deleting the S/N>10 cut keeps all 2000.
    ra_g, dec_g = _grid_radec(1000, 1); ra_b, dec_b = _grid_radec(1000, 2)
    rng = np.random.RandomState(3)
    ra = np.concatenate([ra_g, ra_b]); dec = np.concatenate([dec_g, dec_b])
    dra = np.concatenate([np.full(1000, 0.5), rng.normal(0, 80, 1000)])
    dde = np.concatenate([np.full(1000, 0.5), rng.normal(0, 80, 1000)])
    sn = np.concatenate([np.full(1000, 100.0), np.full(1000, 5.0)])     # junk < 10
    p = str(tmp_path / "sn.fits"); _two_filter_cat(p, ra, dec, dra, dde, sn=sn)
    monkeypatch.setattr(D, "_catalog_candidates", lambda o: [(p, "m8", 8, 1.0)])
    out = D._interfilter_residuals(object(), "F212N")
    assert out is not None and len(out[0]) == 1000                      # junk excluded


def test_interfilter_residuals_partner_is_nearest_wavelength(tmp_path, monkeypatch):
    # F212N with F187N (25 nm away) and F480M (268 nm away) present -> nearest = F187N.
    # A mutation min()->max() would pick F480M.
    ra, dec = _grid_radec(500, 4)
    p = str(tmp_path / "partner.fits")
    _two_filter_cat(p, ra, dec, 0.0, 0.0, extra_partner="f480m")
    monkeypatch.setattr(D, "_catalog_candidates", lambda o: [(p, "m8", 8, 1.0)])
    out = D._interfilter_residuals(object(), "F212N")
    assert out is not None and out[4] == "F187N"


def test_interfilter_residuals_requires_min_stars(tmp_path, monkeypatch):
    # 150 stars (< the 200 floor) -> None; a mutation 200->0 would return a result.
    ra, dec = _grid_radec(150, 5)
    p = str(tmp_path / "few.fits"); _two_filter_cat(p, ra, dec, 0.0, 0.0)
    monkeypatch.setattr(D, "_catalog_candidates", lambda o: [(p, "m8", 8, 1.0)])
    assert D._interfilter_residuals(object(), "F212N") is None


def _sc_deg(ra, dec):
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    return SkyCoord(np.asarray(ra, float) * u.deg, np.asarray(dec, float) * u.deg)


def test_perfilter_interfilter_residuals_crossmatch_bulk_removed_and_partner(tmp_path, monkeypatch):
    # #247: two SEPARATE per-filter jicama catalogues (generic skycoord + flux, no cross-band merge)
    # are cross-matched, the nearest-wavelength partner is chosen, the bulk offset is removed and a
    # position-dependent gradient survives.
    rng = np.random.RandomState(9)
    ra = 266.40 + rng.uniform(0, 0.03, 800); dec = -28.90 + rng.uniform(0, 0.03, 800)
    cosd = np.cos(np.radians(-28.9))
    grad = (ra - ra.mean()) * 2000.0                       # RA-dependent ΔRA (distortion-like)
    sc_f212 = _sc_deg(ra + (60.0 + grad) / 3.6e6 / cosd, dec)   # bulk 60 + gradient
    base = _sc_deg(ra, dec)
    sn = np.full(800, 100.0)
    pos = {"F212N": (sc_f212, sn), "F187N": (base, sn), "F480M": (base, sn)}
    monkeypatch.setattr(D, "_jicama_perfilter_catalog", lambda o, f: f"/fake/{f}.fits")
    monkeypatch.setattr(D, "_jicama_perfilter_filters", lambda o: ["F187N", "F212N", "F480M"])
    monkeypatch.setattr(D, "_jicama_positions", lambda o, f: pos[f])
    out = D._perfilter_interfilter_residuals(_obs(filt="F212N"), "F212N")
    assert out is not None
    rr, dd, dra, dde, f2, name = out
    assert f2 == "F187N"                                   # nearest-wavelength partner (25 nm, not 268)
    assert abs(np.median(dra)) < 2 and abs(np.median(dde)) < 2      # bulk removed
    assert np.corrcoef(rr, dra)[0, 1] > 0.8               # gradient recovered
    assert " + " in name                                  # two-catalogue provenance label


def test_stage8_perfilter_path_is_provisional_and_not_red_flagged(tmp_path, monkeypatch):
    # #247: with only per-filter jicama catalogues (no cross-band merge), stage 8 measures a
    # PROVISIONAL map from them and must NOT red-flag a gross offset -- the filters are not yet
    # cross-tied, so a large inter-filter offset there is pipeline-progress state, not a WCS defect.
    ra, dec = _grid_radec(3000, 47)
    ran = (ra - ra.mean()) / (0.5 * (ra.max() - ra.min()))
    dra = 40.0 * ran; dde = np.zeros(ra.size)             # ~40 mas gradient -> gross by the merged gate
    monkeypatch.setattr(D, "_interfilter_residuals", lambda o, f: None)
    monkeypatch.setattr(D, "_perfilter_interfilter_residuals",
                        lambda o, f: (ra, dec, dra, dde, "F480M", "a.fits + b.fits"))
    monkeypatch.setattr(D, "OUTDIR", str(tmp_path / "figs"))
    png, m = D.stage8_distortion(_obs(filt="F212N"), "F212N")
    assert os.path.exists(png)
    assert m.get("provisional") is True and "provisional_reason" in m
    assert m["binned_amp90_mas"] > 15.0                   # amplitude IS gross...
    assert not m.get("red_flag")                          # ...but is NOT flagged on the provisional path
    assert m["passed"] is True                            # the measurement still succeeded


def _run_stage8(tmp_path, monkeypatch, ra, dec, dra, dde, sn=100.0):
    p = str(tmp_path / "cat.fits"); _two_filter_cat(p, ra, dec, dra, dde, sn=sn)
    monkeypatch.setattr(D, "_catalog_candidates", lambda o: [(p, "m8", 8, 1.0)])
    monkeypatch.setattr(D, "OUTDIR", str(tmp_path / "figs"))
    png, m = D.stage8_distortion(_obs(filt="F212N"), "F212N")
    assert os.path.exists(png)                                          # it renders
    return m


def test_stage8_recovers_gradient_null_significance_and_amp90(tmp_path, monkeypatch):
    # KNOWN coherent RA gradient (peak ~6 mas) + 1 mas noise on 8000 high-S/N stars (enough per
    # cell that the shuffled-position null is well below the coherent amplitude).
    ra, dec = _grid_radec(8000, 11)
    ran = (ra - ra.mean()) / (0.5 * (ra.max() - ra.min()))             # ~[-1, 1]
    rng = np.random.RandomState(12)
    A = 6.0
    dra = A * ran + rng.normal(0, 1.0, ra.size)
    dde = rng.normal(0, 1.0, ra.size)
    m = _run_stage8(tmp_path, monkeypatch, ra, dec, dra, dde)
    assert m["n_stars"] == 8000
    # amp90 recovers the gradient; a mutation amp90->amp50 would report ~0.5*A (~3 mas), so the
    # >4.2 mas floor separates the 90th percentile (~5.4) from the 50th (~3.0).
    assert m["binned_amp90_mas"] > 4.2
    # significance is the NULL ratio (observed / shuffled-position), and it is >> 1 for real signal
    assert m["null_amp90_mas"] < m["binned_amp90_mas"]
    assert m["amp90_significance"] > 3.0 and m["amp90_p_value"] < 0.1
    assert m["cells_total"] == 144 and m["cells_used"] > 0
    assert m["passed"] is True and not m.get("red_flag")               # a real ~mas term is no defect


def test_stage8_metrics_use_full_population_not_map_clip(tmp_path, monkeypatch):
    # The map clip drops wild-coordinate strays so the grid is not stretched, but the PUBLISHED
    # metrics (n_stars, frac_gt_20mas) must be computed on the FULL population BEFORE that clip:
    # the strays are exactly the nearest-neighbour-ambiguous tail frac_gt_20mas is defined to count,
    # so clipping first would silently divide that QA number by ~5.
    ra, dec = _grid_radec(3000, 71)
    rng = np.random.RandomState(71)
    dra = rng.normal(0, 1.0, 3000); dde = rng.normal(0, 1.0, 3000)
    k = 60                                     # strays ~1° off, each carrying a >20 mas residual
    sra = np.concatenate([ra, ra[:k] + 1.0]); sdec = np.concatenate([dec, dec[:k] + 1.0])
    sdra = np.concatenate([dra, np.full(k, 60.0)]); sdde = np.concatenate([dde, np.full(k, 60.0)])
    m = _run_stage8(tmp_path, monkeypatch, sra, sdec, sdra, sdde)
    assert m["n_stars"] == 3000 + k                     # metrics on the FULL population
    assert m["n_stars_mapped"] == 3000                  # strays dropped from the MAP only
    assert m["n_stars_offfield_clipped"] == k
    assert m["frac_gt_20mas"] > 0.015                   # the stray tail is counted (~0.0196), not ~0
    assert m["passed"] is True


def test_stage8_pure_noise_significance_near_one_and_does_not_flip_pass(tmp_path, monkeypatch):
    # pure Gaussian position noise, NO coherent term: significance ~1, and passed stays True
    # (the gate is measurement-success, not amplitude, so noise cannot flip fail->pass).
    ra, dec = _grid_radec(3000, 21)
    rng = np.random.RandomState(22)
    m = _run_stage8(tmp_path, monkeypatch, ra, dec,
                    rng.normal(0, 3.0, ra.size), rng.normal(0, 3.0, ra.size))
    assert m["amp90_significance"] < 2.0                               # no coherent structure
    assert m["passed"] is True and not m.get("red_flag")


def test_stage8_gate_is_measurement_success_not_amplitude(tmp_path, monkeypatch):
    # adding 5 mas of pure noise onto a modest signal must NOT change passed (old gate flipped
    # fail->pass here); passed reflects populated cells only.
    ra, dec = _grid_radec(3000, 31)
    ran = (ra - ra.mean()) / (0.5 * (ra.max() - ra.min()))
    rng = np.random.RandomState(32)
    base = _run_stage8(tmp_path, monkeypatch, ra, dec, 1.0 * ran, np.zeros(ra.size))
    noisy = _run_stage8(tmp_path, monkeypatch, ra, dec,
                        1.0 * ran + rng.normal(0, 5.0, ra.size), rng.normal(0, 5.0, ra.size))
    assert base["passed"] is True and noisy["passed"] is True          # no fail->pass flip
    assert not base.get("red_flag") and not noisy.get("red_flag")      # 5 mas noise is not gross


def test_stage8_gross_offset_red_flags_but_still_passes_measurement(tmp_path, monkeypatch):
    # a huge (~40 mas peak) inter-filter gradient is a genuine per-filter WCS break -> red_flag,
    # but the MEASUREMENT still succeeded so passed stays True.
    ra, dec = _grid_radec(3000, 41)
    ran = (ra - ra.mean()) / (0.5 * (ra.max() - ra.min()))
    m = _run_stage8(tmp_path, monkeypatch, ra, dec, 40.0 * ran, np.zeros(ra.size))
    assert m["binned_amp90_mas"] > 15.0
    assert m.get("red_flag") is True and m["passed"] is True


def test_stage8_too_few_populated_cells_does_not_pass(tmp_path, monkeypatch):
    # >=200 stars but crammed into two tight clusters -> only 2 of 144 cells populated, so the
    # measurement did not really sample the field: passed must be False (kills a forced passed=True,
    # since every well-sampled case legitimately passes).
    rng = np.random.RandomState(61)
    jit = lambda: rng.uniform(-1e-5, 1e-5, 100)
    ra = np.concatenate([266.40 + jit(), 266.43 + jit()])
    dec = np.concatenate([-28.90 + jit(), -28.87 + jit()])
    rng2 = np.random.RandomState(62)
    m = _run_stage8(tmp_path, monkeypatch, ra, dec,
                    rng2.normal(0, 0.5, ra.size), rng2.normal(0, 0.5, ra.size))
    assert m["cells_used"] < 3
    assert m["passed"] is False


def test_stage8_not_applicable_state_is_not_a_red_flag(tmp_path, monkeypatch):
    # single-filter catalogue: no second band -> not-applicable, NOT a red flag and NOT passed.
    from astropy.table import Table
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    ra, dec = _grid_radec(300, 51)
    p = str(tmp_path / "single.fits")
    Table({"skycoord_f212n": SkyCoord(ra * u.deg, dec * u.deg),
           "flux_f212n": np.full(ra.size, 1e4),
           "flux_err_f212n": np.full(ra.size, 1e2)}).write(p, overwrite=True)
    monkeypatch.setattr(D, "_catalog_candidates", lambda o: [(p, "m8", 8, 1.0)])
    monkeypatch.setattr(D, "_perfilter_interfilter_residuals", lambda o, f: None)  # no jicama fallback
    monkeypatch.setattr(D, "OUTDIR", str(tmp_path / "figs"))
    png, m = D.stage8_distortion(_obs(filt="F212N"), "F212N")
    assert os.path.exists(png)
    assert m.get("measurable") is False
    assert m.get("passed") is None                                     # distinct from True/False
    assert not m.get("red_flag")                                       # a non-defect is not flagged




def test_spitzer_for_miri_band_selection(monkeypatch):
    monkeypatch.setattr(D.os.path, "exists", lambda p: True)     # pretend both mosaics present
    assert "IRAC" in D._spitzer_for_miri("F770W")[0]             # 7.7 um -> IRAC 8 um
    assert "MIPS" in D._spitzer_for_miri("F2100W")[0]            # 21 um -> MIPS 24 um
    assert "MIPS" in D._spitzer_for_miri("F2550W")[0]
    assert D._spitzer_for_miri("F999X") is None                  # unknown filter


def test_miri_caption_variants():
    full = D._miri_caption(dict(filt="F2550W", spitzer="gc_mosaic_MIPSGAL.fits",
                                spitzer_footprint_matched=True,
                                sat_median=0.012, sat_max=0.02, sat_n_frames=72, sat_kind="_rate"),
                           "JWST-GC/data-qa")
    assert "MIRI F2550W basics" in full and "Spitzer" in full and "saturation mask" in full
    assert "same footprint" in full and "72" in full and "_rate" in full
    assert "qa_methods.md#stagemiri" in full
    # an un-matched footprint (reproject failed or off-coverage) must NOT claim a shared footprint
    unmatched = D._miri_caption(dict(filt="F1500W", spitzer="mips.fits",
                                     spitzer_footprint_matched=False), "JWST-GC/data-qa")
    assert "same footprint" not in unmatched and "not matched" in unmatched
    rf = D._miri_caption(dict(filt="F770W", red_flag=True, red_flag_reason="no MIRI i2d on disk"),
                         "JWST-GC/data-qa")
    assert rf.startswith("🚩") and "no MIRI i2d" in rf
    # the caption names the image it actually shows: a reduced mosaic is not labelled MAST (#163)
    assert "MAST i2d image" in D._miri_caption(dict(filt="F770W", i2d_source="mast"),
                                               "JWST-GC/data-qa")
    assert "Reduced i2d image" in D._miri_caption(dict(filt="F770W", i2d_source="reduced"),
                                                  "JWST-GC/data-qa")


def test_miri_i2d_pathing(tmp_path, monkeypatch):
    monkeypatch.setattr(D, "BASE", str(tmp_path))
    md = tmp_path / "brick" / "mastDownload"; md.mkdir(parents=True)
    (md / "jw02221-o001_t001_miri_f2550w_i2d.fits").write_text("")
    o = Observation(program="2221", obs="001", target="Brick", release_field="brick",
                    instrument="MIRI", filters=["F2550W"], visits=[], epoch="", notes="")
    assert D._miri_i2d(o, "F2550W").endswith("miri_f2550w_i2d.fits")
    assert D._miri_i2d(o, "F1800W") is None


def test_miri_i2d_recursive_and_tile_token(tmp_path, monkeypatch):
    """L3 products stage under mastDownload/JWST/<product>/ and the tile token is not always
    _t001_ -- the finder must recurse and accept _t002_/_t003_."""
    monkeypatch.setattr(D, "BASE", str(tmp_path))
    sub = tmp_path / "brick" / "mastDownload" / "JWST" / "jw02221-o003_t002_miri_f2550w"
    sub.mkdir(parents=True)
    (sub / "jw02221-o003_t002_miri_f2550w_i2d.fits").write_text("")   # nested + _t002_
    o = Observation(program="2221", obs="003", target="Brick", release_field="brick",
                    instrument="MIRI", filters=["F2550W"], visits=[], epoch="", notes="")
    hit = D._miri_i2d(o, "F2550W")
    assert hit is not None and hit.endswith("jw02221-o003_t002_miri_f2550w_i2d.fits")


def test_miri_i2d_cross_field_fallback(tmp_path, monkeypatch):
    """cloudc's jw02221-o002 mosaic lives in the sibling brick/ tree (cloudc has no mastDownload
    of its own): the field-scoped glob misses it, the cross-field wildcard finds it.  A file from
    a WRONG field must be reachable ONLY through that fallback, never preferred over a scoped hit."""
    monkeypatch.setattr(D, "BASE", str(tmp_path))
    # obs lives under field=cloudc per the registry, but the file is physically under brick/
    brick_sub = tmp_path / "brick" / "mastDownload" / "JWST" / "jw02221-o002_t001_miri_f2550w"
    brick_sub.mkdir(parents=True)
    (brick_sub / "jw02221-o002_t001_miri_f2550w_i2d.fits").write_text("")
    (tmp_path / "cloudc").mkdir()                      # cloudc dir exists but has no mastDownload
    o = Observation(program="2221", obs="002", target="Cloud C", release_field="cloudc",
                    instrument="MIRI", filters=["F2550W"], visits=[], epoch="", notes="")
    hit = D._miri_i2d(o, "F2550W")
    assert hit is not None and hit.endswith("jw02221-o002_t001_miri_f2550w_i2d.fits")

    # PREFER the field-scoped hit: give cloudc its own mastDownload with the same obsid, plus a
    # stray same-obsid file in an unrelated field; the scoped one must win.
    cloudc_sub = tmp_path / "cloudc" / "mastDownload" / "JWST" / "jw02221-o002_t001_miri_f2550w"
    cloudc_sub.mkdir(parents=True)
    (cloudc_sub / "jw02221-o002_t001_miri_f2550w_i2d.fits").write_text("")
    assert "/cloudc/" in D._miri_i2d(o, "F2550W")


def test_saturation_mask_obs_scoped(tmp_path, monkeypatch):
    from astropy.io import fits
    monkeypatch.setattr(D, "BASE", str(tmp_path))
    d = tmp_path / "brick" / "mastDownload" / "JWST" / "jw02221001001_02101_00001_mirimage"
    d.mkdir(parents=True)
    dq = np.zeros((8, 8), dtype=np.int32); dq[0, 0] = 2; dq[1, 1] = 2   # 2 SATURATED pixels of 64
    fits.HDUList([fits.PrimaryHDU(),
                  fits.ImageHDU(np.zeros((8, 8), "float32"), name="SCI"),
                  fits.ImageHDU(dq, name="DQ")]
                 ).writeto(d / "jw02221001001_02101_00001_mirimage_cal.fits")
    o = Observation(program="2221", obs="001", target="Brick", release_field="brick",
                    instrument="MIRI", filters=["F2550W"], visits=[], epoch="", notes="")
    sat = D._saturation_mask(o)
    assert abs(sat["sat_median"] - 2 / 64) < 1e-9 and abs(sat["sat_max"] - 2 / 64) < 1e-9
    assert sat["n_frames"] == 1 and sat["kind"] == "_cal" and "jw02221001001" in sat["source"]
    assert sat["mask"].sum() == 2
    # a DIFFERENT obs (002) must NOT pick up obs-001's cal (the scoping bug)
    o2 = Observation(program="2221", obs="002", target="Brick", release_field="brick",
                     instrument="MIRI", filters=["F2550W"], visits=[], epoch="", notes="")
    assert D._saturation_mask(o2) is None


def test_saturation_mask_aggregates_and_reports_max(tmp_path, monkeypatch):
    """With several readable frames the summary spans them all: median and max of the per-frame
    saturated fraction, n_frames, and the worst frame's mask/name."""
    from astropy.io import fits
    monkeypatch.setattr(D, "BASE", str(tmp_path))
    d = tmp_path / "brick" / "mastDownload" / "JWST" / "obs001"
    d.mkdir(parents=True)
    for i, nsat in enumerate((1, 2, 5)):                 # 3 frames, differing saturation
        dq = np.zeros((8, 8), dtype=np.int32)
        dq.flat[:nsat] = 2
        fits.HDUList([fits.PrimaryHDU(),
                      fits.ImageHDU(np.zeros((8, 8), "float32"), name="SCI"),
                      fits.ImageHDU(dq, name="DQ")]
                     ).writeto(d / f"jw02221001001_0210{i}_00001_mirimage_cal.fits")
    o = Observation(program="2221", obs="001", target="Brick", release_field="brick",
                    instrument="MIRI", filters=["F2550W"], visits=[], epoch="", notes="")
    sat = D._saturation_mask(o)
    assert sat["n_frames"] == 3
    assert abs(sat["sat_median"] - 2 / 64) < 1e-9        # median of (1,2,5)/64 is 2/64
    assert abs(sat["sat_max"] - 5 / 64) < 1e-9
    assert sat["mask"].sum() == 5                        # displayed mask is the worst frame


def _celestial_wcs(crval, crpix, scale_arcsec, rot_deg=0.0):
    """A small 2-D TAN WCS header (for building synthetic mosaics)."""
    from astropy.wcs import WCS
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = list(crval)
    w.wcs.crpix = list(crpix)
    w.wcs.cdelt = [-scale_arcsec / 3600.0, scale_arcsec / 3600.0]
    th = np.deg2rad(rot_deg)
    w.wcs.pc = [[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]]
    return w


def test_miri_overview_reprojects_spitzer(tmp_path, monkeypatch):
    """Smoke test for miri_overview on a synthetic 2-panel figure (MIRI i2d + Spitzer mosaic):
    the figure builds under matplotlib Agg, the Spitzer cutout is reprojected onto the MIRI grid
    (footprint marked matched, output shape == the MIRI shape), and there are 2 axes (MIRI +
    Spitzer, no saturation product on disk)."""
    pytest.importorskip("reproject")
    from astropy.io import fits
    monkeypatch.setattr(D, "BASE", str(tmp_path))
    monkeypatch.setattr(D, "OUTDIR", str(tmp_path / "out"))

    cen = (266.55, -28.72)
    # MIRI i2d: 40x40, 0.11"/pix, rotated ~266 deg (the obs PA), with an exact-zero border
    mwcs = _celestial_wcs(cen, [20, 20], 0.11, rot_deg=266.0)
    md = np.ones((40, 40), "float32")
    yy, xx = np.mgrid[0:40, 0:40]
    md += 50.0 * np.exp(-((xx - 20) ** 2 + (yy - 20) ** 2) / 40.0)
    md[:4, :] = 0.0; md[:, :4] = 0.0                 # zero border -> exercises the _gray non-zero norm
    sub = tmp_path / "brick" / "mastDownload" / "JWST" / "jw02221-o001_t001_miri_f2550w"
    sub.mkdir(parents=True)
    fits.HDUList([fits.PrimaryHDU(),
                  fits.ImageHDU(md, header=mwcs.to_header(), name="SCI")]
                 ).writeto(sub / "jw02221-o001_t001_miri_f2550w_i2d.fits")

    # synthetic MIPSGAL-like mosaic (F2550W -> MIPS 24 um), north-up, coarse, covering the field
    swcs = _celestial_wcs(cen, [50, 50], 2.5, rot_deg=0.0)
    sd = np.ones((100, 100), "float32")
    yy, xx = np.mgrid[0:100, 0:100]
    sd += 20.0 * np.exp(-((xx - 50) ** 2 + (yy - 50) ** 2) / 200.0)
    spath = tmp_path / "mipsgal_mock.fits"
    fits.PrimaryHDU(sd, header=swcs.to_header()).writeto(spath)
    monkeypatch.setattr(D, "SPITZER_MIPS24", str(spath))

    o = Observation(program="2221", obs="001", target="Brick", release_field="brick",
                    instrument="MIRI", filters=["F2550W"], visits=[], epoch="", notes="")

    captured = {}
    orig_save = D._save
    def _cap(fig, name):
        captured["fig"] = fig
        captured["n"] = len(fig.axes)
        return orig_save(fig, name)
    monkeypatch.setattr(D, "_save", _cap)

    png, metrics = D.miri_overview(o)
    assert os.path.exists(png)
    assert metrics.get("passed") is True and not metrics.get("red_flag")
    assert captured["n"] == 2                              # MIRI + Spitzer, no saturation panel
    assert metrics.get("spitzer") == "mipsgal_mock.fits"
    assert metrics.get("spitzer_footprint_matched") is True

    # PIN the PRODUCTION reprojection by inspecting the DRAWN Spitzer panel, not a re-run in the
    # test: the drawn image must be on the MIRI pixel grid.  Deleting reproject_interp from
    # miri_overview leaves panel=cut.data at the coarse Spitzer shape (100x100 here, not 40x40), so
    # this fails -- which the earlier "re-run reproject in the test" form did not.
    spitzer_img = np.asarray(captured["fig"].axes[1].images[0].get_array(), dtype="float32")
    assert spitzer_img.shape == md.shape
    assert np.isfinite(spitzer_img).mean() > 0.5


def test_miri_coverage_measured_on_reprojected_panel(tmp_path, monkeypatch):
    """The footprint-matched gate must read the REPROJECTED panel, not the raw cutout: a Spitzer
    mosaic offset so it barely overlaps the MIRI field leaves the drawn panel mostly NaN, so the
    footprint must NOT be reported as matched (`spitzer_panel_finite_frac` < 0.5)."""
    pytest.importorskip("reproject")
    from astropy.io import fits
    monkeypatch.setattr(D, "BASE", str(tmp_path))
    monkeypatch.setattr(D, "OUTDIR", str(tmp_path / "out"))
    cen = (266.55, -28.72)
    mwcs = _celestial_wcs(cen, [20, 20], 0.11, rot_deg=266.0)
    md = np.ones((40, 40), "float32")
    md += 50.0 * np.exp(-((np.mgrid[0:40, 0:40][1] - 20) ** 2) / 40.0)
    sub = tmp_path / "brick" / "mastDownload" / "JWST" / "jw02221-o001_t001_miri_f2550w"
    sub.mkdir(parents=True)
    fits.HDUList([fits.PrimaryHDU(),
                  fits.ImageHDU(md, header=mwcs.to_header(), name="SCI")]
                 ).writeto(sub / "jw02221-o001_t001_miri_f2550w_i2d.fits")
    # Spitzer mosaic that does not cover the MIRI field: NaN everywhere except a far corner, so the
    # cutout around the MIRI centre -- and thus the reprojected drawn panel -- is essentially all NaN
    # (the sickle-vs-MIPS24 case), even though a raw finite-fraction over the whole array is nonzero.
    swcs = _celestial_wcs(cen, [50, 50], 2.5, rot_deg=0.0)
    sd = np.full((100, 100), np.nan, "float32")
    sd[:12, :12] = 1.0                                    # finite only in a corner, off the field
    spath = tmp_path / "mips_offset.fits"
    fits.PrimaryHDU(sd, header=swcs.to_header()).writeto(spath)
    monkeypatch.setattr(D, "SPITZER_MIPS24", str(spath))
    o = Observation(program="2221", obs="001", target="Brick", release_field="brick",
                    instrument="MIRI", filters=["F2550W"], visits=[], epoch="", notes="")
    _png, metrics = D.miri_overview(o)
    assert metrics.get("spitzer_panel_finite_frac", 1.0) < 0.5
    assert metrics.get("spitzer_footprint_matched") is False


def test_miri_degenerate_i2d_does_not_pass(tmp_path, monkeypatch):
    """A blank/mostly-empty MIRI i2d must NOT read passed=True (the gate has teeth beyond
    'an i2d opened')."""
    from astropy.io import fits
    monkeypatch.setattr(D, "BASE", str(tmp_path))
    monkeypatch.setattr(D, "OUTDIR", str(tmp_path / "out"))
    cen = (266.55, -28.72)
    mwcs = _celestial_wcs(cen, [20, 20], 0.11, rot_deg=0.0)
    md = np.zeros((40, 40), "float32")                    # all-zero -> finite-but-degenerate
    md[20, 20] = 5.0
    md[md == 0] = np.nan                                  # mostly NaN -> finite fraction ~1/1600
    sub = tmp_path / "brick" / "mastDownload" / "JWST" / "jw02221-o001_t001_miri_f2550w"
    sub.mkdir(parents=True)
    fits.HDUList([fits.PrimaryHDU(),
                  fits.ImageHDU(md, header=mwcs.to_header(), name="SCI")]
                 ).writeto(sub / "jw02221-o001_t001_miri_f2550w_i2d.fits")
    monkeypatch.setattr(D, "SPITZER_MIPS24", "/nonexistent")
    o = Observation(program="2221", obs="001", target="Brick", release_field="brick",
                    instrument="MIRI", filters=["F2550W"], visits=[], epoch="", notes="")
    _png, metrics = D.miri_overview(o)
    assert metrics.get("passed") is False
    assert metrics.get("miri_finite_frac", 1.0) < 0.2


def test_miri_overview_unavailable_no_i2d(tmp_path, monkeypatch):
    """No i2d on disk -> a pending figure, available False, passed None, no red flag."""
    monkeypatch.setattr(D, "BASE", str(tmp_path))
    monkeypatch.setattr(D, "OUTDIR", str(tmp_path / "out"))
    (tmp_path / "brick" / "mastDownload").mkdir(parents=True)
    o = Observation(program="2221", obs="009", target="Brick", release_field="brick",
                    instrument="MIRI", filters=["F1800W"], visits=[], epoch="", notes="")
    png, metrics = D.miri_overview(o)
    assert os.path.exists(png)
    assert (metrics.get("available") is False and metrics.get("passed") is None
            and not metrics.get("red_flag"))


# --------------------------------------------------------------------------- input provenance
def test_used_records_absolute_paths_and_dedupes(tmp_path):
    a = tmp_path / "one.fits"; a.write_text("x")
    with D._recording_inputs() as rec:
        D._used(str(a), "role A")
        D._used(str(a), "role A")            # same (role, path) -> recorded once
        D._used(str(a), "role B")            # a second role for the same file is its own entry
    assert rec == [("role A", os.path.abspath(str(a))), ("role B", os.path.abspath(str(a)))]


def test_used_skips_paths_that_do_not_exist(tmp_path):
    # Recording a file that was never opened is a false provenance claim.  _used is called at the
    # READ, and several read sites are inside a try/except that a missing file lands in.
    with D._recording_inputs() as rec:
        D._used(str(tmp_path / "absent.fits"), "role")
        D._used(None, "role")
        D._used("", "role")
    assert rec == []


def test_used_returns_the_path_unchanged(tmp_path):
    # _used wraps a read in place, so it must be transparent.
    a = tmp_path / "one.fits"; a.write_text("x")
    with D._recording_inputs():
        assert D._used(str(a), "role") == str(a)
        assert D._used(None, "role") is None


def test_recording_inputs_restores_the_outer_collector(tmp_path):
    a = tmp_path / "one.fits"; a.write_text("x")
    b = tmp_path / "two.fits"; b.write_text("x")
    with D._recording_inputs() as outer:
        D._used(str(a), "outer")
        with D._recording_inputs() as inner:
            D._used(str(b), "inner")
        assert [r for r, _ in inner] == ["inner"]
        D._used(str(b), "outer again")
    assert [r for r, _ in outer] == ["outer", "outer again"]


def test_build_stage_attaches_inputs(monkeypatch, tmp_path):
    # The choke point must attach provenance whatever the stage returns.
    a = tmp_path / "m.fits"; a.write_text("x")

    def fake(o, sw, lw):
        D._used(str(a), "SW mosaic")
        return "fig.png", dict(stage=1, passed=True)
    monkeypatch.setattr(D, "stage1_mosaics", fake)
    png, m = D.build_stage(_obs(), 1, "F212N", "F405N")
    assert m["inputs"] == [dict(role="SW mosaic", path=os.path.abspath(str(a)))]


def test_build_stage_keeps_inputs_when_the_stage_raises(monkeypatch, tmp_path):
    # A stage that dies partway is exactly when "which files did it read" matters most.
    a = tmp_path / "m.fits"; a.write_text("x")

    def boom(o, sw, lw):
        D._used(str(a), "SW mosaic")
        raise ValueError("nope")
    monkeypatch.setattr(D, "stage1_mosaics", boom)
    with pytest.raises(ValueError):
        D.build_stage(_obs(), 1, "F212N", "F405N")
    assert D._LAST_FAILED_INPUTS == [dict(role="SW mosaic", path=os.path.abspath(str(a)))]


def test_inputs_block_lists_every_path_when_the_set_is_small():
    inputs = [dict(role="reference", path="/data/brick/catalogs/refcat.fits"),
              dict(role="mosaic", path="/data/brick/F212N/pipeline/f212n_i2d.fits")]
    blk = D._inputs_block(dict(inputs=inputs))
    assert "Files read for this stage (2)" in blk
    for d in inputs:                                  # dir + name reconstructs the full path
        assert f"`{os.path.dirname(d['path'])}/`" in blk
        assert f"`{os.path.basename(d['path'])}`" in blk
    assert "not listed here" not in blk               # nothing summarised at this size
    assert "metrics/<obsid>.json" not in blk


def test_inputs_block_says_so_when_it_summarises():
    # A silent truncation would read as "these are all the files the stage used".
    inputs = [dict(role="per-exposure daophot", path=f"/data/brick/F212N/exp{i:05d}.fits")
              for i in range(200)]
    blk = D._inputs_block(dict(inputs=inputs))
    assert "Files read for this stage (200)" in blk   # the COUNT is never truncated
    assert "per-exposure daophot** — 200 files" in blk
    assert "196 more not listed here" in blk
    assert "metrics/<obsid>.json" in blk              # where the complete list lives
    assert "`exp00000.fits`" in blk and "`exp00199.fits`" in blk   # first and last shown
    assert len(blk) < 2000                            # and it stays small enough to post


def test_inputs_block_is_empty_without_inputs():
    assert D._inputs_block(dict(stage=4)) == ""
    assert D._inputs_block(dict(stage=4, inputs=[])) == ""


def test_caption_for_appends_the_inputs_block():
    m = dict(stage=4, offset_med_mas=1.0, n_cells=16, offset_scatter_mas=2.0,
             bulk_source="histogram", inputs=[dict(role="reference", path="/data/ref.fits")])
    cap = D.caption_for(4, m)
    assert "Files read for this stage (1)" in cap and "`/data/`" in cap


def test_inputs_block_headline_counts_distinct_files_not_reads():
    # Stage 5 reads the same 192 per-exposure catalogs for three different purposes.  "Files read
    # for this stage" must be the number of FILES; the per-role counts then sum to more, and the
    # block says why rather than leaving the reader to notice the mismatch.
    inputs = ([dict(role="module positions", path=f"/d/exp{i}.fits") for i in range(5)]
              + [dict(role="S/N cut", path=f"/d/exp{i}.fits") for i in range(5)])
    blk = D._inputs_block(dict(inputs=inputs))
    assert "Files read for this stage (5)" in blk
    assert "module positions** — 5 files" in blk and "S/N cut** — 5 files" in blk
    assert "5 of the entries above are the same file read for a second purpose" in blk
    # and a set with no repeats says nothing about it
    solo = D._inputs_block(dict(inputs=[dict(role="a", path="/d/x.fits")]))
    assert "same file read for a second purpose" not in solo


def test_inputs_block_falls_back_to_one_line_per_role_when_it_would_be_too_large():
    # The per-directory sampling bounds each directory, not the number of ROLES.  A comment that
    # exceeds GitHub's limit fails to post and carries no provenance at all, so past a ceiling the
    # block drops to one line per role and says where the full list is.
    inputs = [dict(role=f"role {r}", path=f"/data/field/dir{r}/file{i}.fits")
              for r in range(400) for i in range(3)]
    blk = D._inputs_block(dict(inputs=inputs))
    assert len(blk) < 65536
    assert "Files read for this stage (1200)" in blk
    assert "Too many to list individually here" in blk
    assert "metrics/<obsid>.json" in blk
    assert "- **role 0** — 3 files in `/data/field/dir0/`" in blk


# --------------------------------------------------------------------- STAGE 10 (JWST1PASS)
_XYMEEE_HEADER = (
    "#    xbar        ybar        mbar        xsig        ysig        msig        qbar    Nf  Ng  Nm\n")


def _write_matchup(path, rows):
    """Write a MATCHUP.XYMEEE with a comment header and whitespace columns
    (x y m ex ey em q Nf Ng Nm)."""
    with open(path, "w") as fh:
        fh.write(_XYMEEE_HEADER)
        for r in rows:
            fh.write("  " + "  ".join(f"{v:.4f}" if i < 7 else f"{int(v):d}"
                                      for i, v in enumerate(r)) + "\n")


def _good_rows(n=200):
    # bright-to-faint instrumental mags, small position/mag RMS, found in all 6 exposures
    import numpy as _np
    mags = _np.linspace(-14.0, -4.0, n)
    return [(1000.0 + i, 2000.0 + i, float(mm), 0.01, 0.011, 0.02, 0.13, 6, 6, 6)
            for i, mm in enumerate(mags)]


def test_read_matchup_xymeee_drops_sentinels_and_unmeasured(tmp_path):
    p = tmp_path / "MATCHUP.XYMEEE"
    rows = _good_rows(120) + [
        (0.0, 0.0, 0.0, 0.01, 0.01, 0.02, 0.13, 1, 1, 1),        # mbar==0 unmeasured
        (5.0, 5.0, -6.0, 9.0, 9.0, 9.0, 9.999, 6, 6, 6),         # position/mag RMS sentinels
        (6.0, 6.0, -6.0, 0.01, 0.01, 0.02, 9.999, 6, 6, 6),      # qfit-only sentinel (good X/Y/mag)
        (7.0, 7.0, -6.0, 0.01, 0.01, 0.02, 0.13, 1, 1, 1),       # Ng<2 single exposure
    ]
    _write_matchup(str(p), rows)
    d = D._read_matchup_xymeee(str(p))
    assert d is not None
    assert d["m"].size == 120                                    # only the good rows survive
    assert (d["m"] < 0).all() and (d["Ng"] >= 2).all()
    assert d["ex"].max() < D._XYMEEE_SENTINEL
    assert d["q"].max() < D._XYMEEE_QFIT_SENTINEL                # qfit 9.999 rows dropped too


def test_read_matchup_xymeee_none_when_too_few(tmp_path):
    p = tmp_path / "MATCHUP.XYMEEE"
    _write_matchup(str(p), _good_rows(10))                       # < 50 usable rows
    assert D._read_matchup_xymeee(str(p)) is None


def test_jwst1pass_matchup_locator(tmp_path, monkeypatch):
    monkeypatch.setitem(D._JWST1PASS_ROOTS, "brick", str(tmp_path))
    o = _obs(field="brick", filt="F182M")
    assert D._jwst1pass_matchup(o, "F182M") is None             # nothing on disk yet
    d = tmp_path / "brick" / "jwst1pass" / "F182M"; d.mkdir(parents=True)
    (d / "MATCHUP.XYMEEE").write_text("# empty\n")
    assert D._jwst1pass_matchup(o, "F182M") == str(d / "MATCHUP.XYMEEE")


def test_jwst1pass_psfperts_locator_and_figure(tmp_path, monkeypatch):
    from astropy.io import fits
    monkeypatch.setitem(D._JWST1PASS_ROOTS, "brick", str(tmp_path))
    o = _obs(field="brick", obs="001", filt="F182M")
    assert D._jwst1pass_psfperts(o, "F182M") == []               # nothing on disk yet
    base = tmp_path / "brick" / "jwst1pass" / "F182M" / "o001"
    for det in ("NRCA1", "NRCB4", "NRCA2"):
        d = base / det; d.mkdir(parents=True)
        img = np.full((171, 570), -0.1, "float32"); img[40:120, 40:300] = 0.01   # fill + interior
        fits.PrimaryHDU(img).writeto(str(d / "LOG.psfperts.fits"))
    pp = D._jwst1pass_psfperts(o, "F182M")
    assert [det for det, _ in pp] == ["NRCA1", "NRCA2", "NRCB4"]  # sorted by detector
    # obs-scoped: a different obs sees nothing
    assert D._jwst1pass_psfperts(_obs(field="brick", obs="004", filt="F182M"), "F182M") == []
    # one full-width row draws, and interior rms EXCLUDES the +/-0.1 fill (0.01 interior -> ~0.01)
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _, a = plt.subplots()
    im, rms = D._draw_psfperts_row(a, pp[0][1], pp[0][0], "F182M")
    assert im is not None and abs(rms - 0.01) < 1e-3


def test_jwst1pass_matchup_per_obs_match_layout(tmp_path, monkeypatch):
    # the real product layout is {FILT}/o{obs}/match/MATCHUP.XYMEEE (issue: stage 10 red-flagged
    # every field because the resolver only globbed one level under {FILT})
    monkeypatch.setitem(D._JWST1PASS_ROOTS, "brick", str(tmp_path))
    monkeypatch.delenv("QA_JWST1PASS_DIR", raising=False)
    o = _obs(field="brick", obs="001", filt="F182M")
    assert D._jwst1pass_matchup(o, "F182M") is None
    d = tmp_path / "brick" / "jwst1pass" / "F182M" / "o001" / "match"; d.mkdir(parents=True)
    (d / "MATCHUP.XYMEEE").write_text("# empty\n")
    assert D._jwst1pass_matchup(o, "F182M") == str(d / "MATCHUP.XYMEEE")
    # obs-scoped: a different obs of the same filter must NOT pick up o001's product
    assert D._jwst1pass_matchup(_obs(field="brick", obs="004", filt="F182M"), "F182M") is None


def test_jwst1pass_matchup_env_override(tmp_path, monkeypatch):
    (tmp_path / "MATCHUP.XYMEEE").write_text("# here\n")
    monkeypatch.setenv("QA_JWST1PASS_DIR", str(tmp_path))
    assert D._jwst1pass_matchup(_obs(field="brick", filt="F182M"), "F182M") == \
        str(tmp_path / "MATCHUP.XYMEEE")


def test_stage10_unavailable_without_product(tmp_path, monkeypatch):
    monkeypatch.setitem(D._JWST1PASS_ROOTS, "brick", str(tmp_path))
    monkeypatch.delenv("QA_JWST1PASS_DIR", raising=False)
    png, m = D.stage10_photometric_consistency(_obs(field="brick", filt="F182M"), "F182M", None)
    assert m.get("available") is False and m.get("passed") is None and not m.get("red_flag")
    assert "no JWST1PASS" in m["na_reason"]
    cap = D.caption_for(10, m)
    assert "pending" in cap and "not yet on disk" in cap


def test_stage10_measures_consistency(tmp_path, monkeypatch):
    monkeypatch.setenv("QA_JWST1PASS_DIR", str(tmp_path))
    _write_matchup(str(tmp_path / "MATCHUP.XYMEEE"), _good_rows(300))
    png, m = D.stage10_photometric_consistency(_obs(field="brick", filt="F182M"), "F182M", None)
    assert m["passed"] and m["filter"] == "F182M"
    assert m["n_stars"] == 300 and m["n_exposures"] == 6
    # x RMS 0.01 META-pixel * 32 mas = 0.32 mas floor; mag RMS floor ~0.02
    assert abs(m["x_rms_floor_mas"] - 0.32) < 0.05
    assert abs(m["mag_rms_floor"] - 0.02) < 0.01
    cap = D.caption_for(10, m)
    assert "across-exposure consistency (F182M)" in cap and "mas/META-pixel" in cap
    assert "meta_scale_assumed_sw" not in m                       # SW filter -> no LW-scale warning


def test_stage10_flags_sw_meta_scale_on_lw_filter(tmp_path, monkeypatch):
    # _META_PIX_MAS is the SW grid; an LW MATCHUP must be flagged, not silently converted 2x small.
    monkeypatch.setenv("QA_JWST1PASS_DIR", str(tmp_path))
    _write_matchup(str(tmp_path / "MATCHUP.XYMEEE"), _good_rows(300))
    o = _obs(field="brick", filt="F405N")
    png, m = D.stage10_photometric_consistency(o, "F405N", None)
    assert m["passed"] and m["filter"] == "F405N"
    assert m.get("meta_scale_assumed_sw") is True
    assert "unconfirmed" in D.caption_for(10, m) and "SW" in D.caption_for(10, m)


# --------------------------------------------------------------------- STAGE 11 (effective PSF)
def _write_peppar_frame(path, n=400, qfit=5.5, seed=0):
    """A peppar per-frame *_iter1_cat.fits with m/x_fit/y_fit/qfit; qfit constant per exposure."""
    from astropy.table import Table
    rng = np.random.default_rng(seed)
    Table({"m": np.linspace(-9.0, -3.0, n),
           "x_fit": rng.uniform(20, 2020, n), "y_fit": rng.uniform(20, 2020, n),
           "qfit": np.full(n, qfit)}).write(str(path), overwrite=True)


def test_exposure_qfit_flags_the_streaked_exposure(tmp_path, monkeypatch):
    # A streaked exposure fits the PSF far worse -> its qfit spikes above the run baseline.
    monkeypatch.setitem(D._PEPPAR_ROOTS, "brick", str(tmp_path))
    det = tmp_path / "brick" / "peppar" / "F212N" / "NRCA1"; det.mkdir(parents=True)
    for k, q in enumerate([5.5, 5.6, 16.0, 5.4, 5.5], start=1):     # exp 3 is the streaked one
        _write_peppar_frame(det / f"jw02221001001_02101_0000{k}_nrca1_cal_brick_iter1_cat.fits",
                            qfit=q, seed=k)
    o = Observation(program="2221", obs="001", target="Brick", release_field="brick",
                    instrument="NIRCam", filters=["F212N"], visits=[], epoch="", notes="")
    qf = D._exposure_qfit(o, "F212N")
    assert len(qf) == 5
    meds = {e.split("_")[-1]: v[0] for e, v in qf.items()}
    assert abs(meds["00003"] - 16.0) < 0.1 and abs(meds["00001"] - 5.5) < 0.1
    # baseline is the median across exposures (~5.5); only exp 3 exceeds 2x it
    base = float(np.median([v[0] for v in qf.values()]))
    over = [e for e, v in meds.items() if v > D._EPSF_QFIT_STREAK_FACTOR * base]
    assert over == ["00003"]


def test_effective_psf_builds_stamp_and_guards(tmp_path):
    from astropy.io import fits
    rng = np.random.default_rng(1)
    img = rng.normal(0.0, 1.0, (400, 400)).astype("float32")
    yy, xx = np.mgrid[0:23, 0:23]
    g = np.exp(-((xx - 11) ** 2 + (yy - 11) ** 2) / (2 * 1.5 ** 2))
    for _ in range(40):                                            # plant 40 well-separated bright stars
        cx, cy = rng.integers(40, 360), rng.integers(40, 360)
        img[cy - 11:cy + 12, cx - 11:cx + 12] += 500.0 * g
    cal = tmp_path / "exp_cal.fits"
    fits.HDUList([fits.PrimaryHDU(),
                  fits.ImageHDU(img, name="SCI")]).writeto(str(cal), overwrite=True)
    stamp, n = D._effective_psf(str(cal))
    assert stamp is not None and n >= 10
    assert stamp.shape == (2 * D._EPSF_HALF + 1, 2 * D._EPSF_HALF + 1)
    assert np.unravel_index(int(np.argmax(stamp)), stamp.shape) == (D._EPSF_HALF, D._EPSF_HALF)
    # a blank frame yields no stamp, not a crash
    fits.HDUList([fits.PrimaryHDU(),
                  fits.ImageHDU(np.zeros((400, 400), "float32"), name="SCI")]).writeto(
        str(tmp_path / "blank.fits"), overwrite=True)
    assert D._effective_psf(str(tmp_path / "blank.fits")) == (None, 0)


def test_effective_psf_excludes_saturated_via_dq(tmp_path):
    # Every star's core flagged SATURATED in the DQ plane -> all excluded -> no stamp (not a crash).
    from astropy.io import fits
    rng = np.random.default_rng(2)
    img = rng.normal(0.0, 1.0, (400, 400)).astype("float32")
    dq = np.zeros((400, 400), "int32")
    yy, xx = np.mgrid[0:23, 0:23]
    g = np.exp(-((xx - 11) ** 2 + (yy - 11) ** 2) / (2 * 1.5 ** 2))
    centres = [(rng.integers(40, 360), rng.integers(40, 360)) for _ in range(40)]
    for cx, cy in centres:
        img[cy - 11:cy + 12, cx - 11:cx + 12] += 500.0 * g
        dq[cy - 1:cy + 2, cx - 1:cx + 2] |= D._DQ_SATURATED          # flag the core saturated
    fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(img, name="SCI"),
                  fits.ImageHDU(dq, name="DQ")]).writeto(str(tmp_path / "sat.fits"), overwrite=True)
    stamp, n = D._effective_psf(str(tmp_path / "sat.fits"))
    assert stamp is None and n < 10                                  # saturated cores all dropped


def test_stage11_unavailable_without_peppar(tmp_path, monkeypatch):
    monkeypatch.setitem(D._PEPPAR_ROOTS, "brick", str(tmp_path))
    png, m = D.stage11_effective_psf(_obs(field="brick", filt="F212N"), "F212N", None)
    assert m.get("available") is False and m.get("passed") is None and not m.get("red_flag")
    assert "no peppar" in m["na_reason"]
    assert "pending" in D.caption_for(11, m)


def _write_o_frames(det_dir, prog, obs, qfits):
    # peppar frames for one observation: filenames jw<prog><obs>001_02101_0000k_nrca1_cal_..._iter1_cat
    for k, q in enumerate(qfits, start=1):
        _write_peppar_frame(det_dir / f"jw{int(prog):05d}{obs}001_02101_0000{k}_nrca1_cal_"
                                      f"brick_iter1_cat.fits", qfit=q, seed=int(obs) * 10 + k)


def test_stage11_flags_streak_and_pins_passed(tmp_path, monkeypatch):
    # A streaked exposure (qfit spike) must set passed=False -- and the grid must be SCOPED to this
    # obs, not pull a neighbouring obs's exposures from the same peppar filter directory.
    monkeypatch.setitem(D._PEPPAR_ROOTS, "brick", str(tmp_path))
    monkeypatch.setattr(D, "_effective_psf", lambda cal, **kw: (np.ones((23, 23)), 30))  # skip cal I/O
    det = tmp_path / "brick" / "peppar" / "F212N" / "NRCA1"; det.mkdir(parents=True)
    _write_o_frames(det, 2221, "001", [5.5, 5.6, 16.0, 5.4])        # THIS obs: exp3 streaked
    _write_o_frames(det, 2221, "002", [5.5, 5.5, 5.5, 5.5, 5.5])    # a DIFFERENT obs, must be ignored
    o = Observation(program="2221", obs="001", target="Brick", release_field="brick",
                    instrument="NIRCam", filters=["F212N"], visits=[], epoch="", notes="")
    png, m = D.stage11_effective_psf(o, "F212N", None)
    assert m["n_exposures"] == 4                                    # only o001's four, not o002's five
    assert m["streaked_exposures"] == ["00003"] and m["n_streaked"] == 1
    assert m["passed"] is False
    assert "flagged" in D.caption_for(11, m)


def test_exposure_qfit_scoped_to_obs(tmp_path, monkeypatch):
    monkeypatch.setitem(D._PEPPAR_ROOTS, "brick", str(tmp_path))
    det = tmp_path / "brick" / "peppar" / "F212N" / "NRCA1"; det.mkdir(parents=True)
    _write_o_frames(det, 2221, "001", [5.5, 5.6])
    _write_o_frames(det, 2221, "002", [5.5, 5.5, 5.5])
    o = Observation(program="2221", obs="001", target="Brick", release_field="brick",
                    instrument="NIRCam", filters=["F212N"], visits=[], epoch="", notes="")
    qf = D._exposure_qfit(o, "F212N")
    assert len(qf) == 2                                             # only o001's two exposures
    assert all(e.startswith("jw02221001") for e in qf)


# ------------------------------------------------ STAGE 6 clean recompute (exclude bad-PSF exposures)
def test_daophot_key_for_token():
    assert D._daophot_key_for_token("jw02045001001_02101_00004") == "vgroup02101_exp00004"


def test_exclude_frames_drops_flagged_exposures():
    cats = ["/x/jw02045001001_02101_00004_nrca1_cal_arches_iter1_cat.fits",
            "/x/jw02045001001_02101_00005_nrca1_cal_arches_iter1_cat.fits"]
    kept = D._exclude_frames(cats, {"jw02045001001_02101_00004"})
    assert kept == [cats[1]]
    assert D._exclude_frames(cats, None) == cats                 # None -> unchanged


def test_streaked_exposures_returns_flagged_token(tmp_path, monkeypatch):
    monkeypatch.setitem(D._PEPPAR_ROOTS, "brick", str(tmp_path))
    det = tmp_path / "brick" / "peppar" / "F212N" / "NRCA1"; det.mkdir(parents=True)
    for k, q in enumerate([5.5, 5.5, 16.0, 5.5], start=1):       # exp3 streaked
        _write_peppar_frame(det / f"jw02221001001_02101_0000{k}_nrca1_cal_brick_iter1_cat.fits",
                            qfit=q, seed=k)
    o = Observation(program="2221", obs="001", target="Brick", release_field="brick",
                    instrument="NIRCam", filters=["F212N"], visits=[], epoch="", notes="")
    assert D._streaked_exposures(o, "F212N") == {"jw02221001001_02101_00003"}


def test_stage6_wrapper_builds_clean_figure_when_streaked(monkeypatch, tmp_path):
    # The wrapper must build a SECOND figure excluding the flagged exposures and expose clean_png.
    def _fake_fig(o, sw, lw, exclude=None, png_suffix=""):
        if exclude:
            return (str(tmp_path / "clean.png"),
                    {"stage": 6, "sw": sw, "lw": lw, "passed": True,
                     "excluded_exposures": sorted(e.split("_")[-1] for e in exclude),
                     "peppar_framestd_floor_mas_f212n": 2.0})
        return str(tmp_path / "main.png"), {"stage": 6, "sw": sw, "lw": lw, "passed": True,
                                            "peppar_framestd_floor_mas_f212n": 2.6}
    monkeypatch.setattr(D, "_stage6_figure", _fake_fig)
    monkeypatch.setattr(D, "_streaked_exposures",
                        lambda o, f: {"jw_00004"} if f == "F212N" else set())
    png, m = D.stage6_astrom_error(_obs(field="arches", filt="F212N"), "F212N", "F323N")
    assert png.endswith("main.png")
    assert m["clean_png"].endswith("clean.png")
    assert m["excluded_exposures"] == ["00004"]
    assert m["clean_peppar_framestd_floor_mas_f212n"] == 2.0
    cap = D.caption_for("6clean", m)
    assert "EXCLUDING bad-PSF" in cap and "00004" in cap and "2.60 → **2.00 mas**" in cap


def test_stage6_wrapper_no_clean_when_no_streak(monkeypatch, tmp_path):
    monkeypatch.setattr(D, "_stage6_figure",
                        lambda o, sw, lw, exclude=None, png_suffix="": (str(tmp_path / "m.png"),
                                                                        {"stage": 6, "passed": True}))
    monkeypatch.setattr(D, "_streaked_exposures", lambda o, f: set())
    png, m = D.stage6_astrom_error(_obs(field="brick", filt="F212N"), "F212N", None)
    assert "clean_png" not in m


# ---- STAGE 6 clean recompute: keep "clean" actually clean (mutation guards from the #118 review) ----
def test_peppar_precision_excluding_skips_combo_starlist(tmp_path, monkeypatch):
    # The combined starlist bakes in ALL exposures, so it must NOT be used when excluding.  With a
    # combo present but every per-frame cat belonging to the excluded exposure, exclude=None yields a
    # frame_std (from the combo) while exclude={that token} yields none (combo skipped, nothing left).
    from astropy.table import Table
    monkeypatch.setitem(D._PEPPAR_ROOTS, "brick", str(tmp_path))
    pdir = tmp_path / "brick" / "peppar" / "F212N"; det = pdir / "NRCA1"; det.mkdir(parents=True)
    Table({"m": np.linspace(-8, -3, 120), "x_wcs_std": np.full(120, 0.004),
           "y_wcs_std": np.full(120, 0.004)}).write(str(pdir / "combo_starlist_F212N_NRCA1.fits"),
                                                    overwrite=True)
    _write_peppar_frame(det / "jw02221001001_02101_00004_nrca1_cal_brick_iter1_cat.fits", qfit=5.5)
    o = Observation(program="2221", obs="001", target="Brick", release_field="brick",
                    instrument="NIRCam", filters=["F212N"], visits=[], epoch="", notes="")
    assert "frame_std" in D._peppar_precision(o, "F212N")           # combo used when not excluding
    assert D._peppar_precision(o, "F212N", exclude={"jw02221001001_02101_00004"}) is None  # combo skipped


def test_stage6_figure_excluding_draws_no_rms_jwst(tmp_path, monkeypatch):
    # rms(jwst) comes from the merged all-exposure std, so the clean figure must NOT draw it (it
    # would be contaminated by the excluded exposure).  Assert _internal_pos_rms is not consulted.
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    monkeypatch.setenv("QA_OUTDIR", str(tmp_path))
    n = 300
    sc = SkyCoord(266.4 + np.arange(n) * 1e-4, np.full(n, -28.9), unit="deg")
    pooled = (sc, np.full(n, 1.0), np.full(n, 1.0), np.linspace(-10, -3, n), np.full(n, 1.0))
    monkeypatch.setattr(D, "_pooled_daophot", lambda o, f, exclude=None: pooled)
    monkeypatch.setattr(D, "_vega_zeropoint", lambda *a, **k: None)
    monkeypatch.setattr(D, "_viraccache_path", lambda o: None)
    monkeypatch.setattr(D, "_refcat_path", lambda o: None)
    monkeypatch.setattr(D, "_mosaic_path", lambda o, f: "x.fits")
    monkeypatch.setattr(D, "_obs_epoch", lambda o, p: None)
    monkeypatch.setattr(D, "_peppar_precision", lambda o, f, exclude=None: None)
    calls = []
    monkeypatch.setattr(D, "_internal_pos_rms", lambda o, f: calls.append(f) or None)
    o = _obs(field="brick", filt="F212N")
    _png, m = D._stage6_figure(o, "F212N", None, exclude={"jw_00004"})
    assert calls == [] and not any(k.startswith("rms_jwst_floor") for k in m)   # clean: no rms(jwst)
    calls.clear()
    D._stage6_figure(o, "F212N", None)                             # normal: rms(jwst) consulted
    assert calls == ["F212N"]


# --------------------------------------------------------------------------- _clipped_locus_fit
def _sparse_locus_with_cloud():
    """34 matches on a unit-slope locus, 4 of them displaced into a red mismatch cloud.

    The upstream stage-3 gate admits a field at 30 matches, so a locus this sparse is where the
    sample floor decides what gets reported (issue #97).
    """
    rng = np.random.default_rng(97)
    x = np.linspace(12.0, 18.0, 34)
    y = x + rng.normal(0.0, 0.10, x.size)         # clean locus, slope 1
    y[-4:] += 2.0                                 # the red/mismatch cloud
    return x, y


def _old_clip_loop(x, y):
    """The pre-#97 loop: the sample floor breaks BEFORE the clip is adopted."""
    import astropy.stats as ast
    xf, yf = x, y
    slope, zp = np.polyfit(xf, yf, 1)
    for _ in range(5):
        resid = yf - (slope * xf + zp)
        loc = np.abs(resid) < 3 * ast.mad_std(resid)
        if loc.all() or loc.sum() < 30:
            break
        xf, yf = xf[loc], yf[loc]
        slope, zp = np.polyfit(xf, yf, 1)
    return float(slope), float(ast.mad_std(yf - (slope * xf + zp))), int(len(xf))


def test_clipped_locus_fit_adopts_the_clip_at_the_sample_floor():
    """A sparse locus carrying a mismatch cloud must report the CLIPPED fit.

    The old loop broke on ``loc.sum() < 30`` before reassigning xf/yf, so it reported the slope,
    scatter and n_locus of the set the clip had NOT been applied to.
    """
    x, y = _sparse_locus_with_cloud()
    old_slope, _old_scat, old_n = _old_clip_loop(x, y)
    slope, zp, scat, n_locus, n_unclipped, clip_exit = D._clipped_locus_fit(x, y)
    assert n_unclipped == 34
    assert clip_exit == "floor"                   # stopped on the floor WITH the clip in hand
    assert n_locus < old_n                        # 27 clipped stars, against 31 reported before
    assert abs(slope - 1.0) < abs(old_slope - 1.0)    # 1.015 against 1.064
    assert scat < 0.10


def test_clipped_locus_fit_reports_convergence_separately_from_the_floor():
    """A clean locus converges; the two exits must be distinguishable in the metrics."""
    rng = np.random.default_rng(3)
    x = np.linspace(12.0, 18.0, 400)
    y = x + rng.normal(0.0, 0.05, x.size)
    slope, zp, scat, n_locus, n_unclipped, clip_exit = D._clipped_locus_fit(x, y)
    assert clip_exit in ("converged", "maxiter")
    assert abs(slope - 1.0) < 0.02


def test_clipped_locus_fit_refuses_a_clip_that_strips_its_own_support(monkeypatch):
    """A clip that would leave too few stars to fit is refused, and SAID to be refused."""
    monkeypatch.setattr(D, "LOCUS_CLIP_MIN_FIT", 33)     # the first pass leaves 31
    x, y = _sparse_locus_with_cloud()
    slope, zp, scat, n_locus, n_unclipped, clip_exit = D._clipped_locus_fit(x, y)
    assert clip_exit == "floor-unclipped"
    assert n_locus == n_unclipped == 34                  # nothing was thrown away


# ------------------------------------------------------- split per-observation reduction trees
def _split_field_tree(tmp_path):
    """A field whose observations were split into per-observation reduction trees, HALF migrated:
    the frames and the per-obs catalogues moved to ``gc2211_o023``, the mosaic and the pooled
    five-pointing catalogue stayed under ``gc2211`` (the real gc2211 layout)."""
    base = tmp_path / "gc2211"
    (base / "catalogs").mkdir(parents=True)
    (base / "images-merged").mkdir(parents=True)
    (base / "offsets").mkdir(parents=True)
    _touch(base / "catalogs", "basic_merged_indivexp_photometry_tables_merged_resbgsub_m7.fits")
    _touch(base / "images-merged", "jw02211-o023_t001_nircam_clear-f200w-merged_i2d.fits")
    _touch(base / "offsets", "Offsets_JWST_gc2211_VIRAC2locked.csv")

    split = tmp_path / "gc2211_o023"
    (split / "catalogs").mkdir(parents=True)
    (split / "F200W").mkdir(parents=True)
    _touch(split / "catalogs",
           "basic_merged_indivexp_photometry_tables_merged_resbgsub_m7_o023.fits")
    _touch(split / "catalogs", "gaia_virac2_refcat_epoch2023.71.fits")
    for det in ("nrca1", "nrcb1"):
        _touch(split / "F200W", f"f200w_{det}_o023_visit001_exp1_m3_daophot_basic.fits")
    return base, split


def test_split_field_reads_the_per_observation_tree_and_the_base_field(tmp_path, monkeypatch):
    """A half-migrated field must resolve products from BOTH trees.

    Before this, the field name alone was the path: the five gc2211 observations found the pooled
    catalogue and zero per-exposure catalogues, and all reported the same stage 2/3/4 numbers
    (#94, #119).
    """
    monkeypatch.setattr(D, "BASE", str(tmp_path))
    base, split = _split_field_tree(tmp_path)
    o = _obs(field="gc2211", obs="023")

    assert D._field_roots(o) == [str(split), str(base)]
    # the per-exposure catalogues live ONLY in the split tree
    got = D._daophot_glob(o, "F200W")
    assert len(got) == 2 and all(str(split) in g for g in got)
    # the mosaic lives ONLY in the base field
    assert D._mosaic_path(o, "F200W") == str(base / "images-merged" /
                                             "jw02211-o023_t001_nircam_clear-f200w-merged_i2d.fits")
    # this observation's own catalogue wins over the pooled multi-pointing one
    cands = [os.path.basename(p) for p, _k, _t, _m in D._catalog_candidates(o)]
    assert "basic_merged_indivexp_photometry_tables_merged_resbgsub_m7_o023.fits" in cands
    # the POOLED five-pointing catalogue is out: it is not this observation's
    assert "basic_merged_indivexp_photometry_tables_merged_resbgsub_m7.fits" not in cands
    # the split tree's untokened refcat is this obs's by location
    assert D._refcat_path(o) == str(split / "catalogs" / "gaia_virac2_refcat_epoch2023.71.fits")


def test_split_field_peppar_cal_resolves_into_the_sibling_tree(tmp_path, monkeypatch):
    """peppar catalogues stay in the base field while their cal frames move; stage 11 and the
    peppar half of stage 6 go blank unless the sibling split tree is searched."""
    monkeypatch.setattr(D, "BASE", str(tmp_path))
    base, split = _split_field_tree(tmp_path)
    pdir = base / "peppar" / "F200W" / "NRCA1"
    pdir.mkdir(parents=True)
    cat = pdir / "jw02211023001_02101_00001_nrca1_cal_gc2211_iter1_cat.fits"
    _touch(pdir, cat.name)
    (split / "F200W" / "pipeline").mkdir(parents=True)
    _touch(split / "F200W" / "pipeline", "jw02211023001_02101_00001_nrca1_cal.fits")

    assert D._peppar_cal_for_cat(str(cat)) == str(
        split / "F200W" / "pipeline" / "jw02211023001_02101_00001_nrca1_cal.fits")


def test_field_without_a_split_tree_is_unchanged(tmp_path, monkeypatch):
    """One root, the old behaviour: no field without a ``<field>_o<obs>`` directory moves."""
    monkeypatch.setattr(D, "BASE", str(tmp_path))
    d = tmp_path / "brick" / "F212N"; d.mkdir(parents=True)
    _touch(d, "f212n_nrca1_visit001_exp1_m3_daophot_basic.fits")
    o = _obs(field="brick", obs="001", filt="F212N")
    assert D._field_roots(o) == [str(tmp_path / "brick")]
    assert len(D._daophot_glob(o, "F212N")) == 1


# --------------------------------------------------------------------------- _ab_overlap
def _ab_module_catalogs(true_dra_mas, true_ddec_mas, n=800, seed=95):
    """Two module catalogues of the SAME stars, A displaced from B by a known offset.

    ``true_dra/ddec`` is the shift that moves A onto B, the quantity ``_ab_overlap`` reports.
    """
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    rng = np.random.default_rng(seed)
    ra0, dec0 = 266.5, -28.6
    cosd = float(np.cos(np.radians(dec0)))
    b_ra = ra0 + rng.uniform(-0.01, 0.01, n)
    b_dec = dec0 + rng.uniform(-0.01, 0.01, n)
    a_ra = b_ra - true_dra_mas / 1000.0 / 3600.0 / cosd
    a_dec = b_dec - true_ddec_mas / 1000.0 / 3600.0
    return (SkyCoord(a_ra * u.deg, a_dec * u.deg), SkyCoord(b_ra * u.deg, b_dec * u.deg))


def test_ab_overlap_refines_a_biased_histogram_peak_onto_the_same_stars(monkeypatch):
    """The reported A−B offset must be the peak CORRECTED by the same stars, not the raw peak.

    A histogram peak measured against a catalogue tracing the same clustered field is pulled by a
    correlated wrong-pair background (issue #95).  Inject that pull directly: hand `_ab_overlap` a
    peak that is 6.0/-4.0 mas off truth and check the returned offset comes back to truth, with the
    raw peak still on record.
    """
    from data_qa import astrometry_audit as aa
    true_dra, true_ddec = 12.0, -7.0
    bias_dra, bias_ddec = 6.0, -4.0
    a_sc, b_sc = _ab_module_catalogs(true_dra, true_ddec)

    def _biased_xcorr(a, b, **kw):
        dra, ddec = true_dra + bias_dra, true_ddec + bias_ddec
        return dict(dra=dra, ddec=ddec, off=float(np.hypot(dra, ddec)),
                    npairs=5000, peak_ratio=25.0)

    monkeypatch.setattr(aa, "xcorr", _biased_xcorr)
    ov = D._ab_overlap(a_sc, b_sc)
    assert ov is not None
    assert ov["bulk_source"] == "same-star"
    # the raw peak is 7.2 mas off truth; the refined offset is back on it
    assert abs(ov["peak_off"] - np.hypot(true_dra + bias_dra, true_ddec + bias_ddec)) < 1e-6
    assert abs(ov["dra"] - true_dra) < 0.5 and abs(ov["dde"] - true_ddec) < 0.5
    assert abs(ov["off"] - np.hypot(true_dra, true_ddec)) < 0.5
    # sign check: ADDING the residual median instead of subtracting doubles the error
    wrong = np.hypot(true_dra + 2 * bias_dra, true_ddec + 2 * bias_ddec)
    assert abs(ov["off"] - wrong) > 5.0
    # ...and pin the residual's SENSE, which is what makes the verb "subtract" rather than "add":
    # a_aligned = a + peak, so a peak too large by d leaves +d (not -d) in a_aligned - b_matched.
    # The docstring and docs/qa_methods.md state the formula in these terms; an alignment direction
    # flipped without updating them would keep `off` right by luck here and fail this.
    assert abs(ov["same_star_dra"] - bias_dra) < 0.5
    assert abs(ov["same_star_ddec"] - bias_ddec) < 0.5


def test_ab_overlap_keeps_the_raw_peak_when_too_few_pairs_to_refine(monkeypatch):
    """Below the pair floor the refinement is refused and the histogram peak stands, labelled."""
    from data_qa import astrometry_audit as aa
    true_dra, true_ddec = 12.0, -7.0
    bias_dra, bias_ddec = 6.0, -4.0
    a_sc, b_sc = _ab_module_catalogs(true_dra, true_ddec)

    def _biased_xcorr(a, b, **kw):
        dra, ddec = true_dra + bias_dra, true_ddec + bias_ddec
        return dict(dra=dra, ddec=ddec, off=float(np.hypot(dra, ddec)),
                    npairs=5000, peak_ratio=25.0)

    monkeypatch.setattr(aa, "xcorr", _biased_xcorr)
    monkeypatch.setattr(D, "AB_SAME_STAR_MINPAIRS", 10_000)
    ov = D._ab_overlap(a_sc, b_sc)
    assert ov is not None
    assert ov["bulk_source"] == "histogram"
    assert ov["off"] == ov["peak_off"]


# --------------------------------------------------------------- product globs (issue #163)
def _miri_obs(field="brick", program="2221", obs="001", filts=("F770W",)):
    return Observation(program=program, obs=obs, target="T", release_field=field,
                       instrument="MIRI", filters=list(filts), visits=[], epoch="", notes="")


def test_download_root_follows_the_monitor_declaration(monkeypatch):
    """The QA download root is READ OFF `mast_monitor.DEFAULT_DOWNLOAD_DIR`, not restated.

    Restating it (`f"{BASE}/ops/downloads"`) makes the two declarations able to drift, which is
    exactly how QA stopped seeing auto-downloaded products (#163) -- and a pin that compares a
    restatement to the constant is green under the default even after the constant moves.  So
    MOVE the constant and require the QA root to follow it."""
    from data_qa import mast_monitor
    monkeypatch.delenv("QA_DOWNLOAD_DIR", raising=False)
    monkeypatch.setattr(D, "BASE", "/orange/adamginsburg/jwst")     # the default QA_BASE
    assert D._download_root() == mast_monitor.DEFAULT_DOWNLOAD_DIR

    monkeypatch.setattr(mast_monitor, "DEFAULT_DOWNLOAD_DIR",
                        "/orange/adamginsburg/jwst/ops/downloads2")
    assert D._download_root() == "/orange/adamginsburg/jwst/ops/downloads2"
    # ... and it is still re-rooted onto a moved BASE, so a tmp_path test tree lines up
    monkeypatch.setattr(D, "BASE", "/tmp/qa")
    assert D._download_root() == "/tmp/qa/ops/downloads2"
    # a monitor downloading OUTSIDE the QA base is taken literally, not spliced onto BASE
    monkeypatch.setattr(mast_monitor, "DEFAULT_DOWNLOAD_DIR", "/scratch/dl")
    assert D._download_root() == "/scratch/dl"
    # an explicit --download-dir run overrides both
    monkeypatch.setenv("QA_DOWNLOAD_DIR", "/elsewhere/dl")
    assert D._download_root() == "/elsewhere/dl"


def test_mast_globs_reach_the_monitor_download_tree(tmp_path, monkeypatch):
    """A product the monitor downloaded lands in `<BASE>/ops/downloads/mastDownload/JWST/...`,
    one level below the `{BASE}/*/mastDownload` wildcard, so no MAST-side stage could see it."""
    monkeypatch.setattr(D, "BASE", str(tmp_path))
    (tmp_path / "gc-treasury").mkdir()                    # field tree exists but holds nothing
    sub = (tmp_path / "ops" / "downloads" / "mastDownload" / "JWST"
           / "jw10678-o088_t001_nircam_clear-f212n")
    sub.mkdir(parents=True)
    _touch(sub, "jw10678-o088_t001_nircam_clear-f212n_i2d.fits")
    _touch(sub, "jw10678-o088_t001_nircam_clear-f212n_cat.fits")
    o = Observation(program="10678", obs="088", target="GC Treasury",
                    release_field="gc-treasury", instrument="NIRCam", filters=["F212N"],
                    visits=[], epoch="", notes="")
    got = D._mast_i2d(o, "F212N")
    assert got is not None and got.endswith("clear-f212n_i2d.fits")
    cat = D._mast_l3_catalog(o, "F212N", allow_download=False)
    assert cat is not None and cat.endswith("clear-f212n_cat.fits")

    # MIRI parallel of the same tile, downloaded into the same tree
    msub = (tmp_path / "ops" / "downloads" / "mastDownload" / "JWST"
            / "jw10678-o088_t001_miri_f770w")
    msub.mkdir(parents=True)
    _touch(msub, "jw10678-o088_t001_miri_f770w_i2d.fits")
    mo = _miri_obs(field="gc-treasury", program="10678", obs="088")
    assert D._miri_i2d(mo, "F770W").endswith("jw10678-o088_t001_miri_f770w_i2d.fits")


def test_download_tree_cannot_pull_in_another_observation(tmp_path, monkeypatch):
    """The download tree is field-less and holds every program the monitor fetched, so the added
    root must stay pinned to this obsid -- a sibling obs's mosaic in the same tree is not a hit."""
    monkeypatch.setattr(D, "BASE", str(tmp_path))
    sub = tmp_path / "ops" / "downloads" / "mastDownload" / "JWST" / "other"
    sub.mkdir(parents=True)
    _touch(sub, "jw10678-o089_t001_nircam_clear-f212n_i2d.fits")     # a DIFFERENT observation
    o = Observation(program="10678", obs="088", target="GC Treasury",
                    release_field="gc-treasury", instrument="NIRCam", filters=["F212N"],
                    visits=[], epoch="", notes="")
    assert D._mast_i2d(o, "F212N") is None


def test_miri_i2d_finds_the_locally_reduced_mosaic(tmp_path, monkeypatch):
    """A treasury F770W tile reduced into `<field>/F770W/pipeline/` with no MAST copy anywhere:
    the MIRI stage globbed mastDownload only, so it red-flagged 'no MIRI i2d on disk'."""
    monkeypatch.setattr(D, "BASE", str(tmp_path))
    pipe = tmp_path / "gc-treasury" / "F770W" / "pipeline"; pipe.mkdir(parents=True)
    _touch(pipe, "jw10678-o088_t003_miri_f770w_i2d.fits")            # non-t001 tile token
    o = _miri_obs(field="gc-treasury", program="10678", obs="088")
    got = D._miri_i2d(o, "F770W")
    assert got is not None and got.endswith("jw10678-o088_t003_miri_f770w_i2d.fits")
    assert D._miri_i2d(o, "F1130W") is None            # a filter with no product stays None


def test_miri_i2d_reduced_layout_skips_outlier_and_residual_products(tmp_path, monkeypatch):
    """`<FILT>/pipeline/` also holds outlier-detection intermediates and photometry model/
    residual mosaics.  `..._f770w_0_o002_outlier_i2d.fits` sorts BEFORE `..._f770w_i2d.fits`,
    so a trailing wildcard would hand the MIRI stage an intermediate as the science mosaic."""
    monkeypatch.setattr(D, "BASE", str(tmp_path))
    pipe = tmp_path / "w51" / "F770W" / "pipeline"; pipe.mkdir(parents=True)
    for name in ("jw06151-o002_t001_miri_f770w_0_o002_outlier_i2d.fits",
                 "jw06151-o002_t001_miri_f770w_15_o002_outlier_i2d.fits",
                 "jw06151-o002_t001_miri_clear-f770w-mirimage_group_m3_"
                 "daophot_basic_mergedcat_residual_i2d.fits",
                 "jw06151-o002_t001_miri_f770w_i2d.fits"):
        _touch(pipe, name)
    o = _miri_obs(field="w51", program="6151", obs="002")
    got = D._miri_i2d(o, "F770W")
    assert os.path.basename(got) == "jw06151-o002_t001_miri_f770w_i2d.fits"
    stem = D._MIRI_REDUCED_STEMS[0].format(obsid=o.obsid, filt="f770w")
    assert [os.path.basename(q) for q in sorted(glob.glob(f"{pipe}/{stem}"))] == \
        ["jw06151-o002_t001_miri_f770w_i2d.fits"]


def test_miri_i2d_prefers_the_mast_delivery_over_the_reduced_mosaic(tmp_path, monkeypatch):
    """The reduced layout is a FALLBACK: where a MAST copy exists it still wins, so no
    observation that resolves today resolves to a different file."""
    monkeypatch.setattr(D, "BASE", str(tmp_path))
    md = tmp_path / "brick" / "mastDownload"; md.mkdir(parents=True)
    _touch(md, "jw02221-o001_t001_miri_f770w_i2d.fits")
    pipe = tmp_path / "brick" / "F770W" / "pipeline"; pipe.mkdir(parents=True)
    _touch(pipe, "jw02221-o001_t001_miri_f770w_i2d.fits")
    got = D._miri_i2d(_miri_obs(field="brick"), "F770W")
    assert "/mastDownload/" in got


def test_miri_obs_from_disk_reads_the_reduced_layout(tmp_path, monkeypatch):
    """Portal-independent MIRI discovery: a locally-reduced tile with no MAST copy."""
    monkeypatch.setattr(D, "BASE", str(tmp_path))
    for filt in ("F770W", "F1130W"):
        pipe = tmp_path / "sickle" / filt / "pipeline"; pipe.mkdir(parents=True)
        _touch(pipe, f"jw03958-o002_t001_miri_{filt.lower()}_i2d.fits")
        _touch(pipe, f"jw03958-o002_t001_miri_{filt.lower()}_4_o002_outlier_i2d.fits")
    o = D._miri_obs_from_disk("3958", "002", base=str(tmp_path))
    assert o is not None and o.instrument == "MIRI"
    assert o.release_field == "sickle"
    assert o.filters == ["F1130W", "F770W"]          # the outlier intermediates add no filter


def test_miri_obs_from_disk_skips_a_field_with_only_byproducts(tmp_path, monkeypatch):
    """A field dir holding only outlier/model/residual mosaics for this obsid must not claim the
    observation: it would return an Observation with no filters and mask the real field.

    Widening the filter regex to also read the cataloging `_data_i2d` stem (#166 review) must not
    widen it to the model/residual mosaics that sit beside it under the SAME `clear-<filt>-
    mirimage_` prefix -- so the decoy dir carries one of each."""
    monkeypatch.setattr(D, "BASE", str(tmp_path))
    junk = tmp_path / "aaa_scratch" / "F770W" / "pipeline"; junk.mkdir(parents=True)
    for name in ("jw03958-o002_t001_miri_f770w_3_o002_outlier_i2d.fits",
                 "jw03958-o002_t001_miri_clear-f770w-mirimage_group_m3_"
                 "daophot_basic_mergedcat_model_i2d.fits",
                 "jw03958-o002_t001_miri_clear-f770w-mirimage_group_m3_"
                 "daophot_basic_mergedcat_residual_i2d.fits",
                 "jw03958-o002_t001_miri_clear-f770w-mirimage_group_m3_"
                 "daophot_basic_mergedcat_residual_smoothed_bg_i2d.fits"):
        _touch(junk, name)
    assert D._miri_filters(sorted(junk.glob("*.fits"))) == []
    pipe = tmp_path / "sickle" / "F770W" / "pipeline"; pipe.mkdir(parents=True)
    _touch(pipe, "jw03958-o002_t001_miri_f770w_i2d.fits")
    o = D._miri_obs_from_disk("3958", "002", base=str(tmp_path))
    assert o is not None and o.release_field == "sickle" and o.filters == ["F770W"]


def test_miri_i2d_finds_the_cataloging_data_mosaic(tmp_path, monkeypatch):
    """Some observations have ONLY the cataloging stage's `_data_i2d` resample under
    `<FILT>/pipeline/` -- sgrb2 `jw05365-o002` F2550W on disk today has no plain
    `jw05365-o002_t001_miri_f2550w_i2d.fits`.  Matching only the reduction stem left it
    unreachable (#166 review B2)."""
    monkeypatch.setattr(D, "BASE", str(tmp_path))
    pipe = tmp_path / "sgrb2" / "F2550W" / "pipeline"; pipe.mkdir(parents=True)
    for name in ("jw05365-o002_t001_miri_clear-f2550w-mirimage_group_m3_"
                 "daophot_basic_mergedcat_model_i2d.fits",
                 "jw05365-o002_t001_miri_clear-f2550w-mirimage_group_m3_"
                 "daophot_basic_mergedcat_residual_i2d.fits",
                 "jw05365-o002_t001_miri_clear-f2550w-mirimage_group_m3_"
                 "daophot_basic_mergedcat_residual_smoothed_bg_i2d.fits",
                 "jw05365-o002_t001_miri_clear-f2550w-mirimage_data_i2d.fits"):
        _touch(pipe, name)
    o = _miri_obs(field="sgrb2", program="5365", obs="002", filts=("F2550W",))
    got = D._miri_i2d(o, "F2550W")
    assert os.path.basename(got) == \
        "jw05365-o002_t001_miri_clear-f2550w-mirimage_data_i2d.fits"
    # EXACTNESS, not sort-luck: `_data_i2d` admits exactly one of the four real names above, so
    # a by-product cannot become the science mosaic by happening to sort ahead of it.  (Today
    # `data` < `group_` < `resbgsub_` lexically, so a loose `*<filt>*_i2d.fits` would return the
    # right file anyway -- and would stop doing so the day a mosaic sorts before `data`.)
    stem = D._MIRI_REDUCED_STEMS[1].format(obsid=o.obsid, filt="f2550w")
    assert [os.path.basename(q) for q in sorted(glob.glob(f"{pipe}/{stem}"))] == \
        ["jw05365-o002_t001_miri_clear-f2550w-mirimage_data_i2d.fits"]
    # ... and it is a FALLBACK: where the reduction's own stage-3 mosaic is there too, it wins
    _touch(pipe, "jw05365-o002_t001_miri_f2550w_i2d.fits")
    assert os.path.basename(D._miri_i2d(o, "F2550W")) == \
        "jw05365-o002_t001_miri_f2550w_i2d.fits"
    # the `_data` stem also declares its filter to `_miri_obs_from_disk`
    os.remove(os.path.join(str(pipe), "jw05365-o002_t001_miri_f2550w_i2d.fits"))
    disk = D._miri_obs_from_disk("5365", "002", base=str(tmp_path))
    assert disk is not None and disk.release_field == "sgrb2" and disk.filters == ["F2550W"]


def test_miri_panel_title_names_the_image_it_shows(tmp_path, monkeypatch):
    """The panel title was hardcoded `MIRI <filt> MAST i2d`, so on the locally-reduced fallback
    the PNG a human opens labelled our own mosaic a MAST delivery (#166 review B1)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from astropy.io import fits
    monkeypatch.setattr(D, "BASE", str(tmp_path))
    monkeypatch.setattr(D, "_spitzer_for_miri", lambda f: None)
    monkeypatch.setattr(D, "_saturation_mask", lambda o: None)
    titles = []

    def _grab(fig, name):
        titles.append(fig.axes[0].get_title()); plt.close(fig); return name
    monkeypatch.setattr(D, "_save", _grab)

    def _make(path):
        hdu = fits.HDUList([fits.PrimaryHDU(),
                            fits.ImageHDU(np.arange(64, dtype="float32").reshape(8, 8),
                                          name="SCI")])
        hdu[1].header.update(dict(CTYPE1="RA---TAN", CTYPE2="DEC--TAN", CRPIX1=4, CRPIX2=4,
                                  CRVAL1=266.0, CRVAL2=-28.9, CDELT1=-3e-5, CDELT2=3e-5))
        hdu.writeto(path, overwrite=True)

    o = _miri_obs(field="w51", program="6151", obs="002")
    pipe = tmp_path / "w51" / "F770W" / "pipeline"; pipe.mkdir(parents=True)
    _make(str(pipe / "jw06151-o002_t001_miri_f770w_i2d.fits"))
    _png, m = D.miri_overview(o)
    assert m["i2d_source"] == "reduced"
    assert titles[-1] == "MIRI F770W reduced i2d"
    assert "MAST" not in titles[-1]

    md = tmp_path / "w51" / "mastDownload" / "JWST" / "p"; md.mkdir(parents=True)
    _make(str(md / "jw06151-o002_t001_miri_f770w_i2d.fits"))
    _png, m = D.miri_overview(o)
    assert m["i2d_source"] == "mast"
    assert titles[-1] == "MIRI F770W MAST i2d"


def test_miri_caption_never_guesses_a_provenance():
    """A metrics JSON written before #163 carries no `i2d_source`, and the no-image red-flag path
    records none either -- the caption must not then assert one."""
    cap = D._miri_caption(dict(filt="F770W"), "JWST-GC/data-qa")
    assert "i2d image" in cap and "MAST" not in cap and "Reduced" not in cap


def test_miri_obs_from_disk_reads_the_monitor_download_tree(tmp_path, monkeypatch):
    """A tile the monitor auto-downloaded that no field tree has a copy of.  The tree is
    field-less, so `release_field` comes from the monitor's own program->field map -- without it
    `_run_miri` reports "portal + on-disk empty" and the observation gets no MIRI stage at all
    (#166 review, non-blocking 3)."""
    from data_qa import mast_monitor
    monkeypatch.setattr(D, "BASE", str(tmp_path))
    monkeypatch.delenv("QA_DOWNLOAD_DIR", raising=False)
    monkeypatch.setattr(mast_monitor, "DEFAULT_DOWNLOAD_DIR",
                        "/orange/adamginsburg/jwst/ops/downloads")
    (tmp_path / "gc-treasury").mkdir()                    # field tree exists but holds nothing
    sub = (tmp_path / "ops" / "downloads" / "mastDownload" / "JWST"
           / "jw10678-o088_t001_miri_f770w")
    sub.mkdir(parents=True)
    _touch(sub, "jw10678-o088_t001_miri_f770w_i2d.fits")
    o = D._miri_obs_from_disk("10678", "088", base=str(tmp_path))
    assert o is not None and o.instrument == "MIRI" and o.filters == ["F770W"]
    assert o.release_field == mast_monitor.field_for(10678, "088") == "gc-treasury"

    # a program the monitor has no field for is NOT guessed at -- no Observation, no wrong field
    assert mast_monitor.field_for(99999, "001") == ""
    assert D._miri_obs_from_disk("99999", "001", base=str(tmp_path)) is None


def test_obs_from_disk_builds_nircam_from_mast_when_no_mosaic(tmp_path, monkeypatch):
    """A delivered tile whose reduce is held has no merged mosaic of ours yet, so the mosaic pass
    finds nothing.  The MAST-only pass then builds the NIRCam observation from the MAST-delivered
    i2d, so the tile still gets a QA issue (JWST-GC/data-qa#161)."""
    monkeypatch.setattr(D, "BASE", str(tmp_path))
    for filt in ("f212n", "f480m"):
        sub = (tmp_path / "gc-treasury" / "mastDownload" / "JWST"
               / f"jw10678-o113_t113_nircam_clear-{filt}")
        sub.mkdir(parents=True)
        _touch(sub, f"jw10678-o113_t113_nircam_clear-{filt}_i2d.fits")
    o = D._obs_from_disk("10678", "113", base=str(tmp_path))
    assert o is not None and o.instrument == "NIRCam"
    assert o.release_field == "gc-treasury"
    assert o.filters == ["F212N", "F480M"]
    assert o.issue_title == "GC Treasury — jw10678-o113 (NIRCam)"


def test_obs_from_disk_prefers_our_mosaic_over_mast(tmp_path, monkeypatch):
    """When our own merged mosaic exists the mosaic pass wins, so a reduced tile keeps its previous
    behaviour and the MAST-only pass never overrides it."""
    monkeypatch.setattr(D, "BASE", str(tmp_path))
    pipe = tmp_path / "gc-treasury" / "F212N" / "pipeline"; pipe.mkdir(parents=True)
    _touch(pipe, "jw10678-o137_t001_nircam_clear-f212n-merged_i2d.fits")
    mast = (tmp_path / "gc-treasury" / "mastDownload" / "JWST"
            / "jw10678-o137_t137_nircam_clear-f480m"); mast.mkdir(parents=True)
    _touch(mast, "jw10678-o137_t137_nircam_clear-f480m_i2d.fits")
    o = D._obs_from_disk("10678", "137", base=str(tmp_path))
    assert o is not None and o.filters == ["F212N"]      # from the mosaic pass, not the MAST i2d


def test_mast_lookups_reach_a_split_field_tree(tmp_path, monkeypatch):
    """`_mast_source_catalog`, `_mast_catalog_positions`, `_mast_i2d` and `_saturation_mask` used
    a literal `{BASE}/{o.field}`, so on a split field (`<field>_o<obs>`, issue #119) they saw only
    the one tree the registry names.  They go through `_field_roots` now, like every other
    lookup, so the split tree and the base field are both reachable."""
    monkeypatch.setattr(D, "BASE", str(tmp_path))
    (tmp_path / "gc2211_o023").mkdir()                   # the split tree: frames went here
    md = (tmp_path / "gc2211" / "mastDownload" / "JWST"      # the MAST copy stayed in the base
          / "jw02211-o023_t001_nircam_clear-f200w")
    md.mkdir(parents=True)
    _touch(md, "jw02211-o023_t001_nircam_clear-f200w_i2d.fits")
    _touch(md, "jw02211-o023_t001_nircam_clear-f200w_cat.ecsv")
    o = Observation(program="2211", obs="023", target="T", release_field="gc2211_o023",
                    instrument="NIRCam", filters=["F200W"], visits=[], epoch="", notes="")
    assert D._field_roots(o)[0].endswith("gc2211_o023")
    assert D._mast_i2d(o, "F200W").endswith("clear-f200w_i2d.fits")
    assert D._mast_source_catalog(o, "F200W").endswith("clear-f200w_cat.ecsv")
# --------------------------------------------------------------------------- STAGE 12 linearity
def test_linearity_fit_flat_is_zero_slope():
    # a constant aperture-minus-PSF offset across brightness = a linear response: ~zero slope, no
    # bright turn-over, baseline at the constant offset.
    rng = np.random.default_rng(0)
    m = np.repeat(np.arange(15.0, 21.0, 0.5), 20)
    dmag = 0.40 + rng.normal(0, 0.002, size=m.size)
    fit = D._linearity_fit(m, dmag)
    assert fit is not None
    assert abs(fit["slope"]) < 0.01
    assert fit["turnover"] is None
    assert abs(fit["baseline"] - 0.40) < 0.02


def test_linearity_fit_detects_bright_turnover():
    # flat at the faint end, departing at the bright end (a saturation roll-over): the turn-over is
    # detected on the bright side and the linear-range slope stays small.
    rng = np.random.default_rng(1)
    centres = np.arange(15.0, 21.0, 0.5)
    parts_m, parts_d = [], []
    for c in centres:
        base = 0.40 if c >= 17.0 else 0.40 + (17.0 - c) * 0.15   # rises going bright
        parts_m.append(np.full(20, c))
        parts_d.append(base + rng.normal(0, 0.002, size=20))
    m = np.concatenate(parts_m); dmag = np.concatenate(parts_d)
    fit = D._linearity_fit(m, dmag)
    assert fit is not None
    assert fit["turnover"] is not None
    assert 16.0 < fit["turnover"] < 17.5          # departure begins around m ~ 16.8
    assert abs(fit["slope"]) < 0.02               # faint (linear) range is flat


def test_linearity_fit_recovers_injected_slope_and_trips_flag():
    # A KNOWN linear trend must be recovered (would fail if the slope were hardcoded to 0), and a
    # slope beyond _LIN_SLOPE_FLAG must set the flag path -- the flag's only exercise.
    rng = np.random.default_rng(2)
    m = np.repeat(np.arange(15.0, 21.0, 0.5), 30)
    for s, expect_flag in [(0.005, False), (0.050, True)]:
        dmag = 0.40 + s * (m - 18.0) + rng.normal(0, 0.002, size=m.size)
        fit = D._linearity_fit(m, dmag)
        assert fit is not None and fit["slope"] is not None
        assert abs(fit["slope"] - s) < 5 * fit["slope_err"]        # trend recovered
        assert abs(fit["slope"] - s) < 0.01                        # and quantitatively close
        assert (abs(fit["slope"]) > D._LIN_SLOPE_FLAG) is expect_flag


def test_linearity_fit_no_turnover_on_pure_trend():
    # A pure global slope with NO saturation feature must NOT report a turn-over: the turn-over is
    # measured against the fitted trend, so a constant slope alone does not trip it.
    rng = np.random.default_rng(3)
    m = np.repeat(np.arange(15.0, 21.0, 0.5), 30)
    dmag = 0.40 + 0.05 * (m - 18.0) + rng.normal(0, 0.002, size=m.size)
    fit = D._linearity_fit(m, dmag)
    assert fit is not None
    assert fit["turnover"] is None
    assert abs(fit["slope"] - 0.05) < 0.01        # the slope still measures the real trend


def _synth_mosaic(tmp_path, name="m.fits", nstars=100):
    """A WCS mosaic with a grid of well-separated Gaussians spanning a range of fluxes; returns
    (path, SkyCoord positions, flux array)."""
    from astropy.io import fits
    from astropy.wcs import WCS
    from astropy.coordinates import SkyCoord
    ny = nx = 520
    yy, xx = np.mgrid[0:ny, 0:nx]
    g = np.linspace(40, nx - 40, 10)
    XX, YY = np.meshgrid(g, g)
    xs = XX.ravel(); ys = YY.ravel()
    flux = np.geomspace(3e3, 3e5, len(xs)); sig = 1.5
    img = np.zeros((ny, nx), "float32")
    for xi, yi, f in zip(xs, ys, flux):
        img += (f / (2 * np.pi * sig ** 2)) * np.exp(-((xx - xi) ** 2 + (yy - yi) ** 2) / (2 * sig ** 2))
    w = WCS(naxis=2)
    w.wcs.crpix = [nx / 2, ny / 2]; w.wcs.cdelt = [-1 / 3600.0, 1 / 3600.0]
    w.wcs.crval = [266.4, -28.7]; w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    mp = str(tmp_path / name)
    fits.HDUList([fits.PrimaryHDU(),
                  fits.ImageHDU(img, header=w.to_header(), name="SCI")]).writeto(mp)
    sc = w.pixel_to_world(xs, ys); sc = SkyCoord(sc.ra, sc.dec)
    return mp, sc, flux


def test_stage12_end_to_end_synthetic(tmp_path, monkeypatch):
    pytest.importorskip("photutils"); pytest.importorskip("scipy")
    monkeypatch.setattr(D, "OUTDIR", str(tmp_path))
    mp, sc, flux = _synth_mosaic(tmp_path)
    monkeypatch.setattr(D, "_psf_flux_positions", lambda o, f: (sc, flux.copy(), "synth.fits"))
    monkeypatch.setattr(D, "_mosaic_path", lambda o, f: mp)
    o = Observation(program="1182", obs="004", target="Brick", release_field="brick",
                    instrument="NIRCam", filters=["F212N", "F444W"], visits=[], epoch="", notes="")
    png, m = D.stage12_photometric_linearity(o, "F212N", "F444W")
    assert not m.get("red_flag")
    assert m["primary_filter"] == "F212N"
    assert set(m["filters_measured"]) == {"F212N", "F444W"}
    # a constant Gaussian shape at every brightness -> the aperture misses the same fraction ->
    # flat aper-minus-PSF -> ~zero linearity slope
    assert abs(m["per_filter"]["F212N"]["slope"]) < 0.03
    # one plot shown by default (primary) + the rest hidden in extra_figures
    assert [f for f, _ in m["extra_figures"]] == ["F444W"]
    assert os.path.exists(png)
    for _f, p in m["extra_figures"]:
        assert os.path.exists(p)


def test_caption_stage12_table_and_primary():
    m = dict(stage=12, sw="F212N", primary_filter="F212N",
             filters_measured=["F212N", "F405N"],
             per_filter={
                 "F212N": dict(slope=0.006, slope_err=0.004, turnover_mag=15.5,
                               aper_corr=0.42, n=1800, flagged=False),
                 "F405N": dict(slope=0.031, slope_err=0.005, turnover_mag=14.0,
                               aper_corr=0.55, n=900, flagged=True),
             },
             slope=0.006, slope_err=0.004, turnover_mag=15.5, aper_corr=0.42, n_isolated=1800,
             n_flagged=1)
    cap = D.caption_for(12, m)
    assert "DOCROOT" not in cap and "qa_methods.md#stage12" in cap
    assert "photometric linearity" in cap
    assert "| filter |" in cap                       # per-filter table present
    assert "F405N" in cap and "🚩" in cap            # flagged filter marked
    assert "+0.006" in cap                           # primary slope reported
    assert "expandable block" in cap                 # says the rest are hidden


def test_details_block_embeds_extra(monkeypatch):
    from data_qa import post_diagnostics as P
    monkeypatch.setattr(P, "upload_asset",
                        lambda repo, token, path, name: f"https://cdn/{name}")
    o = Observation(program="1182", obs="004", target="Brick", release_field="brick",
                    instrument="NIRCam", filters=["F212N"], visits=[], epoch="", notes="")
    block = P._details_block("JWST-GC/data-qa", "tok", o, 12,
                             [("F405N", f"/tmp/{o.obsid}_stage12_F405N.png"),
                              ("F444W", f"/tmp/{o.obsid}_stage12_F444W.png")])
    assert "<details>" in block and "</details>" in block
    assert "Other 2 figure(s): F405N, F444W" in block
    assert "**F405N**" in block and "**F444W**" in block
    assert f"{o.obsid}_stage12_F405N.png" in block          # asset name = the png basename


def test_details_block_asset_name_is_url_safe_with_spaced_label(monkeypatch):
    """The asset name (release-asset URL path) must not carry spaces/parens from a free-text label
    — GitHub rejects control characters in the path (stage 3's 'jicama-m8 vs VIRAC (calibration)')."""
    from data_qa import post_diagnostics as P
    seen = []
    monkeypatch.setattr(P, "upload_asset",
                        lambda repo, token, path, name: seen.append(name) or f"https://cdn/{name}")
    o = Observation(program="10678", obs="132", target="GC Treasury", release_field="gc-treasury",
                    instrument="NIRCam", filters=["F212N"], visits=[], epoch="", notes="")
    block = P._details_block("JWST-GC/data-qa", "tok", o, 3,
                             [("jicama-m8 vs VIRAC (calibration)",
                               f"/tmp/{o.obsid}_stage3_our.png")])
    assert seen == [f"{o.obsid}_stage3_our.png"]           # basename, not the label
    assert " " not in seen[0] and "(" not in seen[0]
    assert "**jicama-m8 vs VIRAC (calibration)**" in block  # label still the visible caption


# --------------------------------------------------------------------------- pagination robustness
def _fake_req(script):
    """Return a stand-in for post_diagnostics._req that yields (200, data, {'Link': link}) for each
    scripted (data, link) in order, so pagination can be exercised without the network."""
    calls = {"n": 0}

    def fake(method, url, token, data=None, headers=None, raw=False, want_headers=False):
        d, link = script[min(calls["n"], len(script) - 1)]
        calls["n"] += 1
        return (200, d, {"Link": link}) if want_headers else (200, d)
    return fake, calls


def test_paged_get_follows_next_cursor(monkeypatch):
    from data_qa import post_diagnostics as P
    page1 = [{"number": i} for i in range(100)]
    page2 = [{"number": 100 + i} for i in range(30)]
    nxt = '<https://api.github.com/x?page=2&after=cur>; rel="next"'
    fake, _ = _fake_req([(page1, nxt), (page2, "")])
    monkeypatch.setattr(P, "_req", fake)
    monkeypatch.setattr(P.time, "sleep", lambda *a: None)
    out = P._paged_get("https://api.github.com/x?page={page}", "tok", "test")
    assert len(out) == 130                         # both pages, cursor followed to the end


def test_paged_get_retries_spurious_empty_next_page(monkeypatch):
    # page 1 advertises a next page; that next page comes back EMPTY once (spurious) then real.
    from data_qa import post_diagnostics as P
    page1 = [{"number": i} for i in range(100)]
    page2 = [{"number": 100 + i} for i in range(20)]
    nxt = '<https://api.github.com/x?page=2&after=cur>; rel="next"'
    fake, _ = _fake_req([(page1, nxt), ([], nxt), (page2, "")])   # empty page2 then real page2
    monkeypatch.setattr(P, "_req", fake)
    monkeypatch.setattr(P.time, "sleep", lambda *a: None)
    out = P._paged_get("https://api.github.com/x?page={page}", "tok", "test")
    assert len(out) == 120                         # the spurious empty did not truncate the listing


def test_issue_number_unions_truncated_scans(monkeypatch):
    # first scan is TRUNCATED (a short page 1, no next, missing the target); a later scan sees it.
    from data_qa import post_diagnostics as P
    target = {"number": 1, "state": "open", "title": "Brick — jw02221-o001 (NIRCam)"}
    truncated = [{"number": 9, "state": "open", "title": "something else"}]
    full = truncated + [target]
    seq = {"n": 0}

    def fake_paged(url, token, what):
        seq["n"] += 1
        return truncated if seq["n"] == 1 else full
    monkeypatch.setattr(P, "_paged_get", fake_paged)
    n = P._issue_number("JWST-GC/data-qa", "tok", "Brick — jw02221-o001 (NIRCam)")
    assert n == 1                                   # union across attempts recovered it


def test_issue_number_absent_title_returns_none(monkeypatch):
    from data_qa import post_diagnostics as P
    monkeypatch.setattr(P, "_paged_get",
                        lambda url, token, what: [{"number": 9, "state": "open", "title": "x"}])
    assert P._issue_number("JWST-GC/data-qa", "tok", "NOPE") is None


def test_load_reference_reads_refmag(tmp_path):
    """The Step-0 gaia_virac2 reference catalogue carries its magnitude in `refmag`, not
    `Ksmag`.  gc-treasury tiles have only this catalogue (no raw VIRAC2 Ksmag cache), so stage 3
    photometric calibration goes blank ("need VIRAC refcat") unless load_reference reads refmag."""
    from astropy.table import Table
    from data_qa import astrometry_audit as aa
    p = tmp_path / "gaia_virac2_refcat_epoch2026.70_o100.fits"
    Table({"RA": np.linspace(266.4, 266.6, 5), "DEC": np.linspace(-28.95, -28.85, 5),
           "refmag": np.array([12.0, 14.0, 16.0, 18.0, 20.0])}).write(p)
    sc, mag = aa.load_reference(str(p), 2026.7)
    assert sc is not None
    assert mag is not None and np.isfinite(mag).all()
    np.testing.assert_allclose(np.sort(mag), [12.0, 14.0, 16.0, 18.0, 20.0])


def test_load_reference_prefers_ksmag_over_refmag(tmp_path):
    """A raw VIRAC2 cache (reduction fields) carries both a real Ksmag and no refmag; where both a
    Ksmag and a refmag exist, Ksmag wins so the calibration uses native VIRAC2 Ks."""
    from astropy.table import Table
    from data_qa import astrometry_audit as aa
    p = tmp_path / "virac2.fits"
    Table({"RAJ2000": np.linspace(266.4, 266.6, 4), "DEJ2000": np.linspace(-28.95, -28.85, 4),
           "Ksmag": np.array([11.0, 13.0, 15.0, 17.0]),
           "refmag": np.array([99.0, 99.0, 99.0, 99.0])}).write(p)
    _, mag = aa.load_reference(str(p), 2026.7)
    np.testing.assert_allclose(np.sort(mag), [11.0, 13.0, 15.0, 17.0])


def test_load_reference_prefers_refmag_over_gaia_g(tmp_path):
    """The physics half of the ranking: with both a NIR `refmag` and Gaia optical `phot_g_mean_mag`
    present, refmag must win — choosing G for an F212N zeropoint is the failure the ranking prevents."""
    from astropy.table import Table
    from data_qa import astrometry_audit as aa
    p = tmp_path / "ref.fits"
    Table({"RA": np.linspace(266.4, 266.6, 4), "DEC": np.linspace(-28.95, -28.85, 4),
           "refmag": np.array([14.0, 15.0, 16.0, 17.0]),
           "phot_g_mean_mag": np.array([90.0, 91.0, 92.0, 93.0])}).write(p)
    _, mag = aa.load_reference(str(p), 2026.7)
    np.testing.assert_allclose(np.sort(mag), [14.0, 15.0, 16.0, 17.0])


def test_stage3_reference_never_uses_refcat_refmag(tmp_path, monkeypatch):
    """The gaia_virac2 refcat's VIRAC2 rows carry J in refmag; stage 3 must not grade a Ks zeropoint
    against it.  With no raw Ks cache and no VizieR result, the reference is absent (ungraded)."""
    from astropy.table import Table
    from data_qa.observations import Observation
    p = tmp_path / "gaia_virac2_refcat_epoch2026.70_o100.fits"
    Table({"RA": np.linspace(266.40, 266.60, 6), "DEC": np.linspace(-28.95, -28.85, 6),
           "source": np.array(["VIRAC2"] * 6), "refmag": np.arange(14.0, 20.0)}).write(p)
    o = Observation(program="10678", obs="100", target="T", release_field="gc-treasury",
                    instrument="NIRCam", filters=["F212N"], visits=[], epoch="", notes="")
    monkeypatch.setattr(D, "_viraccache_path", lambda o: None)
    monkeypatch.setattr(D, "_refcat_path", lambda o: str(p))
    monkeypatch.setenv("QA_VIRAC_DOWNLOAD", "0")
    assert D._stage3_reference(o, 2026.70) == (None, None)


def test_stage3_reference_reads_cached_virac2_ks(tmp_path, monkeypatch):
    """A cached VizieR VIRAC2 table next to the refcat supplies Ksmag, PM-propagated from 2014."""
    from astropy.table import Table
    from data_qa.observations import Observation
    p = tmp_path / "gaia_virac2_refcat_epoch2026.70_o100.fits"
    Table({"RA": [266.5], "DEC": [-28.9], "source": ["VIRAC2"], "refmag": [18.0]}).write(p)
    o = Observation(program="10678", obs="100", target="T", release_field="gc-treasury",
                    instrument="NIRCam", filters=["F212N"], visits=[], epoch="", notes="")
    monkeypatch.setattr(D, "_viraccache_path", lambda o: None)
    monkeypatch.setattr(D, "_refcat_path", lambda o: str(p))
    monkeypatch.setenv("QA_VIRAC_DOWNLOAD", "0")
    cache = D._stage3_ks_cache_path(o)
    import os
    os.makedirs(os.path.dirname(cache))
    Table({"RAJ2000": [266.5, 266.51], "DEJ2000": [-28.9, -28.91], "pmRA": [0.0, 0.0],
           "pmDE": [10.0, 0.0], "Jmag": [18.0, 19.0], "Ksmag": [12.0, 13.0]}).write(cache)
    sc, mag = D._stage3_reference(o, 2026.70)
    np.testing.assert_allclose(mag, [12.0, 13.0])                   # Ks, not J
    assert abs((sc[0].dec.deg - (-28.9)) * 3.6e6 - 127.0) < 1.0     # 10 mas/yr x 12.7 yr


def test_one_to_one_match_rejects_shared_partner():
    """Two VIRAC stars around ONE JWST star: only the mutual pair survives."""
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    ref = SkyCoord([266.5, 266.5 + 0.06 / 3600] * u.deg, [-28.9, -28.9] * u.deg)
    jw = SkyCoord([266.5 + 0.01 / 3600] * u.deg, [-28.9] * u.deg)
    ir, ij = D._one_to_one_match(ref, jw)
    assert list(ir) == [0] and list(ij) == [0]


def test_calibration_figure_recovers_locus_under_bulk_offset(monkeypatch):
    """A catalogue 130 mas off VIRAC (Sgr B2 case) in a dense field: without the shift the 0.1"
    match pairs neighbours; with it the true unit-slope locus comes back and the offset is reported."""
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    from data_qa.observations import Observation
    rng = np.random.default_rng(1)
    n = 6000
    ra = 266.4 + rng.uniform(0, 0.03, n); dec = -28.9 + rng.uniform(0, 0.03, n)
    ks = rng.uniform(12.0, 18.0, n)
    ref = SkyCoord(ra * u.deg, dec * u.deg)
    cosd = np.cos(np.radians(-28.9))
    jra = ra + (-30.0 / 3.6e6) / cosd; jde = dec + 127.0 / 3.6e6       # 130 mas off
    # deep JWST: every VIRAC star + 3x as many faint field stars
    fra = 266.4 + rng.uniform(0, 0.03, 3 * n); fde = -28.9 + rng.uniform(0, 0.03, 3 * n)
    jsc = SkyCoord(np.r_[jra, fra] * u.deg, np.r_[jde, fde] * u.deg)
    jmag = np.r_[ks + rng.normal(0, 0.05, n), rng.uniform(18, 23, 3 * n)]
    o = Observation(program="5365", obs="001", target="T", release_field="sgrb2",
                    instrument="NIRCam", filters=["F212N"], visits=[], epoch="", notes="")
    png, sub = D._calibration_figure(o, "F212N", jsc, jmag, "jicama-m7", ref, ks,
                                     "t_stage3_offset.png")
    assert abs(sub["offset_to_virac_mas"] - 130.0) < 15.0
    assert 0.9 < sub["slope"] < 1.1 and sub["scatter"] < 0.2


def test_stage3_red_flags_misregistered_pipeline_catalogue(monkeypatch):
    o = _stage3_synth(monkeypatch, our=True)
    real = D._calibration_figure

    def fake(o, sw, jsc, jmag, lbl, ref_sc, ref_mag, out, system="Vega"):
        png, sub = real(o, sw, jsc, jmag, lbl, ref_sc, ref_mag, out, system)
        if lbl != "MAST catalogue":
            sub["offset_to_virac_mas"] = 135.0
        return png, sub
    monkeypatch.setattr(D, "_calibration_figure", fake)
    _, m = D.stage3_calibration(o, "F212N")
    assert m["red_flag"] is True and m["passed"] is False
    assert "135 mas off VIRAC" in m["red_flag_reason"]
    assert m["registration_flag"] is True and m["photometry_passed"] is True  # zeropoint clean
    cap = D.caption_for(3, m)
    assert "RED FLAG" in cap and "plot is empty" not in cap     # the locus IS drawn
    assert "slope 1.0" in cap or "slope 0.9" in cap


def _stage3_synth(monkeypatch, our=True, our_offset=3.0, our_system="Vega", our_why=None):
    """Synthetic stage-3 inputs.  The reference spans 11-19 mag (STRADDLING the [13,17] fit window,
    so windowing is actually exercised), and MAST vs our photometry DIFFER (MAST noisy + a slope
    error, ours clean unit-slope) so the grade-source and windowing assertions can't pass by accident
    on identical fixtures."""
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    from data_qa.observations import Observation
    rng = np.random.default_rng(0)
    n = 400
    ra = 266.4 + rng.uniform(0, 0.02, n); dec = -28.9 + rng.uniform(0, 0.02, n)
    ks = rng.uniform(11.0, 19.0, n)                 # straddles the [13,17] window
    ref_sc = SkyCoord(ra * u.deg, dec * u.deg)
    o = Observation(program="10678", obs="100", target="T", release_field="gc-treasury",
                    instrument="NIRCam", filters=["F212N"], visits=[], epoch="", notes="")
    monkeypatch.setattr(D, "_mosaic_path", lambda o, f: None)
    monkeypatch.setattr(D, "_obs_epoch", lambda o, p: 2026.70)
    monkeypatch.setattr(D, "_stage3_reference", lambda o, ep: (ref_sc, ks))
    # MAST: a wrong slope (0.7) + large scatter -> would FAIL if it were graded
    monkeypatch.setattr(D, "_mast_calibration_sources",
                        lambda o, sw: (ref_sc, 0.7 * ks + 5.0 + rng.normal(0, 0.4, n), "Vega"))
    # ours: clean unit slope, tight scatter -> passes
    our_val = ((ref_sc, ks + our_offset + rng.normal(0, 0.05, n), "jicama-m2", our_system, our_why)
               if our else (None,) * 5)
    monkeypatch.setattr(D, "_stage3_our_catalog", lambda o, sw: our_val)
    return o


def test_stage3_our_primary_mast_in_dropdown(monkeypatch):
    o = _stage3_synth(monkeypatch, our=True)
    png, m = D.stage3_calibration(o, "F212N")
    assert m["primary_source"] == "jicama-m2"          # our catalogue is the shown image ...
    assert png.endswith(f"{o.obsid}_stage3.png")
    labels = [lbl for lbl, _ in m["extra_figures"]]
    assert labels == ["MAST catalogue vs VIRAC (calibration)"]   # ... MAST still posted, in dropdown
    assert m["source"] == "jicama-m2"                  # verdict comes from OUR catalogue
    assert m["passed"] is True                          # graded on OURS (unit slope): pass ...
    assert not (0.8 < m["mast_slope"] < 1.2)            # ... NOT on MAST (slope 0.7) -> grade-source pinned
    assert m["our_fit_windowed"] is True and m["our_n_fit"] < m["our_n_matched"]  # window applied


def test_stage3_mast_only_informational(monkeypatch):
    o = _stage3_synth(monkeypatch, our=False)
    _, m = D.stage3_calibration(o, "F212N")
    assert m["available"] is True                       # MAST shown -> stage posts
    assert m["passed"] is None                          # ungraded, NOT red-flagged
    assert m["primary_source"] == "MAST catalogue"
    assert not m.get("extra_figures")                   # nothing graded to add


def test_stage7_offset_recovers_true_offset_not_collapsed():
    """A DEEP catalogue genuinely 60 mas off VIRAC must report ~60, NOT the ~13 a bare same-star
    tie collapses to at its 0.05" radius.  Builds a dense reference, a JWST copy shifted by 60 mas,
    plus many spurious deep sources (the pile-up that drives the collapse), and checks the recovered
    bulk.  This pins the NUMBER (the reviewer's point: monkeypatching same_star_tie only pinned
    routing)."""
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    rng = np.random.default_rng(3)
    n = 4000
    ra = 266.40 + rng.uniform(0, 0.03, n)
    dec = -28.90 + rng.uniform(0, 0.03, n)
    ref = SkyCoord(ra * u.deg, dec * u.deg)
    cosd = np.cos(np.radians(-28.90))
    jra = ra + 60.0 / 3.6e6 / cosd + rng.normal(0, 0.003 / 3600, n)   # true 60 mas RA offset
    jdec = dec + rng.normal(0, 0.003 / 3600, n)
    era = 266.40 + rng.uniform(0, 0.03, 8000)                          # spurious deep sources
    ede = -28.90 + rng.uniform(0, 0.03, 8000)
    jsc = SkyCoord(np.concatenate([jra, era]) * u.deg,
                   np.concatenate([jdec, ede]) * u.deg)
    out = D._bulk_offset(jsc, ref)
    assert out is not None
    assert 50.0 < out[2] < 70.0            # ~60, not the collapsed ~13


def test_offset_panel_title_headlines_same_star_not_histogram():
    """The offset panel must lead with the same-star tie (the authoritative estimator) and demote
    the per-cell histogram median -- if someone simplifies it back to the histogram the figure would
    misstate the measurement while the metrics stay correct, and this catches that."""
    cc = {"n_cells": 12}
    t = D._offset_panel_title(1.2, "same-star", {"off": 1.2, "npairs": 40}, 14.0, cc, 6.0, "gate 75")
    lead, second = t.split("\n")[:2]
    assert "1.2 mas [same-star]" in lead and "same-star pairs" in lead
    assert "histogram median 14 mas" in second and "histogram median" not in lead
    # with no same-star tie, the reported (headline) value is the histogram one, labelled as such
    t2 = D._offset_panel_title(14.0, "histogram", None, 14.0, cc, 6.0, "gate 75")
    assert "14.0 mas [histogram]" in t2.split("\n")[0]


def test_caption_stage7_names_dropdown_when_jicama_primary():
    # jicama per-cell offset figure is the shown image -> the caption says the MAST comparison moved
    base = dict(stage=7, mast_offset_med_mas=40.0, jicama_offset_med_mas=10.0)
    cap = D.caption_for(7, dict(base, primary_figure="jicama_offset"))
    assert "dropdown" in cap and "per-cell offset" in cap
    cap = D.caption_for(7, base)                    # comparison figure shown -> no dropdown note
    assert "dropdown" not in cap


def test_stage7_pick_primary_release_jicama_promoted():
    # a release (jicama) offset figure is shown; the MAST comparison moves to the dropdown
    png, m = D._stage7_pick_primary("o_stage7.png", "o_stage7_jicama_offset.png",
                                    "release:jw10678-o132_f212n_m8.fits", {})
    assert png == "o_stage7_jicama_offset.png"
    assert m["primary_figure"] == "jicama_offset"
    assert m["extra_figures"] == [("MAST vs pipeline (mosaics, depth, offsets)", "o_stage7.png")]


def test_stage7_pick_primary_mast_fallback_keeps_comparison():
    # positions fell back to MAST -> the comparison stays primary, offset figure in the dropdown
    png, m = D._stage7_pick_primary("o_stage7.png", "o_stage7_jicama_offset.png",
                                    "MAST:jw10678-o132_cat.fits", {})
    assert png == "o_stage7.png"
    assert "primary_figure" not in m
    assert m["extra_figures"] == [("jicama vs VIRAC (per-cell offset)", "o_stage7_jicama_offset.png")]


def _stage2_cat(tmp_path, monkeypatch, program, mtime, tier="m8"):
    from astropy.table import Table
    rng = np.random.default_rng(1)
    n = 3000
    lw = rng.uniform(10, 20, n); sw = lw + rng.normal(2.0, 0.3, n)
    cat = tmp_path / f"basic_merged_indivexp_photometry_tables_merged_resbgsub_{tier}_o132.fits"
    Table({"mag_ab_f212n": sw, "mag_ab_f480m": lw}).write(cat, overwrite=True)
    os.utime(cat, (mtime, mtime))
    monkeypatch.setattr(D, "OUTDIR", str(tmp_path))
    monkeypatch.setattr(D, "_catalog_for",
                        lambda o, sw, lw: (str(cat), tier, "mag_ab_f212n", "mag_ab_f480m"))
    return Observation(program=program, obs="132", target="T", release_field="gc-treasury",
                       instrument="NIRCam", filters=["F212N", "F480M"], visits=[], epoch="",
                       notes="")


def test_stage2_adds_sw_cmd(tmp_path, monkeypatch):
    o = _stage2_cat(tmp_path, monkeypatch, "10678", 1.8e9)
    png, m = D.stage2_cmd(o, "F212N", "F480M")
    assert m["passed"] is True and "pending_rebuild" not in m
    (label, extra), = m["extra_figures"]
    assert "F212N on the y axis" in label and os.path.exists(extra)
    assert "F212N on the y axis" in D.caption_for(2, m)

def test_caption_stage3_notes_undetermined_offset():
    m = dict(available=True, passed=True, our_slope=1.0, slope=1.0, scatter=0.1, n_matched=500,
             source="jicama-m8", offset_to_virac_mas=None)
    assert "could not be measured" in D.caption_for(3, m)
    m["offset_to_virac_mas"] = 3.0
    assert "could not be measured" not in D.caption_for(3, m)


def test_caption_stage3_no_photometry_red_flag_keeps_empty_wording():
    m = dict(stage=3, red_flag=True, passed=False, red_flag_reason="no JWST photometry")
    assert "plot is empty" in D.caption_for(3, m)


def test_caption_stage7_cell_figure_unmeasurable_keeps_full_catalogue_offset():
    # o112: per-cell (position-valid) selection unmeasurable, full catalogue 6 mas vs MAST 28 mas.
    m = dict(stage=7, passed=True, primary_figure="jicama_offset", jicama_is_release=True,
             mast_offset_med_mas=28.3, jicama_offset_med_mas=6.4, jicama_cell_offset_med_mas=None,
             jicama_cell_offset_unmeasurable=True, n_jicama_window=93167, n_mast_window=27365)
    cap = D.caption_for(7, m)
    assert "UNMEASURABLE" in cap and "6 mas (jicama) vs 28 mas (MAST)" in cap
    assert "mis-registration" not in cap
    # A measured per-cell offset does not get the UNMEASURABLE note.
    m.update(jicama_cell_offset_med_mas=5.9, jicama_cell_offset_unmeasurable=False)
    assert "UNMEASURABLE" not in D.caption_for(7, m)


def test_stage7_merge_cell_offset_keeps_measured_headline():
    # o112: full-catalogue headline 6.4 mas; the per-cell figure is unmeasurable.  The cell value
    # must not overwrite the headline.
    m = D._stage7_merge_cell_offset(dict(jicama_offset_med_mas=6.4),
                                    dict(offset_med_mas=None, offset_unmeasurable=True))
    assert m["jicama_offset_med_mas"] == 6.4
    assert m["jicama_cell_offset_med_mas"] is None and m["jicama_cell_offset_unmeasurable"]
    # A measured cell value also leaves a measured headline alone.
    m = D._stage7_merge_cell_offset(dict(jicama_offset_med_mas=6.4),
                                    dict(offset_med_mas=464.0, offset_unmeasurable=False))
    assert m["jicama_offset_med_mas"] == 6.4 and m["jicama_cell_offset_med_mas"] == 464.0
    # No headline: fall back to the cell value.
    m = D._stage7_merge_cell_offset(dict(jicama_offset_med_mas=None),
                                    dict(offset_med_mas=5.9, offset_unmeasurable=False))
    assert m["jicama_offset_med_mas"] == 5.9


def test_perfilter_vega_mag_matches_pipeline_conversion():
    """flux = sum of MJy/sr pixels -> Jy via the pixel solid angle -> Vega with the SVO zero point
    (jwst-gc-pipeline merge_catalogs).  jw10678-o086 F212N: instrumental mags sat ~26 mag below Ks."""
    import astropy.units as u
    from astropy.table import Table
    t = Table({"flux": [189685.8, 100.8]}); t.meta["PIXSCALE"] = 0.03122293448156842
    mag, why = D._perfilter_vega_mag(t, "F212N", np.asarray(t["flux"], float))
    assert why is None
    pixar = ((0.03122293448156842 * u.arcsec) ** 2).to(u.sr).value
    expect = -2.5 * np.log10(189685.8 * 1e6 * pixar / 674.83)
    assert abs(mag[0] - expect) < 1e-6 and 12.0 < mag[0] < 14.0     # a K~13 star, not -13


def test_perfilter_vega_mag_explains_instrumental_fallback():
    from astropy.table import Table
    t = Table({"flux": [1000.0]})                                   # no PIXSCALE
    mag, why = D._perfilter_vega_mag(t, "F212N", np.array([1000.0]))
    assert abs(mag[0] + 7.5) < 1e-9 and "PIXSCALE" in why
    t.meta["PIXSCALE"] = 0.031
    _, why = D._perfilter_vega_mag(t, "F999X", np.array([1000.0]))
    assert "zero point" in why


def test_stage3_labels_systems_and_rejects_uncalibrated_vega(monkeypatch):
    # a column labelled Vega that sits 26 mag off Ks is not Vega -> red flag naming the units
    o = _stage3_synth(monkeypatch, our=True, our_offset=-26.0)
    _, m = D.stage3_calibration(o, "F212N")
    assert m["our_mag_system"] == "Vega" and m["our_mag_system_verified"] is False
    assert m["red_flag"] is True and "not Vega" in m["red_flag_reason"]
    cap = D.caption_for(3, m)
    assert "VIRAC Ks in **Vega**" in cap and "FAILED the unit check" in cap
    assert "MAST catalogue F212N in **Vega**" in cap


def test_stage3_instrumental_is_labelled_with_reason(monkeypatch):
    o = _stage3_synth(monkeypatch, our=True, our_offset=-26.0, our_system="instrumental",
                      our_why="the catalogue header carries no PIXSCALE")
    _, m = D.stage3_calibration(o, "F212N")
    assert m["our_mag_system"] == "instrumental" and m["passed"] is True   # slope still graded
    assert not m.get("red_flag")                    # correctly labelled instrumental: not a defect
    cap = D.caption_for(3, m)
    assert "**instrumental**" in cap and "because the catalogue header carries no PIXSCALE" in cap


def test_not_applicable_stage_draws_neutral_card_not_red_flag(monkeypatch):
    # stage 8 with no second band: "not reduced yet" must never be drawn as a RED FLAG
    drawn = []
    monkeypatch.setattr(D, "_red_flag_figure", lambda *a, **k: drawn.append("red") or "r.png")
    monkeypatch.setattr(D, "_note_figure", lambda *a, **k: drawn.append("note") or "n.png")
    monkeypatch.setattr(D, "_interfilter_residuals", lambda o, sw: None)
    monkeypatch.setattr(D, "_perfilter_interfilter_residuals", lambda o, sw: None)
    from data_qa.observations import Observation
    o = Observation(program="10678", obs="086", target="T", release_field="gc-treasury",
                    instrument="NIRCam", filters=["F212N"], visits=[], epoch="", notes="")
    _, m = D.stage8_distortion(o, "F212N")
    assert drawn == ["note"] and m["passed"] is None


def test_missing_sw_reads_pending_not_failed(tmp_path, monkeypatch):
    # 10678 o087: F480M reduced while F212N is still at image2.  Stage 1 must read "not done yet"
    # (passed=None), and the SW-graded stages must report pending instead of crashing on sw=None.
    monkeypatch.setattr(D, "BASE", str(tmp_path))
    red = tmp_path / "gc-treasury" / "F480M" / "pipeline"
    _write_i2d(str(red / "jw10678-o087_t001_nircam_clear-f480m-merged_i2d.fits"))
    o = Observation(program="10678", obs="087", target="GC Treasury", release_field="gc-treasury",
                    instrument="NIRCam", filters=["F480M"], visits=[], epoch="", notes="")
    _png, m1 = D.stage1_mosaics(o, None, "F480M")
    assert m1["passed"] is None
    cap = D.caption_for(1, m1)
    assert "nan" not in cap and "pending" in cap and "RED FLAG" not in cap
    for n in D._STAGES_NEEDING_SW:
        png, m = D._dispatch_stage(o, n, None, "F480M")
        assert png is None and m["available"] is False and m["passed"] is None
    # Every stage must survive sw=None without a red flag: 10678 o077/o084 crashed in stages 8/9
    # (_interfilter_residuals on f1=None) after 2/3/5 were guarded.  This fixture holds only the
    # F480M i2d, so stages outside _STAGES_NEEDING_SW may reach n/a from missing catalogues; it
    # guards against crashes, and a fixture with LW catalogues + MAST products would be needed to
    # exercise their sw=None handling on real inputs.
    for n in range(2, 13):
        _png, m = D._dispatch_stage(o, n, None, "F480M")
        assert m.get("passed") is not False, (n, m)
