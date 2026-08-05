"""Unit tests for the QA diagnostics helpers that previously had ZERO coverage.

Covers the pure numeric helpers (`_binned_stat`), the caption fallback that used to print
"nanσ" (`caption_for` with the significance panel omitted), and the observation-scoping of the
per-exposure daophot glob (`_daophot_glob`) that keeps one observation's cats out of another's
QA on multi-obs fields.  No I/O beyond touching empty files under a temporary QA_BASE.
"""
import os

import numpy as np
import pytest

from data_qa import diagnostics as D
from data_qa.observations import Observation


def _obs(field="gc2211", obs="023", filt="F200W"):
    return Observation(program="2211", obs=obs, target="T", release_field=field,
                       instrument="NIRCam", filters=[filt], visits=[], epoch="", notes="")


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
def test_caption_stage4_no_significance_no_nan():
    # zero-spread / no significance -> still report the median tie over its cells, never "nanσ"
    cap = D.caption_for(4, dict(stage=4, sw="F212N", offset_signif_med=None,
                                offset_med_mas=7.1, n_cells=6, offset_scatter_mas=None))
    assert "nan" not in cap.lower()
    assert "7 mas" in cap and "6 measured cells" in cap


def test_caption_stage4_flags_discontinuity():
    # an adjacency-confirmed deviating region is called out in the caption
    cap = D.caption_for(4, dict(stage=4, offset_med_mas=31, n_cells=12, offset_scatter_mas=8.0,
                                offset_signif_med=4.0, n_cells_confirmed=3, bad_src_frac=0.08,
                                n_cells_dropped=0))
    assert "internal discontinuity" in cap and "8%" in cap


def test_caption_stage1_dropped_filters_noted():
    cap = D.caption_for(1, dict(stage=1, sw="F212N", lw="F405N", dropped_filters=["F444W", "F322W2"]))
    assert "F444W" in cap and "F322W2" in cap and "not reduced" in cap


def test_caption_redflag():
    cap = D.caption_for(3, dict(red_flag=True, red_flag_reason="no catalog for F212N"))
    assert "RED FLAG" in cap and "no catalog" in cap


def test_caption_stage5_no_overlap_two_modules():
    # both modules present but no shared stars -> overlap keys absent; must NOT truncate to a
    # bare fragment and must say the tie is unverified.
    cap = D.caption_for(5, dict(stage=5, intermodule_diff=3.0, passed=False))
    assert "nan" not in cap.lower()
    assert "could not be measured" in cap and "unverified" in cap
    assert "3.0 mas" in cap


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
    assert "TOP-LEFT" in cap and "Reference-free" in cap    # panels labelled + term defined
    assert "S/N > 10" in cap and "marginal" in cap          # the new S/N panel + marginals noted
    assert "TOP-MIDDLE" in cap and "to its right" in cap    # correct panel positions
    # the NRCB2/no-overlap question is answered inline
    assert "NRCB2" in cap and "VIRAC, not NRCA" in cap


def test_caption_stage5_overlap_without_hi_sn_panel():
    # overlap present but no S/N>10 panel (field lacks flux errors) -> no S/N promise; the all-stars
    # panel is TOP-RIGHT (2-column top row), and the footprint is its own full-width row below
    cap = D.caption_for(5, dict(stage=5, intermodule_diff=3.0, intermodule_off=4.1,
                                intermodule_rms=6.2, n_overlap=137, n_overlap_footprint=137))
    assert "S/N > 10" not in cap and "to its right" not in cap
    assert "TOP-RIGHT" in cap and "137 shared stars" in cap
    assert "full-width row" in cap and "dither-overlap strip" in cap   # footprint described


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


def test_caption_stage3_drops_false_claim_and_labels_line():
    cap = D.caption_for(3, dict(stage=3, sw="F212N", n_matched=2603, slope=1.0, scatter=0.28))
    # the untrue "a tight locus means the right stars were matched" claim is gone
    assert "right stars were matched" not in cap
    # positive labelling: the cyan line is named "1:1 line" (no "not a fit")
    assert "1:1 line" in cap and "NOT a fit" not in cap


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
                offset_signif_med=3.0, n_cells_dropped=2),
        5: dict(stage=5, intermodule_diff=3.0, intermodule_off=4.1, intermodule_rms=6.0,
                n_overlap=100, n_overlap_hi=50, intermodule_rms_hi=4.0),
        6: dict(red_flag=True, red_flag_reason="x"),
    }
    for n, m in samples.items():
        cap = D.caption_for(n, m)
        for anc in re.findall(r"qa_methods\.md#([A-Za-z0-9\-]+)", cap):
            assert anc in ids, f"stage {n} caption links #{anc} but no <a id> exists in the doc"


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
    msg = D._offset_failure_reason(_obs(), "F200W", sc, sc, {"peak_ratio": 0.3})
    # must NOT assert an unmeasured cause -- says it's undetermined, and that no VIRAC comparison ran
    assert "cause not determined" in msg
    assert "not measured" in msg


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
    # 99% of sources in cells tied at ~130, 1% at ~28 -> the CATALOG tie is ~130, not 28
    # (the density-biased peak-ratio cut used to keep the sparse 28 mas side; #54 review 🔴1).
    cells = _grid_cells({**{(i, j): (130.0, 0.0, 40000) for i in range(4) for j in range(4) if not (i == 0 and j == 0)},
                         (0, 0): (28.0, 0.0, 400)})
    cc = D._cell_consistency(cells, [])
    assert cc["off_med"] > 100            # source-weighted, not the sparse minority


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


def test_cell_consistency_low_coverage_not_consistent():
    # most sources sit in DROPPED (no-peak) cells -> not adequately sampled to pass
    cells = _grid_cells({(0, 0): (9.0, 0.0, 500), (0, 1): (9.0, 0.0, 500),
                         (1, 0): (9.0, 0.0, 500), (1, 1): (9.0, 0.0, 500)})
    dropped = [dict(i=2, j=2, ra=0.0, dec=0.0, n=100000)]
    cc = D._cell_consistency(cells, dropped)
    assert cc["coverage"] < 0.5 and not cc["consistent"]


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
    cells, dropped = D._cell_offsets(jsc, ref, ncell=2, min_per_cell=50)
    assert len(cells) >= 3
    dra = np.array([c["dra"] for c in cells])
    assert np.all(np.abs(dra - 100.0) < 15)      # each cell recovers ~+100 mas


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
    monkeypatch.setattr(D, "_jwst_sources", lambda o, f: (None, None, None))   # no merged/MAST
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
