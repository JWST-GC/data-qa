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
    # too few cells to estimate an uncertainty -> report the median tie, say consistency not
    # assessed, and never print "nanσ".
    cap = D.caption_for(4, dict(stage=4, sw="F212N", offset_signif_med=None,
                                offset_med_mas=7.1, n_cells=1))
    assert "nan" not in cap.lower()
    assert "not assessed" in cap and "7 mas" in cap


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
    cap = D.caption_for(5, dict(stage=5, intermodule_diff=3.0, intermodule_off=4.1,
                                intermodule_rms=6.2, n_overlap=137))
    assert "137 shared" in cap and "4.1 mas" in cap


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
    r = D._offset_failure_reason(_obs(), "F200W", None, object(), None, 0)
    assert "not catalogued" in r and "F200W" in r


def test_offset_reason_no_reference():
    r = D._offset_failure_reason(_obs(), "F200W", object(), None, None, 0)
    assert "no virac reference" in r.lower()


def test_offset_reason_disjoint_footprint():
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    j = SkyCoord([266.40, 266.41] * u.deg, [-28.90, -28.89] * u.deg)   # north patch
    r = SkyCoord([266.40, 266.41] * u.deg, [-29.20, -29.19] * u.deg)   # south patch, disjoint
    msg = D._offset_failure_reason(_obs(), "F200W", j, r, {"peak_ratio": 0.0}, 0)
    assert "do not" in msg and "overlap" in msg


def test_offset_reason_overlap_but_no_peak(monkeypatch):
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    # no I/O: the reason may quote the JWST mag range, so stub the catalog read out
    monkeypatch.setattr(D, "_jwst_sources", lambda o, f: (None, None, None))
    sc = SkyCoord([266.40, 266.41, 266.42] * u.deg, [-28.90, -28.89, -28.88] * u.deg)
    msg = D._offset_failure_reason(_obs(), "F200W", sc, sc, {"peak_ratio": 0.3}, 0)
    # must NOT assert an unmeasured cause -- says it's undetermined, and that no VIRAC comparison ran
    assert "cause not determined" in msg
    assert "not measured" in msg


# --------------------------------------------------------------------------- cell-based stage-4
def test_cell_stats_uniform_small_spread():
    # cells all near the same offset -> small spread, high significance
    dra = np.array([100.0, 101.0, 99.0, 100.5, 98.5, 101.5])
    dde = np.array([50.0, 49.0, 51.0, 50.5, 49.5, 50.0])
    off, spread, se, sig = D._cell_stats(dra, dde)
    assert abs(off - np.hypot(100.0, 50.0)) < 2
    assert spread < 5 and se is not None and sig is not None and sig > 10


def test_cell_stats_bimodal_large_spread():
    # half the cells at ~0, half at ~130 -> large spread (catches an internal discontinuity)
    dra = np.array([0.0, 2.0, 1.0, 128.0, 130.0, 129.0])
    dde = np.zeros(6)
    off, spread, se, sig = D._cell_stats(dra, dde)
    assert spread > 30            # would fail the _CELL_SPREAD_MAX consistency gate


def test_cell_stats_single_cell_no_uncertainty():
    off, spread, se, sig = D._cell_stats(np.array([40.0]), np.array([10.0]))
    assert spread is None and se is None and sig is None


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
    cells = D._cell_offsets(jsc, ref, ncell=2, min_per_cell=50)
    assert len(cells) >= 3
    dra = np.array([c["dra"] for c in cells])
    assert np.all(np.abs(dra - 100.0) < 15)      # each cell recovers ~+100 mas


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
