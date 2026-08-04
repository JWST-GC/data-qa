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
def test_caption_stage4_omitted_significance_no_nan():
    cap = D.caption_for(4, dict(stage=4, sw="F212N", offset_signif_med=None, bulk_off=7.1))
    assert "nan" not in cap.lower()
    assert "omitted" in cap


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
