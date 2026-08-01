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


# --------------------------------------------------------------------------- _daophot_glob
def _touch(d, name):
    (d / name).write_text("")


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
