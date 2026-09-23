"""The m-stage rows of a per-observation status table must describe THAT observation.

Catalog products for many observations share one field directory and carry the obsid in
their filenames.  A glob without the obsid reports the field-wide newest file for every
stage, so a sibling tile finishing an early stage makes every later stage of every other
tile read "older than an earlier m-stage" -> 🛑 STALE.  These tests build a synthetic
field directory and pin the scoping behaviour.
"""
import os
import time

import pytest

from data_qa import pipeline_status as PS


class _O:
    program = "10678"
    target = "GC Treasury"
    field = "gc-treasury"
    instrument = "NIRCam"

    def __init__(self, obs):
        self.obs = obs
        self.obsid = f"jw10678-o{obs}"


def _touch(path, when):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write("")
    os.utime(path, (when, when))


@pytest.fixture()
def field(tmp_path, monkeypatch):
    """gc-treasury with two tiles: o041 monotone m1->m3, o046 newer but only through m2."""
    P = tmp_path / "gc-treasury"
    t0 = time.time() - 86400 * 5
    for k, dt in ((1, 0), (2, 3600), (3, 7200)):
        _touch(str(P / "catalogs" / f"f212n_merged_o041_indivexp_merged_m{k}_dao_basic.fits"),
               t0 + dt)
    for k, dt in ((1, 10800), (2, 14400)):
        _touch(str(P / "catalogs" / f"f212n_merged_o046_indivexp_merged_m{k}_dao_basic.fits"),
               t0 + dt)
    _touch(str(P / "F212N" / "f212n_nrca1_o041_visit001_vgroup02101_exp00001_m1_daophot_basic.fits"),
           t0)
    monkeypatch.setattr(PS, "BASE", str(tmp_path))
    return P


def _by_label(rows):
    return {label: (status, detail) for label, status, _when, detail in rows}


def test_sibling_tile_does_not_make_this_one_stale(field):
    """o046's newer m2 must not push o041's m3 into 🛑 STALE (the reported defect)."""
    rows = _by_label(PS.stage_rows(_O("041")))
    assert rows["cataloging m1"][0] == PS.DONE
    assert rows["cataloging m2"][0] == PS.DONE
    assert rows["cataloging m3"][0] == PS.DONE
    assert "STALE" not in rows["cataloging m3"][1]


def test_counts_are_this_observation_only(field):
    rows = _by_label(PS.stage_rows(_O("041")))
    assert rows["cataloging m1"][1].startswith("1 catalogs")


def test_stage_this_observation_has_not_reached_reads_pending(field):
    """o046 has no m3, though the field does -- that row is ⬜ pending, not ✅ done."""
    rows = _by_label(PS.stage_rows(_O("046")))
    assert rows["cataloging m2"][0] == PS.DONE
    assert rows["cataloging m3"][0] == PS.PEND


def test_real_regression_within_one_observation_is_still_stale(field):
    """Scoping must not blunt the check: an m3 older than this tile's own m2 is stale."""
    P = field
    old = os.path.getmtime(P / "catalogs" / "f212n_merged_o041_indivexp_merged_m1_dao_basic.fits")
    os.utime(P / "catalogs" / "f212n_merged_o041_indivexp_merged_m3_dao_basic.fits",
             (old - 3600, old - 3600))
    rows = _by_label(PS.stage_rows(_O("041")))
    assert rows["cataloging m3"][0] == PS.STALE


def test_untagged_field_falls_back_to_pooled_with_a_caveat(tmp_path, monkeypatch):
    """Fields whose catalogs predate obsid naming keep reporting, and say they are pooled."""
    P = tmp_path / "w51"
    t0 = time.time() - 86400
    for k in (1, 2):
        _touch(str(P / "catalogs" / f"f405n_merged_indivexp_merged_m{k}_dao_basic.fits"), t0 + k)
    monkeypatch.setattr(PS, "BASE", str(tmp_path))

    class _W(_O):
        program = "1182"
        field = "w51"

    rows = _by_label(PS.stage_rows(_W("001")))
    assert rows["cataloging m1"][0] == PS.DONE
    assert "field-pooled" in rows["cataloging m1"][1]


def test_single_frame_row_is_scoped_too(field):
    rows = _by_label(PS.stage_rows(_O("046")))
    assert rows["single-frame cataloging"][0] == PS.PEND
    rows = _by_label(PS.stage_rows(_O("041")))
    assert rows["single-frame cataloging"][0] == PS.DONE


@pytest.fixture()
def partly_tagged(tmp_path, monkeypatch):
    """w51-shaped field: a few obsid-tagged catalogs beside many that carry no obsid.

    Real proportions (2026-09-21): w51 73 tagged of 1068, cloudc 18 of 589, sgrb2 36 of
    878.  The tagged ones are also the OLDER ones, which is what makes dropping the
    untagged majority move a row's date backwards instead of merely narrowing it.
    """
    P = tmp_path / "w51"
    old = time.time() - 86400 * 90          # tagged, June-era
    new = time.time() - 86400               # untagged, current
    for k in (2, 3, 4):
        _touch(str(P / "catalogs" / f"f405n_merged_o001_indivexp_merged_m{k}_dao_basic.fits"), old)
        _touch(str(P / "catalogs" / f"f405n_merged_o002_indivexp_merged_m{k}_dao_basic.fits"), old)
    for k in (5, 6):    # tagged, but only a SIBLING's -- the case that went blank
        _touch(str(P / "catalogs" / f"f405n_merged_o002_indivexp_merged_m{k}_dao_basic.fits"), old)
    for k in (2, 3, 4, 5, 6):
        for det in ("nrca", "nrcb"):
            _touch(str(P / "catalogs" / f"f405n_{det}_indivexp_merged_m{k}_dao_basic.fits"), new)
    monkeypatch.setattr(PS, "BASE", str(tmp_path))

    class _W(_O):
        program = "1182"
        field = "w51"

    return _W


def test_partial_tagging_keeps_the_untagged_products(partly_tagged):
    """The untagged majority must stay in the row, not be dropped for a tagged subset."""
    rows = _by_label(PS.stage_rows(partly_tagged("001")))
    when = {label: w for label, _st, w, _d in PS.stage_rows(partly_tagged("001"))}
    assert rows["cataloging m2"][0] == PS.DONE
    assert "3 catalogs" in rows["cataloging m2"][1]        # 1 tagged mine + 2 untagged
    # dated from the current untagged files, not the three-month-old tagged pair
    assert when["cataloging m2"] > when["cataloging m2"][:4] + "-01-01 00:00"
    assert rows["cataloging m2"][1].endswith("2 untagged not attributed")


def test_partial_tagging_does_not_blank_a_stage_the_field_has(partly_tagged):
    """m5/m6 have no tagged file of THIS obs, only a sibling's plus untagged ones.

    Scoping alone reported pending here, on a field holding 148 and 115 of them.
    """
    rows = _by_label(PS.stage_rows(partly_tagged("001")))
    for k in (5, 6):
        assert rows[f"cataloging m{k}"][0] == PS.DONE
        assert "2 untagged not attributed" in rows[f"cataloging m{k}"][1]


def test_partial_tagging_says_how_many_were_not_attributed(partly_tagged):
    """The caveat is the point: a silently-narrowed row reads as pipeline state."""
    for label, _st, _w, detail in PS.stage_rows(partly_tagged("001")):
        if label.startswith("cataloging m") and detail:
            assert "untagged not attributed" in detail


def test_partial_tagging_compares_one_population(partly_tagged):
    """Every m-row draws from the same set, so no row goes stale against another's."""
    rows = _by_label(PS.stage_rows(partly_tagged("001")))
    assert not any(st == PS.STALE for label, (st, _d) in rows.items()
                   if label.startswith("cataloging m"))
