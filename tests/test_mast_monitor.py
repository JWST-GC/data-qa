"""Offline unit tests for data_qa.mast_monitor (state diffing; no network)."""
import json

import pytest

from data_qa import mast_monitor as mm
from data_qa.observations import FIELDS

POLL = 60000.0   # fake "now" MJD


def _row(obs_id, calib=3, release=59900.0, filters="F405N;F410M",
         target="GAL_CENTER"):
    return {"obs_id": obs_id, "t_max": release - 10, "t_obs_release": release,
            "calib_level": calib, "instrument_name": "NIRCAM/IMAGE",
            "filters": filters, "target_name": target}


# ---------------------------------------------------------------------- obs_id parse
def test_obsnum_from_dash_obsid():
    assert mm.obsnum_from_obs_id("jw02221-o001_t001_nircam_clear-f405n") == "001"


def test_obsnum_from_flat_obsid():
    assert mm.obsnum_from_obs_id("jw02221001001_02101_00001_nrcalong") == "001"


def test_obsnum_unparseable():
    assert mm.obsnum_from_obs_id("hst_12345") == ""


def test_field_mapping():
    assert mm.field_for(2221, "001") == "brick"
    assert mm.field_for(2221, "002") == "cloudc"
    assert mm.field_for("1182", "004") == "brick"
    assert mm.field_for(1182, "002") == "w51"
    assert mm.field_for(9999, "001") == ""


def test_programs_cross_check_release_fields():
    """Every mapped field that has a public release page is a known FIELDS key."""
    unreleased = {"cloudef", "sgra", "ngc6334"}   # no release page yet
    for prog, obsmap in mm.PROGRAMS.items():
        for field in obsmap.values():
            assert field in FIELDS or field in unreleased, (prog, field)


# --------------------------------------------------------------------------- diffing
def test_new_observation_event():
    new = mm.summarize([_row("jw02221-o001_t001_nircam_clear-f405n")], POLL)
    events = mm.diff_events(2221, {}, new)
    assert [e["event"] for e in events] == ["NEW_OBSERVATION"]
    assert events[0]["field"] == "brick"
    assert events[0]["obsnum"] == "001"


def test_no_events_when_unchanged():
    new = mm.summarize([_row("jw02221-o001_t001_nircam_clear-f405n")], POLL)
    assert mm.diff_events(2221, new, new) == []


def test_newly_released_event():
    row = _row("jw02221-o002_t001_nircam_clear-f405n", release=POLL - 1)
    old = mm.summarize([row], POLL - 100)   # release still in the future then
    assert old[row["obs_id"]]["released"] is False
    new = mm.summarize([row], POLL)
    events = mm.diff_events(2221, old, new)
    assert [e["event"] for e in events] == ["NEWLY_RELEASED"]
    assert events[0]["field"] == "cloudc"


def test_calib_level_up_event():
    row2 = _row("jw04147-o012_t001_nircam_clear-f405n", calib=2)
    row3 = _row("jw04147-o012_t001_nircam_clear-f405n", calib=3)
    old = mm.summarize([row2], POLL)
    new = mm.summarize([row3], POLL)
    events = mm.diff_events(4147, old, new)
    assert [e["event"] for e in events] == ["CALIB_LEVEL_UP"]
    assert events[0]["previous_calib_level"] == 2
    assert events[0]["calib_level"] == 3


def test_release_and_calib_up_together():
    rowa = _row("jw02221-o001_x", calib=2, release=POLL + 5)
    rowb = _row("jw02221-o001_x", calib=3, release=POLL - 5)
    events = mm.diff_events(2221, mm.summarize([rowa], POLL - 100),
                            mm.summarize([rowb], POLL))
    assert sorted(e["event"] for e in events) == ["CALIB_LEVEL_UP", "NEWLY_RELEASED"]


# ----------------------------------------------------------------------------- state
def test_state_roundtrip_atomic(tmp_path):
    path = tmp_path / "sub" / "dir" / "state.json"   # parent auto-created
    state = {"version": 1, "programs": {"2221": {"obs": {}}}}
    mm.save_state(str(path), state)
    assert mm.load_state(str(path)) == state
    assert not list(tmp_path.glob("**/*.tmp.*"))     # tmp file renamed away


def test_load_state_missing_is_empty(tmp_path):
    st = mm.load_state(str(tmp_path / "nope.json"))
    assert st["programs"] == {}


def test_load_state_corrupt_raises(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{not json")
    with pytest.raises(json.JSONDecodeError):
        mm.load_state(str(p))


def test_format_event_readable():
    new = mm.summarize([_row("jw02221-o001_t001_nircam_clear-f405n")], POLL)
    (ev,) = mm.diff_events(2221, {}, new)
    line = mm.format_event(ev)
    assert "NEW_OBSERVATION" in line and "brick" in line and "2221" in line
