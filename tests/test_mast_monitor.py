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


# -------------------------------------------------------------------------- treasury
def test_treasury_program_in_programs():
    assert mm.TREASURY_PROGRAM == 10678
    assert mm.TREASURY_PROGRAM in mm.PROGRAMS


def test_treasury_field_for_any_obsnum():
    """10678 obs numbers are not enumerable in advance: EVERY obsnum maps to
    the gc-treasury field."""
    for obsnum in ("001", "123", "999"):
        assert mm.field_for(10678, obsnum) == mm.TREASURY_FIELD
    assert mm.field_for("10678", "042") == mm.TREASURY_FIELD


def test_treasury_event_carries_tile_name():
    row = _row("jw10678-o017_t017_nircam_clear-f212n", filters="F212N;F480M",
               target="GC_17")
    (ev,) = mm.diff_events(10678, {}, mm.summarize([row], POLL))
    assert ev["field"] == "gc-treasury"
    assert ev["tile"] == "GC_17"
    assert "tile=GC_17" in mm.format_event(ev)


def test_non_treasury_event_has_no_tile():
    new = mm.summarize([_row("jw02221-o001_t001_nircam_clear-f405n")], POLL)
    (ev,) = mm.diff_events(2221, {}, new)
    assert ev["tile"] is None


# ------------------------------------------------------------------------- disk gate
class _Usage:
    def __init__(self, free):
        self.total = 100e12
        self.used = self.total - free
        self.free = free


def test_disk_gate_passes_with_space(monkeypatch, tmp_path):
    monkeypatch.setattr(mm.shutil, "disk_usage", lambda p: _Usage(10e12))
    ok, free_tb, msg = mm.disk_gate(str(tmp_path), min_free_tb=5.0)
    assert ok is True
    assert free_tb == pytest.approx(10.0)
    assert "OK" in msg


def test_disk_gate_fails_below_threshold(monkeypatch, tmp_path):
    monkeypatch.setattr(mm.shutil, "disk_usage", lambda p: _Usage(2e12))
    ok, free_tb, msg = mm.disk_gate(str(tmp_path), min_free_tb=5.0)
    assert ok is False
    assert free_tb == pytest.approx(2.0)
    assert "LOW DISK" in msg and "report-only" in msg


def test_free_terabytes_climbs_to_existing_parent(monkeypatch, tmp_path):
    seen = {}

    def fake_usage(p):
        seen["path"] = p
        return _Usage(7e12)

    monkeypatch.setattr(mm.shutil, "disk_usage", fake_usage)
    missing = tmp_path / "not" / "yet" / "created"
    assert mm.free_terabytes(str(missing)) == pytest.approx(7.0)
    assert seen["path"] == str(tmp_path)   # nearest existing parent


# ------------------------------------------------------------------------- auto mode
def _patch_poll(monkeypatch, rows):
    """Offline main(): canned MAST rows, no login, recorded actions."""
    calls = []
    monkeypatch.setattr(mm, "mast_login_if_token", lambda: False)
    monkeypatch.setattr(mm, "query_program", lambda prog: rows)
    monkeypatch.setattr(
        mm, "act_download", lambda evs, **kw: calls.append(("download", kw)))
    monkeypatch.setattr(
        mm, "act_trigger", lambda evs, **kw: calls.append(("trigger", kw)))
    monkeypatch.setattr(
        mm, "act_report", lambda evs, **kw: calls.append(("report", kw)))
    return calls


def test_auto_healthy_disk_runs_everything(monkeypatch, tmp_path):
    calls = _patch_poll(monkeypatch,
                        [_row("jw02221-o001_t001_nircam_clear-f405n")])
    monkeypatch.setattr(mm.shutil, "disk_usage", lambda p: _Usage(50e12))
    state = tmp_path / "state.json"
    rc = mm.main(["--program", "2221", "--auto", "--state", str(state),
                  "--download-dir", str(tmp_path)])
    assert rc == 0
    acted = dict(calls)
    assert set(acted) == {"download", "trigger", "report"}
    assert acted["download"]["execute"] is True
    assert acted["trigger"]["execute"] is True
    assert acted["report"]["execute"] is True
    assert acted["report"]["notice"] is None
    assert state.exists()                        # --auto commits state


def test_auto_low_disk_downgrades_to_report_only(monkeypatch, tmp_path, capsys):
    calls = _patch_poll(monkeypatch,
                        [_row("jw02221-o001_t001_nircam_clear-f405n")])
    monkeypatch.setattr(mm.shutil, "disk_usage", lambda p: _Usage(1e12))
    state = tmp_path / "state.json"
    rc = mm.main(["--program", "2221", "--auto", "--min-free-tb", "5",
                  "--state", str(state), "--download-dir", str(tmp_path)])
    assert rc == 0
    acted = dict(calls)
    assert set(acted) == {"report"}              # no download, no trigger
    assert acted["report"]["execute"] is True    # the report still posts...
    assert "LOW DISK" in acted["report"]["notice"]   # ...with the loud warning
    assert not state.exists()                    # state NOT committed: re-fires
    assert "LOW DISK" in capsys.readouterr().err
