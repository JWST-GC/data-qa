"""Offline unit tests for data_qa.mast_monitor (state diffing; no network)."""
import datetime
import inspect
import json
import os

import pytest

from data_qa import mast_monitor as mm
from data_qa import paths
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
    unreleased = {"cloudef", "cloudef_controlfield", "sgra", "ngc6334",
                  "arches", "quintuplet"}          # no release page yet
    for prog, obsmap in mm.PROGRAMS.items():
        for field in obsmap.values():
            assert field in FIELDS or field in unreleased, (prog, field)


def test_gc_fields_membership():
    """DEBLEND_SATSTARS keys off GC_FIELDS: every inner-GC/CMZ field is in, and
    the 1182 brick+w51 split must never sweep w51 (or any other non-GC field)
    into the set."""
    # gc2211 is split into one field per observation (pipeline #469, issue #119)
    assert {"gc-treasury", "brick", "cloudc", "sgrc", "sgrb2",
            "arches", "quintuplet", "sickle", "cloudef", "sgra"} <= mm.GC_FIELDS
    assert {"gc2211_o023", "gc2211_o028", "gc2211_o046",
            "gc2211_o049", "gc2211_o050"} <= mm.GC_FIELDS
    assert not {"w51", "wd1", "wd2", "ngc6334"} & mm.GC_FIELDS
    # every GC field is a field the monitor can actually map to
    mappable = {f for obsmap in mm.PROGRAMS.values() for f in obsmap.values()}
    mappable.add(mm.TREASURY_FIELD)
    assert mm.GC_FIELDS <= mappable


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


def _seed_state(path, program=2221, obs_id="jw02221-o009_x"):
    """Write a non-empty baseline state so a run is NOT a first run."""
    mm.save_state(str(path), {
        "version": 1,
        "programs": {str(program): {"obs": {obs_id: {"calib_level": 3}}}}})


def test_auto_healthy_disk_runs_everything(monkeypatch, tmp_path):
    calls = _patch_poll(monkeypatch,
                        [_row("jw02221-o001_t001_nircam_clear-f405n")])
    monkeypatch.setattr(mm.shutil, "disk_usage", lambda p: _Usage(50e12))
    state = tmp_path / "state.json"
    _seed_state(state)                           # baseline: not a first run
    rc = mm.main(["--program", "2221", "--auto", "--state", str(state),
                  "--download-dir", str(tmp_path)])
    assert rc == 0
    acted = dict(calls)
    assert set(acted) == {"download", "trigger", "report"}
    assert acted["download"]["execute"] is True
    assert acted["trigger"]["execute"] is True
    assert acted["report"]["execute"] is True
    assert acted["report"]["notice"] is None
    committed = mm.load_state(str(state))        # --auto commits state
    assert "jw02221-o001_t001_nircam_clear-f405n" in \
        committed["programs"]["2221"]["obs"]


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


# ------------------------------------------------------------- first-run seed (HIGH-1a)
def test_auto_first_run_is_seed_only(monkeypatch, tmp_path, capsys):
    """Missing state + --auto: commit the baseline, act on NOTHING (no herd)."""
    calls = _patch_poll(monkeypatch,
                        [_row("jw02221-o001_t001_nircam_clear-f405n"),
                         _row("jw02221-o002_t001_nircam_clear-f405n")])
    monkeypatch.setattr(mm.shutil, "disk_usage", lambda p: _Usage(50e12))
    state = tmp_path / "state.json"
    rc = mm.main(["--program", "2221", "--auto", "--state", str(state),
                  "--download-dir", str(tmp_path)])
    assert rc == 0
    acted = dict(calls)
    assert set(acted) == {"report"}              # no download, no trigger
    assert "SEED RUN" in acted["report"]["notice"]
    assert state.exists()                        # baseline committed
    committed = mm.load_state(str(state))
    assert len(committed["programs"]["2221"]["obs"]) == 2
    assert "SEED RUN" in capsys.readouterr().err


def test_execute_first_run_is_seed_only(monkeypatch, tmp_path):
    """--download --trigger --execute (non-auto) on an empty state also seeds."""
    calls = _patch_poll(monkeypatch,
                        [_row("jw02221-o001_t001_nircam_clear-f405n")])
    state = tmp_path / "state.json"
    rc = mm.main(["--program", "2221", "--download", "--trigger", "--execute",
                  "--state", str(state)])
    assert rc == 0
    assert calls == []                           # nothing acted (no report asked)
    assert state.exists()                        # but the baseline committed


def test_first_run_dry_run_unchanged(monkeypatch, tmp_path):
    """Without --execute the first run still dry-runs the actions as before."""
    calls = _patch_poll(monkeypatch,
                        [_row("jw02221-o001_t001_nircam_clear-f405n")])
    state = tmp_path / "state.json"
    rc = mm.main(["--program", "2221", "--download", "--trigger",
                  "--state", str(state)])
    assert rc == 0
    acted = dict(calls)
    assert set(acted) == {"download", "trigger"}
    assert acted["trigger"]["execute"] is False
    assert not state.exists()                    # dry-run commits nothing


def test_seed_verb_commits_without_acting(monkeypatch, tmp_path, capsys):
    calls = _patch_poll(monkeypatch,
                        [_row("jw02221-o001_t001_nircam_clear-f405n")])
    state = tmp_path / "state.json"
    rc = mm.main(["--program", "2221", "--seed", "--state", str(state)])
    assert rc == 0
    assert calls == []
    assert state.exists()
    assert "SEED RUN" in capsys.readouterr().err


# ------------------------------------------------------------- submission cap (HIGH-1b)
def _rows_n_groups(n):
    return [_row(f"jw02221-o{i:03d}_t001_nircam_clear-f405n")
            for i in range(1, n + 1)]


def _patch_poll_events(monkeypatch, rows):
    """_patch_poll, but the recorded calls keep the EVENT LIST each action
    received: (name, events, kwargs)."""
    calls = []
    monkeypatch.setattr(mm, "mast_login_if_token", lambda: False)
    monkeypatch.setattr(mm, "query_program", lambda prog: list(rows))
    monkeypatch.setattr(mm, "act_download",
                        lambda evs, **kw: calls.append(("download", list(evs), kw)))
    monkeypatch.setattr(mm, "act_trigger",
                        lambda evs, **kw: calls.append(("trigger", list(evs), kw)))
    monkeypatch.setattr(mm, "act_report",
                        lambda evs, **kw: calls.append(("report", list(evs), kw)))
    monkeypatch.setattr(mm, "act_peppar",
                        lambda evs, **kw: calls.append(("peppar", list(evs), kw)))
    return calls


def test_capped_run_acts_on_cap_and_commits_acted(monkeypatch, tmp_path, capsys):
    """3 actionable groups vs cap 2: act on the 2 oldest, commit exactly those,
    defer the third (its record stays out of the commit, so it re-fires).
    Treasury rows: every 10678 obsnum is field-mapped."""
    rows = [_row(f"jw10678-o{i:03d}_t001_nircam_clear-f212n",
                 filters="F212N;F480M", target=f"GC_{i}") for i in (1, 2, 3)]
    calls = _patch_poll_events(monkeypatch, rows)
    monkeypatch.setattr(mm.shutil, "disk_usage", lambda p: _Usage(50e12))
    state = tmp_path / "state.json"
    _seed_state(state, program=10678, obs_id="jw10678-o999_x")   # not first run
    rc = mm.main(["--program", "10678", "--auto", "--max-submit", "2",
                  "--state", str(state), "--download-dir", str(tmp_path)])
    assert rc == 0
    acted = {name: (evs, kw) for name, evs, kw in calls}
    assert set(acted) == {"download", "trigger", "report"}
    assert set(mm._group_by_obs(acted["trigger"][0])) == {
        (10678, "001", "NIRCam"), (10678, "002", "NIRCam")}
    assert len(acted["report"][0]) == 3          # the report covers everything
    notice = acted["report"][1]["notice"]
    assert "CAPPED" in notice
    assert "10678-o003-NIRCam" in notice         # the deferred group is named
    committed = mm.load_state(str(state))["programs"]["10678"]["obs"]
    assert "jw10678-o001_t001_nircam_clear-f212n" in committed
    assert "jw10678-o002_t001_nircam_clear-f212n" in committed
    assert "jw10678-o003_t001_nircam_clear-f212n" not in committed   # re-fires
    assert "CAPPED" in capsys.readouterr().err


def test_under_cap_runs_everything(monkeypatch, tmp_path):
    calls = _patch_poll(monkeypatch, _rows_n_groups(2))
    monkeypatch.setattr(mm.shutil, "disk_usage", lambda p: _Usage(50e12))
    state = tmp_path / "state.json"
    _seed_state(state)
    rc = mm.main(["--program", "2221", "--auto", "--max-submit", "2",
                  "--state", str(state), "--download-dir", str(tmp_path)])
    assert rc == 0
    assert set(dict(calls)) == {"download", "trigger", "report"}


# --------------------------------------------- actionable-group counting (issue #67)
def _ready_event(program=10678, obsnum="001", instr="NIRCAM/IMAGE",
                 field="gc-treasury", release=59900.0, **over):
    ev = dict(event="NEW_OBSERVATION", program=program, obsnum=obsnum,
              obs_id=f"jw{program:05d}-o{obsnum}_x", field=field, tile=None,
              calib_level=3, released=True, t_obs_release=release,
              instrument_name=instr, filters="F212N;F480M",
              target_name=f"GC_{int(obsnum)}")
    ev.update(over)
    return ev


def test_actionable_groups_truth_table():
    """Only groups the run would ACT on consume --max-submit slots: ready +
    field-mapped, with an un-retired download or (NIRCam-only) trigger side."""
    st = {"triggered": {"10678-o001": "t", "10678-o007": "t"},
          "downloaded": {"10678-o001-NIRCam": "t", "10678-o002-MIRI": "t",
                         "10678-o003-NIRCam": "t"}}
    events = [
        _ready_event(obsnum="001"),                       # both retired -> out
        _ready_event(obsnum="002", instr="MIRI/IMAGE"),   # downloaded; MIRI
                                                          # never triggers -> out
        _ready_event(obsnum="003"),                       # trigger armed -> IN
        _ready_event(obsnum="004", instr="MIRI/IMAGE"),   # download armed -> IN
        _ready_event(obsnum="005", calib_level=-1, released=False,
                     t_obs_release=None),                 # planned -> out
        _ready_event(program=9999, obsnum="006", field=""),   # unmapped -> out
        _ready_event(obsnum="007"),                       # download armed -> IN
        _ready_event(obsnum="008"),                       # fresh -> IN
    ]
    assert set(mm.actionable_groups(st, events)) == {
        (10678, "003", "NIRCam"), (10678, "004", "MIRI"),
        (10678, "007", "NIRCam"), (10678, "008", "NIRCam")}


def test_actionable_groups_trigger_dimension_is_nircam_only():
    """MIRI groups never consume trigger slots (act_trigger SKIPs them)."""
    events = [_ready_event(obsnum="001"),
              _ready_event(obsnum="001", instr="MIRI/IMAGE")]
    groups = mm.actionable_groups({}, events, download=False, trigger=True)
    assert set(groups) == {(10678, "001", "NIRCam")}


def test_actionable_groups_download_dimension_respects_downloaded_map():
    st = {"downloaded": {"10678-o001-NIRCam": "t"}}
    events = [_ready_event(obsnum="001")]
    assert mm.actionable_groups(st, events, download=True, trigger=False) == {}


def test_actionable_groups_peppar_dimension_never_retires():
    """act_peppar has NO one-shot state key, so --peppar keeps a ready NIRCam
    group actionable even when its triggered/downloaded keys are both burned.
    Without this dimension the group falls out of the mapping, is dropped from
    the capped run's truncated event list and is never deferred -- its peppar
    fan-out is lost instead of postponed."""
    st = {"triggered": {"10678-o001": "t"},
          "downloaded": {"10678-o001-NIRCam": "t"}}
    events = [_ready_event(obsnum="001")]
    assert mm.actionable_groups(st, events) == {}          # download+trigger: retired
    assert set(mm.actionable_groups(st, events, peppar=True)) == {
        (10678, "001", "NIRCam")}
    # peppar alone never counts a MIRI group (act_peppar SKIPs it as NIRCam-only)
    miri = [_ready_event(obsnum="002", instr="MIRI/IMAGE")]
    assert mm.actionable_groups({}, miri, download=False, trigger=False,
                                peppar=True) == {}
    # and it stays off unless asked for: default peppar=False
    assert mm.actionable_groups(st, events, download=False, trigger=False) == {}


def test_actionable_groups_ordered_oldest_first():
    events = [_ready_event(obsnum="003", release=59903.0),
              _ready_event(obsnum="001", release=59901.0),
              _ready_event(obsnum="002", release=59902.0)]
    assert list(mm.actionable_groups({}, events)) == [
        (10678, "001", "NIRCam"), (10678, "002", "NIRCam"),
        (10678, "003", "NIRCam")]


def test_actionable_groups_multi_event_group_sorts_on_its_oldest():
    """A group carries several events (NEWLY_RELEASED then CALIB_LEVEL_UP).  It
    takes its slot priority from the OLDEST ready release it holds, so the
    obs that has been waiting longest drains first; ``max`` would rank a group
    by its freshest event and let an old group sit behind a new one."""
    events = [_ready_event(obsnum="001", release=59910.0),
              _ready_event(obsnum="001", release=59902.0,
                           obs_id="jw10678-o001_y", event="CALIB_LEVEL_UP"),
              _ready_event(obsnum="002", release=59905.0)]
    assert list(mm.actionable_groups({}, events)) == [
        (10678, "001", "NIRCam"), (10678, "002", "NIRCam")]


def test_actionable_groups_age_ignores_unreleased_events():
    """Only READY events date a group.  A planned row riding in the same group
    carries a scheduled release far in the past of the real delivery; ranking
    on it would jump the group to the head of the cap queue on the strength of
    data that does not exist yet."""
    events = [_ready_event(obsnum="001", release=59920.0),
              _ready_event(obsnum="001", obs_id="jw10678-o001_planned",
                           release=59000.0, calib_level=-1, released=False),
              _ready_event(obsnum="002", release=59905.0)]
    assert list(mm.actionable_groups({}, events)) == [
        (10678, "002", "NIRCam"), (10678, "001", "NIRCam")]


def test_actionable_groups_undated_group_sorts_last():
    """A ready group whose release date is unknown sorts to the BACK (float
    'inf'), so a dated backlog drains ahead of it; '-inf' would let one
    undated row displace every dated group from the cap."""
    events = [_ready_event(obsnum="001", release=None),
              _ready_event(obsnum="002", release=59905.0)]
    assert list(mm.actionable_groups({}, events)) == [
        (10678, "002", "NIRCam"), (10678, "001", "NIRCam")]


def test_actionable_groups_ties_break_on_group_key():
    """One MAST delivery timestamps a whole visit set identically.  The group
    key breaks the tie, so a capped run acts on the same groups whichever order
    MAST returns its rows in (that order varies between polls)."""
    same = 59905.0
    forward = [_ready_event(obsnum=n, release=same) for n in ("001", "002", "003")]
    assert (list(mm.actionable_groups({}, forward))
            == list(mm.actionable_groups({}, list(reversed(forward))))
            == [(10678, "001", "NIRCam"), (10678, "002", "NIRCam"),
                (10678, "003", "NIRCam")])


def test_revert_deferred_restores_baseline_records():
    """Deferred groups keep their pre-poll baseline: an obs absent from the
    baseline is popped (re-fires NEW), a changed one is restored (re-fires
    NEWLY_RELEASED / CALIB_LEVEL_UP)."""
    state = {"programs": {"2221": {"obs": {
        "a": {"calib_level": 3, "released": True},
        "b": {"calib_level": 3, "released": True}}}}}
    old = {"2221": {"a": {"calib_level": 2, "released": False}}}
    mm._revert_deferred(state, old, [dict(program=2221, obs_id="a"),
                                     dict(program=2221, obs_id="b")])
    assert state["programs"]["2221"]["obs"] == {
        "a": {"calib_level": 2, "released": False}}


def _visit_rows(obsnum, release):
    """One delivered 10678 visit = TWO MAST groups: NIRCam F212N;F480M prime +
    MIRI F770W parallel."""
    return [
        {"obs_id": f"jw10678-o{obsnum}_t001_nircam_clear-f212n",
         "t_max": release - 1, "t_obs_release": release, "calib_level": 3,
         "instrument_name": "NIRCAM/IMAGE", "filters": "F212N;F480M",
         "target_name": f"GC_{int(obsnum)}"},
        {"obs_id": f"jw10678-o{obsnum}_t001_miri_f770w",
         "t_max": release - 1, "t_obs_release": release, "calib_level": 3,
         "instrument_name": "MIRI/IMAGE", "filters": "F770W",
         "target_name": f"GC_{int(obsnum)}"},
    ]


def test_treasury_delivery_morning_drains_incrementally(monkeypatch, tmp_path,
                                                        capsys):
    """The issue-#67 shape: 5 delivered 10678 visits = 10 groups vs cap 4.
    Run 1 acts on the 4 oldest and commits exactly those; run 2 acts on the
    next 4 (the first 4 do NOT recount); run 3 drains the last 2 uncapped."""
    rows = []
    for i in (1, 2, 3, 4, 5):
        rows += _visit_rows(f"{i:03d}", release=59900.0 + i)
    calls = _patch_poll_events(monkeypatch, rows)
    monkeypatch.setattr(mm.shutil, "disk_usage", lambda p: _Usage(50e12))
    state = tmp_path / "state.json"
    _seed_state(state, program=10678, obs_id="jw10678-o999_x")
    args = ["--program", "10678", "--auto", "--max-submit", "4",
            "--state", str(state), "--download-dir", str(tmp_path)]

    assert mm.main(args) == 0                    # ---------------------- run 1
    acted = {name: (evs, kw) for name, evs, kw in calls}
    assert set(mm._group_by_obs(acted["trigger"][0])) == {
        (10678, "001", "MIRI"), (10678, "001", "NIRCam"),
        (10678, "002", "MIRI"), (10678, "002", "NIRCam")}
    assert acted["trigger"][0] == acted["download"][0]   # same acted subset
    assert len(acted["report"][0]) == 10         # the report covers everything
    notice = acted["report"][1]["notice"]
    assert "CAPPED" in notice
    assert "10678-o003-MIRI" in notice           # deferred groups are named
    assert "10678-o005-NIRCam" in notice
    committed = mm.load_state(str(state))["programs"]["10678"]["obs"]
    for i in (1, 2):                             # acted: committed, retired
        assert f"jw10678-o{i:03d}_t001_nircam_clear-f212n" in committed
        assert f"jw10678-o{i:03d}_t001_miri_f770w" in committed
    for i in (3, 4, 5):                          # deferred: NOT committed
        assert f"jw10678-o{i:03d}_t001_nircam_clear-f212n" not in committed
        assert f"jw10678-o{i:03d}_t001_miri_f770w" not in committed
    assert "CAPPED" in capsys.readouterr().err

    calls.clear()
    assert mm.main(args) == 0                    # ---------------------- run 2
    acted = {name: (evs, kw) for name, evs, kw in calls}
    assert set(mm._group_by_obs(acted["trigger"][0])) == {
        (10678, "003", "MIRI"), (10678, "003", "NIRCam"),
        (10678, "004", "MIRI"), (10678, "004", "NIRCam")}
    # the 4 groups run 1 retired do NOT recount: no visit-1/2 event re-fires
    assert not any(ev["obsnum"] in ("001", "002") for ev in acted["report"][0])
    assert "CAPPED" in acted["report"][1]["notice"]

    calls.clear()
    assert mm.main(args) == 0                    # ---------------------- run 3
    acted = {name: (evs, kw) for name, evs, kw in calls}
    assert set(mm._group_by_obs(acted["trigger"][0])) == {
        (10678, "005", "MIRI"), (10678, "005", "NIRCam")}
    assert acted["report"][1]["notice"] is None  # under the cap: no CAPPED


def test_planned_treasury_events_do_not_trip_the_cap(monkeypatch, tmp_path):
    """PLANNED 10678 rows (calib -1, unreleased) fire NEW_OBSERVATION but
    consume no submissions; main() used to count them against --max-submit and
    wedge --auto into report-only long before any data existed."""
    rows = []
    for i in range(101, 109):                    # 8 planned tiles
        rows.append({"obs_id": f"jw10678-o{i}_t001_nircam_clear-f212n",
                     "t_max": None, "t_obs_release": None, "calib_level": -1,
                     "instrument_name": "NIRCAM/IMAGE",
                     "filters": "F212N;F480M", "target_name": f"GC_{i}"})
    rows += _visit_rows("001", release=59901.0)  # one real delivery: 2 groups
    calls = _patch_poll_events(monkeypatch, rows)
    monkeypatch.setattr(mm.shutil, "disk_usage", lambda p: _Usage(50e12))
    state = tmp_path / "state.json"
    _seed_state(state, program=10678, obs_id="jw10678-o999_x")
    rc = mm.main(["--program", "10678", "--auto", "--max-submit", "4",
                  "--state", str(state), "--download-dir", str(tmp_path)])
    assert rc == 0
    acted = {name: (evs, kw) for name, evs, kw in calls}
    assert set(acted) == {"download", "trigger", "report"}   # NOT capped
    assert acted["report"][1]["notice"] is None
    assert len(acted["report"][0]) == 10         # planned tiles still reported


def _nircam_row(obsnum, release, calib=3):
    return {"obs_id": f"jw10678-o{obsnum}_t001_nircam_clear-f212n",
            "t_max": release - 1, "t_obs_release": release,
            "calib_level": calib, "instrument_name": "NIRCAM/IMAGE",
            "filters": "F212N;F480M", "target_name": f"GC_{int(obsnum)}"}


def test_capped_run_does_not_drop_peppar_for_retired_groups(monkeypatch,
                                                            tmp_path):
    """A capped run acts on the counted groups ONLY, so every enabled action
    must be a counted dimension.  o001's triggered+downloaded keys are already
    burned, but --peppar is still live for it (act_peppar keeps no one-shot
    key): it must either be acted on (peppar sees it) or deferred (its record
    stays out of the commit so it re-fires) -- never dropped-and-committed,
    which loses the fan-out permanently."""
    rows = [_nircam_row("001", release=59901.0),          # oldest, keys burned
            _nircam_row("002", release=59902.0),
            _nircam_row("003", release=59903.0)]
    calls = _patch_poll_events(monkeypatch, rows)
    monkeypatch.setattr(mm.shutil, "disk_usage", lambda p: _Usage(50e12))
    state = tmp_path / "state.json"
    mm.save_state(str(state), {
        "version": 1,
        "programs": {"10678": {"obs": {
            # o001 already seen at calib 2 -> this poll fires CALIB_LEVEL_UP
            "jw10678-o001_t001_nircam_clear-f212n": {
                "calib_level": 2, "released": True, "t_obs_release": 59901.0,
                "instrument_name": "NIRCAM/IMAGE", "filters": "F212N;F480M",
                "target_name": "GC_1"}}}},
        "triggered": {"10678-o001": "2026-08-16"},
        "downloaded": {"10678-o001-NIRCam": "2026-08-16"}})
    args = ["--program", "10678", "--auto", "--peppar", "--max-submit", "2",
            "--state", str(state), "--download-dir", str(tmp_path)]

    assert mm.main(args) == 0                    # ---------------------- run 1
    acted = {name: (evs, kw) for name, evs, kw in calls}
    # peppar counts o001, so the two oldest ACTED groups are o001 and o002
    assert set(mm._group_by_obs(acted["peppar"][0])) == {
        (10678, "001", "NIRCam"), (10678, "002", "NIRCam")}
    committed = mm.load_state(str(state))["programs"]["10678"]["obs"]
    assert "jw10678-o003_t001_nircam_clear-f212n" not in committed  # deferred

    calls.clear()
    assert mm.main(args) == 0                    # ---------------------- run 2
    acted = {name: (evs, kw) for name, evs, kw in calls}
    # the deferred group re-fires and gets its peppar fan-out on the next run
    assert set(mm._group_by_obs(acted["peppar"][0])) == {
        (10678, "003", "NIRCam")}


def test_duplicated_program_keeps_the_true_prepoll_baseline(monkeypatch,
                                                            tmp_path):
    """`--program 10678 10678` polls the same program twice.  The second pass
    must not overwrite the pre-poll baseline with the first pass's POST-poll
    records, or _revert_deferred would 'restore' deferred groups to what MAST
    just returned: they would commit as seen and never re-fire."""
    rows = [_nircam_row(f"{i:03d}", release=59900.0 + i) for i in (1, 2, 3)]
    calls = _patch_poll_events(monkeypatch, rows)
    monkeypatch.setattr(mm.shutil, "disk_usage", lambda p: _Usage(50e12))
    state = tmp_path / "state.json"
    _seed_state(state, program=10678, obs_id="jw10678-o999_x")
    rc = mm.main(["--program", "10678", "10678", "--auto", "--max-submit", "2",
                  "--state", str(state), "--download-dir", str(tmp_path)])
    assert rc == 0
    committed = mm.load_state(str(state))["programs"]["10678"]["obs"]
    assert "jw10678-o003_t001_nircam_clear-f212n" not in committed   # deferred
    assert calls                                  # the run really acted


def test_capped_run_does_not_drop_trigger_for_download_retired_groups(
        monkeypatch, tmp_path):
    """Mirror of the peppar test for the TRIGGER dimension: o001's
    ``downloaded`` key is burned while ``triggered`` is still armed.  Dropping
    trigger from the count leaves o001 out of the truncated event list, so it
    is committed as seen and never submitted on any later run."""
    rows = [_nircam_row(f"{i:03d}", release=59900.0 + i) for i in (1, 2, 3, 4, 5)]
    calls = _patch_poll_events(monkeypatch, rows)
    monkeypatch.setattr(mm.shutil, "disk_usage", lambda p: _Usage(50e12))
    state = tmp_path / "state.json"
    mm.save_state(str(state), {
        "version": 1,
        "programs": {"10678": {"obs": {
            # o001 already seen at calib 2 -> this poll fires CALIB_LEVEL_UP
            "jw10678-o001_t001_nircam_clear-f212n": {
                "calib_level": 2, "released": True, "t_obs_release": 59901.0,
                "instrument_name": "NIRCAM/IMAGE", "filters": "F212N;F480M",
                "target_name": "GC_1"}}}},
        "downloaded": {"10678-o001-NIRCam": "2026-08-16"}})
    args = ["--program", "10678", "--auto", "--max-submit", "2",
            "--state", str(state), "--download-dir", str(tmp_path)]

    assert mm.main(args) == 0                    # ---------------------- run 1
    acted = {name: (evs, kw) for name, evs, kw in calls}
    # the trigger dimension counts o001, so it is one of the two oldest ACTED
    assert set(mm._group_by_obs(acted["trigger"][0])) == {
        (10678, "001", "NIRCam"), (10678, "002", "NIRCam")}
    committed = mm.load_state(str(state))["programs"]["10678"]["obs"]
    assert committed["jw10678-o001_t001_nircam_clear-f212n"]["calib_level"] == 3
    for i in (3, 4, 5):
        assert f"jw10678-o{i:03d}_t001_nircam_clear-f212n" not in committed

    calls.clear()
    assert mm.main(args) == 0                    # ---------------------- run 2
    acted = {name: (evs, kw) for name, evs, kw in calls}
    assert set(mm._group_by_obs(acted["trigger"][0])) == {
        (10678, "003", "NIRCam"), (10678, "004", "NIRCam")}
    # o001 was acted on in run 1 and is retired: it never re-fires, so run 1
    # was its only chance to be triggered
    assert not any(ev["obsnum"] == "001" for ev in acted["report"][0])


@pytest.mark.parametrize("action", mm.ACTION_FLAGS)
def test_single_action_run_is_seeded_and_capped(monkeypatch, tmp_path, capsys,
                                                action):
    """EACH action flag on its own makes an acting run, so the first-run seed
    guard and the --max-submit cap both apply to it.  Dropping any one term
    from ``acting`` (or from the seed suppression) lets that flag skip both
    gates: the whole backlog fans out on the very first poll and commits, with
    no later run to pick it up.  Parametrized over ACTION_FLAGS so a new action
    is defended the day it is added."""
    rows = [_nircam_row(f"{i:03d}", release=59900.0 + i) for i in range(1, 6)]
    calls = _patch_poll_events(monkeypatch, rows)
    state = tmp_path / "state.json"
    args = ["--program", "10678", f"--{action}", "--execute", "--report",
            "--commit-state", "--max-submit", "2", "--state", str(state),
            "--download-dir", str(tmp_path)]

    assert mm.main(args) == 0                    # ------- run 1: first run
    assert [name for name, _evs, _kw in calls if name == action] == []
    assert "SEED RUN" in capsys.readouterr().err

    # a second poll returns 5 MORE groups on top of the seeded baseline
    rows.extend(_nircam_row(f"{i:03d}", release=59900.0 + i)
                for i in range(6, 11))
    calls.clear()
    assert mm.main(args) == 0                    # ------- run 2: capped
    acted = {name: (evs, kw) for name, evs, kw in calls}
    assert set(mm._group_by_obs(acted[action][0])) == {
        (10678, "006", "NIRCam"), (10678, "007", "NIRCam")}
    assert "CAPPED" in acted["report"][1]["notice"]


def test_action_flags_match_actionable_groups_dimensions():
    """ACTION_FLAGS is the single list main() spreads into `acting`, the seed
    suppression and the actionable_groups call.  An action added to
    actionable_groups without being added here would be counted but never
    enabled; one added to argparse alone would be enabled but never counted --
    the loss this cap exists to stop.  Pin the two together."""
    params = inspect.signature(mm.actionable_groups).parameters
    dimensions = tuple(name for name in params
                       if name not in ("state", "events"))
    assert dimensions == mm.ACTION_FLAGS
    # the argparse side is pinned by test_single_action_run_is_seeded_and_capped,
    # which passes --<flag> on the command line for every ACTION_FLAGS entry


def test_exactly_at_the_cap_is_not_capped(monkeypatch, tmp_path):
    """The cap trips on STRICT overflow.  `>=` would post a CAPPED notice with
    an empty deferred list at equality and truncate the event list the act_*
    functions see."""
    rows = [_nircam_row(f"{i:03d}", release=59900.0 + i) for i in (1, 2)]
    calls = _patch_poll_events(monkeypatch, rows)
    monkeypatch.setattr(mm.shutil, "disk_usage", lambda p: _Usage(50e12))
    state = tmp_path / "state.json"
    _seed_state(state, program=10678, obs_id="jw10678-o999_x")
    assert mm.main(["--program", "10678", "--auto", "--max-submit", "2",
                    "--state", str(state),
                    "--download-dir", str(tmp_path)]) == 0
    acted = {name: (evs, kw) for name, evs, kw in calls}
    assert acted["report"][1]["notice"] is None
    assert set(mm._group_by_obs(acted["trigger"][0])) == {
        (10678, "001", "NIRCam"), (10678, "002", "NIRCam")}


def test_capped_notice_keeps_an_earlier_notice(monkeypatch, tmp_path):
    """PER-PROGRAM SEED and CAPPED both describe the same run; the CAPPED text
    is appended so the issue comment carries both."""
    rows = {10678: [_nircam_row(f"{i:03d}", release=59900.0 + i)
                    for i in (1, 2, 3)],
            2221: [_row("jw02221-o001_t001_nircam_clear-f405n")]}
    calls = []
    monkeypatch.setattr(mm, "mast_login_if_token", lambda: False)
    monkeypatch.setattr(mm, "query_program", lambda prog: list(rows[prog]))
    for name in ("act_download", "act_trigger", "act_report", "act_peppar"):
        monkeypatch.setattr(
            mm, name,
            (lambda nm: lambda evs, **kw: calls.append((nm, list(evs), kw)))(
                name.removeprefix("act_")))
    monkeypatch.setattr(mm.shutil, "disk_usage", lambda p: _Usage(50e12))
    state = tmp_path / "state.json"
    _seed_state(state, program=10678, obs_id="jw10678-o999_x")   # 2221 unseeded
    assert mm.main(["--program", "10678", "2221", "--auto", "--max-submit", "2",
                    "--state", str(state),
                    "--download-dir", str(tmp_path)]) == 0
    notice = {name: kw for name, _evs, kw in calls}["report"]["notice"]
    assert "PER-PROGRAM SEED" in notice
    assert "CAPPED" in notice


def test_capped_notice_tracks_commit_state(monkeypatch, tmp_path):
    """The notice is posted verbatim onto the QA issue.  An acting run without
    --commit-state commits nothing, so the notice says so."""
    rows = [_nircam_row(f"{i:03d}", release=59900.0 + i) for i in (1, 2, 3)]
    calls = _patch_poll_events(monkeypatch, rows)
    monkeypatch.setattr(mm.shutil, "disk_usage", lambda p: _Usage(50e12))
    state = tmp_path / "state.json"
    _seed_state(state, program=10678, obs_id="jw10678-o999_x")
    base = ["--program", "10678", "--download", "--trigger", "--report",
            "--execute", "--max-submit", "2", "--state", str(state),
            "--download-dir", str(tmp_path)]

    assert mm.main(base) == 0                    # no --commit-state
    notice = {name: kw for name, _evs, kw in calls}["report"]["notice"]
    assert "commits no state" in notice
    assert "is committed now" not in notice
    committed = mm.load_state(str(state))["programs"]["10678"]["obs"]
    assert "jw10678-o001_t001_nircam_clear-f212n" not in committed

    calls.clear()
    assert mm.main(base + ["--commit-state"]) == 0
    notice = {name: kw for name, _evs, kw in calls}["report"]["notice"]
    assert "is committed now" in notice
    assert "commits no state" not in notice


def test_capped_notice_counts_all_actionable_groups(monkeypatch, tmp_path):
    """The head count in the notice is the number of ACTIONABLE groups, which
    is the number an operator reads to size --max-submit.  Reporting the acted
    count there renders the self-contradictory 'N group(s) exceed --max-submit
    N: acting on the N oldest'."""
    rows = [_nircam_row(f"{i:03d}", release=59900.0 + i) for i in range(1, 8)]
    calls = _patch_poll_events(monkeypatch, rows)
    monkeypatch.setattr(mm.shutil, "disk_usage", lambda p: _Usage(50e12))
    state = tmp_path / "state.json"
    _seed_state(state, program=10678, obs_id="jw10678-o999_x")
    assert mm.main(["--program", "10678", "--auto", "--max-submit", "2",
                    "--state", str(state),
                    "--download-dir", str(tmp_path)]) == 0
    notice = {name: kw for name, _evs, kw in calls}["report"]["notice"]
    assert ("CAPPED — 7 actionable group(s) exceed --max-submit 2: "
            "acting on the 2 oldest") in notice


def test_default_max_submit_caps_a_run_that_passes_no_flag(monkeypatch,
                                                           tmp_path):
    """Every other cap test passes --max-submit explicitly, which leaves the
    SHIPPED default undefended -- and the default governs any invocation
    lacking the flag, the live scrontab included until the owed hand-edit
    lands.  Six groups with no flag cap at DEFAULT_MAX_SUBMIT."""
    assert mm.DEFAULT_MAX_SUBMIT == 4
    rows = [_nircam_row(f"{i:03d}", release=59900.0 + i) for i in range(1, 7)]
    calls = _patch_poll_events(monkeypatch, rows)
    monkeypatch.setattr(mm.shutil, "disk_usage", lambda p: _Usage(50e12))
    state = tmp_path / "state.json"
    _seed_state(state, program=10678, obs_id="jw10678-o999_x")
    assert mm.main(["--program", "10678", "--auto", "--state", str(state),
                    "--download-dir", str(tmp_path)]) == 0
    acted = {name: (evs, kw) for name, evs, kw in calls}
    assert set(mm._group_by_obs(acted["trigger"][0])) == {
        (10678, f"{i:03d}", "NIRCam") for i in range(1, 5)}
    assert f"--max-submit {mm.DEFAULT_MAX_SUBMIT}" in acted["report"][1]["notice"]


def test_capped_notice_truncates_long_group_lists(monkeypatch, tmp_path):
    """100 groups against --max-submit 16 named in full is a ~2.3 kB notice in
    every issue comment that run; the tail collapses to 'and N more'."""
    rows = [_nircam_row(f"{i:03d}", release=59900.0 + i) for i in range(1, 101)]
    calls = _patch_poll_events(monkeypatch, rows)
    monkeypatch.setattr(mm.shutil, "disk_usage", lambda p: _Usage(50e12))
    state = tmp_path / "state.json"
    _seed_state(state, program=10678, obs_id="jw10678-o999_x")
    assert mm.main(["--program", "10678", "--auto", "--max-submit", "16",
                    "--state", str(state),
                    "--download-dir", str(tmp_path)]) == 0
    notice = {name: kw for name, _evs, kw in calls}["report"]["notice"]
    assert "and 6 more" in notice                 # 16 acted, 10 named
    assert "and 74 more" in notice                # 84 deferred, 10 named
    assert len(notice) < 900
    # the truncation is display-only: all 16 oldest groups are still acted on
    acted = {name: (evs, kw) for name, evs, kw in calls}
    assert len(mm._group_by_obs(acted["trigger"][0])) == 16


def test_group_label_list_truncates_beyond_the_limit():
    keys = [(10678, f"{i:03d}", "NIRCam") for i in range(1, 15)]
    assert mm._group_label_list(keys[:3]) == (
        "10678-o001-NIRCam, 10678-o002-NIRCam, 10678-o003-NIRCam")
    out = mm._group_label_list(keys)
    assert out.endswith("and 4 more")
    assert out.count("10678-o") == 10
    assert "10678-o011-NIRCam" not in out
    # exactly AT the limit every label is named and the tail is absent: `>=`
    # here emits a bare "and 0 more" onto the QA issue
    exact = mm._group_label_list(keys[:mm.NOTICE_LABEL_LIMIT])
    assert exact.count("10678-o") == mm.NOTICE_LABEL_LIMIT
    assert "more" not in exact


def test_negative_max_submit_is_rejected(tmp_path):
    """keys[:-1] would act on all-but-the-last group -- the opposite of a cap."""
    with pytest.raises(SystemExit):
        mm.main(["--program", "10678", "--max-submit", "-1",
                 "--state", str(tmp_path / "state.json")])


# ----------------------------------------------------------- in-flight dedup (HIGH-1c)
def _trigger_events(obsnum="001", field="brick", filters="F405N;F410M"):
    return [dict(event="NEW_OBSERVATION", program=2221, obsnum=obsnum,
                 obs_id=f"jw02221-o{obsnum}_t001_nircam_clear-f405n",
                 field=field, tile=None, calib_level=3, released=True,
                 t_obs_release=59900.0, instrument_name="NIRCAM/IMAGE",
                 filters=filters, target_name="GAL_CENTER")]


def _patch_submit(monkeypatch):
    from data_qa import pipeline_trigger
    submitted = []
    monkeypatch.setattr(pipeline_trigger, "submit",
                        lambda **kw: submitted.append(kw))
    return submitted


def test_act_trigger_skips_inflight_job(monkeypatch, tmp_path, capsys):
    submitted = _patch_submit(monkeypatch)
    monkeypatch.setattr(mm, "inflight_job_names",
                        lambda: {"brick2221-o001-reduce", "other-job"})
    mm.act_trigger(_trigger_events("001"), execute=True,
                   state={}, state_path=str(tmp_path / "state.json"))
    assert submitted == []
    assert "SKIPPED(in-flight)" in capsys.readouterr().err


def test_act_trigger_skips_already_triggered(monkeypatch, tmp_path, capsys):
    submitted = _patch_submit(monkeypatch)
    monkeypatch.setattr(mm, "inflight_job_names", lambda: set())
    state = {"triggered": {"2221-o001": "2026-07-21 00:00 UTC"}}
    mm.act_trigger(_trigger_events("001"), execute=True,
                   state=state, state_path=str(tmp_path / "state.json"))
    assert submitted == []
    assert "SKIPPED(already-triggered)" in capsys.readouterr().err


def test_act_trigger_records_triggered_immediately(monkeypatch, tmp_path):
    """A successful submit persists the 'triggered' map to DISK at once, even
    though the event baselines are not committed."""
    submitted = _patch_submit(monkeypatch)
    monkeypatch.setattr(mm, "inflight_job_names", lambda: set())
    state_path = tmp_path / "state.json"
    state = {"version": 1, "programs": {}}
    mm.act_trigger(_trigger_events("001"), execute=True,
                   state=state, state_path=str(state_path))
    assert len(submitted) == 1
    on_disk = mm.load_state(str(state_path))
    assert "2221-o001" in on_disk["triggered"]        # persisted immediately
    assert on_disk["programs"] == {}                  # events NOT committed
    assert "2221-o001" in state["triggered"]          # mirrored in memory


def test_act_trigger_not_registered_keeps_key_armed(monkeypatch, tmp_path,
                                                    capsys):
    """The registry preflight failing (issue #68) must skip WITHOUT burning
    the one-shot key; once the registration lands, the SAME events submit."""
    from data_qa import pipeline_trigger

    def deny(**kw):
        raise pipeline_trigger.NotRegisteredInPipelineError(
            "program 10678 obs 001 is not registered in the pipeline")

    monkeypatch.setattr(pipeline_trigger, "submit", deny)
    monkeypatch.setattr(mm, "inflight_job_names", lambda: set())
    state_path = tmp_path / "state.json"
    state = {"version": 1, "programs": {}}
    events = _trigger_events("001")
    mm.act_trigger(events, execute=True, state=state,
                   state_path=str(state_path))
    err = capsys.readouterr().err
    assert "SKIPPED(not-registered)" in err
    assert "stays armed" in err
    assert "triggered" not in state              # key NOT burned...
    assert not state_path.exists()               # ...in memory or on disk
    submitted = _patch_submit(monkeypatch)       # registration lands
    mm.act_trigger(events, execute=True, state=state,
                   state_path=str(state_path))
    assert len(submitted) == 1
    assert "2221-o001" in state["triggered"]


def test_act_trigger_records_jobids_with_key(monkeypatch, tmp_path):
    """The triggered value is now {when, jobids}: the sbatch ids ride along
    for later outcome probing."""
    from data_qa import pipeline_trigger
    monkeypatch.setattr(
        pipeline_trigger, "submit",
        lambda **kw: dict(plan=[], results={}, jobids=["12345", "12346"]))
    monkeypatch.setattr(mm, "inflight_job_names", lambda: set())
    state_path = str(tmp_path / "state.json")
    state = {"version": 1, "programs": {}}
    mm.act_trigger(_trigger_events("001"), execute=True,
                   state=state, state_path=state_path)
    rec = mm.load_state(state_path)["triggered"]["2221-o001"]
    assert rec["jobids"] == ["12345", "12346"]
    assert "when" in rec


def test_already_triggered_legacy_bare_value_honored(monkeypatch, tmp_path,
                                                     capsys):
    """Pre-#68 state files hold bare timestamp strings; they still dedup."""
    submitted = _patch_submit(monkeypatch)
    monkeypatch.setattr(mm, "inflight_job_names", lambda: set())
    state = {"triggered": {"2221-o001": "2026-07-21 00:00 UTC"}}
    mm.act_trigger(_trigger_events("001"), execute=True,
                   state=state, state_path=str(tmp_path / "state.json"))
    assert submitted == []
    err = capsys.readouterr().err
    assert "SKIPPED(already-triggered)" in err
    assert "2026-07-21 00:00 UTC" in err


def test_already_triggered_message_recommends_rearm(monkeypatch, tmp_path,
                                                    capsys):
    """The guidance is the --rearm verb; hand-editing the state file is out."""
    submitted = _patch_submit(monkeypatch)
    monkeypatch.setattr(mm, "inflight_job_names", lambda: set())
    state = {"triggered": {"2221-o001": {"when": "2026-08-16 00:00 UTC",
                                         "jobids": ["12345"]}}}
    mm.act_trigger(_trigger_events("001"), execute=True,
                   state=state, state_path=str(tmp_path / "state.json"))
    assert submitted == []
    err = capsys.readouterr().err
    assert "--rearm 2221-o001" in err
    assert "at 2026-08-16 00:00 UTC (re-arm" in err   # the timestamp, unwrapped
    assert "jobids" not in err                        # not the raw record dict
    assert "delete the 'triggered' entry" not in err


def test_act_trigger_dry_run_does_not_record(monkeypatch, tmp_path):
    submitted = _patch_submit(monkeypatch)
    monkeypatch.setattr(mm, "inflight_job_names",
                        lambda: pytest.fail("squeue must not run on dry-run"))
    state_path = tmp_path / "state.json"
    mm.act_trigger(_trigger_events("001"), execute=False,
                   state={}, state_path=str(state_path))
    assert len(submitted) == 1
    assert submitted[0]["execute"] is False
    assert not state_path.exists()


# ---------------------------------------------------------- --rearm verb (issue #68)
def test_rearm_removes_and_rearms_end_to_end(monkeypatch, tmp_path, capsys):
    """Burn a key with a real (patched-submit) trigger, --rearm via the CLI,
    then the SAME events trigger again."""
    submitted = _patch_submit(monkeypatch)
    monkeypatch.setattr(mm, "inflight_job_names", lambda: set())
    state_path = str(tmp_path / "state.json")
    mm.act_trigger(_trigger_events("001"), execute=True,
                   state={"version": 1, "programs": {}}, state_path=state_path)
    assert len(submitted) == 1                   # burned
    mm.act_trigger(_trigger_events("001"), execute=True,
                   state=mm.load_state(state_path), state_path=state_path)
    assert len(submitted) == 1                   # one-shot holds
    rc = mm.main(["--rearm", "2221-o001", "--state", state_path])
    assert rc == 0
    assert "removed triggered[2221-o001]" in capsys.readouterr().out
    assert "2221-o001" not in mm.load_state(state_path).get("triggered", {})
    mm.act_trigger(_trigger_events("001"), execute=True,
                   state=mm.load_state(state_path), state_path=state_path)
    assert len(submitted) == 2                   # re-armed


def test_rearm_refuses_on_no_match(tmp_path, capsys):
    state_path = str(tmp_path / "state.json")
    mm.save_state(state_path, {"version": 1, "programs": {}})
    rc = mm.main(["--rearm", "2221-o001", "--state", state_path])
    assert rc == 1
    assert "REFUSED" in capsys.readouterr().err


def test_rearm_refuses_malformed_spec(tmp_path, capsys):
    rc = mm.main(["--rearm", "brick-001", "--state", str(tmp_path / "s.json")])
    assert rc == 1
    assert "REFUSED" in capsys.readouterr().err


@pytest.mark.parametrize("spec", [
    "2221-o001\n",                  # '$' would accept a trailing newline
    "2221-o001 --and-something",
    "2221-o001; rm -rf /",
    "2221-o001WRONG",
])
def test_rearm_refuses_a_spec_with_trailing_junk(tmp_path, capsys, spec):
    """fullmatch, not match: the operator gets a refusal on a mistyped spec
    instead of a silent re-arm of whatever prefix happened to parse."""
    state_path = str(tmp_path / "state.json")
    mm.save_state(state_path, {
        "version": 1, "programs": {},
        "triggered": {"2221-o001": {"when": "2026-08-16 00:00 UTC",
                                    "jobids": []}}})
    assert mm.main(["--rearm", spec, "--state", state_path]) == 1
    assert "REFUSED" in capsys.readouterr().err
    assert "2221-o001" in mm.load_state(state_path)["triggered"]


def test_rearm_download_also_clears_download_keys(tmp_path, capsys):
    """--rearm-download clears the obs's instrument-qualified downloaded keys
    too; other observations' keys stay."""
    state_path = str(tmp_path / "state.json")
    mm.save_state(state_path, {
        "version": 1, "programs": {},
        "triggered": {"2221-o002": {"when": "2026-08-16 00:00 UTC",
                                    "jobids": ["7"]}},
        "downloaded": {"2221-o002-NIRCam": "2026-08-16 00:00 UTC",
                       "2221-o002-MIRI": "2026-08-16 00:00 UTC",
                       "2221-o001-NIRCam": "keep-me"}})
    rc = mm.main(["--rearm", "2221-o002", "--rearm-download",
                  "--state", state_path])
    assert rc == 0
    out = capsys.readouterr().out
    assert "downloaded[2221-o002-NIRCam]" in out
    assert "downloaded[2221-o002-MIRI]" in out
    st = mm.load_state(state_path)
    assert st["triggered"] == {}
    assert st["downloaded"] == {"2221-o001-NIRCam": "keep-me"}


def test_rearm_download_does_not_reach_a_joint_observation(tmp_path, capsys):
    """A downloaded key is instrument-qualified ('...-NIRCam', no '-'), while a
    JOINT obs token carries one, so '2221-o001' must not sweep up the separate
    observation '2221-o001-002'.  Joint obsids are real (sgrb2 5365 MIRI
    '002-998', sickle 3958 MIRI '001-002')."""
    state_path = str(tmp_path / "state.json")
    mm.save_state(state_path, {
        "version": 1, "programs": {},
        "triggered": {"2221-o001": {"when": "2026-08-16 00:00 UTC",
                                    "jobids": []}},
        "downloaded": {"2221-o001-NIRCam": "mine",
                       "2221-o001-002-NIRCam": "another observation"}})
    assert mm.main(["--rearm", "2221-o001", "--rearm-download",
                    "--state", state_path]) == 0
    assert mm.load_state(state_path)["downloaded"] == {
        "2221-o001-002-NIRCam": "another observation"}


def test_rearm_download_reaches_the_joint_observations_own_keys(tmp_path):
    """...and naming the joint token clears exactly its own keys."""
    state_path = str(tmp_path / "state.json")
    mm.save_state(state_path, {
        "version": 1, "programs": {},
        "triggered": {"2221-o001-002": {"when": "2026-08-16 00:00 UTC",
                                        "jobids": []}},
        "downloaded": {"2221-o001-NIRCam": "keep-me",
                       "2221-o001-002-NIRCam": "clear-me"}})
    assert mm.main(["--rearm", "2221-o001-002", "--rearm-download",
                    "--state", state_path]) == 0
    assert mm.load_state(state_path)["downloaded"] == {
        "2221-o001-NIRCam": "keep-me"}


def test_rearm_without_download_flag_keeps_downloads(tmp_path):
    state_path = str(tmp_path / "state.json")
    mm.save_state(state_path, {
        "version": 1, "programs": {},
        "triggered": {"2221-o002": "2026-08-16 00:00 UTC"},
        "downloaded": {"2221-o002-NIRCam": "2026-08-16 00:00 UTC"}})
    rc = mm.main(["--rearm", "2221-o002", "--state", state_path])
    assert rc == 0
    assert "2221-o002-NIRCam" in mm.load_state(state_path)["downloaded"]


# ---------------------------------------------------- dated state backups (issue #68)
def test_backup_keeps_the_first_snapshot_of_the_day(tmp_path):
    """ONE backup per day, holding the state as it stood before the day's first
    write.  _record_state_key calls save_state once per recorded key, so
    refreshing on every write would copy a 1.5 MB file several times a run and
    converge the 'backup' on the current state -- nothing to restore from."""
    path = str(tmp_path / "state.json")
    mm.save_state(path, {"version": 1, "gen": 0})    # nothing to back up yet
    assert not list(tmp_path.glob("*.bak-*"))
    mm.save_state(path, {"version": 1, "gen": 1})
    (bak,) = tmp_path.glob("*.bak-*")
    assert json.loads(bak.read_text())["gen"] == 0   # the pre-write content
    mm.save_state(path, {"version": 1, "gen": 2})
    (bak,) = tmp_path.glob("*.bak-*")                # still ONE per day
    assert json.loads(bak.read_text())["gen"] == 0   # and still the first one


def test_backup_leaves_no_partial_file_at_the_restore_path(tmp_path, monkeypatch,
                                                           capsys):
    """An interrupted copy (ENOSPC, RLIMIT_FSIZE) must not leave a truncated
    file sitting at exactly the path a restore would use, for 14 days: the copy
    goes to a tmp sibling and is renamed in."""
    path = str(tmp_path / "state.json")
    mm.save_state(path, {"version": 1, "gen": 1})
    real_copy = mm.shutil.copy2

    def truncated(src, dst, *a, **kw):
        real_copy(src, dst, *a, **kw)
        raise OSError(27, "File too large")

    monkeypatch.setattr(mm.shutil, "copy2", truncated)
    mm.save_state(path, {"version": 1, "gen": 2})
    assert mm.load_state(path)["gen"] == 2           # the write still happened
    assert "backup FAILED" in capsys.readouterr().err
    assert not list(tmp_path.glob("*.bak-*"))        # no half-written 'backup'


def test_backup_path_occupied_by_a_directory_is_reported(tmp_path, capsys):
    """A directory sitting at the backup path used to swallow the copy (copy2
    writes INSIDE it, empty stderr, nothing at the restore path).  The rename
    refuses it and says so, and the state write still happens."""
    path = str(tmp_path / "state.json")
    mm.save_state(path, {"version": 1, "gen": 1})
    bak = tmp_path / f"state.json.bak-{datetime.date.today():%Y%m%d}"
    bak.mkdir()
    mm.save_state(path, {"version": 1, "gen": 2})
    assert mm.load_state(path)["gen"] == 2
    assert "IsADirectoryError" in capsys.readouterr().err
    assert list(bak.iterdir()) == []


def test_backup_prune_failure_is_not_reported_as_a_backup_failure(tmp_path,
                                                                  monkeypatch,
                                                                  capsys):
    """Pruning is a separate step: a prune that fails must not claim the backup
    failed when the backup succeeded."""
    path = str(tmp_path / "state.json")
    mm.save_state(path, {"version": 1, "gen": 1})

    def denied(*a, **kw):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(mm, "prune_state_backups", denied)
    mm.save_state(path, {"version": 1, "gen": 2})
    err = capsys.readouterr().err
    assert "pruning old state backups FAILED" in err
    assert "state backup FAILED" not in err
    assert mm.load_state(path)["gen"] == 2
    (bak,) = tmp_path.glob("*.bak-*")
    assert json.loads(bak.read_text())["gen"] == 1


def test_backup_failure_still_writes_the_state(tmp_path, monkeypatch, capsys):
    """A backup that cannot be written must NOT abort the write it precedes:
    record_triggered runs right after a successful sbatch, so losing that write
    leaves the jobs queued with the one-shot key unarmed -- the next poll would
    re-submit the same observation."""
    path = str(tmp_path / "state.json")
    mm.save_state(path, {"version": 1, "gen": 1})

    def denied(*a, **kw):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(mm.shutil, "copy2", denied)
    mm.save_state(path, {"version": 1, "gen": 2})
    assert mm.load_state(path)["gen"] == 2
    assert "backup FAILED" in capsys.readouterr().err


def test_rearm_accepts_joint_obs_token(tmp_path):
    """The trigger key of a joint observation ('2221-o001-002') is addressable
    -- the --rearm grammar matches pipeline_trigger's obs-token grammar."""
    state_path = str(tmp_path / "state.json")
    mm.save_state(state_path, {
        "version": 1, "programs": {},
        "triggered": {"2221-o001-002": {"when": "2026-08-16 00:00 UTC",
                                        "jobids": []}}})
    assert mm.main(["--rearm", "2221-o001-002", "--state", state_path]) == 0
    assert mm.load_state(state_path)["triggered"] == {}


def test_rearm_download_without_rearm_is_refused(tmp_path):
    """--rearm-download alone would silently do nothing; argparse errors (rc 2)."""
    with pytest.raises(SystemExit) as ex:
        mm.main(["--rearm-download", "--state", str(tmp_path / "s.json")])
    assert ex.value.code == 2


def test_backup_prunes_beyond_keep_days(tmp_path):
    import datetime
    path = str(tmp_path / "state.json")
    mm.save_state(path, {"version": 1})
    today = datetime.date.today()
    old = tmp_path / ("state.json.bak-"
                      + f"{today - datetime.timedelta(days=30):%Y%m%d}")
    kept = tmp_path / ("state.json.bak-"
                       + f"{today - datetime.timedelta(days=3):%Y%m%d}")
    undated = tmp_path / "state.json.bak-manualcopy"
    for p in (old, kept, undated):
        p.write_text("{}")
    mm.save_state(path, {"version": 1, "n": 2})
    names = {p.name for p in tmp_path.glob("state.json.bak-*")}
    assert old.name not in names                     # pruned (> 14 d)
    assert kept.name in names                        # inside the window
    assert undated.name in names                     # non-dated left alone
    assert f"state.json.bak-{today:%Y%m%d}" in names


def test_backup_prune_boundary_keeps_exactly_keep_days(tmp_path):
    """The retention window is inclusive at its edge: a backup exactly
    BACKUP_KEEP_DAYS old survives and the next day's does not, so an operator
    reading '14 days' gets 14 restorable days rather than 13."""
    import datetime
    path = str(tmp_path / "state.json")
    today = datetime.date.today()

    def dated(days):
        p = tmp_path / ("state.json.bak-"
                        + f"{today - datetime.timedelta(days=days):%Y%m%d}")
        p.write_text("{}")
        return p

    edge = dated(mm.BACKUP_KEEP_DAYS)
    over = dated(mm.BACKUP_KEEP_DAYS + 1)
    mm.prune_state_backups(path, today=today)
    assert edge.exists()
    assert not over.exists()


# ------------------------------------------------------- MAST failure isolation (HIGH-2)
def test_query_failure_skips_program_not_poll(monkeypatch, tmp_path, capsys):
    import requests

    def fake_query(prog):
        if int(prog) == 2221:
            raise requests.exceptions.ConnectionError("MAST down")
        return [_row("jw01182-o004_t001_nircam_clear-f405n")]

    calls = []
    monkeypatch.setattr(mm, "mast_login_if_token", lambda: False)
    monkeypatch.setattr(mm, "query_program", fake_query)
    monkeypatch.setattr(mm, "act_report",
                        lambda evs, **kw: calls.append(("report", evs)))
    state = tmp_path / "state.json"
    rc = mm.main(["--program", "2221", "1182", "--report",
                  "--commit-state", "--state", str(state)])
    assert rc == 0
    err = capsys.readouterr().err
    assert "WARNING" in err and "2221" in err
    (name, evs), = calls
    assert [e["program"] for e in evs] == [1182]      # 1182 still processed
    committed = mm.load_state(str(state))
    assert "1182" in committed["programs"]
    assert "2221" not in committed["programs"]        # failed program untouched


# ------------------------------------------------------------- filter parsing (MED-5)
@pytest.mark.parametrize("raw,expected", [
    ("CLEAR;F212N", ["F212N"]),
    ("F444W;F470N", ["F470N"]),            # F444W is F470N's LW blocking filter -> dropped (#35)
    ("F322W2;F323N", ["F323N"]),           # F322W2 is F323N's LW blocking filter -> dropped (#35)
    ("F210M;F212N", ["F212N"]),            # F210M is F212N's SW blocking filter -> dropped
    ("F212N;F480M", ["F212N", "F480M"]),   # SW narrowband + LW medium: different channels, both kept
    ("F150W2;CLEAR", ["F150W2"]),
    ("F770W", ["F770W"]),                  # MIRI 3-digit
    ("F1000W;F770W", ["F1000W", "F770W"]),  # MIRI 4-digit (no narrowband -> no blocker removal)
    ("GRISMR;F322W2", []),                 # WFSS/dispersed: no imaging filter (issue #35)
    ("GRISMC;F444W", []),                  # ditto, other grism
    ("CLEAR;F322W2", ["F322W2"]),          # direct-imaging config of the same filter is kept
    ("F212N;F212N;F480M", ["F212N", "F480M"]),   # dedupe, stable order
    ("MASKRND;WLP8;junk;", []),
    ("", []),
    (None, []),
])
def test_parse_filters(raw, expected):
    assert mm.parse_filters(raw) == expected


def test_act_trigger_drops_clear_token(monkeypatch, tmp_path):
    submitted = _patch_submit(monkeypatch)
    monkeypatch.setattr(mm, "inflight_job_names", lambda: set())
    mm.act_trigger(_trigger_events("001", filters="CLEAR;F212N"), execute=False)
    assert submitted[0]["filters"] == ["F212N"]


def test_act_trigger_all_junk_filters_skips(monkeypatch, capsys):
    submitted = _patch_submit(monkeypatch)
    mm.act_trigger(_trigger_events("001", filters="CLEAR;GRISMR"), execute=False)
    assert submitted == []
    assert "no filters known" in capsys.readouterr().err


# --------------------------------------------------------------- size precheck (MED-4)
def _patch_download(monkeypatch, size, free_tb=10.0):
    from data_qa import retrieve_data
    fetched = []
    monkeypatch.setattr(retrieve_data, "product_list_size_bytes",
                        lambda *a, **kw: size)
    monkeypatch.setattr(retrieve_data, "retrieve",
                        lambda *a, **kw: fetched.append(kw))
    monkeypatch.setattr(mm, "disk_gate",
                        lambda d, m: (free_tb >= m, free_tb, "gate"))
    return fetched


def test_act_download_skips_oversize_group(monkeypatch, capsys):
    # free 10 TB, floor 5 TB -> 5 TB headroom; 6 TB projected -> skip
    fetched = _patch_download(monkeypatch, size=6e12, free_tb=10.0)
    mm.act_download(_trigger_events("001"), execute=True, min_free_tb=5.0)
    assert fetched == []
    assert "SKIPPED(oversize)" in capsys.readouterr().err


def test_act_download_proceeds_when_size_fits(monkeypatch):
    fetched = _patch_download(monkeypatch, size=1e12, free_tb=10.0)
    mm.act_download(_trigger_events("001"), execute=True, min_free_tb=5.0)
    assert len(fetched) == 1
    assert fetched[0]["dry_run"] is False


def test_act_download_unknown_size_skips_by_default(monkeypatch, capsys):
    fetched = _patch_download(monkeypatch, size=None, free_tb=10.0)
    mm.act_download(_trigger_events("001"), execute=True, min_free_tb=5.0)
    assert fetched == []
    assert "SKIPPED(unknown-size)" in capsys.readouterr().err


def test_act_download_unknown_size_forced(monkeypatch, capsys):
    fetched = _patch_download(monkeypatch, size=None, free_tb=10.0)
    mm.act_download(_trigger_events("001"), execute=True, min_free_tb=5.0,
                    force_unknown_size=True)
    assert len(fetched) == 1
    assert "force-download-unknown-size" in capsys.readouterr().err


def test_act_download_rechecks_disk_gate_per_group(monkeypatch, capsys):
    fetched = _patch_download(monkeypatch, size=1e12, free_tb=2.0)   # below floor
    mm.act_download(_trigger_events("001"), execute=True, min_free_tb=5.0)
    assert fetched == []
    assert "SKIPPED(low-disk)" in capsys.readouterr().err


# ------------------------------------------- mid-run download SKIPs (issue #84)
def test_act_download_reports_the_groups_it_left_owed(monkeypatch):
    """Each mid-run SKIP burns no 'downloaded' key, so the download is still
    owed; act_download names those groups so main() can re-arm them."""
    _patch_download(monkeypatch, size=1e12, free_tb=2.0)          # below floor
    assert mm.act_download(_trigger_events("001"), execute=True,
                           min_free_tb=5.0) == {(2221, "001", "NIRCam"):
                                                "low-disk"}
    _patch_download(monkeypatch, size=None, free_tb=10.0)
    assert mm.act_download(_trigger_events("001"), execute=True,
                           min_free_tb=5.0) == {(2221, "001", "NIRCam"):
                                                "unknown-size"}
    _patch_download(monkeypatch, size=6e12, free_tb=10.0)         # 5 TB headroom
    assert mm.act_download(_trigger_events("001"), execute=True,
                           min_free_tb=5.0) == {(2221, "001", "NIRCam"):
                                                "oversize"}


def test_act_download_owes_nothing_for_standing_skips(monkeypatch, tmp_path):
    """A planned tile, an unmapped program and an already-downloaded group are
    NOT owed: re-arming them would re-fire the same group every poll forever."""
    _patch_download(monkeypatch, size=1e12, free_tb=10.0)
    assert mm.act_download(_planned_events(), execute=True,
                           min_free_tb=5.0) == {}
    unmapped = [dict(_trigger_events("001")[0], field=None)]
    assert mm.act_download(unmapped, execute=True, min_free_tb=5.0) == {}
    state = {"version": 1, "downloaded": {"2221-o001-NIRCam": "2026-08-25"}}
    assert mm.act_download(_trigger_events("001"), execute=True,
                           min_free_tb=5.0, state=state,
                           state_path=str(tmp_path / "s.json")) == {}


def _patch_poll_real_download(monkeypatch, rows, gates):
    """main() with the REAL act_download: canned MAST rows, recorded fetches,
    and a disk gate whose verdicts are read from ``gates`` in call order."""
    from data_qa import retrieve_data
    fetched = []
    monkeypatch.setattr(mm, "mast_login_if_token", lambda: False)
    monkeypatch.setattr(mm, "query_program", lambda prog: list(rows))
    monkeypatch.setattr(retrieve_data, "product_list_size_bytes",
                        lambda *a, **kw: 1e11)
    monkeypatch.setattr(retrieve_data, "retrieve",
                        lambda *a, **kw: fetched.append(kw) or "manifest")
    verdicts = list(gates)

    def gate(download_dir, min_free_tb):
        ok = verdicts.pop(0) if verdicts else True
        return (True, 10.0, "gate") if ok else (False, 1.0, "LOW DISK: gate")

    monkeypatch.setattr(mm, "disk_gate", gate)
    return fetched


def test_main_rearms_a_group_the_disk_gate_skipped_mid_run(monkeypatch, tmp_path):
    """The #84 loss: the between-groups re-check trips after group 1 ate the
    headroom, group 2 is skipped, and the end-of-run commit retires group 2's
    event -- the download owed with nothing recording it.  Now group 2 keeps
    its pre-poll baseline and re-fires."""
    rows = [_row("jw02221-o001_t001_nircam_clear-f405n"),
            _row("jw02221-o002_t001_nircam_clear-f405n")]
    fetched = _patch_poll_real_download(monkeypatch, rows, gates=[True, False])
    state = tmp_path / "state.json"
    _seed_state(state)                           # baseline: not a first run
    args = ["--program", "2221", "--download", "--execute", "--commit-state",
            "--state", str(state), "--download-dir", str(tmp_path)]
    assert mm.main(args) == 0
    assert len(fetched) == 1                     # only the first group ran
    committed = mm.load_state(str(state))["programs"]["2221"]["obs"]
    assert "jw02221-o001_t001_nircam_clear-f405n" in committed   # retired
    assert "jw02221-o002_t001_nircam_clear-f405n" not in committed  # re-armed

    # next poll, disk healthy again: the skipped group is re-offered and the
    # already-downloaded one is not
    fetched2 = _patch_poll_real_download(monkeypatch, rows, gates=[True, True])
    assert mm.main(args) == 0
    assert len(fetched2) == 1
    committed = mm.load_state(str(state))["programs"]["2221"]["obs"]
    assert "jw02221-o002_t001_nircam_clear-f405n" in committed
    assert set(mm.load_state(str(state))["downloaded"]) == {"2221-o001-NIRCam",
                                                            "2221-o002-NIRCam"}


def test_main_notices_the_deferred_download(monkeypatch, tmp_path, capsys):
    """The operator reads the debt on the QA issue, not only in the log: the
    notice rides act_report's comment body and counts as a downgrade class, so
    its first appearance posts a NOTIFYING comment."""
    rows = [_row("jw02221-o001_t001_nircam_clear-f405n")]
    _patch_poll_real_download(monkeypatch, rows, gates=[False])
    reported = {}
    monkeypatch.setattr(mm, "act_report",
                        lambda evs, **kw: reported.update(kw))
    state = tmp_path / "state.json"
    _seed_state(state)
    assert mm.main(["--program", "2221", "--download", "--report", "--execute",
                    "--commit-state", "--state", str(state),
                    "--download-dir", str(tmp_path)]) == 0
    assert "DOWNLOAD DEFERRED" in reported["notice"]
    assert "2221-o001-NIRCam (low-disk)" in reported["notice"]
    assert mm.notice_downgrade_reason(reported["notice"]) == "DOWNLOAD DEFERRED"
    assert "DOWNLOAD DEFERRED" in capsys.readouterr().err


def test_act_download_dry_run_skips_prechecks(monkeypatch):
    from data_qa import retrieve_data
    fetched = []
    monkeypatch.setattr(retrieve_data, "product_list_size_bytes",
                        lambda *a, **kw: pytest.fail("no size query on dry-run"))
    monkeypatch.setattr(retrieve_data, "retrieve",
                        lambda *a, **kw: fetched.append(kw))
    mm.act_download(_trigger_events("001"), execute=False)
    assert len(fetched) == 1
    assert fetched[0]["dry_run"] is True


# ------------------------------------------------------------------------- LOW items
def test_act_download_skips_unmapped_program(monkeypatch, capsys):
    from data_qa import retrieve_data
    monkeypatch.setattr(retrieve_data, "retrieve",
                        lambda *a, **kw: pytest.fail("must not download"))
    events = _trigger_events("001", field="")        # no field mapping
    mm.act_download(events, execute=False)
    assert "no field mapping" in capsys.readouterr().err


def test_save_state_unlinks_orphan_tmp_on_failure(tmp_path):
    path = tmp_path / "state.json"
    with pytest.raises(TypeError):                   # sets aren't JSON-able
        mm.save_state(str(path), {"bad": {1, 2, 3}})
    assert not list(tmp_path.glob("*.tmp.*"))        # no orphan tmp left
    assert not path.exists()


def test_act_report_planned_batch_edits_with_monitor_marker(monkeypatch):
    """Purely-planned batches keep the anti-spam edit-in-place path (#71)."""
    from data_qa import status_report
    posted = []
    monkeypatch.setattr(status_report, "post_status",
                        lambda title, body, **kw: posted.append((title, body, kw))
                        or 0)
    mm.act_report(_planned_events(), execute=True)
    (title, body, kw), = posted
    assert kw["update_last"] is True
    assert kw["marker"] == status_report.MONITOR_MARKER
    assert "PLANNED" in body


# ------------------------------------------------- masked MAST values (BLOCKER 2)
class _FakeTable:
    """Minimal stand-in for the astroquery result table (colnames + row[c])."""
    def __init__(self, rows):
        self.rows = rows
        self.colnames = list(rows[0]) if rows else []

    def __iter__(self):
        return iter(self.rows)


def _patch_fake_mast(monkeypatch, rows):
    import sys
    import types
    fake_mast = types.SimpleNamespace(
        Observations=types.SimpleNamespace(
            query_criteria=lambda **kw: _FakeTable(rows)),
        conf=types.SimpleNamespace(timeout=0, pagesize=0))
    monkeypatch.setitem(sys.modules, "astroquery",
                        types.SimpleNamespace(mast=fake_mast))
    monkeypatch.setitem(sys.modules, "astroquery.mast", fake_mast)


def test_scalar_masked_nan_none():
    import numpy as np
    assert mm._scalar(np.ma.masked, int, default=-1) == -1
    assert mm._scalar(np.ma.masked, float) is None
    assert mm._scalar(np.ma.masked, str) is None
    assert mm._scalar(float("nan"), float) is None
    assert mm._scalar(None, str) is None
    assert mm._scalar("3", int) == 3
    assert mm._scalar(59900.5, float) == 59900.5
    assert mm._scalar("junk", int, default=-1) == -1


def test_mjd_to_iso_unknown_for_none_nan_masked():
    import numpy as np
    assert mm.mjd_to_iso(None) == "unknown"
    assert mm.mjd_to_iso(float("nan")) == "unknown"
    assert mm.mjd_to_iso(np.ma.masked) == "unknown"
    assert "UTC" in mm.mjd_to_iso(59900.0)


def test_query_program_masked_planned_row_end_to_end(monkeypatch):
    """A planned/unreleased row (the 10678 watch target: masked calib_level +
    t_obs_release) must survive query_program -> summarize -> diff_events ->
    format_event without raising (int(masked) raises numpy.ma.MaskError;
    float(masked) -> NaN used to crash mjd_to_iso at report time)."""
    import numpy as np
    _patch_fake_mast(monkeypatch, [{
        "obs_id": "jw10678-o101_t001_nircam_clear-f212n",
        "t_max": np.ma.masked, "t_obs_release": np.ma.masked,
        "calib_level": np.ma.masked, "instrument_name": "NIRCAM/IMAGE",
        "filters": "F212N;F480M", "target_name": "GC_101"}])
    (row,) = mm.query_program(10678)
    assert row["calib_level"] == -1              # masked -> -1 (not a crash)
    assert row["t_obs_release"] is None
    assert row["t_max"] is None
    new = mm.summarize([row], POLL)
    (ev,) = mm.diff_events(10678, {}, new)
    assert ev["released"] is False
    assert ev["calib_level"] == -1
    line = mm.format_event(ev)                   # report-time formatting
    assert "PLANNED" in line
    assert "unknown" in line                     # masked release date


# ---------------------------------------------- released/calib gate (BLOCKER 3)
def _planned_events(obsnum="101", program=10678, field="gc-treasury",
                    tile="GC_101"):
    return [dict(event="NEW_OBSERVATION", program=program, obsnum=obsnum,
                 obs_id=f"jw{program:05d}-o{obsnum}_t001_nircam_clear-f212n",
                 field=field, tile=tile, calib_level=-1, released=False,
                 t_obs_release=None, instrument_name="NIRCAM/IMAGE",
                 filters="F212N;F480M", target_name=tile)]


def test_event_ready_gate():
    (planned,) = _planned_events()
    assert mm.event_ready(planned) is False
    (released,) = _trigger_events("001")         # calib 3, released
    assert mm.event_ready(released) is True
    uncal = dict(released, calib_level=1)        # released but uncal-only
    assert mm.event_ready(uncal) is False
    unreleased = dict(released, released=False)  # calibrated but embargoed
    assert mm.event_ready(unreleased) is False


def test_act_trigger_planned_obs_no_trigger_no_burn(monkeypatch, tmp_path, capsys):
    """A planned obs (calib -1, unreleased) must not submit AND must not burn
    the one-shot 'triggered' key (the key burn was refusing the REAL trigger
    when the data later arrived)."""
    submitted = _patch_submit(monkeypatch)
    monkeypatch.setattr(mm, "inflight_job_names", lambda: set())
    state_path = tmp_path / "state.json"
    state = {"version": 1, "programs": {}}
    mm.act_trigger(_planned_events(), execute=True,
                   state=state, state_path=str(state_path))
    assert submitted == []
    assert "SKIPPED(planned)" in capsys.readouterr().err
    assert "triggered" not in state              # key NOT burned...
    assert not state_path.exists()               # ...in memory or on disk


def test_act_trigger_fires_once_after_release(monkeypatch, tmp_path):
    """Planned -> skipped without burning; released later -> triggers exactly
    once; a re-fire is then refused via the burned key."""
    submitted = _patch_submit(monkeypatch)
    monkeypatch.setattr(mm, "inflight_job_names", lambda: set())
    state_path = str(tmp_path / "state.json")
    state = {"version": 1, "programs": {}}
    mm.act_trigger(_planned_events(), execute=True,
                   state=state, state_path=state_path)
    assert submitted == []
    released = _planned_events()
    released[0].update(calib_level=3, released=True, t_obs_release=POLL - 1)
    mm.act_trigger(released, execute=True, state=state, state_path=state_path)
    assert len(submitted) == 1
    assert "10678-o101" in state["triggered"]    # burned on the REAL submit
    mm.act_trigger(released, execute=True, state=state, state_path=state_path)
    assert len(submitted) == 1                   # one-shot holds


def test_act_download_planned_obs_skips(monkeypatch, capsys):
    fetched = _patch_download(monkeypatch, size=1e12, free_tb=10.0)
    mm.act_download(_planned_events(), execute=True, min_free_tb=5.0)
    assert fetched == []
    assert "SKIPPED(planned)" in capsys.readouterr().err


# ------------------------------------------- instrument-aware keying (MED-b)
def test_instrument_class():
    assert mm.instrument_class("NIRCAM/IMAGE") == "NIRCam"
    assert mm.instrument_class("MIRI/IMAGE") == "MIRI"
    assert mm.instrument_class("NIRCAM") == "NIRCam"
    assert mm.instrument_class(None) == ""
    assert mm.instrument_class("") == ""


def _dual_instrument_events():
    """The real jw02221-o002 shape: NIRCam and MIRI deliveries of one obs."""
    base = dict(event="NEW_OBSERVATION", program=2221, obsnum="002",
                field="cloudc", tile=None, calib_level=3, released=True,
                t_obs_release=59900.0, target_name="CLOUDC")
    return [dict(base, obs_id="jw02221-o002_t001_nircam_clear-f405n",
                 instrument_name="NIRCAM/IMAGE", filters="F405N;F212N"),
            dict(base, obs_id="jw02221-o002_t001_miri_f770w",
                 instrument_name="MIRI/IMAGE", filters="F770W")]


def test_group_by_obs_splits_instrument_classes():
    grouped = mm._group_by_obs(_dual_instrument_events())
    assert set(grouped) == {(2221, "002", "NIRCam"), (2221, "002", "MIRI")}


def test_act_trigger_skips_miri_group_triggers_nircam(monkeypatch, tmp_path,
                                                      capsys):
    submitted = _patch_submit(monkeypatch)
    monkeypatch.setattr(mm, "inflight_job_names", lambda: set())
    mm.act_trigger(_dual_instrument_events(), execute=True, state={},
                   state_path=str(tmp_path / "state.json"))
    assert len(submitted) == 1                   # the NIRCam group only
    assert submitted[0]["filters"] == ["F405N", "F212N"]
    err = capsys.readouterr().err
    assert "SKIPPED(not-automated)" in err and "NIRCam-only" in err


def test_act_report_titles_by_instrument(monkeypatch):
    """Comments land on the instrument-matched issue: '(NIRCam)' vs '(MIRI)'."""
    from data_qa import status_report
    posted = []
    monkeypatch.setattr(status_report, "post_status",
                        lambda title, body, **kw: posted.append(title) or 0)
    mm.act_report(_dual_instrument_events(), execute=False)
    assert sorted(posted) == ["Cloud C — jw02221-o002 (MIRI)",
                              "Cloud C — jw02221-o002 (NIRCam)"]


def test_act_download_dual_instrument_separate_keys(monkeypatch, tmp_path):
    """NIRCam and MIRI downloads of one obs burn SEPARATE 'downloaded' keys."""
    from data_qa import retrieve_data
    fetched = []
    monkeypatch.setattr(retrieve_data, "product_list_size_bytes",
                        lambda *a, **kw: 1e9)
    monkeypatch.setattr(retrieve_data, "retrieve",
                        lambda *a, **kw: fetched.append(kw) or "manifest")
    monkeypatch.setattr(mm, "disk_gate", lambda d, m: (True, 10.0, "gate"))
    state_path = str(tmp_path / "state.json")
    state = {"version": 1, "programs": {}}
    mm.act_download(_dual_instrument_events(), execute=True, min_free_tb=5.0,
                    state=state, state_path=state_path)
    assert len(fetched) == 2
    assert {kw["instrument"] for kw in fetched} == {"NIRCam", "MIRI"}
    on_disk = mm.load_state(state_path)
    assert set(on_disk["downloaded"]) == {"2221-o002-NIRCam", "2221-o002-MIRI"}


# ------------------------------------------------ PROGRAMS completeness (MED-c)
# The pipeline's single field registry (its commit ee33bec, 2026-07-31, replaced
# the per-driver literal dicts).  Every proposal/observation the reduction maps
# is declared there, and it is read here with the pipeline's OWN loader
# (jwst_gc_pipeline/fields.py), so data-qa cannot hold a private opinion about
# the registry schema: a spelling the reduction refuses raises here too, and a
# spelling it accepts maps here the same way.
#
# PIPE_ROOT names the checkout; the tests workflow points it at the
# keflavich/jwst-gc-pipeline checkout it makes, so the guard runs in CI instead
# of skipping (the green-by-skip half of #74).  data_qa.paths reads that
# variable for the whole package now (issue #85), so this guard and the code
# under test resolve the same checkout.
_PIPE_ROOT = paths.pipe_root()
_PIPE_FIELDS_PY = os.path.join(_PIPE_ROOT, "jwst_gc_pipeline", "fields.py")
# CI sets this after checking the pipeline out: a checkout step that silently
# produced nothing then fails the suite.  A skip would hide it.
_REQUIRE_PIPE_REGISTRY = os.environ.get("REQUIRE_PIPE_REGISTRY") == "1"
_needs_pipeline = pytest.mark.skipif(
    not os.path.exists(_PIPE_FIELDS_PY) and not _REQUIRE_PIPE_REGISTRY,
    reason=f"jwst-gc-pipeline checkout not available at {_PIPE_ROOT}")
# Globular-cluster programs ride the pipeline for testing only; they are not
# GC-monitor targets.  Arches/Quintuplet (2045) and Sgr A* (1939) ARE GC fields.
_GLOBULAR_PROGRAMS = {1334, 1979, 8322, 12587}
# An obsids entry of '*' declares "every observation of this proposal" -- the
# registry shape for programs whose observation numbers land only as the visits
# execute (the 10678 treasury: 139 visits over ~1668 planned observations).
_WILDCARD_OBS = "*"
# Concrete observation numbers to resolve a wildcard entry with.  Any obsid
# would do -- the treasury answers program-wide through field_for() -- so these
# just span the range real GC_<n> tiles will number into.
_WILDCARD_PROBES = ("001", "042", "139", "742")
# Proposals whose NIRCam observations the registry maps today, globular-cluster
# test programs aside.  The floor exists because an empty answer for a proposal
# is indistinguishable from "not observed with NIRCam" (2526 really is
# MIRI-only): without it, an `obsids:` block that loses its key drops the
# proposal out of the guard and the suite stays green.  Add a proposal here
# when a field is registered; a removal wants a reason in the commit message.
_NIRCAM_PROPOSALS = {1182, 1905, 1939, 2045, 2092, 2211, 2221, 3523, 3958,
                     4147, 5365, 6151, 6778, 7213}
_PIPE_FIELDS_MODULE = []


def _pipeline_fields_module():
    """The pipeline's registry loader, executed straight from the checkout.

    Executed as a standalone module: importing it as
    ``jwst_gc_pipeline.fields`` runs the package ``__init__``, which installs a
    process-wide ``astropy.io.fits.HDUList.writeto`` provenance hook that every
    other test in this session would inherit.  ``fields.py`` has no
    package-level imports of its own, so executing the file alone gives the
    same loader the reduction drivers use.
    """
    if not _PIPE_FIELDS_MODULE:
        import importlib.util
        spec = importlib.util.spec_from_file_location("_pipe_fields",
                                                      _PIPE_FIELDS_PY)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _PIPE_FIELDS_MODULE.append(module)
    return _PIPE_FIELDS_MODULE[0]


def _pipeline_field_map(registry_path=None, instrument="nircam"):
    """``{proposal: {obsid: field}}`` for one instrument, as
    ``jwst_gc_pipeline.fields.field_to_reg_mapping`` answers it.

    ``registry_path`` names an alternative ``fields.yaml`` (the wildcard and
    schema tests write one); the default is the checkout's own registry.
    ``joint_obsids`` tokens ('002-998') arrive as the loader returns them and
    are split by ``_assert_monitor_covers``.

    A proposal the registry declares with no observation for this instrument
    (2526 is MIRI-only) is left out: it maps nothing, so there is nothing for
    the completeness check to compare.
    """
    fields = _pipeline_fields_module()
    loaded = fields.FIELDS
    if registry_path is not None:
        loaded = fields._load(registry_path)[1]
    proposals = sorted({o.proposal for f in loaded for o in f.observations},
                       key=int)
    original = fields.FIELDS
    # The public view reads the module-level FIELDS; swapping it is how an
    # alternative registry gets the same answer the reduction would give.
    fields.FIELDS = loaded
    try:
        out = {p: dict(fields.field_to_reg_mapping(p, instrument))
               for p in proposals}
    finally:
        fields.FIELDS = original
    return {p: obsmap for p, obsmap in out.items() if obsmap}


def _assert_monitor_covers(mapping):
    """Fail unless every observation the pipeline registry maps appears in
    ``mast_monitor.PROGRAMS`` under the same field name (globular-cluster test
    programs excluded).  PROGRAMS may carry extras; this direction catches a
    newly registered field nobody monitors.

    A joint token ('002-998') names several observations cataloged in one run,
    so every part is required, the way ``fields.target_for_obsid`` reads one.
    A wildcard entry stands for every concrete observation of its proposal;
    PROGRAMS cannot enumerate obs numbers that do not exist yet, so that check
    goes through ``field_for()`` with concrete probes."""
    for prog_str, obsmap in mapping.items():
        prog = int(prog_str)
        if prog in _GLOBULAR_PROGRAMS:
            continue
        assert prog in mm.PROGRAMS, \
            f"pipeline maps program {prog} but mast_monitor.PROGRAMS lacks it"
        for obsid, field in obsmap.items():
            if obsid == _WILDCARD_OBS:
                for probe in _WILDCARD_PROBES:
                    assert mm.field_for(prog, probe) == field, \
                        (prog, probe, field)
                continue
            for obsnum in str(obsid).split("-"):
                assert obsnum in mm.PROGRAMS[prog], (prog, obsid, obsnum)
                qa_field = mm.PROGRAMS[prog][obsnum]
                # EXACT match: PROGRAMS must name the same field the pipeline registers.  Where the
                # pipeline split a region into per-obs reduction fields (gc2211 -> gc2211_o023 ...,
                # pipeline #469), PROGRAMS now carries those per-obs keys too (issue #119), so no
                # base-field relaxation is needed -- and re-tightening restores the drift guard the
                # relaxation had been masking.
                assert field == qa_field, (prog, obsnum, field)


def test_pipeline_checkout_present_when_required():
    """CI checks the pipeline out and sets REQUIRE_PIPE_REGISTRY=1, so a
    checkout step that produced nothing fails here.  Without it a broken
    checkout puts the completeness guard back to green-by-skip."""
    if not _REQUIRE_PIPE_REGISTRY:
        pytest.skip("REQUIRE_PIPE_REGISTRY is not set")
    assert os.path.exists(_PIPE_FIELDS_PY), \
        f"REQUIRE_PIPE_REGISTRY=1 but {_PIPE_FIELDS_PY} does not exist"


@_needs_pipeline
def test_programs_complete_vs_pipeline_field_mapping():
    """Every GC program/obs the reduction pipeline maps must be monitored
    (globular-cluster test programs excluded).

    The comparison is the NIRCam map on purpose: PROGRAMS carries the
    NIRCam-primary orientation, and 2221 numbers its NIRCam and MIRI
    observations of the same two fields in opposite order (brick is NIRCam 001
    and MIRI 002; cloudc the reverse), so the MIRI orientation cannot agree
    with it."""
    mapping = _pipeline_field_map()
    assert mapping, "no observations parsed from the pipeline field registry"
    _assert_monitor_covers(mapping)


@_needs_pipeline
def test_pipeline_field_map_covers_every_known_nircam_proposal():
    """Every proposal known to have NIRCam observations still maps some.

    A proposal quietly leaving the derived map is the failure this guard is
    least able to see on its own: the completeness check then has nothing to
    compare for it and passes."""
    mapping = _pipeline_field_map()
    missing = _NIRCAM_PROPOSALS - {int(p) for p in mapping}
    assert not missing, (
        f"proposals {sorted(missing)} map no NIRCam observation; the registry "
        f"lost them or _NIRCAM_PROPOSALS needs updating")


@_needs_pipeline
def test_programs_completeness_detects_missing_and_misnamed(monkeypatch):
    """The guard still FAILS when a pipeline-mapped program is dropped from
    PROGRAMS, or one of its fields renamed -- the two regressions it exists
    to catch."""
    mapping = _pipeline_field_map()
    prog_str, obsmap = next(
        (p, m) for p, m in sorted(mapping.items(), key=lambda kv: int(kv[0]))
        if int(p) not in _GLOBULAR_PROGRAMS
        and any(o != _WILDCARD_OBS for o in m))
    prog = int(prog_str)
    obsnum = next(o for o in obsmap if o != _WILDCARD_OBS)
    with monkeypatch.context() as m:
        m.setattr(mm, "PROGRAMS",
                  {k: v for k, v in mm.PROGRAMS.items() if k != prog})
        with pytest.raises(AssertionError):
            _assert_monitor_covers(mapping)
    with monkeypatch.context() as m:
        renamed = {k: dict(v) for k, v in mm.PROGRAMS.items()}
        renamed[prog][obsnum] = "misnamed-field"
        m.setattr(mm, "PROGRAMS", renamed)
        with pytest.raises(AssertionError):
            _assert_monitor_covers(mapping)


def _write_registry(tmp_path, body):
    """A one-field fields.yaml the pipeline loader can read."""
    path = tmp_path / "fields.yaml"
    path.write_text("roots:\n  blue: /blue\nfields:\n" + body)
    return str(path)


@_needs_pipeline
def test_pipeline_field_map_wildcard_obsids(tmp_path, monkeypatch):
    """A wildcard obsids entry -- the registry shape the 10678 treasury lands
    with -- reads as {'*': field} in both spellings the loader accepts,
    satisfies the completeness check for any concrete observation, and still
    fails it when the program is unmonitored or the field misnamed."""
    for spelling in ("'*'", "['*']"):
        path = _write_registry(tmp_path,
                               "  gc-treasury:\n"
                               "    root: blue\n"
                               "    observations:\n"
                               "      '10678':\n"
                               "        obsids:\n"
                               f"          nircam: {spelling}\n")
        assert _pipeline_field_map(path) == {"10678": {"*": "gc-treasury"}}
    _assert_monitor_covers({"10678": {"*": "gc-treasury"}})
    with pytest.raises(AssertionError):
        _assert_monitor_covers({"10678": {"*": "wrong-field"}})
    with monkeypatch.context() as m:
        m.setattr(mm, "PROGRAMS", {k: v for k, v in mm.PROGRAMS.items()
                                   if k != mm.TREASURY_PROGRAM})
        with pytest.raises(AssertionError):
            _assert_monitor_covers({"10678": {"*": "gc-treasury"}})


@_needs_pipeline
def test_pipeline_field_map_refuses_unknown_instrument_key(tmp_path):
    """An instrument key the reduction's loader refuses raises here too.

    The hand-rolled YAML reader this helper replaced answered ``{}`` for
    ``obsids: {nircam_obs: [...]}``: the proposal left the guard and the suite
    stayed green, which is the #74 failure mode one level finer.  Reading
    through the loader makes data-qa refuse what the reduction refuses."""
    fields = _pipeline_fields_module()
    path = _write_registry(tmp_path,
                           "  gc2211:\n"
                           "    root: blue\n"
                           "    observations:\n"
                           "      '2211':\n"
                           "        obsids:\n"
                           "          nircam_obs: ['023']\n")
    with pytest.raises(fields.FieldRegistryError):
        _pipeline_field_map(path)


def test_assert_monitor_covers_splits_joint_obsids():
    """Every part of a joint token must be monitored.

    ``field_to_reg_mapping`` returns joint tokens alongside concrete obsids
    ('001-002' -> sickle), so the check reads one the way
    ``fields.target_for_obsid`` does: split on '-', require each part."""
    _assert_monitor_covers({"3958": {"001-002": "sickle"}})
    with pytest.raises(AssertionError):
        _assert_monitor_covers({"3958": {"001-777": "sickle"}})


# --------------------------------------------- treasury rolling issue (MED-d)
def test_act_report_treasury_single_rolling_issue(monkeypatch):
    """All treasury events pool into ONE rolling-issue post (not ~1668 per-obs
    rc=3 failures), created with QA + program labels, sharing one issue cache."""
    from data_qa import status_report
    posted = []
    monkeypatch.setattr(status_report, "post_status",
                        lambda title, body, **kw: posted.append((title, body, kw))
                        or 0)
    evs = []
    for n in (101, 102, 103):
        evs += _planned_events(obsnum=str(n), tile=f"GC_{n}")
    evs += _trigger_events("001")                # one regular brick event
    mm.act_report(evs, execute=False)
    treasury = [(t, b, kw) for t, b, kw in posted
                if t == mm.TREASURY_ISSUE_TITLE]
    assert len(treasury) == 1                    # ONE post for all 3 tiles
    title, body, kw = treasury[0]
    assert body.count("NEW_OBSERVATION") == 3
    assert kw["create_labels"] == ["QA", "program:10678"]
    caches = {id(kw["issue_cache"]) for _, _, kw in posted}
    assert len(caches) == 1                      # one shared cache per run


# ---------------------------------------------------- per-program seed (MED-e)
def test_per_program_seed_after_failed_seed_query(monkeypatch, tmp_path, capsys):
    """A program whose query FAILED during the seed run is seeded (baseline
    committed, actions suppressed) on its first successful poll later."""
    import requests
    state = tmp_path / "state.json"
    calls = []
    monkeypatch.setattr(mm, "mast_login_if_token", lambda: False)
    monkeypatch.setattr(mm, "act_download",
                        lambda evs, **kw: calls.append(("download", list(evs))))
    monkeypatch.setattr(mm, "act_trigger",
                        lambda evs, **kw: calls.append(("trigger", list(evs))))
    monkeypatch.setattr(mm, "act_report",
                        lambda evs, **kw: calls.append(("report", list(evs))))
    monkeypatch.setattr(mm.shutil, "disk_usage", lambda p: _Usage(50e12))

    def q_seed(prog):
        if int(prog) == 1182:
            raise requests.exceptions.ConnectionError("MAST down")
        return [_row("jw02221-o001_t001_nircam_clear-f405n")]

    monkeypatch.setattr(mm, "query_program", q_seed)
    rc = mm.main(["--program", "2221", "1182", "--auto", "--state", str(state),
                  "--download-dir", str(tmp_path)])
    assert rc == 0
    assert mm.load_state(str(state))["seeded_programs"] == ["2221"]

    calls.clear()

    def q_later(prog):                           # 1182 back, with a backlog
        if int(prog) == 1182:
            return [_row("jw01182-o004_t001_nircam_clear-f405n")]
        return [_row("jw02221-o001_t001_nircam_clear-f405n")]

    monkeypatch.setattr(mm, "query_program", q_later)
    rc = mm.main(["--program", "2221", "1182", "--auto", "--state", str(state),
                  "--download-dir", str(tmp_path)])
    assert rc == 0
    acted = dict(calls)
    assert acted["trigger"] == []                # 1182 backlog NOT acted on
    assert acted["download"] == []
    assert [e["program"] for e in acted["report"]] == [1182]   # but reported
    assert "PER-PROGRAM SEED" in capsys.readouterr().err
    st = mm.load_state(str(state))
    assert st["seeded_programs"] == ["1182", "2221"]           # now seeded
    assert "jw01182-o004_t001_nircam_clear-f405n" in \
        st["programs"]["1182"]["obs"]


# --------------------------------------- download dedup + missing obs (MED-f)
def test_act_download_records_and_dedups(monkeypatch, tmp_path, capsys):
    """A successful release-gated download burns the 'downloaded' key
    (mirroring 'triggered'); the next run skips it."""
    from data_qa import retrieve_data
    fetched = []
    monkeypatch.setattr(retrieve_data, "product_list_size_bytes",
                        lambda *a, **kw: 1e9)
    monkeypatch.setattr(retrieve_data, "retrieve",
                        lambda *a, **kw: fetched.append(kw) or "manifest")
    monkeypatch.setattr(mm, "disk_gate", lambda d, m: (True, 10.0, "gate"))
    state_path = str(tmp_path / "state.json")
    state = {"version": 1, "programs": {}}
    mm.act_download(_trigger_events("001"), execute=True, min_free_tb=5.0,
                    state=state, state_path=state_path)
    assert len(fetched) == 1
    assert "2221-o001-NIRCam" in mm.load_state(state_path)["downloaded"]
    mm.act_download(_trigger_events("001"), execute=True, min_free_tb=5.0,
                    state=state, state_path=state_path)
    assert len(fetched) == 1                     # deduplicated
    assert "SKIPPED(already-downloaded)" in capsys.readouterr().err


def test_act_download_failed_download_does_not_burn(monkeypatch, tmp_path):
    """retrieve() returning None (no products / failure) must NOT burn the
    'downloaded' key."""
    from data_qa import retrieve_data
    monkeypatch.setattr(retrieve_data, "product_list_size_bytes",
                        lambda *a, **kw: 1e9)
    monkeypatch.setattr(retrieve_data, "retrieve", lambda *a, **kw: None)
    monkeypatch.setattr(mm, "disk_gate", lambda d, m: (True, 10.0, "gate"))
    state_path = str(tmp_path / "state.json")
    state = {"version": 1, "programs": {}}
    mm.act_download(_trigger_events("001"), execute=True, min_free_tb=5.0,
                    state=state, state_path=state_path)
    assert "downloaded" not in state
    assert not (tmp_path / "state.json").exists()


def test_disappeared_obs_kept_with_missing_since(monkeypatch, tmp_path, capsys):
    """An obs that vanishes from MAST is kept in state under 'missing_since'
    (report-only note; no silent drop, no event storm on reappearance)."""
    state = tmp_path / "state.json"
    _seed_state(state, obs_id="jw02221-o009_x")
    monkeypatch.setattr(mm, "mast_login_if_token", lambda: False)
    monkeypatch.setattr(mm, "query_program", lambda prog: [])   # o009 vanished
    rc = mm.main(["--program", "2221", "--commit-state", "--state", str(state)])
    assert rc == 0
    assert "disappeared" in capsys.readouterr().err
    rec = mm.load_state(str(state))["programs"]["2221"]["obs"]["jw02221-o009_x"]
    assert "missing_since" in rec
    assert rec["calib_level"] == 3               # original record preserved
    # reappearance: still in the baseline, so NOT a NEW_OBSERVATION storm
    monkeypatch.setattr(mm, "query_program",
                        lambda prog: [_row("jw02221-o009_x")])
    rc = mm.main(["--program", "2221", "--commit-state", "--state", str(state)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "NEW_OBSERVATION" not in out
    rec = mm.load_state(str(state))["programs"]["2221"]["obs"]["jw02221-o009_x"]
    assert "missing_since" not in rec            # cleared on reappearance


# --------------------------------------- arrival/downgrade notification (#71)
LOW_DISK_NOTICE = ("LOW DISK: only 1.0 TB free on the filesystem of /x "
                   "(< 5.0 TB threshold) -- auto mode downgraded to report-only")
CAPPED_NOTICE = ("CAPPED — actions suppressed: 9 (program,obs) groups would "
                 "act, exceeding --max-submit 4.")


def test_event_is_arrival_truth_table():
    (ready,) = _trigger_events("001")            # NEW_OBSERVATION, calib 3
    assert mm.event_is_arrival(ready) is True    # first seen already released
    assert mm.event_is_arrival(dict(ready, event="NEWLY_RELEASED")) is True
    assert mm.event_is_arrival(dict(ready, event="CALIB_LEVEL_UP",
                                    previous_calib_level=-1)) is True
    assert mm.event_is_arrival(dict(ready, event="CALIB_LEVEL_UP",
                                    previous_calib_level=2)) is False  # 2->3
    (planned,) = _planned_events()
    assert mm.event_is_arrival(planned) is False
    assert mm.event_is_arrival(dict(ready, event="NEWLY_RELEASED",
                                    released=False)) is False   # embargoed
    assert mm.event_is_arrival(dict(ready, event="NEWLY_RELEASED",
                                    calib_level=1)) is False    # uncal-only


def test_notice_downgrade_reason(monkeypatch, tmp_path):
    monkeypatch.setattr(mm.shutil, "disk_usage", lambda p: _Usage(1e12))
    _, _, msg = mm.disk_gate(str(tmp_path), min_free_tb=5.0)   # the REAL text
    assert mm.notice_downgrade_reason(msg) == "LOW DISK"
    assert mm.notice_downgrade_reason(CAPPED_NOTICE) == "CAPPED"
    assert mm.notice_downgrade_reason(None) is None
    assert mm.notice_downgrade_reason("SEED RUN — actions suppressed") is None
    assert mm.notice_downgrade_reason("PER-PROGRAM SEED — program(s) 1182") \
        is None


def _patch_post_status(monkeypatch, rc=0):
    from data_qa import status_report
    posted = []
    monkeypatch.setattr(status_report, "post_status",
                        lambda title, body, **kw: posted.append((title, body, kw))
                        or rc)
    return posted


def test_act_report_arrival_posts_new_comment(monkeypatch):
    """An arrival -> new comment on THAT issue (GitHub notifies for new
    comments only; an edit reaches nobody), while a planned-only issue in the
    same batch stays a quiet edit."""
    posted = _patch_post_status(monkeypatch)
    evs = _planned_events() + _trigger_events("001")
    mm.act_report(evs, execute=True)
    assert len(posted) == 2                      # brick issue + treasury issue
    assert [(title, kw["update_last"]) for title, _, kw in posted] == [
        ("Brick — jw02221-o001 (NIRCam)", False),      # arrival: notify
        (mm.TREASURY_ISSUE_TITLE, True)]               # planned only: edit


def test_act_report_arrival_classified_per_issue(monkeypatch):
    """Classification is per ISSUE, not batch-global: a treasury tile that
    landed notifies on the treasury issue while a planned-only brick batch
    keeps editing in place."""
    posted = _patch_post_status(monkeypatch)
    landed = dict(_planned_events()[0], event="NEWLY_RELEASED",
                  released=True, calib_level=3, t_obs_release=59900.0)
    planned_brick = dict(_trigger_events("001")[0], released=False,
                         calib_level=-1, t_obs_release=None)
    mm.act_report([planned_brick, landed], execute=True)
    assert [(title, kw["update_last"]) for title, _, kw in posted] == [
        ("Brick — jw02221-o001 (NIRCam)", True),       # planned only: edit
        (mm.TREASURY_ISSUE_TITLE, False)]              # arrival: notify


def test_act_report_fresh_downgrade_notifies_every_issue(monkeypatch, tmp_path):
    """The downgrade notice rides on EVERY comment body, so a fresh one
    notifies on every issue even where the events are purely planned."""
    posted = _patch_post_status(monkeypatch)
    evs = _planned_events() + _trigger_events("001")
    mm.act_report(evs, execute=True, notice=LOW_DISK_NOTICE,
                  state={"version": 1, "programs": {}},
                  state_path=str(tmp_path / "state.json"))
    assert [kw["update_last"] for _, _, kw in posted] == [False, False]


def test_act_report_downgrade_first_time_posts_new_and_memos(monkeypatch,
                                                             tmp_path):
    posted = _patch_post_status(monkeypatch)
    state_path = str(tmp_path / "state.json")
    state = {"version": 1, "programs": {}}
    mm.act_report(_planned_events(), execute=True, notice=LOW_DISK_NOTICE,
                  state=state, state_path=state_path)
    (_, body, kw), = posted
    assert kw["update_last"] is False            # NEW comment: notifies
    assert "LOW DISK" in body
    assert mm.load_state(state_path)[mm.NOTIFIED_DOWNGRADE_KEY] == "LOW DISK"
    assert state[mm.NOTIFIED_DOWNGRADE_KEY] == "LOW DISK"


def test_act_report_repeated_downgrade_edits_in_place(monkeypatch, tmp_path):
    posted = _patch_post_status(monkeypatch)
    state = {"version": 1, "programs": {},
             mm.NOTIFIED_DOWNGRADE_KEY: "LOW DISK"}
    mm.act_report(_planned_events(), execute=True, notice=LOW_DISK_NOTICE,
                  state=state, state_path=str(tmp_path / "state.json"))
    (_, _, kw), = posted
    assert kw["update_last"] is True             # repeat: quiet edit
    assert not (tmp_path / "state.json").exists()    # memo untouched


def test_act_report_downgrade_reason_change_posts_new(monkeypatch, tmp_path):
    posted = _patch_post_status(monkeypatch)
    state_path = str(tmp_path / "state.json")
    mm.save_state(state_path, {"version": 1, "programs": {},
                               mm.NOTIFIED_DOWNGRADE_KEY: "LOW DISK"})
    state = mm.load_state(state_path)
    mm.act_report(_planned_events(), execute=True, notice=CAPPED_NOTICE,
                  state=state, state_path=state_path)
    (_, _, kw), = posted
    assert kw["update_last"] is False            # new reason: notify again
    assert mm.load_state(state_path)[mm.NOTIFIED_DOWNGRADE_KEY] == "CAPPED"


def test_act_report_healthy_batch_clears_memo(monkeypatch, tmp_path):
    posted = _patch_post_status(monkeypatch)
    state_path = str(tmp_path / "state.json")
    mm.save_state(state_path, {"version": 1, "programs": {},
                               mm.NOTIFIED_DOWNGRADE_KEY: "LOW DISK"})
    state = mm.load_state(state_path)
    mm.act_report(_planned_events(), execute=True,
                  state=state, state_path=state_path)
    (_, _, kw), = posted
    assert kw["update_last"] is True
    assert mm.NOTIFIED_DOWNGRADE_KEY not in mm.load_state(state_path)
    assert mm.NOTIFIED_DOWNGRADE_KEY not in state


@pytest.mark.parametrize("state", [None, {}])
def test_act_report_reads_memo_from_disk_without_state(monkeypatch, tmp_path,
                                                       state):
    """No usable in-memory state (None, or an EMPTY dict) falls back to
    reading the memo off the state file; an empty dict must not read as
    'nothing notified yet' and re-notify."""
    posted = _patch_post_status(monkeypatch)
    state_path = str(tmp_path / "state.json")
    mm.save_state(state_path, {"version": 1, "programs": {},
                               mm.NOTIFIED_DOWNGRADE_KEY: "LOW DISK"})
    mm.act_report(_planned_events(), execute=True, notice=LOW_DISK_NOTICE,
                  state=state, state_path=state_path)
    (_, _, kw), = posted
    assert kw["update_last"] is True


def test_act_report_dry_run_classifies_but_does_not_memo(monkeypatch,
                                                         tmp_path):
    posted = _patch_post_status(monkeypatch)
    state_path = str(tmp_path / "state.json")
    mm.act_report(_planned_events(), execute=False, notice=LOW_DISK_NOTICE,
                  state={"version": 1, "programs": {}}, state_path=state_path)
    (_, _, kw), = posted
    assert kw["update_last"] is False            # classification still runs
    assert not (tmp_path / "state.json").exists()    # nothing posted: no memo


def test_act_report_failed_post_does_not_memo(monkeypatch, tmp_path):
    """A notifying downgrade whose posts ALL fail leaves the memo unset, so
    the next run notifies again instead of silently editing."""
    posted = _patch_post_status(monkeypatch, rc=4)
    state_path = str(tmp_path / "state.json")
    state = {"version": 1, "programs": {}}
    mm.act_report(_planned_events(), execute=True, notice=LOW_DISK_NOTICE,
                  state=state, state_path=state_path)
    assert len(posted) == 1
    assert mm.NOTIFIED_DOWNGRADE_KEY not in state
    assert not (tmp_path / "state.json").exists()


def test_act_report_arrival_during_repeated_downgrade_still_notifies(
        monkeypatch, tmp_path):
    """Released data waiting while the automation is wedged: keep nagging."""
    posted = _patch_post_status(monkeypatch)
    state = {"version": 1, "programs": {},
             mm.NOTIFIED_DOWNGRADE_KEY: "LOW DISK"}
    mm.act_report(_trigger_events("001"), execute=True, notice=LOW_DISK_NOTICE,
                  state=state, state_path=str(tmp_path / "state.json"))
    (_, _, kw), = posted
    assert kw["update_last"] is False


def test_act_report_explicit_update_last_overrides(monkeypatch):
    posted = _patch_post_status(monkeypatch)
    mm.act_report(_trigger_events("001"), execute=True, update_last=True)
    (_, _, kw), = posted
    assert kw["update_last"] is True             # caller override wins


def _planned_row(obs_id, target="GC_1"):
    """A planned/unexecuted MAST row, as query_program normalizes it
    (masked release date + calib level -> None / -1)."""
    return {"obs_id": obs_id, "t_max": None, "t_obs_release": None,
            "calib_level": -1, "instrument_name": "NIRCAM/IMAGE",
            "filters": "F212N;F480M", "target_name": target}


def test_auto_downgrade_episode_end_to_end(monkeypatch, tmp_path):
    """Through main(): LOW DISK notifies once, repeats edit quietly, a healthy
    committed run ends the episode, and the next downgrade notifies afresh."""
    posted = _patch_post_status(monkeypatch)
    monkeypatch.setattr(mm, "mast_login_if_token", lambda: False)
    monkeypatch.setattr(mm, "act_download", lambda *a, **kw: None)
    monkeypatch.setattr(mm, "act_trigger", lambda *a, **kw: None)
    monkeypatch.setattr(mm, "query_program",
                        lambda prog: [_planned_row("jw10678-o101_t101_nircam")])
    state = tmp_path / "state.json"
    _seed_state(state, program=10678, obs_id="jw10678-o001_x")
    argv = ["--program", "10678", "--auto", "--state", str(state),
            "--download-dir", str(tmp_path)]

    monkeypatch.setattr(mm.shutil, "disk_usage", lambda p: _Usage(1e12))
    assert mm.main(argv) == 0
    assert posted[-1][2]["update_last"] is False     # episode start: notify
    assert mm.main(argv) == 0
    assert posted[-1][2]["update_last"] is True      # repeat: quiet edit

    monkeypatch.setattr(mm.shutil, "disk_usage", lambda p: _Usage(50e12))
    assert mm.main(argv) == 0                        # healthy: commit + clear
    assert posted[-1][2]["update_last"] is True      # planned-only: edit
    assert mm.NOTIFIED_DOWNGRADE_KEY not in mm.load_state(str(state))

    monkeypatch.setattr(mm.shutil, "disk_usage", lambda p: _Usage(1e12))
    monkeypatch.setattr(mm, "query_program",
                        lambda prog: [_planned_row("jw10678-o101_t101_nircam"),
                                      _planned_row("jw10678-o102_t102_nircam",
                                                   target="GC_2")])
    assert mm.main(argv) == 0
    assert posted[-1][2]["update_last"] is False     # new episode: notify


def test_retire_downgrade_memo_keeps_memo_on_downgraded_run():
    """A DOWNGRADED run that still commits keeps its memo, or repeat
    suppression sees an empty memo every poll and every repeat notifies.
    (A LOW DISK run commits nothing today; a CAPPED run that acts on a subset
    and commits for exactly those groups -- issue #67 -- does.)"""
    for notice in (LOW_DISK_NOTICE, CAPPED_NOTICE):
        state = {"version": 1, mm.NOTIFIED_DOWNGRADE_KEY: "LOW DISK"}
        mm.retire_downgrade_memo(state, notice)
        assert state[mm.NOTIFIED_DOWNGRADE_KEY] == "LOW DISK"


def test_retire_downgrade_memo_clears_on_healthy_run():
    for notice in (None, "SEED RUN — actions suppressed",
                   "PER-PROGRAM SEED — program(s) 1182"):
        state = {"version": 1, mm.NOTIFIED_DOWNGRADE_KEY: "LOW DISK"}
        mm.retire_downgrade_memo(state, notice)
        assert mm.NOTIFIED_DOWNGRADE_KEY not in state


def test_main_commit_state_without_report_clears_memo(monkeypatch, tmp_path):
    """A --commit-state poll with --report off (the weekly scrontab entry's
    shape until this PR) never runs act_report, so main()'s commit is the ONLY
    thing that ends a downgrade episode.  Without it a stale 'LOW DISK' memo is committed
    to disk and the next episode's first notice edits in place, reaching
    nobody -- the #71 defect with a one-week fuse."""
    monkeypatch.setattr(mm, "mast_login_if_token", lambda: False)
    monkeypatch.setattr(mm, "query_program",
                        lambda prog: [_planned_row("jw10678-o101_t101_nircam")])
    monkeypatch.setattr(mm, "act_report",
                        lambda *a, **kw: pytest.fail("--report is off"))
    state = tmp_path / "state.json"
    _seed_state(state, program=10678, obs_id="jw10678-o001_x")
    armed = mm.load_state(str(state))
    armed[mm.NOTIFIED_DOWNGRADE_KEY] = "LOW DISK"        # episode in progress
    mm.save_state(str(state), armed)

    assert mm.main(["--program", "10678", "--commit-state",
                    "--state", str(state)]) == 0
    committed = mm.load_state(str(state))
    assert mm.NOTIFIED_DOWNGRADE_KEY not in committed     # episode retired
    assert "jw10678-o101_t101_nircam" in committed["programs"]["10678"]["obs"]


def test_act_report_regular_issue_edits_with_monitor_marker(monkeypatch):
    """The per-observation (non-treasury) branch selects the MONITOR marker.
    With STATUS_MARKER it would edit the pipeline status comment and silently
    delete something a human needed, with a green suite."""
    from data_qa import status_report
    posted = _patch_post_status(monkeypatch)
    planned_brick = dict(_trigger_events("001")[0], released=False,
                         calib_level=-1, t_obs_release=None)
    mm.act_report([planned_brick], execute=True)
    (title, _, kw), = posted
    assert title == "Brick — jw02221-o001 (NIRCam)"      # regular-issue branch
    assert kw["update_last"] is True
    assert kw["marker"] == status_report.MONITOR_MARKER


def test_act_report_partial_notify_failure_does_not_memo(monkeypatch, tmp_path):
    """A fresh downgrade where one notifying comment posts and another FAILS
    leaves the memo unarmed: arming it would send the failed issue back to
    quiet edits forever, and its watchers would never see the notice."""
    from data_qa import status_report
    posted, rcs = [], iter([0, 4])
    monkeypatch.setattr(status_report, "post_status",
                        lambda title, body, **kw: posted.append((title, kw))
                        or next(rcs))
    state_path = str(tmp_path / "state.json")
    state = {"version": 1, "programs": {}}
    mm.act_report(_trigger_events("001") + _planned_events(), execute=True,
                  notice=LOW_DISK_NOTICE, state=state, state_path=state_path)
    assert [kw["update_last"] for _, kw in posted] == [False, False]
    assert mm.NOTIFIED_DOWNGRADE_KEY not in state
    assert not os.path.exists(state_path)


def test_main_weekly_commit_report_without_auto_clears_memo(monkeypatch,
                                                            tmp_path):
    """The deployed weekly entry (--commit-state --report --execute, no
    --auto) never runs the disk gate, so it carries no notice and ends any
    downgrade episode it lands in: an ongoing LOW DISK wedge notifies again on
    the next --auto poll.  Pins what act_report's docstring states."""
    posted = _patch_post_status(monkeypatch)
    monkeypatch.setattr(mm, "mast_login_if_token", lambda: False)
    monkeypatch.setattr(mm.shutil, "disk_usage",
                        lambda p: pytest.fail("no --auto: no disk gate"))
    monkeypatch.setattr(mm, "query_program",
                        lambda prog: [_planned_row("jw10678-o101_t101_nircam")])
    state = tmp_path / "state.json"
    _seed_state(state, program=10678, obs_id="jw10678-o001_x")
    armed = mm.load_state(str(state))
    armed[mm.NOTIFIED_DOWNGRADE_KEY] = "LOW DISK"        # wedge in progress
    mm.save_state(str(state), armed)

    assert mm.main(["--program", "10678", "--commit-state", "--report",
                    "--execute", "--state", str(state)]) == 0
    assert posted[-1][2]["update_last"] is True          # planned only: quiet
    assert mm.NOTIFIED_DOWNGRADE_KEY not in mm.load_state(str(state))

    # Tuesday: the wedge is still on, and it announces itself afresh
    monkeypatch.setattr(mm, "act_download", lambda *a, **kw: None)
    monkeypatch.setattr(mm, "act_trigger", lambda *a, **kw: None)
    monkeypatch.setattr(mm.shutil, "disk_usage", lambda p: _Usage(1e12))
    monkeypatch.setattr(mm, "query_program",
                        lambda prog: [_planned_row("jw10678-o101_t101_nircam"),
                                      _planned_row("jw10678-o102_t102_nircam",
                                                   target="GC_2")])
    assert mm.main(["--program", "10678", "--auto", "--state", str(state),
                    "--download-dir", str(tmp_path)]) == 0
    assert posted[-1][2]["update_last"] is False         # re-notified


def test_main_seed_run_keeps_low_disk_notice_and_memo(monkeypatch, tmp_path):
    """--seed on an --auto run whose disk gate failed keeps the LOW DISK
    warning: the seed text is appended, so the warning reaches every comment
    body and the last-notified memo survives the seed commit.  The seed also
    commits the baseline the disk gate withheld, which is what the gate comment
    at the --auto downgrade calls out as the exception to "the state is NOT
    committed"."""
    posted = _patch_post_status(monkeypatch)
    monkeypatch.setattr(mm, "mast_login_if_token", lambda: False)
    monkeypatch.setattr(mm, "act_download", lambda *a, **kw: None)
    monkeypatch.setattr(mm, "act_trigger", lambda *a, **kw: None)
    monkeypatch.setattr(mm.shutil, "disk_usage", lambda p: _Usage(1e12))
    monkeypatch.setattr(mm, "query_program",
                        lambda prog: [_planned_row("jw10678-o101_t101_nircam")])
    state = tmp_path / "state.json"
    _seed_state(state, program=10678, obs_id="jw10678-o001_x")

    assert mm.main(["--program", "10678", "--auto", "--seed",
                    "--state", str(state),
                    "--download-dir", str(tmp_path)]) == 0
    (_, body, kw), = posted
    assert "LOW DISK" in body and "SEED RUN" in body
    assert kw["update_last"] is False                    # fresh downgrade
    committed = mm.load_state(str(state))
    assert committed[mm.NOTIFIED_DOWNGRADE_KEY] == "LOW DISK"
    # the seed commits the baseline even though the disk gate cleared
    # commit_state: the polled observation is now in the state file
    assert "jw10678-o101_t101_nircam" in committed["programs"]["10678"]["obs"]
