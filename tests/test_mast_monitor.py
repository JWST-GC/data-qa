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
    unreleased = {"cloudef", "sgra", "ngc6334",
                  "arches", "quintuplet"}          # no release page yet
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


def test_capped_run_acts_on_nothing(monkeypatch, tmp_path, capsys):
    calls = _patch_poll(monkeypatch, _rows_n_groups(3))
    monkeypatch.setattr(mm.shutil, "disk_usage", lambda p: _Usage(50e12))
    state = tmp_path / "state.json"
    _seed_state(state)                           # not a first run
    rc = mm.main(["--program", "2221", "--auto", "--max-submit", "2",
                  "--state", str(state), "--download-dir", str(tmp_path)])
    assert rc == 0
    acted = dict(calls)
    assert set(acted) == {"report"}              # all-or-nothing: nothing acted
    assert "CAPPED" in acted["report"]["notice"]
    committed = mm.load_state(str(state))        # state NOT committed: re-fires
    assert "jw02221-o001_t001_nircam_clear-f405n" not in \
        committed["programs"]["2221"]["obs"]
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
    ("F444W;F470N", ["F444W", "F470N"]),
    ("F212N;F480M", ["F212N", "F480M"]),
    ("F150W2;CLEAR", ["F150W2"]),
    ("F770W", ["F770W"]),                  # MIRI 3-digit
    ("F1000W;F770W", ["F1000W", "F770W"]),  # MIRI 4-digit
    ("GRISMR;F322W2", ["F322W2"]),
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


def test_act_report_updates_in_place_with_monitor_marker(monkeypatch):
    from data_qa import status_report
    posted = []
    monkeypatch.setattr(status_report, "post_status",
                        lambda title, body, **kw: posted.append((title, body, kw)))
    mm.act_report(_trigger_events("001"), execute=True, notice="CAPPED — x")
    (title, body, kw), = posted
    assert kw["update_last"] is True
    assert kw["marker"] == status_report.MONITOR_MARKER
    assert "CAPPED" in body


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
_PIPE_FILE = ("/blue/adamginsburg/adamginsburg/repos/jwst-gc-pipeline/"
              "jwst_gc_pipeline/reduction/PipelineRerunNIRCAM-LONG.py")
# Globular-cluster programs ride the pipeline for testing only; they are not
# GC-monitor targets.  Arches/Quintuplet (2045) and Sgr A* (1939) ARE GC fields.
_GLOBULAR_PROGRAMS = {1334, 1979, 8322, 12587}


def _pipeline_field_map():
    import ast
    with open(_PIPE_FILE) as fh:
        tree = ast.parse(fh.read())
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for tgt in node.targets:
                if isinstance(tgt, ast.Name) and tgt.id == "field_to_reg_mapping":
                    val = node.value
                    if isinstance(val, ast.Subscript):
                        val = val.value
                    return ast.literal_eval(val)
    return None


@pytest.mark.skipif(not __import__("os").path.exists(_PIPE_FILE),
                    reason="jwst-gc-pipeline checkout not available")
def test_programs_complete_vs_pipeline_field_mapping():
    """Every GC program/obs the reduction pipeline maps must be monitored
    (globular-cluster test programs excluded)."""
    mapping = _pipeline_field_map()
    assert mapping, "could not parse field_to_reg_mapping from the pipeline"
    for prog_str, obsmap in mapping.items():
        prog = int(prog_str)
        if prog in _GLOBULAR_PROGRAMS:
            continue
        assert prog in mm.PROGRAMS, \
            f"pipeline maps program {prog} but mast_monitor.PROGRAMS lacks it"
        for obsnum, field in obsmap.items():
            assert obsnum in mm.PROGRAMS[prog], (prog, obsnum)
            assert mm.PROGRAMS[prog][obsnum] == field, (prog, obsnum, field)


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
