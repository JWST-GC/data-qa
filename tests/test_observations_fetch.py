"""registry() is built from MAST over the curated PROGRAMS obsid->field map.
Covers: curated-obsid selection, calib_level>=2 gate, per-instrument grouping, and
loud recording of a MAST failure (a network error must never silently look like an
empty registry -- live issue #4: the weekly sync 'synced' an empty registry)."""
import pytest

from data_qa import make_issues, mast_monitor, observations


def _row(obs_id, filters, inst="NIRCAM/IMAGE", calib=2, t_max=59800.0):
    return {"obs_id": obs_id, "t_max": t_max, "t_obs_release": 59900.0,
            "calib_level": calib, "instrument_name": inst, "filters": filters,
            "target_name": "GC"}


@pytest.fixture(autouse=True)
def _clean_errors():
    observations.LAST_FETCH_ERRORS.clear()
    yield
    observations.LAST_FETCH_ERRORS.clear()


def test_registry_from_mast_over_curated_programs(monkeypatch):
    """All curated gc2211 obsids get an Observation from MAST -- including ones the
    web release never listed -- and uncurated obsids of the same program do not."""
    rows = [
        _row("jw02211-o023_t001_nircam", "F200W;F277W"),
        _row("jw02211-o028_t001_nircam", "F150W;F277W"),
        _row("jw02211-o046_t001_nircam", "F200W;F277W"),
        _row("jw02211-o049_t001_nircam", "F200W;F277W"),
        _row("jw02211-o050_t001_nircam", "F200W;F277W"),
        _row("jw02211-o099_t001_nircam", "F200W"),              # NOT in PROGRAMS -> excluded
        _row("jw02211-o028_t001_nircam", "CLEAR;F150W", calib=-1),  # planned dup -> ignored
    ]
    monkeypatch.setattr(mast_monitor, "query_program", lambda prog: rows)
    obs = observations.registry(programs=[2211])
    assert {o.obs for o in obs} == {"023", "028", "046", "049", "050"}
    byobs = {o.obs: o for o in obs}
    assert byobs["028"].filters == ["F150W", "F277W"]           # F277W present (MAST, not on-disk)
    assert all(o.instrument == "NIRCam" and o.target == "GC 2211"
               and o.field == "gc2211" for o in obs)
    assert observations.LAST_FETCH_ERRORS == []


def test_registry_excludes_uncalibrated(monkeypatch):
    """A curated obsid with only uncalibrated products (calib_level < 2) has nothing
    to QA yet -> no Observation."""
    monkeypatch.setattr(mast_monitor, "query_program",
                        lambda prog: [_row("jw02211-o023_t001_nircam", "F200W", calib=1)])
    assert observations.registry(programs=[2211]) == []


def test_registry_groups_by_instrument(monkeypatch):
    """NIRCam and MIRI of the same (program, obs) are separate deliveries -> two
    Observations (two issues)."""
    rows = [_row("jw05365-o001_t001_nircam", "F200W", inst="NIRCAM/IMAGE"),
            _row("jw05365-o001_t001_miri", "F770W", inst="MIRI/IMAGE")]
    monkeypatch.setattr(mast_monitor, "query_program", lambda prog: rows)
    obs = observations.registry(programs=[5365])
    assert {o.instrument for o in obs} == {"NIRCam", "MIRI"}
    assert all(o.obs == "001" and o.field == "sgrb2" for o in obs)


def test_registry_records_mast_failure_loudly(monkeypatch, capsys):
    """A MAST query failure contributes no observations but is RECORDED (never a
    silent empty), so make_issues can refuse to sync."""
    import requests

    def boom(prog):
        raise requests.exceptions.ConnectionError("simulated MAST outage")
    monkeypatch.setattr(mast_monitor, "query_program", boom)
    obs = observations.registry(programs=[2211])
    assert obs == []
    assert observations.LAST_FETCH_ERRORS                       # recorded, not just printed
    assert "MAST query FAILED" in capsys.readouterr().err


def test_make_issues_aborts_on_empty_registry_with_fetch_errors(monkeypatch, capsys):
    def fake_registry(**kwargs):
        observations.LAST_FETCH_ERRORS.append("MAST query FAILED: program 2211: ConnectionError")
        return []
    monkeypatch.setattr(make_issues, "registry", fake_registry)
    rc = make_issues.main(["--dry-run"])
    assert rc == 3                                     # loud abort, distinct code
    err = capsys.readouterr().err
    assert "ABORT" in err and "refusing" in err


def test_make_issues_genuinely_empty_keeps_old_behavior(monkeypatch, capsys):
    monkeypatch.setattr(make_issues, "registry", lambda **kwargs: [])
    rc = make_issues.main(["--dry-run"])
    assert rc == 1                                     # unchanged exit path
    assert "no matching observations" in capsys.readouterr().err
