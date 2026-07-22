"""Fetch-failure loudness: a network error must never silently look like an empty
release manifest (live issue: the weekly sync 'synced' an empty registry)."""
import urllib.error
import urllib.request

import pytest

from data_qa import make_issues, observations


@pytest.fixture(autouse=True)
def _clean_errors():
    observations.LAST_FETCH_ERRORS.clear()
    yield
    observations.LAST_FETCH_ERRORS.clear()


def _raise_urlerror(*args, **kwargs):
    raise urllib.error.URLError("simulated network failure")


def test_read_lines_failure_is_loud(monkeypatch, capsys):
    monkeypatch.setattr(urllib.request, "urlopen", _raise_urlerror)
    out = observations._read_lines("https://example.invalid/brick_images.txt")
    assert out == []
    err = capsys.readouterr().err
    assert "fetch FAILED" in err and "brick_images.txt" in err
    assert observations.LAST_FETCH_ERRORS   # recorded, not just printed


def test_read_lines_local_path_unaffected(tmp_path):
    p = tmp_path / "brick_images.txt"
    p.write_text("a.fits\n\nb.fits\n")
    assert observations._read_lines(str(p)) == ["a.fits", "b.fits"]
    assert observations.LAST_FETCH_ERRORS == []


def test_discover_resets_errors_and_records_failures(monkeypatch, capsys):
    monkeypatch.setattr(urllib.request, "urlopen", _raise_urlerror)
    observations.LAST_FETCH_ERRORS.append("stale-from-earlier")
    obs = observations.discover_from_release(fields={"brick": "Brick"})
    assert obs == []
    assert "stale-from-earlier" not in observations.LAST_FETCH_ERRORS
    assert len(observations.LAST_FETCH_ERRORS) == 1
    assert "fetch FAILED" in capsys.readouterr().err


def test_make_issues_aborts_on_empty_registry_with_fetch_errors(monkeypatch, capsys):
    def fake_registry(**kwargs):
        observations.LAST_FETCH_ERRORS.append("fetch FAILED: manifest: URLError")
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


def test_make_issues_fetch_lines_delegates_loudly(monkeypatch, capsys):
    monkeypatch.setattr(urllib.request, "urlopen", _raise_urlerror)
    assert make_issues._fetch_lines("https://example.invalid/x.txt") == []
    assert "fetch FAILED" in capsys.readouterr().err
