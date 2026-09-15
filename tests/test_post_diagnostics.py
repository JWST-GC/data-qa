"""Tests for data_qa.post_diagnostics.unpost_stage.

The suite is offline: ``_req``, ``_issue_number`` and ``_find_stage_comment`` are stubbed so the
comment-removal path is exercised without the network (the same HTTP-mocking style used by the
pagination tests in data_qa/tests/test_diagnostics.py)."""
from data_qa.observations import Observation


def _obs():
    return Observation(program="2221", obs="001", target="Brick", release_field="brick",
                       instrument="NIRCam", filters=["F212N"], visits=["001"], epoch="", notes="")


def _record_req(calls, status=204):
    def fake(method, url, token, data=None, headers=None, raw=False, want_headers=False):
        calls.append((method, url))
        return (status, {})
    return fake


def test_unpost_stage_deletes_existing_comment(monkeypatch):
    from data_qa import post_diagnostics as P
    calls = []
    monkeypatch.setattr(P, "_issue_number", lambda repo, token, title: 5)
    monkeypatch.setattr(P, "_find_stage_comment", lambda repo, token, num, marker: {"id": 42})
    monkeypatch.setattr(P, "_req", _record_req(calls))
    out = P.unpost_stage(_obs(), 10, "JWST-GC/data-qa", token="tok")
    assert out == 42
    assert len(calls) == 1 and calls[0][0] == "DELETE" and calls[0][1].endswith("/comments/42")


def test_unpost_stage_noop_when_comment_absent(monkeypatch):
    from data_qa import post_diagnostics as P
    calls = []
    monkeypatch.setattr(P, "_issue_number", lambda repo, token, title: 5)
    monkeypatch.setattr(P, "_find_stage_comment", lambda repo, token, num, marker: None)
    monkeypatch.setattr(P, "_req", _record_req(calls))
    out = P.unpost_stage(_obs(), 10, "JWST-GC/data-qa", token="tok")
    assert out is None
    assert calls == []                               # no HTTP call, no error


def test_unpost_stage_noop_when_issue_missing(monkeypatch):
    from data_qa import post_diagnostics as P
    calls = []
    monkeypatch.setattr(P, "_issue_number", lambda repo, token, title: None)

    def _boom(*a, **k):                              # neither lookup nor delete should run
        raise AssertionError("no comment lookup when the issue is missing")
    monkeypatch.setattr(P, "_find_stage_comment", _boom)
    monkeypatch.setattr(P, "_req", _record_req(calls))
    out = P.unpost_stage(_obs(), 10, "JWST-GC/data-qa", token="tok")
    assert out is None
    assert calls == []
