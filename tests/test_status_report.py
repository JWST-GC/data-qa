"""Offline unit tests for data_qa.status_report (rendering + comment routing;
no GitHub, no squeue)."""
from data_qa import _github, status_report as sr


def test_issue_title_matches_make_issues_convention():
    from data_qa.observations import Observation
    o = Observation(program="2221", obs="001", target="Brick", release_field="brick")
    assert sr.issue_title_for(2221, "001", field="brick") == o.issue_title


def test_render_starts_with_marker_and_timestamp():
    body = sr.render_status(field="brick", program=2221, obsnum="001",
                            jobs=[], state=None, release=None,
                            job_prefix="brick2221", now="2026-07-21 12:00 UTC")
    lines = body.splitlines()
    assert lines[0] == sr.STATUS_MARKER
    assert "2026-07-21 12:00 UTC" in lines[1]
    assert "no queued or running jobs" in body
    assert "no state file" in body


def test_render_job_table():
    jobs = [dict(jobid="123", name="brick2221-o001-reduce-F405N", state="RUNNING",
                 elapsed="1:23:45", reason="c0709a-s1")]
    body = sr.render_status(field="brick", jobs=jobs, job_prefix="brick2221")
    assert "| 123 | `brick2221-o001-reduce-F405N` | RUNNING |" in body


def test_render_squeue_unavailable():
    body = sr.render_status(field="brick", jobs=None, job_prefix="brick2221")
    assert "_squeue unavailable_" in body


def test_render_events_comment_marker():
    events = [dict(event="NEW_OBSERVATION", obs_id="jw02221-o001_x",
                   calib_level=3, t_obs_release=59900.0, filters="F405N")]
    body = sr.render_events_comment(events, now="2026-07-21 12:00 UTC")
    assert body.startswith(sr.STATUS_MARKER)
    assert "NEW_OBSERVATION" in body and "jw02221-o001_x" in body


def test_post_status_dry_run_prints(capsys):
    rc = sr.post_status("Some title", "BODY", dry_run=True)
    out = capsys.readouterr().out
    assert rc == 0
    assert "DRY-RUN" in out and "Some title" in out and "BODY" in out


def _patch_github(monkeypatch, comments):
    calls = {"posted": [], "updated": []}
    monkeypatch.setattr(_github, "get_token", lambda: "tok")
    monkeypatch.setattr(_github, "existing_issues",
                        lambda token, repo: {"T": {"number": 5}})
    monkeypatch.setattr(_github, "list_comments",
                        lambda token, repo, number: comments)
    monkeypatch.setattr(_github, "post_comment",
                        lambda token, repo, number, body:
                        calls["posted"].append((number, body)) or (201, {}))
    monkeypatch.setattr(_github, "update_comment",
                        lambda token, repo, cid, body:
                        calls["updated"].append((cid, body)) or (200, {}))
    return calls


def test_post_status_new_comment(monkeypatch):
    calls = _patch_github(monkeypatch, comments=[])
    rc = sr.post_status("T", "BODY", dry_run=False)
    assert rc == 0
    assert calls["posted"] == [(5, "BODY")]
    assert calls["updated"] == []


def test_post_status_update_last_edits_marker_comment(monkeypatch):
    comments = [
        {"id": 1, "body": "a human comment"},
        {"id": 2, "body": f"{sr.STATUS_MARKER}\nold status"},
        {"id": 3, "body": "another human comment"},
    ]
    calls = _patch_github(monkeypatch, comments)
    rc = sr.post_status("T", "NEW", dry_run=False, update_last=True)
    assert rc == 0
    assert calls["updated"] == [(2, "NEW")]     # edited the marker comment
    assert calls["posted"] == []


def test_post_status_update_last_falls_back_to_post(monkeypatch):
    calls = _patch_github(monkeypatch, comments=[{"id": 1, "body": "human"}])
    rc = sr.post_status("T", "NEW", dry_run=False, update_last=True)
    assert rc == 0
    assert calls["posted"] == [(5, "NEW")]


def test_post_status_missing_issue(monkeypatch):
    monkeypatch.setattr(_github, "get_token", lambda: "tok")
    monkeypatch.setattr(_github, "existing_issues", lambda token, repo: {})
    assert sr.post_status("Nope", "B", dry_run=False) == 3
