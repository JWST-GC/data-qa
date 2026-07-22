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
    assert body.startswith(sr.MONITOR_MARKER)      # monitor comments self-mark...
    assert sr.STATUS_MARKER not in body            # ...and never collide w/ status
    assert "NEW_OBSERVATION" in body and "jw02221-o001_x" in body
    assert "WARNING" not in body


def test_render_events_comment_notice_and_tile():
    events = [dict(event="NEW_OBSERVATION", obs_id="jw10678-o017_x",
                   calib_level=1, t_obs_release=59900.0,
                   filters="F212N;F480M", tile="GC_17")]
    body = sr.render_events_comment(events, now="2026-07-22 12:00 UTC",
                                    notice="LOW DISK: only 1.0 TB free")
    assert "> **WARNING — LOW DISK: only 1.0 TB free**" in body
    assert "tile `GC_17`" in body


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


def test_post_status_update_last_marker_selects_monitor_comment(monkeypatch):
    """The monitor's update-in-place path edits ITS comment, not the status one."""
    comments = [
        {"id": 10, "body": f"{sr.STATUS_MARKER}\npipeline status"},
        {"id": 11, "body": f"{sr.MONITOR_MARKER}\nold monitor events"},
        {"id": 12, "body": "a human comment"},
    ]
    calls = _patch_github(monkeypatch, comments)
    rc = sr.post_status("T", "NEW EVENTS", dry_run=False, update_last=True,
                        marker=sr.MONITOR_MARKER)
    assert rc == 0
    assert calls["updated"] == [(11, "NEW EVENTS")]   # the monitor comment
    assert calls["posted"] == []


def test_post_status_update_last_monitor_marker_absent_posts_new(monkeypatch):
    comments = [{"id": 10, "body": f"{sr.STATUS_MARKER}\npipeline status"}]
    calls = _patch_github(monkeypatch, comments)
    rc = sr.post_status("T", "NEW EVENTS", dry_run=False, update_last=True,
                        marker=sr.MONITOR_MARKER)
    assert rc == 0
    assert calls["updated"] == []                     # status comment untouched
    assert calls["posted"] == [(5, "NEW EVENTS")]


def test_post_status_missing_issue(monkeypatch):
    monkeypatch.setattr(_github, "get_token", lambda: "tok")
    monkeypatch.setattr(_github, "existing_issues", lambda token, repo: {})
    assert sr.post_status("Nope", "B", dry_run=False) == 3


# --------------------------------------------- PLANNED tag (unreleased events)
def test_render_events_comment_planned_tag():
    ev = dict(event="NEW_OBSERVATION", obs_id="jw10678-o101_x", calib_level=-1,
              released=False, t_obs_release=None, filters="F212N;F480M",
              tile="GC_101")
    body = sr.render_events_comment([ev], now="2026-07-22 12:00 UTC")
    assert "PLANNED" in body
    assert "release unknown" in body             # masked date -> 'unknown'


def test_render_events_comment_released_has_no_planned_tag():
    ev = dict(event="NEWLY_RELEASED", obs_id="jw02221-o001_x", calib_level=3,
              released=True, t_obs_release=59900.0, filters="F405N")
    body = sr.render_events_comment([ev], now="2026-07-22 12:00 UTC")
    assert "PLANNED" not in body


# ------------------------------------------ shared issue cache + rolling issue
def test_post_status_shared_issue_cache_fetches_once(monkeypatch):
    """One existing_issues() listing per run when callers share an issue_cache
    (the ~1668-treasury-group rate-limit hazard)."""
    fetches = []
    monkeypatch.setattr(_github, "get_token", lambda: "tok")
    monkeypatch.setattr(_github, "existing_issues",
                        lambda token, repo: fetches.append(repo)
                        or {"T": {"number": 5}})
    monkeypatch.setattr(_github, "list_comments", lambda *a: [])
    monkeypatch.setattr(_github, "post_comment", lambda *a: (201, {}))
    cache = {}
    assert sr.post_status("T", "b1", dry_run=False, issue_cache=cache) == 0
    assert sr.post_status("T", "b2", dry_run=False, issue_cache=cache) == 0
    assert len(fetches) == 1                     # fetched ONCE, reused


def test_post_status_creates_rolling_issue_when_missing(monkeypatch):
    """create_labels turns the rc=3 'no issue' failure into issue creation
    (the treasury rolling-issue path), and the shared cache learns it."""
    created, posted = [], []
    monkeypatch.setattr(_github, "get_token", lambda: "tok")
    monkeypatch.setattr(_github, "existing_issues", lambda token, repo: {})
    monkeypatch.setattr(_github, "ensure_labels", lambda token, repo, names: None)
    monkeypatch.setattr(_github, "create_issue",
                        lambda token, repo, title, body, labels=():
                        created.append((title, list(labels)))
                        or (201, {"number": 9}))
    monkeypatch.setattr(_github, "list_comments", lambda *a: [])
    monkeypatch.setattr(_github, "post_comment",
                        lambda token, repo, number, body:
                        posted.append(number) or (201, {}))
    cache = {}
    rc = sr.post_status("GC Treasury — program 10678 deliveries", "events",
                        dry_run=False, issue_cache=cache,
                        create_labels=["QA", "program:10678"])
    assert rc == 0
    assert created == [("GC Treasury — program 10678 deliveries",
                        ["QA", "program:10678"])]
    assert posted == [9]
    rc = sr.post_status("GC Treasury — program 10678 deliveries", "more",
                        dry_run=False, issue_cache=cache,
                        create_labels=["QA", "program:10678"])
    assert rc == 0
    assert len(created) == 1                     # cached: not re-created
    assert posted == [9, 9]


def test_post_status_missing_issue_still_fails_without_create_labels(monkeypatch):
    """Per-obs issues are still make_issues-owned: no silent creation."""
    monkeypatch.setattr(_github, "get_token", lambda: "tok")
    monkeypatch.setattr(_github, "existing_issues", lambda token, repo: {})
    assert sr.post_status("Nope", "B", dry_run=False, issue_cache={}) == 3
