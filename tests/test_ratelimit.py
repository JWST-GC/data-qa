"""GitHub rate-limit handling (data_qa._ratelimit) and the unchanged-post skip in post_stage."""
import json

import pytest

from data_qa import _ratelimit as R


SECONDARY = {"message": "You have exceeded a secondary rate limit. Please wait a few minutes."}


def test_is_rate_limited_distinguishes_limits_from_permission_errors():
    assert R.is_rate_limited(403, SECONDARY, {})
    assert R.is_rate_limited(403, {"message": "API rate limit exceeded for user ID 143715."}, {})
    assert R.is_rate_limited(429, {}, {"Retry-After": "30"})
    assert R.is_rate_limited(403, {}, {"X-RateLimit-Remaining": "0"})
    assert not R.is_rate_limited(403, {"message": "Resource not accessible by integration"}, {})
    assert not R.is_rate_limited(200, SECONDARY, {"Retry-After": "30"})


def test_retry_wait_prefers_retry_after_then_reset_then_backoff():
    assert R.retry_wait({"retry-after": "30"}, 0) == 30
    assert R.retry_wait({"X-RateLimit-Remaining": "0", "X-RateLimit-Reset": "1100"}, 0,
                        now=1000.0) == pytest.approx(101.0)
    for attempt, base in ((0, 60.0), (2, 240.0)):
        assert 0.8 * base <= R.retry_wait({}, attempt) <= 1.2 * base
    assert R.retry_wait({}, 10) <= 1.2 * 900.0


def test_token_bucket_bursts_then_paces_and_honours_pause():
    st = {}
    assert all(R._take(st, 0.0, 0.5, 3) == 0 for _ in range(3))      # burst of 3
    assert R._take(st, 0.0, 0.5, 3) == pytest.approx(2.0)              # 1 token at 0.5/s
    assert R._take(st, 2.0, 0.5, 3) == 0
    st["pause_until"] = 100.0
    assert R._take(st, 40.0, 0.5, 3) == pytest.approx(60.0)


def test_write_budget_is_shared_through_the_state_file(tmp_path, monkeypatch):
    monkeypatch.setenv("QA_WRITE_BUCKET", str(tmp_path / "b" / "bucket.json"))
    monkeypatch.setenv("QA_WRITE_RATE_PER_MIN", "60")
    monkeypatch.setenv("QA_WRITE_BURST", "2")
    clock = {"t": 1000.0}
    slept = []

    def sleep(s):
        slept.append(s)
        clock["t"] += s
    for _ in range(2):                       # two "elements" each take one token: no wait
        R.acquire_write(sleep=sleep, clock=lambda: clock["t"])
    assert slept == []
    R.acquire_write(sleep=sleep, clock=lambda: clock["t"])   # third waits ~1 s for a refill
    assert len(slept) == 1 and 1.0 <= slept[0] <= 1.5
    state = json.loads((tmp_path / "b" / "bucket.json").read_text())
    assert state["tokens"] < 1


def test_call_retries_rate_limited_writes_and_pauses_everyone(tmp_path, monkeypatch):
    monkeypatch.setenv("QA_WRITE_BUCKET", str(tmp_path / "bucket.json"))
    replies = [(403, SECONDARY, {"Retry-After": "5"}), (429, {}, {"Retry-After": "7"}),
               (201, {"id": 1}, {})]
    slept = []
    st, payload, _ = R.call("POST", lambda: replies.pop(0), sleep=slept.append)
    assert st == 201 and payload == {"id": 1}
    assert slept[:1] == [5.0] and 7.0 in slept
    assert json.loads((tmp_path / "bucket.json").read_text())["pause_until"] > 0


def test_call_returns_refusal_once_the_wait_budget_is_spent(monkeypatch):
    monkeypatch.setattr(R, "MAX_TOTAL_WAIT_S", 10.0)
    slept = []
    st, payload, _ = R.call("GET", lambda: (403, SECONDARY, {"Retry-After": "60"}), sleep=slept.append)
    assert st == 403 and slept == []


def test_call_passes_non_limit_errors_straight_through():
    calls = []
    st, _, _ = R.call("PATCH", lambda: calls.append(1) or (404, {"message": "Not Found"}, {}),
                      sleep=lambda s: None)
    assert st == 404 and calls == [1]


# ----------------------------------------------------------------- unchanged-post skip

def _obs():
    from data_qa.observations import Observation
    return Observation(program="10678", obs="066", target="GC Treasury", release_field="gc-treasury",
                       instrument="NIRCam", filters=["F212N", "F480M"])


def _post(monkeypatch, tmp_path, existing_body, caption="cap", png_bytes=b"png"):
    from data_qa import post_diagnostics as P
    calls = []
    png = tmp_path / "jw10678-o066_stage4.png"
    png.write_bytes(png_bytes)
    monkeypatch.setattr(P, "_issue_number", lambda repo, token, title: 5)
    monkeypatch.setattr(P, "_find_stage_comment", lambda repo, token, num, marker:
                        None if existing_body is None else {"id": 42, "body": existing_body, "html_url": "h"})
    monkeypatch.setattr(P, "upload_asset", lambda *a, **k: calls.append("upload") or "url")
    bodies = []

    def req(method, url, token, data=None, **k):
        calls.append(method)
        bodies.append(json.loads(data)["body"])
        return 200, {"html_url": "u"}
    monkeypatch.setattr(P, "_req", req)
    P.post_stage(_obs(), 4, str(png), caption, "JWST-GC/data-qa", token="tok")
    return calls, bodies


def test_post_stage_skips_an_unchanged_comment(tmp_path, monkeypatch):
    calls, bodies = _post(monkeypatch, tmp_path, None)
    assert calls == ["upload", "POST"] and "data-qa:sha:" in bodies[0]
    first = bodies[0]
    calls, _ = _post(monkeypatch, tmp_path, first)                 # identical rebuild
    assert calls == []
    calls, _ = _post(monkeypatch, tmp_path, first, caption="new caption")
    assert calls == ["upload", "PATCH"]
    calls, _ = _post(monkeypatch, tmp_path, first, png_bytes=b"other png")
    assert calls == ["upload", "PATCH"]
    monkeypatch.setenv("QA_FORCE_REPOST", "1")
    calls, _ = _post(monkeypatch, tmp_path, first)
    assert calls == ["upload", "PATCH"]


def test_comment_without_a_hash_is_reposted_once(tmp_path, monkeypatch):
    # comments posted before the hash existed carry none: the first refresh re-posts and stamps it
    calls, bodies = _post(monkeypatch, tmp_path, "<!-- data-qa:diag:stage4 -->\nold body")
    assert calls == ["upload", "PATCH"] and "data-qa:sha:" in bodies[0]
