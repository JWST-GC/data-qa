import pytest
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


def _fake_github(state):
    """Minimal in-memory GitHub releases API for upload_asset: ``state`` maps tag -> {name: id}."""
    import json as _json
    ids = {tag: 1000 + i for i, tag in enumerate(state)}

    def fake(method, url, token, data=None, headers=None, raw=False, want_headers=False):
        state.setdefault("_calls", []).append((method, url))
        if "/releases/tags/" in url:
            tag = url.rsplit("/", 1)[1]
            return (200, {"id": ids[tag], "tag_name": tag}) if tag in ids else (404, {})
        if method == "POST" and url.endswith("/releases"):
            tag = _json.loads(data)["tag_name"]
            ids[tag] = 1000 + len(ids); state[tag] = {}
            return (201, {"id": ids[tag], "tag_name": tag})
        rid_tag = {v: k for k, v in ids.items()}
        if method == "GET" and "/assets?" in url:
            rid = int(url.split("/releases/")[1].split("/")[0])
            page = int(url.rsplit("page=", 1)[1])
            items = [{"name": n, "id": i} for n, i in state[rid_tag[rid]].items()]
            return (200, items[(page - 1) * 100: page * 100])
        if method == "DELETE":
            aid = int(url.rsplit("/", 1)[1])
            for tag in list(ids):
                state[tag] = {n: i for n, i in state.get(tag, {}).items() if i != aid}
            return (204, {})
        if method == "POST" and "assets?name=" in url:
            rid = int(url.split("/releases/")[1].split("/")[0]); name = url.rsplit("=", 1)[1]
            tag = rid_tag[rid]
            if name in state[tag]:
                return (422, {"errors": [{"code": "already_exists"}]})
            if len(state[tag]) >= 1000:
                return (422, {"message": "file_count limited to 1000 assets per release"})
            state[tag][name] = 50000 + sum(len(v) for k, v in state.items() if k != "_calls")
            return (201, {"id": state[tag][name], "browser_download_url": f"https://dl/{tag}/{name}"})
        raise AssertionError((method, url))
    return fake


def test_upload_asset_goes_to_stable_shard_and_frees_legacy(tmp_path, monkeypatch):
    from data_qa import post_diagnostics as P
    png = tmp_path / "x.png"; png.write_bytes(b"png")
    # legacy bucket FULL (1000 assets) and already holding this name
    legacy = {f"old{i}.png": i for i in range(999)}
    legacy["jw10678-o132_stage3.png"] = 999
    state = {"qa-assets": legacy}
    monkeypatch.setattr(P, "_req", _fake_github(state))
    monkeypatch.setattr(P, "_RELEASES", {}); monkeypatch.setattr(P, "_ASSET_INDEX", {})
    url = P.upload_asset("o/r", "tok", str(png), "jw10678-o132_stage3.png")
    tag = P._asset_shard_tag("jw10678-o132_stage3.png")
    assert tag.startswith("qa-assets-") and url == f"https://dl/{tag}/jw10678-o132_stage3.png"
    assert "jw10678-o132_stage3.png" not in state["qa-assets"]      # legacy copy freed
    # re-upload in a fresh process replaces in place (same shard, no already_exists)
    monkeypatch.setattr(P, "_RELEASES", {}); monkeypatch.setattr(P, "_ASSET_INDEX", {})
    assert P.upload_asset("o/r", "tok", str(png), "jw10678-o132_stage3.png") == url
    assert list(state[tag]) == ["jw10678-o132_stage3.png"]


def test_asset_shard_tag_is_stable_and_spread():
    from data_qa import post_diagnostics as P
    names = [f"jw10678-o{o:03d}_stage{s}.png" for o in range(40, 140) for s in range(1, 13)]
    tags = [P._asset_shard_tag(n) for n in names]
    assert tags == [P._asset_shard_tag(n) for n in names]           # deterministic
    from collections import Counter
    assert max(Counter(tags).values()) < 1000                        # every shard under the cap
    assert len(set(tags)) == P.ASSET_SHARDS


def test_upload_failure_keeps_legacy_copy(tmp_path, monkeypatch):
    from data_qa import post_diagnostics as P
    png = tmp_path / "x.png"; png.write_bytes(b"png")
    name = "jw10678-o132_stage3.png"
    tag = P._asset_shard_tag(name)
    state = {"qa-assets": {name: 7}, tag: {f"f{i}.png": 100 + i for i in range(1000)}}  # shard full
    monkeypatch.setattr(P, "_req", _fake_github(state))
    monkeypatch.setattr(P, "_RELEASES", {}); monkeypatch.setattr(P, "_ASSET_INDEX", {})
    with pytest.raises(P.PostError):
        P.upload_asset("o/r", "tok", str(png), name)
    assert state["qa-assets"] == {name: 7}                          # old comment image still live
    posts = [u for m, u in state["_calls"] if m == "POST" and "assets?name=" in u]
    assert len(posts) == 1                                          # a full shard is not retried


def test_upload_replaces_asset_a_concurrent_task_uploaded(tmp_path, monkeypatch):
    """Two array tasks posting the same obs (two issues for 2092 o005): the other task uploads the
    name after this process cached the release index, so the POST hits 422 already_exists."""
    from data_qa import post_diagnostics as P
    png = tmp_path / "x.png"; png.write_bytes(b"png")
    name = "jw02092-o005_stage10.png"
    tag = P._asset_shard_tag(name)
    state = {"qa-assets": {}, tag: {}}
    monkeypatch.setattr(P, "_req", _fake_github(state))
    monkeypatch.setattr(P, "_RELEASES", {}); monkeypatch.setattr(P, "_ASSET_INDEX", {})
    rel = P._ensure_release("o/r", "tok", tag)
    P._asset_index("o/r", "tok", tag, rel)                          # index cached while empty
    state[tag][name] = 77                                           # the other task's upload
    url = P.upload_asset("o/r", "tok", str(png), name)
    assert url == f"https://dl/{tag}/{name}"
    assert list(state[tag]) == [name] and state[tag][name] != 77    # replaced, not duplicated


def test_upload_retries_404_from_concurrent_replace(tmp_path, monkeypatch):
    """#38/#115 share jw02092-o005: while the other task replaces the asset, our POST can get 404.
    The upload backs off, re-reads the index and retries instead of failing the stage."""
    from data_qa import post_diagnostics as P
    png = tmp_path / "x.png"; png.write_bytes(b"png")
    name = "jw02092-o005_stage7.png"
    tag = P._asset_shard_tag(name)
    state = {"qa-assets": {}, tag: {}}
    real = _fake_github(state)
    seen = {"posts": 0}

    def flaky(method, url, token, **kw):
        if method == "POST" and "assets?name=" in url:
            seen["posts"] += 1
            if seen["posts"] <= 2:
                return (404, {"message": "Not Found"})
        return real(method, url, token, **kw)
    monkeypatch.setattr(P, "_req", flaky)
    monkeypatch.setattr(P, "_RELEASES", {}); monkeypatch.setattr(P, "_ASSET_INDEX", {})
    monkeypatch.setattr(P, "_UPLOAD_RACE_SLEEP_S", 0)
    assert P.upload_asset("o/r", "tok", str(png), name) == f"https://dl/{tag}/{name}"
    assert seen["posts"] == 3 and list(state[tag]) == [name]


def test_upload_gives_up_after_bounded_404_retries(tmp_path, monkeypatch):
    from data_qa import post_diagnostics as P
    png = tmp_path / "x.png"; png.write_bytes(b"png")
    name = "jw02092-o005_stage6.png"
    tag = P._asset_shard_tag(name)
    state = {"qa-assets": {}, tag: {}}
    real = _fake_github(state)
    seen = {"posts": 0}

    def always404(method, url, token, **kw):
        if method == "POST" and "assets?name=" in url:
            seen["posts"] += 1
            return (404, {"message": "Not Found"})
        return real(method, url, token, **kw)
    monkeypatch.setattr(P, "_req", always404)
    monkeypatch.setattr(P, "_RELEASES", {}); monkeypatch.setattr(P, "_ASSET_INDEX", {})
    monkeypatch.setattr(P, "_UPLOAD_RACE_SLEEP_S", 0)
    with pytest.raises(P.PostError):
        P.upload_asset("o/r", "tok", str(png), name)
    assert seen["posts"] == 1 + P._UPLOAD_RACE_RETRIES

def test_ensure_release_rereads_after_concurrent_create(monkeypatch):
    from data_qa import post_diagnostics as P
    seen = {"gets": 0}

    def fake(method, url, token, data=None, headers=None, raw=False, want_headers=False):
        if method == "GET":
            seen["gets"] += 1
            # first GET: missing; after our POST loses the race, the re-read finds it
            return (404, {}) if seen["gets"] == 1 else (200, {"id": 9, "tag_name": "qa-assets-03"})
        return (422, {"errors": [{"code": "already_exists"}]})
    monkeypatch.setattr(P, "_req", fake)
    monkeypatch.setattr(P, "_RELEASES", {})
    assert P._ensure_release("o/r", "tok", "qa-assets-03")["id"] == 9
    assert seen["gets"] == 2


def _guarded_post(monkeypatch, existing_body, caption, stage=5):
    from data_qa import post_diagnostics as P
    calls = []
    monkeypatch.setattr(P, "_issue_number", lambda repo, token, title: 5)
    monkeypatch.setattr(P, "_find_stage_comment",
                        lambda repo, token, num, marker: {"id": 42, "body": existing_body})
    monkeypatch.setattr(P, "upload_asset", lambda *a, **k: calls.append("upload") or "url")
    monkeypatch.setattr(P, "_req", lambda *a, **k: calls.append(a[0]) or (200, {"html_url": "u"}))
    P.post_stage(_obs(), stage, "x.png", caption, "JWST-GC/data-qa", token="tok")
    return calls


def test_post_stage_refuses_roll_correction_downgrade(monkeypatch):
    # a run without the roll tag must not overwrite a roll-corrected stage-5 comment (#346)
    from data_qa import post_diagnostics as P
    monkeypatch.delenv("QA_ALLOW_ROLL_DOWNGRADE", raising=False)
    with pytest.raises(P.PostError, match="roll-corrected"):
        _guarded_post(monkeypatch, "... per-visit roll correction applied ...", "plain caption")


def test_post_stage_roll_guard_allows_corrected_and_other_stages(monkeypatch):
    monkeypatch.delenv("QA_ALLOW_ROLL_DOWNGRADE", raising=False)
    old = "... roll correction applied ..."
    assert "PATCH" in _guarded_post(monkeypatch, old, "new: roll correction applied (set v1)")
    assert "PATCH" in _guarded_post(monkeypatch, "uncorrected", "plain caption")
    assert "PATCH" in _guarded_post(monkeypatch, old, "plain caption", stage=4)
    monkeypatch.setenv("QA_ALLOW_ROLL_DOWNGRADE", "1")
    assert "PATCH" in _guarded_post(monkeypatch, old, "plain caption")


def test_roll_guard_phrase_matches_stage5_caption():
    # the guard keys on the caption text: keep the two in step
    from data_qa import diagnostics as D, post_diagnostics as P
    cap = D.caption_for(5, dict(single_module="NRCB", roll_tag="v1", roll_corrected=True))
    assert P._ROLL_APPLIED in cap
    assert P._ROLL_APPLIED not in D.caption_for(5, dict(single_module="NRCB", roll_tag="v1",
                                                        roll_corrected=False))


def test_unpost_stage_refuses_to_delete_roll_corrected_stage5(monkeypatch):
    from data_qa import post_diagnostics as P
    monkeypatch.delenv("QA_ALLOW_ROLL_DOWNGRADE", raising=False)
    calls = []
    monkeypatch.setattr(P, "_issue_number", lambda repo, token, title: 5)
    monkeypatch.setattr(P, "_find_stage_comment", lambda repo, token, num, marker:
                        {"id": 42, "body": "roll correction applied"})
    monkeypatch.setattr(P, "_req", _record_req(calls))
    with pytest.raises(P.RollDowngradeError):
        P.unpost_stage(_obs(), 5, "JWST-GC/data-qa", token="tok")
    assert calls == []
    assert P.unpost_stage(_obs(), 4, "JWST-GC/data-qa", token="tok") == 42   # other stages


def test_refused_roll_downgrade_keeps_prior_stage5_metrics(tmp_path, monkeypatch):
    # the corrected comment stays, so metrics.json must keep the corrected stage-5 numbers
    import json
    from data_qa import diagnostics as D, post_diagnostics as P
    o = _obs()
    monkeypatch.setattr(D, "__file__", str(tmp_path / "diagnostics.py"))
    (tmp_path / "metrics").mkdir()
    mpath = tmp_path / "metrics" / f"{o.obsid}.json"
    prior = {"stage": 5, "roll_tag": "v1", "roll_corrected": True, "passed": True}
    mpath.write_text(json.dumps({"stage5": prior}))
    monkeypatch.setattr(D, "registry", lambda programs=None: [o])
    monkeypatch.setattr(D, "_available_filters", lambda o: ["F212N", "F480M"])
    monkeypatch.setattr(D, "_filters_with_mosaic", lambda o: ["F212N", "F480M"])
    monkeypatch.setattr(D, "build_stage", lambda o, n, sw, lw: ("x.png", {"stage": 5, "passed": True}))
    monkeypatch.setattr(D, "_update_index", lambda o, repo: None)

    def _refuse(*a, **k):
        raise P.RollDowngradeError("refusing")
    monkeypatch.setattr(P, "post_stage", _refuse)
    assert D.main(["--program", o.program, "--obs", o.obs, "--stage", "5", "--post"]) == 0
    assert json.loads(mpath.read_text())["stage5"] == prior
