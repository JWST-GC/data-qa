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
