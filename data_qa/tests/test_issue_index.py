import pytest

from data_qa import post_diagnostics as PD
from data_qa.make_issues import _carry_index
from data_qa.post_diagnostics import INDEX_END, INDEX_START, render_index, splice_index


def _c(i, body):
    return {"html_url": f"https://github.com/o/r/issues/1#issuecomment-{i}", "body": body}


COMMENTS = [
    _c(1, "<!-- data-qa:monitor -->\n**MAST monitor events**"),
    _c(2, "<!-- data-qa:pipeline-status -->\n### Pipeline progress"),
    _c(3, "<!-- data-qa:diag:stage11 -->\n### QA diagnostic — stage 11\n\n**Stage 11 — effective PSF.**"),
    _c(4, "a human comment mentioning stage 2"),
    _c(5, "<!-- data-qa:diag:stage2 -->\n### QA diagnostic — stage 2\n\n**Stage 2 — CMD.**"),
    _c(6, "<!-- data-qa:diag:stage6clean -->\n### QA diagnostic — stage 6clean"),
    _c(7, "<!-- data-qa:diag:stage6 -->\n### QA diagnostic — stage 6\n\n**Stage 6 — astrometric error.**"),
]


def test_index_is_in_stage_order_and_skips_human_comments():
    block = render_index(COMMENTS)
    assert block.startswith(INDEX_START) and block.endswith(INDEX_END)
    order = [ln.split("#issuecomment-")[1].split(")")[0] for ln in block.splitlines()
             if ln.startswith("- [")]
    assert order == ["2", "1", "5", "7", "6", "3"]
    assert "- [Stage 2](https://github.com/o/r/issues/1#issuecomment-5) — CMD" in block
    assert "Stage 11" in block and "effective PSF" in block


def test_splice_appends_then_replaces_in_place():
    body = "intro\n\n---\nfooter\n"
    once = splice_index(body, render_index(COMMENTS[:3]))
    assert once.startswith(body.rstrip("\n")) and once.rstrip().endswith(INDEX_END)
    twice = splice_index(once, render_index(COMMENTS))
    assert twice.count(INDEX_START) == 1 and "Stage 2" in twice
    assert splice_index(twice, render_index(COMMENTS)) == twice


def test_make_issues_carries_index_across_rerender():
    old = splice_index("old body", render_index(COMMENTS))
    new = _carry_index("fresh body\n", old)
    assert new.startswith("fresh body") and new.count(INDEX_START) == 1
    assert _carry_index("fresh body\n", "no index here") == "fresh body\n"


class _O:
    issue_title = "T"


def _fake_api(monkeypatch, body, patch_status=200):
    calls = []
    monkeypatch.setattr(PD, "_issue_number", lambda repo, token, title: 7)
    monkeypatch.setattr(PD, "_paged_get", lambda url, token, what: COMMENTS)

    def req(method, url, token, data=None, **kw):
        calls.append(method)
        if method == "GET":
            return 200, {"body": body}
        return patch_status, {}
    monkeypatch.setattr(PD, "_req", req)
    return calls


def test_update_index_patches_only_on_change(monkeypatch):
    calls = _fake_api(monkeypatch, "body")
    assert PD.update_index(_O(), "o/r", token="t") is True and calls == ["GET", "PATCH"]
    calls = _fake_api(monkeypatch, splice_index("body", render_index(COMMENTS)))
    assert PD.update_index(_O(), "o/r", token="t") is False and calls == ["GET"]


def test_update_index_raises_on_failed_patch(monkeypatch):
    _fake_api(monkeypatch, "body", patch_status=500)
    with pytest.raises(PD.PostError):
        PD.update_index(_O(), "o/r", token="t")
