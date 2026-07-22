"""Offline tests for data_qa.make_issues body rendering (no network: the
release-manifest fetch is stubbed out)."""
import pytest

from data_qa import make_issues as mi
from data_qa.observations import Observation


@pytest.fixture
def obs():
    return Observation(program="2221", obs="001", target="Brick",
                       release_field="brick", instrument="NIRCam",
                       filters=["F212N", "F405N"], visits=["001"],
                       epoch="2022-08-28")


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    monkeypatch.setattr(mi, "_fetch_lines", lambda url: [])


def test_render_body_has_checklist_and_marker(obs):
    body = mi.render_body(obs)
    assert body.startswith(mi.AUTOGEN_MARKER)
    assert "### QA checklist" in body
    assert "`F212N`" in body and "`F405N`" in body


def test_render_body_asks_destreak_decision(obs):
    """Decision 2026-07-22: cataloging defaults to the plain align crf; the QA
    checklist must ASK whether destreak is needed per observation."""
    body = mi.render_body(obs)
    assert ("- [ ] **Destreak**: assessed whether 1/f striping requires "
            "destreak (SW/LW per module); noted decision") in body
    assert "align" in body


# ----------------------------- sticky checkboxes x _github extraction (rebase)
def test_sticky_checkboxes_union_never_unchecks():
    new = "- [ ] Box A\n- [x] Box B\n- [ ] Box C"
    old = "- [x] Box A\n- [ ] Box B\nunrelated line"
    out = mi._sticky_checkboxes(new, old)
    assert "- [x] Box A" in out                  # human check carried over
    assert "- [x] Box B" in out                  # new render's check kept
    assert "- [ ] Box C" in out                  # neither -> unchecked


def test_metrics_drive_auto_checkboxes(obs, monkeypatch):
    """PR #17/#18 feature intact post-rebase: metrics file -> checked boxes."""
    monkeypatch.setattr(mi, "_qa_metrics",
                        lambda o: {"stage1": {"passed": True},
                                   "stage2": {"passed": True}})
    body = mi.render_body(obs)
    assert "- [x] Observation delivered / retrieved" in body
    assert "- [x] Catalog produced and vetted" in body
    assert mi._ck(True) == "x" and mi._ck(False) == " "


def test_sync_observation_preserves_human_checkbox_via_github_plumbing(
        obs, monkeypatch):
    """Regression for the rebase: the sticky-checkbox merge (PR #17/#18) must
    still run on issue UPDATE now that the API plumbing lives in
    data_qa._github (this branch's extraction)."""
    calls = []
    monkeypatch.setattr(mi, "_req",
                        lambda method, url, token, data=None:
                        calls.append((method, url, data)) or (200, {}))
    old_body = mi.render_body(obs).replace(
        "- [ ] Background / stripes / artifacts acceptable",
        "- [x] Background / stripes / artifacts acceptable")
    existing = {obs.issue_title: {"number": 7, "body": old_body}}
    msg = mi.sync_observation(obs, "tok", "own/repo", existing)
    assert msg.startswith("updated #7")
    (method, url, data), = calls
    assert method == "PATCH" and url.endswith("/issues/7")
    assert "- [x] Background / stripes / artifacts acceptable" in data["body"]
