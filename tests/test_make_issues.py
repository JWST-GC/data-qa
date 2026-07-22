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
