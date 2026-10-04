"""Curated ``banner_nircam`` renders as a top-of-body WARNING admonition on NIRCam issues only."""
from data_qa import observations
from data_qa.make_issues import AUTOGEN_MARKER, render_body
from data_qa.observations import Observation


def _body(monkeypatch, instrument, banner="Line one.  \nLine two.\n\nParagraph two."):
    monkeypatch.setitem(observations.CURATED, "jw09999-o001", dict(banner_nircam=banner))
    return render_body(Observation(program="9999", obs="001", target="Test", instrument=instrument))


def test_banner_on_nircam(monkeypatch):
    b = _body(monkeypatch, "NIRCam")
    head = b.split("**Observation")[0]
    assert head.startswith(AUTOGEN_MARKER + "\n> [!WARNING]\n")
    # every banner line stays inside the blockquote, blank lines included
    assert "> Line one.  \n> Line two.\n>\n> Paragraph two.\n\n" in head


def test_no_banner_on_miri(monkeypatch):
    assert "[!WARNING]" not in _body(monkeypatch, "MIRI").split("**Observation")[0]


def test_no_banner_without_entry():
    b = render_body(Observation(program="9999", obs="002", target="Test", instrument="NIRCam"))
    assert "[!WARNING]" not in b.split("**Observation")[0]
