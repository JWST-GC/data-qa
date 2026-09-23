"""Tests for data_qa.make_issues body rendering.  render_body reads no network
(the registry is built from MAST upstream; the body itself only formats)."""
import pytest

from data_qa import make_issues as mi
from data_qa.observations import Observation


@pytest.fixture
def obs():
    return Observation(program="2221", obs="001", target="Brick",
                       release_field="brick", instrument="NIRCam",
                       filters=["F212N", "F405N"], visits=["001"],
                       epoch="2022-08-28")


def test_render_body_has_checklist_and_marker(obs):
    body = mi.render_body(obs)
    assert body.startswith(mi.AUTOGEN_MARKER)
    assert "### QA checklist" in body
    assert "`F212N`" in body and "`F405N`" in body


def test_render_body_has_no_web_release_references(obs):
    """The starformation web release is the last, post-QA step: the release page and its
    direct downloads must never appear in the issue body.  This guard concerns only those; it
    does not require the Aladin viewer link (that is asserted separately), and any starformation
    reference that IS present must be the viewer, which is archive tooling rather than the
    release page."""
    import re
    body = mi.render_body(obs)
    assert "Release page" not in body and "Direct downloads" not in body
    assert "MAST data search" in body            # archive link still present
    hosts = re.findall(r"starformation\.astro\.ufl\.edu\S*", body)
    assert all("avm_images/jwst_gc_aladin.html" in h for h in hosts)


def test_render_body_links_aladin_viewer(obs):
    """The overview links the JWST-GC Aladin viewer.  It is a plain link: the viewer centres
    from its own presets and does not read URL coordinates, so the body claims no per-field
    centring."""
    body = mi.render_body(obs)
    assert mi._ALADIN_PAGE in body
    assert "centred on this field" not in body
    assert "?ra=" not in body


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


@pytest.mark.parametrize("new_label,prior", [(k, p) for k, ps in mi._CK_RENAMED.items() for p in ps])
def test_sticky_checkboxes_carry_a_tick_across_a_reworded_label(new_label, prior):
    # The merge keys on label TEXT, so rewording a checklist item drops every tick a human put on
    # it.  _CK_RENAMED maps a label to every spelling it has had; a tick on ANY of them carries.
    out = mi._sticky_checkboxes(f"- [ ] {new_label}", f"- [x] {prior}")
    assert out == f"- [x] {new_label}"


def test_sticky_checkboxes_carry_a_tick_across_two_renames(monkeypatch):
    # A second rewording must append to the tuple rather than replace it.  With only the
    # immediately-previous spelling recorded, a tick placed before the FIRST rename drops on the
    # second -- and both keys would still be live labels, so the staleness test below cannot see
    # it.  Simulate the two-rename case rather than wait for it to happen.
    monkeypatch.setattr(mi, "_CK_RENAMED", {"third wording": ("second wording", "first wording")})
    assert mi._sticky_checkboxes("- [ ] third wording", "- [x] first wording") == "- [x] third wording"
    assert mi._sticky_checkboxes("- [ ] third wording", "- [x] second wording") == "- [x] third wording"
    # an unrelated label is still left alone
    assert mi._sticky_checkboxes("- [ ] third wording", "- [x] unrelated") == "- [ ] third wording"


def test_sticky_checkboxes_renamed_labels_are_labels_the_body_emits(obs):
    # Every key must be a label render_body actually writes; otherwise the entry is dead and the
    # tick it exists to carry is lost anyway.
    body = mi.render_body(obs)
    for new_label in mi._CK_RENAMED:
        assert new_label in body


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


def test_globus_block_lists_i2d_and_catalog(tmp_path, monkeypatch, obs):
    """The Globus section links one i2d + one catalogue per filter and a scriptable URL
    list, preferring the plain merged science i2d over the _data_i2d resample and skipping
    per-detector and model/residual products."""
    monkeypatch.setattr(mi, "_GLOBUS_ROOT", str(tmp_path))
    pdir = tmp_path / "brick" / "F212N" / "pipeline"
    pdir.mkdir(parents=True)
    for name in (
        "jw02221-o001_t001_nircam_clear-f212n-merged_i2d.fits",           # preferred
        "jw02221-o001_t001_nircam_clear-f212n-merged_data_i2d.fits",      # resample, not preferred
        "jw02221-o001_t001_nircam_clear-f212n-nrca_i2d.fits",            # per-detector, excluded
        "jw02221-o001_t001_nircam_clear-f212n-merged_cat.ecsv",
        "jw02221-o001_t001_nircam_clear-f212n-merged_m2_daophot_basic_mergedcat_model_i2d.fits",
    ):
        (pdir / name).write_text("")
    block = mi._globus_block(obs)
    assert "### Data files (Globus)" in block
    assert "clear-f212n-merged_i2d.fits" in block
    assert "_data_i2d" not in block and "-nrca_" not in block and "_model_" not in block
    assert "clear-f212n-merged_cat.ecsv" in block
    assert f"{mi._GLOBUS_HTTPS_BASE}/brick/F212N/pipeline/" in block
    # command-line download recipe: bearer-token wget + the scriptable URL list
    assert "Authorization: Bearer" in block and "globus-sdk" in block
    assert mi._GLOBUS_COLLECTION_ID in block and "urls.txt" in block


def test_globus_block_empty_when_unreduced(tmp_path, monkeypatch, obs):
    """With no pipeline products on disk the section states that it fills in after reduction."""
    monkeypatch.setattr(mi, "_GLOBUS_ROOT", str(tmp_path))
    (tmp_path / "brick").mkdir()
    assert "fills in once the observation is reduced" in mi._globus_block(obs)
