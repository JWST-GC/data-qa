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
    assert "### Human review" in body and "### Automatic status" in body
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
    """Decision 2026-07-22: cataloging defaults to the plain align crf; the image review row
    asks whether destreaking is needed."""
    body = mi.render_body(obs)
    assert "including whether destreaking is needed" in body


# ----------------------------- #364: status table vs human review
_FULL_METRICS = {"stage1": {"passed": True}, "stage2": {"passed": True, "n_stars": 1234},
                 "stage3": {"passed": True, "slope": 1.0, "scatter": 0.1, "sw": "F212N"},
                 "stage4": {"passed": True, "bulk_off": 5.0, "spatial_assessed": True,
                            "source": "release: x"},
                 "stage5": {"intermodule_off": 2.0}, "stage7": {"jicama_is_release": True}}


def _review_lines(body):
    return [ln for ln in body.splitlines() if "<!-- qa:" in ln]


def test_flags_fill_the_status_table_and_never_tick_a_review_box(obs, monkeypatch):
    monkeypatch.setattr(mi, "_qa_metrics", lambda o: _FULL_METRICS)
    body = mi.render_body(obs)
    status = mi._STATUS_RE.search(body).group(0)
    assert "- [" not in status                                  # no checkboxes in the table
    assert "✅ 1,234 sources" in status and "not a depth or completeness test" in status
    assert "✅ 2.0 mas, below the 15 mas flag threshold" in status
    assert "not inferred from stage 3" in status
    rows = _review_lines(body)
    assert len(rows) == len(mi._REVIEW_ROWS_NIRCAM)
    assert all(ln.startswith("- [ ] ") for ln in rows)
    assert "- [x]" not in body


def test_status_table_marks_unmeasured_items_pending(obs, monkeypatch):
    monkeypatch.setattr(mi, "_qa_metrics", lambda o: {"stage4": {"passed": False, "bulk_off": 90.0}})
    status = mi._STATUS_RE.search(mi.render_body(obs)).group(0)
    assert "⚠️ field offset 90.0 mas" in status
    assert "| Observation delivered; per-filter mosaics on disk | ⏳ not yet measured |" in status


def test_review_rows_link_the_stage_comments(obs):
    body = mi.render_body(obs, stage_urls={"1": "https://u/1", "7": "https://u/7"})
    images = next(ln for ln in body.splitlines() if "<!-- qa:images -->" in ln)
    assert "(stage [1](https://u/1), [7](https://u/7))" in images
    assert "(stage 2)" in next(ln for ln in body.splitlines() if "<!-- qa:population -->" in ln)


def test_stage_urls_parsed_from_index():
    from data_qa.post_diagnostics import render_index
    idx = render_index([{"body": "<!-- data-qa:diag:stage4 -->\n**Stage 4 — x.**", "html_url": "U4"},
                        {"body": "<!-- data-qa:diag:stagemiri -->", "html_url": "UM"}])
    assert mi._stage_urls("body\n" + idx) == {"4": "U4", "miri": "UM"}


def test_miri_issue_gets_its_own_review_rows():
    o = Observation(program="1939", obs="001", target="Sgr C", release_field="sgrc",
                    instrument="MIRI", filters=["F770W"])
    keys = [ln.split("<!-- qa:")[1][:-4] for ln in _review_lines(mi.render_body(o))]
    assert keys == ["images", "triage"]


def _tick(body, key):
    return "\n".join(ln.replace("- [ ] ", "- [x] ", 1) if f"<!-- qa:{key} -->" in ln else ln
                     for ln in body.splitlines())


def _stages(o):
    return {k: st for k, _, st in mi._review_rows(o)}


def _edits(*bodies_logins):
    return [{"diff": b, "editor": {"login": who}, "editedAt": at + "T00:00:00Z"}
            for b, who, at in bodies_logins]


def test_new_human_tick_is_stamped_with_the_reviewer_from_the_edit_history(obs):
    new = mi.render_body(obs)
    ticked = _tick(new, "images")
    hist = _edits((ticked, "alice", "2026-10-13"), (new, "bot", "2026-10-12"))
    out = mi._merge_review(new, ticked, {}, "2026-10-14", _stages(obs), edits=lambda: hist)
    line = next(ln for ln in out.splitlines() if "<!-- qa:images -->" in ln)
    assert line.startswith("- [x] ") and "— @alice, 2026-10-13" in line
    assert "<!-- qa-rev:images|alice|2026-10-13|- -->" in line
    # a second sync keeps the stamp and does not consult the history again
    out2 = mi._merge_review(new, out, {}, "2026-10-20", _stages(obs), edits=lambda: 1 / 0)
    assert line in out2


def test_unattributable_tick_says_so(obs):
    new = mi.render_body(obs)
    out = mi._merge_review(new, _tick(new, "triage"), {}, "2026-10-14", _stages(obs))
    assert "reviewer not recorded, 2026-10-14" in out


def test_changed_evidence_unticks_with_a_re_review_note(obs, tmp_path):
    f = tmp_path / "cat.fits"; f.write_text("a")
    M = {"stage2": {"inputs": [{"path": str(f)}]}}
    new = mi.render_body(obs)
    first = mi._merge_review(new, _tick(new, "population"), M, "2026-10-14", _stages(obs))
    assert "- [x] **Inspect the source population" in first
    # unchanged inputs: stays ticked; a sync without metrics: stays ticked
    assert first == mi._merge_review(new, first, M, "2026-10-15", _stages(obs))
    assert "- [x] **Inspect the source population" in mi._merge_review(new, first, {}, "x", _stages(obs))
    f.write_text("bb")                                          # the reviewed catalogue changed
    second = mi._merge_review(new, first, M, "2026-10-16", _stages(obs))
    line = next(ln for ln in second.splitlines() if "<!-- qa:population -->" in ln)
    assert line.startswith("- [ ] ") and "evidence changed since @unknown's review on 2026-10-14" in line
    # the note stays until someone re-ticks, then a fresh stamp replaces it
    assert line in mi._merge_review(new, second, M, "2026-10-17", _stages(obs))
    third = mi._merge_review(new, _tick(second, "population"), M, "2026-10-18", _stages(obs))
    line3 = next(ln for ln in third.splitlines() if "<!-- qa:population -->" in ln)
    assert line3.startswith("- [x] ") and "qa-stale" not in line3 and "2026-10-18" in line3


@pytest.mark.parametrize("mark", ["- N/A ", "- [ ] N/A ", "- n/a "])
def test_na_rows_are_kept_without_a_checkbox(obs, mark):
    new = mi.render_body(obs)
    old = "\n".join(ln.replace("- [ ] ", mark, 1) if "<!-- qa:jwst1pass -->" in ln else ln
                    for ln in new.splitlines())
    out = mi._merge_review(new, old, {}, "2026-10-14", _stages(obs))
    line = next(ln for ln in out.splitlines() if "<!-- qa:jwst1pass -->" in ln)
    assert line.startswith("- N/A — **Review JWST1PASS") and "[ ]" not in line


def test_legacy_ticks_are_listed_not_carried(obs):
    old = ("- [x] Observation delivered / retrieved\n- [x] Background / stripes / artifacts acceptable\n"
           "- [ ] **Purity**: minimal junk detections in PSF wings and in extended-emission regions")
    hist = _edits((old, "rainer", "2026-10-13"),
                  ("- [ ] Observation delivered / retrieved\n- [ ] Background / stripes / artifacts "
                   "acceptable", "keflavich", "2026-10-12"))
    block = mi._legacy_block(old, edits=lambda: hist, sync_login="keflavich")
    assert "Observation delivered / retrieved — ticked by @rainer on 2026-10-13" in block
    assert "Background / stripes / artifacts acceptable — ticked by @rainer on 2026-10-13" in block
    assert "Purity" not in block
    # a flag-driven row ticked by the sync account is not attributed to a person
    hist2 = _edits((old, "keflavich", "2026-10-13"), ("", "keflavich", "2026-10-01"))
    block2 = mi._legacy_block(old, edits=lambda: hist2, sync_login="keflavich")
    assert "retrieved — set from a diagnostic flag (or by hand from @keflavich)" in block2
    # sync account unknown (e.g. an Actions token): a flag row is never attributed
    assert "retrieved — set from a diagnostic flag" in mi._legacy_block(old, edits=lambda: hist)
    # carried unchanged once written; nothing to list -> no block
    assert mi._legacy_block("x\n" + block + "\n- [x] other") == block
    assert mi._legacy_block("- [ ] Observation delivered / retrieved") == ""


def test_sync_migrates_a_legacy_body_and_keeps_the_index(obs, monkeypatch):
    calls = []
    monkeypatch.setattr(mi, "_req", lambda method, url, token, data=None:
                        calls.append((method, url, data)) or (200, {"data": {}}))
    monkeypatch.setattr(mi, "_qa_metrics", lambda o: {})
    from data_qa.post_diagnostics import render_index
    idx = render_index([{"body": "<!-- data-qa:diag:stage2 -->", "html_url": "U2"}])
    old = mi.AUTOGEN_MARKER + "\n### QA checklist\n- [x] Catalog produced and vetted\n" + idx
    msg = mi.sync_observation(obs, "tok", "own/repo", {obs.issue_title: {"number": 7, "body": old}},
                              today="2026-10-14", sync_login="keflavich")
    assert msg.startswith("updated #7")
    body = calls[-1][2]["body"]
    assert calls[-1][0] == "PATCH" and calls[-1][1].endswith("/issues/7")
    assert all(ln.startswith("- [ ] ") for ln in _review_lines(body))
    assert "Catalog produced and vetted — set from a diagnostic flag" in body
    assert "(stage [2](U2))" in body
    assert "data-qa:index" in body


def test_sync_without_metrics_keeps_the_status_table_and_the_ticks(obs, monkeypatch):
    calls = []
    monkeypatch.setattr(mi, "_req", lambda method, url, token, data=None:
                        calls.append(data) or (200, {}))
    monkeypatch.setattr(mi, "_qa_metrics", lambda o: _FULL_METRICS)
    old = _tick(mi.render_body(obs), "frame")
    old = mi._merge_review(mi.render_body(obs), old, _FULL_METRICS, "2026-10-14", _stages(obs))
    monkeypatch.setattr(mi, "_qa_metrics", lambda o: {})
    mi.sync_observation(obs, "tok", "own/repo", {obs.issue_title: {"number": 7, "body": old}},
                        today="2026-10-15")
    body = calls[-1]["body"]
    assert mi._STATUS_RE.search(body).group(0) == mi._STATUS_RE.search(old).group(0)
    assert "- [x] **Check reference-frame alignment" in body and "2026-10-14" in body


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
