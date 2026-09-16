"""Unit tests for the project-board sync's pure classification logic.

`scripts/` is not on the CI import path (the workflow runs `pytest tests data_qa`), so load the
module by file path.  These exercise the metrics -> category mapping that decides each card's
`Measured` value -- the part that must never call a missing/partial/crashed result "clean".
"""
import importlib.util
import os

import pytest

_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "sync_project_board.py")
_spec = importlib.util.spec_from_file_location("sync_project_board", _PATH)
S = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(S)


def _m(stages):
    return {f"stage{n}": v for n, v in stages.items()}


def test_stage_status_glyphs_and_error():
    line, nrf, st = S._stage_status(_m({1: {"passed": True}, 2: {"red_flag": True},
                                        3: {"passed": False}, 4: {"error": "boom", "passed": False}}))
    assert st[1] == "ok" and st[2] == "RF" and st[3] == "fail" and st[4] == "err"
    assert nrf == 1
    for g in ("✅", "🚩", "⚠️", "🟥"):
        assert g in line


def test_classify_miri_regardless_of_metrics():
    assert S._classify("MIRI", _m({1: {"passed": True}}), {1: "ok"}) == "MIRI"


def test_classify_absent_and_corrupt_are_nometrics():
    assert S._classify("NIRCam", None, {}) == "nometrics"
    assert S._classify("NIRCam", "corrupt", {}) == "nometrics"


def test_classify_partial_is_incomplete_not_clean():
    # stages 1-3 present and passing, 4-6 never ran -> '?' -> incomplete, NOT clean
    _, _, st = S._stage_status(_m({1: {"passed": True}, 2: {"passed": True}, 3: {"passed": True}}))
    assert S._classify("NIRCam", {"stage1": {}}, st) == "incomplete"


def test_classify_crash_is_error_not_attention():
    d = {n: {"passed": True} for n in (1, 2, 3, 5, 6)}
    d[4] = {"error": "x", "passed": False}
    _, _, st = S._stage_status(_m(d))
    assert S._classify("NIRCam", {"stage1": {}}, st) == "error"


def test_classify_all_pass_is_clean():
    _, _, st = S._stage_status(_m({n: {"passed": True} for n in range(1, 7)}))
    assert S._classify("NIRCam", {"stage1": {}}, st) == "clean"


def test_classify_redflag_wins():
    _, _, st = S._stage_status(_m({1: {"red_flag": True}, 2: {"error": "x"}, 3: {"passed": False}}))
    assert S._classify("NIRCam", {"stage1": {}}, st) == "redflag"


def test_status_options_derived_from_cat_map():
    # the two must never drift -- _MEASURED_OPTIONS is built from _CAT_TO_OPTION.values()
    assert S._MEASURED_OPTIONS == list(S._CAT_TO_OPTION.values())
    assert set(S._CAT_TO_OPTION) >= {"redflag", "error", "incomplete", "clean", "nometrics"}


def test_review_status_maps_completeness():
    # incomplete / no-metrics -> still running; every stage having run -> ready for review
    assert S._review_status("incomplete") == "In progress"
    assert S._review_status("nometrics") == "In progress"
    for cat in ("clean", "attention", "redflag", "error"):
        assert S._review_status(cat) == "Ready for review"
    for cat in ("MIRI", "meta"):
        assert S._review_status(cat) is None


def test_gh_token_routing_by_subcommand(monkeypatch):
    monkeypatch.setenv("QA_REPO_TOKEN", "repo-tok")
    monkeypatch.setenv("QA_PROJECT_TOKEN", "proj-tok")
    monkeypatch.delenv("GH_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    assert S._token_for(("issue", "list")) == "repo-tok"
    assert S._token_for(("api", "repos/x")) == "repo-tok"
    assert S._token_for(("project", "list")) == "proj-tok"
    assert S._token_for(("api", "graphql")) == "proj-tok"


# --------------------------------------------------------------- SAFETY properties (docstring)
import json as _json

_WRITES = ("item-edit", "item-add", "item-archive", "field-create", "field-delete")


def _run(monkeypatch, tmp_path, argv):
    """Drive main() with _gh stubbed (canned project/fields/items/issues, all writes recorded)
    and _METRICS_DIR pointed at tmp_path.  Returns (rc, list-of-gh-arg-tuples)."""
    calls = []
    fields = {
        "Measured": {"name": "Measured", "id": "F_meas",
                     "options": [{"name": o, "id": f"m{i}"} for i, o in enumerate(S._MEASURED_OPTIONS)]},
        "Workflow": {"name": "Workflow", "id": "F_work",
                     "options": [{"name": o, "id": f"w{i}"} for i, o in enumerate(S._WORKFLOW_OPTIONS)]},
        "Review status": {"name": "Review status", "id": "F_rev",
                          "options": [{"name": o, "id": f"r{i}"} for i, o in enumerate(S._REVIEW_OPTIONS)]},
        "Stages 1-6": {"name": "Stages 1-6", "id": "F_st"},
        "Red flags": {"name": "Red flags", "id": "F_rf"},
        "Offset (mas)": {"name": "Offset (mas)", "id": "F_off"},
    }

    def fake_gh(*args, check=True):
        calls.append(args)
        a = list(args)
        if a[:2] == ["project", "list"]:
            return _json.dumps({"projects": [{"title": "Data-QA field status",
                                              "id": "P", "number": 2, "url": "https://x"}]}), 0
        if a[:2] == ["project", "field-list"]:
            return _json.dumps({"fields": list(fields.values())}), 0
        if a[:2] == ["issue", "list"]:
            return _json.dumps([{"number": 98,
                                 "title": "GC Treasury — jw10678-o098 (NIRCam)",
                                 "url": "https://i/98"}]), 0
        if a[:2] == ["project", "item-list"]:
            return _json.dumps({"items": []}), 0
        if a[:2] == ["project", "item-add"]:
            return _json.dumps({"id": "ITEM1"}), 0
        return "", 0

    monkeypatch.setattr(S, "_gh", fake_gh)
    monkeypatch.setattr(S, "_METRICS_DIR", str(tmp_path))
    monkeypatch.setenv("GH_TOKEN", "x")
    rc = S.main(argv)
    return rc, calls


def test_dry_run_writes_nothing(monkeypatch, tmp_path):
    rc, calls = _run(monkeypatch, tmp_path, [])
    assert not [c for c in calls if any(w in c for w in _WRITES)], calls


def test_apply_never_writes_the_human_workflow_field(monkeypatch, tmp_path):
    (tmp_path / "jw10678-o098.json").write_text(_json.dumps({"stage1": {"passed": True}}))
    rc, calls = _run(monkeypatch, tmp_path, ["--apply"])
    edits = [c for c in calls if "item-edit" in c]
    assert edits, "nothing was written at all; the test is not exercising --apply"
    assert not [c for c in edits if "F_work" in c]


def test_apply_refuses_when_metrics_are_missing(monkeypatch, tmp_path):
    with pytest.raises(SystemExit):
        _run(monkeypatch, tmp_path, ["--apply", "--max-missing", "0"])


def _stub_issue_list(monkeypatch, tmp_path, issues):
    monkeypatch.setattr(S, "_METRICS_DIR", str(tmp_path))
    monkeypatch.setattr(S, "_gh", lambda *a, **k: (_json.dumps(issues), 0))


def test_rows_skips_non_observation_issues_by_default(monkeypatch, tmp_path):
    _stub_issue_list(monkeypatch, tmp_path, [
        {"number": 98, "title": "GC Treasury — jw10678-o098 (NIRCam)", "url": "u98"},
        {"number": 161, "title": "A delivered tile gets no per-observation QA issue", "url": "u161"},
    ])
    assert {r["num"] for r in S._rows("repo")} == {98}          # dev/tracking issue is off the board


def test_rows_include_meta_opts_the_escape_hatch_back_in(monkeypatch, tmp_path):
    _stub_issue_list(monkeypatch, tmp_path, [
        {"number": 98, "title": "GC Treasury — jw10678-o098 (NIRCam)", "url": "u98"},
        {"number": 161, "title": "A delivered tile gets no per-observation QA issue", "url": "u161"},
    ])
    rows = S._rows("repo", include_meta=True)
    assert {r["num"] for r in rows} == {98, 161}
    assert any(r["cat"] == "meta" for r in rows)               # the non-obs issue is a meta card
