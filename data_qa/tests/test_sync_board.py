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
