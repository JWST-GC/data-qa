"""Offline unit tests for data_qa.pipeline_trigger + pipeline_policy (command
generation only -- no sbatch and no real policy-probe subprocess is ever run)."""
import json
import subprocess

import pytest

from data_qa import pipeline_policy as pp
from data_qa import pipeline_trigger as pt

FILTERS = ["F405N", "F410M", "F466N", "F212N"]

TREASURY_POLICY = {"each_suffix": "destreak_o001_crf",
                   "destreaks": {"F212N": True, "F480M": True}}


@pytest.fixture(autouse=True)
def _fresh_policy_cache():
    """The probe caches per (field, obs) within a run; tests need isolation."""
    pp.clear_cache()
    yield
    pp.clear_cache()


def _stub_run(stdout="", rc=0, stderr="", calls=None):
    """A subprocess.run stand-in returning a canned CompletedProcess."""
    def run(argv, capture_output=True, text=True, timeout=None):
        if calls is not None:
            calls.append(argv)
        return subprocess.CompletedProcess(argv, rc, stdout=stdout,
                                           stderr=stderr)
    return run


def _probe_root(tmp_path):
    """A pipe_root that passes the probe's package-presence guard."""
    (tmp_path / "jwst_gc_pipeline").mkdir(exist_ok=True)
    return str(tmp_path)


def test_reduction_golden_command():
    step = pt.reduction_step(2221, "001", "brick", FILTERS, pipe_root="/pipe")
    assert pt.shell_line(step) == (
        "MODULES=nrca,nrcb,merged "
        "sbatch --parsable --job-name=brick2221-o001-reduce --array=0-3 "
        "'--export=ALL,PROPOSAL=2221,FIELD=001,SKIP=0,"
        "FILTERS=F405N F410M F466N F212N' "
        "/pipe/scripts/reduction/submit_reduction.sbatch")


def test_reduction_skip_step12():
    step = pt.reduction_step(2221, "001", "brick", ["F405N"], pipe_root="/pipe",
                             skip_step12=True)
    assert "SKIP=1" in " ".join(step["argv"])
    assert "--array=0-0" in step["argv"]


def test_modules_never_in_export():
    """Comma-valued MODULES must ride the environment, not the --export list
    (the SLURM --export comma trap)."""
    step = pt.reduction_step(2221, "001", "brick", FILTERS, pipe_root="/pipe")
    export_arg = next(a for a in step["argv"] if a.startswith("--export="))
    assert "MODULES" not in export_arg
    assert step["env"]["MODULES"] == "nrca,nrcb,merged"


def test_cataloging_golden_command():
    """Fallback EACH_SUFFIX (no probe result) is the plain no-destreak crf form
    (align_o<obs>_crf): fix_alignment always runs, and the no-destreak reduction
    path names the per-exposure crfs *_align_o<field>_crf.fits.  brick is a GC
    field, so DEBLEND_SATSTARS=1 rides the env too."""
    step = pt.cataloging_step(2221, "001", "brick", FILTERS, pipe_root="/pipe")
    assert pt.shell_line(step) == (
        "DEBLEND_SATSTARS=1 DEP='<REDUCTION_JOBID>' EACH_SUFFIX=align_o001_crf "
        "FIELD=001 FILTERS='F405N F410M F466N F212N' MODULES=merged "
        "PROPOSAL=2221 "
        "TARGET=brick /pipe/scripts/reduction/submit_cataloging_chain.sh")


def test_cataloging_destreak_optin():
    """--destreak selects the destreaked products' suffix."""
    step = pt.cataloging_step(2221, "001", "brick", FILTERS, pipe_root="/pipe",
                              destreak=True)
    assert step["env"]["EACH_SUFFIX"] == "destreak_o001_crf"


def test_cataloging_each_suffix_override_wins():
    step = pt.cataloging_step(2221, "001", "brick", FILTERS, pipe_root="/pipe",
                              each_suffix="custom_o001_crf", destreak=True)
    assert step["env"]["EACH_SUFFIX"] == "custom_o001_crf"


def test_cataloging_guard_vars_all_present():
    """submit_cataloging.sbatch hard-fails unless these travel together."""
    step = pt.cataloging_step(1182, "004", "brick", ["F200W"], pipe_root="/pipe")
    for var in ("PROPOSAL", "FIELD", "TARGET", "EACH_SUFFIX", "MODULES"):
        assert var in step["env"], var


def test_build_plan_field_from_programs_map():
    # /pipe has no jwst_gc_pipeline package, so the policy probe declines
    # (no subprocess) and EACH_SUFFIX is the historical align fallback
    plan = pt.build_plan(4147, "012", filters=["F405N"], pipe_root="/pipe")
    assert plan[0]["argv"][2] == "--job-name=sgrc4147-o012-reduce"
    assert plan[1]["env"]["TARGET"] == "sgrc"
    assert plan[1]["env"]["EACH_SUFFIX"] == "align_o012_crf"


def test_build_plan_destreak_flag_threads_through():
    plan = pt.build_plan(4147, "012", filters=["F405N"], pipe_root="/pipe",
                         destreak=True)
    assert plan[1]["env"]["EACH_SUFFIX"] == "destreak_o012_crf"


def test_build_plan_requires_field_mapping():
    with pytest.raises(ValueError, match="no field mapping"):
        pt.build_plan(9999, "001", filters=["F405N"])


def test_build_plan_requires_filters():
    with pytest.raises(ValueError, match="filters required"):
        pt.build_plan(2221, "001")


def test_missing_scripts_refuses_execute(tmp_path):
    """--execute against a pipe-root without the pipeline scripts must refuse
    without ever invoking sbatch."""
    rc = pt.main(["--program", "2221", "--obs", "001", "--filters", "F405N",
                  "--pipe-root", str(tmp_path), "--execute"])
    assert rc == 1


def test_dry_run_prints_plan(tmp_path, capsys):
    rc = pt.main(["--program", "2221", "--obs", "001", "--filters", "F405N",
                  "F410M", "--pipe-root", str(tmp_path)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "dry-run" in out
    assert "brick2221-o001-reduce" in out
    assert "submit_cataloging_chain.sh" in out


# ------------------------------------------------------------ policy probe (#69)
def test_probe_parses_destreak_policy(tmp_path, monkeypatch):
    """A destreak-policy probe result (gc-treasury destreaks) parses through."""
    calls = []
    monkeypatch.setattr(pp.subprocess, "run",
                        _stub_run(json.dumps(TREASURY_POLICY), calls=calls))
    got = pp.probe_policy("gc-treasury", "001", ["F212N", "F480M"],
                          pipe_root=_probe_root(tmp_path))
    assert got == TREASURY_POLICY
    # the probe runs the pipeline env python with pipe_root + field/obs/filters
    (argv,) = calls
    assert argv[0] == pp.pipeline_python()
    assert argv[1] == "-c"
    assert argv[2] == pp.PROBE_CODE
    assert argv[3:] == [str(tmp_path), "gc-treasury", "001", "F212N", "F480M"]


def test_probe_parses_align_policy(tmp_path, monkeypatch):
    """An align-policy probe result (extended-emission field) parses through."""
    payload = {"each_suffix": "align_o002_crf",
               "destreaks": {"F150W": False, "F480M": False}}
    monkeypatch.setattr(pp.subprocess, "run", _stub_run(json.dumps(payload)))
    got = pp.probe_policy("w51", "002", ["F150W", "F480M"],
                          pipe_root=_probe_root(tmp_path))
    assert got == payload


def test_probe_failure_warns_and_returns_none(tmp_path, monkeypatch, capsys):
    """rc!=0 -> None + a loud warning naming the m1 zero-inputs mismatch risk."""
    monkeypatch.setattr(pp.subprocess, "run",
                        _stub_run(rc=1, stderr="ModuleNotFoundError: boom"))
    got = pp.probe_policy("gc-treasury", "001", ["F212N"],
                          pipe_root=_probe_root(tmp_path))
    assert got is None
    err = capsys.readouterr().err
    assert "WARNING" in err
    assert "align_o001_crf" in err
    assert "data-qa#69" in err


def test_probe_garbage_output_warns_and_returns_none(tmp_path, monkeypatch,
                                                     capsys):
    monkeypatch.setattr(pp.subprocess, "run", _stub_run("not json"))
    got = pp.probe_policy("gc-treasury", "001", ["F212N"],
                          pipe_root=_probe_root(tmp_path))
    assert got is None
    assert "WARNING" in capsys.readouterr().err


def test_probe_refuses_bogus_pipe_root(tmp_path, monkeypatch, capsys):
    """A pipe_root without the package must decline WITHOUT subprocessing --
    the pipeline env has an installed copy that would otherwise answer for it."""
    def boom(*a, **k):
        raise AssertionError("no subprocess may run for a bogus pipe_root")
    monkeypatch.setattr(pp.subprocess, "run", boom)
    assert pp.probe_policy("brick", "001", ["F212N"],
                           pipe_root=str(tmp_path)) is None
    assert "WARNING" in capsys.readouterr().err


def test_probe_cached_per_field_obs(tmp_path, monkeypatch):
    """One subprocess per (field, obs) per run, even for repeated build_plans."""
    calls = []
    monkeypatch.setattr(pp.subprocess, "run",
                        _stub_run(json.dumps(TREASURY_POLICY), calls=calls))
    root = _probe_root(tmp_path)
    for _ in range(3):
        pp.probe_policy("gc-treasury", "001", ["F212N", "F480M"], pipe_root=root)
    assert len(calls) == 1


# ------------------------------------------- probed env defaults in the plan (#69)
def test_build_plan_probed_each_suffix(tmp_path, monkeypatch):
    """The auto-trigger path (no overrides) catalogs what the reduction writes:
    EACH_SUFFIX comes from the probed destreak policy, so gc-treasury globs
    destreak_o001_crf while its reduction destreaks."""
    monkeypatch.setattr(pp.subprocess, "run",
                        _stub_run(json.dumps(TREASURY_POLICY)))
    plan = pt.build_plan(10678, "001", filters=["F212N", "F480M"],
                         pipe_root=_probe_root(tmp_path))
    assert plan[1]["env"]["TARGET"] == "gc-treasury"
    assert plan[1]["env"]["EACH_SUFFIX"] == "destreak_o001_crf"
    # the reduction is left on the driver's own destreak_policy default
    assert "NO_DESTREAK" not in plan[0]["env"]


def test_build_plan_probe_failure_keeps_today_default(tmp_path, monkeypatch,
                                                      capsys):
    """Probe failure degrades to today's behavior (align fallback + warning),
    never crashes the trigger."""
    monkeypatch.setattr(pp.subprocess, "run", _stub_run(rc=1))
    plan = pt.build_plan(10678, "001", filters=["F212N", "F480M"],
                         pipe_root=_probe_root(tmp_path))
    assert plan[1]["env"]["EACH_SUFFIX"] == "align_o001_crf"
    assert "NO_DESTREAK" not in plan[0]["env"]
    assert "WARNING" in capsys.readouterr().err


def test_build_plan_explicit_override_skips_probe(monkeypatch):
    """An operator destreak choice wins outright; no subprocess runs."""
    def boom(*a, **k):
        raise AssertionError("probe must not run under an explicit override")
    monkeypatch.setattr(pp, "probe_policy", boom)
    plan = pt.build_plan(4147, "012", filters=["F405N"], pipe_root="/pipe",
                         destreak=True)
    assert plan[1]["env"]["EACH_SUFFIX"] == "destreak_o012_crf"


def test_build_plan_no_destreak_consistent_pair():
    """destreak=False reaches BOTH steps: align cataloging + NO_DESTREAK=1 on
    the reduction, so the pair cannot split once the sbatch passthrough lands."""
    plan = pt.build_plan(2221, "001", filters=["F405N"], pipe_root="/pipe",
                         destreak=False)
    assert plan[0]["env"]["NO_DESTREAK"] == "1"
    assert plan[1]["env"]["EACH_SUFFIX"] == "align_o001_crf"


def test_build_plan_mixed_destreak_policy_warns(tmp_path, monkeypatch, capsys):
    """A per-filter split (sickle: SW destreaks, LW does not) cannot ride one
    EACH_SUFFIX -- the plan says so instead of silently mis-globbing."""
    payload = {"each_suffix": "destreak_o002_crf",
               "destreaks": {"F212N": True, "F480M": False}}
    monkeypatch.setattr(pp.subprocess, "run", _stub_run(json.dumps(payload)))
    plan = pt.build_plan(3958, "002", filters=["F212N", "F480M"],
                         pipe_root=_probe_root(tmp_path))
    assert plan[1]["env"]["EACH_SUFFIX"] == "destreak_o002_crf"
    assert "wrong for part of the filter" in capsys.readouterr().err


# --------------------------------------------------- DEBLEND_SATSTARS + SKYMATCH
def test_deblend_satstars_set_for_gc_fields():
    """The pipeline README marks DEBLEND_SATSTARS=1 required for crowded GC
    fields; every treasury tile is inner-CMZ."""
    for field, program, obsnum in (("gc-treasury", 10678, "001"),
                                   ("gc2211", 2211, "023"),
                                   ("arches", 2045, "001")):
        step = pt.cataloging_step(program, obsnum, field, ["F212N"],
                                  pipe_root="/pipe")
        assert step["env"]["DEBLEND_SATSTARS"] == "1", field


def test_deblend_satstars_absent_for_non_gc_fields():
    for field, program, obsnum in (("wd1", 1905, "001"), ("w51", 6151, "001"),
                                   ("ngc6334", 6778, "001")):
        step = pt.cataloging_step(program, obsnum, field, ["F150W"],
                                  pipe_root="/pipe")
        assert "DEBLEND_SATSTARS" not in step["env"], field


def test_skymatch_passthrough_when_operator_sets_it(monkeypatch):
    """SKYMATCH rides through to submit_reduction.sbatch only when set; the
    trigger invents no default (jwst-gc-pipeline#419 owns that decision)."""
    monkeypatch.setenv("SKYMATCH", "match")
    step = pt.reduction_step(10678, "001", "gc-treasury", ["F212N"],
                             pipe_root="/pipe")
    assert step["env"]["SKYMATCH"] == "match"


def test_skymatch_absent_by_default(monkeypatch):
    monkeypatch.delenv("SKYMATCH", raising=False)
    step = pt.reduction_step(10678, "001", "gc-treasury", ["F212N"],
                             pipe_root="/pipe")
    assert "SKYMATCH" not in step["env"]


def test_reduction_no_destreak_only_when_explicitly_off():
    step = pt.reduction_step(2221, "001", "brick", FILTERS, pipe_root="/pipe",
                             destreak=False)
    assert step["env"]["NO_DESTREAK"] == "1"
    for destreak in (None, True):
        step = pt.reduction_step(2221, "001", "brick", FILTERS,
                                 pipe_root="/pipe", destreak=destreak)
        assert "NO_DESTREAK" not in step["env"]


# ------------------------------------------------------- end-to-end dry-run lines
def test_dry_run_env_lines_carry_policy(tmp_path, monkeypatch, capsys):
    """The printed sbatch env lines carry the probed EACH_SUFFIX and the GC
    DEBLEND_SATSTARS -- what a treasury auto-trigger would really submit."""
    monkeypatch.setattr(pp.subprocess, "run",
                        _stub_run(json.dumps(TREASURY_POLICY)))
    monkeypatch.delenv("SKYMATCH", raising=False)
    rc = pt.main(["--program", "10678", "--obs", "001", "--filters", "F212N",
                  "F480M", "--pipe-root", _probe_root(tmp_path)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "EACH_SUFFIX=destreak_o001_crf" in out
    assert "DEBLEND_SATSTARS=1" in out
    assert "SKYMATCH" not in out
    assert "--job-name=gc-treasury10678-o001-reduce" in out


def test_dry_run_no_probe_flag(tmp_path, monkeypatch, capsys):
    """--no-probe keeps the hardcoded default and never subprocesses."""
    def boom(*a, **k):
        raise AssertionError("probe must not run under --no-probe")
    monkeypatch.setattr(pp, "probe_policy", boom)
    rc = pt.main(["--program", "10678", "--obs", "001", "--filters", "F212N",
                  "--pipe-root", str(tmp_path), "--no-probe"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "EACH_SUFFIX=align_o001_crf" in out
