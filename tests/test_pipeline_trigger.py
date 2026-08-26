"""Offline unit tests for data_qa.pipeline_trigger + pipeline_policy (command
generation only -- no sbatch is ever run).  The policy probe is stubbed
throughout; the registry preflight is stubbed except in its own tests, which
drive it with fake interpreters plus one real-interpreter case against a
tmp_path checkout."""
import json
import os
import subprocess

import pytest

from data_qa import mast_monitor as mm
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


# the un-stubbed preflight, for its own tests below (the autouse fixture
# replaces the module attribute before each test runs)
_REAL_PREFLIGHT = pt.registry_preflight


@pytest.fixture(autouse=True)
def preflight_calls(monkeypatch):
    """Keep the command-generation tests offline: the registry preflight
    (a subprocess of the pipeline env python) is stubbed to a recorder.
    Preflight tests call _REAL_PREFLIGHT with a fake interpreter instead."""
    calls = []
    monkeypatch.setattr(
        pt, "registry_preflight",
        lambda program, obs, **kw: calls.append((program, obs, kw)))
    return calls


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


def test_cataloging_each_suffix_ladder_precedence():
    """The four EACH_SUFFIX levels rank explicit suffix > explicit destreak >
    probed policy > historical fallback, asserted on cataloging_step directly so
    the contract holds for callers build_plan's gating cannot reach."""
    policy = {"each_suffix": "destreak_o001_crf",
              "destreaks": {"F212N": True}}
    common = dict(pipe_root="/pipe")

    def suffix(**kw):
        return pt.cataloging_step(2221, "001", "brick", ["F212N"],
                                  **common, **kw)["env"]["EACH_SUFFIX"]

    # 1 explicit each_suffix outranks everything
    assert suffix(each_suffix="custom_o001_crf", destreak=False,
                  policy=policy) == "custom_o001_crf"
    # 2 an explicit destreak choice outranks the probed policy
    assert suffix(destreak=False, policy=policy) == "align_o001_crf"
    assert suffix(destreak=True, policy={"each_suffix": "align_o001_crf",
                                         "destreaks": {"F212N": False}}
                  ) == "destreak_o001_crf"
    # 3 the probed policy outranks the historical fallback
    assert suffix(policy=policy) == "destreak_o001_crf"
    # 4 the fallback with nothing supplied
    assert suffix() == "align_o001_crf"


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
    """rc!=0 -> None + a loud warning naming the m1 zero-inputs mismatch risk.

    The warning carries the return code and the stderr tail, which is the whole
    operator-facing diagnostic for the likeliest real failure (a pipeline env
    that cannot import the package)."""
    monkeypatch.setattr(pp.subprocess, "run",
                        _stub_run(rc=1, stderr="Traceback...\n"
                                               "ModuleNotFoundError: boom"))
    got = pp.probe_policy("gc-treasury", "001", ["F212N"],
                          pipe_root=_probe_root(tmp_path))
    assert got is None
    err = capsys.readouterr().err
    assert "WARNING" in err
    assert "rc=1" in err
    assert "ModuleNotFoundError: boom" in err
    assert "align_o001_crf" in err
    assert "data-qa#69" in err


def test_probe_nonzero_rc_rejects_even_a_valid_payload(tmp_path, monkeypatch,
                                                       capsys):
    """A probe that FAILED is not trusted for what it printed: rc!=0 with a
    well-formed payload on stdout (a partial run, a wrapper that echoes a
    cached answer then exits nonzero) still degrades to the fallback."""
    monkeypatch.setattr(pp.subprocess, "run",
                        _stub_run(json.dumps(TREASURY_POLICY), rc=2,
                                  stderr="probe wrapper failed"))
    got = pp.probe_policy("gc-treasury", "001", ["F212N"],
                          pipe_root=_probe_root(tmp_path))
    assert got is None
    err = capsys.readouterr().err
    assert "rc=2" in err
    assert "probe wrapper failed" in err


def test_probe_garbage_output_warns_and_returns_none(tmp_path, monkeypatch,
                                                     capsys):
    monkeypatch.setattr(pp.subprocess, "run", _stub_run("not json"))
    got = pp.probe_policy("gc-treasury", "001", ["F212N"],
                          pipe_root=_probe_root(tmp_path))
    assert got is None
    assert "WARNING" in capsys.readouterr().err


def test_probe_timeout_warns_and_returns_none(tmp_path, monkeypatch, capsys):
    """A wedged env (hung filesystem/python) must not hang or crash the poll:
    TimeoutExpired -> None + the fallback warning."""
    def hang(argv, capture_output=True, text=True, timeout=None):
        raise subprocess.TimeoutExpired(argv, timeout)
    monkeypatch.setattr(pp.subprocess, "run", hang)
    got = pp.probe_policy("gc-treasury", "001", ["F212N"],
                          pipe_root=_probe_root(tmp_path), timeout=7)
    assert got is None
    err = capsys.readouterr().err
    assert "timed out after 7s" in err
    assert "align_o001_crf" in err


def test_probe_unrunnable_python_warns_and_returns_none(tmp_path, monkeypatch,
                                                        capsys):
    """A missing/non-executable $PIPELINE_PYTHON raises OSError out of
    subprocess.run -- caught, warned, fallback, so no traceback escapes into
    the trigger loop."""
    def missing(argv, capture_output=True, text=True, timeout=None):
        raise FileNotFoundError(2, "No such file or directory", argv[0])
    monkeypatch.setattr(pp.subprocess, "run", missing)
    got = pp.probe_policy("gc-treasury", "001", ["F212N"],
                          pipe_root=_probe_root(tmp_path),
                          python="/nonexistent/python")
    assert got is None
    err = capsys.readouterr().err
    assert "could not run /nonexistent/python" in err
    assert "align_o001_crf" in err


@pytest.mark.parametrize("payload", [
    json.dumps({"suffix": "destreak_o001_crf"}),         # older probe shape
    json.dumps({"each_suffix": None, "destreaks": {}}),   # null suffix
    json.dumps({"each_suffix": "destreak_o001_crf"}),     # destreaks missing
    json.dumps(["destreak_o001_crf"]),                    # not a mapping
    json.dumps("destreak_o001_crf"),                      # bare string
])
def test_probe_wrong_shape_payload_warns_and_returns_none(payload, tmp_path,
                                                          monkeypatch, capsys):
    """Valid JSON of the WRONG shape (older checkout, stray print merged into
    stdout, partial write) must be rejected.  Caching it as a truthy policy
    would send it to cataloging_step as ``policy["each_suffix"]``, and the
    KeyError there aborts act_trigger mid-loop, leaving every later observation
    in that poll unsubmitted."""
    monkeypatch.setattr(pp.subprocess, "run", _stub_run(payload))
    got = pp.probe_policy("gc-treasury", "001", ["F212N"],
                          pipe_root=_probe_root(tmp_path))
    assert got is None
    assert "WARNING" in capsys.readouterr().err
    # ... and the plan still builds, on today's hardcoded default
    plan = pt.build_plan(10678, "001", filters=["F212N"],
                         pipe_root=_probe_root(tmp_path))
    assert plan[1]["env"]["EACH_SUFFIX"] == "align_o001_crf"


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
    """One subprocess per distinct (field, obs, filters, pipe_root, python) per
    run, even for repeated build_plans."""
    calls = []
    monkeypatch.setattr(pp.subprocess, "run",
                        _stub_run(json.dumps(TREASURY_POLICY), calls=calls))
    root = _probe_root(tmp_path)
    for _ in range(3):
        pp.probe_policy("gc-treasury", "001", ["F212N", "F480M"], pipe_root=root)
    assert len(calls) == 1


def test_probe_cache_key_includes_filters(tmp_path, monkeypatch):
    """The filter list is part of the key, because the policy is per filter:
    sickle o002 F212N -> destreak_o002_crf while F480M -> align_o002_crf, so a
    second filter list must not be served the first list's answer."""
    calls = []
    monkeypatch.setattr(pp.subprocess, "run",
                        _stub_run(json.dumps(TREASURY_POLICY), calls=calls))
    root = _probe_root(tmp_path)
    pp.probe_policy("sickle", "002", ["F212N"], pipe_root=root)
    pp.probe_policy("sickle", "002", ["F480M"], pipe_root=root)
    assert len(calls) == 2
    assert calls[0][-1] == "F212N"
    assert calls[1][-1] == "F480M"
    # ... and the same list still hits the cache
    pp.probe_policy("sickle", "002", ["F480M"], pipe_root=root)
    assert len(calls) == 2


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


def test_build_plan_each_suffix_override_skips_probe(monkeypatch):
    """An explicit --each-suffix names the glob outright, so no subprocess runs."""
    def boom(*a, **k):
        raise AssertionError("probe must not run under an explicit each_suffix")
    monkeypatch.setattr(pp, "probe_policy", boom)
    plan = pt.build_plan(4147, "012", filters=["F405N"], pipe_root="/pipe",
                         each_suffix="custom_o012_crf")
    assert plan[1]["env"]["EACH_SUFFIX"] == "custom_o012_crf"


def test_build_plan_destreak_override_wins_but_still_probes(tmp_path,
                                                            monkeypatch):
    """An operator destreak choice wins the EACH_SUFFIX ladder, and the probe
    still runs -- the policy is what can tell the operator the flag contradicts
    what the reduction will write."""
    calls = []
    monkeypatch.setattr(pp.subprocess, "run",
                        _stub_run(json.dumps(TREASURY_POLICY), calls=calls))
    plan = pt.build_plan(10678, "001", filters=["F212N"], destreak=True,
                         pipe_root=_probe_root(tmp_path))
    assert plan[1]["env"]["EACH_SUFFIX"] == "destreak_o001_crf"
    assert len(calls) == 1


def test_build_plan_no_destreak_contradicting_policy_warns(tmp_path,
                                                           monkeypatch, capsys):
    """--no-destreak on a field the policy destreaks reproduces the #69 zero
    glob: the driver applies destreak_policy itself and writes
    destreak_o001_crf, while the chain globs align_o001_crf.  NO_DESTREAK=1
    cannot stop it (no consumer in the pipeline), so the plan says so."""
    payload = {"each_suffix": "destreak_o001_crf",
               "destreaks": {"F405N": True, "F410M": True}}
    monkeypatch.setattr(pp.subprocess, "run", _stub_run(json.dumps(payload)))
    plan = pt.build_plan(2221, "001", filters=["F405N", "F410M"],
                         destreak=False, pipe_root=_probe_root(tmp_path))
    assert plan[1]["env"]["EACH_SUFFIX"] == "align_o001_crf"   # the flag wins
    err = capsys.readouterr().err
    assert "--no-destreak contradicts" in err
    assert "F405N, F410M" in err
    assert "ZERO inputs at m1" in err
    assert "NO_DESTREAK=1 has no consumer" in err


def test_build_plan_destreak_on_policy_off_field_warns(tmp_path, monkeypatch,
                                                       capsys):
    """The mirror direction: --destreak on an extended-emission field globs
    destreak_o001_crf against an align_* reduction."""
    payload = {"each_suffix": "align_o001_crf", "destreaks": {"F150W": False}}
    monkeypatch.setattr(pp.subprocess, "run", _stub_run(json.dumps(payload)))
    plan = pt.build_plan(6151, "001", filters=["F150W"], destreak=True,
                         pipe_root=_probe_root(tmp_path))
    assert plan[1]["env"]["EACH_SUFFIX"] == "destreak_o001_crf"
    err = capsys.readouterr().err
    assert "--destreak contradicts" in err
    assert "F150W" in err
    # the NO_DESTREAK note belongs to the other direction only
    assert "NO_DESTREAK=1 has no consumer" not in err


def test_build_plan_destreak_agreeing_with_policy_is_quiet(tmp_path,
                                                           monkeypatch, capsys):
    """An explicit flag that matches the policy warns about nothing."""
    monkeypatch.setattr(pp.subprocess, "run",
                        _stub_run(json.dumps(TREASURY_POLICY)))
    pt.build_plan(10678, "001", filters=["F212N", "F480M"], destreak=True,
                  pipe_root=_probe_root(tmp_path))
    assert "contradicts" not in capsys.readouterr().err


def test_build_plan_no_destreak_consistent_pair():
    """destreak=False reaches BOTH steps: align cataloging + NO_DESTREAK=1 on
    the reduction, so the pair cannot split once the sbatch passthrough lands."""
    plan = pt.build_plan(2221, "001", filters=["F405N"], pipe_root="/pipe",
                         destreak=False)
    assert plan[0]["env"]["NO_DESTREAK"] == "1"
    assert plan[1]["env"]["EACH_SUFFIX"] == "align_o001_crf"


def test_build_plan_mixed_destreak_policy_warns(tmp_path, monkeypatch, capsys):
    """A per-filter split (sickle: SW destreaks, LW does not) cannot ride one
    EACH_SUFFIX -- the plan says so, so those filters go by hand."""
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
                                   ("gc2211_o023", 2211, "023"),   # per-obs split field (#119)
                                   ("arches", 2045, "001")):
        step = pt.cataloging_step(program, obsnum, field, ["F212N"],
                                  pipe_root="/pipe")
        assert step["env"]["DEBLEND_SATSTARS"] == "1", field


def test_deblend_satstars_off_for_non_gc_fields():
    """Off is the EMPTY value, set explicitly: submit_cataloging.sbatch tests
    `[ -n "$DEBLEND_SATSTARS" ]`, so empty is the only value that means off."""
    for field, program, obsnum in (("wd1", 1905, "001"), ("w51", 6151, "001"),
                                   ("ngc6334", 6778, "001")):
        step = pt.cataloging_step(program, obsnum, field, ["F150W"],
                                  pipe_root="/pipe")
        assert step["env"]["DEBLEND_SATSTARS"] == "", field


@pytest.mark.parametrize("ambient", ["1", "0", "off"])
def test_deblend_satstars_non_gc_overrides_ambient_env(monkeypatch, ambient):
    """An ambient DEBLEND_SATSTARS must not reach a non-GC chain: run_plan
    composes dict(os.environ, **step["env"]) and the chain exports ALL, and
    `[ -n ... ]` treats ANY non-empty value -- "0" included -- as on."""
    monkeypatch.setenv("DEBLEND_SATSTARS", ambient)
    step = pt.cataloging_step(6151, "001", "w51", ["F150W"], pipe_root="/pipe")
    composed = dict(os.environ, **step["env"])
    assert composed["DEBLEND_SATSTARS"] == ""
    gc_step = pt.cataloging_step(10678, "001", "gc-treasury", ["F212N"],
                                 pipe_root="/pipe")
    assert dict(os.environ, **gc_step["env"])["DEBLEND_SATSTARS"] == "1"


def test_skymatch_rejects_a_value_skymatchstep_cannot_take(monkeypatch):
    """SKYMATCH rides through unchecked to --skymatch-method, so a typo would
    survive Detector1 + Image2 and die in Image3 hours later; build time is
    where it costs nothing."""
    monkeypatch.setenv("SKYMATCH", "matchh")
    with pytest.raises(ValueError, match="SkyMatchStep skymethod"):
        pt.reduction_step(10678, "001", "gc-treasury", ["F212N"],
                          pipe_root="/pipe")


@pytest.mark.parametrize("method", ["local", "global", "match",
                                    "global+match", "user"])
def test_skymatch_accepts_every_skymatchstep_method(monkeypatch, method):
    monkeypatch.setenv("SKYMATCH", method)
    step = pt.reduction_step(10678, "001", "gc-treasury", ["F212N"],
                             pipe_root="/pipe")
    assert step["env"]["SKYMATCH"] == method


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


# ------------------------------------------------- registry preflight (issue #68)
def _fake_python(tmp_path, body, module="/pipe/jwst_gc_pipeline/fields.py"):
    """An executable stand-in for the pipeline env python.  It announces the
    fields module it 'imported' first, as the real preflight code does; pass
    ``module=None`` for a child that never got that far (an import failure)."""
    lines = ["#!/bin/sh"]
    if module:
        lines.append(f'echo "{pt._MODULE_MARK}{module}" >&2')
    lines.append(body)
    script = tmp_path / "fakepython"
    script.write_text("\n".join(lines) + "\n")
    script.chmod(0o755)
    return str(script)


def test_build_plan_runs_registry_preflight(preflight_calls):
    """build_plan consults the pipeline registry BEFORE returning any step."""
    pt.build_plan(2221, "001", filters=["F405N"], pipe_root="/pipe")
    assert preflight_calls == [(2221, "001", {"pipe_root": "/pipe"})]


def test_preflight_precedes_the_policy_probe(tmp_path, monkeypatch):
    """The two pipeline-env subprocesses compose in one order: the registry
    preflight (issue #68) gates, then the destreak-policy probe (issue #69)
    supplies EACH_SUFFIX.  An observation the registry rejects never reaches the
    probe, and the probe's result still lands in the plan when it passes."""
    order = []
    monkeypatch.setattr(pt, "registry_preflight",
                        lambda program, obs, **kw: order.append("preflight"))
    monkeypatch.setattr(pp, "probe_policy",
                        lambda *a, **k: order.append("probe") or TREASURY_POLICY)
    plan = pt.build_plan(10678, "001", field="gc-treasury", filters=["F212N"],
                         pipe_root=_probe_root(tmp_path))
    assert order == ["preflight", "probe"]
    assert plan[1]["env"]["EACH_SUFFIX"] == TREASURY_POLICY["each_suffix"]

    order.clear()

    def deny(program, obs, **kw):
        order.append("preflight")
        raise pt.NotRegisteredInPipelineError("not in fields.yaml")

    monkeypatch.setattr(pt, "registry_preflight", deny)
    with pytest.raises(pt.NotRegisteredInPipelineError):
        pt.build_plan(10678, "001", field="gc-treasury", filters=["F212N"],
                      pipe_root=_probe_root(tmp_path))
    assert order == ["preflight"]


def test_unregistered_obs_raises_before_any_sbatch(monkeypatch):
    """The typed error escapes submit() with sbatch never reached."""
    def deny(program, obs, **kw):
        raise pt.NotRegisteredInPipelineError("not in fields.yaml")

    monkeypatch.setattr(pt, "registry_preflight", deny)
    monkeypatch.setattr(pt, "run_plan",
                        lambda plan: pytest.fail("sbatch must not run"))
    with pytest.raises(pt.NotRegisteredInPipelineError):
        pt.submit(10678, "001", field="gc-treasury", filters=["F212N"],
                  execute=True)


def test_registry_preflight_unregistered_raises(tmp_path):
    """rc 3 + the verdict line = the registry's OWN answer -> block."""
    py = _fake_python(
        tmp_path,
        'echo "REGISTRY-VERDICT: KeyError: proposal 10678 observation" >&2\n'
        f'exit {pt.PREFLIGHT_NOT_REGISTERED_RC}')
    with pytest.raises(pt.NotRegisteredInPipelineError,
                       match="10678 obs 001.*KeyError") as ex:
        _REAL_PREFLIGHT(10678, "001", pipe_root="/pipe", python=py)
    # the checkout the operator has to go register the observation in is the
    # whole point of the provenance work -- the message names it
    assert "in the pipeline at /pipe" in str(ex.value)


@pytest.mark.parametrize("body,tell", [
    ('echo "ModuleNotFoundError: No module named astropy" >&2\nexit 1',
     "ModuleNotFoundError"),
    ("exit 137", "rc=137"),                     # OOM-killed subprocess
    ("exit 2", "rc=2"),
])
def test_registry_preflight_broken_check_fails_open(tmp_path, capsys, body, tell):
    """ONLY rc 3 blocks.  A pipeline env that cannot import (conda update,
    half-installed dependency, bad PYTHONPATH) or a killed subprocess must NOT
    be reported as 'not registered' -- that would silence EVERY trigger, for
    every already-registered program, on every poll until the env is repaired.
    The rc verdict is read BEFORE the registry-provenance check, so a child
    that died before naming its fields module still reports the rc."""
    py = _fake_python(tmp_path, body, module=None)
    assert _REAL_PREFLIGHT(2221, "001", pipe_root="/pipe", python=py) is None
    err = capsys.readouterr().err
    assert "FAILED to reach a verdict" in err and "fail-open" in err
    assert tell in err
    assert "not registered" not in err


def test_registry_preflight_broken_check_after_naming_its_module(tmp_path,
                                                                 capsys):
    """A child that imported its fields module and THEN died still fails open,
    and the warning quotes the failure -- the protocol lines are filtered out of
    the reported tail, so the operator reads the traceback line rather than
    'REGISTRY-MODULE: ...'."""
    py = _fake_python(tmp_path,
                      'echo "RuntimeError: registry file unreadable" >&2\nexit 1')
    assert _REAL_PREFLIGHT(2221, "001", pipe_root="/pipe", python=py) is None
    err = capsys.readouterr().err
    assert "FAILED to reach a verdict" in err and "fail-open" in err
    assert "RuntimeError: registry file unreadable" in err
    assert pt._MODULE_MARK not in err


def test_registry_preflight_broken_check_with_no_output(tmp_path, capsys):
    """Only the protocol line on stderr: the warning still reports the rc and
    says so rather than quoting the marked line as if it were the failure."""
    py = _fake_python(tmp_path, "exit 1")
    assert _REAL_PREFLIGHT(2221, "001", pipe_root="/pipe", python=py) is None
    err = capsys.readouterr().err
    assert "rc=1: no output" in err and "fail-open" in err
    assert pt._MODULE_MARK not in err


def test_registry_preflight_rejects_weird_program(tmp_path, monkeypatch):
    """``program`` is interpolated into the -c code too; ``int()`` is its whole
    sanitizer, so a non-numeric program must be refused before any subprocess."""
    monkeypatch.setattr(pt.subprocess, "run",
                        lambda *a, **k: pytest.fail("no subprocess may run"))
    with pytest.raises(ValueError):
        _REAL_PREFLIGHT("2221'); import os; ('", "001", pipe_root="/pipe",
                        python=_fake_python(tmp_path, "exit 0"))


def test_registry_preflight_child_code_matches_the_real_registry(tmp_path):
    """The rc-3 protocol is a contract with the pipeline's fields module, so
    exercise it against the REAL interpreter + registry: a registered obs
    passes, a nonexistent proposal raises (not fails open)."""
    py = pt.DEFAULT_PIPELINE_PYTHON
    pipe_root = pt.DEFAULT_PIPE_ROOT
    if not (os.path.exists(py) and os.path.isdir(pipe_root)):
        pytest.skip("pipeline env/checkout not on this host")
    assert _REAL_PREFLIGHT(2221, "001", pipe_root=pipe_root, python=py) is None
    with pytest.raises(pt.NotRegisteredInPipelineError, match="99999 obs 001"):
        _REAL_PREFLIGHT(99999, "001", pipe_root=pipe_root, python=py)


def test_registry_preflight_registered_passes(tmp_path, capsys):
    """A clean pass is SILENT: any warning here would print on every poll for
    every registered observation, which is how a fail-open branch hides."""
    py = _fake_python(tmp_path, "exit 0")
    assert _REAL_PREFLIGHT(2221, "001", pipe_root="/pipe", python=py) is None
    assert capsys.readouterr().err == ""


def test_registry_preflight_timeout_fails_open(tmp_path, capsys):
    """A wedged pipeline import must not silence real triggers: TIMEOUT warns
    and proceeds (fail-open), never raises."""
    py = _fake_python(tmp_path, "sleep 5")
    _REAL_PREFLIGHT(2221, "001", pipe_root="/pipe", python=py, timeout_s=0.2)
    err = capsys.readouterr().err
    assert "TIMED OUT" in err and "fail-open" in err


def test_registry_preflight_missing_interpreter_fails_open(tmp_path, capsys):
    _REAL_PREFLIGHT(2221, "001", pipe_root="/pipe",
                    python=str(tmp_path / "no-such-python"))
    assert "could not run" in capsys.readouterr().err


def test_registry_preflight_pythonpath_fronts_pipe_root(tmp_path):
    """The checkout being submitted against is the registry consulted."""
    out = tmp_path / "pythonpath.txt"
    py = _fake_python(tmp_path, f'echo "$PYTHONPATH" > {out}\nexit 0')
    _REAL_PREFLIGHT(2221, "001", pipe_root="/my/pipe", python=py)
    assert out.read_text().startswith("/my/pipe")


def test_registry_preflight_rejects_weird_obs_token(tmp_path):
    """Only digit/'-' obs tokens may reach the interpolated -c code."""
    with pytest.raises(ValueError, match="observation token"):
        _REAL_PREFLIGHT(2221, "001'; import os", pipe_root="/pipe",
                        python=_fake_python(tmp_path, "exit 0"))


def test_registry_preflight_rejects_trailing_newline_obs_token(tmp_path):
    """fullmatch, not match+'$': '001\\n' would otherwise reach the -c code and
    die of a syntax error (rc 1)."""
    with pytest.raises(ValueError, match="observation token"):
        _REAL_PREFLIGHT(2221, "001\n", pipe_root="/pipe",
                        python=_fake_python(tmp_path, "exit 0"))


def test_registry_preflight_accepts_joint_obs_token(tmp_path):
    _REAL_PREFLIGHT(2221, "001-002", pipe_root="/pipe",
                    python=_fake_python(tmp_path, "exit 0"))


def test_obs_token_grammar_is_shared_with_rearm():
    """--rearm must be able to address every key the preflight lets the monitor
    write, so the two validators are ONE pattern ('\\d' is Unicode-aware and
    unbounded; '[0-9]{1,4}' is neither, and they used to disagree)."""
    assert pt._OBS_TOKEN_RE.pattern == mm.OBS_TOKEN_PATTERN
    for token, ok in [("001", True), ("001-002", True), ("1", True),
                      ("00001", False), ("12345", False), ("١٢٣", False),
                      ("001\n", False), ("001x", False), ("-001", False)]:
        assert bool(pt._OBS_TOKEN_RE.fullmatch(token)) is ok, token
        assert bool(mm._REARM_RE.fullmatch(f"2221-o{token}")) is ok, token


def test_registry_preflight_verdict_from_another_checkout_fails_open(tmp_path,
                                                                     capsys):
    """jwst_gc_pipeline is pip-installed in the pipeline env, so a pipe_root
    holding no importable package leaves the PYTHONPATH prepend inert and the
    INSTALLED registry answers.  A refusal naming a checkout it never consulted
    is not a verdict: warn and proceed."""
    py = _fake_python(tmp_path, f"exit {pt.PREFLIGHT_NOT_REGISTERED_RC}",
                      module="/somewhere/else/jwst_gc_pipeline/fields.py")
    assert _REAL_PREFLIGHT(10678, "001", pipe_root="/pipe", python=py) is None
    err = capsys.readouterr().err
    assert "/somewhere/else" in err and "not under /pipe" in err
    assert "fail-open" in err


def test_registry_preflight_sibling_checkout_fails_open(tmp_path, capsys):
    """'/pipe-wt' is not under '/pipe'.  The provenance check compares against
    ``root + os.sep`` for that reason: '<root>' / '<root>-wt' is the worktree
    naming convention this repo works in, and a plain string prefix would
    accept a sibling checkout's verdict -- refusing a REGISTERED observation on
    an answer from a registry that was never consulted."""
    py = _fake_python(tmp_path, f"exit {pt.PREFLIGHT_NOT_REGISTERED_RC}",
                      module="/pipe-wt/jwst_gc_pipeline/fields.py")
    assert _REAL_PREFLIGHT(2221, "001", pipe_root="/pipe", python=py) is None
    err = capsys.readouterr().err
    assert "/pipe-wt" in err and "not under /pipe" in err and "fail-open" in err


def test_registry_preflight_accepts_a_symlinked_pipe_root(tmp_path):
    """The child reports ``os.path.realpath(fields.__file__)``, so the parent
    resolves pipe_root the same way before comparing.  Without that resolution a
    pipe_root reached through any symlinked component -- routine on this
    filesystem, and guaranteed under the worktree workflow -- never prefix-
    matches, so every call fails open with the suite still green."""
    real = tmp_path / "real-checkout"
    (real / "jwst_gc_pipeline").mkdir(parents=True)
    link = tmp_path / "link-checkout"
    link.symlink_to(real)
    fields = os.path.realpath(str(real / "jwst_gc_pipeline" / "fields.py"))
    py = _fake_python(tmp_path, f"exit {pt.PREFLIGHT_NOT_REGISTERED_RC}",
                      module=fields)
    with pytest.raises(pt.NotRegisteredInPipelineError):
        _REAL_PREFLIGHT(10678, "001", pipe_root=str(link), python=py)


def test_registry_preflight_unidentified_registry_fails_open(tmp_path, capsys):
    """A child that exits 3 without naming the fields module it imported has
    not answered the protocol -- rc alone must not block."""
    py = _fake_python(tmp_path, f"exit {pt.PREFLIGHT_NOT_REGISTERED_RC}",
                      module=None)
    assert _REAL_PREFLIGHT(10678, "001", pipe_root="/pipe", python=py) is None
    assert "unidentified registry" in capsys.readouterr().err


def test_registry_preflight_verdict_quotes_the_registrys_own_line(tmp_path):
    """The child's stderr also carries whatever the pipeline import printed
    (deprecation warnings, CRDS chatter); the raised message quotes the MARKED
    verdict line, not the last thing on stderr."""
    py = _fake_python(
        tmp_path,
        'echo "REGISTRY-VERDICT: KeyError: proposal 10678 not in fields.yaml" >&2\n'
        'echo "WARNING: some astropy deprecation" >&2\n'
        f"exit {pt.PREFLIGHT_NOT_REGISTERED_RC}")
    with pytest.raises(pt.NotRegisteredInPipelineError) as ex:
        _REAL_PREFLIGHT(10678, "001", pipe_root="/pipe", python=py)
    assert "not in fields.yaml" in str(ex.value)
    assert "deprecation" not in str(ex.value)


def _real_pipeline_env():
    return (os.path.exists(pt.DEFAULT_PIPELINE_PYTHON)
            and os.path.isdir(pt.DEFAULT_PIPE_ROOT))


def test_preflight_child_code_fails_open_on_a_broken_checkout(tmp_path, capsys):
    """The child's ``from jwst_gc_pipeline import fields`` sits OUTSIDE its try
    so an ImportError is rc 1 (fail-open), never a verdict.  Run the REAL
    _PREFLIGHT_CODE through the real interpreter against a checkout whose
    package cannot import: moving that import inside the try and widening the
    except would otherwise turn a broken env into 'not registered' with a green
    suite."""
    if not os.path.exists(pt.DEFAULT_PIPELINE_PYTHON):
        pytest.skip("pipeline env python not on this host")
    pkg = tmp_path / "jwst_gc_pipeline"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("import definitely_not_a_real_module\n")
    (pkg / "fields.py").write_text("raise AssertionError('never reached')\n")
    assert _REAL_PREFLIGHT(2221, "001", pipe_root=str(tmp_path),
                           python=pt.DEFAULT_PIPELINE_PYTHON) is None
    err = capsys.readouterr().err
    assert "FAILED to reach a verdict" in err and "fail-open" in err
    assert "definitely_not_a_real_module" in err
    assert "not registered" not in err


def test_preflight_instrument_selects_the_registry_section():
    """Joint obs tokens are a MIRI reality (sgrb2 5365 '002-998'); the nircam
    section answers 'not registered' for them, so the section the preflight
    asks must be selectable."""
    if not _real_pipeline_env():
        pytest.skip("pipeline env/checkout not on this host")
    kw = dict(pipe_root=pt.DEFAULT_PIPE_ROOT, python=pt.DEFAULT_PIPELINE_PYTHON)
    assert _REAL_PREFLIGHT(5365, "002-998", instrument="miri", **kw) is None
    with pytest.raises(pt.NotRegisteredInPipelineError, match=r"\(nircam\)"):
        _REAL_PREFLIGHT(5365, "002-998", instrument="nircam", **kw)


def test_registry_preflight_rejects_weird_instrument(tmp_path):
    """The instrument is interpolated into the -c code too."""
    with pytest.raises(ValueError, match="instrument"):
        _REAL_PREFLIGHT(2221, "001", instrument="nircam'; import os",
                        pipe_root="/pipe",
                        python=_fake_python(tmp_path, "exit 0"))


# ----------------------------------------------------- jobid capture (issue #68)
def test_parse_jobids_parsable_and_submitted_lines():
    """Per step: the --parsable step's own first line, plus every step's
    'Submitted batch job' lines, deduped in first-seen order."""
    results = {"reduction": "12345;hpg\n",
               "cataloging-chain": ("Submitted batch job 12346\n"
                                    "Submitted batch job 12347\n")}
    assert pt.parse_jobids(results) == ["12345", "12346", "12347"]
    assert pt.parse_jobids({}) == []
    assert pt.parse_jobids("") == []
    assert pt.parse_jobids(None) == []


def test_parse_jobids_rejects_a_bare_number_that_is_not_a_jobid():
    """Issue #87: a year or a count printed by the cataloging chain is not a
    SLURM job id.  Recording it would hand the planned sacct outcome probe
    (#68c) an id sacct has never heard of, and 'no such job' would read as an
    outcome for a submission that ran fine."""
    results = {"cataloging-chain": "chain built 2026\n7\n"
                                   "Submitted batch job 12346\n"}
    assert pt.parse_jobids(results) == ["12346"]
    # ... and the same line noise inside the --parsable step's capture: the id
    # is read POSITIONALLY (first line), so the trailing bare numbers are not
    # ids either
    assert pt.parse_jobids({"reduction": "12345\n2026\n7\n"}) == ["12345"]
    # a --parsable capture that is not a job id at all yields nothing
    assert pt.parse_jobids({"reduction": "sbatch: error: whatever\n"}) == []


def test_run_plan_refuses_a_reduction_capture_that_is_not_a_jobid(monkeypatch):
    """rc=0 with unparsable stdout used to become DEP=<that text> on the
    cataloging chain (--dependency=afterok:<garbage>)."""
    class _Proc:
        returncode = 0
        stdout = "sbatch: submitted, probably\n"
        stderr = ""

    monkeypatch.setattr(pt.subprocess, "run", lambda *a, **k: _Proc())
    plan = pt.build_plan(2221, "001", field="brick", filters=["F405N"],
                         pipe_root="/pipe", probe=False)
    with pytest.raises(RuntimeError, match="not a job id"):
        pt.run_plan(plan)


def test_submit_dry_run_returns_plan_without_jobids(tmp_path):
    out = pt.submit(2221, "001", filters=["F405N"], pipe_root=str(tmp_path))
    assert [s["name"] for s in out["plan"]] == ["reduction", "cataloging-chain"]
    assert out["jobids"] == []
    assert out["results"] == {}
