"""Build (and optionally submit) the jwst-gc-pipeline SLURM sequence for one
observation: reduction filter-array job, then the cataloging chain dependency-gated
on it (``DEP=<reduction jobid>`` -> ``--dependency=afterok``).

Wraps the pipeline's own submitters (never re-implements them):
  scripts/reduction/submit_reduction.sbatch        (array over filters)
  scripts/reduction/submit_cataloging_chain.sh     (per-filter array + m7 finalize)

Conventions honored (see jwst-gc-pipeline CLAUDE.md):
  * job names at SUBMIT time: <target><program>-o<obsid>-<stage>
  * comma-valued vars (MODULES) go through the process ENVIRONMENT + --export=ALL,
    never inside the --export list (the SLURM --export comma trap);
  * the cataloging guard needs PROPOSAL/FIELD/TARGET/EACH_SUFFIX/MODULES together;
  * env defaults come from the pipeline's OWN policy (pipeline_policy.probe_policy
    -> destreak_policy.crf_suffix), the same derivation run_pipeline.build_plan
    uses, so the triggered reduction and cataloging cannot disagree about what
    the crf products are called (issue #69).  A failed probe degrades to the
    historical hardcoded default with a loud warning.

Registry preflight (issue #68): build_plan verifies the observation is registered
in the pipeline's fields.yaml BEFORE any sbatch, by subprocessing the pipeline
env python (``fields.target_for_obsid``).  An unregistered obs raises
NotRegisteredInPipelineError in-process -- sbatch would have ACCEPTED the jobs
and KeyError'd on-node minutes later, burning the monitor's one-shot trigger key.

Stdlib-only.  Dry-run (default) prints the exact commands; --execute submits and
threads the parsed reduction job id into DEP.

Usage:
    python -m data_qa.pipeline_trigger --program 2221 --obs 001 \\
        --filters F405N F410M F466N F212N            # dry-run print
    python -m data_qa.pipeline_trigger --program 2221 --obs 001 \\
        --filters F405N F410M --execute              # really sbatch
"""
from __future__ import annotations

import argparse
import os
import re
import shlex
import subprocess
import sys
from typing import Dict, List, Optional

from . import paths, pipeline_policy
from .mast_monitor import GC_FIELDS, OBS_TOKEN_PATTERN, field_for

#: single home for the pipe-root default (data_qa.paths owns the path and the
#: $PIPE_ROOT override; re-exported here so the trigger's callers and CLI keep
#: the historical name).  Resolve with paths.pipe_root(), which reads the env.
DEFAULT_PIPE_ROOT = paths.DEFAULT_PIPE_ROOT
REDUCTION_SBATCH = "scripts/reduction/submit_reduction.sbatch"
CATALOGING_CHAIN = "scripts/reduction/submit_cataloging_chain.sh"
DEP_PLACEHOLDER = "<REDUCTION_JOBID>"

#: SkyMatchStep.spec's ``skymethod`` options.  submit_reduction.sbatch passes
#: SKYMATCH straight through as --skymatch-method and nothing downstream checks
#: it, so an unknown value survives Detector1 + Image2 (hours) and dies in
#: Image3; the trigger rejects it at build time instead.
SKYMATCH_METHODS = ("local", "global", "match", "global+match", "user")

# The registry parser (fields.yaml) lives in jwst_gc_pipeline, whose environment
# is far heavier than this stdlib-only module -- so the preflight subprocesses
# the pipeline env python instead of importing.  Override the interpreter with
# $PIPELINE_PYTHON (the same knob the pipeline's own submit scripts honour).
DEFAULT_PIPELINE_PYTHON = ("/blue/adamginsburg/adamginsburg/miniconda3/envs/"
                           "python313/bin/python")
PREFLIGHT_TIMEOUT_S = 30

# obs tokens are digits, optionally '-'-joined ('001-002' names a JOINT
# observation: several observations cataloged as one unit); anything else must
# never be interpolated into the -c code.  The pattern is imported from
# mast_monitor so --rearm and this validator share ONE grammar (they used to
# drift: '\d+' is Unicode-aware and unbounded, '[0-9]{1,4}' is neither).
# fullmatch, not match+'$': '$' also matches before a trailing newline, and
# '001\n' would reach the -c code as a syntax error (rc 1) instead of a
# rejected token.
_OBS_TOKEN_RE = re.compile(OBS_TOKEN_PATTERN)
# instruments name a fields.yaml section ('nircam'/'miri'/...); same reasoning
_INSTRUMENT_RE = re.compile(r"[a-z]{2,10}")

# The child answers with a STATUS, not merely non-zero: only rc 3 is the
# registry's own verdict ("this observation is not in fields.yaml").  Any other
# non-zero rc means the CHECK broke -- a failed import (half-installed env, bad
# PYTHONPATH), an OOM kill, a syntax error -- and must fail OPEN, exactly like
# the timeout/OSError paths.  Reporting every non-zero rc as 'not registered'
# would let one broken pipeline env silence EVERY trigger, for every already
# registered program, on every poll.
PREFLIGHT_NOT_REGISTERED_RC = 3
_VERDICT_MARK = "REGISTRY-VERDICT: "
# The child also names the fields module it actually imported: jwst_gc_pipeline
# is pip-installed in the pipeline env, so a pipe_root holding no importable
# package leaves the PYTHONPATH prepend inert and the INSTALLED registry
# answers -- a verdict from a checkout other than the one the error message
# names.  The parent verifies the reported path lies under pipe_root and fails
# open when it does not.
_MODULE_MARK = "REGISTRY-MODULE: "
_PREFLIGHT_CODE = (
    # the import sits OUTSIDE the try on purpose: an ImportError here is a
    # broken check (rc 1 -> fail-open), never a registry verdict
    "import os, sys\n"
    "from jwst_gc_pipeline import fields\n"
    "sys.stderr.write('{modmark}%s\\n' % (os.path.realpath(fields.__file__),))\n"
    "try:\n"
    "    fields.target_for_obsid('{program}', '{obs}', instrument='{instrument}')\n"
    "except (KeyError, fields.FieldRegistryError) as ex:\n"
    "    sys.stderr.write('{mark}%s\\n' % (ex,))\n"
    "    sys.exit({rc})\n"
)


def _marked(lines, mark) -> List[str]:
    """The payloads of the child's ``<MARK>...`` protocol lines."""
    return [ln[len(mark):] for ln in lines if ln.startswith(mark)]


class NotRegisteredInPipelineError(RuntimeError):
    """The (program, obs) has no fields.yaml entry in the pipeline registry:
    sbatch would ACCEPT the submission and the job would KeyError on-node
    minutes later, burning the monitor's one-shot trigger key (issue #68)."""


def registry_preflight(program, obs, pipe_root=None, python=None,
                       timeout_s=PREFLIGHT_TIMEOUT_S, instrument="nircam"):
    """Verify (program, obs) is registered in the pipeline's fields.yaml BEFORE
    any sbatch; raises NotRegisteredInPipelineError when it is not.

    ``pipe_root`` fronts PYTHONPATH so the checkout being submitted against is
    the one consulted; the interpreter is ``python`` / $PIPELINE_PYTHON /
    DEFAULT_PIPELINE_PYTHON.  ``instrument`` selects the fields.yaml section --
    joint obs tokens ('002-998') exist under miri, and the nircam section
    answers "not registered" for them.  ``instrument`` is a DIRECT-CALL
    argument: build_plan wraps the NIRCam reduction/cataloging submitters only,
    so it asks the nircam section and grows an instrument of its own when the
    trigger learns to submit MIRI.

    ``pipe_root`` is compared as a REALPATH against the realpath the child
    reports, and with a trailing separator: this account's checkouts are reached
    through symlinked components and its worktrees are named '<root>-<slug>'
    beside the root, so an unresolved compare would fail open on every call and
    a separator-less prefix would accept a sibling checkout's verdict.

    ONLY the child's rc 3 (PREFLIGHT_NOT_REGISTERED_RC, written by the child
    when target_for_obsid raises KeyError/FieldRegistryError) blocks, and only
    when the child reports importing the fields module from UNDER pipe_root.
    Every other outcome -- subprocess TIMEOUT, an interpreter that cannot
    start, any other non-zero rc (import failure in a half-installed env, OOM
    kill), and a verdict reached from a registry outside pipe_root -- warns and
    PROCEEDS.  All of them mean the CHECK is broken, and a broken check must
    not silence real triggers: a false skip leaves delivered data unreduced
    with no error recorded anywhere and the operator pointed at a registration
    that is already correct, while proceeding reproduces the pre-preflight
    burn-on-submit behaviour for that one observation."""
    obs_token = str(obs)
    if not _OBS_TOKEN_RE.fullmatch(obs_token):
        raise ValueError(
            f"obs {obs!r} is not a plausible observation token "
            "(1-4 digits, optionally '-'-joined for a joint observation, "
            "e.g. '001' or '001-002')")
    instrument = str(instrument).lower()
    if not _INSTRUMENT_RE.fullmatch(instrument):
        raise ValueError(f"instrument {instrument!r} is not a plausible "
                         "instrument name (lowercase letters, e.g. 'nircam')")
    python = (python or os.environ.get("PIPELINE_PYTHON")
              or DEFAULT_PIPELINE_PYTHON)
    pipe_root = pipe_root or paths.pipe_root()
    code = _PREFLIGHT_CODE.format(program=int(program), obs=obs_token,
                                  instrument=instrument, mark=_VERDICT_MARK,
                                  modmark=_MODULE_MARK,
                                  rc=PREFLIGHT_NOT_REGISTERED_RC)
    env = dict(os.environ)
    env["PYTHONPATH"] = pipe_root + os.pathsep + env.get("PYTHONPATH", "")
    try:
        proc = subprocess.run([python, "-c", code], env=env,
                              capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        print(f"pipeline_trigger: registry preflight TIMED OUT after "
              f"{timeout_s}s ({python}); proceeding WITHOUT the registry "
              "check (fail-open: a wedged pipeline import must not silence "
              "real triggers)", file=sys.stderr)
        return
    except OSError as ex:
        print(f"pipeline_trigger: registry preflight could not run "
              f"({ex.__class__.__name__}: {ex}); proceeding WITHOUT the "
              "registry check (fail-open, same rationale as the timeout)",
              file=sys.stderr)
        return
    lines = [ln.strip() for ln
             in ((proc.stderr or "") + (proc.stdout or "")).splitlines()
             if ln.strip()]
    if proc.returncode not in (0, PREFLIGHT_NOT_REGISTERED_RC):
        # the check itself broke -> fail open (see the docstring)
        plain = [ln for ln in lines
                 if not ln.startswith((_VERDICT_MARK, _MODULE_MARK))]
        raise_line = plain[-1] if plain else "no output"
        print(f"pipeline_trigger: registry preflight FAILED to reach a verdict "
              f"for program {int(program)} obs {obs_token} (rc="
              f"{proc.returncode}: {raise_line}); proceeding WITHOUT the "
              "registry check (fail-open: a broken pipeline env must not "
              "silence real triggers)", file=sys.stderr)
        return
    consulted = ([""] + _marked(lines, _MODULE_MARK))[-1]
    root = os.path.realpath(pipe_root)
    if not (consulted == root or consulted.startswith(root + os.sep)):
        # the answer came from a registry other than the checkout being
        # submitted against (pip-installed package, inert PYTHONPATH prepend)
        print(f"pipeline_trigger: registry preflight consulted "
              f"{consulted or 'an unidentified registry'}, which is not under "
              f"{pipe_root}; proceeding WITHOUT the registry check (fail-open: "
              "a verdict must come from the checkout being submitted against)",
              file=sys.stderr)
        return
    if proc.returncode == 0:
        return
    verdict = _marked(lines, _VERDICT_MARK)
    detail = verdict[-1] if verdict else ""
    raise NotRegisteredInPipelineError(
        f"program {int(program)} obs {obs_token} ({instrument}) is not "
        f"registered in the pipeline at {pipe_root} (fields.target_for_obsid"
        + (f": {detail}" if detail else "") + ")")


def missing_scripts(pipe_root=None) -> List[str]:
    """The pipeline submitter scripts NOT found under pipe_root (empty = all good)."""
    pipe_root = pipe_root or paths.pipe_root()
    return [rel for rel in (REDUCTION_SBATCH, CATALOGING_CHAIN)
            if not os.path.exists(os.path.join(pipe_root, rel))]


def reduction_step(program, obs, field, filters, pipe_root=None,
                   modules="nrca,nrcb,merged", skip_step12=False,
                   destreak=None) -> dict:
    """The reduction array submission: one array task per filter.

    ``skip_step12=False`` (default) sets SKIP=0: a NEW observation has no *_cal.fits
    yet, so Detector1/Image2 must run.  Set True to reuse existing cal files.

    ``destreak`` mirrors the cataloging step's choice so the pair stays
    consistent: None (default) leaves the driver on its own destreak_policy;
    False adds NO_DESTREAK=1 so an operator forcing the align products gets an
    align reduction to match (the sbatch passthrough for it is the pipeline-side
    half of issue #69; harmless where not yet honored).  True needs nothing --
    the driver already destreaks wherever the policy allows (the policy can only
    turn destreaking OFF, so destreak=True cannot override a policy-off field).
    """
    pipe_root = pipe_root or paths.pipe_root()
    job = f"{field}{int(program)}-o{obs}-reduce"
    export = (f"ALL,PROPOSAL={int(program)},FIELD={obs},"
              f"SKIP={1 if skip_step12 else 0},FILTERS={' '.join(filters)}")
    argv = ["sbatch", "--parsable", f"--job-name={job}",
            f"--array=0-{len(filters) - 1}", f"--export={export}",
            os.path.join(pipe_root, REDUCTION_SBATCH)]
    # MODULES is comma-valued -> environment + --export=ALL (the --export comma trap)
    env = {"MODULES": modules}
    if destreak is False:
        env["NO_DESTREAK"] = "1"
    # SKYMATCH: submit_reduction.sbatch reads it from the environment
    # (--skymatch-method).  Passed through ONLY when the operator sets it; the
    # trigger invents no default.
    # TODO(keflavich/jwst-gc-pipeline#419): once the treasury F480M skymatch
    # policy decision lands (a skymatch_policy module beside destreak_policy, or
    # SKYMATCH=match for program 10678), derive this from the policy probe the
    # same way EACH_SUFFIX is.
    skymatch = os.environ.get("SKYMATCH")
    if skymatch:
        if skymatch not in SKYMATCH_METHODS:
            raise ValueError(
                f"SKYMATCH={skymatch!r} is not a SkyMatchStep skymethod "
                f"{SKYMATCH_METHODS}; the reduction would run Detector1 and "
                "Image2 for hours before Image3 rejects it")
        env["SKYMATCH"] = skymatch
    return dict(name="reduction", argv=argv, env=env)


def cataloging_step(program, obs, field, filters, pipe_root=None,
                    modules="merged", each_suffix=None, destreak=None,
                    policy=None, dep: Optional[str] = DEP_PLACEHOLDER) -> dict:
    """The cataloging chain (env-var driven; DEP gates it on the reduction array).

    EACH_SUFFIX resolution, strongest first:
      1. ``each_suffix`` -- an explicit operator override, used verbatim;
      2. ``destreak`` True/False -- an explicit operator choice of the
         ``destreak_o<obs>_crf`` / ``align_o<obs>_crf`` form;
      3. ``policy`` -- the pipeline's own destreak policy as probed by
         ``pipeline_policy.probe_policy`` (``policy["each_suffix"]``), the same
         value the pipeline's run_pipeline.build_plan derives, so the chain
         globs exactly what the triggered reduction writes (issue #69);
      4. the historical hardcoded fallback ``align_o<obs>_crf`` (the reduction
         ALWAYS runs fix_alignment, and the no-destreak path copies
         ``*_cal.fits`` -> ``*_align.fits`` before Image3, so the per-exposure
         crf products are ``*_align_o<field>_crf.fits``).
    """
    pipe_root = pipe_root or paths.pipe_root()
    if each_suffix:
        suffix = each_suffix
    elif destreak is not None:
        suffix = pipeline_policy.crf_suffix(obs, destreak)
    elif policy:
        suffix = policy["each_suffix"]
    else:
        suffix = pipeline_policy.crf_suffix(obs, False)
    env = {
        "PROPOSAL": str(int(program)),
        "FIELD": obs,
        "TARGET": field,
        "MODULES": modules,
        "EACH_SUFFIX": suffix,
        "FILTERS": " ".join(filters),
    }
    # ZEROFRAME satstar deblend: the pipeline README marks DEBLEND_SATSTARS=1
    # "required for crowded GC fields (gc2211/arches/quintuplet/sgra)"
    # (jwst-gc-pipeline scripts/reduction/README.md), and every treasury tile
    # is inner-CMZ.  It auto-degrades to legacy where a frame lacks a sibling
    # _ramp.fits ZEROFRAME, so it is safe across all GC fields.
    # submit_cataloging.sbatch reads it from the environment (--export=ALL).
    # The non-GC value is the EMPTY string, set explicitly: run_plan composes
    # dict(os.environ, **step["env"]) and the chain exports ALL, so an omitted
    # key lets an ambient DEBLEND_SATSTARS through to
    # `[ -n "$DEBLEND_SATSTARS" ] && DEBLEND_ARG=--deblend-satstars`
    # (submit_cataloging.sbatch:119-121), where ANY non-empty value -- "0"
    # included -- turns the deblend on.  Empty is that check's off value.
    # SCOPE: this takes effect on exposures with no cached satstar catalog,
    # i.e. newly reduced data -- the normal auto-trigger case.  The pipeline
    # caches *_satstar_catalog.fits skip-if-exists and keys that cache on the
    # RECOVERY signature only, not on the deblend flag
    # (crowdsource_catalogs_long.load_or_make_satstar_catalog ->
    # cataloging._satstar_recovery_signature), so a re-catalog of an
    # ALREADY-cataloged field (brick/cloudc/sgrc) reuses its non-deblended
    # caches and the deblend is a no-op there until
    # keflavich/jwst-gc-pipeline#427 lands or those caches are cleared.
    env["DEBLEND_SATSTARS"] = "1" if field in GC_FIELDS else ""
    if dep:
        env["DEP"] = dep
    return dict(name="cataloging-chain", env=env,
                argv=[os.path.join(pipe_root, CATALOGING_CHAIN)])


def _warn_destreak_contradiction(field, obs, destreak, policy) -> List[str]:
    """Warn when an explicit --destreak/--no-destreak fights the probed policy.

    The flag sets EACH_SUFFIX, and the reduction driver consults
    ``destreak_policy.destreaks`` on its own
    (``PipelineRerunNIRCAM-LONG.py:289-291``), so a contradiction means the
    chain globs a suffix the reduction never writes: zero inputs at m1, hours
    after submission (issue #69).  Returns the disagreeing filters.
    """
    disagree = sorted(f for f, v in policy["destreaks"].items()
                      if bool(v) != bool(destreak))
    if not disagree:
        return []
    asked = "--destreak" if destreak else "--no-destreak"
    suffix = pipeline_policy.crf_suffix(obs, destreak)
    print(f"--trigger: WARNING {asked} contradicts the pipeline's destreak "
          f"policy for {field} o{obs} on {', '.join(disagree)} "
          f"(destreak_policy.destreaks() -> {policy['destreaks']}).  The "
          f"reduction driver applies that policy itself, so it writes "
          f"{policy['each_suffix']} for those filters while the chained "
          f"cataloging globs EACH_SUFFIX={suffix}: ZERO inputs at m1 "
          f"(data-qa#69).", file=sys.stderr)
    if not destreak:
        # NO_DESTREAK=1 is the flag's only reduction-side effect and the
        # pipeline has no consumer for it yet (the pipeline-side half of #69),
        # so --no-destreak cannot make the reduction match the align suffix.
        print(f"--trigger: WARNING NO_DESTREAK=1 has no consumer in the "
              f"pipeline yet, so the reduction destreaks {field} regardless.  "
              f"Drop --no-destreak, or pass --each-suffix "
              f"{policy['each_suffix']} to catalog what it writes.",
              file=sys.stderr)
    return disagree


def build_plan(program, obs, field=None, filters=None, pipe_root=None,
               modules="nrca,nrcb,merged", catalog_modules="merged",
               each_suffix=None, destreak=None, skip_step12=False,
               probe=True) -> List[dict]:
    """The full submission sequence for one observation (list of step dicts).

    With no explicit ``each_suffix`` the pipeline's destreak policy is probed.
    On the --auto trigger path (no override at all) it sets EACH_SUFFIX, so the
    chain globs the crf products the reduction really writes.  Under an explicit
    ``destreak`` the flag still wins, and the probe serves the contradiction
    warning (``_warn_destreak_contradiction``).  ``probe=False`` skips the
    subprocess and keeps the historical hardcoded default.
    """
    pipe_root = pipe_root or paths.pipe_root()
    field = field or field_for(program, obs)
    if not field:
        raise ValueError(f"no field mapping for program {program} obs {obs}; "
                         "pass --field or add it to mast_monitor.PROGRAMS")
    if not filters:
        raise ValueError("filters required (e.g. --filters F405N F410M)")
    # registry preflight (issue #68): an unregistered obs must fail IN-PROCESS,
    # before sbatch accepts jobs that will KeyError on-node.  It runs BEFORE the
    # destreak-policy probe (issue #69): both subprocess the pipeline env, and
    # an observation the registry rejects has nothing to probe a policy for.
    # The nircam section is the one asked: every step this plan submits is a
    # NIRCam wrapper (submit_reduction.sbatch / submit_cataloging_chain.sh).
    registry_preflight(program, obs, pipe_root=pipe_root)
    policy = None
    if probe and each_suffix is None:
        # probed even under an explicit --destreak/--no-destreak: the flag still
        # wins the EACH_SUFFIX ladder, and the policy is what tells us the flag
        # contradicts what the reduction will write.  An explicit --each-suffix
        # names the glob outright, so it needs no probe.
        policy = pipeline_policy.probe_policy(field, obs, filters,
                                              pipe_root=pipe_root)
    if policy and destreak is None and len(set(policy["destreaks"].values())) > 1:
        # sickle-style split (SW destreaks, LW does not): the chain takes ONE
        # EACH_SUFFIX, so some filters glob the wrong products -- that field
        # needs per-filter manual submissions (run_pipeline suffix_by_filter)
        print(f"--trigger: WARNING field {field} destreaks per-filter "
              f"({policy['destreaks']}); a single EACH_SUFFIX="
              f"{policy['each_suffix']} is wrong for part of the filter "
              "list -- submit those filters by hand", file=sys.stderr)
    if policy and destreak is not None:
        _warn_destreak_contradiction(field, obs, destreak, policy)
    return [
        reduction_step(program, obs, field, filters, pipe_root=pipe_root,
                       modules=modules, skip_step12=skip_step12,
                       destreak=destreak),
        cataloging_step(program, obs, field, filters, pipe_root=pipe_root,
                        modules=catalog_modules, each_suffix=each_suffix,
                        destreak=destreak, policy=policy),
    ]


_PARSABLE_JOBID_RE = re.compile(r"(?m)^\s*(\d+)(?:;[\w.-]+)?\s*$")
_SUBMITTED_JOBID_RE = re.compile(r"Submitted batch job (\d+)")


def parse_jobids(text) -> List[str]:
    """Every SLURM job id in captured submitter stdout: bare ``--parsable``
    lines (``<jobid>[;cluster]``, the reduction sbatch) plus ``Submitted batch
    job <id>`` lines (the cataloging chain's sbatch calls), deduped in
    first-seen order."""
    ids = [m.group(1) for m in _PARSABLE_JOBID_RE.finditer(text or "")]
    ids += _SUBMITTED_JOBID_RE.findall(text or "")
    return list(dict.fromkeys(ids))


def shell_line(step: dict) -> str:
    """Exact reproducible shell line for a step (env prefix + quoted argv)."""
    prefix = " ".join(f"{k}={shlex.quote(v)}" for k, v in sorted(step["env"].items()))
    cmd = shlex.join(step["argv"])
    return f"{prefix} {cmd}" if prefix else cmd


def run_plan(plan: List[dict]) -> Dict[str, str]:
    """Execute the plan: sbatch the reduction, parse its job id (--parsable), thread
    it into the chain's DEP.  Returns {step name: captured stdout}."""
    results, reduction_jobid = {}, None
    for step in plan:
        env = dict(os.environ, **step["env"])
        if env.get("DEP") == DEP_PLACEHOLDER:
            if not reduction_jobid:
                raise RuntimeError("cataloging DEP placeholder but no reduction "
                                   "job id was captured")
            env["DEP"] = reduction_jobid
        print(f"[{step['name']}] {shell_line(step)}")
        proc = subprocess.run(step["argv"], env=env, capture_output=True, text=True)
        if proc.stdout:
            print(proc.stdout.rstrip())
        if proc.stderr:
            print(proc.stderr.rstrip(), file=sys.stderr)
        if proc.returncode != 0:
            raise RuntimeError(f"{step['name']} failed (rc={proc.returncode}); "
                               "aborting the remaining steps")
        results[step["name"]] = proc.stdout.strip()
        if step["name"] == "reduction":
            # `sbatch --parsable` prints just "<jobid>[;cluster]"
            reduction_jobid = proc.stdout.strip().split(";")[0]
            print(f"[reduction] job id {reduction_jobid} -> DEP for cataloging")
    return results


def submit(program, obs, field=None, filters=None, pipe_root=None, execute=False,
           **kwargs) -> dict:
    """Build + print the plan; submit it when execute=True.  Returns
    ``{"plan": steps, "results": {step: stdout}, "jobids": [ids]}`` --
    results/jobids are empty on dry-run; jobids are parsed from the captured
    sbatch output so the caller (act_trigger) can record them alongside the
    one-shot trigger key."""
    pipe_root = pipe_root or paths.pipe_root()
    plan = build_plan(program, obs, field=field, filters=filters,
                      pipe_root=pipe_root, **kwargs)
    missing = missing_scripts(pipe_root)
    results: Dict[str, str] = {}
    if execute:
        if missing:
            raise FileNotFoundError(
                f"refusing --execute: missing under {pipe_root}: {missing}")
        results = run_plan(plan)
    else:
        print(f"# dry-run (submission sequence for program {program} obs {obs}):")
        for step in plan:
            print(shell_line(step))
        if missing:
            print(f"# WARNING: missing under {pipe_root}: {missing} "
                  "(--execute would refuse)", file=sys.stderr)
    jobids = parse_jobids("\n".join(results.values()))
    return dict(plan=plan, results=results, jobids=jobids)


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--program", required=True, help="JWST program id, e.g. 2221")
    ap.add_argument("--obs", required=True, help="observation number, e.g. 001")
    ap.add_argument("--field", default=None,
                    help="target/field name (default: from the PROGRAMS map)")
    ap.add_argument("--filters", nargs="+", required=True,
                    help="filter list; becomes the array dimension")
    ap.add_argument("--pipe-root", default=None,
                    help="jwst-gc-pipeline checkout (default: $PIPE_ROOT, "
                         f"else {DEFAULT_PIPE_ROOT})")
    ap.add_argument("--modules", default="nrca,nrcb,merged",
                    help="reduction MODULES (default nrca,nrcb,merged)")
    ap.add_argument("--catalog-modules", default="merged",
                    help="cataloging MODULES (default merged)")
    ap.add_argument("--each-suffix", default=None,
                    help="cataloging EACH_SUFFIX override (default: probe the "
                         "pipeline's destreak policy; align_o<obs>_crf if the "
                         "probe fails)")
    ap.add_argument("--destreak", dest="destreak", action="store_true",
                    default=None,
                    help="force cataloging the destreaked products "
                         "(EACH_SUFFIX destreak_o<obs>_crf)")
    ap.add_argument("--no-destreak", dest="destreak", action="store_false",
                    help="force the plain align products (EACH_SUFFIX "
                         "align_o<obs>_crf + NO_DESTREAK=1 to the reduction)")
    ap.add_argument("--no-probe", action="store_true",
                    help="skip the destreak-policy probe subprocess and keep "
                         "the hardcoded EACH_SUFFIX default")
    ap.add_argument("--skip-step12", action="store_true",
                    help="SKIP=1: reuse existing *_cal.fits (default SKIP=0 "
                         "for fresh data)")
    ap.add_argument("--execute", action="store_true",
                    help="really submit via sbatch (default: dry-run print)")
    args = ap.parse_args(argv)

    try:
        submit(args.program, args.obs, field=args.field, filters=args.filters,
               pipe_root=args.pipe_root, execute=args.execute,
               modules=args.modules, catalog_modules=args.catalog_modules,
               each_suffix=args.each_suffix, destreak=args.destreak,
               skip_step12=args.skip_step12, probe=not args.no_probe)
    except (ValueError, FileNotFoundError, RuntimeError) as ex:
        print(f"ERROR: {ex}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
