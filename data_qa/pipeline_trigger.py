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
import shlex
import subprocess
import sys
from typing import Dict, List, Optional

from . import pipeline_policy
from .mast_monitor import GC_FIELDS, field_for

DEFAULT_PIPE_ROOT = "/blue/adamginsburg/adamginsburg/repos/jwst-gc-pipeline"
REDUCTION_SBATCH = "scripts/reduction/submit_reduction.sbatch"
CATALOGING_CHAIN = "scripts/reduction/submit_cataloging_chain.sh"
DEP_PLACEHOLDER = "<REDUCTION_JOBID>"


def missing_scripts(pipe_root) -> List[str]:
    """The pipeline submitter scripts NOT found under pipe_root (empty = all good)."""
    return [rel for rel in (REDUCTION_SBATCH, CATALOGING_CHAIN)
            if not os.path.exists(os.path.join(pipe_root, rel))]


def reduction_step(program, obs, field, filters, pipe_root=DEFAULT_PIPE_ROOT,
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
        env["SKYMATCH"] = skymatch
    return dict(name="reduction", argv=argv, env=env)


def cataloging_step(program, obs, field, filters, pipe_root=DEFAULT_PIPE_ROOT,
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
    if each_suffix:
        suffix = each_suffix
    elif destreak is not None:
        suffix = f"{'destreak' if destreak else 'align'}_o{obs}_crf"
    elif policy:
        suffix = policy["each_suffix"]
    else:
        suffix = f"align_o{obs}_crf"
    env = {
        "PROPOSAL": str(int(program)),
        "FIELD": obs,
        "TARGET": field,
        "MODULES": modules,
        "EACH_SUFFIX": suffix,
        "FILTERS": " ".join(filters),
    }
    if field in GC_FIELDS:
        # ZEROFRAME satstar deblend: the pipeline README marks DEBLEND_SATSTARS=1
        # "required for crowded GC fields (gc2211/arches/quintuplet/sgra)"
        # (jwst-gc-pipeline scripts/reduction/README.md), and every treasury tile
        # is inner-CMZ.  It auto-degrades to legacy where a frame lacks a sibling
        # _ramp.fits ZEROFRAME, so it is safe across all GC fields.
        # submit_cataloging.sbatch reads it from the environment (--export=ALL).
        env["DEBLEND_SATSTARS"] = "1"
    if dep:
        env["DEP"] = dep
    return dict(name="cataloging-chain", env=env,
                argv=[os.path.join(pipe_root, CATALOGING_CHAIN)])


def build_plan(program, obs, field=None, filters=None, pipe_root=DEFAULT_PIPE_ROOT,
               modules="nrca,nrcb,merged", catalog_modules="merged",
               each_suffix=None, destreak=None, skip_step12=False,
               probe=True) -> List[dict]:
    """The full submission sequence for one observation (list of step dicts).

    With no explicit ``each_suffix``/``destreak`` override (the --auto trigger
    path), the pipeline's destreak policy is probed so EACH_SUFFIX names the crf
    products the reduction really writes; ``probe=False`` skips the subprocess
    and keeps the historical hardcoded default.
    """
    field = field or field_for(program, obs)
    if not field:
        raise ValueError(f"no field mapping for program {program} obs {obs}; "
                         "pass --field or add it to mast_monitor.PROGRAMS")
    if not filters:
        raise ValueError("filters required (e.g. --filters F405N F410M)")
    policy = None
    if probe and each_suffix is None and destreak is None:
        policy = pipeline_policy.probe_policy(field, obs, filters,
                                              pipe_root=pipe_root)
        if policy and len(set(policy["destreaks"].values())) > 1:
            # sickle-style split (SW destreaks, LW does not): the chain takes ONE
            # EACH_SUFFIX, so some filters glob the wrong products -- that field
            # needs per-filter manual submissions (run_pipeline suffix_by_filter)
            print(f"--trigger: WARNING field {field} destreaks per-filter "
                  f"({policy['destreaks']}); a single EACH_SUFFIX="
                  f"{policy['each_suffix']} is wrong for part of the filter "
                  "list -- submit those filters by hand", file=sys.stderr)
    return [
        reduction_step(program, obs, field, filters, pipe_root=pipe_root,
                       modules=modules, skip_step12=skip_step12,
                       destreak=destreak),
        cataloging_step(program, obs, field, filters, pipe_root=pipe_root,
                        modules=catalog_modules, each_suffix=each_suffix,
                        destreak=destreak, policy=policy),
    ]


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
           **kwargs) -> List[dict]:
    """Build + print the plan; submit it when execute=True.  Returns the plan."""
    pipe_root = pipe_root or DEFAULT_PIPE_ROOT
    plan = build_plan(program, obs, field=field, filters=filters,
                      pipe_root=pipe_root, **kwargs)
    missing = missing_scripts(pipe_root)
    if execute:
        if missing:
            raise FileNotFoundError(
                f"refusing --execute: missing under {pipe_root}: {missing}")
        run_plan(plan)
    else:
        print(f"# dry-run (submission sequence for program {program} obs {obs}):")
        for step in plan:
            print(shell_line(step))
        if missing:
            print(f"# WARNING: missing under {pipe_root}: {missing} "
                  "(--execute would refuse)", file=sys.stderr)
    return plan


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--program", required=True, help="JWST program id, e.g. 2221")
    ap.add_argument("--obs", required=True, help="observation number, e.g. 001")
    ap.add_argument("--field", default=None,
                    help="target/field name (default: from the PROGRAMS map)")
    ap.add_argument("--filters", nargs="+", required=True,
                    help="filter list; becomes the array dimension")
    ap.add_argument("--pipe-root", default=DEFAULT_PIPE_ROOT,
                    help=f"jwst-gc-pipeline checkout (default {DEFAULT_PIPE_ROOT})")
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
