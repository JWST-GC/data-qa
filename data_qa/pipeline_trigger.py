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
  * the cataloging guard needs PROPOSAL/FIELD/TARGET/EACH_SUFFIX/MODULES together.

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

from .mast_monitor import PROGRAMS

DEFAULT_PIPE_ROOT = "/blue/adamginsburg/adamginsburg/repos/jwst-gc-pipeline"
REDUCTION_SBATCH = "scripts/reduction/submit_reduction.sbatch"
CATALOGING_CHAIN = "scripts/reduction/submit_cataloging_chain.sh"
DEP_PLACEHOLDER = "<REDUCTION_JOBID>"


def missing_scripts(pipe_root) -> List[str]:
    """The pipeline submitter scripts NOT found under pipe_root (empty = all good)."""
    return [rel for rel in (REDUCTION_SBATCH, CATALOGING_CHAIN)
            if not os.path.exists(os.path.join(pipe_root, rel))]


def reduction_step(program, obs, field, filters, pipe_root=DEFAULT_PIPE_ROOT,
                   modules="nrca,nrcb,merged", skip_step12=False) -> dict:
    """The reduction array submission: one array task per filter.

    ``skip_step12=False`` (default) sets SKIP=0: a NEW observation has no *_cal.fits
    yet, so Detector1/Image2 must run.  Set True to reuse existing cal files.
    """
    job = f"{field}{int(program)}-o{obs}-reduce"
    export = (f"ALL,PROPOSAL={int(program)},FIELD={obs},"
              f"SKIP={1 if skip_step12 else 0},FILTERS={' '.join(filters)}")
    argv = ["sbatch", "--parsable", f"--job-name={job}",
            f"--array=0-{len(filters) - 1}", f"--export={export}",
            os.path.join(pipe_root, REDUCTION_SBATCH)]
    # MODULES is comma-valued -> environment + --export=ALL (the --export comma trap)
    return dict(name="reduction", argv=argv, env={"MODULES": modules})


def cataloging_step(program, obs, field, filters, pipe_root=DEFAULT_PIPE_ROOT,
                    modules="merged", each_suffix=None,
                    dep: Optional[str] = DEP_PLACEHOLDER) -> dict:
    """The cataloging chain (env-var driven; DEP gates it on the reduction array)."""
    env = {
        "PROPOSAL": str(int(program)),
        "FIELD": obs,
        "TARGET": field,
        "MODULES": modules,
        "EACH_SUFFIX": each_suffix or f"destreak_o{obs}_crf",
        "FILTERS": " ".join(filters),
    }
    if dep:
        env["DEP"] = dep
    return dict(name="cataloging-chain", env=env,
                argv=[os.path.join(pipe_root, CATALOGING_CHAIN)])


def build_plan(program, obs, field=None, filters=None, pipe_root=DEFAULT_PIPE_ROOT,
               modules="nrca,nrcb,merged", catalog_modules="merged",
               each_suffix=None, skip_step12=False) -> List[dict]:
    """The full submission sequence for one observation (list of step dicts)."""
    field = field or PROGRAMS.get(int(program), {}).get(obs, "")
    if not field:
        raise ValueError(f"no field mapping for program {program} obs {obs}; "
                         "pass --field or add it to mast_monitor.PROGRAMS")
    if not filters:
        raise ValueError("filters required (e.g. --filters F405N F410M)")
    return [
        reduction_step(program, obs, field, filters, pipe_root=pipe_root,
                       modules=modules, skip_step12=skip_step12),
        cataloging_step(program, obs, field, filters, pipe_root=pipe_root,
                        modules=catalog_modules, each_suffix=each_suffix),
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
                    help="cataloging EACH_SUFFIX (default destreak_o<obs>_crf)")
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
               each_suffix=args.each_suffix, skip_step12=args.skip_step12)
    except (ValueError, FileNotFoundError, RuntimeError) as ex:
        print(f"ERROR: {ex}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
