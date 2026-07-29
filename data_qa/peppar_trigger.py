"""Fan a peppar PSF-photometry job out per (filter, detector) for a GC observation.

peppar (Matt Hosek's package, /blue/adamginsburg/adamginsburg/repos/peppar) does single-frame
PSF photometry on the per-exposure ``*_cal.fits``.  Our pipeline already stores those at
``/orange/adamginsburg/jwst/<field>/<FILT>/*_cal.fits`` -- exactly the layout peppar globs --
so this trigger just enumerates the (filter, detector) combos that have cal files and submits
``run_peppar_generic.sbatch`` (env-driven) for each, one SLURM job per detector.

Mirrors ``data_qa.pipeline_trigger``: DRY-RUN by default (prints the exact sbatch commands);
``--execute`` really submits.  In-flight dedup skips a (field, filter, detector) whose job is
already queued/running, so re-firing on every MAST poll is safe.

peppar is per-detector single-frame, so a job is scoped to (field, FILTER, detector) over the
whole filter dir -- not per-obs.  The obs only decides WHEN to (re)trigger the field.

Stdlib-only, so it runs from cron / mast_monitor with just SLURM on PATH.

    python -m data_qa.peppar_trigger --program 2221 --obs 001                 # dry-run, all filt/det
    python -m data_qa.peppar_trigger --program 2221 --obs 001 --execute       # really sbatch
    python -m data_qa.peppar_trigger --program 2221 --obs 001 \
        --filters F212N F405N --dets NRCA1 NRCB1 --execute
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import subprocess
import sys
from typing import Dict, List, Optional

BASE = os.environ.get("QA_BASE", "/orange/adamginsburg/jwst")
PEPPAR_REPO = os.environ.get("PEPPAR_REPO", "/blue/adamginsburg/adamginsburg/repos/peppar")
SBATCH = os.path.join(PEPPAR_REPO, "run_peppar_generic.sbatch")

# a NIRCam detector token in a cal filename: nrca1..nrcb4 (SW) or nrcalong/nrcblong (LW)
_DET_RE = re.compile(r"_(nrc[ab](?:[1-4]|long))_cal\.fits$", re.I)


def field_for(program: str, obs: str, base: str = BASE) -> Optional[str]:
    """Field dir under ``base`` whose FILT subdirs hold this obs's cal files."""
    stem = f"jw{int(program):05d}{obs}"
    for d in sorted(glob.glob(f"{base}/*/")):
        fld = os.path.basename(d.rstrip("/"))
        if glob.glob(f"{base}/{fld}/*/{stem}*_cal.fits"):
            return fld
    return None


def enumerate_filt_det(field: str, base: str = BASE) -> Dict[str, List[str]]:
    """{FILTER: [DET, ...]} for every filter dir under the field that holds ``*_cal.fits``.
    Detectors are read from the filenames (upper-cased)."""
    out: Dict[str, set] = {}
    for fdir in sorted(glob.glob(f"{base}/{field}/*/")):
        filt = os.path.basename(fdir.rstrip("/"))
        if not re.fullmatch(r"F\d{3,4}[WNM]", filt, re.I):     # only filter dirs
            continue
        dets = set()
        for p in glob.glob(f"{fdir}/*_cal.fits"):
            m = _DET_RE.search(os.path.basename(p))
            if m:
                dets.add(m.group(1).upper())
        if dets:
            out[filt.upper()] = sorted(dets)
    return out


def _queued_jobnames() -> set:
    """Names of the user's currently queued/running jobs (for in-flight dedup)."""
    try:
        r = subprocess.run(["squeue", "--me", "-h", "-o", "%j"],
                           capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return set()
    return {ln.strip() for ln in r.stdout.splitlines() if ln.strip()}


def job_name(field: str, filt: str, det: str) -> str:
    return f"peppar-{field}-{filt}-{det}".lower()


def build_jobs(program: str, obs: str, field: Optional[str] = None,
               filters: Optional[List[str]] = None, dets: Optional[List[str]] = None,
               base: str = BASE) -> List[dict]:
    """The list of peppar jobs (one per filter/detector) for this observation."""
    field = field or field_for(program, obs, base)
    if not field:
        raise SystemExit(f"peppar_trigger: no field dir under {base} with cal files for "
                         f"jw{int(program):05d}-o{obs}")
    fd = enumerate_filt_det(field, base)
    want_f = {f.upper() for f in filters} if filters else None
    want_d = {d.upper() for d in dets} if dets else None
    jobs = []
    for filt, det_list in fd.items():
        if want_f and filt not in want_f:
            continue
        data_dir = f"{base}/{field}/{filt}"
        for det in det_list:
            if want_d and det not in want_d:
                continue
            stf_dir = f"{data_dir}/peppar_{det.lower()}"
            jobs.append(dict(field=field, filt=filt, det=det, data_dir=data_dir,
                             stf_dir=stf_dir, name=job_name(field, filt, det)))
    return jobs


def sbatch_argv(job: dict) -> List[str]:
    exports = ",".join([
        "ALL",
        f"PEPPAR_DATA_DIR={job['data_dir']}",
        f"PEPPAR_STF_DIR={job['stf_dir']}",
        f"PEPPAR_FILT={job['filt']}",
        f"PEPPAR_DET={job['det']}",
        f"PEPPAR_REPO={PEPPAR_REPO}",
    ])
    return ["sbatch", "--parsable", f"--job-name={job['name']}", f"--export={exports}", SBATCH]


def submit_all(program: str, obs: str, field: Optional[str] = None,
               filters: Optional[List[str]] = None, dets: Optional[List[str]] = None,
               execute: bool = False, base: str = BASE) -> dict:
    """Build and (optionally) submit every peppar job for this obs.  Shared by the CLI and by
    mast_monitor's --peppar hook.  Returns {jobs, submitted, skipped}.  DRY-RUN prints the
    exact sbatch commands and submits nothing."""
    jobs = build_jobs(program, obs, field=field, filters=filters, dets=dets, base=base)
    field = jobs[0]["field"] if jobs else field
    print(f"peppar_trigger: {len(jobs)} (filter,detector) jobs for {field} "
          f"jw{int(program):05d}-o{obs}  ({'EXECUTE' if execute else 'DRY-RUN'})")
    queued = _queued_jobnames() if execute else set()
    submitted, skipped = 0, 0
    for job in jobs:
        argv_ = sbatch_argv(job)
        if not execute:
            print("  DRY  " + " ".join(argv_))
            continue
        if job["name"] in queued:
            print(f"  SKIP (already queued): {job['name']}")
            skipped += 1
            continue
        try:
            r = subprocess.run(argv_, capture_output=True, text=True, timeout=120)
        except (OSError, subprocess.SubprocessError) as e:
            print(f"  FAILED to sbatch {job['name']}: {e}", file=sys.stderr)
            continue
        if r.returncode != 0:
            print(f"  FAILED {job['name']}: {r.stderr.strip()}", file=sys.stderr)
            continue
        print(f"  submitted {job['name']} -> job {r.stdout.strip()}")
        submitted += 1
    if execute:
        print(f"peppar_trigger: submitted {submitted}, skipped {skipped} (in-flight)")
    return dict(jobs=len(jobs), submitted=submitted, skipped=skipped)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--program", required=True)
    ap.add_argument("--obs", required=True)
    ap.add_argument("--field", default=None, help="override on-disk field key (else auto-detected)")
    ap.add_argument("--filters", nargs="+", default=None, help="subset of filters (default: all)")
    ap.add_argument("--dets", nargs="+", default=None, help="subset of detectors (default: all)")
    ap.add_argument("--execute", action="store_true",
                    help="actually sbatch (default: dry-run, print commands only)")
    ap.add_argument("--base", default=BASE)
    args = ap.parse_args(argv)

    if not os.path.exists(SBATCH):
        print(f"peppar_trigger: sbatch script not found: {SBATCH}", file=sys.stderr)
        return 2

    try:
        res = submit_all(args.program, args.obs, field=args.field, filters=args.filters,
                         dets=args.dets, execute=args.execute, base=args.base)
    except SystemExit as e:
        print(str(e), file=sys.stderr)
        return 1
    return 0 if res["jobs"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
