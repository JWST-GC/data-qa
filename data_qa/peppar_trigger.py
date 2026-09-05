"""Fan a peppar PSF-photometry job out per (filter, detector) for a GC observation.

peppar (Matt Hosek's package, /blue/adamginsburg/adamginsburg/repos/peppar) does single-frame
PSF photometry on the per-exposure ``*_cal.fits``.  A "cal file" is one JWST stage-2
calibrated product: a single exposure of a single detector in a single filter.  This trigger
enumerates the (filter, detector) combos that have cal files under
``/orange/adamginsburg/jwst/<field>/<FILT>/`` and submits ``run_peppar_generic.sbatch``
(env-driven) for each, one SLURM job per detector.

Two on-disk layouts hold the cal files (issue #73).  The reduction writes
``<field>/<FILT>/pipeline/*_cal.fits`` (``PipelineRerunNIRCAM-LONG.py`` outputs to
``{basepath}{filtername}/pipeline/``); older reductions left them flat at
``<field>/<FILT>/*_cal.fits``, and many fields carry BOTH.  Census 2026-08-16 over
``/orange/adamginsburg/jwst`` (flat/pipeline), FILTER-dir scope -- what
``enumerate_filt_det`` sees, so what decides jobs --

  pipeline-only: arches 0/120, cloudc 0/556, cloudef 0/480, quintuplet 0/120, sgra 0/216,
                 sgrb2 0/1500, sgrc 0/242, sickle 0/309, w51 0/560, wd1 0/696, wd2 0/328
  both layouts:  brick 720/1296, gc2211 740/740, m92 80/80, ngc6334 1250/1250

``field_for`` resolves the field by cal-file COUNT, since a field dir can hold a handful of
another observation's frames: jw02221 obs 001 reduces to brick and obs 002 to cloudc, yet
brick also carries 96 obs-002 frames and cloudc 72 obs-001 ones, so first-match answered
brick for both and left cloudc's 480 obs-002 files unreachable.  It matches ANY subdir, so
in its wider scope brick becomes 1296/1296 (``dolphot/``
adds 576 flat) and three more fields hold both -- m4 150/150 and ngc6397 120/120 (their cal
files sit in ``F150W2``/``F322W2``) and w51 64/560 (64 flat in ``dolphot/``) -- for SEVEN
both-layout fields.  ``field_for`` also sees the flat-only jw02731 760/0, jw02732 1058/0 and
jwebbinar_prep 6/0, which no filter-dir census lists.  The filter-dir census above predates
issue #82: m4's and ngc6397's cal files sit in the WIDE-filter dirs ``F150W2``/``F322W2``,
which ``_FILT_RE`` now accepts, so both fields enumerate 10 (filter, detector) jobs each.

Across the 3060 basename pairs present in both layouts, 1600 (52%, all of m4 / m92 /
ngc6334 / ngc6397) are HARDLINKS -- one file with two names, where the choice of layout
changes nothing.  The other 1460 (brick and gc2211) are separate files, and the ``pipeline/``
copy is the newer one in all 1460: e.g.
``brick/F182M/jw02221001001_07101_00001_nrca1_cal.fits`` is 117538560 B at both paths, dated
2022-12-31 flat against 2024-07-17 under ``pipeline/``, with differing md5 over the first MB.
So preferring ``pipeline/`` selects the newer reduction where the two differ at all.

Discovery unions the two layouts, counting a basename present in both once.  The order
``_cal_files`` returns is immaterial -- both callers use the result as a set of detectors.
The copy that actually reaches peppar is chosen by ``cal_data_dir``, which resolves
``PEPPAR_DATA_DIR`` per (filter, DETECTOR): the runner globs ``PEPPAR_DATA_DIR`` flat, so it
sees one layout, and a mid-migration filter dir can hold some detectors under ``pipeline/``
while another is still flat.

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
# PEPPAR_REPO is the peppar PACKAGE checkout (for `from peppar import peppar`), imported via
# PYTHONPATH in the sbatch.  The runner + sbatch themselves are OURS -- they live in this repo
# (scripts/peppar), NOT in the peppar package.
PEPPAR_REPO = os.environ.get("PEPPAR_REPO", "/blue/adamginsburg/adamginsburg/repos/peppar")
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PEPPAR_SCRIPTS = os.path.join(_REPO_ROOT, "scripts", "peppar")
SBATCH = os.path.join(_PEPPAR_SCRIPTS, "run_peppar_generic.sbatch")
RUNNER = os.path.join(_PEPPAR_SCRIPTS, "run_peppar_generic.py")

# A NIRCam detector token in a cal filename.  SW (short-wavelength channel) detectors are
# nrca1..nrcb4; the LW (long-wavelength) channel is one detector per module, nrcalong /
# nrcblong.
_DET_RE = re.compile(r"_(nrc[ab](?:[1-4]|long))_cal\.fits$", re.I)

# A filter-dir name.  The trailing "2?" admits the NIRCam WIDE filters F150W2 and F322W2
# (issue #82): m4 and ngc6397 keep every cal file under F150W2/ and F322W2/, so without it
# both fields resolved in ``field_for`` and then enumerated nothing.  Those products are
# genuine wide-filter imaging -- the cal headers read FILTER='F150W2'/'F322W2' with
# PUPIL='CLEAR', so the dir name IS the bandpass.  Reaching them was necessary but not
# sufficient: both bandpasses ALSO needed two fixes in ``scripts/peppar/run_peppar_generic.py``
# (the F150W2 PSF grid cannot be built at stpsf's default spectral sampling, and peppar
# keys the LW detectors NRCA5/NRCB5) before a job emitted here could write a catalog.
# ``peppar.setup_filter_props()`` carrying both filters says only that the zeropoint/FWHM
# table has a row -- it does not exercise either of those paths.  Live scope of the widening
# (census 2026-08-25 over /orange/adamginsburg/jwst): m4 F150W2 120 cal files / 8 SW
# detectors + F322W2 30 / 2 LW, ngc6397 F150W2 96 / 8 + F322W2 24 / 2, and no other field
# holds a wide-filter dir -- 20 new (filter, detector) jobs.  All 20 are manual-CLI only:
# proposal 1979 (m4, ngc6397) is not in ``mast_monitor.PROGRAMS``, so the auto fan-out
# never reaches them.
_FILT_RE = re.compile(r"F\d{3,4}[WNM]2?", re.I)


def _cal_files(fdir: str, stem: str = "") -> List[str]:
    """Cal files under one filter dir, across both layouts: ``pipeline/`` first, then the
    legacy flat dir.  A file present in both (same basename) is counted once, preferring
    the ``pipeline/`` copy."""
    fdir = fdir.rstrip("/")
    pipe = sorted(glob.glob(f"{fdir}/pipeline/{stem}*_cal.fits"))
    seen = {os.path.basename(p) for p in pipe}
    flat = sorted(p for p in glob.glob(f"{fdir}/{stem}*_cal.fits")
                  if os.path.basename(p) not in seen)
    return pipe + flat


def cal_data_dir(fdir: str, det: Optional[str] = None) -> str:
    """The single directory the peppar runner should glob for this filter dir, resolved
    per DETECTOR when ``det`` is given.

    The runner globs ``PEPPAR_DATA_DIR`` flat
    (``scripts/peppar/run_peppar_generic.py::run_peppar_script``), so it sees ONE layout
    while discovery enumerates the union of both.  Two ways that bites, both fixed here by
    choosing the layout that serves the most cal files for the scope asked about (ties go
    to ``pipeline/``, the newer reduction):

    * a detector present only in the flat dir, handed ``{FILT}/pipeline``, clears the
      runner's "no images" guard on its SIBLINGS' files and then dies in
      ``peppar.setup_dict_images_for_run``, which indexes ``dict_images[filt][det]``
      with no guard -- a KeyError after the job has taken its queue slot;
    * a detector whose exposures are SPLIT across the two layouts runs quietly on
      whichever subset the chosen dir holds.  Choosing the larger side removes the loss
      whenever one layout is a superset of the other (the ordinary half-migrated shape);
      a genuinely disjoint split still loses the remainder, so that case warns on stderr
      instead of passing unremarked.

    What this rule maximises is the file COUNT, which can select the older reduction: the
    module docstring's ``pipeline/`` preference holds where both layouts hold the same
    basename (the dedup in ``_cal_files``, and the tie here), while a half-migrated
    detector whose flat side is bigger is served the flat copies, older on brick and
    gc2211.  Serving the whole detector beats serving the newer part of it, and the
    disjoint case says so on stderr.  On the live archive flat is a subset of
    ``pipeline/`` by basename in all 539 (filter, detector) pairs, so ``pipeline/`` wins
    wherever it exists.

    Without ``det`` the answer is the filter-level one, by the same rule."""
    fdir = fdir.rstrip("/")
    pat = f"*_{det.lower()}_cal.fits" if det else "*_cal.fits"
    pipe = {os.path.basename(p) for p in glob.glob(f"{fdir}/pipeline/{pat}")}
    flat = {os.path.basename(p) for p in glob.glob(f"{fdir}/{pat}")}
    if not pipe:
        return fdir
    pipe_wins = len(pipe) >= len(flat)
    chosen, served = (f"{fdir}/pipeline", pipe) if pipe_wins else (fdir, flat)
    union = pipe | flat
    if len(served) < len(union):
        print(f"peppar_trigger: WARNING {fdir}"
              f"{' ' + det.upper() if det else ''} splits across both layouts; "
              f"{chosen} serves {len(served)} of {len(union)} cal files "
              f"({len(union) - len(served)} exist only in the other layout)",
              file=sys.stderr)
    return chosen


def field_cal_counts(program: str, obs: str, base: str = BASE) -> Dict[str, int]:
    """{field: number of this obs's cal files under it} over every field dir in ``base``,
    counting both layouts and a basename present in both once.  Fields holding none are
    omitted."""
    stem = f"jw{int(program):05d}{obs}"
    counts: Dict[str, int] = {}
    for d in sorted(glob.glob(f"{base}/*/")):
        fld = os.path.basename(d.rstrip("/"))
        n = sum(len(_cal_files(fdir, stem))
                for fdir in sorted(glob.glob(f"{base}/{fld}/*/")))
        if n:
            counts[fld] = n
    return counts


def field_for(program: str, obs: str, base: str = BASE) -> Optional[str]:
    """Field dir under ``base`` holding the MOST of this obs's cal files (in either the
    ``pipeline/`` or the legacy flat layout); ``None`` when no field holds any.

    Several field dirs hold a few frames of an observation that reduces into a DIFFERENT
    field, so "the first field that matches" answers the wrong one.  jw02221 is the live
    case: obs 001 reduces to brick and obs 002 to cloudc, while brick's LW filter dirs
    also carry 96 obs-002 frames and cloudc's F2550W carries 72 obs-001 frames.  Returning
    the first sorted match gave brick for BOTH observations, so cloudc's 480 obs-002 cal
    files -- 30 (filter, detector) jobs -- were unreachable from ``--peppar`` for every
    observation of the only program that reaches them, silently and with a populated
    ``data_dir``.  Counting answers brick for 2221/001 (1056 against 72) and cloudc for
    2221/002 (480 against 144), and reproduces ``mast_monitor.PROGRAMS`` -- the reduction's
    own field registry -- for every registered observation that resolves on disk.

    A contested observation warns on stderr with the per-field counts, since a field
    holding a minority of another observation's frames is a misfiling worth seeing.
    An exact tie resolves alphabetically, so the answer is stable across polls.

    ``mast_monitor.act_peppar`` passes ``field=`` from that registry, so this is the
    resolver for the CLI and for any caller without a registry entry.  It reads every
    field dir (~1.6 s over the 55 fields on the live archive), which suits the poll
    cadence the trigger runs at."""
    counts = field_cal_counts(program, obs, base)
    if not counts:
        return None
    best = max(sorted(counts), key=counts.get)
    if len(counts) > 1:
        rest = ", ".join(f"{f} {counts[f]}" for f in sorted(counts) if f != best)
        print(f"peppar_trigger: WARNING jw{int(program):05d}-o{obs} has cal files under "
              f"{len(counts)} fields; running {best} ({counts[best]} cal files), leaving "
              f"{rest}", file=sys.stderr)
    return best


def enumerate_filt_det(field: str, base: str = BASE) -> Dict[str, List[str]]:
    """{FILTER: [DET, ...]} for every filter dir under the field that holds ``*_cal.fits``
    (in either layout).  Detectors are read from the filenames (upper-cased)."""
    out: Dict[str, set] = {}
    for fdir in sorted(glob.glob(f"{base}/{field}/*/")):
        filt = os.path.basename(fdir.rstrip("/"))
        if not _FILT_RE.fullmatch(filt):                       # only filter dirs
            continue
        dets = set()
        for p in _cal_files(fdir):
            m = _DET_RE.search(os.path.basename(p))
            if m:
                dets.add(m.group(1).upper())
        if dets:
            out[filt.upper()] = sorted(dets)
    return out


def nonfilter_cal_dirs(field: str, base: str = BASE) -> List[str]:
    """Subdir names under the field that hold ``*_cal.fits`` (either layout) and are NOT
    filter dirs by ``_FILT_RE``.  These are the reason a field can resolve in ``field_for``
    -- which matches any subdir -- and then enumerate nothing: ``dolphot/`` (brick, w51).
    The wide-filter dirs ``F150W2``/``F322W2`` (m4, ngc6397) used to land here too and now
    enumerate normally (issue #82).  Named in the no-jobs error so it reports the pattern
    rather than reading as "no data on disk"."""
    out = []
    for fdir in sorted(glob.glob(f"{base}/{field}/*/")):
        name = os.path.basename(fdir.rstrip("/"))
        if not _FILT_RE.fullmatch(name) and _cal_files(fdir):
            out.append(name)
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
        filt_dir = f"{base}/{field}/{filt}"
        for det in det_list:
            if want_d and det not in want_d:
                continue
            # per DETECTOR: the layout that actually holds THIS detector's cal files
            data_dir = cal_data_dir(filt_dir, det)
            # outputs stay at the filter level in both layouts (they are products, and
            # the reduction owns pipeline/)
            stf_dir = f"{filt_dir}/peppar_{det.lower()}"
            jobs.append(dict(field=field, filt=filt, det=det, data_dir=data_dir,
                             stf_dir=stf_dir, name=job_name(field, filt, det)))
    if not jobs and not (filters or dets):
        msg = (f"peppar_trigger: field {field} under {base} has no filter dir "
               f"holding NIRCam *_cal.fits (checked both layouts)")
        other = nonfilter_cal_dirs(field, base)
        if other:
            msg += (f"; these subdirs DO hold cal files and are skipped because their "
                    f"name fails the filter-dir pattern {_FILT_RE.pattern}: "
                    + ", ".join(other))
        raise SystemExit(msg)
    return jobs


def sbatch_argv(job: dict) -> List[str]:
    exports = ",".join([
        "ALL",
        f"PEPPAR_DATA_DIR={job['data_dir']}",
        f"PEPPAR_STF_DIR={job['stf_dir']}",
        f"PEPPAR_FILT={job['filt']}",
        f"PEPPAR_DET={job['det']}",
        f"PEPPAR_REPO={PEPPAR_REPO}",      # peppar package checkout (PYTHONPATH)
        f"PEPPAR_RUNNER={RUNNER}",         # our runner (in this repo)
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
