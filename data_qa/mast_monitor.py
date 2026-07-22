"""Poll MAST for new / newly-released / newly-calibrated JWST GC observations.

Stateful monitor: each run queries MAST per program, diffs against a state file, and
reports events.  Default is report-only (print events, touch nothing); every side
effect is individually gated:

  --commit-state   update the state file (otherwise the same events re-report)
  --download       fetch the new observation's products via data_qa.retrieve_data
  --trigger        build the reduction+cataloging submission (data_qa.pipeline_trigger)
  --report         comment the events on the per-observation QA issue (status_report)
  --execute        actually do it (download / sbatch / post); without it the actions
                   above are dry-run prints
  --auto           fully automatic: --download --trigger --report --commit-state
                   --execute, gated ONLY by available file space (--min-free-tb,
                   checked against the --download-dir filesystem); below the
                   threshold it downgrades to report-only with a LOW DISK warning

Events:
  NEW_OBSERVATION  obs_id not previously in the state file
  NEWLY_RELEASED   t_obs_release passed since the last committed poll
  CALIB_LEVEL_UP   calib_level increased (e.g. 2 -> 3: mosaics available)

Anonymous MAST metadata queries work without auth; ``~/.mast_api_token`` is used if
present (exclusive-access programs).  astroquery is imported lazily so the module
stays stdlib-importable.

Usage:
    python -m data_qa.mast_monitor --limit-programs 2221            # poll + print
    python -m data_qa.mast_monitor --json                           # machine-readable
    python -m data_qa.mast_monitor --commit-state                   # accept as seen
    python -m data_qa.mast_monitor --download --trigger --report    # dry-run actions
    python -m data_qa.mast_monitor --download --trigger --report --execute
    python -m data_qa.mast_monitor --auto --min-free-tb 5 \\
        --download-dir /orange/adamginsburg/jwst/ops/downloads
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import shutil
import sys
import time
from typing import Dict, List

# Monitored programs: program id -> {obs number -> release field}.  Mirrors the
# reduction's field_to_reg_mapping (PipelineRerunNIRCAM-LONG.py) restricted to the
# GC-treasury/QA programs; field names match data_qa.observations.FIELDS keys where
# released (cloudef/sgra/ngc6334 have no public release page yet).
#
# ================================ PRIORITY WATCH =================================
# 10678 is THE GC Treasury program: ~1668 planned observations tiling the GC as
# GC_<n> targets (NIRCam F212N;F480M + MIRI F770W).  As of 2026-07-22 EVERY
# observation is calib_level=-1 / unreleased, so ANY event from 10678 is the first
# sign of treasury data arriving.  Its obs numbers cannot be enumerated in advance
# (they land as the tiles execute), so it maps program-wide to the 'gc-treasury'
# field via field_for(); the GC_<n> tile name rides each event as 'tile'.
# =================================================================================
TREASURY_PROGRAM = 10678
TREASURY_FIELD = "gc-treasury"

PROGRAMS: Dict[int, Dict[str, str]] = {
    TREASURY_PROGRAM: {},   # GC Treasury: every obs -> gc-treasury (see field_for)
    2221: {"001": "brick", "002": "cloudc"},
    1182: {"004": "brick", "002": "w51"},           # w51 = broadband
    2211: {"023": "gc2211", "028": "gc2211", "046": "gc2211",
           "049": "gc2211", "050": "gc2211"},
    4147: {"012": "sgrc"},
    5365: {"001": "sgrb2"},
    3958: {"001": "sickle", "002": "sickle", "007": "sickle"},
    2092: {"002": "cloudef", "005": "cloudef"},
    1939: {"001": "sgra"},
    1905: {"001": "wd1", "003": "wd1"},
    3523: {"003": "wd2", "005": "wd2"},
    6778: {"001": "ngc6334"},
    7213: {"001": "ngc6334"},
}

DEFAULT_STATE = "/orange/adamginsburg/jwst/ops/mast_state.json"

# Columns kept from the MAST observation table (t_* are MJD floats).
MONITOR_COLUMNS = ("obs_id", "t_max", "t_obs_release", "calib_level",
                   "instrument_name", "filters", "target_name")

# calib-level-3 products: jw02221-o001_t001_nircam_...; level<=2: jw02221001001_...
_OBS_DASH_RE = re.compile(r"^jw(\d{5})-o(\d{3})")
_OBS_FLAT_RE = re.compile(r"^jw(\d{5})(\d{3})\d{3}")


def obsnum_from_obs_id(obs_id: str) -> str:
    """'jw02221-o001_t001_...' or 'jw02221001001_...' -> '001' ('' if unparseable)."""
    m = _OBS_DASH_RE.match(obs_id) or _OBS_FLAT_RE.match(obs_id)
    return m.group(2) if m else ""


def field_for(program, obsnum: str) -> str:
    if int(program) == TREASURY_PROGRAM:
        # Treasury tiles (GC_<n>) all reduce into the one gc-treasury field; the
        # per-obs tile name is carried on the event as 'tile' instead.
        return TREASURY_FIELD
    return PROGRAMS.get(int(program), {}).get(obsnum, "")


def now_mjd() -> float:
    return time.time() / 86400.0 + 40587.0


def mjd_to_iso(mjd) -> str:
    if mjd is None:
        return "?"
    ts = (float(mjd) - 40587.0) * 86400.0
    return datetime.datetime.fromtimestamp(
        ts, tz=datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


# ------------------------------------------------------------------------------ MAST
def mast_login_if_token(token_path="~/.mast_api_token") -> bool:
    """Log in to MAST if a token file exists; anonymous otherwise (fine for public
    metadata).  Returns True when logged in."""
    path = os.path.expanduser(token_path)
    if not os.path.exists(path):
        return False
    with open(path) as fh:
        token = fh.read().strip()
    if not token:
        return False
    os.environ["MAST_API_TOKEN"] = token
    from astroquery.exceptions import LoginError
    from astroquery.mast import Observations
    try:
        Observations.login(token)
    except (LoginError, ValueError, ConnectionError, OSError) as ex:
        print(f"MAST login failed ({ex.__class__.__name__}: {ex}); "
              "continuing anonymously", file=sys.stderr)
        return False
    return True


def query_program(program) -> List[dict]:
    """MAST observations for one program as plain-python dict rows (lazy astroquery)."""
    from astroquery.mast import Observations
    tbl = Observations.query_criteria(proposal_id=str(int(program)),
                                      obs_collection="JWST")
    cols = [c for c in MONITOR_COLUMNS if c in tbl.colnames]
    rows = []
    for r in tbl:
        row = {}
        for c in cols:
            v = r[c]
            if c in ("t_max", "t_obs_release"):
                try:
                    row[c] = float(v)
                except (TypeError, ValueError):
                    row[c] = None
            elif c == "calib_level":
                try:
                    row[c] = int(v)
                except (TypeError, ValueError):
                    row[c] = None
            else:
                row[c] = None if v is None else str(v)
        rows.append(row)
    return rows


# ------------------------------------------------------------------------------ state
def summarize(rows: List[dict], poll_mjd: float) -> Dict[str, dict]:
    """Query rows -> per-obs_id state records (what the state file stores)."""
    out = {}
    for row in rows:
        rel = row.get("t_obs_release")
        out[row["obs_id"]] = {
            "calib_level": row.get("calib_level"),
            "t_obs_release": rel,
            "t_max": row.get("t_max"),
            "released": bool(rel is not None and rel <= poll_mjd),
            "instrument_name": row.get("instrument_name"),
            "filters": row.get("filters"),
            "target_name": row.get("target_name"),
        }
    return out


def diff_events(program, old_obs: Dict[str, dict], new_obs: Dict[str, dict]):
    """Compare the previous state records to the fresh ones -> event dicts."""
    events = []
    for obs_id, rec in sorted(new_obs.items()):
        obsnum = obsnum_from_obs_id(obs_id)
        # Treasury events carry the GC_<n> tile name (the MAST target_name)
        tile = (rec.get("target_name")
                if int(program) == TREASURY_PROGRAM else None)
        base = dict(program=int(program), obs_id=obs_id, obsnum=obsnum,
                    field=field_for(program, obsnum), tile=tile,
                    calib_level=rec.get("calib_level"),
                    released=rec.get("released"),
                    t_obs_release=rec.get("t_obs_release"),
                    instrument_name=rec.get("instrument_name"),
                    filters=rec.get("filters"),
                    target_name=rec.get("target_name"))
        prev = old_obs.get(obs_id)
        if prev is None:
            events.append(dict(event="NEW_OBSERVATION", **base))
            continue
        if rec.get("released") and not prev.get("released", False):
            events.append(dict(event="NEWLY_RELEASED", **base))
        if (rec.get("calib_level") or 0) > (prev.get("calib_level") or 0):
            events.append(dict(event="CALIB_LEVEL_UP",
                               previous_calib_level=prev.get("calib_level"), **base))
    return events


def load_state(path) -> dict:
    """State file -> dict; a missing file is an empty (first-run) state.
    A CORRUPT file raises (better loud than treating everything as new)."""
    if not os.path.exists(path):
        return {"version": 1, "programs": {}}
    with open(path) as fh:
        return json.load(fh)


def save_state(path, state: dict):
    """Atomic write (tmp + rename); auto-creates the parent directory."""
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}"
    with open(tmp, "w") as fh:
        json.dump(state, fh, indent=1, sort_keys=True)
        fh.write("\n")
    os.replace(tmp, path)


def format_event(ev: dict) -> str:
    extra = (f" (level {ev['previous_calib_level']} -> {ev['calib_level']})"
             if ev["event"] == "CALIB_LEVEL_UP"
             else f" calib={ev['calib_level']} release={mjd_to_iso(ev['t_obs_release'])}")
    field = ev["field"] or "?unmapped?"
    tile = f" tile={ev['tile']}" if ev.get("tile") else ""
    return (f"{ev['event']:16s} {ev['program']} {ev['obs_id']} "
            f"[field={field}{tile} filters={ev.get('filters') or '?'}]" + extra)


def _group_by_obs(events):
    """Events -> {(program, obsnum): [events]} for per-observation actions
    (obs-level only: skips events whose obsnum could not be parsed)."""
    grouped = {}
    for ev in events:
        if ev["obsnum"]:
            grouped.setdefault((ev["program"], ev["obsnum"]), []).append(ev)
    return grouped


# -------------------------------------------------------------------------- disk gate
DEFAULT_MIN_FREE_TB = 5.0
DEFAULT_DOWNLOAD_DIR = "./data"          # matches retrieve_data.retrieve's default


def free_terabytes(path) -> float:
    """Free space (TB, 1e12 bytes) on the filesystem holding ``path``.  Climbs to
    the nearest EXISTING parent so a not-yet-created download dir still reports
    its destination filesystem."""
    p = os.path.abspath(path)
    while not os.path.exists(p):
        parent = os.path.dirname(p)
        if parent == p:
            break
        p = parent
    return shutil.disk_usage(p).free / 1e12


def disk_gate(download_dir, min_free_tb=DEFAULT_MIN_FREE_TB):
    """The ONLY gate on --auto: (ok, free_tb, message).  Below-threshold means
    auto must downgrade to report-only (no download/trigger/commit-state)."""
    free_tb = free_terabytes(download_dir)
    if free_tb >= min_free_tb:
        return True, free_tb, (f"disk gate OK: {free_tb:.1f} TB free at "
                               f"{download_dir} (threshold {min_free_tb:.1f} TB)")
    return False, free_tb, (
        f"LOW DISK: only {free_tb:.1f} TB free on the filesystem of "
        f"{download_dir} (< {min_free_tb:.1f} TB threshold) -- auto mode "
        f"downgraded to report-only; NOT downloading, NOT triggering, NOT "
        f"committing state (events will re-fire once space is freed)")


# ---------------------------------------------------------------------------- actions
def act_download(events, execute=False, download_dir=DEFAULT_DOWNLOAD_DIR):
    from .retrieve_data import retrieve   # lazy: astroquery
    for (program, obsnum), evs in sorted(_group_by_obs(events).items()):
        print(f"--download: program {program} obs {obsnum} "
              f"({len(evs)} event(s); dry_run={not execute})")
        retrieve(program, obsnum, product_type=("uncal", "i2d"),
                 download_dir=download_dir, dry_run=not execute)


def act_trigger(events, execute=False, pipe_root=None):
    from .pipeline_trigger import submit   # stdlib-only
    for (program, obsnum), evs in sorted(_group_by_obs(events).items()):
        field = evs[0]["field"]
        if not field:
            print(f"--trigger: SKIP program {program} obs {obsnum}: no field mapping "
                  "(add it to mast_monitor.PROGRAMS)", file=sys.stderr)
            continue
        filters = sorted({f for ev in evs for f in (ev.get("filters") or "").split(";")
                          if f and f != "?"})
        if not filters:
            print(f"--trigger: SKIP program {program} obs {obsnum}: no filters known",
                  file=sys.stderr)
            continue
        submit(program=program, obs=obsnum, field=field, filters=filters,
               pipe_root=pipe_root, execute=execute)


def act_report(events, execute=False, repo=None, update_last=False, notice=None):
    from . import status_report
    for (program, obsnum), evs in sorted(_group_by_obs(events).items()):
        field = evs[0]["field"]
        title = status_report.issue_title_for(program, obsnum, field=field)
        body = status_report.render_events_comment(evs, notice=notice)
        status_report.post_status(title, body, repo=repo, update_last=update_last,
                                  dry_run=not execute)


# ------------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--program", "--limit-programs", dest="programs", nargs="*",
                    help="program id(s) to poll (default: all in PROGRAMS)")
    ap.add_argument("--state", default=DEFAULT_STATE,
                    help=f"state JSON path (default {DEFAULT_STATE})")
    ap.add_argument("--json", dest="as_json", action="store_true",
                    help="emit events as JSON instead of text lines")
    ap.add_argument("--commit-state", action="store_true",
                    help="write the updated state file (atomic tmp+rename)")
    ap.add_argument("--download", action="store_true",
                    help="retrieve products for new observations (data_qa.retrieve_data)")
    ap.add_argument("--trigger", action="store_true",
                    help="build the reduction+cataloging submissions (pipeline_trigger)")
    ap.add_argument("--report", action="store_true",
                    help="comment events on the per-observation QA issue")
    ap.add_argument("--auto", action="store_true",
                    help="AUTO mode: --download --trigger --report --commit-state "
                         "--execute in one flag, gated ONLY by the disk-space "
                         "check (--min-free-tb); below threshold it downgrades "
                         "to report-only with a loud LOW DISK warning")
    ap.add_argument("--min-free-tb", type=float, default=DEFAULT_MIN_FREE_TB,
                    help="minimum free space (TB) on the --download-dir "
                         f"filesystem for --auto (default {DEFAULT_MIN_FREE_TB})")
    ap.add_argument("--download-dir", default=DEFAULT_DOWNLOAD_DIR,
                    help="download destination for --download/--auto "
                         f"(default {DEFAULT_DOWNLOAD_DIR})")
    ap.add_argument("--pipe-root", default=None,
                    help="jwst-gc-pipeline checkout for --trigger")
    ap.add_argument("--repo", default=None, help="owner/name for --report")
    ap.add_argument("--execute", action="store_true",
                    help="really download/submit/post (default: dry-run actions)")
    args = ap.parse_args(argv)

    notice = None
    if args.auto:
        args.download = args.trigger = args.report = True
        args.commit_state = args.execute = True
        ok, _free, msg = disk_gate(args.download_dir, args.min_free_tb)
        print(f"--auto: {msg}", file=sys.stderr if not ok else sys.stdout)
        if not ok:
            # report-only downgrade: the issue comment still posts (with the
            # LOW DISK warning), but nothing is downloaded/submitted and the
            # state is NOT committed so the events re-fire next run.
            args.download = args.trigger = args.commit_state = False
            notice = msg

    programs = ([int(p) for p in args.programs] if args.programs
                else sorted(PROGRAMS))
    unknown = [p for p in programs if p not in PROGRAMS]
    if unknown:
        print(f"note: program(s) {unknown} not in PROGRAMS; polled anyway, "
              "but events carry no field mapping", file=sys.stderr)

    state = load_state(args.state)
    mast_login_if_token()
    poll_mjd = now_mjd()

    all_events = []
    for prog in programs:
        rows = query_program(prog)
        new_obs = summarize(rows, poll_mjd)
        old_obs = state.get("programs", {}).get(str(prog), {}).get("obs", {})
        all_events.extend(diff_events(prog, old_obs, new_obs))
        state.setdefault("programs", {})[str(prog)] = {"obs": new_obs}
    state["version"] = 1
    state["last_poll_mjd"] = poll_mjd
    state["last_poll_utc"] = mjd_to_iso(poll_mjd)

    if args.as_json:
        print(json.dumps(all_events, indent=2))
    else:
        for ev in all_events:
            print(format_event(ev))
        if not all_events:
            print(f"no new events across {len(programs)} program(s)")

    if all_events:
        if args.download:
            act_download(all_events, execute=args.execute,
                         download_dir=args.download_dir)
        if args.trigger:
            act_trigger(all_events, execute=args.execute, pipe_root=args.pipe_root)
        if args.report:
            act_report(all_events, execute=args.execute, repo=args.repo,
                       notice=notice)

    if args.commit_state:
        save_state(args.state, state)
        print(f"state committed: {args.state}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
