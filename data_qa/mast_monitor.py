"""Poll MAST for new / newly-released / newly-calibrated JWST GC observations.

Stateful monitor: each run queries MAST per program, diffs against a state file, and
reports events.  Default is report-only (print events, touch nothing); every side
effect is individually gated:

  --commit-state   update the state file (otherwise the same events re-report)
  --download       fetch the new observation's products via data_qa.retrieve_data
  --trigger        build the reduction+cataloging submission (data_qa.pipeline_trigger)
  --report         comment the events on the per-observation QA issue (status_report);
                   an issue whose events include an ARRIVAL (calibrated data
                   landing: NEWLY_RELEASED, a CALIB_LEVEL_UP crossing the
                   actionable level, or a NEW_OBSERVATION already released), and
                   every issue on a run carrying a LOW DISK / CAPPED notice whose
                   reason differs from the last one announced, gets a NEW comment
                   (GitHub notifies watchers); anything else edits the previous
                   monitor comment in place (see act_report)
  --execute        actually do it (download / sbatch / post); without it the actions
                   above are dry-run prints
  --auto           fully automatic: --download --trigger --report --commit-state
                   --execute, gated by the disk-space check (--min-free-tb,
                   against the --download-dir filesystem), the first-run seed
                   guard, and the --max-submit cap (overflow acts on the oldest
                   N actionable groups and defers the rest); a failed disk gate
                   downgrades the run to report-only with a loud notice
  --seed           baseline run: commit the full current state, act on NOTHING
                   (use once at deployment so the backlog never fires as "new")
  --max-submit N   cap on the actionable (program,obs,instrument) groups one
                   acting run may touch; overflow acts on the N oldest and
                   defers the rest to later runs (default 4)

Safety gates on acting runs (--auto, or --download/--trigger/--peppar with
--execute):
  * FIRST-RUN SEED: a missing/empty state file means EVERY observation would
    fire as NEW_OBSERVATION; an acting run instead behaves like --seed
    (commit state, report "SEED RUN", no downloads/submissions/fan-outs).
    Plain dry-run invocations are unchanged.
  * SUBMISSION CAP (--max-submit): only groups the run would ACT on count
    (see ``actionable_groups``) -- event_ready events with a field mapping,
    minus groups the one-shot ``triggered``/``downloaded`` maps already
    retired; the trigger dimension counts NIRCam groups only (the MIRI
    trigger path is manual today), and --peppar counts every ready NIRCam
    group (act_peppar keeps no one-shot key, so it never retires).  Every
    action enabled on an acting run is a counted dimension, because the capped
    run acts on the counted groups ONLY: a group an enabled action needs but
    the count omits is dropped from the run and committed as seen, losing that
    action outright where a counted group would merely be postponed.  This is
    why ``--peppar --execute`` on its own also counts as an acting run: it
    would otherwise skip this whole block and fan out the entire backlog.
    ``ACTION_FLAGS`` is the one list main() spreads into the acting gate, the
    first-run seed suppression and the ``actionable_groups`` dimensions, so
    those three stay in step as actions are added.
    Groups queue strictly OLDEST-READY-FIRST: a group is dated by the earliest
    t_obs_release among its event_ready events (an undated ready group sorts
    last), ties broken on the group key so the acted set is stable across MAST
    row orderings.
    Overflow acts on the N OLDEST groups and commits state for exactly those
    plus the groups that consume no slot at all (planned tiles, unmapped
    programs, retired keys); the deferred groups keep their pre-poll
    baselines, so their events re-fire (and count again) on later runs until
    the backlog drains.  The CAPPED notice names the acted and deferred groups
    (long lists are truncated with an "and N more" tail).
    (Counting every grouped event and downgrading all-or-nothing without a
    commit wedged --auto permanently once the group count passed the cap:
    the same events re-fired and re-counted every later morning -- the
    treasury-delivery failure of issue #67.)
    Vocabulary used throughout this block and in ``actionable_groups``:
      one-shot key  an entry in the state file's ``triggered`` /
                    ``downloaded`` map.  Written the moment an action
                    succeeds and never rewritten, so it fires at most once per
                    obs for the life of the state file; ARMED means the key is
                    absent (the action is still owed), BURNED / RETIRED means
                    the key is present (the action already happened).  Delete
                    the entry by hand to re-arm.
      dimension     one action (download / trigger / peppar) that
                    ``actionable_groups`` evaluates per group.
      slot          one unit of the --max-submit budget.  A group consumes one
                    slot when at least one dimension is armed for it,
                    regardless of how many dimensions are; a group with every
                    dimension retired consumes none.
      skip-chatter  events that reach the act_* functions and print SKIP
                    without doing anything (planned tiles, unmapped programs,
                    MIRI on the trigger path).  They consume no slot.
      fan-out       act_peppar's per-filter/per-detector job submission for
                    one obs -- many SLURM jobs from one event.
  * IN-FLIGHT DEDUP (--trigger): a group is skipped when squeue already has a
    job named ``<field><program>-o<obs>-*`` or when the state file's
    ``triggered`` map marks the obs as already submitted (the map is written
    immediately on every successful submission -- with the sbatch job ids --
    even when the event state is not committed, so a partial failure cannot
    double-submit; re-arm with ``--rearm <program>-o<obs>``).  ARMED = the obs
    has no key in that map, so the monitor is allowed to fire its one-shot
    action for it; recording the key DISARMS (burns) it, and --rearm deletes
    the key to arm it again.
  * REGISTRY PREFLIGHT (--trigger): pipeline_trigger.build_plan verifies the
    obs is registered in the pipeline's fields.yaml BEFORE any sbatch; an
    unregistered obs prints SKIPPED(not-registered) and the one-shot key
    stays armed for the poll after the registration lands.

Events:
  NEW_OBSERVATION  obs_id not previously in the state file
  NEWLY_RELEASED   t_obs_release passed since the last committed poll
  CALIB_LEVEL_UP   calib_level increased (e.g. 2 -> 3: mosaics available)

  ARRIVAL is the subset that announces calibrated data actually landing, the
  class --report must interrupt a human for: NEWLY_RELEASED, a CALIB_LEVEL_UP
  that crossed MIN_ACTIONABLE_CALIB_LEVEL, or a NEW_OBSERVATION first seen
  already released and calibrated (see event_is_arrival).  A planned tile and
  a 2 -> 3 reprocessing bump stay routine: they edit the monitor comment in
  place.

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
    python -m data_qa.mast_monitor --rearm 10678-o001    # re-arm a burned trigger
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from typing import Dict, List, Optional, Set

# Monitored programs: program id -> {obs number -> release field}.  Mirrors the
# reduction's field registry (jwst_gc_pipeline/fields.yaml, read through
# jwst_gc_pipeline.fields.field_to_reg_mapping; it lived in a literal dict in
# PipelineRerunNIRCAM-LONG.py until pipeline commit ee33bec) restricted to the
# GC-treasury/QA programs; field names match data_qa.observations.FIELDS keys where
# released (cloudef/sgra/ngc6334 have no public release page yet).
# tests/test_mast_monitor.py::test_programs_complete_vs_pipeline_field_mapping
# compares the two and fails when this table falls behind the registry.
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
    5365: {"001": "sgrb2"},                          # NIRCam + MIRI obs
    6151: {"001": "w51"},
    2045: {"001": "arches", "003": "quintuplet"},
    3958: {"001": "sickle", "002": "sickle", "007": "sickle"},
    2092: {"002": "cloudef", "005": "cloudef_controlfield"},   # o005 = the control/offset pointing
    1939: {"001": "sgra"},
    1905: {"001": "wd1", "003": "wd1"},
    3523: {"003": "wd2", "005": "wd2"},
    6778: {"001": "ngc6334"},
    7213: {"001": "ngc6334"},
}

# Programs whose fields sit in the inner Galactic Center / CMZ -- the crowded
# fields the pipeline README marks DEBLEND_SATSTARS=1 "required for crowded GC
# fields" (jwst-gc-pipeline scripts/reduction/README.md).  Membership is decided
# per FIELD (the cataloging TARGET), derived from these programs' PROGRAMS rows.
# 1182 is left out on purpose: it maps BOTH brick (GC -- already covered via
# 2221) and w51 (non-GC), so deriving fields from it would wrongly sweep w51 in.
GC_PROGRAMS = (TREASURY_PROGRAM, 2221, 2211, 4147, 5365, 2045, 3958, 2092, 1939)

GC_FIELDS = frozenset(
    field for prog in GC_PROGRAMS for field in PROGRAMS[prog].values()
) | {TREASURY_FIELD}

DEFAULT_STATE = "/orange/adamginsburg/jwst/ops/mast_state.json"

# Columns kept from the MAST observation table (t_* are MJD floats).
MONITOR_COLUMNS = ("obs_id", "t_max", "t_obs_release", "calib_level",
                   "instrument_name", "filters", "target_name",
                   # the pointing centre, carried so a per-tile reference
                   # catalog can be built without a second MAST round-trip
                   # (keflavich/jwst-gc-pipeline#415).  Every row of one
                   # observation reports the same pair: measured 2026-08-20
                   # across all 139 planned 10678 observations (12 rows each)
                   # and all three delivered 2221 observations, max spread
                   # 0.00" in both.
                   "s_ra", "s_dec")

#: Columns read as floats rather than strings.
_FLOAT_COLUMNS = ("t_max", "t_obs_release", "s_ra", "s_dec")

# calib-level-3 products: jw02221-o001_t001_nircam_...; level<=2: jw02221001001_...
_OBS_DASH_RE = re.compile(r"^jw(\d{5})-o(\d{3})")
_OBS_FLAT_RE = re.compile(r"^jw(\d{5})(\d{3})\d{3}")


def obsnum_from_obs_id(obs_id: str) -> str:
    """'jw02221-o001_t001_...' or 'jw02221001001_...' -> '001' ('' if unparseable)."""
    m = _OBS_DASH_RE.match(obs_id) or _OBS_FLAT_RE.match(obs_id)
    return m.group(2) if m else ""


# Valid JWST filter token: F212N, F480M, F444W, F150W2, MIRI F1000W...  MAST's
# `filters` column mixes in pupil/CLEAR/GRISM tokens (CLEAR;F212N, F444W;F470N)
# that must never reach the reduction's FILTERS array.
FILTER_TOKEN = re.compile(r"^F\d{3,4}[WNM]2?$")


def parse_filters(filters_str) -> List[str]:
    """MAST `filters` string -> validated filter tokens.

    Splits on ';' (also tolerating ','), validates each token against
    FILTER_TOKEN, drops CLEAR/pupil/GRISM/junk, dedupes, keeps first-seen
    (stable) order."""
    out: List[str] = []
    for tok in (filters_str or "").replace(",", ";").split(";"):
        tok = tok.strip().upper()
        if FILTER_TOKEN.match(tok) and tok not in out:
            out.append(tok)
    return out


# Only observations with RELEASED, pipeline-ready products may trigger/download.
# calib_level -1 (masked on MAST) = planned/unexecuted; 1 = uncal-only; >= 2 has
# the *_cal/*_i2d products the reduction consumes.
MIN_ACTIONABLE_CALIB_LEVEL = 2


def event_ready(ev: dict) -> bool:
    """True when the event's observation has released, calibrated data
    (``released`` and ``calib_level >= 2``; both ride on the event payload).

    Planned/unreleased rows -- the 10678 treasury watch target -- fire
    NEW_OBSERVATION long before any data exists; they must REPORT (tagged
    PLANNED) but never trigger/download, and crucially must NOT burn the
    one-shot ``triggered`` key, or the real trigger is refused when the data
    finally arrives."""
    return (bool(ev.get("released"))
            and (ev.get("calib_level") or 0) >= MIN_ACTIONABLE_CALIB_LEVEL)


def event_is_arrival(ev: dict) -> bool:
    """True when the event announces calibrated data actually LANDING (the
    event class that must interrupt a human, issue #71): the observation is
    event_ready AND the event is the arrival edge -- NEWLY_RELEASED, a
    CALIB_LEVEL_UP that CROSSED MIN_ACTIONABLE_CALIB_LEVEL, or a
    NEW_OBSERVATION first seen with released calibrated data (a treasury tile
    that executed between polls fires this way; act_download/act_trigger act
    on it just like the other two).  A CALIB_LEVEL_UP entirely above the
    threshold (2 -> 3 reprocessing) is routine and is NOT an arrival."""
    if not event_ready(ev):
        return False
    if ev["event"] == "CALIB_LEVEL_UP":
        prev = ev.get("previous_calib_level")
        return (prev if prev is not None else 0) < MIN_ACTIONABLE_CALIB_LEVEL
    return ev["event"] in ("NEW_OBSERVATION", "NEWLY_RELEASED")


def instrument_class(instrument_name) -> str:
    """MAST ``instrument_name`` ('NIRCAM/IMAGE', 'MIRI/IMAGE') -> 'NIRCam' /
    'MIRI' ('' when absent/unknown).  NIRCam and MIRI observations of the same
    (program, obs) are DIFFERENT deliveries: separate QA issues, and only the
    NIRCam side is trigger-automated today."""
    name = (instrument_name or "").split("/")[0].strip().upper()
    if name.startswith("NIRCAM"):
        return "NIRCam"
    if name.startswith("MIRI"):
        return "MIRI"
    return name.title() if name else ""


def field_for(program, obsnum: str) -> str:
    if int(program) == TREASURY_PROGRAM:
        # Treasury tiles (GC_<n>) all reduce into the one gc-treasury field; the
        # per-obs tile name is carried on the event as 'tile' instead.
        return TREASURY_FIELD
    return PROGRAMS.get(int(program), {}).get(obsnum, "")


def now_mjd() -> float:
    return time.time() / 86400.0 + 40587.0


def mjd_to_iso(mjd) -> str:
    """MJD -> ISO string; ``"unknown"`` for None / NaN / masked / uncastable
    (planned-but-unexecuted MAST rows have a masked ``t_obs_release``, which
    reaches here as None or NaN -- it must never crash the report)."""
    if mjd is None:
        return "unknown"
    try:
        import numpy as np
    except ImportError:
        np = None
    if np is not None and np.ma.is_masked(mjd):
        return "unknown"
    try:
        ts = (float(mjd) - 40587.0) * 86400.0
    except (TypeError, ValueError):
        return "unknown"
    if math.isnan(ts):
        return "unknown"
    return datetime.datetime.fromtimestamp(
        ts, tz=datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _scalar(val, cast, default=None):
    """One MAST table cell -> a plain-python scalar via ``cast``; ``default`` for
    masked / NaN / None / uncastable cells.

    Planned/unreleased rows (e.g. every 10678 treasury tile) carry MASKED
    ``calib_level`` / ``t_obs_release``: ``int(np.ma.masked)`` raises
    ``numpy.ma.MaskError`` (NOT a TypeError/ValueError) and ``float(np.ma.masked)``
    silently returns NaN that later crashes ``mjd_to_iso`` -- so masked-ness is
    checked explicitly (lazy numpy import keeps the module stdlib-importable)."""
    if val is None:
        return default
    try:
        import numpy as np
    except ImportError:
        np = None
    if np is not None and np.ma.is_masked(val):
        return default
    try:
        out = cast(val)
    except (TypeError, ValueError):
        return default
    if isinstance(out, float) and math.isnan(out):
        return default
    return out


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
    """MAST observations for one program as plain-python dict rows (lazy astroquery).

    The request is time-bounded (retrieve_data.configure_mast: astroquery.mast.conf
    timeout/pagesize -- the 22h-hang lesson); network failures propagate as
    ``retrieve_data.mast_query_errors()`` for the caller's per-program isolation."""
    from astroquery.mast import Observations
    from .retrieve_data import configure_mast
    configure_mast()
    tbl = Observations.query_criteria(proposal_id=str(int(program)),
                                      obs_collection="JWST")
    cols = [c for c in MONITOR_COLUMNS if c in tbl.colnames]
    rows = []
    for r in tbl:
        row = {}
        for c in cols:
            v = r[c]
            if c in _FLOAT_COLUMNS:
                row[c] = _scalar(v, float, default=None)
            elif c == "calib_level":
                # planned/unreleased rows have a MASKED calib_level: -1 marks
                # "no calibrated products" and keeps the row un-actionable
                # (event_ready requires calib_level >= 2)
                row[c] = _scalar(v, int, default=-1)
            else:
                row[c] = _scalar(v, str, default=None)
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
            "s_ra": row.get("s_ra"),
            "s_dec": row.get("s_dec"),
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


# A botched hand-edit or bad write used to cost the whole state file (incl. the
# 1668-row treasury seeding); a dated pre-write backup caps the loss at one day.
BACKUP_KEEP_DAYS = 14


def prune_state_backups(path, today=None, keep_days=BACKUP_KEEP_DAYS):
    """Delete ``<path>.bak-YYYYMMDD`` siblings older than ``keep_days`` days;
    files without a parseable date suffix are left alone."""
    today = today or datetime.date.today()
    parent = os.path.dirname(os.path.abspath(path))
    prefix = os.path.basename(path) + ".bak-"
    for name in os.listdir(parent):
        if not name.startswith(prefix):
            continue
        try:
            date = datetime.datetime.strptime(name[len(prefix):], "%Y%m%d").date()
        except ValueError:
            continue
        if (today - date).days > keep_days:
            os.unlink(os.path.join(parent, name))


def backup_state(path, today=None) -> Optional[str]:
    """Copy the existing state file to ``<path>.bak-YYYYMMDD`` before a write:
    ONE backup per day, the FIRST write of that day, so the sibling holds the
    state as it stood before today's run touched it.  A later same-day write
    keeps that first snapshot -- refreshing it would converge the "backup" on
    the current state (``_record_state_key`` writes once per recorded key, so a
    single run writes several times) and leave nothing to restore from.
    Returns the backup path (None when there is nothing on disk to back up).

    The copy goes to a tmp sibling and is renamed into place, so an interrupted
    copy (ENOSPC, RLIMIT_FSIZE) cannot leave a truncated file sitting at
    exactly the path a restore would use.

    Consequence of the one-per-day rule: an existing ``.bak-<today>`` is taken
    as today's snapshot whatever it holds and whatever its mode, so a file an
    operator put there by hand, or a read-only one, is kept silently.  A
    restore reads the sibling, and ``json.load`` is the check that it is a
    state file."""
    if not os.path.exists(path):
        return None
    today = today or datetime.date.today()
    bak = f"{path}.bak-{today.strftime('%Y%m%d')}"
    if os.path.isfile(bak):
        return bak
    tmp = f"{bak}.tmp.{os.getpid()}"
    try:
        shutil.copy2(path, tmp)
        os.replace(tmp, bak)
    finally:
        try:
            os.unlink(tmp)          # no-op on success: os.replace consumed it
        except FileNotFoundError:
            pass
    return bak


def save_state(path, state: dict):
    """Atomic write (tmp + rename); auto-creates the parent directory.  A failure
    between write and replace unlinks the orphan tmp file (the exception still
    propagates).  The previous on-disk state is first copied to a dated
    ``.bak-YYYYMMDD`` sibling (backup_state); a backup that CANNOT be written
    (read-only or foreign-owned leftover, ENOSPC) warns and the write PROCEEDS
    -- the backup is a convenience, while a lost write loses a just-recorded
    trigger key and re-submits that observation on the next poll.  Pruning old
    backups is reported separately: a prune that fails says so, instead of
    claiming the backup failed when it succeeded."""
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    try:
        backup_state(path)
    except OSError as ex:
        print(f"mast_monitor: state backup FAILED for {path} "
              f"({ex.__class__.__name__}: {ex}); writing the new state anyway",
              file=sys.stderr)
    try:
        prune_state_backups(path)
    except OSError as ex:
        print(f"mast_monitor: pruning old state backups FAILED for {path} "
              f"({ex.__class__.__name__}: {ex}); writing the new state anyway",
              file=sys.stderr)
    tmp = f"{path}.tmp.{os.getpid()}"
    try:
        with open(tmp, "w") as fh:
            json.dump(state, fh, indent=1, sort_keys=True)
            fh.write("\n")
        os.replace(tmp, path)
    finally:
        try:
            os.unlink(tmp)          # no-op on success: os.replace consumed it
        except FileNotFoundError:
            pass


# ---------------------------------------------------------------- trigger dedup state
def trigger_key(program, obsnum) -> str:
    """Key in the state file's ``triggered`` map: '<program>-o<obs>'."""
    return f"{int(program)}-o{obsnum}"


def download_key(program, obsnum, instrument="NIRCam") -> str:
    """Key in the state file's ``downloaded`` map: '<program>-o<obs>-<instr>'
    (instrument-qualified: the NIRCam and MIRI product sets of one (program,obs)
    are separate downloads)."""
    return f"{int(program)}-o{obsnum}-{instrument or 'NIRCam'}"


def _record_state_key(state_path, mapname: str, key: str, value,
                      state: Optional[dict] = None):
    """Persist a one-shot action key (``triggered``/``downloaded`` map)
    IMMEDIATELY (fresh read-modify-write of the on-disk state), so a crash or a
    deliberately uncommitted run cannot re-fire the same action.  Only the named
    map is touched on disk -- the event baselines (``programs``) keep whatever
    commit decision the caller made.  Also mirrors the entry into the caller's
    in-memory ``state`` so a later full commit carries it."""
    disk = load_state(state_path)
    disk.setdefault(mapname, {})[key] = value
    save_state(state_path, disk)
    if state is not None:
        state.setdefault(mapname, {})[key] = value


def record_triggered(state_path, key: str, when: str,
                     state: Optional[dict] = None, jobids=None):
    """Persist a successful submission into the ``triggered`` map (see
    ``_record_state_key``) as ``{"when": ..., "jobids": [...]}`` -- the sbatch
    job ids ride along so a later run can probe their outcome (sacct) instead
    of trusting acceptance.  Only release-gated successful submits reach this
    (act_trigger's event_ready gate), so a planned/unreleased observation can
    never burn its one-shot key."""
    _record_state_key(state_path, "triggered", key,
                      {"when": when, "jobids": list(jobids or [])}, state=state)


def trigger_record_when(value) -> str:
    """A ``triggered`` map value -> its timestamp.  Values are dicts
    ``{"when", "jobids"}`` now; bare timestamp strings in pre-#68 state files
    are still honored."""
    if isinstance(value, dict):
        return value.get("when", "?")
    return value


def record_downloaded(state_path, key: str, when: str, state: Optional[dict] = None):
    """Persist a successful download into the ``downloaded`` map (mirror of the
    ``triggered`` map; release-gated the same way)."""
    _record_state_key(state_path, "downloaded", key, when, state=state)


# --auto downgrade notices that must interrupt a human when they FIRST appear
# (issue #71).  Matched by PREFIX against the run notice, and the memo below
# keys on this coarse class: the LOW DISK text embeds the fluctuating free-TB
# figure, so keying on the full message would re-notify on every 0.1 TB wiggle.
DOWNGRADE_NOTICE_PREFIXES = ("LOW DISK", "CAPPED")

# Top-level state-file key remembering which downgrade reason was last
# announced with a NEW (notifying) comment.
NOTIFIED_DOWNGRADE_KEY = "last_notified_downgrade"


def notice_downgrade_reason(notice: Optional[str]) -> Optional[str]:
    """Coarse downgrade class of a run notice ('LOW DISK' / 'CAPPED'), or None
    for no notice or a non-downgrade notice.

    Scoped to the NOTICE alone: SEED RUN / PER-PROGRAM SEED are deliberate
    baselines, so the notice itself never needs to interrupt anyone.  A seed
    run's EVENTS are classified independently -- a backlog holding released
    observations still carries arrivals, and act_report posts a notifying
    comment for them."""
    for prefix in DOWNGRADE_NOTICE_PREFIXES:
        if notice is not None and notice.startswith(prefix):
            return prefix
    return None


def _memo_notified_downgrade(state_path, reason: Optional[str],
                             state: Optional[dict] = None):
    """Persist (str) or clear (None) the last-notified downgrade reason
    IMMEDIATELY (fresh read-modify-write, like the ``triggered`` map: a LOW
    DISK run commits no state at all, so the memo must ride its own write).
    No disk write when the stored value already matches; the caller's
    in-memory ``state`` is mirrored so a later full commit carries it."""
    disk = load_state(state_path)
    if disk.get(NOTIFIED_DOWNGRADE_KEY) != reason:
        if reason is None:
            disk.pop(NOTIFIED_DOWNGRADE_KEY, None)
        else:
            disk[NOTIFIED_DOWNGRADE_KEY] = reason
        save_state(state_path, disk)
    if state is not None:
        if reason is None:
            state.pop(NOTIFIED_DOWNGRADE_KEY, None)
        else:
            state[NOTIFIED_DOWNGRADE_KEY] = reason


def retire_downgrade_memo(state: dict, notice: Optional[str]) -> dict:
    """Commit-time end of a downgrade episode: drop the last-notified memo
    from ``state`` when the run carried NO downgrade notice, so the next
    LOW DISK / CAPPED notice posts a fresh (notifying) comment.  Returns the
    same dict.

    This is the ONLY clearing path on a ``--commit-state`` run without
    ``--report`` (act_report, the other one, does not run) -- the shape the
    weekly scrontab poll had until issue #71 (docs/scrontab.example now adds
    --report --execute so a mid-wedge weekly run announces the arrivals its
    commit retires, but --commit-state alone stays a supported invocation).

    Gated on the notice instead of clearing unconditionally: a DOWNGRADED run
    that still commits must KEEP its memo.  A LOW DISK run commits nothing, so
    the gate is inert for it today, but a CAPPED run that acts on a subset and
    commits for exactly those groups (issue #67) would otherwise wipe the memo
    on every poll and re-notify each repeat."""
    if notice_downgrade_reason(notice) is None:
        state.pop(NOTIFIED_DOWNGRADE_KEY, None)
    return state


# An observation token: 1-4 digits, optionally '-'-joined.  A JOINT token
# ('001-002') names several observations cataloged as one unit -- Sgr B2's MIRI
# 002 and 998 tile one field between them, so they are reduced and cataloged
# together and the state file keys them as one.  SINGLE SOURCE of the grammar:
# pipeline_trigger imports this pattern for the obs token it interpolates into
# its preflight -c code, so every key the monitor can write is addressable by
# --rearm.  '[0-9]' rather than '\d' (which also matches Arabic-Indic digits)
# and a length bound, on both sides.
OBS_TOKEN_PATTERN = r"[0-9]{1,4}(?:-[0-9]{1,4})*"
# '<program>-o<obs>' as --rearm spells it
_REARM_RE = re.compile(rf"([0-9]+)-o({OBS_TOKEN_PATTERN})")


def rearm(state_path, spec: str, include_download=False) -> int:
    """--rearm PROGRAM-oNNN: delete the one-shot ``triggered`` key (and, with
    ``include_download``, the obs's ``downloaded`` keys) so the next poll may
    act again -- the observation goes back to ARMED, the state in which the
    monitor is allowed to fire its one-shot action for it.  The sanctioned
    replacement for hand-editing the state file.  Prints every removed entry;
    REFUSES (rc 1) on a malformed spec or when nothing matches, so a typo
    cannot silently 'succeed' (which also makes it non-idempotent: a second
    --rearm of the same observation exits 1).  Returns a process return
    code."""
    m = _REARM_RE.fullmatch(spec or "")
    if not m:
        print(f"--rearm: REFUSED: {spec!r} does not look like "
              "'<program>-o<obs>' (e.g. 10678-o001)", file=sys.stderr)
        return 1
    key = trigger_key(m.group(1), m.group(2))
    state = load_state(state_path)
    removed = []
    trig = state.get("triggered", {})
    if key in trig:
        removed.append(("triggered", key, trig.pop(key)))
    if include_download:
        dl = state.get("downloaded", {})
        # downloaded keys are instrument-qualified: '<program>-o<obs>-<instr>',
        # and an instrument name carries no '-' while a JOINT obs token does --
        # so match the suffix exactly, or '--rearm 2221-o001 --rearm-download'
        # would also clear the SEPARATE observation '2221-o001-002'
        dl_re = re.compile(re.escape(key) + r"-[A-Za-z0-9]+")
        for dkey in [k for k in sorted(dl) if dl_re.fullmatch(k)]:
            removed.append(("downloaded", dkey, dl.pop(dkey)))
    if not removed:
        print(f"--rearm: REFUSED: no triggered"
              + ("/downloaded" if include_download else "")
              + f" entry matches {key} in {state_path}", file=sys.stderr)
        return 1
    save_state(state_path, state)
    for mapname, k, value in removed:
        print(f"--rearm: removed {mapname}[{k}] "
              f"(was: {trigger_record_when(value)})")
    return 0


def pointings(state_path, program, released_only=False) -> List[dict]:
    """Per-observation pointing centres for one program, from the state file.

    One row per observation, newest poll wins: ``{obsnum, tile, ra, dec,
    calib_level, released, instrument}``.  Every MAST row of an observation
    reports the same centre (verified 2026-08-20: 139 planned 10678
    observations and 3 delivered 2221 ones, 0.00" spread within each), so the
    first row carrying finite coordinates answers for the observation.

    This exists so the per-tile reference catalogs of
    keflavich/jwst-gc-pipeline#415 can be built from what the monitor already
    polled, rather than each builder invocation re-querying MAST for a centre
    the state file has held since the observation was planned.  Observations
    with no coordinates on record are omitted rather than reported at (0, 0).
    """
    state = load_state(state_path)
    obs = (state.get("programs", {}).get(str(int(program)), {})
           .get("obs", {}))
    out = {}
    for obs_id, rec in sorted(obs.items()):
        ra, dec = rec.get("s_ra"), rec.get("s_dec")
        if ra is None or dec is None:
            continue
        if released_only and not rec.get("released"):
            continue
        obsnum = obsnum_from_obs_id(obs_id)
        if not obsnum or obsnum in out:
            continue
        out[obsnum] = dict(obsnum=obsnum, tile=rec.get("target_name"),
                           ra=float(ra), dec=float(dec),
                           calib_level=rec.get("calib_level"),
                           released=bool(rec.get("released")),
                           instrument=rec.get("instrument_name"))
    return [out[k] for k in sorted(out)]


def print_pointings(state_path, program, released_only=False) -> int:
    """--pointings: one TSV line per observation, for a shell/driver caller.

    TSV rather than JSON because the consumer is a build loop in another
    repository, which cannot import this package.  Exits 1 when the program
    has no coordinates on record, so a driver reading an empty list does not
    mistake it for "no tiles to build".
    """
    rows = pointings(state_path, program, released_only=released_only)
    if not rows:
        scope = "released " if released_only else ""
        print(f"--pointings: no {scope}observations of program {program} "
              f"carry coordinates in {state_path}", file=sys.stderr)
        return 1
    print("obsnum\ttile\tra_deg\tdec_deg\tcalib\treleased")
    for r in rows:
        print(f"{r['obsnum']}\t{r['tile'] or ''}\t{r['ra']:.6f}\t"
              f"{r['dec']:.6f}\t{r['calib_level']}\t{int(r['released'])}")
    return 0


def inflight_job_names(timeout_s=30) -> Optional[Set[str]]:
    """Job names currently in ``squeue --me`` (set), or None when squeue is
    unavailable/failing (caller proceeds with a warning -- it cannot check)."""
    try:
        proc = subprocess.run(["squeue", "--me", "--noheader", "--format=%j"],
                              capture_output=True, text=True, timeout=timeout_s)
    except (FileNotFoundError, subprocess.TimeoutExpired) as ex:
        print(f"mast_monitor: squeue unavailable ({ex.__class__.__name__}); "
              "cannot check for in-flight jobs", file=sys.stderr)
        return None
    if proc.returncode != 0:
        print(f"mast_monitor: squeue failed: {proc.stderr.strip()}; "
              "cannot check for in-flight jobs", file=sys.stderr)
        return None
    return {ln.strip() for ln in proc.stdout.splitlines() if ln.strip()}


def format_event(ev: dict) -> str:
    extra = (f" (level {ev['previous_calib_level']} -> {ev['calib_level']})"
             if ev["event"] == "CALIB_LEVEL_UP"
             else f" calib={ev['calib_level']} release={mjd_to_iso(ev['t_obs_release'])}")
    field = ev["field"] or "?unmapped?"
    tile = f" tile={ev['tile']}" if ev.get("tile") else ""
    planned = "" if event_ready(ev) else " PLANNED(no released calibrated data yet)"
    return (f"{ev['event']:16s} {ev['program']} {ev['obs_id']} "
            f"[field={field}{tile} filters={ev.get('filters') or '?'}]"
            + extra + planned)


def _group_by_obs(events):
    """Events -> {(program, obsnum, instrument_class): [events]} for
    per-observation actions (obs-level only: skips events whose obsnum could not
    be parsed).  Instrument-qualified: NIRCam and MIRI deliveries of the same
    (program, obs) -- e.g. jw02221-o002 -- are distinct groups with distinct QA
    issues, and only NIRCam groups are trigger-automated."""
    grouped = {}
    for ev in events:
        if ev["obsnum"]:
            key = (ev["program"], ev["obsnum"],
                   instrument_class(ev.get("instrument_name")))
            grouped.setdefault(key, []).append(ev)
    return grouped


# The submitting actions, in one place.  Each name is simultaneously an argparse
# dest (--download / --trigger / --peppar), a keyword of actionable_groups, and a
# term of main()'s `acting` gate; main() spreads this list into all three so a
# new action is enabled, seeded and counted together.  An action enabled without
# being counted is dropped from a capped run and committed as seen, which loses
# it outright -- the failure this module's SUBMISSION CAP block exists to stop.
ACTION_FLAGS = ("download", "trigger", "peppar")


def actionable_groups(state, events, download=True, trigger=True, peppar=False):
    """The (program, obsnum, instrument) groups this run would actually ACT on:
    ``{group_key: [events]}``, ordered oldest ready event first.  Pure (no
    network, no filesystem): exactly what the --max-submit cap must count.

    A group consumes a submission slot only when it has a field mapping, at
    least one event_ready event (planned/unreleased rows report only), and an
    un-retired action dimension:
      * download: its ``downloaded`` one-shot key is not burned;
      * trigger:  NIRCam only (act_trigger SKIPs MIRI as not-automated) and
        its ``triggered`` one-shot key is not burned;
      * peppar:   NIRCam only, and armed for EVERY ready group while --peppar is
        on -- act_peppar has no one-shot state key (its only dedup is the
        in-flight squeue job name), so each ready NIRCam group is a live
        fan-out.  It has to be a dimension of its own: a group whose download
        and trigger keys are both burned would otherwise fall out of this
        mapping, be dropped from the capped run's truncated event list, and --
        never having been deferred -- lose its peppar submission permanently.
    Everything else -- planned treasury tiles, unmapped programs, groups the
    one-shot maps already retired, MIRI trigger-side -- is skip-chatter, and
    counting it against the cap is how the first 10678 delivery morning
    wedged --auto into permanent report-only (issue #67).

    The result is an UPPER BOUND on the submissions a run makes: the act_*
    functions apply later gates this pure function cannot see (act_trigger's
    no-filters SKIP and squeue in-flight dedup, act_download's unknown-size /
    oversize / low-disk SKIPs), so a counted group can consume a slot without
    submitting anything.  Counting high keeps the cap conservative."""
    triggered = (state or {}).get("triggered", {})
    downloaded = (state or {}).get("downloaded", {})
    out = {}
    for key, evs in _group_by_obs(events).items():
        program, obsnum, instr = key
        if not evs[0]["field"]:
            continue          # unmapped: act_download/act_trigger both SKIP it
        if not any(event_ready(ev) for ev in evs):
            continue          # planned/unreleased: report-only, no slot burned
        would_download = (download and
                          download_key(program, obsnum, instr or "NIRCam")
                          not in downloaded)
        would_trigger = (trigger and instr == "NIRCam"
                         and trigger_key(program, obsnum) not in triggered)
        would_peppar = peppar and instr == "NIRCam"
        if would_download or would_trigger or would_peppar:
            out[key] = evs

    def _age(item):
        key, evs = item
        rels = [ev["t_obs_release"] for ev in evs
                if event_ready(ev) and ev.get("t_obs_release") is not None]
        return (min(rels) if rels else float("inf"), key)

    return dict(sorted(out.items(), key=_age))


def _group_label(key) -> str:
    """(program, obsnum, instrument) group key -> '10678-o001-NIRCam'."""
    program, obsnum, instr = key
    return f"{program}-o{obsnum}-{instr or '?'}"


NOTICE_LABEL_LIMIT = 10


def _group_label_list(keys, limit: int = NOTICE_LABEL_LIMIT) -> str:
    """Comma-joined group labels for the CAPPED notice, capped in length.
    Beyond ``limit`` labels the tail collapses to 'and N more', keeping the
    notice readable when a treasury backlog runs to dozens of groups (the
    notice is posted verbatim into a GitHub issue comment)."""
    keys = list(keys)
    shown = ", ".join(_group_label(k) for k in keys[:limit])
    if len(keys) > limit:
        return f"{shown}, and {len(keys) - limit} more"
    return shown


def _revert_deferred(state, old_obs_by_prog, deferred_events):
    """Restore the deferred groups' obs records to their pre-poll baseline, so
    the end-of-run commit retires exactly the acted groups: a deferred event
    diffs out identically (and re-fires) on the next poll.  A record absent
    from the baseline is popped (re-fires as NEW_OBSERVATION); a changed one is
    restored (re-fires as NEWLY_RELEASED / CALIB_LEVEL_UP)."""
    for ev in deferred_events:
        prog = str(ev["program"])
        obs_map = state.get("programs", {}).get(prog, {}).get("obs", {})
        prev = old_obs_by_prog.get(prog, {}).get(ev["obs_id"])
        if prev is None:
            obs_map.pop(ev["obs_id"], None)
        else:
            obs_map[ev["obs_id"]] = prev


# -------------------------------------------------------------------------- disk gate
DEFAULT_MIN_FREE_TB = 5.0
DEFAULT_MAX_SUBMIT = 4
# Absolute (matches docs/scrontab.example): a relative "./data" filled whatever
# directory the scrontab happened to run from.  NOTE this ops download tree is a
# STAGING copy for QA inspection of what arrived -- the reduction pipeline
# downloads its own inputs into its working tree and does NOT consume this copy
# (deliberately not unified for now).
DEFAULT_DOWNLOAD_DIR = "/orange/adamginsburg/jwst/ops/downloads"


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
def act_download(events, execute=False, download_dir=DEFAULT_DOWNLOAD_DIR,
                 min_free_tb=DEFAULT_MIN_FREE_TB, force_unknown_size=False,
                 state=None, state_path=None):
    """Download the products for each actionable (program, obs, instrument) group.

    The download tree is a STAGING copy for QA inspection -- the reduction
    downloads its own inputs (see DEFAULT_DOWNLOAD_DIR).  Release-gated
    (event_ready) and deduplicated via the state file's ``downloaded`` map
    (burned only on a successful release-gated download)."""
    from . import retrieve_data   # lazy: astroquery
    downloaded = (state or {}).get("downloaded", {})
    for (program, obsnum, instr), evs in sorted(_group_by_obs(events).items()):
        if not evs[0]["field"]:
            # mirror act_trigger: an unmapped program has nowhere to reduce, so
            # do not fill the disk for it either
            print(f"--download: SKIP program {program} obs {obsnum}: no field "
                  "mapping (add it to mast_monitor.PROGRAMS)", file=sys.stderr)
            continue
        if not any(event_ready(ev) for ev in evs):
            print(f"--download: SKIPPED(planned) program {program} obs {obsnum} "
                  f"({instr or '?'}): no released calib_level>="
                  f"{MIN_ACTIONABLE_CALIB_LEVEL} data yet -- report-only",
                  file=sys.stderr)
            continue
        instrument = instr or "NIRCam"
        dkey = download_key(program, obsnum, instrument)
        if dkey in downloaded:
            print(f"--download: SKIPPED(already-downloaded) program {program} "
                  f"obs {obsnum} ({instrument}): state marks a download at "
                  f"{downloaded[dkey]} (delete the 'downloaded' entry in the "
                  "state file to re-arm)", file=sys.stderr)
            continue
        if execute:
            # re-check the disk gate BETWEEN groups: an earlier group's download
            # may have eaten the headroom the run-level gate saw
            ok, free_tb, msg = disk_gate(download_dir, min_free_tb)
            if not ok:
                print(f"--download: SKIPPED(low-disk) program {program} obs "
                      f"{obsnum}: {msg}", file=sys.stderr)
                continue
            headroom_tb = free_tb - min_free_tb
            try:
                size = retrieve_data.product_list_size_bytes(
                    program, obsnum, product_type=("uncal", "i2d"),
                    instrument=instrument)
            except retrieve_data.mast_query_errors() as ex:
                print(f"--download: WARNING program {program} obs {obsnum}: "
                      f"size query failed ({ex.__class__.__name__}: {ex})",
                      file=sys.stderr)
                size = None
            if size is None:
                if not force_unknown_size:
                    print(f"--download: SKIPPED(unknown-size) program {program} "
                          f"obs {obsnum}: could not determine the projected "
                          "download size; rerun with --force-download-unknown-size "
                          "to download anyway", file=sys.stderr)
                    continue
                print(f"--download: WARNING program {program} obs {obsnum}: "
                      "unknown projected size; proceeding under "
                      "--force-download-unknown-size", file=sys.stderr)
            elif size / 1e12 > headroom_tb:
                print(f"--download: SKIPPED(oversize) program {program} obs "
                      f"{obsnum}: projected {size / 1e12:.2f} TB exceeds the "
                      f"{headroom_tb:.2f} TB headroom ({free_tb:.1f} TB free - "
                      f"{min_free_tb:.1f} TB --min-free-tb floor)", file=sys.stderr)
                continue
        print(f"--download: program {program} obs {obsnum} ({instrument}; "
              f"{len(evs)} event(s); dry_run={not execute})")
        result = retrieve_data.retrieve(
            program, obsnum, product_type=("uncal", "i2d"),
            instrument=instrument, download_dir=download_dir,
            dry_run=not execute)
        if execute and state_path and result is not None:
            # burned only on a successful, release-gated download (mirrors the
            # 'triggered' map semantics)
            record_downloaded(state_path, dkey, mjd_to_iso(now_mjd()), state=state)


def act_trigger(events, execute=False, pipe_root=None, state=None, state_path=None):
    from . import pipeline_trigger   # stdlib-only
    from .pipeline_trigger import NotRegisteredInPipelineError
    triggered = (state or {}).get("triggered", {})
    inflight = inflight_job_names() if execute else None
    for (program, obsnum, instr), evs in sorted(_group_by_obs(events).items()):
        field = evs[0]["field"]
        if not field:
            print(f"--trigger: SKIP program {program} obs {obsnum}: no field mapping "
                  "(add it to mast_monitor.PROGRAMS)", file=sys.stderr)
            continue
        ready = [ev for ev in evs if event_ready(ev)]
        if not ready:
            # planned/unreleased (calib_level -1, future/absent release): report
            # only.  Crucially does NOT record_triggered -- the one-shot key
            # stays armed for the REAL trigger when the data arrives.
            print(f"--trigger: SKIPPED(planned) program {program} obs {obsnum} "
                  f"({instr or '?'}): no released calib_level>="
                  f"{MIN_ACTIONABLE_CALIB_LEVEL} data yet -- report-only; "
                  "trigger stays armed for the data arrival", file=sys.stderr)
            continue
        if instr != "NIRCam":
            # the trigger path (submit_reduction.sbatch + cataloging chain) is
            # NIRCam-only today; MIRI (e.g. 5365, treasury F770W) is manual
            print(f"--trigger: SKIPPED(not-automated) program {program} obs "
                  f"{obsnum} ({instr or 'unknown instrument'}): the trigger "
                  "path is NIRCam-only today -- reduce by hand", file=sys.stderr)
            continue
        filters: List[str] = []
        for ev in ready:
            for tok in parse_filters(ev.get("filters")):
                if tok not in filters:
                    filters.append(tok)
        if not filters:
            print(f"--trigger: SKIP program {program} obs {obsnum}: no filters known",
                  file=sys.stderr)
            continue
        key = trigger_key(program, obsnum)
        if key in triggered:
            print(f"--trigger: SKIPPED(already-triggered) program {program} obs "
                  f"{obsnum}: state marks a submission at "
                  f"{trigger_record_when(triggered[key])} (re-arm with "
                  f"`python -m data_qa.mast_monitor --rearm {key}`)",
                  file=sys.stderr)
            continue
        prefix = f"{field}{int(program)}-o{obsnum}-"
        if inflight and any(name.startswith(prefix) for name in inflight):
            print(f"--trigger: SKIPPED(in-flight) program {program} obs {obsnum}: "
                  f"squeue --me already has a job named {prefix}*", file=sys.stderr)
            continue
        try:
            outcome = pipeline_trigger.submit(
                program=program, obs=obsnum, field=field, filters=filters,
                pipe_root=pipe_root, execute=execute)
        except NotRegisteredInPipelineError as ex:
            # the registry preflight failed in-process, BEFORE any sbatch:
            # report-only, and crucially does NOT record_triggered -- the
            # one-shot key stays armed for the poll after the registration
            # lands in the pipeline's fields.yaml
            print(f"--trigger: SKIPPED(not-registered) program {program} obs "
                  f"{obsnum}: {ex} -- register it in the pipeline's "
                  "fields.yaml; trigger stays armed", file=sys.stderr)
            continue
        if execute and state_path:
            # written IMMEDIATELY (not at the end-of-run commit) so a partial
            # failure in a later group cannot re-fire this submission; the
            # sbatch job ids ride along for later outcome probing
            jobids = (outcome.get("jobids", [])
                      if isinstance(outcome, dict) else [])
            record_triggered(state_path, key, mjd_to_iso(now_mjd()),
                             state=state, jobids=jobids)


def act_peppar(events, execute=False):
    """Fan peppar PSF-photometry jobs out (per filter/detector) for each newly-arrived NIRCam
    obs.  Reuses peppar_trigger.submit_all (which dedups in-flight by SLURM job name), so it is
    safe to re-fire every poll.  Same NIRCam-only / released-data gates as act_trigger.

    NOT wired into --auto: the peppar env needs repair (a dev photutils shadows the pinned
    1.12.0) before these jobs can succeed; enable with --peppar once it imports."""
    from .peppar_trigger import submit_all   # stdlib-only
    for (program, obsnum, instr), evs in sorted(_group_by_obs(events).items()):
        field = evs[0]["field"]
        if not field:
            print(f"--peppar: SKIP program {program} obs {obsnum}: no field mapping",
                  file=sys.stderr)
            continue
        if instr != "NIRCam":
            print(f"--peppar: SKIP program {program} obs {obsnum} ({instr or '?'}): "
                  "peppar is NIRCam-only", file=sys.stderr)
            continue
        if not any(event_ready(ev) for ev in evs):
            print(f"--peppar: SKIPPED(planned) program {program} obs {obsnum}: no released "
                  "data yet", file=sys.stderr)
            continue
        try:
            submit_all(program=program, obs=obsnum, field=field, execute=execute)
        except SystemExit as e:                # no cal files on disk yet, etc.
            print(f"--peppar: SKIP program {program} obs {obsnum}: {e}", file=sys.stderr)


# All treasury deliveries report into ONE rolling issue: per-obs issues would
# mean ~1668 title lookups/creations (rate-limit hazard + issue spam).
TREASURY_ISSUE_TITLE = f"GC Treasury — program {TREASURY_PROGRAM} deliveries"


def act_report(events, execute=False, repo=None, update_last=None, notice=None,
               state=None, state_path=None):
    """Comment the event batch on the QA issue(s).

    NOTIFICATION POLICY (issue #71): GitHub sends NO notification for a
    comment EDIT, so pure update-in-place silently swallowed the one event
    class that must interrupt a human.  With ``update_last=None`` each ISSUE
    is classified on its OWN events:

      * post a NEW comment (watchers notified) when that issue's events
        carry an ARRIVAL (``event_is_arrival``: calibrated data actually
        landed), or when the run carries a downgrade notice (LOW DISK /
        CAPPED) whose coarse reason is FRESH -- differing from the memo
        (state key ``last_notified_downgrade``) that records the reason last
        announced with a notifying comment.  The notice rides on EVERY
        comment body, so a fresh downgrade notifies on every issue;
      * edit the previous MONITOR_MARKER comment in place for that issue's
        purely-planned events and for REPEATS of an already-notified
        downgrade reason (a LOW DISK run commits no state, so the identical
        batch re-fires every poll -- the memo keeps those repeats from
        stacking).

    Per-issue rather than batch-global: one brick arrival must not make the
    planned-only treasury tiles post notifying comments too.

    A SEED RUN / PER-PROGRAM SEED backlog that contains released observations
    notifies: those arrivals were never announced, and the seed notice itself
    is not a downgrade (``notice_downgrade_reason`` -> None).

    The anti-spam memo is written on an executing run immediately after EVERY
    notifying downgrade comment in the batch posted successfully (a partial
    failure leaves it unarmed, so the next poll re-notifies: a duplicate
    comment on the issues that did post is cheaper than an issue whose
    watchers never hear the notice).  It is cleared by any executed batch
    without a downgrade notice, plus by main()'s commit path,
    ``retire_downgrade_memo`` -- the only clearing path when --report is off.

    A downgrade episode therefore notifies once per uninterrupted run of
    downgraded polls.  Any downgrade-free run in between ends the episode and
    re-arms the notification, including a poll that carries no notice because
    it never ran the disk gate: the weekly ``--commit-state --report
    --execute`` entry (docs/scrontab.example) has no --auto, so a LOW DISK
    wedge spanning a Monday re-notifies on the Tuesday poll.

    A downgraded run that keeps re-firing ARRIVAL events keeps posting new
    comments on purpose: released data is waiting while the automation is
    wedged, at the deployed poll cadence of once a day
    (docs/scrontab.example).  Pass ``update_last=True/False`` to override the
    classification for every issue.
    """
    from . import status_report
    reason = notice_downgrade_reason(notice)
    if state:
        last_notified = state.get(NOTIFIED_DOWNGRADE_KEY)
    elif state_path:
        last_notified = load_state(state_path).get(NOTIFIED_DOWNGRADE_KEY)
    else:
        last_notified = None
    fresh_downgrade = reason is not None and reason != last_notified
    override = update_last

    def _update_last(evs) -> bool:
        """Edit-in-place (True) vs new notifying comment (False) for ONE
        issue's events."""
        if override is not None:
            return override
        return not (fresh_downgrade
                    or any(event_is_arrival(ev) for ev in evs))

    # ONE existing_issues() fetch for the whole run, shared by every
    # post_status call below (a per-group refetch is the ~1668-treasury-group
    # rate-limit hazard); post_status fills it on first use.
    issue_cache: dict = {}
    posts: List[tuple] = []                      # (update_last, rc) per issue
    treasury = [ev for ev in events if ev.get("field") == TREASURY_FIELD]
    regular = [ev for ev in events if ev.get("field") != TREASURY_FIELD]
    for (program, obsnum, instr), evs in sorted(_group_by_obs(regular).items()):
        field = evs[0]["field"]
        # instrument-qualified title: NIRCam vs MIRI deliveries of the same
        # (program, obs) have separate QA issues (e.g. jw02221-o002)
        title = status_report.issue_title_for(program, obsnum, field=field,
                                              instrument=instr or "NIRCam")
        body = status_report.render_events_comment(evs, notice=notice)
        edit = _update_last(evs)
        posts.append((edit, status_report.post_status(
            title, body, repo=repo, update_last=edit,
            marker=status_report.MONITOR_MARKER,
            dry_run=not execute, issue_cache=issue_cache)))
    if treasury:
        # single rolling issue (auto-created if absent) instead of per-obs
        # rc=3 "no issue titled ..." failures for every treasury tile
        body = status_report.render_events_comment(treasury, notice=notice)
        edit = _update_last(treasury)
        posts.append((edit, status_report.post_status(
            TREASURY_ISSUE_TITLE, body, repo=repo, update_last=edit,
            marker=status_report.MONITOR_MARKER, dry_run=not execute,
            issue_cache=issue_cache,
            create_labels=["QA", f"program:{TREASURY_PROGRAM}"])))
    if execute and state_path:
        notifying = [rc for edit, rc in posts if not edit]
        # EVERY notifying comment must have landed before repeats are
        # suppressed: arming on a partial success (issue A posted, issue B
        # returned rc!=0) would leave B editing quietly forever and B's
        # watchers would never see the notice
        notified = bool(notifying) and all(rc == 0 for rc in notifying)
        if reason is not None and notified:
            # every notifying comment carrying the downgrade notice went out:
            # arm repeat suppression
            _memo_notified_downgrade(state_path, reason, state=state)
        elif reason is None:
            # downgrade-free batch: the episode is over; the NEXT downgrade
            # notice posts a fresh (notifying) comment
            _memo_notified_downgrade(state_path, None, state=state)


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
                         "--execute in one flag, gated by the disk-space check "
                         "(--min-free-tb), the first-run seed guard and the "
                         "--max-submit cap (overflow acts on the oldest N "
                         "actionable groups and defers the rest); a failed "
                         "disk gate downgrades the run to report-only with a "
                         "loud warning")
    ap.add_argument("--seed", action="store_true",
                    help="baseline run: commit the full current state, take NO "
                         "download/trigger actions (use once at deployment so "
                         "the existing backlog never fires as new)")
    ap.add_argument("--rearm", metavar="PROGRAM-oNNN", default=None,
                    help="delete the one-shot 'triggered' key for the given "
                         "observation (e.g. --rearm 10678-o001) so the next "
                         "poll may submit it again; prints what was removed, "
                         "refuses (rc 1) when nothing matches -- so a repeat "
                         "of the same --rearm also exits 1, which an '&&' "
                         "chain will read as a failure.  Exits without "
                         "polling MAST.")
    ap.add_argument("--rearm-download", action="store_true",
                    help="with --rearm: also delete the obs's 'downloaded' "
                         "key(s) so the products re-download")
    ap.add_argument("--pointings", metavar="PROGRAM", default=None,
                    help="print one TSV line per observation of PROGRAM with "
                         "its tile name and pointing centre, from the state "
                         "file, then exit; drives the per-tile reference "
                         "catalog build (keflavich/jwst-gc-pipeline#415).  "
                         "Exits 1 when no observation carries coordinates.")
    ap.add_argument("--released-only", action="store_true",
                    help="with --pointings, list only released observations")
    ap.add_argument("--max-submit", type=int, default=DEFAULT_MAX_SUBMIT,
                    help="max actionable (program,obs,instrument) groups an "
                         "acting run may touch; overflow acts on the N oldest "
                         "and defers the rest to later runs "
                         f"(default {DEFAULT_MAX_SUBMIT})")
    ap.add_argument("--force-download-unknown-size", action="store_true",
                    help="download even when the projected product size cannot "
                         "be determined (default: skip with a warning)")
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
    ap.add_argument("--peppar", action="store_true",
                    help="also fan out peppar PSF-photometry jobs (per filter/detector) for each "
                         "newly-arrived NIRCam obs. OPT-IN (not part of --auto): the peppar env "
                         "must be repaired first. Honours --execute like the other actions.")
    args = ap.parse_args(argv)
    if args.max_submit < 0:
        # a negative cap would slice keys[:negative] -- "act on all but the last
        # |N| groups", the opposite of a cap.  0 means act on nothing.
        ap.error(f"--max-submit must be >= 0 (got {args.max_submit})")

    if args.rearm_download and not args.rearm:
        ap.error("--rearm-download requires --rearm PROGRAM-oNNN")
    if args.pointings:
        return print_pointings(args.state, args.pointings,
                               released_only=args.released_only)
    if args.rearm:
        # state-surgery verb: touch only the one-shot maps, no MAST poll
        return rearm(args.state, args.rearm,
                     include_download=args.rearm_download)

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
            # EXCEPTION: a --seed run (or a first run that seeds itself) sets
            # commit_state back to True below, since a seed exists to write the
            # baseline; a hand-run `--seed --auto` under LOW DISK therefore
            # retires the events this gate withheld.  The LOW DISK notice and
            # the last-notified memo survive that path (the seed text is
            # appended to `notice` rather than replacing it).
            args.download = args.trigger = args.commit_state = False
            notice = msg

    programs = ([int(p) for p in args.programs] if args.programs
                else sorted(PROGRAMS))
    unknown = [p for p in programs if p not in PROGRAMS]
    if unknown:
        print(f"note: program(s) {unknown} not in PROGRAMS; polled anyway, "
              "but events carry no field mapping", file=sys.stderr)

    state = load_state(args.state)
    # per-program seed baseline: a program counts as seeded once it has EITHER
    # an entry in the 'seeded_programs' set or (back-compat with older state
    # files) a committed obs baseline.  A program whose query FAILED during the
    # seed run gets seeded on its first successful poll later (actions
    # suppressed for it that run) instead of firing its whole backlog.
    seeded = set(state.get("seeded_programs", []))
    seeded |= {p for p, rec in state.get("programs", {}).items()
               if (rec or {}).get("obs")}
    # first run = nothing seeded anywhere (missing/empty state)
    first_run = not seeded
    mast_login_if_token()
    poll_mjd = now_mjd()

    from .retrieve_data import mast_query_errors
    all_events, failed_programs, newly_seeded = [], [], []
    # pre-poll baselines, kept so a capped run can revert its DEFERRED groups'
    # records before the commit (see _revert_deferred)
    old_obs_by_prog: Dict[str, dict] = {}
    for prog in programs:
        try:
            rows = query_program(prog)
        except mast_query_errors() as ex:
            # per-program isolation: one hung/failed MAST request must not kill
            # the whole poll; the program's old state is left untouched so its
            # events fire on the next successful poll
            print(f"WARNING: MAST query for program {prog} failed "
                  f"({ex.__class__.__name__}: {ex}); skipping this program "
                  "this poll", file=sys.stderr)
            failed_programs.append(prog)
            continue
        new_obs = summarize(rows, poll_mjd)
        old_obs = state.get("programs", {}).get(str(prog), {}).get("obs", {})
        # setdefault keeps the FIRST pass's baseline: a duplicated --program
        # argument polls the same program twice, and the second pass's
        # "pre-poll" records are already the first pass's POST-poll records.
        # Overwriting with those would have _revert_deferred "restore"
        # deferred groups to what MAST just returned, so they would commit as
        # seen and never re-fire.
        old_obs_by_prog.setdefault(str(prog), old_obs)
        all_events.extend(diff_events(prog, old_obs, new_obs))
        # an obs that DISAPPEARED from MAST is kept under a 'missing_since'
        # note (report-only; no event, no silent drop) until it reappears
        merged_obs = dict(new_obs)
        for obs_id, rec in old_obs.items():
            if obs_id not in merged_obs:
                kept = dict(rec)
                kept.setdefault("missing_since", mjd_to_iso(poll_mjd))
                merged_obs[obs_id] = kept
                print(f"note: program {prog} obs {obs_id} disappeared from "
                      f"MAST; kept in state (missing_since "
                      f"{kept['missing_since']})", file=sys.stderr)
        state.setdefault("programs", {})[str(prog)] = {"obs": merged_obs}
        if str(prog) not in seeded:
            newly_seeded.append(str(prog))
    state["seeded_programs"] = sorted(seeded | set(newly_seeded))
    state["version"] = 1
    state["last_poll_mjd"] = poll_mjd
    state["last_poll_utc"] = mjd_to_iso(poll_mjd)

    if args.as_json:
        print(json.dumps(all_events, indent=2))
    else:
        for ev in all_events:
            print(format_event(ev))
        if not all_events:
            print(f"no new events across {len(programs)} program(s)"
                  + (f" ({len(failed_programs)} query failure(s))"
                     if failed_programs else ""))

    # FIRST-RUN SEED gate: with no baseline, every observation fires as NEW, so
    # an acting run would submit the entire backlog at once.  The first run
    # seeds: commit state, act on nothing.
    # EVERY submitting action counts here.  ACTION_FLAGS is the single list the
    # three coupled sites read (this gate, the seed suppression below, and the
    # actionable_groups dimensions at the SUBMISSION CAP), so an action can only
    # be enabled on a run that also seeds it and counts it.  Each omission cost a
    # review round: --peppar-only left `acting` False and fanned the whole
    # backlog out on the first run.
    acting = args.execute and any(getattr(args, flag) for flag in ACTION_FLAGS)
    seed = args.seed or (first_run and bool(all_events) and acting)
    actionable = all_events
    if seed:
        seed_notice = (f"SEED RUN — actions suppressed: committing the current "
                       f"state ({len(all_events)} event(s)) as the baseline; "
                       "nothing downloaded, submitted or fanned out.  "
                       "Subsequent runs act only on genuinely new events."
                       + ("" if args.seed else "  (state file was missing/empty)"))
        print(f"--seed: {seed_notice}", file=sys.stderr)
        # A LOW DISK notice from the --auto disk gate keeps the FRONT of the
        # run notice.  Overwriting it dropped the warning from every comment
        # body AND (the coarse reason is prefix-matched) cleared the
        # last-notified memo mid-episode, so the wedge announced itself to
        # nobody -- the #71 defect inside a --seed --auto invocation.  One
        # line, since render_events_comment renders the notice as a blockquote.
        notice = f"{notice}  {seed_notice}" if notice else seed_notice
        for flag in ACTION_FLAGS:
            setattr(args, flag, False)
        args.commit_state = True
    elif acting:
        # PER-PROGRAM SEED: a program polled successfully for the first time
        # (e.g. its query failed during the seed run) fires its whole backlog
        # as NEW -- commit it as baseline, but suppress actions for it this run
        if newly_seeded:
            suppressed_progs = sorted({str(ev["program"]) for ev in all_events
                                       if str(ev["program"]) in set(newly_seeded)})
            actionable = [ev for ev in all_events
                          if str(ev["program"]) not in set(newly_seeded)]
            if suppressed_progs:
                msg = (f"PER-PROGRAM SEED — program(s) "
                       f"{', '.join(suppressed_progs)} polled successfully for "
                       "the first time: their events are committed as baseline; "
                       "actions suppressed for them this run.")
                print(f"--seed(per-program): {msg}", file=sys.stderr)
                if notice is None:
                    notice = msg
        # SUBMISSION CAP: count only the groups this run would ACT on (planned
        # tiles, unmapped programs, retired one-shot keys and the MIRI trigger
        # side are skip-chatter -- see actionable_groups).  Overflow acts on
        # the --max-submit OLDEST groups; the deferred groups' baselines are
        # reverted so the commit retires exactly the acted groups and the
        # deferred events re-fire (and count again) next run.  EVERY action
        # enabled here must be a counted dimension: the truncated `actionable`
        # below feeds all of them, so a group an action needs but the count
        # omits is dropped from this run AND committed as seen -- that action is
        # lost outright, with no later run to pick it up.  The dimensions are
        # spread from ACTION_FLAGS, the same list `acting` above reads, so the
        # two stay in step by construction (test_action_flags_match_dimensions).
        groups = actionable_groups(state, actionable,
                                   **{flag: getattr(args, flag)
                                      for flag in ACTION_FLAGS})
        if len(groups) > args.max_submit:
            keys = list(groups)
            acted_keys = keys[:args.max_submit]
            deferred_keys = keys[args.max_submit:]
            actionable = [ev for key in acted_keys for ev in groups[key]]
            deferred = [ev for key in deferred_keys for ev in groups[key]]
            _revert_deferred(state, old_obs_by_prog, deferred)
            # the commit sentence tracks --commit-state: this notice is posted
            # verbatim onto the QA issue, so an operator reads it to know what
            # re-fires.  A dry-state acting run (--download/--trigger/--peppar
            # --execute without --commit-state) commits nothing at all.
            if args.commit_state:
                commit_clause = (
                    "The DEFERRED groups keep their pre-poll baselines and "
                    "re-fire next run; every other group in this poll -- the "
                    "acted ones and the ones that consume no slot (planned "
                    "tiles, unmapped programs, groups the one-shot maps "
                    "already retired) -- is committed now and does not "
                    "re-fire.")
            else:
                commit_clause = (
                    "This run commits no state (--commit-state is off), so "
                    "every group in this poll re-fires next run.")
            capped = (f"CAPPED — {len(groups)} actionable group(s) exceed "
                      f"--max-submit {args.max_submit}: acting on the "
                      f"{len(acted_keys)} oldest "
                      f"({_group_label_list(acted_keys)}); "
                      f"deferred to later runs: "
                      f"{_group_label_list(deferred_keys)}.  " + commit_clause)
            # keep an earlier notice (PER-PROGRAM SEED, LOW DISK): both
            # describe this run and the operator needs both
            notice = f"{notice}  {capped}" if notice else capped
            print(f"--max-submit: {capped}", file=sys.stderr)

    if all_events:
        if args.download:
            act_download(actionable, execute=args.execute,
                         download_dir=args.download_dir,
                         min_free_tb=args.min_free_tb,
                         force_unknown_size=args.force_download_unknown_size,
                         state=state, state_path=args.state)
        if args.trigger:
            act_trigger(actionable, execute=args.execute, pipe_root=args.pipe_root,
                        state=state, state_path=args.state)
        if args.peppar:
            act_peppar(actionable, execute=args.execute)
        if args.report:
            # report EVERYTHING (incl. per-program-seed-suppressed events);
            # state/state_path carry the last-notified-downgrade memo that
            # decides new-comment-vs-edit (see act_report)
            act_report(all_events, execute=args.execute, repo=args.repo,
                       notice=notice, state=state, state_path=args.state)

    if args.commit_state:
        # end the downgrade episode when this run was not itself downgraded,
        # so the NEXT downgrade notice posts a fresh (notifying) comment; a
        # downgraded run that still commits keeps its memo (see
        # retire_downgrade_memo).  This is the only clearing path when
        # --report is off (a plain --commit-state poll).
        retire_downgrade_memo(state, notice)
        save_state(args.state, state)
        print(f"state committed: {args.state}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
