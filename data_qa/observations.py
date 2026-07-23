"""Registry of JWST Galactic Center observations.

Single source of truth that drives the QA tracking issues (``make_issues.py``) and the
data retrieval (``retrieve_data.py``).  One :class:`Observation` == one JWST observation
(a program's obs-number, per instrument) == one tracking issue; executions (visits) are
listed inside.

The registry is built from **MAST** -- the authoritative record of what has been
observed and made public.  ``mast_monitor.query_program`` returns every observation of a
program (obs number, instrument, filters, calibration level, epoch); an Observation is
emitted for each obs that has released calibrated products (``calib_level >= 2``) and is
one of the curated QA obsids in :data:`mast_monitor.PROGRAMS` (which maps obsid -> field).
Curated hand-notes are overlaid per obsid via :data:`CURATED`.

Deliberately NOT sourced from the public web release (the starformation.astro.ufl.edu
portal): that portal is the LAST step, published only after an observation's QA issue is
closed, so it can never be the discovery source without inverting the pipeline.  The QA
process tracks an observation from the moment its data is public on MAST; the web release
is downstream of QA and is not referenced here or in the issues.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field as _dc_field
from typing import Dict, List

# QA fields -> display name.  ``mast_monitor.PROGRAMS`` maps each program's curated
# obsids to one of these field keys; the field key is the on-disk basepath subdir
# (/orange/adamginsburg/jwst/<field>/) and the target's display name is looked up here.
# Keep a display name for EVERY field key referenced by ``mast_monitor.PROGRAMS``:
# ``target`` feeds ``issue_title`` (the idempotency key), so a later-added name would
# rename an existing issue and spawn a duplicate.
FIELDS: Dict[str, str] = {
    "brick": "Brick",
    "cloudc": "Cloud C",
    "cloudef": "Cloud E/F",
    "gc2211": "GC 2211",
    "gc-treasury": "GC Treasury",
    "arches": "Arches",
    "quintuplet": "Quintuplet",
    "sgra": "Sgr A*",
    "sgrb2": "Sgr B2",
    "sgrc": "Sgr C",
    "sickle": "Sickle",
    "ngc6334": "NGC 6334",
    "w51": "W51",
    "wd1": "Westerlund 1",
    "wd2": "Westerlund 2",
}

# Per-obsid curated overlay (optional): hand notes, epoch (DATE-OBS) override, visits.
CURATED: Dict[str, dict] = {
    "jw02221-o001": dict(
        epoch="2022-08-28", visits=["001"],
        notes="Narrow/medium-band NIRCam. F410M/NRCA5 carries a known per-module "
              "distortion+filteroffset offset corrected via a per-module split in the "
              "locked offsets table.",
    ),
    "jw01182-o004": dict(
        epoch="2022-09-14", visits=["001", "002"],
        notes="Wide-band NIRCam. LW filteroffset module-swap in the 2024 cal was fixed "
              "2026-07; a residual ~20 mas inter-module offset is addressed by a "
              "per-module 2-shift tie.",
    ),
}


@dataclass(frozen=True)
class Observation:
    program: str                 # e.g. "2221"
    obs: str                     # zero-padded 3 char, e.g. "001"
    target: str = ""             # display name, e.g. "Brick"
    release_field: str = ""      # field key / on-disk subdir, e.g. "brick" (defaults to target.lower())
    instrument: str = "NIRCam"   # "NIRCam" | "MIRI"
    filters: List[str] = _dc_field(default_factory=list)
    visits: List[str] = _dc_field(default_factory=list)
    epoch: str = ""
    notes: str = ""
    merged_obsids: List[str] = _dc_field(default_factory=list)  # other obs drizzled into the same tile

    @property
    def obsid(self) -> str:
        return f"jw{int(self.program):05d}-o{self.obs}"

    @property
    def field(self) -> str:
        return self.release_field or self.target.lower()

    @property
    def issue_title(self) -> str:
        """Stable idempotency key for the tracking issue."""
        return f"{self.target} — {self.obsid} ({self.instrument})"

    @property
    def mosaic_obsid(self) -> str:
        """Obsid stem the on-disk mosaics actually carry.  When this observation is
        drizzled together with others into one combined tile (e.g. jw05365-o002-998),
        the i2d is named for the combined id, not the bare obsid -- so the on-disk path
        must use this, not ``obsid``."""
        if self.merged_obsids:
            return f"{self.obsid}-{'-'.join(self.merged_obsids)}"
        return self.obsid

    # ---- links (MAST / archive only; the web release portal is intentionally absent) ----
    @property
    def mast_program_url(self) -> str:
        """Public APT program summary (the old get-proposal-info cgi now 404s)."""
        return f"https://www.stsci.edu/jwst/phase2-public/{int(self.program)}.pdf"

    @property
    def mast_search_url(self) -> str:
        """MAST data search filtered to this program."""
        return f"https://mast.stsci.edu/search/ui/#/jwst?proposal_id={int(self.program)}"

    def product_glob(self, basepath: str = "/orange/adamginsburg/jwst") -> str:
        # wildcard after the obs number so combined tiles (jw05365-o002-998_t001...) match,
        # not only the bare-obsid single-observation products.
        inst = self.instrument.lower()
        return (f"{basepath}/{self.field}/*/pipeline/"
                f"{self.obsid}*_t001_{inst}_*i2d.fits")


# --------------------------------------------------------------------------- discovery
# MAST-query failures recorded since the last registry() build.  A failed program query
# contributes no observations, but must be LOUD: make_issues checks this to distinguish
# "MAST genuinely returned nothing" from "the MAST query errored" and abort rather than
# sync an empty registry (live issue #4: a silent [] made the weekly sync a stale no-op).
LAST_FETCH_ERRORS: List[str] = []


def _epoch_from_tmax(t_max_values) -> str:
    """Earliest MAST ``t_max`` (MJD) of an observation -> ISO date ('' if unavailable)."""
    from . import mast_monitor as MM
    vals = [t for t in t_max_values if t is not None]
    if not vals:
        return ""
    iso = MM.mjd_to_iso(min(vals))
    return iso[:10] if iso and iso != "unknown" else ""


def _observations_for_program(program) -> List["Observation"]:
    """Emit one Observation per (obs, instrument) for the curated QA obsids of ``program``
    (``mast_monitor.PROGRAMS``), with metadata from MAST.  Only observations with released
    calibrated products (``calib_level >= 2``) are emitted.  A MAST failure is recorded to
    :data:`LAST_FETCH_ERRORS` and yields no observations for the program (never a silent []).
    """
    from . import mast_monitor as MM
    from .retrieve_data import mast_query_errors

    want = MM.PROGRAMS.get(int(program), {})     # curated obsnum -> field
    if not want:
        return []                                # uncurated / treasury (dynamic obs)

    try:
        rows = MM.query_program(program)
    except ImportError as ex:
        # astroquery/astropy absent (the issue-sync Action installs runtime deps;
        # a missing one must be LOUD-but-guarded, not an uncaught crash).  Caught
        # BEFORE mast_query_errors() below: evaluating that tuple itself imports
        # astroquery.exceptions, which would re-raise the very ModuleNotFoundError.
        msg = (f"MAST query dependency MISSING: program {int(program)}: "
               f"{ex.__class__.__name__}: {ex}")
        print(f"data_qa.observations: {msg}", file=sys.stderr)
        LAST_FETCH_ERRORS.append(msg)
        return []
    except mast_query_errors() as ex:            # network / MAST service error
        msg = f"MAST query FAILED: program {int(program)}: {ex.__class__.__name__}: {ex}"
        print(f"data_qa.observations: {msg}", file=sys.stderr)
        LAST_FETCH_ERRORS.append(msg)
        return []

    grouped: Dict[tuple, dict] = {}
    for r in rows:
        obsnum = MM.obsnum_from_obs_id(r.get("obs_id", ""))
        if obsnum not in want:
            continue                             # not a curated QA observation of this field
        if (r.get("calib_level") or -1) < MM.MIN_ACTIONABLE_CALIB_LEVEL:
            continue                             # planned / uncal-only: nothing to QA yet
        inst = MM.instrument_class(r.get("instrument_name")) or "NIRCam"
        g = grouped.setdefault((obsnum, inst), {"filters": [], "t_max": []})
        for f in MM.parse_filters(r.get("filters")):
            if f not in g["filters"]:
                g["filters"].append(f)
        g["t_max"].append(r.get("t_max"))

    out = []
    for (obsnum, inst), g in sorted(grouped.items()):
        field = want[obsnum]
        oid = f"jw{int(program):05d}-o{obsnum}"
        cur = CURATED.get(oid, {})
        out.append(Observation(
            program=str(int(program)), obs=obsnum,
            target=FIELDS.get(field, field.replace("_", " ").title()),
            release_field=field, instrument=inst,
            filters=sorted(g["filters"]),
            visits=cur.get("visits", []),
            epoch=cur.get("epoch") or _epoch_from_tmax(g["t_max"]),
            notes=cur.get("notes", ""),
        ))
    return out


def registry(programs=None, target=None) -> List["Observation"]:
    """The QA observation registry, built from MAST over the curated
    :data:`mast_monitor.PROGRAMS` obsid->field map -- one Observation per (program, obs,
    instrument) that has released calibrated products (``calib_level >= 2``).  Optionally
    filtered by program id(s) or target name.  MAST-query failures are recorded to
    :data:`LAST_FETCH_ERRORS` (checked by make_issues to avoid syncing an empty registry)."""
    from . import mast_monitor as MM
    LAST_FETCH_ERRORS.clear()
    if programs:
        progs = [int(p) for p in programs]
    else:
        progs = [p for p in MM.PROGRAMS if p != MM.TREASURY_PROGRAM]
    obs: List[Observation] = []
    for prog in progs:
        obs.extend(_observations_for_program(prog))
    if target:
        t = target.lower()
        obs = [o for o in obs if o.target.lower() == t or o.field == t]
    return obs
