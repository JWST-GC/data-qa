"""Registry of JWST Galactic Center observations.

Single source of truth that drives the QA tracking issues (``make_issues.py``) and the
data retrieval (``retrieve_data.py``).  One :class:`Observation` == one JWST observation
(a program's obs-number, per instrument) == one tracking issue; executions (visits) are
listed inside.

The registry is **discovered from the public release itself**: each field's
``{field}_images.txt`` on the JWST-GC portal lists the released mosaics, from which the
(program, obs, instrument, filters) of every observation are parsed.  Curated metadata
(known issues, epoch, visits) is overlaid per obsid via :data:`CURATED`.  Add a newly
released field to :data:`FIELDS`; add hand notes to :data:`CURATED`.
"""
from __future__ import annotations

import os
import re
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field as _dc_field
from typing import Dict, List

# JWST-GC public data-release portal. Per-field overview pages live at
# <base>/<field>.html; the authoritative direct-download URL lists (pointing at the
# Globus-hosted FITS) at <base>/<field>_{images,catalogs}.txt.
RELEASE_BASE = "https://starformation.astro.ufl.edu/jwst-gc"

# Released fields -> display name.  Add a field here when its products go public.
FIELDS: Dict[str, str] = {
    "brick": "Brick",
    "cloudc": "Cloud C",
    "gc2211": "GC 2211",
    "sgrb2": "Sgr B2",
    "sgrc": "Sgr C",
    "sickle": "Sickle",
    "w51": "W51",
    "wd1": "Westerlund 1",
    "wd2": "Westerlund 2",
}

# Per-obsid curated overlay (optional): known issues, epoch (DATE-OBS), visits.
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

# Parse a mosaic filename: program, obs (tile suffix stripped), instrument, filter.
#   jw02221-o001_t001_nircam_clear-f182m-merged_i2d.fits
#   jw05365-o002-998_t001_miri_clear-f770w-mirimage_data_i2d.fits
#   jw02221-o002_t001_miri_f2550w_i2d.fits
_MOSAIC_RE = re.compile(
    r"jw(\d{5})-o(\d{3})(?:-(\d+))?_t\d+_(nircam|miri)_(?:clear-)?(f\d{3,4}[wnm])", re.I)


@dataclass(frozen=True)
class Observation:
    program: str                 # e.g. "2221"
    obs: str                     # zero-padded 3 char, e.g. "001"
    target: str = ""             # display name, e.g. "Brick"
    release_field: str = ""      # release-page basename, e.g. "brick" (defaults to target.lower())
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
        """Obsid stem the RELEASE mosaics actually carry.  When this observation is drizzled
        together with others into one combined tile (e.g. jw05365-o002-998), the released
        i2d is named for the combined id, not the bare obsid -- so the on-disk path and the
        download links must use this, not ``obsid``."""
        if self.merged_obsids:
            return f"{self.obsid}-{'-'.join(self.merged_obsids)}"
        return self.obsid

    # ---- links ----
    @property
    def mast_program_url(self) -> str:
        """Public APT program summary (the old get-proposal-info cgi now 404s)."""
        return f"https://www.stsci.edu/jwst/phase2-public/{int(self.program)}.pdf"

    @property
    def mast_search_url(self) -> str:
        """MAST data search that actually RUNS the query for this program.

        The bare ``#/jwst?proposal_id=N`` form only opens a blank, unexecuted search
        form.  The executed-results route is ``#/jwst/results?...&program_id=N``; the
        ``search_key`` in a shared MAST URL is a session-specific saved-search hash and
        is intentionally omitted (``useStore=false`` builds the query fresh from the
        explicit params instead)."""
        return ("https://mast.stsci.edu/search/ui/#/jwst/results?resolve=true"
                "&data_types=spectrum,timeseries,image,other"
                "&instruments=MIRI,NIRCAM,NIRSPEC,NIRISS,FGS"
                f"&program_id={int(self.program)}&useStore=false")

    @property
    def release_url(self) -> str:
        """Field overview page on the JWST-GC portal."""
        return f"{RELEASE_BASE}/{self.field}.html"

    @property
    def images_list_url(self) -> str:
        return f"{RELEASE_BASE}/{self.field}_images.txt"

    @property
    def catalogs_list_url(self) -> str:
        return f"{RELEASE_BASE}/{self.field}_catalogs.txt"

    def product_glob(self, basepath: str = "/orange/adamginsburg/jwst") -> str:
        # wildcard after the obs number so combined tiles (jw05365-o002-998_t001...) match,
        # not only the bare-obsid single-observation products.
        inst = self.instrument.lower()
        return (f"{basepath}/{self.field}/*/pipeline/"
                f"{self.obsid}*_t001_{inst}_*i2d.fits")


# --------------------------------------------------------------------------- discovery
# Fetch failures recorded by _read_lines since the last discover_from_release() call.
# A failed fetch returns [] (so partial discovery still works) but must be LOUD:
# consumers (make_issues) check this to distinguish "manifest genuinely lists
# nothing" from "the network fetch errored" and abort rather than sync an empty
# registry (live issue #4: silent [] made the weekly sync a stale no-op).
LAST_FETCH_ERRORS: List[str] = []


def _read_lines(url_or_path):
    """Read a newline-delimited list from a local path or URL.

    Empty on fetch failure, but never silently: failures print to stderr and append
    to :data:`LAST_FETCH_ERRORS`.
    """
    if os.path.exists(url_or_path):
        with open(url_or_path) as fh:
            return [ln.strip() for ln in fh if ln.strip()]
    req = urllib.request.Request(url_or_path, headers={"User-Agent": "jwst-gc-data-qa"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return [ln.strip() for ln in r.read().decode().splitlines() if ln.strip()]
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError) as ex:
        msg = f"fetch FAILED: {url_or_path}: {ex.__class__.__name__}: {ex}"
        print(f"data_qa.observations: {msg}", file=sys.stderr)
        LAST_FETCH_ERRORS.append(msg)
        return []


def discover_from_release(base: str = RELEASE_BASE, fields: Dict[str, str] = None,
                          local_dir: str = None) -> List["Observation"]:
    """Build the observation list from each field's released image manifest.

    One :class:`Observation` per (field, program, obs, instrument); its filters are every
    filter with a released science mosaic.  Curated metadata is overlaid by obsid.
    ``local_dir`` reads ``{local_dir}/{field}_images.txt`` instead of fetching (offline).
    """
    fields = fields or FIELDS
    LAST_FETCH_ERRORS.clear()
    grouped: Dict[tuple, set] = {}
    merged: Dict[tuple, set] = {}
    for fld, disp in fields.items():
        src = (os.path.join(local_dir, f"{fld}_images.txt") if local_dir
               else f"{base}/{fld}_images.txt")
        for ln in _read_lines(src):
            low = ln.lower()
            if "_i2d.fits" not in low or "resbgsub" in low:
                continue                       # science mosaics only (skip residual/model)
            m = _MOSAIC_RE.search(low)
            if not m:
                continue
            prog, ob, tile, inst, filt = m.groups()
            key = (fld, disp, str(int(prog)), ob, inst.lower())
            grouped.setdefault(key, set()).add(filt.upper())
            if tile:                                   # combined-tile obsid: jw..-oOOO-TTT
                merged.setdefault(key, set()).add(tile)

    out = []
    for (fld, disp, prog, ob, inst), filts in sorted(grouped.items()):
        oid = f"jw{int(prog):05d}-o{ob}"
        cur = CURATED.get(oid, {})
        out.append(Observation(
            program=prog, obs=ob, target=disp, release_field=fld,
            instrument="NIRCam" if inst == "nircam" else "MIRI",
            filters=sorted(filts), visits=cur.get("visits", []),
            epoch=cur.get("epoch", ""), notes=cur.get("notes", ""),
            merged_obsids=sorted(merged.get((fld, disp, prog, ob, inst.lower()), set())),
        ))
    return out


def registry(programs=None, target=None, local_dir=None) -> List["Observation"]:
    """Discovered observations, optionally filtered by program id(s) / target name."""
    obs = discover_from_release(local_dir=local_dir)
    if programs:
        progs = {str(int(p)) for p in programs}
        obs = [o for o in obs if o.program in progs]
    if target:
        t = target.lower()
        obs = [o for o in obs if o.target.lower() == t or o.field == t]
    return obs
