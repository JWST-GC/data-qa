"""Registry of JWST Galactic Center observations.

This is the single source of truth that drives the QA tracking issues
(``make_issues.py``) and the data retrieval (``retrieve_data.py``).  One
:class:`Observation` == one JWST observation (a program's obs-number) == one
tracking issue.  Executions (visits) are listed inside the observation.

Add a new observation by appending to :data:`OBSERVATIONS` (or, for bulk
discovery, use :func:`discover_from_mast`).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

# JWST-GC public data-release portal. Per-field pages live at <base>/<field>.html and
# the authoritative direct-download URL lists at <base>/<field>_{images,catalogs}.txt
# (those point at the Globus-hosted FITS). make_issues fetches + filters them per obs.
RELEASE_BASE = "https://starformation.astro.ufl.edu/jwst-gc"


@dataclass(frozen=True)
class Observation:
    program: str                 # JWST program id, e.g. "2221" (no leading jw/zeros)
    obs: str                     # observation number, zero-padded 3 char, e.g. "001"
    target: str                  # human name, e.g. "Brick"
    instrument: str = "NIRCam"   # JWST instrument
    filters: List[str] = field(default_factory=list)   # e.g. ["F182M", "F187N", ...]
    visits: List[str] = field(default_factory=list)    # executions, e.g. ["001", "002"]
    epoch: str = ""              # observation date (DATE-OBS), e.g. "2022-08-28"
    notes: str = ""              # free-text (known issues, links to related datasets)

    @property
    def obsid(self) -> str:
        """Canonical JWST observation id, e.g. 'jw02221-o001'."""
        return f"jw{int(self.program):05d}-o{self.obs}"

    @property
    def issue_title(self) -> str:
        """Stable title used as the idempotency key for the tracking issue."""
        return f"{self.target} — {self.obsid} ({self.instrument})"

    # ---- links to the underlying data products ----
    @property
    def mast_program_url(self) -> str:
        return (f"https://www.stsci.edu/cgi-bin/get-proposal-info?"
                f"id={int(self.program)}&observatory=JWST")

    @property
    def mast_search_url(self) -> str:
        # deep link into the MAST portal filtered to this program
        return f"https://mast.stsci.edu/search/ui/#/jwst?proposal_id={int(self.program)}"

    @property
    def field(self) -> str:
        """Release-page basename for this dataset (target name, lowercased)."""
        return self.target.lower()

    @property
    def release_url(self) -> str:
        """The public release page for this field on the JWST-GC portal."""
        return f"{RELEASE_BASE}/{self.field}.html"

    @property
    def images_list_url(self) -> str:
        """Authoritative list of this field's mosaic-image direct-download URLs."""
        return f"{RELEASE_BASE}/{self.field}_images.txt"

    @property
    def catalogs_list_url(self) -> str:
        """Authoritative list of this field's catalog direct-download URLs."""
        return f"{RELEASE_BASE}/{self.field}_catalogs.txt"

    def product_glob(self, basepath: str = "/orange/adamginsburg/jwst") -> str:
        """glob for the combined per-filter i2d mosaics of this observation on disk."""
        field_dir = self.target.lower()
        return (f"{basepath}/{field_dir}/*/pipeline/"
                f"{self.obsid}_t001_nircam_clear-*-merged_i2d.fits")


# ---------------------------------------------------------------------------
# Registry.  Seeded with the two Brick datasets; extend as products are made.
# (jw02221-o002 is Cloud C, jw01182 has many non-Brick observations across the
# Galactic plane -- add those as their products are produced/registered.)
# ---------------------------------------------------------------------------
OBSERVATIONS: List[Observation] = [
    Observation(
        program="2221", obs="001", target="Brick", instrument="NIRCam",
        filters=["F182M", "F187N", "F212N", "F405N", "F410M", "F466N"],
        visits=["001"], epoch="2022-08-28",
        notes="Narrow/medium-band NIRCam. F410M/NRCA5 carries a known per-module "
              "distortion+filteroffset offset corrected via a per-module split in the "
              "locked offsets table.",
    ),
    Observation(
        program="1182", obs="004", target="Brick", instrument="NIRCam",
        filters=["F115W", "F200W", "F356W", "F444W"],
        visits=["001", "002"], epoch="2022-09-14",
        notes="Wide-band NIRCam. LW filteroffset module-swap in the 2024 cal was fixed "
              "2026-07; a residual ~20 mas inter-module offset is addressed by a "
              "per-module 2-shift tie.",
    ),
]


def registry(programs=None, target=None) -> List[Observation]:
    """Return registered observations, optionally filtered by program id(s)/target."""
    obs = OBSERVATIONS
    if programs:
        progs = {str(int(p)) for p in programs}
        obs = [o for o in obs if o.program in progs]
    if target:
        obs = [o for o in obs if o.target.lower() == target.lower()]
    return obs


def discover_from_mast(program, instrument="NIRCam"):
    """Discover observations for a program from MAST (astroquery).  Returns a list of
    :class:`Observation` with filters/visits/epoch populated from the product table.

    This is the hook for auto-registering NEW observations: run it, diff against the
    static registry, and append what is missing.  Kept import-light so the module loads
    without astroquery installed.
    """
    from collections import defaultdict
    from astroquery.mast import Observations as MastObs

    tbl = MastObs.query_criteria(obs_collection="JWST", proposal_id=str(int(program)),
                                 instrument_name=f"{instrument}*")
    by_obs = defaultdict(lambda: {"filters": set(), "visits": set(), "epoch": ""})
    for row in tbl:
        oid = str(row["obs_id"])            # e.g. jw02221-o001_t001_nircam_...
        if "-o" not in oid:
            continue
        obsnum = oid.split("-o")[1][:3]
        rec = by_obs[obsnum]
        filt = str(row["filters"]).upper()
        for f in filt.replace(";", ",").split(","):
            f = f.strip()
            if f.startswith("F") and f not in ("CLEAR", "F"):
                rec["filters"].add(f)
        try:
            rec["epoch"] = str(row["t_min"])   # MJD; caller may convert
        except (KeyError, TypeError):
            pass
    out = []
    for obsnum, rec in sorted(by_obs.items()):
        out.append(Observation(program=str(int(program)), obs=obsnum, target="",
                               filters=sorted(rec["filters"]), visits=sorted(rec["visits"]),
                               epoch=rec["epoch"]))
    return out
