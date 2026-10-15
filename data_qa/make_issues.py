"""Create/refresh the per-observation QA tracking issues on GitHub.

For each registered :class:`~data_qa.observations.Observation` this renders a filled
QA template (metadata + links to the data products + an automatic status table + a human
review checklist) and creates a GitHub issue for it.  Idempotent: keyed on the issue title, so
re-running updates the existing issue body instead of duplicating.  Stdlib-only (urllib) so it runs in CI with
just ``GITHUB_TOKEN``.

Usage:
    python -m data_qa.make_issues --program 2221 1182 --target Brick
    python -m data_qa.make_issues --program 2221 1182 --dry-run
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

from . import observations
from ._github import API, REPO, ensure_labels, existing_issues, request as _req
from .observations import Observation, registry

# Marker so we can recognize (and update) an auto-generated body without clobbering
# human discussion, which lives in the comments, not the body.
AUTOGEN_MARKER = "<!-- data-qa:autogen -->"


# --------------------------------------------------------------------------- body
def _qa_metrics(o: Observation) -> dict:
    """Load the per-obs diagnostic metrics (written by ``data_qa.diagnostics``) that fill the
    automatic status table.  Absent file (e.g. the scheduled CI run, which has no cluster
    ``metrics/``) -> empty dict -> :func:`sync_observation` keeps the table already on the issue."""
    path = os.path.join(os.path.dirname(__file__), "metrics", f"{o.obsid}.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


def _guidestar_json():
    path = os.path.join(os.path.dirname(__file__), "guidestar.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


def _guidestar_block(o: Observation) -> str:
    gs = _guidestar_json().get(o.obsid)
    if not gs:
        return ""
    gid = gs.get("gdstarid")
    ra, dec = gs.get("gs_ra"), gs.get("gs_dec")
    mag, ver = gs.get("gs_mag"), gs.get("gsc_ver")
    from .guidestar import gsss_webform_url, POS_SOURCE_DOC
    coord = f"{ra:.6f}, {dec:.6f}" if isinstance(ra, (int, float)) and isinstance(dec, (int, float)) else "—"
    lines = ["### Guide star",
             f"- ID (`GDSTARID`): `{gid}`" + (f" · catalog `{ver}`" if ver else ""),
             f"- Position (ICRS): `{coord}`" + (f" · mag `{mag:.2f}`" if isinstance(mag, (int, float)) else ""),
             f"- Resolve full entry / `posSource`: [GSSS webform]({gsss_webform_url(gid)}) "
             f"(HST_ID pre-filled, catalog GSC3.1) → [source-code table]({POS_SOURCE_DOC})"]
    return "\n".join(lines) + "\n"


# JWST-GC Globus guest collection (rooted at /orange/adamginsburg/jwst): its HTTPS data
# plane serves collection-relative paths, so a released product's download URL is
# ``_GLOBUS_HTTPS_BASE + <path relative to _GLOBUS_ROOT>``.  Kept in step with
# scripts/release/stage_release.py in jwst-gc-pipeline.
_GLOBUS_HTTPS_BASE = "https://g-92a536.55ba.08cc.data.globus.org"
_GLOBUS_ROOT = "/orange/adamginsburg/jwst"
_GLOBUS_COLLECTION_ID = "d9873d5e-0fbd-4980-aedf-4ca56f65a045"      # "JWST root" guest collection
# Globus tutorial native client, used only to mint a short-lived HTTPS token; the same recipe
# ships in jwst-gc-pipeline scripts/release/make_webpage.py for the public release.
_GLOBUS_NATIVE_CLIENT = "3b1925c0-a87b-452b-a492-2c9921d3bd14"
# per-detector and intermediate products that are not the released mosaic + its catalogue
_GLOBUS_PROD_EXCLUDE = ("outlier", "_model_", "_residual_", "smoothed_bg",
                        "destreak", "_crf", "_cr_", "segm", "nrca", "nrcb")

# These files are shared with the Treasury team over Globus (a free Globus/ORCID login), NOT a
# public release, so the HTTPS URLs redirect to Globus auth.  This is the command-line recipe to
# fetch them with a short-lived bearer token; it is the same per issue (only the URL list varies).
_GLOBUS_DOWNLOAD_HELP = (
    "These are pipeline products on the JWST-GC Globus collection (`" + _GLOBUS_COLLECTION_ID
    + "`), not a public release, so the URLs redirect to Globus auth.  Access is granted per "
    "field: program 10678 (`gc-treasury`) is shared with the Treasury team group today; other "
    "fields may need a read rule added first, so a 403 after logging in means \"ask for access\" "
    "rather than a broken link.  Fetch the files from the command line with a short-lived bearer "
    "token:\n\n"
    "```\n"
    "pip install globus-sdk                 # once\n"
    "python - <<'PY'                        # opens an ORCID/Globus login; prints a ~48 h token\n"
    "import globus_sdk\n"
    'C = "' + _GLOBUS_COLLECTION_ID + '"\n'
    'cl = globus_sdk.NativeAppAuthClient("' + _GLOBUS_NATIVE_CLIENT + '")\n'
    'cl.oauth2_start_flow(requested_scopes=f"https://auth.globus.org/scopes/{C}/https")\n'
    'print("Log in here:", cl.oauth2_get_authorize_url())\n'
    'tok = cl.oauth2_exchange_code_for_tokens(input("code: ").strip())\n'
    'print("TOKEN:", tok.by_resource_server[C]["access_token"])\n'
    "PY\n"
    "TOKEN=<paste the printed token>\n"
    'wget --header="Authorization: Bearer $TOKEN" -i urls.txt      # urls.txt = the list below\n'
    "```\n\n"
    "Or transfer with the Globus CLI: `globus login`, then `globus transfer "
    + _GLOBUS_COLLECTION_ID + ":<path> <your-endpoint>:<dest>`."
)


def _globus_products(o: Observation):
    """``(filter, kind, url)`` for this obs's pipeline i2d + catalogue files, as Globus HTTPS
    download URLs, from the on-disk pipeline dir.  ``kind`` is ``i2d`` or ``catalog``; one of
    each per filter, preferring the merged mosaic + merged catalogue and skipping per-detector
    and intermediate products."""
    import glob
    inst = o.instrument.lower()
    rows = []
    for f in o.filters:
        pdir = f"{_GLOBUS_ROOT}/{o.field}/{f}/pipeline"
        for kind, suffix in (("i2d", "i2d.fits"), ("catalog", "cat.ecsv")):
            hits = [p for p in glob.glob(f"{pdir}/{o.obsid}*_t001_{inst}_*_{suffix}")
                    if not any(t in os.path.basename(p).lower() for t in _GLOBUS_PROD_EXCLUDE)]
            # prefer the merged mosaic, and the plain science i2d over the cataloging
            # ``_data_i2d`` resample, then shortest name.
            hits.sort(key=lambda p: ("merged" not in (b := os.path.basename(p).lower()),
                                     "_data_i2d" in b, len(b), b))
            if hits:
                rel = os.path.relpath(hits[0], _GLOBUS_ROOT)
                rows.append((f, kind, f"{_GLOBUS_HTTPS_BASE}/{rel}"))
    return rows


def _globus_block(o: Observation) -> str:
    """Markdown block: per-filter i2d + catalogue links on the Globus collection, plus a
    collapsed command-line recipe (bearer-token ``wget``) and the plain URL list it consumes.
    The files need a Globus login, so a bare URL is not fetchable without the token."""
    rows = _globus_products(o)
    if not rows:
        return ("### Data files (Globus)\n_No pipeline `i2d`/catalogue on disk yet; this "
                "section fills in once the observation is reduced._\n\n")
    by_filt = {}
    for f, kind, url in rows:
        by_filt.setdefault(f, {})[kind] = url
    lines = ["### Data files (Globus)",
             "Pipeline products for this observation on the JWST-GC Globus collection. Access is "
             "granted per field (10678/`gc-treasury` is shared with the Treasury team; other fields "
             "may need a read rule added). See the download recipe below."]
    for f in o.filters:
        d = by_filt.get(f)
        if d:
            parts = [f"[{k}]({d[k]})" for k in ("i2d", "catalog") if k in d]
            lines.append(f"- `{f}` — " + " · ".join(parts))
    urls = "\n".join(url for _, _, url in rows)
    lines.append("\n<details><summary>Download from the command line</summary>\n\n"
                 + _GLOBUS_DOWNLOAD_HELP
                 + "\n\n**`urls.txt`:**\n```\n" + urls + "\n```\n</details>\n")
    return "\n".join(lines) + "\n\n"


# The JWST-GC Aladin viewer (all-survey HiPS overlays).  A plain link: the page centres
# itself from its own preset buttons and does not read URL coordinates, so the issue links
# the viewer rather than claiming a per-field centring the page does not do.  Per-field
# centring is a follow-up that first brings the page under version control (it is a
# hand-maintained file in the live docroot today).
_ALADIN_PAGE = "https://starformation.astro.ufl.edu/avm_images/jwst_gc_aladin.html"


# --------------------------------------------------------------------------- status + review
# JWST-GC/data-qa#364: automatic flags and product availability go in a status TABLE (no
# checkboxes, so GitHub's task count covers human review only); the checklist holds human review
# actions alone.  A checkmark means the review action is complete -- a review that found a problem
# is still complete.  No automatic flag ever ticks a review box.
STATUS_START = "<!-- data-qa:status -->"
STATUS_END = "<!-- /data-qa:status -->"
_STATUS_RE = re.compile(re.escape(STATUS_START) + r".*?" + re.escape(STATUS_END), re.S)
LEGACY_START = "<!-- data-qa:legacy-checklist -->"
LEGACY_END = "<!-- /data-qa:legacy-checklist -->"
_LEGACY_RE = re.compile(re.escape(LEGACY_START) + r".*?" + re.escape(LEGACY_END), re.S)

# (key, label, evidence stages).  The key is written into the line as a hidden ``<!-- qa:key -->``
# so a tick survives any later rewording of the label.  Wording follows #364.
_REVIEW_ROWS_NIRCAM = (
    ("images", "**Inspect the images:** reviewed coverage, background, stripes and artifacts in "
               "each filter, including whether destreaking is needed", ("1", "7")),
    ("population", "**Inspect the source population:** reviewed CMDs and luminosity functions for "
                   "unusual structure or unexpected depth", ("2",)),
    ("photometry", "**Check photometric behaviour:** reviewed reference calibration, PSF/aperture "
                   "agreement and brightness-dependent trends in the available filters",
     ("3", "9", "12")),
    ("frame", "**Check reference-frame alignment:** reviewed bulk and spatial offsets for the "
              "delivered and pipeline catalogs", ("4", "7")),
    ("internal", "**Check internal alignment:** reviewed detector/module agreement and "
                 "inter-filter/intra-detector residual patterns", ("5", "8")),
    ("precision", "**Assess astrometric precision:** reviewed empirical repeatability versus "
                  "magnitude, formal errors and external scatter", ("6",)),
    ("jwst1pass", "**Review JWST1PASS consistency:** inspected position/magnitude repeatability, "
                  "fit quality, saturation behaviour and PSF perturbations", ("10",)),
    ("psfs", "**Inspect exposure PSFs:** checked for exposure/chip anomalies that may require "
             "exclusion or down-weighting", ("11",)),
    ("fitting", "**Inspect fitting and detections:** reviewed the iteration animation and the "
                "residual maps by tool for missed/spurious sources and subtraction problems",
     ("14", "15")),
    ("neighbours", "**Check neighbouring tiles:** reviewed overlap astrometry/photometry for "
                   "disagreements", ("13",)),
    ("triage", "**Triage known issues:** addressed concerns raised during review and linked any "
               "needed follow-up issues", ()),
)
# MIRI issues carry only the MIRI overview; their review actions are scoped to it (#364: other
# instruments need their own applicable rows).
_REVIEW_ROWS_MIRI = (
    ("images", "**Inspect the images:** reviewed coverage, background, saturation and artifacts "
               "in each filter against the Spitzer comparison", ("miri",)),
    ("triage", "**Triage known issues:** addressed concerns raised during review and linked any "
               "needed follow-up issues", ()),
)


def _review_rows(o: Observation):
    return _REVIEW_ROWS_MIRI if o.instrument == "MIRI" else _REVIEW_ROWS_NIRCAM


def _stage_refs(stages, stage_urls):
    if not stages:
        return ""
    refs = [f"[{'MIRI overview' if t == 'miri' else t}]({stage_urls[t]})"
            if stage_urls and t in stage_urls else ("MIRI overview" if t == "miri" else t)
            for t in stages]
    return " (stage " + ", ".join(refs) + ")" if stages != ("miri",) else " (" + refs[0] + ")"


def _review_block(o: Observation, stage_urls=None) -> str:
    rows = "\n".join(f"- [ ] {label}{_stage_refs(stages, stage_urls)} <!-- qa:{key} -->"
                     for key, label, stages in _review_rows(o))
    return ("### Human review\n"
            "<sub>A checkmark means the review action is complete; a review that found a problem "
            "is still complete (say what you found in a comment). Leave a blocked review "
            "unchecked. To mark a row inapplicable, replace its `[ ]` with `N/A`; N/A rows drop "
            "out of the task count. Review every applicable filter and the expandable figures in "
            "each linked comment. The reviewer and date are recorded on the next sync, and a box "
            "is unticked with a note when the evidence behind it changes.</sub>\n" + rows)


def _fmt(v, nd=1):
    return "—" if v is None else f"{v:.{nd}f}"


def _status_block(o: Observation, M: dict, f: dict) -> str:
    """Automatic status table: product availability and diagnostic flags, never review sign-off.
    Each state is ✅ (flag met / present), ⚠️ (flag not met / absent), ⏳ (not yet measured)."""
    if o.instrument == "MIRI":
        rows = [("MIRI overview built", "✅" if M.get("miri") else "⏳ not yet built", "MIRI overview")]
    else:
        s1, s2, s3, s4 = (M.get(f"stage{n}", {}) for n in (1, 2, 3, 4))

        def st(stage, ok, good, bad):
            return "⏳ not yet measured" if not stage else (f"✅ {good}" if ok else f"⚠️ {bad}")
        dropped = s1.get("dropped_filters") or []
        n2 = s2.get("n_stars")
        im, thr = f["im"], f["thresh_im"]
        rows = [
            ("Observation delivered; per-filter mosaics on disk",
             st(s1, f["delivered"], "present",
                "missing " + (", ".join(dropped) if dropped else "mosaic(s)")), "stage 1"),
            ("jicama merged catalogue", st(M, f["jicama"], "present", "not found"),
             "stages 3, 4, 7"),
            ("JWST1PASS products", st(M, f["jwst1pass"], "present", "not found"), "stage 10"),
            ("peppar products", st(M, f["peppar"], "present", "not found"), "stages 6, 11"),
            ("CMD source count",
             st(s2, f["catalog_ok"], f"{n2:,} sources" if n2 else "flag met",
                f"{n2:,} sources" if n2 else "flag not met"),
             "stage 2; flag = more than 500 finite magnitude pairs, not a depth or completeness test"),
            ("Short-wave photometry vs VIRAC Ks",
             st(s3, f["phot_ok"], f"slope {_fmt(s3.get('slope'), 3)}, scatter "
                f"{_fmt(s3.get('scatter'), 3)} mag (Vega)", "flag not met"),
             f"stage 3 ({s3.get('sw') or 'SW'} only); not a cross-filter or cross-module calibration"),
            ("Registration onto VIRAC2/Gaia",
             st(s4, f["frame_ok"], f"field offset {_fmt(s4.get('bulk_off'))} mas, cells consistent",
                f"field offset {_fmt(s4.get('bulk_off'))} mas"
                + ("" if s4.get("spatial_assessed", True) else ", per-cell check not run")),
             f"stage 4 (positions: {s4.get('source') or '—'})"),
            ("NRCA–NRCB overlap offset",
             "⏳ not yet measured" if im is None else
             (f"✅ {im:.1f} mas, below the {thr:.0f} mas flag threshold" if im < thr else
              f"⚠️ {im:.1f} mas, above the {thr:.0f} mas flag threshold"),
             "stage 5 (else stage 4); a measured offset against a threshold, not a proper-motion "
             "certification"),
            ("Cross-module / cross-tile zero points",
             "aggregate only",
             "[stage 13 zero-point network](https://github.com/JWST-GC/data-qa/blob/main/docs/"
             "qa_methods.md#stage13-zeropoints); not inferred from stage 3"),
        ]
    table = "\n".join(f"| {a} | {b} | {c} |" for a, b, c in rows)
    return (f"{STATUS_START}\n### Automatic status\n<sub>Product availability and automatic "
            f"diagnostic flags, from `data_qa.diagnostics`. Flags are suggestions for the reviewer "
            f"and never tick a review box.</sub>\n\n| item | status | basis |\n|---|---|---|\n"
            f"{table}\n{STATUS_END}")


# ---- review-state merge across regeneration
_ROW_RE = re.compile(r"^\s*- (?:\[(?P<ck>[ xX])\]\s*)?(?P<na>N/?A\b)?.*<!-- qa:(?P<key>[\w-]+) -->",
                     re.I)
_REV_RE = re.compile(r"<!-- qa-rev:(?P<key>[\w-]+)\|(?P<by>[^|]*)\|(?P<at>[^|]*)\|(?P<ev>[^|]*) -->")
_STALE_RE = re.compile(r"<!-- qa-stale:(?P<key>[\w-]+)\|(?P<by>[^|]*)\|(?P<at>[^|]*) -->")


def _row_states(body):
    """{key: dict(state='x'|' '|'na', rev=(by, at, ev) or None, stale=(by, at) or None)}."""
    out = {}
    for ln in (body or "").splitlines():
        m = _ROW_RE.match(ln)
        if not m:
            continue
        state = "na" if m.group("na") else ("x" if (m.group("ck") or " ") in "xX" else " ")
        r, s_ = _REV_RE.search(ln), _STALE_RE.search(ln)
        out[m.group("key")] = dict(state=state,
                                   rev=(r.group("by"), r.group("at"), r.group("ev")) if r else None,
                                   stale=(s_.group("by"), s_.group("at")) if s_ else None)
    return out


def _evidence_version(M, stages):
    """Short hash of the inputs behind ``stages`` (path, size, mtime of every file the diagnostics
    recorded reading), or None when no metrics are at hand (scheduled CI run).  A change means the
    reviewed products changed, so the review has to be redone."""
    import hashlib
    parts = []
    for t in stages:
        st = M.get(t if t == "miri" else f"stage{t}") or {}
        if not st:
            continue
        files = []
        for rec in st.get("inputs") or []:
            path = rec.get("path") if isinstance(rec, dict) else rec
            if not path:
                continue
            try:
                fs = os.stat(path)
                files.append(f"{path}\0{fs.st_size}\0{fs.st_mtime_ns}")
            except OSError:
                files.append(f"{path}\0missing")
        parts.append(f"{t}\0" + "\0".join(sorted(files)))
    if not parts:
        return None
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()[:12]


def _who_ticked(edits, key):
    """(login, date) of the edit that ticked row ``key``, from the issue's edit history (newest
    first; each edit's ``diff`` is the full body at that version).  The sync never ticks a review
    row, so the edit that turned it on is a person's.  None when the history does not show it."""
    found = None
    for e in edits:
        st = _row_states(e.get("diff") or "").get(key)
        if st and st["state"] == "x":
            found = ((e.get("editor") or {}).get("login") or "unknown", (e.get("editedAt") or "")[:10])
        else:
            break
    return found


def _fetch_edits(token, repo, num):
    owner, name = repo.split("/")
    q = ("query($o:String!,$n:String!,$i:Int!){repository(owner:$o,name:$n){issue(number:$i){"
         "userContentEdits(first:100){nodes{editedAt editor{login} diff}}}}}")
    status, data = _req("POST", f"{API}/graphql", token,
                        {"query": q, "variables": {"o": owner, "n": name, "i": num}})
    if status >= 300 or not isinstance(data, dict) or data.get("errors"):
        return []
    issue = ((data.get("data") or {}).get("repository") or {}).get("issue") or {}
    return (issue.get("userContentEdits") or {}).get("nodes") or []


def _merge_review(new_body, old_body, M, today, row_stages, edits=lambda: []):
    """Carry human review state from the current remote body into the regenerated one.

    Per row (keyed on ``<!-- qa:key -->``): N/A stays N/A (no checkbox); a tick stays ticked with
    a visible reviewer/date stamp and a hidden evidence version; a tick without a stamp (placed by
    a person since the last sync) is stamped now, the reviewer read from the edit history; a
    stamped tick whose evidence version no longer matches the current inputs is unticked with a
    re-review note.  An evidence version is compared only when both sides are known, so a sync
    without metrics never unticks anything.  ``row_stages`` maps a row key to its evidence stages.
    ``edits`` is called lazily (one GraphQL query, only
    when a new tick needs attributing)."""
    old = _row_states(old_body)
    cache = []

    def history():
        if not cache:
            cache.append(edits())
        return cache[0]

    out = []
    for ln in new_body.splitlines():
        m = _ROW_RE.match(ln)
        if not m or m.group("key") not in old:
            out.append(ln)
            continue
        key, st = m.group("key"), old[m.group("key")]
        text = re.sub(r"^\s*- \[ \] ", "", ln)
        ev_now = _evidence_version(M, row_stages.get(key, ()))
        if st["state"] == "na":
            out.append(f"- N/A — {text}")
            continue
        if st["state"] == "x":
            by, at, ev = st["rev"] if st["rev"] else (None, None, None)
            if by is None:
                by, at = _who_ticked(history(), key) or ("unknown", today)
                ev = ev_now or "-"
            if ev not in (None, "-") and ev_now is not None and ev != ev_now:
                out.append(f"- [ ] {text} <sub>⚠️ evidence changed since @{by}'s review on {at}; "
                           f"re-review</sub><!-- qa-stale:{key}|{by}|{at} -->")
                continue
            if ev == "-" and ev_now is not None:
                ev = ev_now
            who = f"@{by}" if by != "unknown" else "reviewer not recorded"
            out.append(f"- [x] {text} <sub>— {who}, {at}</sub><!-- qa-rev:{key}|{by}|{at}|{ev} -->")
            continue
        if st["stale"]:
            by, at = st["stale"]
            out.append(f"- [ ] {text} <sub>⚠️ evidence changed since @{by}'s review on {at}; "
                       f"re-review</sub><!-- qa-stale:{key}|{by}|{at} -->")
            continue
        out.append(ln)
    return "\n".join(out)


def render_body(o: Observation, stage_urls=None) -> str:
    """Issue body for ``o``.  ``stage_urls`` maps a stage tag ("1", "14", "miri") to its diagnostic
    comment URL (parsed from the current body's index by :func:`sync_observation`) so each review
    row links its evidence; without it the rows name the stages unlinked."""
    M = _qa_metrics(o)
    s1, s2, s3, s4, s5 = (M.get(f"stage{n}", {}) for n in (1, 2, 3, 4, 5))
    s6, s7, s10, s11 = (M.get(f"stage{n}", {}) for n in (6, 7, 10, 11))
    from . import astrometry_audit as aa
    THRESH_IM = aa.THRESH["intermodule"]
    delivered = bool(s1.get("passed"))
    # Registration onto VIRAC: use stage 4's own PASS flag, which requires a small field offset
    # AND cells that AGREE.  A bare `bulk_off < THRESH` passes a bimodal frame (gc2211
    # o050: field 57 mas with a 58 mas cell-to-cell spread), which is an internally-inconsistent
    # mosaic and must never show as a pass (PR #54 review).
    # Also require that the per-cell spatial-consistency check actually RAN: a small or sparse
    # field measured WHOLE-FIELD (spatial_assessed False, issue #13) has no per-cell check, so a
    # sub-region discontinuity could hide behind a passing field offset.
    frame_ok = bool(s4.get("passed")) and s4.get("spatial_assessed", True)
    # inter-module: prefer stage 5's overlap offset (JWST against itself), else stage 4's.
    # Absent = 'not yet measured'.
    im = s5.get("intermodule_off", s4.get("intermodule_off"))
    phot_ok = bool(s3.get("passed"))
    catalog_ok = bool(s2.get("passed"))
    # Product-existence flags (from the diagnostics; each stage red-flags when
    # its product is absent, so "not red-flagged" == "product present").
    #  - jicama = the merged/release catalogue: stage 7 says so directly, else stage 4 read positions
    #    from a "release:" catalogue, else stage 3's Vega calibration (which needs the merged cat) ran.
    jicama_present = (bool(s7.get("jicama_is_release"))
                     or str(s4.get("source", "")).startswith("release")
                     or bool(s3.get("passed")))
    #  - JWST1PASS (stage 10): a MATCHUP.XYMEEE / LOG.psfperts product exists.
    jwst1pass_present = bool(s10) and s10.get("available") is not False and not s10.get("red_flag")
    #  - peppar: stage 6 drew a peppar curve (peppar_kind set) or stage 11 built the ePSF grid.
    peppar_present = bool(s6.get("peppar_kind")) or (
        bool(s11) and s11.get("available") is not False and not s11.get("red_flag"))

    # surface nominal filters with no product on disk (stage 1) so an unreduced filter is visible
    dropped = s1.get("dropped_filters") or []
    dropped_note = (f"\n> ⚠️ **No data on disk for nominal filter(s):** "
                    f"{', '.join(f'`{f}`' for f in dropped)} — observed but not reduced, or not "
                    f"delivered. These are excluded from the QA below.\n" if dropped else "")
    visits = ", ".join(o.visits) or "—"
    notes = f"\n> **Notes:** {o.notes}\n" if o.notes else ""
    guidestar = _guidestar_block(o)
    globus = _globus_block(o)

    # combined-tile note: released mosaics carry a merged obsid (jw..-oOOO-TTT), so say so
    merged_note = (f" (mosaic merges obs {o.obs} + {' + '.join(o.merged_obsids)}; "
                   f"product id `{o.mosaic_obsid}`)" if o.merged_obsids else "")

    banner_txt = (observations.CURATED.get(o.obsid, {}).get("banner_nircam")
                  if o.instrument == "NIRCam" else None)
    # quote EVERY line so a multi-paragraph banner stays inside the admonition
    banner = ("> [!WARNING]\n" + "\n".join(f"> {ln}" if ln else ">" for ln in banner_txt.split("\n"))
              + "\n\n") if banner_txt else ""

    return f"""{AUTOGEN_MARKER}
{banner}**Observation `{o.obsid}`** — {o.target} / {o.instrument}

| field | value |
|-------|-------|
| Program | `{int(o.program)}` |
| Observation | `{o.obs}` (`{o.obsid}`){merged_note} |
| Target | {o.target} |
| Instrument | {o.instrument} |
| Filters | {", ".join(f"`{f}`" for f in o.filters) or "—"} |
| Executions (visits) | {visits} |
| Epoch (DATE-OBS) | {o.epoch or "—"} |

### Archive & data
- APT program (PDF): {o.mast_program_url}
- MAST data search: {o.mast_search_url}
- JWST-GC Aladin viewer: {_ALADIN_PAGE}
- On-disk mosaics: `{o.product_glob()}`

{globus}{dropped_note}{guidestar}{notes}
{_status_block(o, M, dict(delivered=delivered, frame_ok=frame_ok, im=im, thresh_im=THRESH_IM,
                          phot_ok=phot_ok, catalog_ok=catalog_ok, jicama=jicama_present,
                          jwst1pass=jwst1pass_present, peppar=peppar_present))}

{_review_block(o, stage_urls)}

---
*Auto-generated by `data_qa/make_issues.py` from the observation registry. Metadata is
kept in sync on re-runs; **discuss issues in the comments** (the body is regenerated; review ticks,
N/A marks and reviewer stamps are kept).*
"""


def labels_for(o: Observation):
    return ["QA", o.instrument, f"program:{int(o.program)}", f"target:{o.target}"]


# --------------------------------------------------------------------------- main
# GitHub API plumbing (_req/existing_issues/ensure_labels) lives in data_qa._github
# (shared with status_report.py); behavior is unchanged.
_CK_LINE = re.compile(r"^(\s*- \[)([ xX])(\] )(.*)$")

# Rows of the checklist used before #364.  Several were ticked by diagnostic flags, the rest by
# hand, and the body never recorded which; their ticks are therefore NOT carried into the review
# rows (#364: no human approval inferred from legacy boxes).  They are listed once, collapsed, so
# nothing a person ticked is lost.
_LEGACY_AUTO = (
    "Observation delivered / retrieved",
    "Per-filter mosaics (`i2d`) present and complete",
    "**Astrometry**: catalogue registered onto VIRAC2/Gaia within survey noise",
    "**Astrometry**: absolute frame tie (VIRAC2/Gaia) within survey noise",
    "**Astrometry**: no inter-module (NRCA/NRCB) offset (proper-motion grade)",
    "**Photometry**: zeropoints consistent across filters/modules",
    "Catalog produced and vetted",
    "**Products**: jicama (merged/release catalogue) present",
    "**Products**: JWST1PASS products present",
    "**Products**: peppar products present",
    "**Depth**: detection luminosity functions reach the expected depth (not missing stars we "
    "should be detecting)",
)


def _legacy_ticker(edits, label):
    """(login, date) of the edit that ticked legacy row ``label``.  The sync never ticked a manual
    row, so for those rows the edit that turned the tick on is a person's."""
    line = f"- [x] {label}"
    found = None
    for e in edits:
        if any(ln.strip().lower().startswith(line.lower()) for ln in (e.get("diff") or "").splitlines()):
            found = ((e.get("editor") or {}).get("login") or "unknown", (e.get("editedAt") or "")[:10])
        else:
            break
    return found


def _legacy_block(old_body, edits=lambda: [], sync_login=None):
    """Collapsed list of the ticks on a pre-#364 checklist, or "" when there are none.  An
    existing legacy block is carried unchanged.  Ticks are attributed from the issue's edit history
    when it shows the tick being placed.  A flag-driven row is attributed only when the sync's own
    account (``sync_login``) is known and the tick came from a different, non-bot account, since
    the sync itself ticked those rows."""
    m = _LEGACY_RE.search(old_body or "")
    if m:
        return m.group(0)
    ticked, cache = [], []
    for ln in (old_body or "").splitlines():
        c = _CK_LINE.match(ln)
        if c and c.group(2) in "xX" and "<!-- qa:" not in ln:
            label = c.group(4).strip()
            if not cache:
                cache.append(edits())
            who = _legacy_ticker(cache[0], label)
            if label in _LEGACY_AUTO:
                ticked.append(f"- {label} — " + (
                    f"ticked by @{who[0]} on {who[1]}"
                    if who and sync_login and who[0] not in (sync_login, "unknown")
                    and not who[0].endswith("[bot]") else "set from a diagnostic flag" + (f" (or by hand from @{sync_login})"
                                                         if sync_login else "")))
                continue
            ticked.append(f"- {label} — " + (f"ticked by @{who[0]} on {who[1]}" if who else
                                             "manual row; who ticked it was not recorded"))
    if not ticked:
        return ""
    return (f"{LEGACY_START}\n<details><summary>Ticks on the previous checklist (before "
            f"JWST-GC/data-qa#364) — not counted as reviews</summary>\n\n"
            + "\n".join(ticked) + f"\n\nThe checklist above starts unticked; re-check each review "
            f"action and tick it there.\n</details>\n{LEGACY_END}")


_INDEX_LINK = re.compile(r"^- \[(Stage (\w+)|MIRI overview)\]\((\S+?)\)", re.M)


def _stage_urls(old_body):
    """{stage tag: comment URL} from the index block ``post_diagnostics.update_index`` maintains."""
    from .post_diagnostics import _INDEX_RE
    m = _INDEX_RE.search(old_body or "")
    if not m:
        return {}
    return {(g2 or "miri"): url for _, g2, url in _INDEX_LINK.findall(m.group(0))}


def _carry_status(new_body, old_body, have_metrics):
    """Without metrics (scheduled CI run) keep the status table already on the issue rather than
    replacing it with an all-pending one."""
    m = _STATUS_RE.search(old_body or "")
    if have_metrics or not m:
        return new_body
    return _STATUS_RE.sub(lambda _: m.group(0), new_body)


def _splice_legacy(new_body, block):
    if not block:
        return new_body
    return new_body.replace("\n---\n*Auto-generated", f"\n{block}\n\n---\n*Auto-generated", 1)


def _carry_index(new_body: str, old_body: str) -> str:
    """Carry the comment-index block (written by ``post_diagnostics.update_index``) from the
    remote body into the regenerated body, which does not know the comment URLs."""
    from .post_diagnostics import _INDEX_RE, splice_index
    m = _INDEX_RE.search(old_body or "")
    return splice_index(new_body, m.group(0)) if m else new_body


def sync_observation(o, token, repo, existing, dry_run=False, today=None, sync_login=None):
    import datetime
    title, labels = o.issue_title, labels_for(o)
    if title in existing:
        it = existing[title]
        num = it["number"]
        if dry_run:
            return f"UPDATE #{num}: {title}"
        old = it.get("body", "")
        body = render_body(o, stage_urls=_stage_urls(old))
        M = _qa_metrics(o)
        today = today or datetime.date.today().isoformat()
        body = _carry_status(body, old, bool(M))
        cache = []

        def history():                                          # one GraphQL query, on demand
            if not cache:
                cache.append(_fetch_edits(token, repo, num))
            return cache[0]
        body = _merge_review(body, old, M, today, {k: st for k, _, st in _review_rows(o)},
                             edits=history)
        body = _splice_legacy(body, _legacy_block(old, edits=history, sync_login=sync_login))
        body = _carry_index(body, old)                          # keep the comment index block
        _req("PATCH", f"{API}/repos/{repo}/issues/{num}", token,
             {"body": body, "labels": labels})
        return f"updated #{num}: {title}"
    body = render_body(o)
    if dry_run:
        return f"CREATE: {title}"
    ensure_labels(token, repo, labels)
    status, data = _req("POST", f"{API}/repos/{repo}/issues", token,
                        {"title": title, "body": body, "labels": labels})
    if status >= 300:
        return f"FAILED ({status}) {title}: {data.get('message')}"
    return f"created #{data['number']}: {title}"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--program", nargs="*", help="program id(s) to include (default: all)")
    ap.add_argument("--target", help="restrict to a target name (e.g. Brick)")
    ap.add_argument("--repo", default=REPO, help=f"owner/name (default {REPO})")
    ap.add_argument("--dry-run", action="store_true", help="print actions, do not call GitHub")
    args = ap.parse_args(argv)

    obs = registry(programs=args.program, target=args.target)
    if not obs:
        if observations.LAST_FETCH_ERRORS:
            # A manifest fetch FAILED and the registry came back empty: this is a
            # network problem, not an empty release.  Refuse to "sync" (which would
            # render stale/empty issue bodies) and exit loudly for CI.
            print("ABORT: registry empty AND manifest fetch(es) failed -- refusing "
                  "to sync an empty registry:", file=sys.stderr)
            for msg in observations.LAST_FETCH_ERRORS:
                print(f"  {msg}", file=sys.stderr)
            return 3
        print("no matching observations in the registry", file=sys.stderr)
        return 1
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if not token and not args.dry_run:
        print("GITHUB_TOKEN not set (use --dry-run to preview)", file=sys.stderr)
        return 2

    existing = existing_issues(token, args.repo) if token else {}
    login = None
    if token and not args.dry_run:
        st, me = _req("GET", f"{API}/user", token)
        login = me.get("login") if st < 300 and isinstance(me, dict) else None
    for o in obs:
        print(sync_observation(o, token, args.repo, existing, dry_run=args.dry_run,
                               sync_login=login))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
