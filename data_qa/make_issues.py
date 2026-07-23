"""Create/refresh the per-observation QA tracking issues on GitHub.

For each registered :class:`~data_qa.observations.Observation` this renders a filled
QA template (metadata + links to the data products + a QA checklist) and creates a
GitHub issue for it.  Idempotent: keyed on the issue title, so re-running updates the
existing issue body instead of duplicating.  Stdlib-only (urllib) so it runs in CI with
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
    """Load the per-obs diagnostic metrics (written by ``data_qa.diagnostics``) that drive
    checkbox state.  Absent file -> empty dict -> every box renders unchecked (as before)."""
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


def _ck(cond) -> str:
    return "x" if cond else " "


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


def render_body(o: Observation) -> str:
    M = _qa_metrics(o)
    s1, s2, s3, s4, s5 = (M.get(f"stage{n}", {}) for n in (1, 2, 3, 4, 5))
    from . import astrometry_audit as aa
    THRESH_ABS, THRESH_IM = aa.THRESH["absolute"], aa.THRESH["intermodule"]
    delivered = bool(s1.get("passed"))
    frame_ok = s4.get("bulk_off") is not None and s4["bulk_off"] < THRESH_ABS
    # inter-module: prefer stage 5's reference-free overlap offset, else stage 4's.  Absent =
    # 'not yet measured' -> left unchecked (the sticky-merge won't downgrade a prior check).
    im = s5.get("intermodule_off", s4.get("intermodule_off"))
    interm_ok = im is not None and im < THRESH_IM
    phot_ok = bool(s3.get("passed"))
    catalog_ok = bool(s2.get("passed"))

    filt_rows = "\n".join(f"  - [ ] `{f}` — mosaic reviewed; astrometry + photometry OK"
                          for f in o.filters) or "  - (filters TBD)"
    visits = ", ".join(o.visits) or "—"
    notes = f"\n> **Notes:** {o.notes}\n" if o.notes else ""
    guidestar = _guidestar_block(o)

    # combined-tile note: released mosaics carry a merged obsid (jw..-oOOO-TTT), so say so
    merged_note = (f" (mosaic merges obs {o.obs} + {' + '.join(o.merged_obsids)}; "
                   f"product id `{o.mosaic_obsid}`)" if o.merged_obsids else "")

    return f"""{AUTOGEN_MARKER}
**Observation `{o.obsid}`** — {o.target} / {o.instrument}

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
- On-disk mosaics: `{o.product_glob()}`

{guidestar}{notes}
### QA checklist
<sub>boxes with a ✓ are auto-set from the diagnostic replies below (`data_qa.diagnostics`); the rest are manual.</sub>
- [{_ck(delivered)}] Observation delivered / retrieved
- [{_ck(delivered)}] Per-filter mosaics (`i2d`) present and complete
{filt_rows}
- [{_ck(frame_ok)}] **Astrometry**: absolute frame tie (VIRAC2/Gaia) within survey noise
- [{_ck(interm_ok)}] **Astrometry**: no inter-module (NRCA/NRCB) offset (proper-motion grade)
- [{_ck(phot_ok)}] **Photometry**: zeropoints consistent across filters/modules
- [ ] Background / stripes / artifacts acceptable
- [ ] **Destreak**: assessed whether 1/f striping requires destreak (SW/LW per module); noted decision (cataloging defaults to the plain `align` crf products)
- [{_ck(catalog_ok)}] Catalog produced and vetted
- [{_ck(catalog_ok)}] **Depth**: detection luminosity functions reach the expected depth (not missing stars we should be detecting)
- [ ] **Purity**: minimal junk detections in PSF wings and in extended-emission regions
- [ ] **Residuals**: PSF-subtracted residual histogram is narrow and centered on zero (no systematic over/under-subtraction)
- [ ] Known issues triaged (comment below)

---
*Auto-generated by `data_qa/make_issues.py` from the observation registry. Metadata is
kept in sync on re-runs; **discuss issues in the comments** (the body is overwritten).*
"""


def labels_for(o: Observation):
    return ["QA", o.instrument, f"program:{int(o.program)}", f"target:{o.target}"]


# --------------------------------------------------------------------------- main
# GitHub API plumbing (_req/existing_issues/ensure_labels) lives in data_qa._github
# (shared with status_report.py); behavior is unchanged.
_CK_LINE = re.compile(r"^(\s*- \[)([ xX])(\] )(.*)$")


def _sticky_checkboxes(new_body: str, old_body: str) -> str:
    """Carry checked marks from the CURRENT remote body into the regenerated body.

    The body is machine-overwritten every run, which otherwise (a) unchecks every
    metrics-derived box on the scheduled CI run (which has no cluster ``metrics/`` file) and
    (b) clobbers boxes a human ticked.  Rule: a box CHECKED in either the new render or the
    remote body stays checked (sticky/union), keyed on the checklist label text.  Never
    unchecks -- a regression is surfaced in the diagnostic reply, not by silently unticking.
    """
    old_checked = set()
    for ln in (old_body or "").splitlines():
        m = _CK_LINE.match(ln)
        if m and m.group(2) in "xX":
            old_checked.add(m.group(4).strip())
    out = []
    for ln in new_body.splitlines():
        m = _CK_LINE.match(ln)
        if m and m.group(2) == " " and m.group(4).strip() in old_checked:
            ln = f"{m.group(1)}x{m.group(3)}{m.group(4)}"
        out.append(ln)
    return "\n".join(out)


def sync_observation(o, token, repo, existing, dry_run=False):
    title, body, labels = o.issue_title, render_body(o), labels_for(o)
    if title in existing:
        it = existing[title]
        num = it["number"]
        if dry_run:
            return f"UPDATE #{num}: {title}"
        body = _sticky_checkboxes(body, it.get("body", ""))     # preserve human + prior marks
        _req("PATCH", f"{API}/repos/{repo}/issues/{num}", token,
             {"body": body, "labels": labels})
        return f"updated #{num}: {title}"
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
    for o in obs:
        print(sync_observation(o, token, args.repo, existing, dry_run=args.dry_run))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
