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
import urllib.error
import urllib.request

from .observations import Observation, registry

REPO = os.environ.get("QA_REPO", "JWST-GC/data-qa")
API = "https://api.github.com"

# Marker so we can recognize (and update) an auto-generated body without clobbering
# human discussion, which lives in the comments, not the body.
AUTOGEN_MARKER = "<!-- data-qa:autogen -->"


# --------------------------------------------------------------------------- release links
FILTER_TOKEN = re.compile(r"^(f\d{3}[wnm])[_-]")


def _fetch_lines(url):
    """Best-effort fetch of a newline-delimited URL list. Empty on any failure."""
    req = urllib.request.Request(url, headers={"User-Agent": "jwst-gc-data-qa"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return [ln.strip() for ln in r.read().decode().splitlines() if ln.strip()]
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError):
        return []


def release_links(o: Observation):
    """Filter the field's authoritative release lists down to THIS observation.

    Returns (mosaics, per_filter_catalogs, field_catalogs):
      - mosaics: [(FILTER, url)] science i2d mosaics carrying this obsid
      - per_filter_catalogs: [(FILTER, url)] vetted catalogs for this obs's filters
      - field_catalogs: [url] field-level catalogs (merged/seed; shared by the field)
    """
    filt_lower = [f.lower() for f in o.filters]
    mosaics = []
    for u in _fetch_lines(o.images_list_url):
        low = u.lower()
        if o.obsid.lower() in low and "_i2d.fits" in low and "resbgsub" not in low:
            filt = next((f.upper() for f in filt_lower if f in low), "?")
            mosaics.append((filt, u))
    per_filter, field_cats = [], []
    for u in _fetch_lines(o.catalogs_list_url):
        base = u.rsplit("/", 1)[-1].lower()
        m = FILTER_TOKEN.match(base)
        if m:
            tok = m.group(1)
            if tok in filt_lower:                 # per-filter catalog for THIS obs
                per_filter.append((tok.upper(), u))
            # else: another observation's filter -> skip
        else:
            field_cats.append(u)                  # field-level (merged/seed)
    mosaics.sort()
    per_filter.sort()
    return mosaics, per_filter, field_cats


def _fmt_links(items):
    """items: [(label, url)] -> markdown bullets; '' if empty."""
    return "\n".join(f"  - [{lab}]({url})" for lab, url in items)


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


def _ck(cond) -> str:
    return "x" if cond else " "


def render_body(o: Observation) -> str:
    M = _qa_metrics(o)
    s1, s2, s3, s4 = (M.get(f"stage{n}", {}) for n in (1, 2, 3, 4))
    THRESH_ABS, THRESH_IM = 75.0, 15.0     # mas; mirror data_qa.astrometry_audit
    delivered = bool(s1.get("passed"))
    frame_ok = s4.get("bulk_off") is not None and s4["bulk_off"] < THRESH_ABS
    interm_ok = s4.get("intermodule_off") is not None and s4["intermodule_off"] < THRESH_IM
    phot_ok = bool(s3.get("passed"))
    catalog_ok = bool(s2.get("passed"))

    filt_rows = "\n".join(f"  - [ ] `{f}` — mosaic reviewed; astrometry + photometry OK"
                          for f in o.filters) or "  - (filters TBD)"
    visits = ", ".join(o.visits) or "—"
    notes = f"\n> **Notes:** {o.notes}\n" if o.notes else ""

    mosaics, per_filter, field_cats = release_links(o)
    dl = []
    if mosaics:
        dl.append("**Mosaics (`i2d`):**\n" + _fmt_links(mosaics))
    if per_filter:
        dl.append("**Per-filter catalogs (vetted):**\n" + _fmt_links(per_filter))
    if field_cats:
        dl.append("**Field catalogs:**\n"
                  + _fmt_links((u.rsplit("/", 1)[-1], u) for u in field_cats))
    downloads = ("\n\n".join(dl) if dl
                 else f"_(no release files listed yet — see {o.release_url})_")

    return f"""{AUTOGEN_MARKER}
**Observation `{o.obsid}`** — {o.target} / {o.instrument}

| field | value |
|-------|-------|
| Program | `{int(o.program)}` |
| Observation | `{o.obs}` (`{o.obsid}`) |
| Target | {o.target} |
| Instrument | {o.instrument} |
| Filters | {", ".join(f"`{f}`" for f in o.filters) or "—"} |
| Executions (visits) | {visits} |
| Epoch (DATE-OBS) | {o.epoch or "—"} |

### Release
- Release page: {o.release_url}
- MAST program: {o.mast_program_url}
- On-disk mosaics: `{o.product_glob()}`

### Direct downloads
{downloads}
{notes}
### QA checklist
<sub>boxes with a ✓ are auto-set from the diagnostic replies below (`data_qa.diagnostics`); the rest are manual.</sub>
- [{_ck(delivered)}] Observation delivered / retrieved
- [{_ck(delivered)}] Per-filter mosaics (`i2d`) present and complete
{filt_rows}
- [{_ck(frame_ok)}] **Astrometry**: absolute frame tie (VIRAC2/Gaia) within survey noise
- [{_ck(interm_ok)}] **Astrometry**: no inter-module (NRCA/NRCB) offset (proper-motion grade)
- [{_ck(phot_ok)}] **Photometry**: zeropoints consistent across filters/modules
- [ ] Background / stripes / artifacts acceptable
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


# --------------------------------------------------------------------------- GitHub API
def _req(method, url, token, data=None):
    body = json.dumps(data).encode() if data is not None else None
    req = urllib.request.Request(url, data=body, method=method)
    req.add_header("Authorization", f"token {token}")
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("User-Agent", "jwst-gc-data-qa")
    if body is not None:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode() or "{}")


def existing_issues(token, repo):
    """title -> issue dict, over all states (paginated)."""
    out, page = {}, 1
    while True:
        url = f"{API}/repos/{repo}/issues?state=all&per_page=100&page={page}"
        status, data = _req("GET", url, token)
        if status != 200 or not data:
            break
        for it in data:
            if "pull_request" in it:      # issues endpoint also returns PRs
                continue
            out[it["title"]] = it
        if len(data) < 100:
            break
        page += 1
    return out


def ensure_labels(token, repo, names):
    """Create any missing labels (best-effort; ignores 'already exists')."""
    palette = {"QA": "0e8a16", "NIRCam": "1d76db", "MIRI": "5319e7"}
    for n in names:
        color = palette.get(n, "ededed")
        _req("POST", f"{API}/repos/{repo}/labels", token,
             {"name": n, "color": color})


# --------------------------------------------------------------------------- main
def sync_observation(o, token, repo, existing, dry_run=False):
    title, body, labels = o.issue_title, render_body(o), labels_for(o)
    if title in existing:
        it = existing[title]
        num = it["number"]
        if dry_run:
            return f"UPDATE #{num}: {title}"
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
