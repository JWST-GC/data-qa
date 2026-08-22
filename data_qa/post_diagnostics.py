"""Post a QA diagnostic figure as an idempotent reply (comment) on the observation's
tracking issue.

The image LIVES IN THE ISSUE, not the repo source tree: the PNG bytes are uploaded to the
GitHub CDN as an asset on a single ``qa-assets`` bucket release, and the comment embeds
that URL.  One comment per (issue, stage), keyed on a hidden marker, so re-running UPDATES
the existing comment (new image + caption) instead of piling up duplicates.

Stdlib-only (urllib) so it runs in CI with just ``GITHUB_TOKEN``.
"""
from __future__ import annotations

import json
import mimetypes
import os
import urllib.error
import urllib.request

from .observations import Observation

API = "https://api.github.com"


class PostError(Exception):
    """A GitHub post/upload step failed; caller decides whether to continue other stages."""
UPLOADS = "https://uploads.github.com"
ASSET_RELEASE_TAG = os.environ.get("QA_ASSET_TAG", "qa-assets")
DIAG_MARKER = "<!-- data-qa:diag:stage{n} -->"

# The function in data_qa/diagnostics.py that builds each stage's figure + numbers, so every
# posted comment can link straight to the source that produced it (the caption already links the
# narrative method doc + glossary terms).
STAGE_FUNC = {
    1: "stage1_mosaics", 2: "stage2_cmd", 3: "stage3_calibration", 4: "stage4_offsets",
    5: "stage5_intermodule", 6: "stage6_astrom_error", 7: "stage7_mast_vs_pipeline",
    8: "stage8_distortion",
    9: "stage9_psf_vs_aper",
    10: "stage10_photometric_consistency", 11: "stage11_effective_psf",
    "6clean": "stage6_astrom_error",          # stage 6 recomputed excluding bad-PSF exposures
    "miri": "miri_overview",
}


def _provenance_footer(repo, stage):
    """One-line 'how this was made' footer: the narrative method doc + the exact source function."""
    base = f"https://github.com/{repo}/blob/main"
    anchor = "stage6" if stage == "6clean" else f"stage{stage}"   # 6clean shares stage 6's doc
    doc = f"{base}/docs/qa_methods.md#{anchor}"
    func = STAGE_FUNC.get(stage)
    src = (f"[`data_qa/diagnostics.py` → `{func}()`]({base}/data_qa/diagnostics.py)"
           if func else f"[`data_qa/diagnostics.py`]({base}/data_qa/diagnostics.py)")
    return f"📖 [how this plot & its numbers are made]({doc}) · 🛠 source: {src}"


def _token():
    tok = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if not tok:
        raise PostError("GITHUB_TOKEN not set")
    return tok


def _req(method, url, token, data=None, headers=None, raw=False):
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Authorization", f"token {token}")
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("User-Agent", "jwst-gc-data-qa")
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    try:
        with urllib.request.urlopen(req) as r:
            body = r.read()
            return r.status, (body if raw else json.loads(body.decode() or "{}"))
    except urllib.error.HTTPError as e:
        body = e.read()
        try:
            return e.code, json.loads(body.decode() or "{}")
        except ValueError:
            return e.code, {"raw": body}


# --------------------------------------------------------------------------- release-asset host
def _ensure_release(repo, token):
    """Return the ``qa-assets`` bucket release, creating it once if missing."""
    st, rel = _req("GET", f"{API}/repos/{repo}/releases/tags/{ASSET_RELEASE_TAG}", token)
    if st == 200:
        return rel
    st, rel = _req("POST", f"{API}/repos/{repo}/releases", token, data=json.dumps({
        "tag_name": ASSET_RELEASE_TAG,
        "name": "QA diagnostic assets",
        "body": "Bucket release hosting QA diagnostic figures embedded in issue comments. "
                "Managed by data_qa.post_diagnostics; do not edit by hand.",
        "prerelease": True,
    }).encode())
    if st >= 300:
        raise PostError(f"could not create {ASSET_RELEASE_TAG} release: {rel}")
    return rel


def upload_asset(repo, token, png_path, asset_name):
    """Upload ``png_path`` as ``asset_name`` on the bucket release; replace if it exists.
    Returns the browser_download_url (renders inline in markdown)."""
    rel = _ensure_release(repo, token)
    for a in rel.get("assets", []):
        if a["name"] == asset_name:
            _req("DELETE", f"{API}/repos/{repo}/releases/assets/{a['id']}", token)
    with open(png_path, "rb") as fh:
        blob = fh.read()
    ctype = mimetypes.guess_type(png_path)[0] or "image/png"
    url = f"{UPLOADS}/repos/{repo}/releases/{rel['id']}/assets?name={asset_name}"
    st, data = _req("POST", url, token, data=blob, headers={"Content-Type": ctype})
    if st >= 300:
        raise PostError(f"asset upload failed ({st}): {data}")
    return data["browser_download_url"]


# --------------------------------------------------------------------------- issue + comment
def _issue_number(repo, token, title):
    """Number of the canonical issue with ``title``.  Titles can be duplicated (a closed
    dup + the live one), so collect ALL matches and prefer an OPEN issue; never post to a
    closed duplicate."""
    matches, page = [], 1
    while True:
        st, data = _req("GET", f"{API}/repos/{repo}/issues?state=all&per_page=100&page={page}", token)
        if st != 200:
            # Same "transient error indistinguishable from a real negative" trap as
            # _find_stage_comment: a 5xx/rate-limit mid-scan must NOT masquerade as "no such
            # issue" (-> a silent per-issue no-op on a cron run).  Raise instead.
            raise PostError(f"issue lookup failed ({st}) for {title!r}: {data}")
        if not data:                       # successful empty page -> end of pagination
            break
        for it in data:
            if "pull_request" in it:
                continue
            if it["title"] == title:
                matches.append((it["state"], it["number"]))
        if len(data) < 100:
            break
        page += 1
    if not matches:
        return None
    open_ = [n for s, n in matches if s == "open"]
    return min(open_) if open_ else min(n for _, n in matches)


def _find_stage_comment(repo, token, num, marker):
    """Return the existing marker-keyed comment, or None if it genuinely does not exist.

    CRITICAL: a transient API failure (5xx / rate-limit) must NOT be reported as "not
    found" -- the caller would then POST a duplicate, spamming the issue on every hiccup.
    So raise on an API error and only return None when a listing SUCCEEDED without the
    marker."""
    page = 1
    while True:
        st, data = _req("GET", f"{API}/repos/{repo}/issues/{num}/comments?per_page=100&page={page}", token)
        if st != 200:
            raise PostError(f"comment lookup failed ({st}) on #{num}; refusing to risk a "
                            f"duplicate post: {data}")
        if not data:                       # successful empty page -> paginated past the end
            return None
        for c in data:
            if marker in (c.get("body") or ""):
                return c
        if len(data) < 100:
            return None
        page += 1


def post_stage(o: Observation, stage, png_path, caption, repo, token=None):
    """Idempotently post/update the stage-N comment on ``o``'s issue with the figure."""
    token = token or _token()
    num = _issue_number(repo, token, o.issue_title)
    if num is None:
        raise PostError(f"no issue titled {o.issue_title!r} in {repo}")
    # Resolve the existing comment BEFORE uploading the asset: _find_stage_comment can raise on
    # a transient lookup failure, and doing it first keeps that a clean no-op instead of leaving
    # an already-replaced release asset behind.
    marker = DIAG_MARKER.format(n=stage)
    existing = _find_stage_comment(repo, token, num, marker)
    asset_name = f"{o.obsid}_stage{stage}.png"
    img_url = upload_asset(repo, token, png_path, asset_name)
    body = (f"{marker}\n### QA diagnostic — stage {stage}\n\n"
            f"{caption}\n\n"
            f"![{asset_name}]({img_url})\n\n"
            f"<sub>{_provenance_footer(repo, stage)}</sub>\n"
            f"<sub>auto-posted by `data_qa.diagnostics`; updates in place as the pipeline advances.</sub>")
    if existing:
        st, data = _req("PATCH", f"{API}/repos/{repo}/issues/comments/{existing['id']}", token,
                        data=json.dumps({"body": body}).encode())
        action = "updated"
    else:
        st, data = _req("POST", f"{API}/repos/{repo}/issues/{num}/comments", token,
                        data=json.dumps({"body": body}).encode())
        action = "created"
    if st >= 300:
        raise PostError(f"comment {action} failed ({st}): {data}")
    print(f"  stage {stage}: {action} comment on #{num} -> {data.get('html_url')}")
    return data
