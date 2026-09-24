"""Post a QA diagnostic figure as an idempotent reply (comment) on the observation's
tracking issue.

The image LIVES IN THE ISSUE, not the repo source tree: the PNG bytes are uploaded to the
GitHub CDN as an asset on a single ``qa-assets`` bucket release, and the comment embeds
that URL.  One comment per (issue, stage), keyed on a hidden marker, so re-running UPDATES
the existing comment (new image + caption) instead of piling up duplicates.

Stdlib-only (urllib) so it runs in CI with just ``GITHUB_TOKEN``.
"""
from __future__ import annotations

import hashlib
import json
import mimetypes
import os
import re
import time
import urllib.error
import urllib.request

from .observations import Observation

API = "https://api.github.com"


class PostError(Exception):
    """A GitHub post/upload step failed; caller decides whether to continue other stages."""
UPLOADS = "https://uploads.github.com"
ASSET_RELEASE_TAG = os.environ.get("QA_ASSET_TAG", "qa-assets")
# GitHub caps a release at 1000 assets ("file_count limited to 1000 assets per release"), and the
# QA figures (~15 per observation x ~160 issues) exceed that, so the bucket is sharded: each asset
# goes to ``{ASSET_RELEASE_TAG}-NN`` chosen by a stable hash of its name, which keeps same-name
# replacement working.  The original un-sharded release is the legacy bucket: an asset of the same
# name there is deleted on re-upload so it stops holding a slot.
ASSET_SHARDS = int(os.environ.get("QA_ASSET_SHARDS", "16"))
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
    12: "stage12_photometric_linearity",
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


def _req(method, url, token, data=None, headers=None, raw=False, want_headers=False):
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Authorization", f"token {token}")
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("User-Agent", "jwst-gc-data-qa")
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    try:
        with urllib.request.urlopen(req) as r:
            body = r.read()
            payload = body if raw else json.loads(body.decode() or "{}")
            return (r.status, payload, dict(r.headers)) if want_headers else (r.status, payload)
    except urllib.error.HTTPError as e:
        body = e.read()
        try:
            payload = json.loads(body.decode() or "{}")
        except ValueError:
            payload = {"raw": body}
        return (e.code, payload, dict(e.headers)) if want_headers else (e.code, payload)


# --------------------------------------------------------------------------- release-asset host
_RELEASES = {}                                  # (repo, tag) -> release object, per process


def _ensure_release(repo, token, tag=ASSET_RELEASE_TAG, create=True):
    """Return the bucket release for ``tag`` (cached per process), creating it once if missing.
    With ``create=False`` a missing release returns None."""
    if (repo, tag) in _RELEASES:
        return _RELEASES[(repo, tag)]
    st, rel = _req("GET", f"{API}/repos/{repo}/releases/tags/{tag}", token)
    if st != 200:
        if not create:
            return None
        st, rel = _req("POST", f"{API}/repos/{repo}/releases", token, data=json.dumps({
            "tag_name": tag,
            "name": f"QA diagnostic assets ({tag})",
            "body": "Bucket release hosting QA diagnostic figures embedded in issue comments. "
                    "Managed by data_qa.post_diagnostics; do not edit by hand.",
            "prerelease": True,
        }).encode())
        if st >= 300:
            # a concurrent array task may have created it first -> re-read before failing
            st, rel = _req("GET", f"{API}/repos/{repo}/releases/tags/{tag}", token)
            if st != 200:
                raise PostError(f"could not create {tag} release: {rel}")
    _RELEASES[(repo, tag)] = rel
    return rel


def _asset_shard_tag(asset_name):
    """Stable shard release tag for ``asset_name`` (same name -> same shard on every run)."""
    h = int(hashlib.sha1(asset_name.encode()).hexdigest()[:8], 16)
    return f"{ASSET_RELEASE_TAG}-{h % ASSET_SHARDS:02d}"


def _release_assets(repo, token, rel):
    """All assets of ``rel`` by name, via the paginated assets endpoint (the list embedded in the
    release object is not guaranteed complete)."""
    out, page = {}, 1
    while True:
        st, data = _req("GET", f"{API}/repos/{repo}/releases/{rel['id']}/assets"
                               f"?per_page=100&page={page}", token)
        if st != 200 or not data:
            return out
        out.update({a["name"]: a["id"] for a in data})
        if len(data) < 100:
            return out
        page += 1


_ASSET_INDEX = {}                               # (repo, tag) -> {name: id}, per process


def _asset_index(repo, token, tag, rel):
    if (repo, tag) not in _ASSET_INDEX:
        _ASSET_INDEX[(repo, tag)] = _release_assets(repo, token, rel)
    return _ASSET_INDEX[(repo, tag)]


def _delete_asset(repo, token, tag, rel, asset_name):
    """Delete ``asset_name`` from ``rel`` if present; returns True when one was removed."""
    idx = _asset_index(repo, token, tag, rel)
    aid = idx.get(asset_name)
    if aid is None:
        return False
    st, data = _req("DELETE", f"{API}/repos/{repo}/releases/assets/{aid}", token)
    if st >= 300 and st != 404:
        raise PostError(f"could not delete existing asset {asset_name} ({st}): {data}")
    idx.pop(asset_name, None)
    return True


def upload_asset(repo, token, png_path, asset_name):
    """Upload ``png_path`` as ``asset_name`` on its shard release; replace if it exists there, and
    drop a same-name copy from the legacy un-sharded release.  Returns the browser_download_url
    (renders inline in markdown)."""
    tag = _asset_shard_tag(asset_name)
    rel = _ensure_release(repo, token, tag)
    _delete_asset(repo, token, tag, rel, asset_name)
    with open(png_path, "rb") as fh:
        blob = fh.read()
    ctype = mimetypes.guess_type(png_path)[0] or "image/png"
    url = f"{UPLOADS}/repos/{repo}/releases/{rel['id']}/assets?name={asset_name}"
    st, data = _req("POST", url, token, data=blob, headers={"Content-Type": ctype})
    if st >= 300:
        raise PostError(f"asset upload failed ({st}): {data}")
    _asset_index(repo, token, tag, rel)[asset_name] = data.get("id")
    # free the legacy copy only once the shard copy exists, so a failed upload leaves the old
    # comment's image resolvable
    legacy = _ensure_release(repo, token, ASSET_RELEASE_TAG, create=False)
    if legacy is not None:
        _delete_asset(repo, token, ASSET_RELEASE_TAG, legacy, asset_name)
    return data["browser_download_url"]


# --------------------------------------------------------------------------- issue + comment
# GitHub's list endpoints intermittently return a SPURIOUS EMPTY page even when more items follow
# -- observed directly: repeated identical calls to the repo issues list gave `page1:100,
# page2:empty` (dropping issue #1, the oldest, which lives on page 2) and even `page1:empty` on some
# calls, interleaved with correct results, and the empty can persist across several retries.
# Treating an empty page as end-of-pagination silently truncates the listing: a real issue reads as
# "no such issue", and a real marker comment reads as absent -> a DUPLICATE post.  The issues
# endpoint uses CURSOR pagination (a `Link` header with rel="next" carrying an `after` cursor; no
# rel="last"), so the end is signalled by the ABSENCE of a rel="next" link, not by an empty page.
# A page reached via a rel="next" therefore MUST have rows; a spurious empty one is re-fetched.
_EMPTY_RETRIES = 8
_EMPTY_BACKOFF_S = 0.4
_PER_PAGE = 100
_NEXT_URL_RE = re.compile(r'<([^>]+)>;\s*rel="next"')


def _fetch_page(url, token, what, must_have_data):
    """Fetch one page URL and return (data, next_url_or_None).  Re-fetches a response that cannot be
    trusted as the true end of the listing: a spurious empty page that ``must_have_data`` (it was
    reached via a rel="next"), or a FULL page (``_PER_PAGE`` rows) carrying no rel="next" (a
    truncated/cached Link header would otherwise read as 'listing complete' and drop later pages)."""
    for _ in range(_EMPTY_RETRIES):
        st, d, hdrs = _req("GET", url, token, want_headers=True)
        if st != 200:
            raise PostError(f"{what} failed ({st}): {d}")
        nxt = _NEXT_URL_RE.search(hdrs.get("Link", ""))
        nxt_url = nxt.group(1) if nxt else None
        empty_bad = not d and must_have_data                 # a page that must exist came back empty
        truncated_bad = d and nxt_url is None and len(d) >= _PER_PAGE  # full page, no next -> suspect
        if not empty_bad and not truncated_bad:
            return d, nxt_url
        time.sleep(_EMPTY_BACKOFF_S)
    # exhausted retries: for a full-but-no-next page, return what we have (a real 100-row last page is
    # possible); for a must-have-data empty, that is a genuine failure.
    if empty_bad:
        raise PostError(f"{what}: a page reached via rel=\"next\" kept returning empty")
    return d, nxt_url


def _paged_get(url_tmpl, token, what):
    """Fetch every item across all pages of a GitHub list endpoint (``url_tmpl`` carries ``{page}``
    for page 1), following rel="next" cursors to the end and re-fetching a page that cannot be
    trusted as the end (see _fetch_page).  A non-200 raises -- a transient error must not masquerade
    as a real negative."""
    data, nxt = [], None
    for _ in range(_EMPTY_RETRIES):
        data, nxt = _fetch_page(url_tmpl.format(page=1), token, what, must_have_data=False)
        if data:
            break
        time.sleep(_EMPTY_BACKOFF_S)
    if not data:
        return []
    items = list(data)
    while nxt:                                   # follow the cursor to the last page
        data, nxt = _fetch_page(nxt, token, what, must_have_data=True)
        items.extend(data)
    return items


# The issues listing can also come back TRUNCATED -- a short page 1 with no rel="next" at all (the
# Link header itself wrong/cached), which no in-scan check can detect.  Different calls truncate
# differently, so the scan is run up to _SCAN_ATTEMPTS times and the matches are UNIONed; a title
# present in the repo surfaces within a few attempts (a full-but-no-next page is already retried in
# _fetch_page, so only a SHORT truncated page -- indistinguishable in-scan from a real last page --
# reaches here), and a genuinely-absent title stays absent across all of them.
_SCAN_ATTEMPTS = 8


def _issue_number(repo, token, title):
    """Number of the canonical issue with ``title``.  Titles can be duplicated (a closed
    dup + the live one), so collect ALL matches and prefer an OPEN issue; never post to a
    closed duplicate."""
    url = f"{API}/repos/{repo}/issues?state=all&per_page=100&page={{page}}"
    states = {}                                  # number -> state, unioned across attempts
    for _ in range(_SCAN_ATTEMPTS):
        for it in _paged_get(url, token, f"issue lookup for {title!r}"):
            if "pull_request" not in it and it["title"] == title:
                states[it["number"]] = it["state"]
        if states:                               # found it (union only grows) -> done
            break
    if not states:
        return None
    open_ = [n for n, s in states.items() if s == "open"]
    return min(open_) if open_ else min(states)


def _find_stage_comment(repo, token, num, marker):
    """Return the existing marker-keyed comment, or None if it genuinely does not exist.

    CRITICAL: a transient API failure (5xx / rate-limit) or a spurious empty page must NOT be
    reported as "not found" -- the caller would then POST a duplicate, spamming the issue on every
    hiccup.  ``_paged_get`` raises on an API error and confirms an empty page before ending, so a
    None return here means a COMPLETE listing genuinely lacked the marker."""
    data = _paged_get(f"{API}/repos/{repo}/issues/{num}/comments?per_page=100&page={{page}}",
                      token, f"comment lookup on #{num}")
    for c in data:
        if marker in (c.get("body") or ""):
            return c
    return None


def _details_block(repo, token, o, stage, extra_images):
    """Upload each ``(label, png_path)`` and return an expandable ``<details>`` block embedding them,
    so a multi-figure stage (stage 12: one plot per filter) shows one plot by default and hides the
    rest.  The asset name is the figure's own basename (already a stable, URL-safe
    ``{obsid}_stage{stage}_{tag}.png``) so it updates in place -- deriving it from the free-text
    ``label`` instead put spaces/parens in the release-asset URL (e.g. 'jicama-m8 vs VIRAC
    (calibration)'), which the GitHub API rejects as control characters in the path."""
    labels = ", ".join(label for label, _ in extra_images)
    parts = [f"\n\n<details><summary>Other {len(extra_images)} figure(s): {labels}</summary>\n"]
    for label, path in extra_images:
        aname = os.path.basename(path)
        url = upload_asset(repo, token, path, aname)
        parts.append(f"\n**{label}**\n\n![{aname}]({url})\n")
    parts.append("\n</details>")
    return "".join(parts)


def post_stage(o: Observation, stage, png_path, caption, repo, token=None, extra_images=None):
    """Idempotently post/update the stage-N comment on ``o``'s issue with the figure.

    ``extra_images`` (optional) is a list of ``(label, png_path)`` for a multi-figure stage; they
    are uploaded and embedded in a collapsed ``<details>`` block after the primary image."""
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
    extra_block = _details_block(repo, token, o, stage, extra_images) if extra_images else ""
    body = (f"{marker}\n### QA diagnostic — stage {stage}\n\n"
            f"{caption}\n\n"
            f"![{asset_name}]({img_url})\n"
            f"{extra_block}\n\n"
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


def unpost_stage(o: Observation, stage, repo, token=None):
    """Remove the stage-``stage`` diagnostic comment on ``o``'s issue, if present.

    Used when a stage's input data are not yet on disk: the stage is excluded from the issue
    until the data land, so a comment left from an earlier run is deleted.  A no-op when the
    issue or the comment does not exist."""
    token = token or _token()
    num = _issue_number(repo, token, o.issue_title)
    if num is None:
        return None
    marker = DIAG_MARKER.format(n=stage)
    existing = _find_stage_comment(repo, token, num, marker)
    if not existing:
        return None
    st, data = _req("DELETE", f"{API}/repos/{repo}/issues/comments/{existing['id']}", token)
    if st >= 300:
        raise PostError(f"comment delete failed ({st}): {data}")
    print(f"  stage {stage}: removed (data unavailable) comment on #{num}")
    return existing["id"]
