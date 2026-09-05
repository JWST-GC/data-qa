"""Shared GitHub REST helpers (stdlib-only urllib).

Used by ``make_issues.py`` (issue create/update) and ``status_report.py`` (status
comments).  Token comes from ``GITHUB_TOKEN``/``GH_TOKEN``, else the PAT file at
``~/.config/data-qa/github_token``, else a ``gh auth token`` fallback for interactive
use on the cluster.  No third-party dependencies so the CI issue-sync stays
stdlib-only.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request

REPO = os.environ.get("QA_REPO", "JWST-GC/data-qa")
API = "https://api.github.com"


# 600-perm PAT file, the headless-safe source.  scripts/refresh_all_issues.sh reads the
# same path and records why: under scron/cron `gh auth token` can hand back an INVALID
# token, so the file has to be tried BEFORE the `gh` rung, not after it.
TOKEN_FILE = "~/.config/data-qa/github_token"

# Failure statuses `request()` returns instead of raising.  Both are >= 300 and != 200,
# so every existing caller (`status >= 300`, `status != 200`) already reads them as the
# failures they are -- nothing here turns an error into a success.
BAD_TOKEN_STATUS = 598          # malformed token: the request was never sent
NETWORK_ERROR_STATUS = 599      # transport failure (DNS/refused/timeout): nothing landed

# A GitHub token is ONE line of printable, space-free ASCII (`ghp_`, `github_pat_`,
# `gho_`, 40-hex).  Anything else -- a PAT file saved with a second line, a pasted
# "Bearer x", a stray CR -- must never reach an Authorization header: http.client
# raises ValueError("Invalid header value b'token <the token>...'"), which both ECHOES
# THE TOKEN into the log and, from the monitor's report path, aborts mast_monitor.main()
# three lines before its save_state().  Reject at the source instead.
_PRINTABLE_ASCII_RUN = re.compile(r"\A[!-~]+\Z")

_MALFORMED = ("not a single line of printable ASCII (whitespace or control "
              "characters); expected ONE token on ONE line")


def valid_token(tok) -> bool:
    """True when `tok` can be sent in an Authorization header."""
    return bool(tok) and _PRINTABLE_ASCII_RUN.match(tok) is not None


def _accept(tok, source):
    """`tok` if it is well-formed, else None plus a stderr line naming the SOURCE.

    The token's VALUE is never printed -- only where it came from.
    """
    if not tok:
        return None
    if valid_token(tok):
        return tok
    print(f"ignoring the GitHub token from {source}: {_MALFORMED}", file=sys.stderr)
    return None


def token_from_file(path=None):
    """Token read from `path` (default TOKEN_FILE); None if unreadable, empty or
    malformed.

    Never raises: this runs inside the monitor's report path, which sits before
    ``mast_monitor.main()``'s ``save_state``, so anything raising out of here loses a
    whole poll's committed state.
    """
    path = os.path.expanduser(path or TOKEN_FILE)
    try:
        with open(path, encoding="utf-8") as fh:
            raw = fh.read()
    except (OSError, UnicodeDecodeError):
        return None
    return _accept(raw.strip(), path)


def get_token():
    """GITHUB_TOKEN / GH_TOKEN, else the PAT file, else `gh auth token` (else None).

    A malformed rung is skipped (with a stderr line naming it) rather than returned:
    a token carrying an interior newline is not usable and raises when sent.
    """
    for var in ("GITHUB_TOKEN", "GH_TOKEN"):
        tok = _accept((os.environ.get(var) or "").strip(), f"${var}")
        if tok:
            return tok
    tok = token_from_file()
    if tok:
        return tok
    try:
        r = subprocess.run(["gh", "auth", "token"], capture_output=True, text=True,
                           timeout=10)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if r.returncode != 0:
        return None
    return _accept(r.stdout.strip(), "`gh auth token`")


_AUTH_CHECKED: dict = {}


#: A 403 whose body names the rate limit.  GitHub uses 403 for both "your token
#: may not do this" and "you are going too fast"; only the first is an auth
#: verdict, and the body is what separates them.
_RATE_LIMIT_HINTS = ("rate limit", "abuse detection", "secondary rate")


def _is_rate_limited(status, message):
    """Is this 403 the rate limiter rather than a permission refusal?"""
    return status == 403 and any(h in message.lower() for h in _RATE_LIMIT_HINTS)


def check_auth(token, force=False):
    """``(ok, detail)`` from ``GET /user`` -- the preflight this package lacked.

    ``scripts/refresh_all_issues.sh`` pairs the same token ladder with a loud
    ``gh api user`` check (:43-51) because an invalid/expired token gives a 401 that
    the enumeration swallows: ``_paginate`` breaks on a non-200, ``existing_issues``
    returns ``{}``, and the caller reports ``no issue titled '<t>'`` rc=3 -- a message
    that names nothing about auth.  A PAT file never refreshes itself, so this is the
    expiry path, not a hypothetical.

    Memoized per token, so a run posting to many issues costs ONE extra request rather
    than one per issue.

    Three statuses are NOT auth verdicts and return ``ok=True`` ("unchecked"),
    leaving the run to fail where it would have failed before rather than being
    blocked by the preflight: a transport failure, any 5xx, and a 403 whose body
    names the rate limit.  Answering ``False`` to any of those refuses a post
    that would have landed -- and a run touching the ~1668 treasury groups is
    where a secondary rate limit actually appears.
    """
    if not force and token in _AUTH_CHECKED:
        return _AUTH_CHECKED[token]
    status, data = request("GET", f"{API}/user", token)
    msg = str(data.get("message") or "")
    if status == 200:
        result = (True, str(data.get("login") or ""))
    elif status == NETWORK_ERROR_STATUS or status >= 500 or _is_rate_limited(
            status, msg):
        # NOT an auth verdict.  A 5xx is GitHub being unwell, and a 403 naming
        # the rate limit is the token working too well -- neither says the
        # token is bad, and answering `False` here refuses a post that would
        # have landed.  A run touching the ~1668 treasury groups is exactly
        # where a secondary rate limit shows up, so this is the delivery-window
        # path, not a hypothetical.  Fail where it would have failed before.
        result = (True, f"unchecked -- HTTP {status}: {msg}".strip())
    else:
        result = (False, f"HTTP {status}: {msg}".strip())
    _AUTH_CHECKED[token] = result
    return result


def request(method, url, token, data=None):
    """One API call -> (status, decoded json). HTTP errors return (code, body).

    Returns a FAILURE status instead of raising for the two non-HTTP ways this can go
    wrong, because the monitor's report path runs before ``save_state``: a malformed
    token (``BAD_TOKEN_STATUS``, nothing sent -- and the token itself is never put in
    the message) and a transport failure (``NETWORK_ERROR_STATUS``).
    """
    if not valid_token(token):
        # do not build the header: http.client would raise ValueError AND put the
        # token in the exception text
        return BAD_TOKEN_STATUS, {"message": f"malformed GitHub token: {_MALFORMED}"}
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
        # an error body is not always JSON (a proxy or rate-limit page is HTML)
        raw = e.read().decode(errors="replace")
        try:
            return e.code, json.loads(raw or "{}")
        except json.JSONDecodeError:
            return e.code, {"message": raw.strip()[:200] or f"HTTP {e.code}"}
    except urllib.error.URLError as e:                  # DNS/refused/timeout
        return NETWORK_ERROR_STATUS, {"message": f"network error: {e.reason}"}


def _paginate(token, url_fmt):
    """GET all pages of url_fmt (must contain {page}); concatenated list."""
    out, page = [], 1
    while True:
        status, data = request("GET", url_fmt.format(page=page), token)
        if status != 200 or not data:
            break
        out.extend(data)
        if len(data) < 100:
            break
        page += 1
    return out


def existing_issues(token, repo):
    """title -> issue dict, over all states (paginated; PRs excluded)."""
    items = _paginate(token, f"{API}/repos/{repo}/issues?state=all&per_page=100"
                             "&page={page}")
    return {it["title"]: it for it in items if "pull_request" not in it}


def ensure_labels(token, repo, names,
                  palette={"QA": "0e8a16", "NIRCam": "1d76db", "MIRI": "5319e7"}):
    """Create any missing labels (best-effort; ignores 'already exists')."""
    for n in names:
        request("POST", f"{API}/repos/{repo}/labels", token,
                {"name": n, "color": palette.get(n, "ededed")})


# ------------------------------------------------------------------- issues/comments
def create_issue(token, repo, title, body, labels=()):
    return request("POST", f"{API}/repos/{repo}/issues", token,
                   {"title": title, "body": body, "labels": list(labels)})


def update_issue(token, repo, number, **fields):
    return request("PATCH", f"{API}/repos/{repo}/issues/{number}", token, fields)


def close_issue(token, repo, number):
    return update_issue(token, repo, number, state="closed")


def list_comments(token, repo, number):
    """All comments on an issue, oldest first (paginated)."""
    return _paginate(token, f"{API}/repos/{repo}/issues/{number}/comments"
                            "?per_page=100&page={page}")


def post_comment(token, repo, number, body):
    return request("POST", f"{API}/repos/{repo}/issues/{number}/comments", token,
                   {"body": body})


def update_comment(token, repo, comment_id, body):
    return request("PATCH", f"{API}/repos/{repo}/issues/comments/{comment_id}", token,
                   {"body": body})
