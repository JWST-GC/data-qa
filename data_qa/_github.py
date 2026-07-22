"""Shared GitHub REST helpers (stdlib-only urllib).

Used by ``make_issues.py`` (issue create/update) and ``status_report.py`` (status
comments).  Token comes from ``GITHUB_TOKEN``/``GH_TOKEN``, with a ``gh auth token``
fallback for interactive use on the cluster.  No third-party dependencies so the CI
issue-sync stays stdlib-only.
"""
from __future__ import annotations

import json
import os
import subprocess
import urllib.error
import urllib.request

REPO = os.environ.get("QA_REPO", "JWST-GC/data-qa")
API = "https://api.github.com"


def get_token():
    """GITHUB_TOKEN / GH_TOKEN, else `gh auth token` (None if unavailable)."""
    tok = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if tok:
        return tok
    try:
        r = subprocess.run(["gh", "auth", "token"], capture_output=True, text=True,
                           timeout=10)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if r.returncode != 0:
        return None
    return r.stdout.strip() or None


def request(method, url, token, data=None):
    """One API call -> (status, decoded json). HTTP errors return (code, body)."""
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
