"""Collect pipeline status for an observation and post it as a COMMENT on its
per-observation QA tracking issue.

Collected (each part failure-tolerant -- a missing squeue / state file / release dir
degrades to "unavailable", never crashes the report):
  * SLURM queue: ``squeue --me`` rows whose job name matches the field/program
    prefix (the ``<target><program>-o<obsid>-<stage>`` naming convention);
  * monitor state summary from the mast_monitor state file;
  * latest release-stage marker (cheap glob under the releases root).

The comment body starts with ``<!-- data-qa:status -->`` + a UTC timestamp.  The
issue is found by exact title (``make_issues`` conventions); the autogen issue BODY
is never touched -- status lives in the comments.  ``--update-last`` edits the bot's
previous status comment (matched by the marker) instead of stacking new ones.

Stdlib-only.  Dry-run by default; --execute posts.

Usage:
    python -m data_qa.status_report --field brick --program 2221 --obs 001
    python -m data_qa.status_report --issue-title 'TEST issue' --execute
    python -m data_qa.status_report --field brick --program 2221 --obs 001 \\
        --execute --update-last
"""
from __future__ import annotations

import argparse
import datetime
import glob
import json
import os
import subprocess
import sys
from typing import List, Optional

from . import _github
from .observations import FIELDS

STATUS_MARKER = "<!-- data-qa:status -->"
# mast_monitor --report comments carry their own marker so successive monitor
# reports (incl. recurring LOW DISK / CAPPED downgrades) edit ONE comment per
# issue instead of stacking, without clobbering the status_report comment.
MONITOR_MARKER = "<!-- data-qa:monitor -->"
DEFAULT_RELEASES_ROOT = "/orange/adamginsburg/jwst/releases"
DEFAULT_STATE = "/orange/adamginsburg/jwst/ops/mast_state.json"
_SQUEUE_FIELDS = ("jobid", "name", "state", "elapsed", "reason")


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def issue_title_for(program, obsnum, field="", instrument="NIRCam") -> str:
    """The make_issues idempotency title for one observation."""
    target = FIELDS.get(field, field) or "?"
    return f"{target} — jw{int(program):05d}-o{obsnum} ({instrument})"


# ------------------------------------------------------------------------ collectors
def collect_squeue(prefix: str = "") -> Optional[List[dict]]:
    """Our queued/running jobs (name matching prefix); None if squeue unavailable."""
    cmd = ["squeue", "--me", "--noheader", "--format=%i|%j|%T|%M|%R"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except (FileNotFoundError, subprocess.TimeoutExpired) as ex:
        print(f"status_report: squeue unavailable ({ex.__class__.__name__})",
              file=sys.stderr)
        return None
    if proc.returncode != 0:
        print(f"status_report: squeue failed: {proc.stderr.strip()}", file=sys.stderr)
        return None
    rows = []
    for ln in proc.stdout.splitlines():
        parts = ln.strip().split("|")
        if len(parts) != len(_SQUEUE_FIELDS):
            continue
        row = dict(zip(_SQUEUE_FIELDS, parts))
        if not prefix or row["name"].startswith(prefix):
            rows.append(row)
    return rows


def state_summary(state_path: str, program=None) -> Optional[dict]:
    """Monitor state file digest; None when absent/unreadable."""
    try:
        with open(state_path) as fh:
            state = json.load(fh)
    except (OSError, json.JSONDecodeError) as ex:
        print(f"status_report: no monitor state ({ex.__class__.__name__}: {ex})",
              file=sys.stderr)
        return None
    progs = state.get("programs", {})
    if program is not None:
        progs = {k: v for k, v in progs.items() if k == str(int(program))}
    return {
        "last_poll_utc": state.get("last_poll_utc", "?"),
        "n_programs": len(progs),
        "n_obs": sum(len(p.get("obs", {})) for p in progs.values()),
    }


def latest_release_marker(field: str,
                          releases_root: str = DEFAULT_RELEASES_ROOT) -> Optional[str]:
    """Newest release-stage path mentioning the field (cheap glob; None if none)."""
    if not field:
        return None
    try:
        hits = glob.glob(os.path.join(releases_root, "*", field))
        hits += glob.glob(os.path.join(releases_root, field))
        hits = sorted(hits, key=os.path.getmtime)
    except OSError as ex:
        print(f"status_report: release glob failed ({ex})", file=sys.stderr)
        return None
    return hits[-1] if hits else None


# ------------------------------------------------------------------------- rendering
def render_status(field="", program=None, obsnum="", jobs=None, state=None,
                  release=None, job_prefix="", now=None) -> str:
    """The status markdown block (starts with the marker + UTC timestamp)."""
    lines = [STATUS_MARKER,
             f"**Pipeline status** — `{field or '?'}`"
             + (f" / program `{int(program)}`" if program else "")
             + (f" / obs `{obsnum}`" if obsnum else "")
             + f" — {now or utc_now()}", ""]
    if jobs is None:
        lines += [f"**SLURM** (prefix `{job_prefix}`): _squeue unavailable_"]
    elif not jobs:
        lines += [f"**SLURM** (prefix `{job_prefix}`): no queued or running jobs"]
    else:
        lines += [f"**SLURM** (prefix `{job_prefix}`): {len(jobs)} job(s)", "",
                  "| jobid | name | state | elapsed | reason/node |",
                  "|-------|------|-------|---------|-------------|"]
        lines += [f"| {j['jobid']} | `{j['name']}` | {j['state']} | {j['elapsed']} "
                  f"| {j['reason']} |" for j in jobs]
    lines += [""]
    if state:
        lines += [f"**Monitor state:** {state['n_obs']} observation(s) known across "
                  f"{state['n_programs']} program(s); last poll {state['last_poll_utc']}"]
    else:
        lines += ["**Monitor state:** no state file (monitor not yet committed)"]
    lines += [f"**Latest release stage:** `{release}`" if release
              else "**Latest release stage:** none found"]
    lines += ["", "_Posted by `data_qa/status_report.py`._"]
    return "\n".join(lines)


def render_events_comment(events: List[dict], now=None, notice=None) -> str:
    """Markdown comment body for mast_monitor --report (MONITOR_MARKER header, so
    the monitor's update-in-place path finds and edits its own comment).

    ``notice`` (e.g. the --auto LOW DISK / SEED / CAPPED downgrade message)
    renders as a loud warning blockquote above the event list."""
    from .mast_monitor import mjd_to_iso   # stdlib-only
    lines = [MONITOR_MARKER,
             f"**MAST monitor events** — {now or utc_now()}", ""]
    if notice:
        lines += [f"> **WARNING — {notice}**", ""]
    for ev in events:
        rel = mjd_to_iso(ev.get("t_obs_release"))
        tile = f", tile `{ev['tile']}`" if ev.get("tile") else ""
        lines.append(f"- **{ev['event']}**: `{ev['obs_id']}` "
                     f"(calib level {ev.get('calib_level')}, release {rel}, "
                     f"filters `{ev.get('filters') or '?'}`{tile})")
    lines += ["", "_Posted by `data_qa/mast_monitor.py --report`._"]
    return "\n".join(lines)


# --------------------------------------------------------------------------- posting
def find_last_marked_comment(token, repo, number,
                             marker=STATUS_MARKER) -> Optional[dict]:
    """The most recent comment on the issue that starts with ``marker``."""
    ours = [c for c in _github.list_comments(token, repo, number)
            if (c.get("body") or "").lstrip().startswith(marker)]
    return ours[-1] if ours else None


def find_last_status_comment(token, repo, number) -> Optional[dict]:
    """The most recent comment on the issue that starts with the status marker."""
    return find_last_marked_comment(token, repo, number, marker=STATUS_MARKER)


def post_status(title: str, body: str, repo=None, update_last=False, dry_run=True,
                marker=STATUS_MARKER):
    """Post (or, update_last, edit-in-place) the status comment on the issue with
    this exact title.  ``marker`` selects WHICH bot comment update_last edits
    (STATUS_MARKER for status reports, MONITOR_MARKER for monitor events).
    Returns 0 on success / dry-run, nonzero on failure."""
    repo = repo or _github.REPO
    if dry_run:
        print(f"DRY-RUN: would {'update last status comment' if update_last else 'comment'} "
              f"on issue titled {title!r} in {repo}:")
        print(body)
        return 0
    token = _github.get_token()
    if not token:
        print("no GitHub token (GITHUB_TOKEN/GH_TOKEN or `gh auth login`)",
              file=sys.stderr)
        return 2
    issue = _github.existing_issues(token, repo).get(title)
    if issue is None:
        print(f"no issue titled {title!r} in {repo}", file=sys.stderr)
        return 3
    number = issue["number"]
    if update_last:
        prev = find_last_marked_comment(token, repo, number, marker=marker)
        if prev is not None:
            status, data = _github.update_comment(token, repo, prev["id"], body)
            print(f"updated status comment {prev['id']} on #{number} ({status})")
            return 0 if status < 300 else 4
    status, data = _github.post_comment(token, repo, number, body)
    print(f"posted status comment on #{number} ({status})")
    return 0 if status < 300 else 4


# ------------------------------------------------------------------------------ main
def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--field", default="", help="release field name, e.g. brick")
    ap.add_argument("--program", default=None, help="program id, e.g. 2221")
    ap.add_argument("--obs", default="", help="observation number, e.g. 001")
    ap.add_argument("--instrument", default="NIRCam")
    ap.add_argument("--issue-title", default=None,
                    help="exact issue title override (testing / odd titles)")
    ap.add_argument("--job-prefix", default=None,
                    help="squeue job-name prefix (default <field><program>)")
    ap.add_argument("--state", default=DEFAULT_STATE,
                    help="mast_monitor state file to summarize")
    ap.add_argument("--releases-root", default=DEFAULT_RELEASES_ROOT)
    ap.add_argument("--repo", default=_github.REPO)
    ap.add_argument("--update-last", action="store_true",
                    help="edit the previous status comment instead of adding one")
    ap.add_argument("--execute", action="store_true",
                    help="really post to GitHub (default: dry-run print)")
    args = ap.parse_args(argv)

    if args.issue_title:
        title = args.issue_title
    elif args.program and args.obs:
        title = issue_title_for(args.program, args.obs, field=args.field,
                                instrument=args.instrument)
    else:
        print("need --issue-title, or --program and --obs (with --field)",
              file=sys.stderr)
        return 1

    prefix = args.job_prefix
    if prefix is None:
        prefix = f"{args.field}{int(args.program)}" if (args.field and args.program) \
            else args.field
    body = render_status(
        field=args.field, program=args.program, obsnum=args.obs,
        jobs=collect_squeue(prefix),
        state=state_summary(args.state, program=args.program),
        release=latest_release_marker(args.field, args.releases_root),
        job_prefix=prefix)
    return post_status(title, body, repo=args.repo, update_last=args.update_last,
                       dry_run=not args.execute)


if __name__ == "__main__":
    raise SystemExit(main())
