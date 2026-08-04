#!/usr/bin/env python3
"""Sync the "Data-QA field status" GitHub Projects (v2) board from the QA metrics + open issues.

The board is a HUMAN-FACING surface: one card per observation issue, grouped by an overall
``QA Status`` (🚩 Red flag / ⚠️ Off-tie / ✅ Clean / 🔭 MIRI-other / 📋 Meta), with the per-stage
pass/flag line and the red-flag count as extra fields.  Humans drag cards between statuses and
close issues from the board; this script only (re)populates cards and refreshes the field values
from what the diagnostics last wrote -- it never re-runs any analysis and never moves a human's
manual placement other than to reflect the current measured status.

Idempotent: an issue already on the board is updated in place (matched by issue number); a new
issue is added.  Safe to run on any cadence -- e.g. after ``refresh_all_issues`` in the daily
cron.

IDs (project number, field ids, single-select option ids) are resolved BY NAME at runtime, so the
script keeps working if the board is recreated.  Requires ``GH_TOKEN``/``GITHUB_TOKEN`` with org
Projects read+write.

Usage:
    python scripts/sync_project_board.py [--owner JWST-GC] [--repo JWST-GC/data-qa]
                                         [--title "Data-QA field status"] [--dry-run]
"""
import argparse
import json
import os
import re
import subprocess
import sys

_OBS_RE = re.compile(r"jw(\d{5})-o(\d{3}).*\((NIRCam|MIRI|Niriss|NIRSpec)\)", re.I)
_STAGE_RE = re.compile(r"stage(\d)")
_GLYPH = {"RF": "🚩", "ok": "✅", "fail": "⚠️", "?": "·"}
_STATUS_OPTIONS = ["🚩 Red flag", "⚠️ Off-tie (measured)", "✅ Clean", "🔭 MIRI/other", "📋 Meta"]
_CAT_TO_OPTION = {"redflag": "🚩 Red flag", "offtie": "⚠️ Off-tie (measured)",
                  "clean": "✅ Clean", "MIRI": "🔭 MIRI/other", "meta": "📋 Meta"}
_METRICS_DIR = os.path.join(os.path.dirname(__file__), "..", "data_qa", "metrics")


def _gh(*args, check=True):
    r = subprocess.run(["gh", *args], capture_output=True, text=True)
    if check and r.returncode != 0:
        raise RuntimeError("gh " + " ".join(args) + " -> " + r.stderr.strip())
    return r.stdout


def _stage_status(metrics):
    """(per-stage glyph line, red-flag count, {stage:code}) from a metrics dict."""
    st = {}
    for k, v in (metrics or {}).items():
        m = _STAGE_RE.match(k)
        if m and isinstance(v, dict):
            n = int(m.group(1))
            st[n] = ("RF" if v.get("red_flag") else
                     "ok" if v.get("passed") is True else
                     "fail" if v.get("passed") is False else "?")
    line = "".join(_GLYPH[st.get(i, "?")] for i in range(1, 7))
    nrf = sum(1 for c in st.values() if c == "RF")
    return line, nrf, st


def _classify(inst, metrics, st):
    if inst and inst.lower() != "nircam":
        return "MIRI"                                  # MIRI/NIRISS/NIRSpec: pipeline-status only
    if metrics is None:
        return "meta"
    if any(c == "RF" for c in st.values()):
        return "redflag"
    if any(c == "fail" for c in st.values()):
        return "offtie"
    return "clean"


def _rows(repo):
    issues = json.loads(_gh("issue", "list", "--repo", repo, "--state", "open",
                            "--json", "number,title,url", "--limit", "100"))
    rows = []
    for it in issues:
        m = _OBS_RE.search(it["title"])
        if not m:
            rows.append(dict(num=it["number"], url=it["url"], cat="meta", line="", nrf=0))
            continue
        obsid = f"jw{m.group(1)}-o{m.group(2)}"
        inst = m.group(3)
        mp = os.path.join(_METRICS_DIR, f"{obsid}.json")
        metrics = json.load(open(mp)) if os.path.exists(mp) else None
        line, nrf, st = _stage_status(metrics)
        cat = _classify(inst, metrics, st)
        if cat in ("MIRI", "meta"):
            line, nrf = "", 0                          # shared-obsid metrics don't apply to MIRI cards
        rows.append(dict(num=it["number"], url=it["url"], cat=cat, line=line, nrf=nrf))
    return rows


def _project(owner, title):
    projs = json.loads(_gh("project", "list", "--owner", owner, "--format", "json"))["projects"]
    for p in projs:
        if p["title"] == title:
            return p
    raise SystemExit(f"no project titled {title!r} under {owner} (create it first)")


def _fields(owner, number):
    fs = json.loads(_gh("project", "field-list", str(number), "--owner", owner,
                        "--format", "json", "--limit", "50"))["fields"]
    by_name = {f["name"]: f for f in fs}
    return by_name


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--owner", default="JWST-GC")
    ap.add_argument("--repo", default="JWST-GC/data-qa")
    ap.add_argument("--title", default="Data-QA field status")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    if not (os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")):
        raise SystemExit("GH_TOKEN/GITHUB_TOKEN not set (needs org Projects read+write)")

    proj = _project(args.owner, args.title)
    pid, pnum = proj["id"], proj["number"]
    fields = _fields(args.owner, pnum)
    fstatus = fields["QA Status"]
    opt_id = {o["name"]: o["id"] for o in fstatus["options"]}
    missing = [o for o in _STATUS_OPTIONS if o not in opt_id]
    if missing:
        raise SystemExit(f"QA Status field is missing options {missing} -- recreate the field")
    fstages = fields.get("Stages 1-6")
    fnrf = fields.get("Red flags")

    existing = {}
    items = json.loads(_gh("project", "item-list", str(pnum), "--owner", args.owner,
                           "--format", "json", "--limit", "200"))["items"]
    for it in items:
        c = it.get("content") or {}
        if c.get("number") is not None:
            existing[c["number"]] = it["id"]

    rows = _rows(args.repo)
    for r in rows:
        line = r["line"]
        opt = _CAT_TO_OPTION[r["cat"]]
        if args.dry_run:
            print(f"#{r['num']:>3} {opt:22s} {line}")
            continue
        iid = existing.get(r["num"])
        if iid is None:
            iid = json.loads(_gh("project", "item-add", str(pnum), "--owner", args.owner,
                                 "--url", r["url"], "--format", "json"))["id"]
        _gh("project", "item-edit", "--id", iid, "--project-id", pid,
            "--field-id", fstatus["id"], "--single-select-option-id", opt_id[opt])
        if fstages is not None:
            # blank the text for MIRI/meta cards, else the per-stage glyph line
            _gh("project", "item-edit", "--id", iid, "--project-id", pid,
                "--field-id", fstages["id"], "--text", line or "—")
        if fnrf is not None and r["cat"] not in ("MIRI", "meta"):
            _gh("project", "item-edit", "--id", iid, "--project-id", pid,
                "--field-id", fnrf["id"], "--number", str(r["nrf"]))
        print(f"#{r['num']:>3} {opt:22s} {line}")
    print(f"synced {len(rows)} items to {proj['url']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
