#!/usr/bin/env python3
"""Sync the "Data-QA field status" GitHub Projects (v2) board from the QA metrics + open issues.

The board is a human-facing surface: one card per open observation issue.  This script owns two
value fields and never touches a third:

* **Measured** (single-select, SCRIPT-OWNED): the current measured state of the observation, from
  what ``data_qa.diagnostics`` last wrote — 🚩 Red flag / 🟥 Error / ⚠️ Attention / ◻️ Incomplete /
  ✅ Clean / ❔ No metrics / 🔭 MIRI-other / 📋 Meta.  Rewritten every run.  Group the board by this.
* **Workflow** (single-select, HUMAN-ONLY): the script NEVER writes it, so a human's placement is
  never clobbered.  Lifecycle: Data taken -> QA assigned / under examination -> Needs discussion /
  Needs attention -> Done.
* plus read-only detail fields: **Stages 1-6** (glyph line), **Red flags** (count),
  **Offset (mas)** (stage-4 median tie).

It re-runs no analysis and detects nothing; it only reflects the metrics on disk.

SAFETY:
* **Dry-run is the DEFAULT.** It prints a `#num  Measured: <old> -> <new>` diff and changes nothing.
  Pass ``--apply`` to write.
* The metrics live in a git-IGNORED directory (``data_qa/metrics/``).  A checkout that never ran
  diagnostics has none, which would flip every card to "No metrics".  The script prints the
  metrics dir + newest mtime at startup and REFUSES to ``--apply`` when more than ``--max-missing``
  observation issues lack a metrics file (override with ``--allow-missing``), so a stale/empty tree
  cannot silently wipe the board.

Idempotent: a card already present is matched by issue number and updated in place; a closed issue's
card is archived.  IDs (project, fields, options) are resolved BY NAME and provisioned if missing,
so a freshly-created board works.  Requires ``GH_TOKEN``/``GITHUB_TOKEN`` with org Projects
read+write.

Usage:
    python scripts/sync_project_board.py                 # dry-run: print the diff
    python scripts/sync_project_board.py --apply         # write it
"""
import argparse
import json
import os
import re
import subprocess
import sys

_OBS_RE = re.compile(r"jw(\d{5})-o(\d{3}).*\((NIRCam|MIRI|Niriss|NIRSpec)\)", re.I)
_STAGE_RE = re.compile(r"stage(\d)")
_GLYPH = {"RF": "🚩", "ok": "✅", "fail": "⚠️", "err": "🟥", "?": "·"}

# Measured single-select: category -> option label.  _MEASURED_OPTIONS is derived from this so the
# two never drift.
_CAT_TO_OPTION = {
    "redflag": "🚩 Red flag", "error": "🟥 Error", "attention": "⚠️ Attention (stage failed)",
    "incomplete": "◻️ Incomplete", "clean": "✅ Clean", "nometrics": "❔ No metrics",
    "MIRI": "🔭 MIRI/other", "meta": "📋 Meta",
}
_MEASURED_OPTIONS = list(_CAT_TO_OPTION.values())
# Human-owned workflow lifecycle (the script NEVER writes it): data taken -> QA assigned ->
# needs discussion / needs attention -> done.
_WORKFLOW_OPTIONS = ["Data taken", "QA assigned / under examination", "Needs discussion",
                     "Needs attention", "Done"]
_METRICS_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "data_qa", "metrics"))


def _gh(*args, check=True):
    r = subprocess.run(["gh", *args], capture_output=True, text=True)
    if check and r.returncode != 0:
        raise RuntimeError("gh " + " ".join(args) + " -> " + r.stderr.strip())
    return r.stdout, r.returncode


# --------------------------------------------------------------------------- metrics -> category
def _load_metrics(obsid):
    """Return the metrics dict, None (absent), or 'corrupt' (unreadable/truncated -- see the
    incremental writes in diagnostics.py, which a SLURM timeout can leave mid-write)."""
    mp = os.path.join(_METRICS_DIR, f"{obsid}.json")
    if not os.path.exists(mp):
        return None
    try:
        with open(mp) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return "corrupt"


def _stage_status(metrics):
    """(per-stage glyph line, red-flag count, {stage:code}) from a metrics dict.  A crashed stage
    (``error`` key, written by diagnostics on an exception) is 'err', NOT a measured 'fail'."""
    st = {}
    for k, v in (metrics or {}).items():
        m = _STAGE_RE.match(k)
        if m and isinstance(v, dict):
            n = int(m.group(1))
            st[n] = ("err" if v.get("error") else
                     "RF" if v.get("red_flag") else
                     "ok" if v.get("passed") is True else
                     "fail" if v.get("passed") is False else "?")
    line = "".join(_GLYPH[st.get(i, "?")] for i in range(1, 7))
    nrf = sum(1 for c in st.values() if c == "RF")
    return line, nrf, st


def _classify(inst, metrics, st):
    """Measured category.  A missing/partial/crashed result is NEVER 'clean': absent metrics ->
    no-metrics, a crashed stage -> error, a stage that did not run ('?') -> incomplete."""
    if inst and inst.lower() != "nircam":
        return "MIRI"                                     # MIRI/NIRISS/NIRSpec: pipeline-status only
    if metrics is None or metrics == "corrupt" or not st:
        return "nometrics"
    if any(c == "RF" for c in st.values()):
        return "redflag"
    if any(c == "err" for c in st.values()):
        return "error"
    if any(st.get(n, "?") == "?" for n in range(1, 7)):  # any of stages 1-6 never ran -> not clean
        return "incomplete"
    if any(c == "fail" for c in st.values()):
        return "attention"
    return "clean"


def _rows(repo):
    out, _ = _gh("issue", "list", "--repo", repo, "--state", "open",
                 "--json", "number,title,url", "--limit", "300")
    issues = json.loads(out)
    if len(issues) >= 300:
        print("WARNING: hit the 300-issue list cap; some open issues may be missing", file=sys.stderr)
    rows = []
    for it in issues:
        m = _OBS_RE.search(it["title"])
        if not m:
            rows.append(dict(num=it["number"], url=it["url"], cat="meta", line="", nrf=0,
                             offset=None, has_metrics=True))          # meta issue: not expected to
            continue
        obsid = f"jw{m.group(1)}-o{m.group(2)}"
        inst = m.group(3)
        metrics = _load_metrics(obsid)
        line, nrf, st = _stage_status(metrics if isinstance(metrics, dict) else None)
        cat = _classify(inst, metrics, st)
        s4 = metrics.get("stage4", {}) if isinstance(metrics, dict) else {}
        offset = s4.get("offset_med_mas")
        if cat in ("MIRI", "meta", "nometrics"):
            line, nrf = "", 0
        rows.append(dict(num=it["number"], url=it["url"], cat=cat, line=line, nrf=nrf,
                         offset=offset,
                         # a NIRCam obs issue that SHOULD have metrics but doesn't:
                         has_metrics=not (inst.lower() == "nircam" and metrics is None)))
    return rows


# --------------------------------------------------------------------------- board plumbing
def _project(owner, title):
    out, _ = _gh("project", "list", "--owner", owner, "--format", "json")
    for p in json.loads(out)["projects"]:
        if p["title"] == title:
            return p
    raise SystemExit(f"no project titled {title!r} under {owner} (create it first)")


def _fields(owner, number):
    out, _ = _gh("project", "field-list", str(number), "--owner", owner, "--format", "json",
                 "--limit", "60")
    return {f["name"]: f for f in json.loads(out)["fields"]}


def _ensure_single_select(owner, number, name, options, fields, apply):
    """Ensure a single-select field with EXACTLY `options` exists; create (or delete+recreate on an
    option mismatch) when --apply.  Returns the field dict or None (dry-run, absent)."""
    f = fields.get(name)
    if f is not None and [o["name"] for o in f.get("options", [])] == options:
        return f
    if not apply:
        return f                                          # dry-run: don't mutate the board schema
    if f is not None:
        _gh("project", "field-delete", "--id", f["id"])   # options changed -> recreate
    _gh("project", "field-create", str(number), "--owner", owner, "--name", name,
        "--data-type", "SINGLE_SELECT", "--single-select-options", ",".join(options))
    return _fields(owner, number)[name]


def _ensure_field(owner, number, name, data_type, fields, apply):
    f = fields.get(name)
    if f is not None or not apply:
        return f
    _gh("project", "field-create", str(number), "--owner", owner, "--name", name,
        "--data-type", data_type)
    return _fields(owner, number)[name]


def _item_field(item, field_name):
    """Read a field value off an item-list row (gh camel-cases the field name into the key)."""
    want = field_name.lower()
    for k, v in item.items():
        if k.lower() == want:
            return v
    return None


# --------------------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--owner", default="JWST-GC")
    ap.add_argument("--repo", default="JWST-GC/data-qa")
    ap.add_argument("--title", default="Data-QA field status")
    ap.add_argument("--apply", action="store_true", help="write to the board (default: dry-run)")
    ap.add_argument("--max-missing", type=int, default=3,
                    help="refuse --apply if more than this many NIRCam issues lack metrics")
    ap.add_argument("--allow-missing", action="store_true", help="apply even if metrics are missing")
    args = ap.parse_args(argv)
    apply = args.apply

    if not (os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")):
        raise SystemExit("GH_TOKEN/GITHUB_TOKEN not set (needs org Projects read+write)")
    # scope preflight: Projects v2 needs read:project; fail LOUDLY, not with a later traceback.
    _, rc = _gh("project", "list", "--owner", args.owner, "--format", "json", check=False)
    if rc != 0:
        raise SystemExit("cannot list org projects -- token lacks Projects (read:project) scope "
                         "or org approval; grant it and retry")

    # metrics provenance + a guard against a stale/empty tree silently wiping the board
    newest = 0.0
    if os.path.isdir(_METRICS_DIR):
        for fn in os.listdir(_METRICS_DIR):
            if fn.endswith(".json"):
                newest = max(newest, os.path.getmtime(os.path.join(_METRICS_DIR, fn)))
    import time
    newest_str = time.strftime("%Y-%m-%d %H:%M", time.localtime(newest)) if newest else "NONE"
    print(f"metrics dir: {_METRICS_DIR}  (newest: {newest_str})")
    print(f"mode: {'APPLY' if apply else 'DRY-RUN (pass --apply to write)'}")

    rows = _rows(args.repo)
    n_missing = sum(1 for r in rows if not r["has_metrics"])
    if n_missing:
        print(f"WARNING: {n_missing} NIRCam issue(s) have NO metrics file -> 'No metrics'")
    if apply and n_missing > args.max_missing and not args.allow_missing:
        raise SystemExit(f"REFUSING to apply: {n_missing} NIRCam issues lack metrics "
                         f"(> --max-missing {args.max_missing}).  This tree likely never ran "
                         f"diagnostics; re-run them or pass --allow-missing.")

    proj = _project(args.owner, args.title)
    pid, pnum = proj["id"], proj["number"]
    fields = _fields(args.owner, pnum)
    fmeas = _ensure_single_select(args.owner, pnum, "Measured", _MEASURED_OPTIONS, fields, apply)
    _ensure_single_select(args.owner, pnum, "Workflow", _WORKFLOW_OPTIONS, fields, apply)  # human-only
    fstages = _ensure_field(args.owner, pnum, "Stages 1-6", "TEXT", fields, apply)
    fnrf = _ensure_field(args.owner, pnum, "Red flags", "NUMBER", fields, apply)
    foff = _ensure_field(args.owner, pnum, "Offset (mas)", "NUMBER", fields, apply)
    meas_opt = {o["name"]: o["id"] for o in (fmeas or {}).get("options", [])}

    out, _ = _gh("project", "item-list", str(pnum), "--owner", args.owner, "--format", "json",
                 "--limit", "300")
    items = json.loads(out)["items"]
    existing = {c["number"]: it for it in items
                if (c := it.get("content") or {}).get("number") is not None}

    changed = 0
    for r in rows:
        want = _CAT_TO_OPTION[r["cat"]]
        cur_item = existing.get(r["num"])
        old = _item_field(cur_item, "Measured") if cur_item else None
        if old != want:
            changed += 1
            print(f"#{r['num']:>3}  Measured: {old or '—'} -> {want}")
        if not apply:
            continue
        iid = cur_item["id"] if cur_item else None
        if iid is None:
            o2, _ = _gh("project", "item-add", str(pnum), "--owner", args.owner,
                        "--url", r["url"], "--format", "json")
            iid = json.loads(o2)["id"]
        if fmeas and want in meas_opt:
            _gh("project", "item-edit", "--id", iid, "--project-id", pid,
                "--field-id", fmeas["id"], "--single-select-option-id", meas_opt[want])
        if fstages:
            _gh("project", "item-edit", "--id", iid, "--project-id", pid,
                "--field-id", fstages["id"], "--text", r["line"] or "—")
        if fnrf and r["cat"] not in ("MIRI", "meta", "nometrics"):
            _gh("project", "item-edit", "--id", iid, "--project-id", pid,
                "--field-id", fnrf["id"], "--number", str(r["nrf"]))
        if foff and r["offset"] is not None:
            _gh("project", "item-edit", "--id", iid, "--project-id", pid,
                "--field-id", foff["id"], "--number", str(round(float(r["offset"]), 1)))

    # archive cards whose issue is no longer open (closed from the board -> stop inflating counts)
    open_nums = {r["num"] for r in rows}
    stale = [(n, it) for n, it in existing.items() if n not in open_nums]
    for n, it in stale:
        print(f"#{n:>3}  issue closed -> archive card")
        if apply:
            _gh("project", "item-archive", str(pnum), "--owner", args.owner, "--id", it["id"])

    print(f"{'applied' if apply else 'would change'} {changed} card(s), "
          f"{len(stale)} to archive; {len(rows)} open issues -> {proj['url']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
