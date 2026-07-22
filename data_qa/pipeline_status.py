"""Auto-updating pipeline-stage progress table for a per-observation QA issue.

Derives, from the on-disk products, WHICH reduction/cataloging stages have run for an
observation, WHEN (newest product mtime), and HOW they did (counts + measured astrometric
offset).  Rendered as one idempotent comment on the tracking issue (marker-keyed, updated
in place), so the issue always shows the live pipeline state as it advances.

Stage order (rising):
    CRF -> destreak -> single-frame cataloging -> cross-frame merge -> refcat comparison
    -> JWST reference-frame creation (w/ measured offset) -> re-alignment
    -> cataloging m1 -> ... -> m8

Status: ✅ done   🔄 running/queued   ⬜ pending   ⚠️ done, flagged   ⏭️ skipped

Stdlib + glob only for the status; posting reuses the GitHub helpers.

Usage:
    python -m data_qa.pipeline_status --program 5365 --obs 001            # print block
    python -m data_qa.pipeline_status --program 5365 --obs 001 --post     # post/update
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from datetime import datetime, timezone

BASE = os.environ.get("QA_BASE", "/orange/adamginsburg/jwst")
MARKER = "<!-- data-qa:pipeline-status -->"
DONE, RUN, PEND, WARN, SKIP = "✅", "🔄", "⬜", "⚠️", "⏭️"


def _newest(pats, stat_cap=400):
    """(mtime, count) for a glob set.  Count is exact (glob doesn't stat); the newest mtime
    is taken over at most ``stat_cap`` files -- on NFS a full stat of a 16k-file per-frame
    set costs minutes, and an approximate 'last run' timestamp is all the status needs."""
    files = []
    for pat in pats:
        files += glob.glob(pat)
    if not files:
        return None, 0
    sample = files if len(files) <= stat_cap else files[-stat_cap:]
    return max(os.path.getmtime(f) for f in sample), len(files)


def _ts(mtime):
    if mtime is None:
        return "—"
    return datetime.fromtimestamp(mtime, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")


def _fld(o):
    return getattr(o, "field", getattr(o, "release_field", "")) or o.target.lower()


def stage_rows(o, offset_mas=None, offset_thresh=75.0):
    """Ordered [(label, status, when, detail)] derived from on-disk products.

    ``offset_mas`` is the measured JWST↔refcat bulk offset (from the diagnostics stage-4
    metric); pass it in to fill the refcat-comparison row's HOW.
    """
    fld = _fld(o)
    P = f"{BASE}/{fld}"
    prop = f"jw{int(o.program):05d}"
    oid = f"{prop}-o{o.obs}"
    rows = []

    def add(label, mtime, n, status=None, detail=""):
        if status is None:
            status = DONE if mtime else PEND
        rows.append((label, status, _ts(mtime), detail))
        return mtime

    crf_t, crf_n = _newest([f"{P}/*/pipeline/{prop}*-o{o.obs}*_crf.fits",
                            f"{P}/*/pipeline/{prop}{o.obs}*_crf.fits"])
    add("CRF (Image3 outlier detection)", crf_t, crf_n, detail=f"{crf_n} frames" if crf_n else "")
    ds_t, ds_n = _newest([f"{P}/*/pipeline/{prop}{o.obs}*_destreak.fits"])
    add("destreak", ds_t, ds_n, detail=f"{ds_n} frames" if ds_n else "")
    sf_t, sf_n = _newest([f"{P}/*/*visit*_daophot_basic.fits"])
    add("single-frame cataloging", sf_t, sf_n, detail=f"{sf_n} per-exposure catalogs" if sf_n else "")
    m7_t, m7_n = _newest([f"{P}/catalogs/*m7*.fits"])
    add("cross-frame catalog merge (m7)", m7_t, m7_n, detail="merged multi-band" if m7_t else "")

    # refcat comparison + JWST reference-frame creation
    off_txt = ""
    off_status = PEND
    if offset_mas is not None:
        flagged = offset_mas >= offset_thresh
        off_status = WARN if flagged else DONE
        off_txt = (f"bulk **{offset_mas:.0f} mas** vs VIRAC2 "
                   f"({'OFF-frame, needs re-tie' if flagged else 'within noise'})")
    rows.append(("refcat comparison (vs VIRAC2/Gaia)", off_status,
                 _ts(datetime.now(tz=timezone.utc).timestamp()) if offset_mas is not None else "—",
                 off_txt))
    tbl_t, tbl_n = _newest([f"{P}/offsets/*VIRAC2locked.csv"])
    add("JWST reference-frame creation (offsets table)", tbl_t, tbl_n,
        detail="per-exposure VIRAC2-locked tie" if tbl_t else "")

    # re-alignment: crf regenerated AFTER the offsets table was written = table applied
    if crf_t and tbl_t:
        if crf_t >= tbl_t:
            add("re-alignment of frames", crf_t, 1, status=DONE, detail="crf regenerated on current tie")
        else:
            add("re-alignment of frames", None, 0, status=RUN, detail="tie updated; re-reduce pending/queued")
    else:
        add("re-alignment of frames", None, 0, status=PEND)

    # cataloging m1..m8
    for k in range(1, 9):
        mt, mn = _newest([f"{P}/catalogs/*_m{k}_*.fits", f"{P}/catalogs/*_m{k}.fits"])
        st = DONE if mt else PEND
        add(f"cataloging m{k}", mt, mn, status=st, detail=f"{mn} catalogs" if mn else "")
    return rows


def render_status_block(o, offset_mas=None):
    rows = stage_rows(o, offset_mas=offset_mas)
    now = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    head = (f"{MARKER}\n### Pipeline progress — `{o.obsid}` ({o.target})\n"
            f"<sub>auto-updated {now} · ✅ done · 🔄 running/queued · ⬜ pending · "
            f"⚠️ done, flagged · ⏭️ skipped</sub>\n\n"
            f"| stage | status | last run (UTC) | detail |\n"
            f"|-------|:------:|----------------|--------|\n")
    body = "\n".join(f"| {lab} | {st} | {when} | {det} |" for lab, st, when, det in rows)
    return head + body + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--program", required=True)
    ap.add_argument("--obs", required=True)
    ap.add_argument("--field", default=None, help="on-disk field dir (default: infer)")
    ap.add_argument("--offset-mas", type=float, default=None,
                    help="measured JWST↔refcat bulk offset for the comparison row")
    ap.add_argument("--target", default=None)
    ap.add_argument("--post", action="store_true")
    ap.add_argument("--repo", default=os.environ.get("QA_REPO", "JWST-GC/data-qa"))
    args = ap.parse_args(argv)

    class _O:                       # light stand-in so this runs without the portal registry
        program = str(int(args.program)); obs = args.obs
        field = args.field or ""; target = args.target or (args.field or "").title()
        instrument = "NIRCam"
        obsid = f"jw{int(args.program):05d}-o{args.obs}"
        issue_title = f"{target} — {obsid} (NIRCam)"
    o = _O()
    block = render_status_block(o, offset_mas=args.offset_mas)
    print(block)
    if args.post:
        from .post_diagnostics import _token, _issue_number, _find_stage_comment, _req, API
        import json
        token = _token()
        num = _issue_number(args.repo, token, o.issue_title)
        if num is None:
            print(f"no issue titled {o.issue_title!r}", file=sys.stderr); return 1
        existing = _find_stage_comment(args.repo, token, num, MARKER)
        if existing:
            _req("PATCH", f"{API}/repos/{args.repo}/issues/comments/{existing['id']}", token,
                 data=json.dumps({"body": block}).encode())
            print(f"updated status comment on #{num}")
        else:
            _req("POST", f"{API}/repos/{args.repo}/issues/{num}/comments", token,
                 data=json.dumps({"body": block}).encode())
            print(f"created status comment on #{num}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
