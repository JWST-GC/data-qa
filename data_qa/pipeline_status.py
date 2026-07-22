"""Auto-updating pipeline-stage progress table for a per-observation QA issue.

Derives, from the on-disk products, WHICH reduction/cataloging stages have run for an
observation, WHEN (newest product mtime), and HOW they did (counts + measured astrometric
offset).  Rendered as one idempotent comment on the tracking issue (marker-keyed, updated
in place), so the issue always shows the live pipeline state as it advances.

Stage order (rising):
    CRF -> destreak -> single-frame cataloging -> cross-frame merge -> refcat comparison
    -> JWST reference-frame creation (w/ measured offset) -> re-alignment
    -> cataloging m1 -> ... -> m8

Status: ✅ done   🔄 running/queued   ⬜ pending   ⚠️ done, flagged   🛑 stale   ⏭️ skipped

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
DONE, RUN, PEND, WARN, SKIP, STALE = "✅", "🔄", "⬜", "⚠️", "⏭️", "🛑"


def _newest(pats):
    """(mtime, count) for a glob set.  mtime is the true max over ALL matches (correctness:
    an arbitrary slice can miss the newest and flip the re-alignment gate).  Scoping the
    globs to one obsid (below) keeps the set small enough to stat on NFS."""
    files = []
    for pat in pats:
        files += glob.glob(pat)
    if not files:
        return None, 0
    return max(os.path.getmtime(f) for f in files), len(files)


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

    # obsid-scoped products (carry -o{obs}/{obs} in the filename)
    crf_t, crf_n = _newest([f"{P}/*/pipeline/{prop}*-o{o.obs}*_crf.fits",
                            f"{P}/*/pipeline/{prop}{o.obs}*_crf.fits"])
    add("CRF (Image3 outlier detection)", crf_t, crf_n, detail=f"{crf_n} frames" if crf_n else "")
    ds_t, ds_n = _newest([f"{P}/*/pipeline/{prop}*-o{o.obs}*_destreak.fits",
                          f"{P}/*/pipeline/{prop}{o.obs}*_destreak.fits"])
    add("destreak", ds_t, ds_n, detail=f"{ds_n} frames" if ds_n else "")

    # FIELD-SHARED products: per-frame cats + merged catalogs carry no obsid, so on a field
    # whose dir holds more than one program (e.g. Brick = 2221 + 1182) these counts can
    # include a sibling observation's products.  Flag that rather than falsely obs-attribute.
    n_programs = len({os.path.basename(p)[2:7]
                      for p in glob.glob(f"{P}/*/pipeline/jw?????*-o*_crf.fits")})
    shared = " · field-shared, may include sibling obs" if n_programs > 1 else ""
    sf_t, sf_n = _newest([f"{P}/*/*visit*_daophot_basic.fits"])
    add("single-frame cataloging", sf_t, sf_n,
        detail=(f"{sf_n} per-exposure catalogs" + shared) if sf_n else "")
    m7_t, m7_n = _newest([f"{P}/catalogs/*m7*.fits"])
    add("cross-frame catalog merge (m7)", m7_t, m7_n,
        detail=("merged multi-band" + shared) if m7_t else "")

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

    # cataloging m1..m8 (field-shared: see caveat above).  These must be monotonically
    # non-decreasing in time (m8 is derived from m7, ...).  A later stage whose newest file
    # is OLDER than an earlier stage means it was NOT regenerated after that earlier stage
    # changed -> it is STALE (🛑), not done: the pipeline must re-run it.
    newest_earlier = None
    for k in range(1, 9):
        mt, mn = _newest([f"{P}/catalogs/*_m{k}_*.fits", f"{P}/catalogs/*_m{k}.fits"])
        if not mt:
            st, det = PEND, ""
        elif newest_earlier is not None and mt < newest_earlier - 1.0:
            st = STALE
            det = f"{mn} catalogs · STALE (older than an earlier m-stage; re-run)" + shared
        else:
            st = DONE
            det = f"{mn} catalogs" + shared
        add(f"cataloging m{k}", mt, mn, status=st, detail=det)
        if mt:
            newest_earlier = mt if newest_earlier is None else max(newest_earlier, mt)
    return rows


def render_status_block(o, offset_mas=None):
    rows = stage_rows(o, offset_mas=offset_mas)
    now = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    head = (f"{MARKER}\n### Pipeline progress — `{o.obsid}` ({o.target})\n"
            f"<sub>auto-updated {now} · ✅ done · 🔄 running/queued · ⬜ pending · "
            f"⚠️ done, flagged · 🛑 stale (out-of-order timestamp) · ⏭️ skipped</sub>\n\n"
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

    obs = f"{int(args.obs):03d}"     # zero-pad so globs use -o001 not -o1
    # display name must match the issue title exactly -- use the FIELDS map ("cloudc"->"Cloud
    # C", "sgrb2"->"Sgr B2"), NOT field.title() ("Cloudc"), else the issue lookup misses and
    # no status comment is posted.
    from .observations import FIELDS
    _target = args.target or FIELDS.get(args.field or "", (args.field or "").title())

    class _O:                       # light stand-in so this runs without the portal registry
        program = str(int(args.program))
        field = args.field or ""; target = _target
        instrument = "NIRCam"
        obsid = f"jw{int(args.program):05d}-o{obs}"
        issue_title = f"{target} — {obsid} (NIRCam)"
    _O.obs = obs
    o = _O()
    block = render_status_block(o, offset_mas=args.offset_mas)
    print(block)
    if args.post:
        try:
            from .post_diagnostics import _token, _issue_number, _find_stage_comment, _req, API
        except ImportError:
            print("post requires data_qa.post_diagnostics (merges with PR #17); skipping post",
                  file=sys.stderr)
            return 3
        import json
        token = _token()
        num = _issue_number(args.repo, token, o.issue_title)
        if num is None:
            print(f"no issue titled {o.issue_title!r}", file=sys.stderr); return 1
        existing = _find_stage_comment(args.repo, token, num, MARKER)
        if existing:
            st, data = _req("PATCH", f"{API}/repos/{args.repo}/issues/comments/{existing['id']}",
                            token, data=json.dumps({"body": block}).encode())
            action = "updated"
        else:
            st, data = _req("POST", f"{API}/repos/{args.repo}/issues/{num}/comments",
                            token, data=json.dumps({"body": block}).encode())
            action = "created"
        if st >= 300:
            print(f"comment {action} FAILED ({st}): {data}", file=sys.stderr); return 1
        print(f"{action} status comment on #{num}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
