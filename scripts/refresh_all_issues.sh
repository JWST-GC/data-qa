#!/bin/bash
# Refresh diagnostics (stages 1-5) + the pipeline-status table on EVERY open QA issue.
#
# Idempotent: every comment is marker-keyed and updated in place, so re-running never
# duplicates comments -- safe to run on any cadence.  Drives off the OPEN ISSUES (not the
# release registry) so not-yet-released observations (e.g. gc2211 o028/o046/o049, whose
# mosaics are only in images-merged/) are covered too.
#
# NIRCam issues  -> diagnostics (1-5) + pipeline status.
# MIRI issues    -> pipeline status only (diagnostics is NIRCam-only).
#
# Env:
#   GITHUB_TOKEN   required (repo-scoped PAT; or GH_TOKEN)
#   QA_REPO        default JWST-GC/data-qa
#   QA_BASE        default /orange/adamginsburg/jwst   (on-disk products)
#   QA_OUTDIR      scratch dir for the rendered PNGs   (default: mktemp)
#   REFRESH_STAGES default "1 2 3 4 5"
set -uo pipefail

REPO="${QA_REPO:-JWST-GC/data-qa}"
STAGES="${REFRESH_STAGES:-1 2 3 4 5}"
: "${GITHUB_TOKEN:=${GH_TOKEN:-$(gh auth token 2>/dev/null)}}"
export GITHUB_TOKEN
[ -n "${GITHUB_TOKEN:-}" ] || { echo "GITHUB_TOKEN not set" >&2; exit 2; }
export QA_OUTDIR="${QA_OUTDIR:-$(mktemp -d)}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Enumerate open issues -> "<program>\t<obs>\t<instrument>\t<display name>" from titles like
#   "GC 2211 — jw02211-o023 (NIRCam)"   (display name disambiguates shared programs)
mapfile -t SPECS < <(
  gh api "repos/$REPO/issues?state=open&per_page=100" --paginate -q '.[].title' 2>/dev/null \
    | python3 -c '
import re, sys
pat = re.compile(r"^(.*?)\s+[—-]\s+jw0*(\d+)-o(\d{3})\s+\((NIRCam|MIRI)\)", re.I)
for line in sys.stdin:
    m = pat.match(line.strip())
    if m:
        disp, prog, obs, inst = m.groups()
        print(f"{prog}\t{obs}\t{inst}\t{disp}")'
)
echo "refresh_all_issues: ${#SPECS[@]} open observation issues in $REPO"

rc_any=0
for spec in "${SPECS[@]}"; do
  IFS=$'\t' read -r prog obs inst disp <<< "$spec"
  echo "===== $disp — jw$(printf %05d "$prog")-o$obs ($inst) ====="
  if [ "${inst,,}" = "nircam" ]; then
    python3 -m data_qa.diagnostics --program "$prog" --obs "$obs" --target "$disp" --stage $STAGES --post \
      2>&1 | grep -iE "SW=|stage [0-9]:|created|updated|FAILED|no obs" || rc_any=1
  fi
  python3 -m data_qa.pipeline_status --program "$prog" --obs "$obs" --target "$disp" --instrument "$inst" --post \
    2>&1 | grep -iE "created|updated|status comment|no issue|FAILED" | tail -1 || rc_any=1
done
echo "refresh_all_issues: done (rc_any=$rc_any)"
exit "$rc_any"
