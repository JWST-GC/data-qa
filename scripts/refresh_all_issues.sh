#!/bin/bash
# Refresh diagnostics (stages 1-5) + the pipeline-status table on every open QA issue.
#
# Idempotent: every comment is marker-keyed and updated in place, and every image asset is
# replaced by name -- re-running never duplicates comments or accumulates assets, so it is
# safe on any cadence.  Drives off the OPEN ISSUES (not the release registry) so not-yet-
# released observations (e.g. gc2211 o028/o046/o049, mosaics only in images-merged/) are
# covered too.
#
# NIRCam issues -> diagnostics + pipeline status.   MIRI issues -> MIRI basics overview + status.
#
# Non-GC fields are SKIPPED (W51, Westerlund 1/2, NGC 6334, globular clusters). Override the
# skip list with QA_EXCLUDE_FIELDS (space-separated field keys) and/or QA_EXCLUDE_RE (a
# display-name regex).
#
# Env:
#   GITHUB_TOKEN        required (repo PAT; or GH_TOKEN; or ~/.config/data-qa/github_token)
#   QA_REPO             default JWST-GC/data-qa
#   QA_BASE             default /orange/adamginsburg/jwst   (on-disk products)
#   QA_OUTDIR           scratch dir for the rendered PNGs   (default: mktemp)
#   REFRESH_STAGES      default "1 2 3 4 5 6"
#   QA_EXCLUDE_FIELDS   default "w51 wd1 wd2 ngc6334"       (field keys to skip)
#   QA_EXCLUDE_RE       default "westerlund|ngc ?6334|globular|w51"  (display-name skip regex)
set -uo pipefail

REPO="${QA_REPO:-JWST-GC/data-qa}"
STAGES="${REFRESH_STAGES:-1 2 3 4 5 6 7 8 9}"

# Token: exported env, else a 600-perm PAT file, else gh's stored creds.
if [ -z "${GITHUB_TOKEN:-}" ]; then
    if [ -n "${GH_TOKEN:-}" ]; then
        GITHUB_TOKEN="$GH_TOKEN"
    elif [ -f "$HOME/.config/data-qa/github_token" ]; then
        GITHUB_TOKEN="$(cat "$HOME/.config/data-qa/github_token")"
    else
        GITHUB_TOKEN="$(gh auth token 2>/dev/null)"
    fi
fi
export GITHUB_TOKEN
[ -n "${GITHUB_TOKEN:-}" ] || { echo "GITHUB_TOKEN not set" >&2; exit 2; }
export QA_OUTDIR="${QA_OUTDIR:-$(mktemp -d)}"

# Auth preflight: gh falls back to a possibly-stale token in hosts.yml, and in a headless
# (scron/cron) env `gh auth token` can hand back an INVALID token -> a 401 that the
# enumeration below would swallow, silently "refreshing" 0 issues.  Fail LOUDLY instead so a
# broken token is obvious, not a quiet no-op every day.  (Provide a valid PAT via
# ~/.config/data-qa/github_token or the GITHUB_TOKEN env.)
if ! gh api user -q .login >/dev/null 2>&1; then
    echo "FATAL: GitHub auth failed (token invalid/expired). In a headless env, put a valid" >&2
    echo "       PAT in ~/.config/data-qa/github_token (Issues+Contents write on $REPO)." >&2
    exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Enumerate open issues -> "<program>\t<obs>\t<instrument>\t<display name>", reverse-mapping
# the title display name to its on-disk field key (via FIELDS) to apply the non-GC skip list.
mapfile -t SPECS < <(
  gh api "repos/$REPO/issues?state=open&per_page=100" --paginate -q '.[].title' 2>/dev/null \
    | QA_EXCLUDE_FIELDS="${QA_EXCLUDE_FIELDS:-w51 wd1 wd2 ngc6334}" \
      QA_EXCLUDE_RE="${QA_EXCLUDE_RE:-westerlund|ngc ?6334|globular|w51}" \
      python3 -c '
import re, sys, os
try:
    from data_qa.observations import FIELDS
except ImportError:
    FIELDS = {}
rev = {d.lower(): f for f, d in FIELDS.items()}          # "W51" -> "w51"
excl = set((os.environ.get("QA_EXCLUDE_FIELDS") or "").split())
excl_re = re.compile(os.environ.get("QA_EXCLUDE_RE") or r"(?!x)x", re.I)
pat = re.compile(r"^(.*?)\s+[—-]\s+jw0*(\d+)-o(\d{3})\s+\((NIRCam|MIRI)\)", re.I)
obsish = re.compile(r"jw\d{5}-o\d{3}", re.I)
for line in sys.stdin:
    t = line.strip()
    m = pat.match(t)
    if not m:
        # A title that CARRIES an obsid but did not match the full pattern (odd dash, altered
        # suffix, rename) would otherwise vanish silently -- this PR is about not letting QA
        # signals disappear quietly, so say so.  Titles with no obsid at all are meta issues.
        if obsish.search(t):
            print(f"WARN unmatched obs-issue title (skipped): {t!r}", file=sys.stderr)
        continue
    disp, prog, obs, inst = m.groups()
    field = rev.get(disp.lower(), disp.lower().replace(" ", ""))
    if field in excl or excl_re.search(disp):
        print(f"skip non-GC: {disp} (field={field})", file=sys.stderr)
        continue
    print(f"{prog}\t{obs}\t{inst}\t{disp}")'
)
echo "refresh_all_issues: ${#SPECS[@]} in-scope observation issues in $REPO"

rc_any=0
# rc_any is a REAL failure signal: set it on a non-zero exit or an error keyword in the
# output, NOT merely because the display-grep matched nothing (a quiet success prints little).
note_failure() { case "$1" in *FAILED*|*"no obs"*|*"no issue"*) return 0;; esac; [ "$2" -ne 0 ]; }
for spec in "${SPECS[@]}"; do
  IFS=$'\t' read -r prog obs inst disp <<< "$spec"
  echo "===== $disp — jw$(printf %05d "$prog")-o$obs ($inst) ====="
  if [ "${inst,,}" = "nircam" ]; then
    out=$(python3 -m data_qa.diagnostics --program "$prog" --obs "$obs" --target "$disp" --stage $STAGES --post 2>&1); drc=$?
    echo "$out" | grep -iE "SW=|stage [0-9]:|created|updated|FAILED|no obs" || true
    note_failure "$out" "$drc" && rc_any=1
  elif [ "${inst,,}" = "miri" ]; then
    # MIRI: basics overview (MAST i2d + Spitzer side-by-side + saturation mask)
    out=$(python3 -m data_qa.diagnostics --program "$prog" --obs "$obs" --target "$disp" --miri --post 2>&1); drc=$?
    echo "$out" | grep -iE "MIRI|created|updated|FAILED|no MIRI" || true
    note_failure "$out" "$drc" && rc_any=1
  fi
  sout=$(python3 -m data_qa.pipeline_status --program "$prog" --obs "$obs" --target "$disp" --instrument "$inst" --post 2>&1); src=$?
  echo "$sout" | grep -iE "created|updated|status comment|no issue|FAILED" | tail -1 || true
  note_failure "$sout" "$src" && rc_any=1
done
echo "refresh_all_issues: done (rc_any=$rc_any)"
exit "$rc_any"
