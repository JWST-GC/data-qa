#!/bin/bash
# Refresh diagnostics (stages 1-11) + the pipeline-status table on every open QA issue.
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
# ORDER MATTERS: this job has a 2 h wall (refresh_all_issues.sbatch) and has already been
# CANCELLED DUE TO TIME LIMIT with 21 issues in the list (job 40173146, 2026-09-02, 18 of 21
# done).  `gh api issues?state=open` returns NEWEST FIRST, so the per-tile treasury issues
# (program 10678: up to 139 of them over the campaign, one per delivered obs+instrument)
# would sort ahead of the 21 established field issues and the truncation would fall on the
# established fields instead of on the tiles.  The work list is therefore PARTITIONED --
# every non-treasury issue first, treasury tiles after -- so a walltime cut can only ever
# drop the newest treasury tiles.  QA_TREASURY_LAST=0 turns the partition off; the real
# capacity fix is the --array fan-out (#162).  Until #162 lands, tiles queued past the wall
# are NOT refreshed at all -- the partition chooses WHO gets dropped, it does not add
# capacity, and the run says so on the work-list line below.
#
# Env:
#   GITHUB_TOKEN        required (repo PAT; or GH_TOKEN; or ~/.config/data-qa/github_token)
#   QA_REPO             default JWST-GC/data-qa
#   QA_BASE             default /orange/adamginsburg/jwst   (on-disk products)
#   QA_OUTDIR           scratch dir for the rendered PNGs   (default: mktemp)
#   REFRESH_STAGES      default "1 2 3 4 5 6 7 8 9 10 11"
#   QA_EXCLUDE_FIELDS   default "w51 wd1 wd2 ngc6334"       (field keys to skip)
#   QA_EXCLUDE_RE       default "westerlund|ngc ?6334|globular|w51"  (display-name skip regex)
#   QA_TREASURY_LAST    default 1  (order program-10678 tiles after every other issue)
#   QA_TREASURY_PROGRAM default 10678 (must match data_qa.mast_monitor.TREASURY_PROGRAM)
#   QA_TREASURY_PENDING_DAYS default 14 (how long a tile may have NO products on disk
#                       before "waiting for the delivery" is reported as a failure)
set -uo pipefail

REPO="${QA_REPO:-JWST-GC/data-qa}"
STAGES="${REFRESH_STAGES:-1 2 3 4 5 6 7 8 9 10 11}"
# Kept in step with data_qa.mast_monitor.TREASURY_PROGRAM by
# tests/test_mast_monitor.py::test_refresh_script_treasury_program_matches_the_module.
TREASURY_PROGRAM="${QA_TREASURY_PROGRAM:-10678}"
TREASURY_LAST="${QA_TREASURY_LAST:-1}"
TREASURY_PENDING_DAYS="${QA_TREASURY_PENDING_DAYS:-14}"

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

# Enumerate open issues -> "<program>\t<obs>\t<instrument>\t<display name>\t<created_at>",
# reverse-mapping the title display name to its on-disk field key (via FIELDS) to apply the
# non-GC skip list.  The treasury tiles are moved to the END of the list (see ORDER MATTERS
# above).  created_at is what bounds the treasury PENDING window (see pending_tile below), so
# it is carried per issue rather than guessed.
mapfile -t SPECS < <(
  gh api "repos/$REPO/issues?state=open&per_page=100" --paginate \
      -q '.[] | [.title, .created_at] | @tsv' 2>/dev/null \
    | QA_EXCLUDE_FIELDS="${QA_EXCLUDE_FIELDS:-w51 wd1 wd2 ngc6334}" \
      QA_EXCLUDE_RE="${QA_EXCLUDE_RE:-westerlund|ngc ?6334|globular|w51}" \
      QA_TREASURY_PROGRAM="$TREASURY_PROGRAM" QA_TREASURY_LAST="$TREASURY_LAST" \
      QA_TREASURY_PENDING_DAYS="$TREASURY_PENDING_DAYS" \
      python3 -c '
import re, sys, os
try:
    from data_qa.observations import FIELDS
except ImportError:
    FIELDS = {}
rev = {d.lower(): f for f, d in FIELDS.items()}          # "W51" -> "w51"
excl = set((os.environ.get("QA_EXCLUDE_FIELDS") or "").split())
excl_re = re.compile(os.environ.get("QA_EXCLUDE_RE") or r"(?!x)x", re.I)
treasury_program = int(os.environ.get("QA_TREASURY_PROGRAM") or 10678)
treasury_last = (os.environ.get("QA_TREASURY_LAST") or "1") not in ("0", "", "no")
pat = re.compile(r"^(.*?)\s+[—-]\s+jw0*(\d+)-o(\d{3})\s+\((NIRCam|MIRI)\)", re.I)
obsish = re.compile(r"jw\d{5}-o\d{3}", re.I)
established, tiles = [], []
for line in sys.stdin:
    t, _, created = line.rstrip("\n").partition("\t")
    t = t.strip()
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
    # The issue API hands these back NEWEST FIRST, and this job is walltime-bound: a
    # per-tile treasury issue must never push an established field issue past the wall,
    # so the tiles go LAST and keep their own newest-first order among themselves.
    bucket = tiles if (treasury_last and int(prog) == treasury_program) else established
    bucket.append(f"{prog}\t{obs}\t{inst}\t{disp}\t{created.strip()}")
if tiles:
    print(f"work list: {len(established)} established + {len(tiles)} treasury tile(s), "
          f"tiles last (QA_TREASURY_LAST=0 to interleave)", file=sys.stderr)
    # The partition decides WHO is dropped by the 2 h wall, not how many fit.  Say what this
    # run will actually cover so "the tiles have issues" is never read as "the tiles have QA".
    print(f"NOTE: the serial loop covers ~21 issues within its 2 h wall; {len(tiles)} tile(s) "
          f"queued after {len(established)} established issue(s) may not be reached this run "
          f"-- the capacity fix is the --array fan-out (#162)", file=sys.stderr)
for row in established + tiles:
    print(row)'
)
echo "refresh_all_issues: ${#SPECS[@]} in-scope observation issues in $REPO"

rc_any=0
# --- classifiers (extracted VERBATIM by tests/test_mast_monitor.py; keep them self-contained)
# rc_any is a REAL failure signal: set it on a non-zero exit or an error keyword in the
# output, NOT merely because the display-grep matched nothing (a quiet success prints little).
failure_keyword() { case "$1" in *FAILED*|*"no obs"*|*"no issue"*) return 0;; esac; return 1; }
note_failure() { failure_keyword "$1" && return 0; [ "$2" -ne 0 ]; }

# ...with one EXPECTED, BOUNDED and NON-ABSORBING exception.  A treasury tile's QA issue is
# opened by `data_qa.mast_monitor --report` when MAST RELEASES the tile; the calibrated
# products land on our disk days to weeks later, and 10678 is uncurated so the portal
# registry returns nothing for it either.  Until then diagnostics prints
#   no obs for program 10678 obs 088 (portal + on-disk both empty)
# and exits 1 -- the expected state of a just-delivered tile, not a broken field.  rc_any is
# the single pass/fail signal shared with the 20+ curated field issues, so without this the
# first treasury arrival would leave it red for the rest of the campaign.
#
# Three things keep the exemption from becoming a blanket green:
#   * it NEVER ABSORBS A CO-OCCURRING FAILURE.  The pending line is stripped and the keyword
#     test re-applied to what is left, and the exit code must be diagnostics' own no-obs 1.
#     Output carrying both the pending message and "stage 4: FAILED" is a FAILURE.
#   * it is TIME-BOUNDED by the tile issue's own created_at.  Past
#     QA_TREASURY_PENDING_DAYS (14) a tile with still nothing visible is reported as a
#     failure and named as stale, so a delivery that stalled -- or a QA glob that cannot
#     reach products that ARE on disk (#163: the monitor's own
#     /orange/adamginsburg/jwst/ops/downloads/mastDownload tree sits one level below every
#     MAST glob, and a locally-reduced MIRI mosaic is not globbed at all) -- ESCALATES
#     instead of sitting green forever.  An unknown/unparseable created_at is a failure too
#     (fail closed).
#   * it is treasury-only: a curated field printing the same message is a failure.
PENDING_AGE_DAYS=""
pending_tile() {   # $1=program $2=output $3=exit code $4=issue created_at
                   # rc 0 = fresh PENDING (green), 2 = PAST the window (red, stale), 1 = not this case
  PENDING_AGE_DAYS=""
  [ "$1" = "$TREASURY_PROGRAM" ] || return 1
  case "$2" in *"portal + on-disk"*) ;; *) return 1;; esac
  # the pending message must be the ONLY failure signal
  [ "$3" -eq 1 ] || return 1
  rest=$(printf '%s\n' "$2" | grep -v "portal + on-disk")
  failure_keyword "$rest" && return 1
  # ...and the wait must be inside the window
  [ -n "${4:-}" ] || return 1
  opened=$(date -d "$4" +%s 2>/dev/null) || return 1
  [ -n "$opened" ] || return 1
  PENDING_AGE_DAYS=$(( ( $(date +%s) - opened ) / 86400 ))
  [ "$PENDING_AGE_DAYS" -lt "$TREASURY_PENDING_DAYS" ] || return 2
  return 0
}

# Establish the failure FIRST, then narrowly downgrade it -- the exemption can only ever
# soften a verdict note_failure has already reached, never pre-empt the classification.
classify() {   # $1=program $2=output $3=exit code $4=created_at $5=what is missing
  note_failure "$2" "$3" || return 0
  pending_tile "$1" "$2" "$3" "$4"; ptrc=$?
  if [ "$ptrc" -eq 0 ]; then
    echo "PENDING treasury tile (${PENDING_AGE_DAYS}d): $5 (not a failure; window ${TREASURY_PENDING_DAYS}d)"
    return 0
  fi
  if [ "$ptrc" -eq 2 ]; then
    echo "FAILED (stale treasury tile): $5 after ${PENDING_AGE_DAYS}d open, past the" \
         "${TREASURY_PENDING_DAYS}d QA_TREASURY_PENDING_DAYS window -- the delivery stalled" \
         "or a QA glob cannot see products that are on disk (#163)"
  fi
  rc_any=1
}
# --- end classifiers
for spec in "${SPECS[@]}"; do
  IFS=$'\t' read -r prog obs inst disp created <<< "$spec"
  echo "===== $disp — jw$(printf %05d "$prog")-o$obs ($inst) ====="
  if [ "${inst,,}" = "nircam" ]; then
    out=$(python3 -m data_qa.diagnostics --program "$prog" --obs "$obs" --target "$disp" --stage $STAGES --post 2>&1); drc=$?
    echo "$out" | grep -iE "SW=|stage [0-9]+:|created|updated|FAILED|no obs" || true
    classify "$prog" "$out" "$drc" "$created" "no products on disk yet"
  elif [ "${inst,,}" = "miri" ]; then
    # MIRI: basics overview (MAST i2d + Spitzer side-by-side + saturation mask)
    out=$(python3 -m data_qa.diagnostics --program "$prog" --obs "$obs" --target "$disp" --miri --post 2>&1); drc=$?
    echo "$out" | grep -iE "MIRI|created|updated|FAILED|no MIRI" || true
    classify "$prog" "$out" "$drc" "$created" "no MIRI products on disk yet"
  fi
  sout=$(python3 -m data_qa.pipeline_status --program "$prog" --obs "$obs" --target "$disp" --instrument "$inst" --post 2>&1); src=$?
  echo "$sout" | grep -iE "created|updated|status comment|no issue|FAILED" | tail -1 || true
  note_failure "$sout" "$src" && rc_any=1
done
echo "refresh_all_issues: done (rc_any=$rc_any)"
exit "$rc_any"
