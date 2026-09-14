#!/bin/bash
# Submit the QA issue refresh as a SLURM array -- one element per open issue (#162).
#
#   bash scripts/submit_refresh_array.sh
#
# Enumerates the board ONCE, writes that snapshot to a work-list file, and sizes --array from
# it.  The single snapshot is the point: an element that re-enumerated for itself would race
# an issue being opened or closed mid-run, shifting every later row so that one issue is
# covered twice and another not at all.
#
# Env:
#   QA_ARRAY_THROTTLE  default 6   concurrent elements (the %N in --array)
#   QA_WORK_LIST_DIR   default /orange/adamginsburg/jwst/logs   where the snapshot is kept
#   ...plus every variable scripts/refresh_all_issues.sh reads (QA_REPO, REFRESH_STAGES,
#      QA_EXCLUDE_FIELDS, QA_TREASURY_LAST, ...), which are passed through to the elements.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

THROTTLE="${QA_ARRAY_THROTTLE:-6}"
LIST_DIR="${QA_WORK_LIST_DIR:-/orange/adamginsburg/jwst/logs}"
mkdir -p "$LIST_DIR"
LIST="$LIST_DIR/qa_worklist_$(date -u +%Y%m%dT%H%M%SZ).tsv"

QA_LIST_ONLY=1 bash scripts/refresh_all_issues.sh > "$LIST"
N=$(wc -l < "$LIST")
[ "$N" -gt 0 ] || { echo "work list is empty; nothing to submit" >&2; exit 1; }
echo "work list: $N issue(s) -> $LIST"

# The elements read the snapshot, so it has to outlive this shell; --export=ALL carries the
# QA_* settings this run was given so an element refreshes the same board it was sized for.
sbatch --array="0-$((N - 1))%${THROTTLE}" \
       --export=ALL,QA_WORK_LIST="$LIST" \
       scripts/refresh_all_issues_array.sbatch
