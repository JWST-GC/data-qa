#!/bin/bash
# One QA issue per SLURM array element -- the capacity fix for #162.
#
# scripts/refresh_all_issues.sbatch runs the whole board serially in a 2 h window and covers
# ~21 issues, so the treasury tiles queued behind the established fields were never reached
# (job 40173146 was cancelled at the wall on 2026-09-13 with 8 tiles unstarted).  Here each
# element refreshes ONE issue, so the board is bounded by the slowest single issue rather
# than by their sum.
#
# Do not submit this directly -- scripts/submit_refresh_array.sh materialises the shared work
# list and sizes --array from it.  Submitting by hand without QA_WORK_LIST makes every
# element re-enumerate, which races the board.
#
# Posting is marker-keyed and idempotent and assets are replaced by name, so concurrent
# elements never conflict and a re-run never duplicates a comment.
#SBATCH --qos=astronomy-dept-b
#SBATCH --account=astronomy-dept
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --time=2:00:00
#SBATCH --job-name=data-qa-refresh-issue
#SBATCH --output=/orange/adamginsburg/jwst/logs/dataqa_refresh_%A_%a.out
set -uo pipefail

REPO_ROOT="${QA_REPO_ROOT:-/blue/adamginsburg/adamginsburg/repos/data-qa}"

# Token: prefer an already-exported GITHUB_TOKEN; else read a 600-perm PAT file.
if [ -z "${GITHUB_TOKEN:-}" ] && [ -f "$HOME/.config/data-qa/github_token" ]; then
    GITHUB_TOKEN="$(cat "$HOME/.config/data-qa/github_token")"
fi
export GITHUB_TOKEN

# Per-element scratch: a shared QA_OUTDIR would have concurrent elements overwrite each
# other's PNGs between rendering one and uploading it.
export QA_OUTDIR="${QA_OUTDIR:-${SLURM_TMPDIR:-/tmp}/qafigs_${SLURM_ARRAY_JOB_ID:-$$}_${SLURM_ARRAY_TASK_ID:-0}}"
mkdir -p "$QA_OUTDIR"

export QA_WORK_INDEX="${SLURM_ARRAY_TASK_ID:?this wrapper is only meaningful as an array job}"

cd "$REPO_ROOT"
bash scripts/refresh_all_issues.sh
