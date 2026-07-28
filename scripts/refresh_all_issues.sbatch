#!/bin/bash
# Refresh diagnostics + pipeline-status on every open QA issue, as a burst-QOS SLURM job.
# Keeps the (single-core, ~35 min) source-detection work OFF the login node.
#
# Submit directly:   sbatch scripts/refresh_all_issues.sbatch
# Or from cron:      see scripts/README or the crontab example.
#
# Requires GITHUB_TOKEN to be exported into the job (SBATCH does NOT inherit your login env
# by default -- pass it with `sbatch --export=ALL,GITHUB_TOKEN=... ` or store a PAT file and
# source it below).
#SBATCH --qos=astronomy-dept-b
#SBATCH --account=astronomy-dept
#SBATCH --cpus-per-task=4
#SBATCH --mem=32gb
#SBATCH --time=2:00:00
#SBATCH --job-name=data-qa-refresh-all-issues
#SBATCH --output=/orange/adamginsburg/jwst/logs/dataqa_refresh_%j.out
set -euo pipefail

REPO_ROOT="${QA_REPO_ROOT:-/blue/adamginsburg/adamginsburg/repos/data-qa}"

# Token: prefer an already-exported GITHUB_TOKEN; else read a 600-perm PAT file.
if [ -z "${GITHUB_TOKEN:-}" ] && [ -f "$HOME/.config/data-qa/github_token" ]; then
    GITHUB_TOKEN="$(cat "$HOME/.config/data-qa/github_token")"
fi
export GITHUB_TOKEN

# Respect a caller-supplied QA_OUTDIR; only pick a default when it is unset (SLURM_TMPDIR is
# usually unset here, so keying the default off it would clobber the caller's dir).
if [ -z "${QA_OUTDIR:-}" ]; then
    export QA_OUTDIR="${SLURM_TMPDIR:-/tmp}/qafigs_${SLURM_JOB_ID:-$$}"
fi

cd "$REPO_ROOT"
bash scripts/refresh_all_issues.sh
