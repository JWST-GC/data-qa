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

REPO_ROOT="${QA_REPO_ROOT:-/blue/adamginsburg/adamginsburg/repos/data-qa}"

# Token: prefer an already-exported GITHUB_TOKEN; else read a 600-perm PAT file.
if [ -z "${GITHUB_TOKEN:-}" ] && [ -f "$HOME/.config/data-qa/github_token" ]; then
    GITHUB_TOKEN="$(cat "$HOME/.config/data-qa/github_token")"
fi
export GITHUB_TOKEN

export QA_OUTDIR="${QA_OUTDIR:-$SLURM_TMPDIR/qafigs}"
[ -n "${SLURM_TMPDIR:-}" ] || export QA_OUTDIR="/tmp/qafigs_$$"

cd "$REPO_ROOT"
bash scripts/refresh_all_issues.sh
