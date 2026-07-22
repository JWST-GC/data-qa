#!/bin/bash
#SBATCH --job-name=gc-treasury-hips
#SBATCH --account=astronomy-dept
#SBATCH --qos=astronomy-dept-b
#SBATCH --partition=hpg-default
#SBATCH --cpus-per-task=8
#SBATCH --mem=64gb
#SBATCH --time=12:00:00
#SBATCH --output=/blue/adamginsburg/adamginsburg/slurm_logs/%x_%A.out
#
# Fold one field's F212N + long-band mosaics into the jwst-gc-treasury-hips
# mono HiPS substrates (data_qa.hips_treasury build).  Reprojecting the real
# multi-GB mosaics into HiPS tiles is a compute job, not a login-node run.
#
# Submit WITH the job name at submit time (standing SLURM naming rule --
# target+stage; the runtime rename only fires once the job starts):
#
#   sbatch --job-name=gc-treasury-hips-<field> \
#     docs/submit_treasury_hips.sbatch /path/to/treasury_hips_spec.json <field>
#
# After all fields are in, derive the color layer (cheap; login node is fine):
#   python -m data_qa.hips_treasury color --spec /path/to/treasury_hips_spec.json
#
set -euo pipefail

SPEC="${1:?usage: sbatch submit_treasury_hips.sbatch <spec.json> <field> [extra args...]}"
FIELD="${2:?usage: sbatch submit_treasury_hips.sbatch <spec.json> <field> [extra args...]}"
shift 2 || true

export MPLBACKEND=Agg
export TMPDIR="${SLURM_TMPDIR:-/blue/adamginsburg/adamginsburg/tmp}"

# data-qa repo root = one level up from this script's docs/ dir.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
# jwst-gc-pipeline provides cmz.hips (the actual HiPS builder/merger).
PIPE_ROOT="${PIPE_ROOT:-/blue/adamginsburg/adamginsburg/repos/jwst-gc-pipeline}"
export PYTHONPATH="$REPO:$PIPE_ROOT:${PYTHONPATH:-}"

PYTHON="${PYTHON:-/blue/adamginsburg/adamginsburg/miniconda3/envs/python313/bin/python}"

echo "treasury HiPS build: spec=$SPEC field=$FIELD extra='$*' repo=$REPO"
"$PYTHON" -m data_qa.hips_treasury build --spec "$SPEC" --field "$FIELD" \
    --threads "${SLURM_CPUS_PER_TASK:-8}" "$@"
echo "treasury HiPS build done."
