#!/bin/bash
#SBATCH --account=astronomy-dept
#SBATCH --qos=astronomy-dept-b          # burst; astronomy-dept / adamginsburg priority QOS forbidden
#SBATCH --partition=hpg-default
#SBATCH --cpus-per-task=4
#SBATCH --mem=16gb
#SBATCH --time=04:00:00
#SBATCH --output=/orange/adamginsburg/jwst/logs/peppar_%x_%j.out
# Runs ONE (filter, detector) peppar extraction for our download-and-push HiPerGator pipeline.
# This wrapper + its runner (run_peppar_generic.py) are OURS (JWST-GC/data-qa), NOT part of
# Matt Hosek's peppar package -- they only drive it. The peppar PACKAGE is imported from a
# checkout on PYTHONPATH (PEPPAR_REPO); the runner script is ours (PEPPAR_RUNNER).
#
# Env supplied by data_qa.peppar_trigger via `sbatch --export=ALL,PEPPAR_*=...`:
#   PEPPAR_DATA_DIR / PEPPAR_STF_DIR / PEPPAR_FILT / PEPPAR_DET   (see run_peppar_generic.py)
#   PEPPAR_RUNNER   abs path to our run_peppar_generic.py (default: alongside this script)
#   PEPPAR_REPO     peppar package checkout (for `from peppar import peppar`)
#   PEPPAR_ENV      conda env prefix
#
# NOTE (2026-07-29): the peppar env currently imports a dev photutils (repos/photutils 3.0.1)
# that shadows the pinned photutils==1.12.0 peppar needs -> `import peppar` fails
# (cannot import name 'IntegratedGaussianPRF'). This job will not succeed until that env is
# repaired -- fix in the env, not here.
set -euo pipefail

PEPPAR_REPO="${PEPPAR_REPO:-/blue/adamginsburg/adamginsburg/repos/peppar}"
PEPPAR_ENV="${PEPPAR_ENV:-/orange/adamginsburg/miniconda3/envs/peppar}"
# default the runner to the copy sitting next to this sbatch
_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PEPPAR_RUNNER="${PEPPAR_RUNNER:-${_HERE}/run_peppar_generic.py}"

source "${PEPPAR_ENV}/bin/activate"
export PYTHONPATH="${PEPPAR_REPO}"                 # peppar package only -- no dev-checkout paths
export PYSYN_CDBS="${PYSYN_CDBS:-/orange/adamginsburg/synphot/grp/hst/cdbs}"   # webbpsf phoenix grid

echo "peppar: FILT=${PEPPAR_FILT} DET=${PEPPAR_DET} DATA_DIR=${PEPPAR_DATA_DIR} STF_DIR=${PEPPAR_STF_DIR}"
echo "peppar: runner=${PEPPAR_RUNNER} pkg=${PEPPAR_REPO}"
python "${PEPPAR_RUNNER}"
