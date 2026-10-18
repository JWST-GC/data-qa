#!/bin/bash
#SBATCH --job-name=jwst-gctreasury-10678-area-t3
#SBATCH --qos=astronomy-dept-b --account=astronomy-dept
#SBATCH --cpus-per-task=2 --mem=48gb --time=04:00:00
#SBATCH --output=/orange/adamginsburg/jwst/logs/%x_%j.log
# Submit from this directory:  sbatch --export=ALL,AREA375_WORKDIR=/path/to/workdir <this file>
HERE=${SLURM_SUBMIT_DIR}
: "${AREA375_WORKDIR:?set AREA375_WORKDIR (directory for frames/, *.json, net_*/)}"
export AREA375_WORKDIR
mkdir -p "$AREA375_WORKDIR" && cd "$AREA375_WORKDIR" || exit 1
PY=${PY:-/blue/adamginsburg/adamginsburg/miniconda3/envs/python313/bin/python}
$PY -u "$HERE/t3_external_ref.py"
