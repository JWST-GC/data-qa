#!/bin/bash
#SBATCH --job-name=jwst-gctreasury-10678-area-build
#SBATCH --qos=astronomy-dept-b --account=astronomy-dept
#SBATCH --cpus-per-task=1 --mem=12gb --time=01:30:00
#SBATCH --array=1-68%34
#SBATCH --output=/orange/adamginsburg/jwst/logs/%x_%A_%a.log
# Submit from this directory:  sbatch --export=ALL,AREA375_WORKDIR=/path/to/workdir <this file>
HERE=${SLURM_SUBMIT_DIR}
: "${AREA375_WORKDIR:?set AREA375_WORKDIR (directory for frames/, *.json, net_*/)}"
export AREA375_WORKDIR
mkdir -p "$AREA375_WORKDIR" && cd "$AREA375_WORKDIR" || exit 1
PY=${PY:-/blue/adamginsburg/adamginsburg/miniconda3/envs/python313/bin/python}
OBS=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$HERE/obslist.txt")
$PY "$HERE/build_frames.py" $OBS
