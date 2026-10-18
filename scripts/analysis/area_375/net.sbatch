#!/bin/bash
#SBATCH --job-name=jwst-gctreasury-10678-area-network
#SBATCH --qos=astronomy-dept-b --account=astronomy-dept
#SBATCH --cpus-per-task=16 --mem=64gb --time=03:00:00
#SBATCH --output=/orange/adamginsburg/jwst/logs/%x_%j.log
# Submit from this directory:  sbatch --export=ALL,AREA375_WORKDIR=/path/to/workdir <this file>
HERE=${SLURM_SUBMIT_DIR}
: "${AREA375_WORKDIR:?set AREA375_WORKDIR (directory for frames/, *.json, net_*/)}"
export AREA375_WORKDIR
mkdir -p "$AREA375_WORKDIR" && cd "$AREA375_WORKDIR" || exit 1
PY=${PY:-/blue/adamginsburg/adamginsburg/miniconda3/envs/python313/bin/python}
# AREA_VARIANT selects the area/temperature model: none area1 areak pT1 pTk pTvk pTvp
: "${AREA_VARIANT:?set AREA_VARIANT}"
export QA_VIRAC_DOWNLOAD=0
echo "AREA_VARIANT=$AREA_VARIANT"
$PY "$HERE/test4_network.py"
