#!/bin/bash
#SBATCH --job-name=jwst-gctreasury-10678-distcmp-empirical-F212N
#SBATCH --qos=astronomy-dept-b
#SBATCH --account=astronomy-dept
#SBATCH --cpus-per-task=4
#SBATCH --mem=160gb
#SBATCH --time=12:00:00
#SBATCH --output=/orange/adamginsburg/jwst/logs/%x_%j.log
# Submit from this directory:  sbatch --export=ALL,DISTCMP_WORKDIR=/path/to/workdir <this file>
HERE=${SLURM_SUBMIT_DIR}
: "${DISTCMP_WORKDIR:?set DISTCMP_WORKDIR (directory for matches/, empirical/, ...)}"
PY=${PY:-/blue/adamginsburg/adamginsburg/miniconda3/envs/python313/bin/python}
cd "$HERE" || exit 1
$PY empirical_test.py --filters F212N --matchdir "$DISTCMP_WORKDIR"/matches --outdir "$DISTCMP_WORKDIR"/empirical
