#!/bin/bash
#SBATCH --job-name=jwst-gctreasury-10678-distcmp-catimpact
#SBATCH --qos=astronomy-dept-b
#SBATCH --account=astronomy-dept
#SBATCH --cpus-per-task=2
#SBATCH --mem=96gb
#SBATCH --time=02:00:00
#SBATCH --output=/orange/adamginsburg/jwst/logs/%x_%j.log
# Submit from this directory:  sbatch --export=ALL,DISTCMP_WORKDIR=/path/to/workdir <this file>
HERE=${SLURM_SUBMIT_DIR}
: "${DISTCMP_WORKDIR:?set DISTCMP_WORKDIR (directory for matches/, empirical/, ...)}"
PY=${PY:-/blue/adamginsburg/adamginsburg/miniconda3/envs/python313/bin/python}
cd "$HERE" || exit 1
$PY catalog_impact.py --filters F212N F480M --matchdir "$DISTCMP_WORKDIR"/matches --empdir "$DISTCMP_WORKDIR"/empirical --out "$DISTCMP_WORKDIR"/empirical/catalog_impact.json
