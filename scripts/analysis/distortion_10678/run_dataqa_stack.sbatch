#!/bin/bash
#SBATCH --job-name=jwst-gctreasury-10678-distcmp-dataqastack
#SBATCH --qos=astronomy-dept-b
#SBATCH --account=astronomy-dept
#SBATCH --cpus-per-task=16
#SBATCH --mem=96gb
#SBATCH --time=06:00:00
#SBATCH --output=/orange/adamginsburg/jwst/logs/%x_%j.log
# Submit from this directory:  sbatch --export=ALL,DISTCMP_WORKDIR=/path/to/workdir <this file>
HERE=${SLURM_SUBMIT_DIR}
: "${DISTCMP_WORKDIR:?set DISTCMP_WORKDIR (directory for matches/, empirical/, ...)}"
PY=${PY:-/blue/adamginsburg/adamginsburg/miniconda3/envs/python313/bin/python}
cd "$HERE/../../.." || exit 1
$PY -m data_qa.intradetector_distortion \
  --catdir /orange/adamginsburg/jwst/gc-treasury/catalogs_rollcorr/v1_dataqa346_perframe \
  --filters F212N F480M --outdir "$DISTCMP_WORKDIR"/dataqa_stack --nproc 16
