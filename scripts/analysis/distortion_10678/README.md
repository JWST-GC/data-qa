# GC Treasury 10678: CRDS distortion against STDGDC and the data-qa map

Analysis scripts behind [data-qa#376](https://github.com/JWST-GC/data-qa/issues/376). Three
intra-detector distortion solutions are compared on the 68 tiles of 10678 (8 SW detectors in F212N, 2 LW
detectors in F480M):

1. CRDS as used (`jwst_1596.pmap`);
2. Jay Anderson's STDGDC (F212N v1/v2 only), `/orange/adamginsburg/jwst/distortion/jayander_stdgdc/NIRCam`;
3. the data-qa measured map (all-pointing stack of the stage 8 residual, `python -m data_qa.intradetector_distortion`).

`results/tables.md` and `results/summary.json` hold the numbers quoted on the issue; figures are on the issue.

## Main finding
CRDS leaves a repeatable residual of 0.57-0.80 mas (F212N) and 0.52-0.53 mas (F480M) binned RMS. STDGDC
lowers it to 0.27-0.49 mas and the cross-validated data-qa map to 0.22-0.29 mas (F212N) and 0.25 mas
(F480M), improving every tile. The effect on consensus catalogue positions is 0.41-0.58 mas RMS in F212N.
Recommendation on the issue: keep CRDS for the catalogue; use an empirical per-filter correction only
for sub-mas work such as proper motions.

## Files
| file | role |
|---|---|
| `fields.py` | distortion fields: CRDS (asdf), STDGDC, data-qa maps; affine anchoring |
| `extract_matches.py` | frame <-> tile-consensus matched pairs -> `matches/matches_<filt>.npz` |
| `extract_refcat.py` | frame <-> Gaia DR3 / VIRAC2 refcat pairs -> `matches/refmatches_<filt>.npz` |
| `model_compare.py` | (a) direct model-to-model comparison on a grid -> `model_compare/` |
| `empirical_test.py` | (b) data tests: binned residual per solution, cross-validated maps, dilution -> `empirical/` |
| `catalog_impact.py` | (c) shift of consensus star positions under each correction |
| `refcat_paired.py` | absolute (refcat) check with paired differences |
| `summarise.py` | `tables.md` + `summary.json` from the above |
| `make_figures.py` | figures |

## Running
Python: `/blue/adamginsburg/adamginsburg/miniconda3/envs/python313/bin/python` (override with `PY=`).
Submit from this directory with `DISTCMP_WORKDIR` set; logs go to `/orange/adamginsburg/jwst/logs/`.

```
cd scripts/analysis/distortion_10678
export DISTCMP_WORKDIR=/path/to/workdir
sbatch --export=ALL run_extract.sbatch          # -> matches/matches_*.npz
sbatch --export=ALL run_extractref.sbatch       # -> matches/refmatches_*.npz
sbatch --export=ALL run_dataqa_stack.sbatch     # -> dataqa_stack/intradet_*.json
$PY model_compare.py --dqdir $DISTCMP_WORKDIR/dataqa_stack --outdir $DISTCMP_WORKDIR/model_compare   # after dataqa_stack
sbatch --export=ALL run_empirical_F212N.sbatch  # after the extracts
sbatch --export=ALL run_empirical_F480M.sbatch
sbatch --export=ALL run_catimpact.sbatch        # after empirical
sbatch --export=ALL run_refpaired.sbatch        # after empirical
$PY summarise.py --distcmp $DISTCMP_WORKDIR
$PY make_figures.py --distcmp $DISTCMP_WORKDIR
```
