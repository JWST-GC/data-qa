# #375: pixel area (AREA map) against flux accuracy, GC Treasury 10678

Analysis scripts behind [data-qa#375](https://github.com/JWST-GC/data-qa/issues/375)
(results comment: https://github.com/JWST-GC/data-qa/issues/375#issuecomment-6242876984; context #362).
They test whether the NIRCam pixel-area map (`AREA`) term is correctly carried into F212N and F480M
photometry for all 68 tiles, against VIRAC2 / GNS Ks and the stage 13 zero-point network.

`results/REPORT.md` is the full write-up; `results/` also holds `tables.md` and the small JSON outputs
(`t1_*.json`, `test2.json`, `test3.json`, `net_<variant>_summary.json`). Large intermediates
(`frames/*.npz`, `test2_maps.npz`, `t1_psfgrid_nodes.npz`) and figures are not committed; the figures are on
the issue.

The numbers in `results/` came from the original scratch-directory copies of these scripts. The
repository copies differ only in paths (work directory, log directory, `data_qa` and pipeline
imports via `data_qa.paths.pipe_root()`, which honours `$PIPE_ROOT`), and the renames `test2.py` -> `t2_repeat_stars.py` and `test3.py` -> `t3_external_ref.py` (so a bare `pytest` run does not collect them); they have not been re-run.

## Main finding
STPSF grids sum to AREA^-0.97, so a PSF fit with correctly paired grid planes already includes the area
term. jwst-gc-pipeline 3393a5f5 (#1154, 2026-10-09) fixed a transposed plane pairing; every non-hybrid
`resbgsub_m7` per-frame catalogue (and therefore m8) predates it and carries approximately
a(x,y) - a(y,x), where a = 2.5 log10 AREA. Aperture photometry on crf frames needs x AREA; the i2d mosaics
do not.

## Files
| file | role |
|---|---|
| `common.py` | shared helpers: per-frame table loading, model terms (a, sT, pT), fixed-effects fits |
| `centres.json` | tile centres (RA, Dec) per obs, from the #362 analysis |
| `obslist.txt` | the 68 obs ids, one per line (array index for `build.sbatch`) |
| `t1_flat_vs_area.py` | test 1a: large-scale flat against AREA (CRDS references) |
| `t1_psfgrid_sum.py` | test 1b: STPSF grid sum against AREA; writes `t1_psfgrid_nodes.npz` |
| `build_frames.py <obs>` | per-frame star tables (PSF, hybrid PSF, aperture on crf) -> `frames/<filt>_o<obs>.npz` |
| `t2_repeat_stars.py` | test 2: frame-to-frame repeat-star differences against a / pT |
| `t3_external_ref.py` | test 3: absolute comparison with VIRAC2 and GNS Ks (Vega) |
| `test4_network.py` | test 4: stage 13 zero-point network with each area/grid correction variant |
| `tables.py`, `figs.py` | tables and figures for the report |

## Running
Python: `/blue/adamginsburg/adamginsburg/miniconda3/envs/python313/bin/python` (override with `PY=`).
Outputs go to `AREA375_WORKDIR` (Python scripts default to the current directory). Submit the batch
scripts from this directory, because they find the Python files through `$SLURM_SUBMIT_DIR`:

```
cd scripts/analysis/area_375
export AREA375_WORKDIR=/path/to/workdir
(cd $AREA375_WORKDIR && $PY $OLDPWD/t1_flat_vs_area.py && $PY $OLDPWD/t1_psfgrid_sum.py)
sbatch --export=ALL build.sbatch                       # array 1-68 -> frames/
sbatch --export=ALL t2_repeat_stars.sbatch                       # -> test2.json, test2_maps.npz
AREA375_GNS_DIR=/path/to/gns_tiles sbatch --export=ALL t3_external_ref.sbatch   # -> test3.json
for v in none area1 pT1 areak pTk pTvp pTvk; do sbatch --export=ALL,AREA_VARIANT=$v net.sbatch; done
(cd $AREA375_WORKDIR && $PY $OLDPWD/tables.py >| tables.md && $PY $OLDPWD/figs.py)
```
`t3_external_ref.py` reads GNS Ks tiles cut around each pointing (`AREA375_GNS_DIR`, default
`$AREA375_WORKDIR/gns_tiles`); these were built during the #362 analysis and are not part of this
directory. VIRAC2 tiles come from `/orange/adamginsburg/jwst/gc-treasury/catalogs/virac2_ks_cache/`.
Logs go to `/orange/adamginsburg/jwst/logs/<job-name>_<jobid>.log`.
