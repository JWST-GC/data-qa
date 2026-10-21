# #362: where the F212N per-detector offsets enter

Analysis scripts behind the 2026-10-21 comment on
[data-qa#362](https://github.com/JWST-GC/data-qa/issues/362) (per-detector part of fig C).

## Finding
The per-detector offsets enter at the merge. Each per-frame `flux_fit` is a sum of MJy/sr over that
detector's native pixels, so its flux in Jy needs that detector's `PIXAR_SR`. `merge_daophot`
(jwst-gc-pipeline `photometry/merge_catalogs.py`, 7c3a5ab2) multiplies the frame-averaged flux by one
pixel area per filter (the merged header's `212PIXS`, 0.0312229 arcsec, 2.2914e-14 sr, close to the NRCA1
value). The eight SW `PIXAR_SR` values span 2.222-2.312e-14 sr, so the merged magnitudes carry
+2.5 log10(PIXAR_SR / 2.2914e-14): A1 +0.2, A2 -31.2, A3 +9.7, A4 -21.9, B1 -33.6, B2 -2.0, B3 -24.6,
B4 +7.1 mmag (positive = too faint). F480M: ALONG +0.2, BLONG +3.5 mmag.

Measured on 10 tiles, merged `mag_vega` minus the per-frame hybrid-PSF magnitude (AB, frame `PIXAR_SR`)
matches the centred prediction to 1.8 mmag rms, with 1.1 mmag tile-to-tile rms. Subtracting it from
fig C removes the 27 mmag west-east column step and lowers the per-detector rms from 18 to 11 mmag.

## Files
| file | role |
|---|---|
| `t7_merge_vs_newframe.py` | catalogue-only test: current per-frame `_resbgsub_hybpsf_m7_daophot_basic.fits` against the merged hybpsf m7 catalogue, per detector |
| `t7_fig.py` | figure: fig C, measured merge term, `PIXAR_SR` prediction and the residual, per detector |
| `t6_merge_vs_frame.py` | the same comparison on the `area_375` frame tables (`build_frames.py` output); needs `AREA375_WORKDIR` |
| `results/t7.json` | t7 output, tiles o040 044 061 066 072 080 100 112 124 133 |
| `results/t7_fig_summary.json` | per-detector `PIXAR_SR`, fig C, merge term, prediction, residual |
| `results/figC_detectors.json` | fig C per-detector offsets (input to `t7_fig.py`), from the 2026-10-19 hybrid m7 snapshot |
| `results/t6_frames_2026-10-17.json` | t6 on the 2026-10-17 frame tables (pre-10-19 refit); same per-detector pattern |

## Usage
    python t7_merge_vs_newframe.py /orange/adamginsburg/jwst/gc-treasury/catalogs <out_dir> 040 044 ...
    python t7_fig.py <figC_detectors.json> <out_dir>/t7.json <out.png>
    AREA375_WORKDIR=<workdir> python t6_merge_vs_frame.py <catalog_dir>

`results/figC_detectors.json` is the output of the #362 fig C script (per-detector hybrid m7 minus VIRAC Ks,
both Vega); that script and its zero-point network inputs are not in this directory.
