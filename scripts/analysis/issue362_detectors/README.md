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

## Follow-up (2026-10-23): per-frame photometry against Ks
`t5_detector_offsets.py` on frame tables rebuilt 2026-10-22 from the current hybpsf m7 per-frame fits
(68 tiles). Hybrid PSF minus aperture on the same crf frame is within 3.5 mmag on every detector for
the r2 (6 px) aperture and within 4.8 mmag for r1 (3.5 px; A3 +4.2, B3 -4.8). The plain STPSF per-frame
fits go further (A3 +9.5 / B3 -5.9 against r2, +8.4 / -7.7 against r1; bootstrap errors < 1 mmag); those
catalogues predate jwst-gc-pipeline #1154 (#375). The A3/B3 internal offsets have the sign of the Ks
pattern below, so the PSF carries a small share of it (up to about 5 mmag for the hybrid PSF). Against
VIRAC Ks, the per-frame PSF and the
aperture photometry show the same per-detector pattern (hybrid PSF, centred: A1 -6, A2 -4, A3 +16,
A4 +23, B1 +10, B2 -1, B3 -23, B4 -15 mmag, bootstrap errors about 2). Aperture photometry on the crf
frames carries it too, so it enters before the PSF fit and the merge: in the image calibration or in the
Ks comparison.
t6 on the rebuilt tables (68 tiles) reproduces the merge's pixel-area term.

## Files
| file | role |
|---|---|
| `t7_merge_vs_newframe.py` | catalogue-only test: current per-frame `_resbgsub_hybpsf_m7_daophot_basic.fits` against the merged hybpsf m7 catalogue, per detector |
| `t7_fig.py` | figure: fig C, measured merge term, `PIXAR_SR` prediction and the residual, per detector |
| `t5_detector_offsets.py` | per detector: hybrid PSF minus aperture (internal), and PSF / aperture minus VIRAC Ks with (tile, exposure) offsets and colour terms (fixed-effects fit); needs `AREA375_WORKDIR` with rebuilt `frames/` |
| `t3_lib.py` | `ref_tile` / `match` from `area_375/t3_external_ref.py`, imported by t5 |
| `t6_merge_vs_frame.py` | the same comparison on the `area_375` frame tables (`build_frames.py` output); needs `AREA375_WORKDIR` |
| `results/t7.json` | t7 output, tiles o040 044 061 066 072 080 100 112 124 133 |
| `results/t7_fig_summary.json` | per-detector `PIXAR_SR`, fig C, merge term, prediction, residual |
| `results/figC_detectors.json` | fig C per-detector offsets (input to `t7_fig.py`), from the 2026-10-19 hybrid m7 snapshot |
| `results/t5.json` | t5 output (2026-10-22 frame tables) |
| `results/t6_frames_2026-10-22.json` | t6 on the 2026-10-22 frame tables against the live hybpsf m7 catalogues |
| `results/t6_frames_2026-10-17.json` | t6 on the 2026-10-17 frame tables (pre-10-19 refit); same per-detector pattern. Its `sources` read `m8_dedup`: that run used the #362 `s13zp_hyb` snapshot, whose files are the 2026-10-19 hybpsf m7 merged catalogues under an `m8_dedup` name |

## Usage
    python t7_merge_vs_newframe.py /orange/adamginsburg/jwst/gc-treasury/catalogs <out_dir> 040 044 ...
    python t7_fig.py <figC_detectors.json> <out_dir>/t7.json <out.png>
    AREA375_WORKDIR=<workdir> python t6_merge_vs_frame.py <catalog_dir>
    AREA375_WORKDIR=<workdir> python t5_detector_offsets.py

`results/figC_detectors.json` is the output of the #362 fig C script (per-detector hybrid m7 minus VIRAC Ks,
both Vega); that script and its zero-point network inputs are not in this directory.

The scripts read the GC Treasury tree (`/orange/adamginsburg/jwst/gc-treasury`) directly; `t7_fig.py`'s
"68 tiles" refers to fig C, which used all 68 tiles.
