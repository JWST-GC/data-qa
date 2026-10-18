# #375: pixel area (AREA map) against flux accuracy, GC Treasury 10678

Scope: F212N (8 SW detectors) and F480M (2 LW detectors), all 68 tiles. Every number below comes from the
scripts in this directory (commands listed at the end). Magnitude systems: per-frame tables carry AB magnitudes
derived from MJy/sr via PIXAR_SR (tests 2 and 3; in test 2 only frame-to-frame differences enter, so the
constant cancels); VIRAC2 and GNS Ks are Vega; the stage-13 network works on m8 Vega magnitudes.

Sign convention: m_frame = m_true + k * term. With a = 2.5 log10 AREA, k = 1 means the area factor is
entirely missing (the correction is m - 2.5 log10 AREA, i.e. flux x AREA), and k = 0 means it is already
handled. sT is the STPSF grid-sum term, 2.5 log10(sum of ePSF), under the plane pairing that the pre-#1154
loader used, so pT = a + sT is the error expected for a PSF fit with the pre-fix grids.

## Headline

1. **Provenance.** The cal/crf SCI arrays are surface brightness: the large-scale flat follows AREA
   (log flat = beta log AREA with beta of 0.45 to 1.54 on the SW detectors and 0.97 on nrcalong; nrcblong
   -0.37). Point-source flux on crf therefore requires x AREA. The STPSF grids (2.2.0, NORMALIZ=first,
   DISTORT=True) carry the factor themselves: the grid sum follows AREA^beta with beta = -0.95 to -0.99 on all
   10 detectors (about 1 mmag residual), so a PSF fit with the correct plane pairing returns flux with the area
   term included. Before jwst-gc-pipeline #1154 (3393a5f5, 2026-10-09) the planes were loaded transposed, so
   the fitted flux carries pT = a(x,y) + sT, which is approximately a(x,y) - a(y,x). Every non-hybrid
   resbgsub_m7 per-frame catalogue (F212N 3232/3232, F480M 808/808) predates the fix and m8 is built from
   them; every hybpsf m7 catalogue postdates it.
2. **External check (test 3, VIRAC2 / GNS).** Pooled k on a: STPSF m7 F212N +1.11 ± 0.07 / +0.91 ± 0.09;
   hybrid m7 F212N +0.12 ± 0.07 / -0.05 ± 0.08; aperture on crf F212N +1.02 ± 0.08 / +0.88 ± 0.08. F480M: STPSF
   +0.67 ± 0.07 / +0.26 ± 0.20, hybrid -0.24 ± 0.06 / -0.61 ± 0.20, aperture +0.58 to +0.66 (VIRAC2). For the
   STPSF m7 photometry the per-detector k on a spreads from +0.06 to +2.26 in F212N; the pT model reduces the
   spread (+0.60 to +1.44 on seven detectors) and gives pooled k = +1.22 ± 0.04 (VIRAC2), +1.15 ± 0.06 (GNS).
   This is the pattern the provenance predicts: the pre-fix STPSF catalogues carry the transposed-grid error,
   the post-fix hybrid catalogues carry almost no area term in F212N, and aperture photometry on crf needs
   the full x AREA.
3. **Internal check (test 2, frame minus tile consensus).** Within a tile the same star moves only by the
   dither offsets, so the within-star differencing keeps 31-34 per cent (SW) and 12-13 per cent (LW) of the
   area-term variance. The binned 8x8 residual maps hold 1.5-5.8 mmag of structure (shuffled null 0.2-1.0
   mmag), and the area terms describe only part of it: cross-validated over tile halves, the fitted k lowers
   the binned RMS on some detectors (F212N nrcb4 5.8 -> 4.3 mmag with pT; F480M aperture r1 2.7/3.4 -> 1.4/1.3
   mmag with a) and raises it on others (F212N nrca1 2.4 -> 4.8 mmag). Per-detector internal k values range
   from -1.5 to +2.4 and change sign between detectors, so I do not treat them as estimates of the area
   coefficient. The F480M aperture k drops from +2.1 (r1) to +1.2 (r2), which indicates that encircled-energy
   variation across the detector adds to the area term for small apertures.
4. **Network (test 4).** With the physical correction for the current m8 (-(a+sT), k = 1): F212N tile RMS
   0.142 -> 0.086 mag, Gx -0.0249 -> -0.0168 mag/arcmin, E-W step -0.052 -> -0.030 mag, NRCA-NRCB +0.035 ->
   +0.018 mag; F480M tile RMS 0.081 -> 0.057, Gx -0.0147 -> +0.0108, step -0.028 -> +0.026, NRCA-NRCB
   +0.043 -> -0.010. With the pooled VIRAC2 k (F212N 1.22, F480M 0.80): F212N 0.073 mag, Gx -0.0150, step
   -0.025, NRCA-NRCB +0.014; F480M 0.033 mag, Gx +0.0058, step +0.015, NRCA-NRCB +0.000. The tile+G RMS
   stays at 0.019-0.039 mag in every variant because G absorbs the linear gradient. The area/grid term
   accounts for roughly 40 per cent of the F212N E-W gradient; about -0.015 mag/arcmin remains and needs
   another explanation. In F480M a correction with k of 0.8-1 reverses the sign of the gradient, so the F480M
   gradient is smaller than the area term predicts.
5. **Recommendation.** Fix it in the PSF chain: rebuild m8 from post-#1154 m7 catalogues (the hybrid m7
   already postdates the fix, or re-run the STPSF m7 with the fixed loader). Do not apply x AREA as a
   catalogue patch to post-fix PSF photometry, because the corrected STPSF grid already carries 1/AREA. If an
   interim patch of the current m8 is needed, -k(a+sT) with the pooled VIRAC2 k performed best in the network
   test, with the caveats below. Aperture photometry measured on crf needs x AREA; the pipeline aperture
   photometry on the i2d (constant pixel area on a TAN grid) does not.

## Test 1: provenance

| product | where pixel solid angle enters | consequence for point sources |
|---|---|---|
| cal / crf (jwst photom) | scalar PHOTMJSR -> MJy/sr; AREA extension and PIXAR_SR attached only (photom.py save_area_info, calc_nircam) | SCI is surface brightness once the flat has removed the area pattern; flux = sum(SCI) x PIXAR_SR x AREA |
| CRDS flat | pixel flat normalised by a 2nd-order polynomial, then delta L-flats from on-sky data (1063, Cycle 1) | large-scale flat follows AREA (beta below), so the area pattern is divided out of SCI |
| i2d (stcal drizzle, in_units=cps, iscale=1) | pixmap from the full distortion; surface brightness conserved | area handled geometrically; i2d aperture photometry with a constant pixel area expects k = 0 |
| PSF chain (merge_catalogs.py L2246) | flux_jy = flux_fit x proj_plane_pixel_area, one value per frame | per-pixel area enters only through the PSF model normalisation |
| STPSF grids (gc-treasury/psfs, 2.2.0, NORMALIZ=first, DISTORT=True) | grid sum proportional to AREA^-0.95..-0.99 | post-#1154 fit includes the area term (k = 0 expected); pre-#1154 transposed planes give error pT = a + sT |
| pipeline aperture_photometry.py | i2d, constant pixel area | correct for a TAN grid |
| m8 | built from non-hybrid resbgsub_m7 (all pre-#1154) | carries pT |

Note: astrometry_gdc/GDC_EXPERIMENT_REPORT.md section 12 describes the brick grids as unit-normalised, which
does not match the treasury grids measured here.

## Figures

- `fig1_maps_f212n_psf.png`, `fig1_maps_f212n_hyb.png`, `fig1_maps_f212n_ap1.png`, `fig1_maps_f480m_psf.png`,
  `fig1_maps_f480m_hyb.png`, `fig1_maps_f480m_ap1.png`: per detector, 16x16 binned maps of the raw a, the raw
  pT, the observed within-star Δm, the k = 1 predictions for a and pT after the same differencing, and the
  residual after the fitted k on pT (mmag).
- `fig2_k_per_detector.png`: k per detector and filter with tile-bootstrap errors; top internal (test 2),
  bottom external against VIRAC2 (filled) and GNS (open).
- `fig3_network.png`: network quantities for all variants (tile RMS, tile+G RMS, Gx, VIRAC2-anchor Gx, E-W
  step, NRCA-NRCB).

## Tables
### Test 1: large-scale flat and STPSF grid sum against AREA (log-log slope beta)

| filter:detector | flat beta (log flat = beta log AREA) | grid sum beta, post-#1154 pairing | resid (mmag) | grid sum beta, pre-#1154 (transposed) | resid (mmag) |
|---|---|---|---|---|---|
| F212N:nrca1 | +0.96 ± 0.03 | -0.978 ± 0.017 | 0.9 | +0.12 ± 0.26 | 13.3 |
| F212N:nrca2 | +1.37 ± 0.05 | -0.970 ± 0.019 | 0.6 | +0.15 ± 0.26 | 8.2 |
| F212N:nrca3 | +0.45 ± 0.03 | -0.988 ± 0.017 | 0.9 | +0.70 ± 0.19 | 10.3 |
| F212N:nrca4 | +0.88 ± 0.05 | -0.971 ± 0.022 | 0.8 | +0.89 ± 0.11 | 4.0 |
| F212N:nrcb1 | +0.87 ± 0.03 | -0.961 ± 0.017 | 0.6 | -0.17 ± 0.25 | 8.3 |
| F212N:nrcb2 | +0.68 ± 0.04 | -0.984 ± 0.019 | 1.0 | -0.10 ± 0.26 | 13.2 |
| F212N:nrcb3 | +1.54 ± 0.03 | -0.962 ± 0.023 | 0.9 | -0.88 ± 0.11 | 4.1 |
| F212N:nrcb4 | +0.94 ± 0.04 | -0.977 ± 0.018 | 1.0 | -0.69 ± 0.19 | 10.2 |
| F480M:nrcalong | +0.97 ± 0.07 | -0.947 ± 0.014 | 1.1 | +0.44 ± 0.22 | 18.6 |
| F480M:nrcblong | -0.37 ± 0.04 | -0.947 ± 0.014 | 1.2 | -0.53 ± 0.21 | 17.9 |

### Test 2: internal k (frame minus tile consensus), pooled over detectors; bootstrap over tiles

| filter | photometry | N rows | stars | sigma (mmag) | k on a | k on a+sT |
|---|---|---|---|---|---|---|
| F212N | PSF (STPSF grid, m7) | 1532728 | 382461 | 11.8 | +0.45 ± 0.04 | +1.51 ± 0.01 |
| F212N | PSF (hybrid grid, m7) | 1425628 | 361906 | 11.8 | -0.14 ± 0.03 | +0.34 ± 0.00 |
| F212N | aperture r1 on crf | 360774 | 110516 | 12.1 | +0.64 ± 0.03 | +0.23 ± 0.01 |
| F212N | aperture r2 on crf | 351004 | 110154 | 14.9 | +0.44 ± 0.03 | +0.27 ± 0.01 |
| F480M | PSF (STPSF grid, m7) | 546790 | 147848 | 6.3 | +0.47 ± 0.02 | +1.16 ± 0.02 |
| F480M | PSF (hybrid grid, m7) | 495797 | 136094 | 7.3 | -0.00 ± 0.02 | +0.17 ± 0.02 |
| F480M | aperture r1 on crf | 122020 | 37476 | 6.1 | +2.14 ± 0.03 | +0.70 ± 0.03 |
| F480M | aperture r2 on crf | 121311 | 37458 | 6.4 | +1.23 ± 0.02 | +0.46 ± 0.02 |

### Test 2: per-detector k and cross-validated 8x8 binned residual RMS (mmag; mean of A->B and B->A)

| filter | photometry | detector | k on a | k on a+sT | RMS before (null) | k=1 on a | CV k on a | CV k on a+sT |
|---|---|---|---|---|---|---|---|---|
| F212N | psf | nrca1 | +2.23 ± 0.03 | +1.44 ± 0.01 | 2.43 (0.33) | 3.22 | 5.34 | 4.84 |
| F212N | psf | nrca2 | +1.41 ± 0.04 | +0.91 ± 0.02 | 2.55 (0.35) | 3.55 | 4.05 | 3.72 |
| F212N | psf | nrca3 | +2.37 ± 0.03 | +1.46 ± 0.01 | 5.59 (0.95) | 5.18 | 5.93 | 4.79 |
| F212N | psf | nrca4 | +2.35 ± 0.04 | +1.27 ± 0.01 | 3.24 (0.37) | 2.31 | 3.05 | 3.07 |
| F212N | psf | nrcb1 | +0.64 ± 0.05 | +1.00 ± 0.02 | 4.48 (0.38) | 4.15 | 4.21 | 4.62 |
| F212N | psf | nrcb2 | -1.53 ± 0.04 | +1.31 ± 0.01 | 3.72 (0.45) | 4.84 | 4.17 | 4.94 |
| F212N | psf | nrcb3 | -0.35 ± 0.04 | +1.35 ± 0.03 | 2.89 (0.33) | 4.24 | 2.57 | 2.89 |
| F212N | psf | nrcb4 | -1.31 ± 0.03 | +2.23 ± 0.01 | 5.77 (0.83) | 6.89 | 5.84 | 4.31 |
| F212N | hyb | nrca1 | +0.46 ± 0.03 | +0.56 ± 0.02 | 2.66 (0.34) | 2.46 | 2.31 | 2.41 |
| F212N | hyb | nrca2 | +1.10 ± 0.04 | +0.62 ± 0.03 | 1.76 (0.33) | 2.24 | 2.34 | 2.02 |
| F212N | hyb | nrca3 | -0.19 ± 0.03 | +0.09 ± 0.01 | 3.48 (0.34) | 4.53 | 3.41 | 3.52 |
| F212N | hyb | nrca4 | +0.88 ± 0.03 | +0.30 ± 0.02 | 2.27 (0.29) | 2.76 | 2.64 | 2.37 |
| F212N | hyb | nrcb1 | -0.09 ± 0.04 | +0.35 ± 0.01 | 3.60 (0.33) | 3.61 | 3.63 | 3.72 |
| F212N | hyb | nrcb2 | -0.76 ± 0.03 | +0.55 ± 0.01 | 2.50 (0.34) | 3.60 | 2.66 | 3.04 |
| F212N | hyb | nrcb3 | -0.36 ± 0.03 | +1.76 ± 0.02 | 2.90 (0.33) | 3.82 | 2.80 | 2.58 |
| F212N | hyb | nrcb4 | -0.60 ± 0.03 | +1.28 ± 0.01 | 4.22 (0.40) | 4.93 | 4.49 | 3.37 |
| F212N | ap1 | nrca1 | +1.15 ± 0.03 | +0.25 ± 0.02 | 2.28 (0.70) | 2.90 | 3.14 | 2.26 |
| F212N | ap1 | nrca2 | +1.43 ± 0.06 | +0.16 ± 0.05 | 2.11 (0.83) | 2.60 | 3.13 | 2.13 |
| F212N | ap1 | nrca3 | +0.72 ± 0.03 | +0.09 ± 0.02 | 2.98 (0.53) | 3.76 | 3.46 | 3.01 |
| F212N | ap1 | nrca4 | +1.14 ± 0.04 | +0.04 ± 0.03 | 1.97 (0.60) | 2.67 | 2.83 | 1.99 |
| F212N | ap1 | nrcb1 | +1.00 ± 0.05 | +0.27 ± 0.03 | 4.22 (0.85) | 4.07 | 4.07 | 4.27 |
| F212N | ap1 | nrcb2 | +0.06 ± 0.05 | +0.28 ± 0.02 | 3.10 (0.88) | 4.21 | 3.14 | 3.40 |
| F212N | ap1 | nrcb3 | +0.51 ± 0.04 | +1.49 ± 0.04 | 3.00 (0.74) | 3.70 | 3.28 | 2.69 |
| F212N | ap1 | nrcb4 | +0.50 ± 0.05 | +0.95 ± 0.03 | 4.15 (1.37) | 4.98 | 4.30 | 3.21 |
| F480M | psf | nrcalong | +0.61 ± 0.03 | +0.65 ± 0.04 | 1.51 (0.20) | 2.24 | 1.86 | 1.51 |
| F480M | psf | nrcblong | +0.28 ± 0.03 | +1.48 ± 0.02 | 1.85 (0.19) | 1.11 | 1.55 | 1.28 |
| F480M | hyb | nrcalong | -0.23 ± 0.03 | -0.42 ± 0.04 | 2.24 (0.18) | 3.19 | 2.10 | 2.20 |
| F480M | hyb | nrcblong | +0.29 ± 0.02 | +0.53 ± 0.03 | 1.62 (0.18) | 1.91 | 1.58 | 1.58 |
| F480M | ap1 | nrcalong | +2.12 ± 0.03 | +1.08 ± 0.07 | 2.74 (0.35) | 1.63 | 1.36 | 2.35 |
| F480M | ap1 | nrcblong | +2.17 ± 0.02 | +0.42 ± 0.05 | 3.41 (0.49) | 2.20 | 1.28 | 3.24 |

### Test 3: external k (m_frame AB minus Ks Vega, colour terms fitted), pooled; bootstrap over tiles

| filter | reference | photometry | N | k on a | k on a+sT | cKs | c(H-Ks) | sigma (mmag) |
|---|---|---|---|---|---|---|---|---|
| F212N | VIRAC | PSF (STPSF grid, m7) | 243276 | +1.11 ± 0.07 | +1.22 ± 0.04 | -0.038 | +0.026 | 152 |
| F212N | VIRAC | PSF (hybrid grid, m7) | 236528 | +0.12 ± 0.07 | +0.24 ± 0.04 | -0.036 | +0.027 | 151 |
| F212N | VIRAC | aperture r1 on crf | 110994 | +1.02 ± 0.08 | +0.63 ± 0.04 | -0.040 | +0.027 | 149 |
| F212N | VIRAC | aperture r2 on crf | 110869 | +1.02 ± 0.08 | +0.64 ± 0.04 | -0.046 | +0.027 | 142 |
| F212N | GNS | PSF (STPSF grid, m7) | 348425 | +0.91 ± 0.09 | +1.15 ± 0.06 | -0.000 | +0.046 | 81 |
| F212N | GNS | PSF (hybrid grid, m7) | 336384 | -0.05 ± 0.08 | +0.18 ± 0.06 | -0.000 | +0.046 | 80 |
| F212N | GNS | aperture r1 on crf | 159281 | +0.88 ± 0.08 | +0.63 ± 0.05 | -0.001 | +0.047 | 76 |
| F212N | GNS | aperture r2 on crf | 159294 | +0.91 ± 0.09 | +0.65 ± 0.06 | -0.012 | +0.050 | 75 |
| F480M | VIRAC | PSF (STPSF grid, m7) | 301431 | +0.67 ± 0.07 | +0.80 ± 0.05 | -0.187 | -0.675 | 141 |
| F480M | VIRAC | PSF (hybrid grid, m7) | 288764 | -0.24 ± 0.06 | -0.10 ± 0.05 | -0.189 | -0.672 | 140 |
| F480M | VIRAC | aperture r1 on crf | 149802 | +0.66 ± 0.07 | +0.40 ± 0.05 | -0.188 | -0.672 | 139 |
| F480M | VIRAC | aperture r2 on crf | 149700 | +0.58 ± 0.06 | +0.32 ± 0.05 | -0.194 | -0.664 | 137 |
| F480M | GNS | PSF (STPSF grid, m7) | 301619 | +0.26 ± 0.20 | +0.64 ± 0.16 | -0.110 | -0.607 | 112 |
| F480M | GNS | PSF (hybrid grid, m7) | 288392 | -0.61 ± 0.20 | -0.25 ± 0.15 | -0.113 | -0.603 | 111 |
| F480M | GNS | aperture r1 on crf | 149451 | +0.34 ± 0.20 | +0.27 ± 0.14 | -0.114 | -0.601 | 111 |
| F480M | GNS | aperture r2 on crf | 149494 | +0.26 ± 0.19 | +0.20 ± 0.14 | -0.123 | -0.593 | 116 |

### Test 3: external per-detector k vs VIRAC2 (k on a / k on a+sT)

| photometry | nrca1 | nrca2 | nrca3 | nrca4 | nrcb1 | nrcb2 | nrcb3 | nrcb4 |
|---|---|---|---|---|---|---|---|---|
| F212N psf | +1.40±0.14 / +1.39±0.08 | +0.45±0.23 / +0.70±0.13 | +2.26±0.13 / +1.44±0.07 | +2.10±0.17 / +1.18±0.09 | +1.16±0.21 / +0.60±0.17 | +0.68±0.16 / +1.01±0.11 | +0.07±0.18 / +0.99±0.39 | +0.06±0.13 / +1.22±0.15 |
| F212N hyb | +0.39±0.14 / +0.48±0.09 | -0.12±0.24 / +0.21±0.13 | +0.05±0.14 / +0.14±0.07 | +0.19±0.17 / +0.18±0.08 | +0.80±0.20 / +0.13±0.16 | -0.09±0.17 / +0.17±0.11 | +0.18±0.19 / +1.75±0.40 | -0.15±0.14 / +0.38±0.16 |
| F212N ap1 | +1.15±0.15 / +0.78±0.10 | +0.64±0.29 / +0.46±0.15 | +0.97±0.16 / +0.61±0.08 | +1.02±0.20 / +0.60±0.09 | +1.51±0.22 / +0.60±0.19 | +0.83±0.21 / +0.54±0.12 | +1.02±0.25 / +1.86±0.52 | +1.05±0.15 / +0.75±0.18 |
| F212N ap2 | +1.22±0.14 / +0.86±0.10 | +0.66±0.26 / +0.45±0.14 | +0.88±0.16 / +0.54±0.07 | +1.09±0.20 / +0.62±0.09 | +1.74±0.21 / +0.74±0.18 | +0.81±0.20 / +0.60±0.11 | +1.08±0.26 / +2.14±0.51 | +0.95±0.14 / +0.82±0.16 |

| photometry | nrcalong | nrcblong |
|---|---|---|
| F480M psf | +1.13±0.09 / +0.87±0.06 | +0.11±0.08 / +0.53±0.10 |
| F480M hyb | -0.20±0.08 / -0.04±0.05 | -0.29±0.08 / -0.31±0.09 |
| F480M ap1 | +0.74±0.09 / +0.45±0.05 | +0.53±0.08 / +0.12±0.10 |
| F480M ap2 | +0.62±0.09 / +0.38±0.05 | +0.52±0.08 / +0.04±0.09 |

### Test 4: stage-13 network on m8 (Vega) with per-star corrections

| variant | filter | tile RMS (mag) | tile+G RMS (mag) | Gx (mag/arcmin) | VIRAC anchor Gx | E-W step (mag) | NRCA-NRCB (mag) | tile+G chi2/dof |
|---|---|---|---|---|---|---|---|---|
| none | F212N | 0.142 | 0.024 | -0.0249 ± 0.0022 | -0.0256 | -0.052 ± 0.003 | +0.035 ± 0.001 | 146.5 |
| none | F480M | 0.081 | 0.019 | -0.0147 ± 0.0016 | -0.0148 | -0.028 ± 0.003 | +0.043 ± 0.002 | 43.4 |
| area1 | F212N | 0.090 | 0.024 | -0.0176 ± 0.0022 | -0.0182 | -0.033 ± 0.003 | +0.036 ± 0.001 | 134.2 |
| area1 | F480M | 0.049 | 0.022 | +0.0106 ± 0.0016 | +0.0106 | +0.017 ± 0.002 | +0.045 ± 0.001 | 42.4 |
| areak | F212N | 0.086 | 0.034 | -0.0210 ± 0.0022 | -0.0209 | -0.042 ± 0.002 | +0.007 ± 0.001 | 81.4 |
| areak | F480M | 0.024 | 0.021 | -0.0034 ± 0.0015 | -0.0034 | -0.011 ± 0.002 | +0.037 ± 0.001 | 34.7 |
| pT1 | F212N | 0.086 | 0.025 | -0.0168 ± 0.0021 | -0.0174 | -0.030 ± 0.002 | +0.018 ± 0.001 | 141.0 |
| pT1 | F480M | 0.057 | 0.021 | +0.0108 ± 0.0015 | +0.0110 | +0.026 ± 0.004 | -0.010 ± 0.001 | 44.8 |
| pTk | F212N | 0.093 | 0.039 | -0.0226 ± 0.0021 | -0.0231 | -0.034 ± 0.002 | -0.004 ± 0.001 | 87.3 |
| pTk | F480M | 0.075 | 0.026 | +0.0095 ± 0.0016 | +0.0097 | +0.016 ± 0.007 | -0.031 ± 0.002 | 65.7 |
| pTvp | F212N | 0.073 | 0.027 | -0.0150 ± 0.0021 | -0.0156 | -0.025 ± 0.002 | +0.014 ± 0.001 | 142.8 |
| pTvp | F480M | 0.033 | 0.020 | +0.0058 ± 0.0015 | +0.0059 | +0.015 ± 0.003 | +0.000 ± 0.001 | 38.8 |
| pTvk | F212N | 0.118 | 0.029 | -0.0230 ± 0.0021 | -0.0237 | -0.037 ± 0.002 | +0.023 ± 0.001 | 130.6 |
| pTvk | F480M | 0.034 | 0.021 | +0.0043 ± 0.0015 | +0.0045 | +0.013 ± 0.002 | +0.013 ± 0.001 | 35.3 |

Network variants: none = published m8; area1 = a with k = 1; areak = a with test-2 per-detector PSF k;
pT1 = a + sT with k = 1; pTk = a + sT with test-2 per-detector k; pTvp = a + sT with test-3 VIRAC2 pooled k
(F212N 1.22, F480M 0.80); pTvk = a + sT with test-3 VIRAC2 per-detector k. Each star receives
dm = -2.5 log10(mean_i 10^(0.4 k p_i)) over the crf frames that cover it; the VIRAC2 anchor goes through the
same loader. The reproduction (none) matches #368 (F212N tile+G Gx -0.0249 ± 0.0022, F480M -0.0147 ± 0.0016).

## Caveats

- External fits have 75-152 mmag scatter (crowding, blending, VIRAC2 and GNS depth); the F480M - Ks
  comparison needs a large colour term (c(H-Ks) about -0.6 to -0.7), so the F480M external k values are less
  secure than the F212N ones. GNS errors are 2-3 times the VIRAC2 errors.
- The internal test only sees the area term along the dither offsets (31-34 per cent SW, 12-13 per cent LW
  of its variance) and is dominated by other position-dependent terms (PSF-model mismatch, flat residuals,
  encircled-energy variation for apertures). The test-2 per-detector k values used in areak/pTk therefore
  mix those terms in; the improvement in tile+G chi2/dof for areak (F212N 146 -> 81) comes with a higher
  tile+G RMS (0.024 -> 0.034).
- The flat-vs-AREA slope also absorbs real QE structure; the plane variant of that fit is degenerate and is
  not used.
- The grid-sum term is interpolated bilinearly between the 4x4 nodes (clamped at the outer nodes) and is
  limited to the grid field of view (fovp101, samp2).
- The hybrid grids differ from STPSF in the core, so the hybrid k values combine the area handling with any
  hybrid normalisation; F480M hybrid shows k = -0.24 ± 0.06 against VIRAC2, which merits a check.
- m8 merges several frames per star, so a per-star mean correction is an approximation of a rebuild from
  corrected m7.
- The remaining F212N gradient (about -0.015 mag/arcmin after correction) is not explained by the area or
  grid terms.

## Commands (as originally run, in the work directory; python = /blue/adamginsburg/adamginsburg/miniconda3/envs/python313/bin/python)

With the copies in this repository, run them as in ../README.md (AREA375_WORKDIR selects the work directory; test2.py/test3.py are now t2_repeat_stars.py/t3_external_ref.py).

```
python t1_flat_vs_area.py                      # -> t1_flat_vs_area.json
python t1_psfgrid_sum.py                       # -> t1_psfgrid_sum.json, t1_psfgrid_nodes.npz
sbatch build.sbatch                            # array 1-68 over obslist.txt: python build_frames.py <obs> -> frames/<filt>_o<obs>.npz (job 46479489)
sbatch test2.sbatch                            # python -u test2.py -> test2.json, test2_maps.npz (job 46481491; NBOOT=1000)
sbatch test3.sbatch                            # python -u test3.py -> test3.json (job 46481492)
for v in none area1 pT1 areak pTk pTvp pTvk; do sbatch --export=ALL,AREA_VARIANT=$v net.sbatch; done
                                               # python test4_network.py -> net_<v>/summary.json
python tables.py >| tables.md                  # tables in this report
python figs.py                                 # fig1_maps_*.png, fig2_k_per_detector.png, fig3_network.png
```
All SLURM jobs used --qos=astronomy-dept-b --account=astronomy-dept with job names
jwst-gctreasury-10678-area-<step>. Shared helpers are in common.py.
