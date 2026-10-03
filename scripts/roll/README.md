# Field roll (rotation) measurement scripts

Research scripts behind the per-field roll corrections discussed in JWST-GC/data-qa#346 and
tabulated in `jwst_gc_pipeline/reduction/roll_corrections.csv` (keflavich/jwst-gc-pipeline).
They are not part of the `data_qa` package or the automated issue refresh.

- `measure_rotation.py PROG OBS [FILT]`: fits a similarity transform (rotation, scale, shift) of
  the per-exposure daophot catalogues of one observation onto VIRAC2 and Gaia DR3 (swept-window
  bulk tie, mutual nearest neighbours, de-blend, flux-vet, iterative 3-sigma clip, bootstrap over
  unique reference stars). Writes `results/rot_*.json`.
- `arches_joint2.py`: arches (2045 o001) F212N m7 NRCA+NRCB fit with a single-shift model and a
  joint-module model (common rotation and scale, free per-module shift), with closure checks at
  pre-rotations of 0, 8.3 and 13.0 arcsec. Evidence for the 8.3 +- 2.0 arcsec row
  (data-qa#346 comment 5972261192, jwst-gc-pipeline#1047).

theta > 0 rotates JWST positions from North toward East to land on the reference.

`arches_joint2.py` reads the m7 catalogues from `ARCHES_CATDIR` (default
`/orange/adamginsburg/jwst/arches/catalogs`).

Reference-catalogue caches (`refcache/`) and outputs go next to the scripts by default; set
`ROLL_WORKDIR` to put them elsewhere. Queries go to VizieR (II/387/virac2, I/355/gaiadr3) when no
cache exists.
