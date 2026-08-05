# QA diagnostics — how every plot and number is made

This document is the reference for the automated QA diagnostics posted on each observation's
tracking issue. Every diagnostic comment links back here (to the matching stage section) and to
the exact source function that built it, so no shorthand on a plot is left undefined.

- Source: [`data_qa/diagnostics.py`](../data_qa/diagnostics.py) builds every figure and metric;
  [`data_qa/post_diagnostics.py`](../data_qa/post_diagnostics.py) posts them.
- The diagnostics are **NIRCam-only**. MIRI issues carry only the pipeline-status table.
- Each stage writes a metrics JSON (`data_qa/metrics/<obsid>.json`) and posts/updates one
  marker-keyed comment per stage, so re-running **updates in place** — it never duplicates.

Jump to: [Glossary](#glossary) ·
[Stage 1](#stage1) · [Stage 2](#stage2) · [Stage 3](#stage3) · [Stage 4](#stage4) ·
[Stage 5](#stage5) · [Stage 6](#stage6) · [Stage 7](#stage7)

---

<a id="glossary"></a>
## Glossary

<a id="glossary-mtier"></a>
### Catalog tiers: MAST default, `m1` … `m8`, `m8_dedup`

The pipeline refines a field's catalog through numbered **merge/refinement stages**. Higher =
more processed; the QA always shows the **highest tier present on disk** for the observation
(`_catalog_priority` in the source). The ordering is:

> **MAST-delivered source catalog** (lowest) < `m1` < `m2` < … < `m7` < **`m8_dedup`** (highest).

- **MAST-delivered source catalog** — the STScI level-3 pipeline's own source list shipped with
  the mosaic (`*_cat.fits`). Single-band, aperture photometry, no cross-band merge. This is the
  "raw" baseline the pipeline improves on (see [stage 7](#stage7)).
- **`mN`** — the `N`th [jicama](#glossary-jicama)-pipeline pass. Successive passes add
  PSF-fit photometry, per-exposure combination, cross-band forced photometry, and quality
  vetting. In particular `m7` seeds each filter's forced photometry at the cross-band source
  positions, and **`m8_dedup`** adds forced cross-band fill plus de-duplication of near-neighbour
  merge artifacts — the deepest, cleanest product. Exact per-stage semantics live in the
  [jwst-gc-pipeline](https://github.com/keflavich/jwst-gc-pipeline) merge code; for QA the only
  thing that matters is the ordering above and that a **higher tier is a more processed catalog**.
- A catalog labelled **`crossmatch`** in a caption means no single merged catalog held both
  requested filters, so the CMD was built by cross-matching the two single-band catalogs. Its
  colour width is set by the positional match tolerance, not the catalog's colour precision.

<a id="glossary-jicama"></a>
### jicama

`jicama` is the name of our PSF-photometry catalog pipeline (as distinct from the MAST/STScI
source catalog, and from the `peppar` and `STARFINDER` catalogs used in method comparisons).
"jicama-produced catalog" = the `mN`/`m8_dedup` products above.

<a id="glossary-virac"></a>
### VIRAC / VIRAC2 / Ks

**VIRAC2** = the VISTA Variables in the Vía Láctea Infrared Astrometric Catalogue, version 2
(VizieR `II/387`). It is **tied to the Gaia DR3 frame at epoch 2014.0** and carries near-IR
**Ks**-band (~2.15 µm) magnitudes and per-star **proper motions (PM)**. QA uses it as the
external reference for both astrometry (frame tie, stages 4/5) and photometry (zeropoint,
stage 3). Ks (~2.15 µm) is the VIRAC band closest to JWST F212N/F210M, which is why those are
the SW filters compared against it.

Before any comparison, the VIRAC positions are **proper-motion-propagated** from 2014.0 to the
JWST observation epoch (`_obs_epoch`, read from the mosaic or a companion refcat), so a real
stellar motion over the ~8 yr baseline is not mistaken for a frame error.

<a id="glossary-crossmatch"></a>
### JWST ↔ VIRAC cross-match and the selection criteria

"The JWST catalog cross-matched to VIRAC" means: take the JWST catalog source positions (read
straight from the release/`mN` catalog — **never re-detected** by QA) and pair them with the
PM-propagated VIRAC reference. The **selection criteria** differ by what is being measured:

- **Photometric zeropoint** ([stage 3](#stage3)): anchor on the **sparse, Ks-bright VIRAC**
  stars and take each one's **nearest** JWST catalog source within **0.1″**. Anchoring on VIRAC
  (not the much deeper JWST catalog) prevents faint JWST sources from being paired with the wrong
  bright VIRAC star. A 3σ-clipped locus (iterated to convergence) then isolates the stellar ridge
  from the red mismatch cloud, and the scatter about that ridge is the zeropoint scatter.
- **Frame tie** ([stage 4](#stage4)): the offset is measured by an
  [x-correlation histogram peak](#glossary-xcorr) **per spatial cell**, not by nearest-neighbour
  pairing (see why under [bulk offset](#glossary-bulk)).
- **Inter-module / per-detector tie** ([stage 5](#stage5)): the per-detector residual uses
  VIRAC matches within **0.15″**; the reference-free A↔B tie uses no external catalog at all.

<a id="glossary-xcorr"></a>
### x-correlation histogram peak (`aa.xcorr`)

At Galactic-Centre stellar density, a plain nearest-neighbour match fabricates pairs: within
0.3″ there are hundreds of thousands of chance coincidences, and their median offset **collapses
toward zero the further the frame is actually displaced** (it reads ~1.8 mas even at a real 2″
shift). So QA instead builds the 2-D histogram of all JWST−reference separation vectors and takes
its **peak** — the displacement at which real matched pairs pile up. `peak_ratio` = peak height ÷
background density; a match is accepted only when `peak_ratio ≥ MIN_PEAK_RATIO` with enough
pairs. This is crowding-robust where a median is not.

<a id="glossary-bulk"></a>
### "bulk" offset and the per-cell tie

The **bulk offset** is the single field-wide shift between the JWST catalog and VIRAC — the
number you would apply to the whole frame to register it onto the Gaia/VIRAC frame. Because a
field can have an **internal discontinuity** (one sub-region tied differently — e.g. a stale
visit block), QA does not report a single scalar. Instead ([`_cell_offsets`](../data_qa/diagnostics.py)):

1. Split the JWST footprint into a 4×4 spatial grid.
2. In each cell, measure the JWST↔VIRAC offset by the [xcorr peak](#glossary-xcorr) against the
   local VIRAC reference (cell + 2″ margin). Cells with enough sources but **no clear peak** are
   recorded as *dropped* (shown hollow grey), not silently discarded.
3. The reported field tie is the **source-count-weighted median** of the per-cell offsets — the
   offset the *catalog as a whole* experiences, not the offset of whichever cell had the sharpest
   peak.

<a id="glossary-tie-uncertainty"></a>
### Tie offset and its uncertainty (what "σ" means in stage 4)

For the stage-4 tie, the "offset significance" is **offset ÷ (its uncertainty)**, where the
uncertainty is the **cell-to-cell standard error**:

- spread = `mad_std` of the per-cell offset vectors (how much the cells disagree),
- uncertainty (standard error) = spread ÷ √(number of measured cells),
- significance σ = tie offset ÷ standard error.

This is **not** the RMS of per-star offsets, and **not** a per-star propagated position error —
it is the uncertainty on the *field-average tie*, driven by how consistently the cells agree.
(The per-star **propagated** position σ = `hypot(VIRAC base position error, baseline × PM error)`
is a different quantity, used only in [stage 6](#stage6) for the astrometric-precision curve.)
A frame passes stage 4 only when the tie is small **and** the cells are spatially consistent (no
adjacency-confirmed sub-region off by >30 mas holding >2% of the sources).

<a id="glossary-reffree"></a>
### "reference-free" inter-module tie

**Reference-free** means the offset is measured by matching **JWST against itself** — module NRCA
directly against module NRCB — using **no external catalog** (no VIRAC, no Gaia). It isolates an
internal instrument tie from any error in the external reference. See [stage 5](#stage5).

<a id="glossary-quiver"></a>
### The per-detector quiver, and how NRCB detectors get a vector

The stage-5 quiver has **one arrow per detector** (up to 8 SW detectors: `nrca1–4`, `nrcb1–4`).
Each arrow is that detector's **median position residual against VIRAC**, after the field-wide
bulk offset is removed — i.e. each detector is compared to the **external VIRAC frame**, *not* to
NRCA. That is why every detector, including e.g. `nrcb2`, gets a vector even though it never
physically overlaps NRCA on the sky: the common reference is VIRAC, which covers the whole field.
The arrow is placed at the detector's mean sky position; the number of matched stars behind each
arrow is annotated on the plot.

The separate **A↔B overlap** panel is the [reference-free](#glossary-reffree) measurement: during
the dither pattern the detectors sweep across the sky, so a star that lands on an NRCA detector in
one exposure can land on an NRCB detector in another. Those genuinely-shared stars (the
NRCA∩NRCB overlap set) tie the two modules to each other directly.

<a id="glossary-lf"></a>
### Luminosity function (LF) and its turnover

The **luminosity function** is the histogram of source counts versus magnitude. Its **turnover**
is the magnitude of the peak bin: counts rise toward fainter magnitudes, then fall once the
catalog stops being complete, so the turnover magnitude is a rough **depth** indicator (fainter
turnover ⇒ deeper catalog). In stage 2 the LF is drawn as a right-side marginal whose magnitude
axis is locked to the CMD.

<a id="glossary-snr"></a>
### S/N and the S/N > 10 cut

Where a plot is restricted to **S/N > 10**, "S/N" is the **flux measurement signal-to-noise** of
each star, `flux_fit / flux_err` from the per-exposure PSF fit. The high-S/N cut restricts an
offset plot to the best-measured stars, so a residual scatter reflects the astrometric tie rather
than photon-noise centroiding of faint sources.

---

<a id="stage1"></a>
## Stage 1 — first mosaics

**What it shows.** Grayscale SW and LW `i2d` mosaic thumbnails.
**How.** `stage1_mosaics` loads the merged `i2d` for the chosen SW/LW filters and renders a
ZScale/asinh grayscale. **Purpose:** confirm the observation was delivered and the mosaics are
present and not obviously corrupt. A nominal (proposed) filter with no mosaic on disk is listed
explicitly, never dropped silently.
**Source:** [`data_qa/diagnostics.py` → `stage1_mosaics`](../data_qa/diagnostics.py).

<a id="stage2"></a>
## Stage 2 — colour–magnitude diagram (CMD)

**What it shows.** A 2-D density (hexbin) colour–magnitude diagram — LW magnitude versus
(SW − LW) colour — from the highest-tier [catalog](#glossary-mtier) on disk, with the
[luminosity function](#glossary-lf) drawn as a right-side marginal on a shared magnitude axis.
**How.** `stage2_cmd` reads the [`mN`/`m8_dedup`](#glossary-mtier) catalog (or, if no single
catalog holds both bands, cross-matches the two single-band catalogs — labelled `crossmatch`).
It plots log-N density and marks the LF turnover magnitude.
**Numbers.** `n_stars` = finite (SW, LW) pairs; `lf_turnover` = magnitude of the LF peak bin
(depth proxy — see [LF](#glossary-lf)).
**Source:** [`data_qa/diagnostics.py` → `stage2_cmd`](../data_qa/diagnostics.py).

<a id="stage3"></a>
## Stage 3 — photometric calibration (zeropoint)

**What it shows.** A 2-D histogram (colour = number of stars) of JWST SW catalog magnitude versus
[VIRAC Ks](#glossary-virac) for the [cross-matched](#glossary-crossmatch) stars. The **cyan line
is the ideal 1:1 (unit-slope) relation — it is NOT a fit**; a well-calibrated catalog should lie
along it. The line is anchored on the densest stellar ridge (the mode of JWST−Ks).
**How.** `stage3_calibration` anchors on VIRAC (nearest JWST source within 0.1″), then makes a
3σ-clipped robust linear fit iterated to convergence to measure the slope and the scatter about
the locus. The **fitted slope** is reported in the title (and gated) but is *not* drawn, because
a free-slope line wanders with the red mismatch cloud and misreads as a bad fit.
**Numbers.** `slope` (gate 0.8–1.2), `scatter` mag about the locus (gate < 0.8), `zeropoint`,
`n_matched`, `n_locus`.
**Source:** [`data_qa/diagnostics.py` → `stage3_calibration`](../data_qa/diagnostics.py).

<a id="stage4"></a>
## Stage 4 — positional offsets (JWST ↔ VIRAC frame tie)

**What it shows.** How well the JWST catalog is registered onto the [VIRAC/Gaia
frame](#glossary-virac), measured **[per spatial cell](#glossary-bulk)** (not one scalar).
- **LEFT** — a spatial map of the 4×4 [per-cell tie](#glossary-bulk): filled squares are measured
  cells (colour = offset), hollow grey squares are cells with sources but no clear
  [xcorr peak](#glossary-xcorr), and a green outline marks cells that coherently deviate.
- **MIDDLE** — the same per-cell offsets as (ΔRA, ΔDec) points sized by source count, with the
  **source-weighted median tie** (black +), the 75 mas pass gate (dotted circle), and
  **marginal histograms of ΔRA and ΔDec along each axis**.
- **RIGHT** (when present) — the [reference-free](#glossary-reffree) NRCA-vs-NRCB inter-module
  offset.

**What "bulk" / "σ" mean here.** The reported **tie** is the [bulk offset](#glossary-bulk) — the
source-weighted median cell offset of the JWST catalog cross-matched to VIRAC. Its
**uncertainty** is the [cell-to-cell standard error](#glossary-tie-uncertainty) (spread ÷ √cells),
and "Nσ" = tie ÷ that standard error. It is **not** a per-star RMS or a per-star propagated error.
**Pass** requires a small tie **and** spatially consistent cells **and** no inter-module offset.
**Source:** [`data_qa/diagnostics.py` → `stage4_offsets`](../data_qa/diagnostics.py)
(cells: `_cell_offsets`; aggregation/uncertainty: `_cell_consistency`).

<a id="stage5"></a>
## Stage 5 — inter-detector / inter-module tie

**What it shows.**
- **TOP-LEFT** — the [per-detector residual quiver](#glossary-quiver): one arrow per SW detector,
  each the detector's median residual **against VIRAC** with the field bulk offset removed, placed
  at the detector's mean sky position and annotated with the **number of matched stars**. (This is
  why an NRCB detector with no NRCA sky overlap still gets a vector — the common reference is
  VIRAC, not NRCA. See [the quiver note](#glossary-quiver).)
- **TOP-RIGHT** — the [reference-free](#glossary-reffree) NRCA∩NRCB overlap tie: the offset and
  RMS of the **same stars** seen in both modules (matched JWST-to-JWST, no external catalog), with
  **marginal histograms** of the residual ΔRA/ΔDec.
- **BOTTOM STRIP** — a cutout gallery of overlap stars from the SW merged `i2d`. A good tie shows
  one round PSF; a mis-tie doubles or elongates the star (the same source drizzled twice at offset
  positions).

**S/N > 10 variant.** Because the all-stars overlap panel is dominated by faint, noisily-centroided
sources, stage 5 also produces the same A↔B overlap restricted to [S/N > 10](#glossary-snr) stars,
so the residual scatter reflects the tie rather than photon noise.
**Source:** [`data_qa/diagnostics.py` → `stage5_intermodule`](../data_qa/diagnostics.py)
(per-detector: `_per_detector_offsets`; modules: `_module_positions`).

<a id="stage6"></a>
## Stage 6 — astrometric precision

**What it shows.** Per-star position error σ_pos (mas) versus Vega magnitude, one curve per
channel, from the per-exposure PSF fits. The bright-end floor is the astrometric systematic
limit; the faint-end rise tracks S/N. Shaded band = 16–84th percentile. Here σ_pos is the
**per-star propagated** position error (see [tie uncertainty](#glossary-tie-uncertainty)), the
formal PSF-fit `dra`/`ddec`.
**Source:** [`data_qa/diagnostics.py` → `stage6_astrom_error`](../data_qa/diagnostics.py).

<a id="stage7"></a>
## Stage 7 — MAST vs pipeline (improvement over the delivered products)

**What it shows.** The gain of the pipeline over the raw **MAST-delivered** products, side by
side:
- **i2d before/after** — the STScI/MAST merged `i2d` mosaic next to our pipeline mosaic (same
  filter, same stretch).
- **catalog depth** — brightness histograms of the [MAST source catalog](#glossary-mtier) versus
  the [jicama](#glossary-jicama) catalog (jicama recovers more and fainter stars; shown briefly).
- **astrometry (main)** — the per-star offset against [VIRAC](#glossary-virac) for the MAST
  catalog versus the jicama catalog, showing the frame-tie tightening.

**Source:** [`data_qa/diagnostics.py` → `stage7_mast_vs_pipeline`](../data_qa/diagnostics.py).
