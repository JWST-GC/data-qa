# QA diagnostics — how every plot and number is made

This document is the reference for the automated QA diagnostics posted on each observation's
tracking issue. Every diagnostic comment links back here (to the matching stage section) and to
the exact source function that built it, so no shorthand on a plot is left undefined.

- Source: [`data_qa/diagnostics.py`](../data_qa/diagnostics.py) builds every figure and metric;
  [`data_qa/post_diagnostics.py`](../data_qa/post_diagnostics.py) posts them.
- The diagnostics are **NIRCam-only**. MIRI issues carry only the pipeline-status table.
- Each stage writes a metrics JSON (`data_qa/metrics/<obsid>.json`) and posts/updates one
  marker-keyed comment per stage, so re-running **updates in place** — it never duplicates.
- **None of these stages is an automated gate.** The `passed` flag is a suggestion for the human
  reviewer, not an enforced pass/fail; each stage below says what "consider it passing" means.

Jump to: [Glossary](#glossary) ·
[Stage 1](#stage1) · [Stage 2](#stage2) · [Stage 3](#stage3) · [Stage 4](#stage4) ·
[Stage 5](#stage5) · [Stage 6](#stage6) · [Stage 7](#stage7)

---

<a id="glossary"></a>
## Glossary

<a id="glossary-mtier"></a>
### Catalog tiers: MAST default, `m1` … `m8`

The jicama pipeline refines a field's catalog through numbered merge/refinement stages. Higher =
more processed; the QA always shows the **highest tier present on disk** for the observation
(`_catalog_priority` in the source). The ordering is:

> **MAST-delivered source catalog** (lowest) < `m1` < `m2` < … < `m7` < `m8` (highest).

- **MAST-delivered source catalog** — the STScI level-3 pipeline's own source list shipped with
  the mosaic (`*_cat.fits`). Single-band, aperture photometry, no cross-band merge. This is the
  "raw" baseline the pipeline improves on (see [stage 7](#stage7)). It is used by the catalog
  stages **only as a last resort**, when no jicama `mN` catalog exists yet for the field.
- **`mN`** — the `N`th [jicama](#glossary-jicama)-pipeline pass. Successive passes add
  PSF-fit photometry, per-exposure combination, cross-band forced photometry, and quality
  vetting; `m7` seeds each filter's forced photometry at the cross-band source positions.
- **`m8` / `m8_dedup`** — the deepest tier. `m8_dedup` is an `m8` catalog with the duplicate rows
  that the cross-band merge can produce removed ("dedup" = duplicate-removed). `_catalog_priority`
  gives plain `m8` and `m8_dedup` the **same** top tier, breaking the tie by most-recent-on-disk,
  so whichever is newer wins. Exact per-stage semantics live in the
  [jwst-gc-pipeline](https://github.com/keflavich/jwst-gc-pipeline) merge code; for QA the only
  thing that matters is the ordering above and that a **higher tier is a more processed catalog**.
- A catalog labelled **`crossmatch`** in a caption means no single merged catalog held both
  requested filters, so the CMD was built by cross-matching the two single-band catalogs. Its
  colour width is set by the positional match tolerance, not the catalog's colour precision.

<a id="glossary-jicama"></a>
### jicama

`jicama` is the name of our PSF-photometry catalog pipeline (as distinct from the MAST/STScI
source catalog, and from the `peppar` and `STARFINDER` catalogs used in method comparisons).
"jicama catalog" = the `mN` products above.

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
straight from the jicama `mN` catalog — **never re-detected** by QA) and pair them with the
PM-propagated VIRAC reference. The **selection criteria** differ by what is being measured:

- **Photometric zeropoint** ([stage 3](#stage3)): anchor on the sparse, Ks-bright VIRAC stars and
  take each one's nearest JWST catalog source within **0.1″**. Anchoring on VIRAC (not the much
  deeper JWST catalog) prevents faint JWST sources from being paired with the wrong bright VIRAC
  star. A sigma-clipped locus (below) then isolates the stellar ridge from the red mismatch cloud.
- **Frame tie** ([stage 4](#stage4)): the offset is measured by an
  [x-correlation histogram peak](#glossary-xcorr) **per spatial cell**, not by nearest-neighbour
  pairing (see why under [bulk offset](#glossary-bulk)).
- **Inter-module / per-detector tie** ([stage 5](#stage5)): the per-detector residual uses
  VIRAC matches within **0.15″**; the reference-free A↔B tie uses no external catalog at all.

<a id="glossary-xcorr"></a>
### x-correlation histogram peak (`aa.xcorr`)

At Galactic-Centre stellar density a plain nearest-neighbour match fabricates pairs: within 0.3″
of a JWST source there are tens of thousands of chance VIRAC coincidences (≈32,000 measured for a
brick F212N field), and their median offset **collapses toward zero the further the frame is
actually displaced** — it reads ~1–2 mas even at a real 2″ shift. So QA instead builds the 2-D
histogram of all JWST−reference separation vectors and takes its **peak**: the displacement at
which real matched pairs pile up. `peak_ratio` = peak height ÷ background density; a match is
accepted only when `peak_ratio ≥ MIN_PEAK_RATIO` with enough pairs. This is crowding-robust where
a median is not. (When both catalogs are dense — the JWST-NRCA↔JWST-NRCB case in [stage 5](#stage5)
— the chance-pair count is far higher, ~400,000, which is why that tie also uses the peak.)

<a id="glossary-bulk"></a>
### "bulk" offset and the per-cell tie

The **bulk offset** is the single field-wide shift between the JWST catalog and VIRAC — the number
you would apply to the whole frame to register it onto the Gaia/VIRAC frame. Because a field can
have an **internal discontinuity** (one sub-region tied differently — e.g. a stale visit block),
QA does not report a single scalar. Instead ([`_cell_offsets`](../data_qa/diagnostics.py)):

1. Split the JWST footprint into a **4×4 grid (16 cells)**.
2. In each cell, measure the JWST↔VIRAC offset by the [xcorr peak](#glossary-xcorr) against the
   local VIRAC reference (that cell + a 2″ margin). Cells with enough sources but **no clear peak**
   are recorded as *dropped* (shown hollow grey).
3. The reported field offset is the **source-count-weighted median of the 16 cell offsets** — the
   offset the catalog as a whole experiences, not the offset of whichever cell had the sharpest
   peak.

This is a QA **cross-check** of a tie the jicama pipeline has already applied per exposure; it
measures the residual whole-catalog offset that survives in the delivered catalog, not the
pipeline's internal per-exposure solution.

<a id="glossary-tie-uncertainty"></a>
### The stage-4 tie offset and its uncertainty (what "σ" means)

For the stage-4 tie, the "offset significance" is **offset ÷ (its uncertainty)**, where the
uncertainty is the **cell-to-cell standard error**:

- spread = `mad_std` of the 16 per-cell offset vectors (how much the cells disagree),
- uncertainty (standard error) = spread ÷ √(number of measured cells),
- significance σ = tie offset ÷ standard error.

This is the uncertainty on the *field-average tie*, driven by how consistently the cells agree —
distinct from the per-star scatter, which [stage 6](#stage6) shows directly as rms(offset).

<a id="glossary-reffree"></a>
### "reference-free" inter-module tie

**Reference-free** means the offset is measured by matching **JWST against itself** — module NRCA
directly against module NRCB — using **no external catalog** (no VIRAC, no Gaia). It isolates an
internal instrument tie from any error in the external reference. See [stage 5](#stage5).

<a id="glossary-quiver"></a>
### The per-detector quiver, and how NRCB detectors get a vector

The stage-5 quiver has **one arrow per detector** (up to 8 SW detectors: `nrca1–4`, `nrcb1–4`).
Each arrow is that detector's median position residual against VIRAC, after the field-wide bulk
offset is removed — i.e. each detector is compared to the **external VIRAC frame**. That is why
every detector, including e.g. `nrcb2`, gets a vector even though it never physically overlaps NRCA
on the sky: the common reference is VIRAC, which covers the whole field. The arrow is placed at the
detector's mean sky position; the number of matched stars behind each arrow is annotated on the
plot.

The separate **A↔B overlap** panel is the [reference-free](#glossary-reffree) measurement: during
the dither pattern the detectors sweep across the sky, so a star that lands on an NRCA detector in
one exposure can land on an NRCB detector in another. Those genuinely-shared stars (the NRCA∩NRCB
overlap set) tie the two modules to each other directly.

<a id="glossary-lf"></a>
### Luminosity function (LF) and its turnover

The **luminosity function** is the histogram of source counts versus magnitude. Its **turnover**
is the magnitude of the peak bin — computed as the centre of the `argmax` bin of the exact
histogram that is plotted (no KDE or smoothing). Counts rise toward fainter magnitudes then fall
once the catalog stops being complete, so the turnover magnitude is a rough **depth** indicator
(fainter turnover ⇒ deeper catalog). In stage 2 the LF is drawn as a right-side marginal whose
magnitude axis is locked to the CMD.

<a id="glossary-snr"></a>
### S/N and the S/N > 10 cut

Where a plot is restricted to **S/N > 10**, "S/N" is the flux measurement signal-to-noise of each
star, `flux_fit / flux_err` from the per-exposure PSF fit. It is computed **per detection** (one
value per star per exposure, using that exposure's fitted flux and its formal flux error) — not a
mean flux over exposures divided by a mean error. The high-S/N cut keeps the best-measured stars,
so a residual scatter reflects the astrometric tie rather than photon-noise centroiding.

---

<a id="stage1"></a>
## Stage 1 — first mosaics

Stage 1 shows grayscale SW and LW `i2d` mosaic thumbnails, loaded from the `merged_i2d.fits`
images (from MAST or the pipeline, depending on the label) in ZScale/asinh grayscale. Check these
for double-stars or other weird visual artifacts. A nominal (proposed) filter with no mosaic on
disk is listed explicitly, never dropped silently.

Source: [`data_qa/diagnostics.py` → `stage1_mosaics`](../data_qa/diagnostics.py).

<a id="stage2"></a>
## Stage 2 — colour–magnitude diagram (CMD)

Stage 2 shows a 2-D density (hexbin) colour–magnitude diagram — LW magnitude versus (SW − LW)
colour — from the highest-tier [catalog](#glossary-mtier) on disk, with the
[luminosity function](#glossary-lf) drawn as a right-side marginal on a shared magnitude axis. Two
versions are produced: **all stars**, and one **limited to S/N > 10 in both bands** (the cleaner
locus).

`stage2_cmd` reads the jicama [`mN`](#glossary-mtier) catalog (or, if no single catalog holds both
bands, cross-matches the two single-band catalogs — labelled `crossmatch`); the MAST L3 product is
used only when no jicama catalog exists. `n_stars` = finite (SW, LW) pairs; `lf_turnover` =
magnitude of the LF peak bin. Consider it passing if the CMD shows a coherent stellar locus and
`n_stars` is not far below the field's expected depth.

Source: [`data_qa/diagnostics.py` → `stage2_cmd`](../data_qa/diagnostics.py).

<a id="stage3"></a>
## Stage 3 — photometric calibration (zeropoint)

Stage 3 shows a 2-D histogram (colour = number of stars) of JWST SW catalog magnitude versus
[VIRAC Ks](#glossary-virac) for the [cross-matched](#glossary-crossmatch) stars. The **cyan 1:1
line** is anchored on the densest stellar ridge (the mode of JWST−Ks); a well-calibrated catalog
lies along it.

`stage3_calibration` anchors on VIRAC (nearest JWST source within 0.1″), then makes a robust
linear fit with **up to 5 sigma-clip iterations** (3σ, early-break — bounded, not run to strict
convergence) to measure the slope and the scatter about the locus. The fitted slope is reported in
the title but not drawn (a free-slope line wanders with the red mismatch cloud and misreads as a
bad fit). `slope`, `scatter` (mag about the locus), `zeropoint`, `n_matched`, `n_locus`. Consider
it passing if the slope is 0.8–1.2 and the scatter < 0.8 mag.

Source: [`data_qa/diagnostics.py` → `stage3_calibration`](../data_qa/diagnostics.py).

<a id="stage4"></a>
## Stage 4 — positional offsets (JWST ↔ VIRAC frame tie)

Stage 4 checks how well the JWST catalog — the **jicama `mN` catalog**, the same one stages 2/3
use, not the MAST products — is registered onto the [VIRAC/Gaia frame](#glossary-virac). The
offset is measured in each cell of a **4×4 grid (16 cells)** over the footprint by the
[xcorr histogram peak](#glossary-xcorr) against the local VIRAC reference, and the reported field
offset is the source-count-weighted median across those cells (see [bulk offset](#glossary-bulk)).

- **LEFT** — the 4×4 grid, each cell filled with its offset colour (a contiguous map, so a coherent
  patch stands out). Cells with sources but no clear [xcorr peak](#glossary-xcorr) are hollow grey;
  a **green outline** marks cells that coherently deviate (an adjacency-confirmed sub-region tied
  differently from the rest of the field).
- **MIDDLE** — the same 16 cell offsets as (ΔRA, ΔDec) points sized by source count, with the
  source-weighted median tie (black +), the 75 mas gate (dotted circle), and ΔRA/ΔDec marginal
  histograms.
- **RIGHT** (when both modules are present) — the [reference-free](#glossary-reffree) NRCA-vs-NRCB
  inter-module offset, printed as its two numbers (ΔRA, ΔDec).

The reported **tie** is the [bulk offset](#glossary-bulk); its
[uncertainty](#glossary-tie-uncertainty) is the cell-to-cell standard error and "Nσ" = tie ÷ that
standard error. Consider it passing if: the tie is small (< 75 mas); the cells are spatially
consistent — no adjacency-confirmed sub-region is off by more than 30 mas while holding more than
2% of the stars (a lone mis-peaked cell does not count; a coherent block does); at least 4 cells
were measurable with ≥ 50% coverage of the sources; and the inter-module offset is < 15 mas.

Source: [`data_qa/diagnostics.py` → `stage4_offsets`](../data_qa/diagnostics.py)
(cells: `_cell_offsets`; aggregation/uncertainty: `_cell_consistency`).

<a id="stage5"></a>
## Stage 5 — inter-detector / inter-module tie

Stage 5 checks how well the detectors and the two modules agree, on the SW filter. The top row has
three panels; when a field lacks flux errors the S/N>10 panel is omitted (two-panel top row).

- **TOP-LEFT** — the [per-detector residual quiver](#glossary-quiver): one arrow per SW detector,
  each the detector's median residual against VIRAC with the field bulk offset removed, placed at
  the detector's mean sky position and annotated with its matched-star count.
- **TOP-MIDDLE** — the [reference-free](#glossary-reffree) NRCA∩NRCB overlap tie: the offset and
  RMS of the same stars seen in both modules (matched JWST-to-JWST, no external catalog), with
  ΔRA/ΔDec marginal histograms.
- **TOP-RIGHT** — the same overlap tie restricted to [S/N > 10](#glossary-snr) stars, so the
  residual scatter reflects the tie rather than photon-noise centroiding of faint sources.
- **BOTTOM STRIP** — a cutout gallery of overlap stars from the SW merged `i2d`. A good tie shows
  one round PSF; a mis-tie doubles or elongates the star (the same source drizzled twice at offset
  positions).

Consider it passing if the reference-free NRCA∩NRCB offset is small (< 15 mas) and the cutouts show
round, single PSFs. (A genuine single-module observation, e.g. NRCB-only, has no A↔B tie to fail
and passes by default.)

Source: [`data_qa/diagnostics.py` → `stage5_intermodule`](../data_qa/diagnostics.py)
(per-detector + high-S/N positions are pooled once from the per-exposure daophot catalogs and then
filtered, in `_module_positions`).

<a id="stage6"></a>
## Stage 6 — astrometric precision

Stage 6 shows three per-star error curves versus Vega magnitude, one set per channel:

1. **σ_pos** (solid) — the formal per-star PSF-fit position error (`dra`/`ddec` from the
   per-exposure daophot fit, pooled). The *predicted* precision of a single measurement.
2. **rms(jwst)** (dotted) — the empirical scatter of a star's position **across its exposures**
   (`std_ra`/`std_dec` from the merged catalog). The JWST *internal repeatability* — what σ_pos
   predicts, actually measured.
3. **rms(offset)** (dashed) — the RMS of the per-star JWST−VIRAC offset per magnitude bin. The
   *external* scatter against the reference; it includes the VIRAC error floor, so it sits highest.

The three separate "predicted precision" (σ_pos) from "internal repeatability" (rms(jwst)) from
"agreement with an external frame" (rms(offset)). The bright-end σ_pos floor is the astrometric
systematic limit; the faint-end rise tracks S/N. Shaded band = 16–84th percentile.

Source: [`data_qa/diagnostics.py` → `stage6_astrom_error`](../data_qa/diagnostics.py).

<a id="stage7"></a>
## Stage 7 — MAST vs pipeline (improvement over the delivered products)

> 🚧 **Planned — implemented in the stage-7 follow-up PR** (not yet in this branch). Documented
> here so the glossary/stage cross-references resolve; the source function below lands with that PR.

Stage 7 shows the gain of the pipeline over the raw MAST-delivered products, over one common
window of the mosaic: the MAST L3 `i2d` next to the pipeline `i2d` (before/after); magnitude
histograms of the MAST catalog vs the [jicama](#glossary-jicama) catalog (depth/count); and the
[bulk offset to VIRAC](#glossary-bulk) for each (the astrometric tightening).

Source: [`data_qa/diagnostics.py` → `stage7_mast_vs_pipeline`](../data_qa/diagnostics.py).
