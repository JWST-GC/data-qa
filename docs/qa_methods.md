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
- **JWST−VIRAC offset** ([stage 4](#stage4)): measured by an
  [x-correlation histogram peak](#glossary-xcorr) **per spatial cell**. Nearest-neighbour pairing
  fails at this density; see [field offset](#glossary-bulk).
- **Inter-module / per-detector offset** ([stage 5](#stage5)): the per-detector residual uses
  VIRAC matches within **0.15″**; the A↔B comparison uses no external catalog at all.

<a id="glossary-xcorr"></a>
### x-correlation histogram peak (`aa.xcorr`)

At Galactic-Centre stellar density a plain nearest-neighbour match fabricates pairs: within 0.3″
of a JWST source there are tens of thousands of chance VIRAC coincidences (≈32,000 measured for a
brick F212N field), and their median offset **collapses toward zero the further the frame is
actually displaced** — it reads ~1–2 mas even at a real 2″ shift. So QA instead builds the 2-D
histogram of all JWST−reference separation vectors and takes its **peak**: the displacement at
which real matched pairs pile up. `peak_ratio` = peak height ÷ background density; a match is
accepted only when `peak_ratio ≥ MIN_PEAK_RATIO` with enough pairs. The peak survives crowding
because the wrong pairs spread out over the whole search window while the right ones stack at one
displacement. (When both catalogs are dense — the JWST-NRCA↔JWST-NRCB case in [stage 5](#stage5) —
the chance-pair count is far higher, ~400,000, so that comparison uses the peak as well.)

The peak has a bias of its own worth knowing about: two catalogs tracing the same clustered field
make a wrong-pair background that is itself clumpy, and it pulls the peak by several mas. On brick
2221-o001 the peak reads 9–17 mas where the same stars, matched one to one, are 0.4–1.6 mas apart.
So the peak is what detects that an offset is small, and the [same-star](#glossary-bulk)
measurement is what reports how small.

<a id="glossary-bulk"></a>
### The field offset, and the per-cell offsets behind it

The **field offset** is the single shift between the JWST catalog and VIRAC — the number you would
apply to the whole frame to register it onto the Gaia/VIRAC frame. A field can have an **internal
discontinuity**, where one sub-region is registered differently from the rest (a stale visit block,
say), and one scalar hides that, so QA measures the offset region by region
([`_cell_offsets`](../data_qa/diagnostics.py)):

1. Split the JWST footprint into a **4×4 grid (up to 16 cells)**.
2. In each cell, measure the JWST−VIRAC offset by the [xcorr peak](#glossary-xcorr) against the
   local VIRAC reference (that cell + a 2″ margin). Cells with enough sources but **no clear peak**
   are recorded as *dropped* (drawn as a solid grey cell). A cell too sparse to reach the minimum
   source count is skipped, so fewer than 16 cells may survive.
3. Take the **weighted median over cells** of ΔRA, and separately of ΔDec, each cell weighted by
   its source count. A weighted median sorts the cells by that component and reads off the value at
   which the running source count passes half the total, so a cell holding a tenth of the catalog
   counts for a tenth of the catalog. The reported field offset is the length of the resulting
   (ΔRA, ΔDec) vector.

The two components are taken separately, so the result is the length of the median vector. Four
cells at (+50, 0), (−50, 0), (0, +50) and (0, −50) mas therefore give a field offset of 0 while
every cell sits 50 mas from it. The spatial-consistency test in
[`_cell_consistency`](../data_qa/diagnostics.py) is what catches that arrangement, by flagging each
cell that differs from the field value by more than 30 mas.

When the cells show the field offset is small, the reported value is re-measured from **the same
star seen in both catalogs** (mutual nearest pairs within 0.05″, median of their separations). That
measurement is only meaningful once a small offset is established: if the frame were really 2″ out,
the nearest VIRAC star to a JWST source would not be the same star, and its median would be
meaningless. The **pass gate tests the larger of the two values**, so a real mis-registration that
the histogram sees survives the re-measurement — every same-star pair is matched within 0.05″, so
that median alone can never exceed the 75 mas gate.

Stage 4 is a QA **cross-check** of a registration the jicama pipeline has already applied per
exposure. It measures the offset left over in the delivered catalog, reading the catalog alone.

<a id="glossary-tie-uncertainty"></a>
### What the stage-4 cell-to-cell spread means (and why no "σ" is quoted)

Alongside the field offset, stage 4 reports the **cell-to-cell spread**: the `mad_std` of the
per-cell ΔRA and ΔDec, combined in quadrature. It says how much the cells disagree with each other.
[Stage 6](#stage6) shows the per-star scatter separately, as rms(offset).

Stage 4 used to divide the field offset by spread ÷ √(number of cells) and report the result as
"N σ from zero". That number carried no information and has been removed. The field offset is a
length built from the same two medians whose sampling error is the denominator, so when the true
offset is zero the ratio still lands at ≈1.1 for 9 or more cells (≈1.5 for 4) — whatever the scale
of the scatter, and whatever the size of the field offset. A 780 mas offset reported at "1σ" was
the statistic sitting at its floor. In simulation, with the true offset set to zero:

| cells | cell scatter | quoted σ (median) | fraction reading > 3σ |
|------:|-------------:|------------------:|----------------------:|
| 4     | 5–500 mas    | 1.5               | 11% |
| 9     | 5–500 mas    | 1.1               | 3% |
| 16    | 5–500 mas    | 1.1               | 2% |

Dividing by √(number of cells) also assumes the cells are independent measurements of the same
quantity, and the error that dominates here is common to all of them: the several-mas pull that a
dense reference puts on every cell's [histogram peak](#glossary-xcorr) is the same pull in every
cell, so it does not shrink with more cells. [Stage 8](#stage8) measures its significance against a
shuffled-position null instead, which is the shape a significance for this kind of measurement has
to take.

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
star. In stage 5 (per-exposure daophot) it is `flux_fit / flux_err` of a single detection, computed **per detection** (one
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
versions are produced: **all stars**, and one **limited to [S/N > 10](#glossary-snr) in both bands** (the cleaner
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
## Stage 4 — positional offsets (JWST catalog − VIRAC)

Stage 4 measures how far a star in the JWST catalog sits from the same star in VIRAC, and so how
well the catalog is registered onto the [VIRAC/Gaia frame](#glossary-virac). The catalog it reads
is the **jicama `mN` catalog**, the same one stages 2 and 3 use. The offset is measured in each cell
of a **4×4 grid (up to 16 cells)** over the footprint by the [xcorr histogram peak](#glossary-xcorr)
against the local VIRAC reference; the [field offset](#glossary-bulk) is the source-count-weighted
median over those cells.

- **LEFT** — the 4×4 grid, each cell filled with its offset colour (a contiguous map, so a coherent
  patch stands out). Cells with sources but no clear [xcorr peak](#glossary-xcorr) are a solid grey
  fill; a **red outline** marks cells that deviate together (an adjacency-confirmed sub-region
  registered differently from the rest of the field).
- **MIDDLE** — the same per-cell offsets (up to 16) as (ΔRA, ΔDec) hollow-circle points sized by
  source count, with the field offset (black +), the 75 mas gate (dotted circle), the cell-to-cell
  spread (dashed circle), and ΔRA/ΔDec marginal histograms.
- **RIGHT** (when both modules are present) — the NRCA-minus-NRCB offset, measured
  [without any external catalog](#glossary-reffree) and printed as its two numbers (ΔRA, ΔDec).

The reported number is the [field offset](#glossary-bulk), re-measured from the same stars matched
one to one once the cells establish that it is small; the pass gate tests the larger of the two
values. Beside it is the [cell-to-cell spread](#glossary-tie-uncertainty), which says how much the
cells disagree with each other. Consider it passing if: the field offset is small (< 75 mas); the
cells are spatially consistent — no adjacency-confirmed sub-region is off by more than 30 mas while
holding more than 2% of the stars (a coherent block counts, a lone mis-peaked cell does not); at
least 4 cells were measurable, covering ≥ 50% of the sources; and the inter-module offset is
< 15 mas.

Source: [`data_qa/diagnostics.py` → `stage4_offsets`](../data_qa/diagnostics.py)
(cells: `_cell_offsets`; combining them: `_cell_consistency`).

<a id="stage5"></a>
## Stage 5 — inter-detector / inter-module tie

Stage 5 checks how well the detectors and the two modules agree, on the SW filter. The top row has
three panels; when a field lacks flux errors the S/N>10 panel is omitted (two-panel top row).

- **TOP-LEFT** — the [per-detector residual quiver](#glossary-quiver): one arrow per SW detector,
  each the detector's median residual **against VIRAC** with the field bulk offset removed, placed
  at the detector's mean sky position and annotated with the **number of matched stars**. (This is
  why an NRCB detector with no NRCA sky overlap still gets a vector — the common reference is
  VIRAC, not NRCA. See [the quiver note](#glossary-quiver).)
- **TOP-MIDDLE** — the [reference-free](#glossary-reffree) NRCA∩NRCB overlap tie: the offset and
  RMS of the **same stars** seen in both modules (matched JWST-to-JWST, no external catalog), with
  **marginal histograms** of the residual ΔRA/ΔDec. Matching is **one-to-one**: after the bulk A→B
  shift (the xcorr histogram peak) the nearest B source within 80 mas is taken for each A source and
  duplicate B are dropped (closest A kept), so the reported count is distinct overlap stars — NOT the
  many-to-many pair count a fixed-radius ball match would return in a crowded field.
- **TOP-RIGHT** — the same overlap tie restricted to [S/N > 10](#glossary-snr) stars (when the
  field has flux errors), so the scatter reflects the tie rather than faint-source centroiding.
- **FULL-WIDTH ROW (below the top panels)** — the A↔B overlap **footprint**: the overlap stars
  mapped on the sky (RA/Dec), coloured by each star's |A−B| residual, using the S/N > 10 set. The
  overlap is a thin, long strip, so this row spans the figure width (data-driven aspect) to make
  the per-star colour readable; it verifies the shared stars trace the NRCA∩NRCB dither-overlap
  strip (not the whole field) and flags any sub-region where the tie degrades.
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

<a id="stage8"></a>
## Stage 8 — inter-filter distortion residual

The **inter-filter** position residual as a function of position across the field: filter `sw`
minus a second JWST filter of the same field, using the **same source rows** in the merged
catalog, [S/N > 10](#glossary-snr) in both bands, [bulk](#glossary-bulk) removed. Two filters of
one field share the frames, the offsets table, the DVA correction and the reference tie, so the
only thing that differs is a **per-filter WCS term** — exactly the position-dependent distortion
residual this stage is for. There is **no external-catalog noise** (VIRAC's ~20 mas per-star PM
error would swamp a few-mas residual). The cross-band match radius did **not** disappear: the rows
were paired by a mutual-nearest-neighbour cross-band match at ~100 mas per band in
`merge_catalogs.py` (visible in the data as a hard truncation of the kept separations near 100 mas).
That blind spot survives, but at ~100× the ~1 mas signal it is far less binding than the ~0.15″
match in a VIRAC-referenced version. The partner filter is the nearest in wavelength that has a
`skycoord_<f>` column in the catalog.

- **LEFT / MIDDLE** — binned-median ΔRA and ΔDec maps (12×12, diverging colour; RA bins carry
  `cos δ` so cells span equal on-sky distance).
- **RIGHT** — a per-cell residual quiver.

A flat map means the two filters' solutions agree; a coherent gradient or swirl is a differential
distortion residual. Significance is quoted against a **shuffled-position null**: the residual
vectors are permuted across the fixed cells (~20 times) and the 90th-percentile cell amplitude is
recomputed, so the null carries the nearest-neighbour-ambiguous tail and the median's efficiency
penalty. On the brick the observed amplitude (~1.1 mas) sits at ~5–6× the null (~0.2 mas). The
per-cell standard error (per-star scatter ÷ √stars-per-cell) is reported too, but it runs ~2×
smaller than the null because the ~7–8% of rows with |Δ| > 20 mas (nearest-neighbour ambiguity
within the match radius) inflate the sampling noise of a cell median beyond scatter/√n; the
null-based figure is the one to trust. This complements [stage 4](#stage4) (the *bulk* tie + cell
consistency) by exposing *spatially-structured* residuals.

**Pass/fail semantics.** A real ~1 mas inter-filter distortion term is an *expected measurement*,
not a defect, so `passed` reflects only whether the measurement **succeeded** (enough populated
cells) — it is **not** gated on the amplitude versus a self-derived noise level, so injecting noise
cannot flip it. A single-filter or not-yet-merged obs has no second band to difference: that is a
distinct *not-applicable* state (no `passed`, no red flag). A **red flag** is raised only on a
**gross** absolute inter-filter offset (fixed `binned_amp90_mas` > 15 mas), which would indicate a
genuine per-filter WCS break rather than normal distortion.

Metrics: `n_stars`, `resid_rms_mas` (per-star), `binned_amp90_mas` (90th-percentile cell
amplitude), `null_amp90_mas` and `amp90_significance` (observed ÷ null; also `amp90_p_value`),
`per_cell_sem_mas` (reported, ~2× optimistic), `frac_gt_20mas`, `stars_per_cell`,
`cells_used`/`cells_total`.

**Source:** [`data_qa/diagnostics.py` → `stage8_distortion`](../data_qa/diagnostics.py)
(`_interfilter_residuals`, `_binned_median_2d`).

<a id="stage9"></a>
## Stage 9 — PSF vs aperture photometry

The jicama catalog reports **PSF-fit** fluxes. Stage 9 **re-measures** simple **aperture**
photometry on the mosaic at the catalog positions (a 3 px circular aperture with a 6–9 px annulus
for local background) and compares the two, so a PSF-model or crowding problem shows up as a
disagreement. (This is a stand-in until the jwst-gc-pipeline emits aperture catalogs of its own;
aperture photometry is cheap to re-measure.)

To keep a neighbour's light out of the aperture, the comparison is restricted to **isolated**
stars — nearest catalog neighbour more than **8 px** away. The LEFT panel plots aperture vs PSF
instrumental magnitude with the 1:1 + aperture-correction line; the RIGHT panel plots
(aperture − PSF) vs PSF magnitude. `n_isolated`, `aper_corr_med` (the median offset = the aperture
correction), `aper_psf_scatter`. Consider it passing if the (aperture − PSF) locus is flat at a
constant offset with small scatter (< ~0.15 mag); curvature or large scatter flags a PSF-model or
crowding problem.

Source: [`data_qa/diagnostics.py` → `stage9_psf_vs_aper`](../data_qa/diagnostics.py)
(`_psf_flux_positions`).

<a id="stage7"></a>
## Stage 7 — MAST vs pipeline (improvement over the delivered products)

**What it shows.** The gain of the pipeline over the raw **MAST-delivered** products, over one
common central window of the mosaic:
- **TOP — i2d before/after**: the STScI/MAST merged `i2d` mosaic next to our pipeline mosaic (same
  filter, same sky region, same stretch).
- **BOTTOM-LEFT — catalog depth (brief)**: magnitude histograms of the **MAST catalogue** vs the
  [jicama](#glossary-jicama) catalogue in the common window. The MAST catalogue is the
  MAST-delivered L3 `_cat.fits` **when it is archived** (fetched from MAST if not already local);
  when MAST did not archive one, it is **approximated** by running DAOStarFinder at 5σ over a
  Background2D on the MAST i2d. That detection is an approximation of, not a match to, the STScI
  `SourceCatalogStep` (which uses image segmentation with deblending), so the two source lists
  differ — read the count as a depth indicator, not a reproduction of the STScI catalogue. jicama
  recovers more and fainter stars by construction (the point is the count/depth; the two use
  different zeropoints).
- **BOTTOM-RIGHT — astrometry (main)**: the [bulk offset](#glossary-bulk) to
  [VIRAC](#glossary-virac) for each catalogue, measured by the [xcorr histogram
  peak](#glossary-xcorr) (crowding-robust — a nearest-neighbour-to-VIRAC distance is meaningless
  at GC density, where VIRAC's ~250 mas source spacing swamps the tie). Shown as a 2-D (ΔRA, ΔDec)
  cloud with marginals whose CENTRE is the bulk tie: the MAST cloud sits at its raw-WCS bulk
  offset, the jicama cloud typically near the origin. The panel title and issue caption are derived
  from the sign of (jicama tie − MAST tie): they assert the pipeline "tightens the tie" only when
  both ties are measured **and** jicama is tighter, and otherwise report both numbers without
  claiming an improvement. The cloud's **width** is bounded by the 0.1″ `search_around_sky`
  cross-match radius — it shows the match distribution, **not** the per-star astrometric precision;
  see [stage 5](#stage5)/[stage 6](#stage6) for the actual per-star RMS. When a tie is unmeasurable
  (no xcorr peak within 1.5″, e.g. a grossly mis-registered product): if only the MAST tie is
  unmeasurable the comparison is flagged as **unavailable** (a possible MAST mis-registration, not a
  defect in our product, so the stage does not fail on that alone); if the jicama tie is
  unmeasurable that is our product failing, so the stage does not pass and is red-flagged.

**Source:** [`data_qa/diagnostics.py` → `stage7_mast_vs_pipeline`](../data_qa/diagnostics.py)
(MAST catalogue: `_mast_l3_catalog`/`_load_mast_catalog`; detection fallback: `_detect_on_mosaic`;
bulk tie: `_tie_cloud`).



<a id="stagemiri"></a>
## MIRI basics (posted on MIRI issues)

The NIRCam stages above don't apply to MIRI; MIRI issues instead get a **basics** overview
(`miri_overview`, posted with the `data-qa:diag:stagemiri` marker):

- **MAST i2d image** — the MIRI level-3 mosaic, grayscale (ZScale/asinh). Check for delivery and
  gross artifacts.
- **Spitzer side-by-side** — the nearer archival Spitzer band, **reprojected onto the MIRI WCS +
  pixel grid** (GLIMPSE is GLON-CAR and MIPSGAL is RA-TAN north-up, both a quarter-turn from the
  MIRI observing PA, so a raw cutout would not line up): **IRAC 8 µm** (GLIMPSE) below ~14 µm (the
  geometric midpoint of the two bands, so F1280W maps to the closer IRAC), **MIPS 24 µm** (MIPSGAL)
  above. The "same footprint" note appears only when the cutout both covers the field and reprojects;
  otherwise the panel is only wavelength-matched. Shows what MIRI resolves that Spitzer could not.
  (Mosaics: `QA_IRAC4` / `QA_MIPS24`, GC-wide by default.)
- **Saturation mask** — which pixels are flagged **SATURATED** in the MAST DQ (from the per-exposure
  `_cal`, falling back to `_rate`; the i2d carries no DQ). Reports the **median** and **max**
  saturated fraction across every readable frame of the obs (a single frame is one exposure of many)
  and shows the worst frame's mask. Omitted when no readable DQ is staged locally.

Source: [`data_qa/diagnostics.py` → `miri_overview`](../data_qa/diagnostics.py)
(`_miri_i2d`, `_spitzer_for_miri`, `_saturation_mask`).
