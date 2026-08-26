# QA diagnostics — how every plot and number is made

This document is the reference for the automated QA diagnostics posted on each observation's
tracking issue. Every diagnostic comment links back here (to the matching stage section) and to
the exact source function that built it.
- Source: [`data_qa/diagnostics.py`](../data_qa/diagnostics.py) builds every figure and metric;
  [`data_qa/post_diagnostics.py`](../data_qa/post_diagnostics.py) posts them.
- The diagnostics are **NIRCam-only**. MIRI issues carry only the pipeline-status table.
- Each stage writes a metrics JSON (`data_qa/metrics/<obsid>.json`) and posts/updates the extisting comment.
-  **The `passed` and `failed` flags should only serve as suggestions for the reviewer**. Each stage below defines "passing" criteria.
- Every stage reports the **full path of every file it read** — see
  [which files a stage used](#inputs).

Jump to: [Glossary](#glossary) ·
[Stage 1](#stage1) · [Stage 2](#stage2) · [Stage 3](#stage3) · [Stage 4](#stage4) ·
[Stage 5](#stage5) · [Stage 6](#stage6) · [Stage 7](#stage7)

---

<a id="inputs"></a>
## Which files a stage used

Each posted comment ends with a collapsed **Files read for this stage** block giving the full
path of every file that stage opened, grouped by what it was used for and then by directory. The same list is the `inputs` key of that stage in
`data_qa/metrics/<obsid>.json`.

Due to GitHub's 65 kB comment limit, if more than 12 files were used then only the first three and last filenames are displayed. In such cases, <o.obsid>.json file contains all paths that were used. 

<a id="glossary"></a>
## Glossary

<a id="glossary-mtier"></a>
### Catalog tiers: MAST default, `m1` … `m8`

The jicama pipeline refines a field's catalog through numbered merge/refinement stages. Higher =
more processed; the QA always chooses the **highest tier present on disk** for the observation with the MAST-delivered catalog being the lowest priority and `m8` being the highest. 
- **MAST-delivered source catalog** — the STScI level-3 pipeline's own source list shipped with
  the mosaic (`*_cat.fits`). Single-band, aperture photometry, no cross-band merge. This is the
  "raw" baseline the pipeline improves on (see [stage 7](#stage7)).
- **`mN`** — the `N`th [jicama](#glossary-jicama)-pipeline pass. Successive passes add
  PSF-fit photometry, per-exposure combination, cross-band forced photometry, and quality
  vetting; `m7` seeds each filter's forced photometry at the cross-band source positions.
- **`m8` / `m8_dedup`** — the final iteration. `m8_dedup` removes duplicate rows that may be present in `m8`due to the cross-band merge. The most-recent-on-disk of the two is used. The cataloging process is described in https://github.com/keflavich/jwst-gc-pipeline/blob/main/PHOTOMETRY_PIPELINE.md (most of the text there is AI generated for now).

- A catalog labelled **`crossmatch`** in a caption means no single merged catalog held both
  requested filters, so the CMD was built by cross-matching the two single-band catalogs. 

<a id="glossary-jicama"></a>
### jicama

`jicama` is the name of our PSF-photometry catalog pipeline, distinct from the MAST/STScI
source catalog, and from the `peppar` and `STARFINDER` catalogs).
The `mN` catalogs are a product of `jicama` pipeline.

<a id="glossary-virac"></a>
### VIRAC / VIRAC2 / Ks

**VIRAC2** = the VISTA Variables in the Vía Láctea Infrared Astrometric Catalogue, version 2
(VizieR `II/387`). It is **tied to the Gaia DR3 frame at epoch 2014.0** and carries near-IR
**Ks**-band (~2.15 µm) magnitudes and per-star **proper motions (PM)**. QA uses it as the
external reference for both astrometry (stages 4/5) and photometry (zeropoint,
stage 3).

Before any comparison, the VIRAC positions are **proper-motion-propagated** from 2014.0 to the
JWST observation epoch.

<a id="glossary-crossmatch"></a>
### JWST ↔ VIRAC cross-match and the selection criteria

"The JWST catalog cross-matched to VIRAC" means: take the JWST catalog source positions and pair them with the
PM-propagated VIRAC reference. The **selection criteria** differ by what is being measured:

- **Photometric zeropoint** ([stage 3](#stage3)): anchor on the sparse, Ks-bright VIRAC stars and
  take each one's nearest JWST catalog source within **0.1″**. Anchoring on VIRAC prevents faint JWST sources from being paired with the wrong bright VIRAC
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

<a id="glossary-adjacency"></a>
### "deviate together": the adjacency rule for outlined cells

A cell is **deviating** when its per-cell offset sits more than `_CELL_SPREAD_MAX` (30 mas) from the
[source-weighted field offset](#glossary-bulk). But a single deviating cell is not, on its own,
evidence that the *frame* is wrong — it is usually just a mis-measurement in that one cell (a
[histogram peak](#glossary-xcorr) that landed on the wrong bump). So a deviating cell is **confirmed**
(and drawn with an outline) **only when an orthogonally-adjacent cell also deviates**. That is what
"deviate together" means: outlined cells form a coherent block of neighbours that share the
deviation, which is the signature of a real sub-region registered differently from the rest of the
field — an [inter-module](#glossary-reffree) seam, a bad detector, a local distortion residual. A
lone deviating cell amid agreeing neighbours stays unoutlined and does **not** fail the frame.

The frame's spatial-consistency check fails when the adjacency-confirmed cells hold more than 2% of
the measured sources (a real block, not a speck), or when less than 50% of the field could be
measured at all.

Why adjacency rather than a spread statistic: an earlier version used `mad_std` of the cell offsets,
which cannot see a *minority* discontinuity — a defect covering a few cells is averaged away by the
many good ones. Requiring adjacency makes the test sensitive to a coherent minority while ignoring
isolated noise (introduced in PR #54, in response to that review).

<a id="glossary-reffree"></a>
### Measuring JWST against itself ("reference-free")

**Reference-free** means the offset is measured by matching **JWST against itself** — module NRCA
directly against module NRCB — using **no external catalog** (no VIRAC, no Gaia). What it measures
is internal to the instrument, so any error in an external reference stays out of it. See
[stage 5](#stage5).

<a id="glossary-quiver"></a>
### The per-detector quiver, and how NRCB detectors get a vector

The stage-5 quiver has **one arrow per detector** (up to 8 SW detectors: `nrca1–4`, `nrcb1–4`).
Each arrow is that detector's median position residual against VIRAC, after the field-wide bulk
offset is removed — i.e. each detector is compared to the **external VIRAC frame**. That is why
every detector, including e.g. `nrcb2`, gets a vector even though it never physically overlaps NRCA
on the sky: the common reference is VIRAC, which covers the whole field. The arrow is placed at the
detector's mean sky position; the number of matched stars behind each arrow is annotated on the
plot.

The separate **A↔B overlap** panel compares [JWST against itself](#glossary-reffree): during the
dither pattern the detectors sweep across the sky, so a star that lands on an NRCA detector in one
exposure can land on an NRCB detector in another. Those genuinely-shared stars (the NRCA∩NRCB
overlap set) measure the two modules against each other directly.

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
star. In stage 5 (per-exposure daophot) it is `flux_fit / flux_err` of a single detection,
computed **per detection**: one value per star per exposure, from that exposure's fitted flux and
its own formal flux error. (A mean flux over exposures divided by a mean error would be a different
quantity.) The high-S/N cut keeps the best-measured stars, which **bounds** the centroid noise
admitted rather than removing it: a star sitting at the cut still carries σ_pos ≈ FWHM/(2.35·S/N)
≈ 2.8 mas for F200W, or ~4 mas into an A−B difference, against the 15 mas inter-module line drawn
on the same panel. Above the cut the residual scatter is dominated by how well the frames agree.

---

<a id="stage1"></a>
## Stage 1 — first mosaics


Stage 1 shows grayscale `i2d` mosaic thumbnails, loaded from the `merged_i2d.fits`
images in ZScale/asinh grayscale to provide the first look of the data.


<details> 
<summary>What to do?</summary>
<br>
Check the images for double-stars or other visual artifacts. 
</details>
<br>


<details> 
<summary>Other details</summary>

* For redundancy, if a filter was observed, but no data exists on disk then this will be explicitly stated here.

* The images gets loaded from the pipeline folder or, if not available, from MAST. See "files read" for details.
</details>
<br>

Source: [`data_qa/diagnostics.py` → `stage1_mosaics`](../data_qa/diagnostics.py).

<a id="stage2"></a>
## Stage 2 — color–magnitude diagram (CMD)

Stage 2 shows a 2-D density color–magnitude diagram — LW magnitude versus (SW − LW)
color — from the highest-tier [catalog](#glossary-mtier) on disk, with the
[luminosity function](#glossary-lf) drawn as a right-side marginal on a shared magnitude axis. Two
versions are produced: **all stars**, and one **limited to [S/N > 10](#glossary-snr) in both bands**.

`stage2_cmd` reads the jicama [`mN`](#glossary-mtier) catalog (or, if no single catalog holds both
bands, cross-matches the two single-band catalogs — labelled `crossmatch`); the MAST L3 product is
used only when no jicama catalog exists. `lf_turnover` =
magnitude of the LF peak bin. 

<details> 
<summary>What to do?</summary>
<br>
Look for a coherent stellar locus. Check whether the number of stars is reasonable based on other similar fields.
</details>
<br>

Source: [`data_qa/diagnostics.py` → `stage2_cmd`](../data_qa/diagnostics.py).

<a id="stage3"></a>
## Stage 3 — photometric calibration (zeropoint)

Stage 3 shows a 2-D histogram (colour = number of stars) of JWST SW catalog magnitude versus
[VIRAC Ks](#glossary-virac) for the [cross-matched](#glossary-crossmatch) stars. The **cyan 1:1
line** is anchored on the densest stellar ridge (the mode of JWST−Ks); a well-calibrated catalog
lies along it.

`stage3_calibration` anchors on VIRAC (nearest JWST source within 0.1″), then makes a robust
linear fit with **up to 5 sigma-clip iterations** (3σ) to measure the slope and the scatter about
the locus. The loop stops on either of two conditions: the clip is stable (nothing more to
reject), **or** the next clip would leave fewer than 30 stars. The second exit returns before the
clipped set is adopted, so on a sparse field the reported `slope`, `scatter` and `n_locus` are
those of the **unclipped** match set — `n_locus == n_matched` is the signature, and it reads the
same as a locus so clean the first pass rejected nothing. The fitted slope is reported in the title and
left undrawn: a free-slope line wanders with the red mismatch cloud and reads as a bad fit. `slope`, `scatter` (mag about the locus), `zeropoint`, `n_matched`, `n_locus`. Consider
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
  fill; a **red outline** marks a group of adjacent cells that each deviate from the field value
  by more than 30 mas — a sub-region registered differently from the rest of the field. A lone
  deviating cell is a mis-measurement in that one cell and is left unoutlined.
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
## Stage 5 — inter-detector / inter-module agreement

Stage 5 measures how far a star seen in one detector sits from the same star seen in another, on
the SW filter. The top row has three panels; when a field lacks flux errors the S/N>10 panel is
omitted (two-panel top row).

- **TOP-LEFT** — the [per-detector residual quiver](#glossary-quiver): one arrow per SW detector,
  each the detector's median residual **against VIRAC** with the field offset removed, placed at
  the detector's mean sky position and annotated with the **number of matched stars**. VIRAC covers
  the whole field, so an NRCB detector that shares no sky with NRCA still gets a vector. See
  [the quiver note](#glossary-quiver).
- **TOP-MIDDLE** — NRCA against NRCB in their overlap, [JWST matched to itself with no external
  catalog](#glossary-reffree): the offset and the scatter of the **same stars** seen in both
  modules, with **marginal histograms** of the residual ΔRA/ΔDec. Matching is **one-to-one**: after
  the bulk A→B shift (the xcorr histogram peak) the nearest B source within 80 mas is taken for each
  A source and duplicate B are dropped (closest A kept), so the reported count is distinct overlap
  stars. A fixed-radius ball match in a crowded field returns a many-to-many pair count instead,
  which is the ~34k figure this replaced.
- **TOP-RIGHT** — the same comparison restricted to [S/N > 10](#glossary-snr) stars (when the field
  has flux errors), where the scatter measures how well the modules agree with the centroid noise
  of faint stars taken out of it.
- **FULL-WIDTH ROW (below the top panels)** — the A↔B overlap **footprint**: the overlap stars
  mapped on the sky (RA/Dec), coloured by each star's |A−B| residual, using the S/N > 10 set. The
  overlap is a thin, long strip, so this row spans the figure width (data-driven aspect) to make
  the per-star colour readable. It shows the shared stars tracing the NRCA∩NRCB dither-overlap
  strip, and flags any part of that strip where the two modules agree less well.
- **BOTTOM STRIP** — a cutout gallery of overlap stars from the SW merged `i2d`. Where the modules
  agree, each star is one round PSF; where they disagree it doubles or elongates, the same source
  drizzled twice at offset positions.

The scatter quoted on the A↔B panels is `hypot` of the ΔRA and ΔDec `mad_std`. Comparing it with a
[stage 6](#stage6) curve takes **two** factors, not one:

- `hypot` **combines** the two axes, while stage 6 divides by √2 to stay **per-axis** — √2.
- each residual is a **difference**, A − B, of two independent measurements of the same star, so
  its per-axis scatter is already √2 above what one module alone contributes — √2 again.

All three [stage 6](#stage6) curves — `σ_pos`, `rms(jwst)` and `rms(offset)` — are per-axis
single-measurement quantities, so the stage-5 number runs **2×** above each of them for isotropic
scatter. Measured: a 10.00 mas per-axis single-module error gives 20.00 mas here and 10.00 mas
there. (Aligning A onto B before differencing removes two numbers across ~1000 stars, which does
not change this.)

`rms(offset)` does sit above `σ_pos` on a real field, for a reason that is **not** a convention:
it is measured against VIRAC and carries VIRAC's ~20 mas per-star error. Its estimator is
`sqrt(mean(r²))` over a residual already divided by √2, which reads 1.00× a per-axis σ, the same
as `σ_pos`.

Consider it passing if the NRCA∩NRCB offset is small (< 15 mas) and the cutouts show round, single
PSFs. A genuine single-module observation (NRCB-only, say) has no A↔B comparison to make and passes
by default.

Source: [`data_qa/diagnostics.py` → `stage5_intermodule`](../data_qa/diagnostics.py)
(per-detector + high-S/N positions are pooled once from the per-exposure daophot catalogs and then
filtered, in `_module_positions`).

<a id="stage6"></a>
## Stage 6 — astrometric precision

Stage 6 shows per-star error curves versus Vega magnitude, one set per channel, over a parallel
lower panel of source counts:

1. **formal σ_fit** (solid) — the PSF fitter's formal per-detection 1σ position error (`dra`/`ddec`
   from the per-exposure daophot fit, pooled). A formal error bar has **no systematic in it by
   construction**, so its ~0.06 mas bright-end floor is the noise-limited *fit* uncertainty, **not**
   the achieved astrometric precision. Do not cite it as the delivered precision.
2. **rms(jwst)** (dotted) — the empirical scatter of a star's position **across its exposures**
   (`std_ra`/`std_dec` from the merged catalog). This is the **achieved internal repeatability**:
   a sub-mas floor, well above the formal σ_fit (on brick jw02221-o001, `floor_mas` for F212N is
   0.76 mas vs a 0.06 mas formal floor — ~12×; other bands 0.6–0.8 mas). The headline `floor_mas`
   metric reports this number when per-exposure catalogs are available (with `floor_is_empirical`
   true), and `formal_sigma_floor_mas` keeps the formal value; without them `floor_mas` falls back
   to the formal floor and `floor_is_empirical` is false.
3. **rms(offset)** (dashed) — the RMS of the per-star JWST−VIRAC offset per magnitude bin. The
   *external* scatter against the reference; it includes the VIRAC error floor, so it sits highest.

The three separate the *fit uncertainty* (σ_fit) from the *achieved repeatability* (rms(jwst)) from
*agreement with an external frame* (rms(offset)); their faint-end rise tracks S/N, shaded band =
16–84th percentile. The **lower-left panel** histograms the number of sources in each 0.5-mag Vega
bin — the sample size behind each curve point, and where it runs out at the faint end.

**Right column — the independent `peppar` (Hosek WebbPSF) catalogues**, a cross-check that shares
none of jicama's detection/fit choices (peppar magnitudes are instrumental, no zero-point). Two
curves per channel:

- **per-frame formal σ_fit** (dashed) — the peppar PSF-fit position error (`x_err`/`y_err`), the
  noise-limited *predicted* precision, the analogue of jicama's formal σ_fit.
- **frame-to-frame σ** (solid) — the standard deviation of each star's **sky position across the
  exposures it appears in**, the analogue of jicama's rms(jwst): the *achieved* repeatability,
  ~20–100× the formal error. It is read straight from a combined starlist's `x_wcs_std`/`y_wcs_std`
  when one exists; otherwise it is **computed** by matching the per-frame catalogues in sky
  coordinates (via each frame's cal WCS, since the exposures are dithered and mosaicked so a star
  lands on different pixels — and different chips — each frame), grouping detections within 40 mas
  and taking the scatter of groups seen in ≥3 exposures. Its own source-count histogram sits below
  it. Metrics: `peppar_framestd_floor_mas_<filt>` (the achieved floor, also the headline
  `peppar_floor_mas`), `peppar_formal_floor_mas_<filt>`.

**Second figure when exposures are flagged.** When [stage 11](#stage11) flags one or more
**bad-PSF (streaked/broadened) exposures** for the obs, stage 6 posts an **additional** figure (its
own comment, marker `stage6clean`) recomputed with those exposures **left out of the per-exposure
pools** — so a momentary tracking failure no longer inflates the precision. The `formal σ_fit`,
`rms(offset)` and the peppar `frame-to-frame σ` / `formal` curves are all rebuilt without the bad
exposures; the `rms(jwst)` curve is omitted there (it comes from the merged catalogue's all-exposure
`std`, which cannot be re-derived per-exposure), so the peppar frame-to-frame σ carries the
achieved-repeatability comparison. Metrics: `excluded_exposures` and `clean_*_floor_mas`.

Source: [`data_qa/diagnostics.py` → `stage6_astrom_error`](../data_qa/diagnostics.py)
(`_stage6_figure`, `_streaked_exposures`, `_peppar_precision`, `_peppar_frame_std`).

<a id="stage8"></a>
## Stage 8 — inter-filter distortion residual

The **inter-filter** position residual as a function of position across the field: filter `sw`
minus a second JWST filter of the same field, using the **same source rows** in the merged
catalog, [S/N > 10](#glossary-snr) in both bands, [field offset](#glossary-bulk) removed. Two
filters of one field share the frames, the offsets table, the DVA correction and the registration
onto VIRAC, which leaves a **per-filter WCS term** as the one thing that differs between them —
the position-dependent distortion residual this stage is for. No external catalog enters the
measurement, which matters because VIRAC's ~20 mas per-star PM error would swamp a few-mas
residual.

The rows were paired across bands upstream, by a mutual-nearest-neighbour match at ~100 mas per
band in `merge_catalogs.py`, visible in the data as a hard truncation of the kept separations near
100 mas. That radius is ~100× the ~1 mas signal, so it does little to shape this map (a
VIRAC-referenced version would carry a ~0.15″ match instead). The partner filter is the nearest in
wavelength that has a `skycoord_<f>` column in the catalog.

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
null-based figure is the one to trust. This complements [stage 4](#stage4), which measures the
field offset and the cell-to-cell consistency, by exposing *spatially-structured* residuals.

**Pass/fail semantics.** A real ~1 mas inter-filter distortion term is an *expected measurement*,
so `passed` reflects only whether the measurement **succeeded** (enough populated cells). It is
free of any gate on the amplitude against a self-derived noise level, which is what keeps injected
noise from flipping it. A single-filter or not-yet-merged obs has no second band to difference and
lands in a distinct *not-applicable* state (no `passed`, no red flag). A **red flag** is raised on
a **gross** absolute inter-filter offset alone (fixed `binned_amp90_mas` > 15 mas), the size that
indicates a genuine per-filter WCS break on top of the normal distortion.

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

<a id="stage10"></a>
## Stage 10 — JWST1PASS across-exposure consistency

**What it shows.** Whether every exposure of one filter is "in family" — measuring the same star's
position and brightness to the same value — from Jay Anderson's **JWST1PASS** pipeline, independent
of jicama and of the STScI level-3 products.

**How JWST1PASS gets there** (run outside data-qa, per filter; this stage consumes the result). For
each `_cal` frame, `jwst1pass` fits an empirical library PSF in the **STDPSF** format (a 5×5 grid of
PSFs across each NIRCam chip) with a **STDGDC** geometric-distortion correction, and — with `PERT=1`
— derives a single **perturbation PSF** from the fit residuals of bright isolated stars and adds it
to the whole 5×5 grid (temporal PSF variation is usually orthogonal to the spatial variation). The
per-chip `.xympqsuvw` catalogues are combined into one **META** frame per exposure
(`convert_nrcab2nrczz_meta` with the `map2avg` linear transforms), matched across exposures
(`xym2mat`, first exposure as the reference frame) and collated (`xym2bar`, a star must appear in
≥2 exposures) into **`MATCHUP.XYMEEE`**: one row per star holding the mean position (`xbar`, `ybar`)
and instrumental magnitude (`mbar`) in the reference frame and the RMS of each across the exposures
it was found in (`xsig`, `ysig`, `msig`), plus the mean quality-of-fit (`qbar`).

**The panels** reproduce Jay's `show_matchup.sm`, each versus instrumental magnitude: **X RMS** and
**Y RMS** (position repeatability, META pixels → mas at 32 mas/pixel), **magnitude RMS**
(photometric repeatability), and **quality of fit**. A tight, flat bright-end floor that rises only
at the **faint** (S/N) and **bright** (saturation) ends is in family; a raised or structured floor
flags a photometric or distortion problem in that filter's frames. Metrics: `x_rms_floor_mas`,
`y_rms_floor_mas`, `mag_rms_floor`, `qfit_floor`, `n_stars`, `n_exposures`, `saturation_turnover_mag`
(brightest magnitude where the mag-RMS has doubled above its floor — the saturation onset). Saturated
stars are recovered to ≈0.05 mag RMS up to a few magnitudes above saturation.

The stage reads `{root}/{field}/jwst1pass/{FILT}/MATCHUP.XYMEEE` (`QA_JWST1PASS_DIR` overrides the
lookup for a one-off product directory); it red-flags when JWST1PASS has not been run for the
obs/filter. The perturbation-PSF `LOG.psfperts.fits` per chip is a further check Jay notes (small,
exposure-to-exposure consistent variations); a per-exposure panel for it is a planned follow-up.

Source: [`data_qa/diagnostics.py` → `stage10_photometric_consistency`](../data_qa/diagnostics.py)
(`_read_matchup_xymeee`, `_jwst1pass_matchup`).

<a id="stage11"></a>
## Stage 11 — effective PSF per exposure

**What it shows.** Whether any exposure has a **streaked or broadened PSF** — a momentary tracking
failure or guide-star glitch (e.g. arches jw02045-o001 exposure 4, "tracking failed for a second").
For each exposure of a representative detector, the **empirical (effective) PSF** is the **mean** of
the peak-normalised cutouts of its bright, isolated, **unsaturated** stars detected directly on the
cal image. Saturated stars are excluded via the **DQ** plane (their flat-topped cores are the
brightest and hide the effect, so they would otherwise dominate); a cosmic-ray-dominated cutout is
skipped by requiring the peak at the stamp centre; and the mean — not the median — keeps the
structure. The stamps use a **log** stretch so the faint wings (and the broadened halo) show. A good
exposure gives a sharp core with the six-spike NIRCam diffraction pattern; a glitched exposure gives
a **broadened, washed-out** stamp — a larger halo and a lower, less-peaked core (the broadening is
roughly symmetric, so it is a *breadth* change, not a clean elongation). Each panel is labelled with
its **star count**, its ePSF **rms radius** (`r`, the breadth), and its `qfit`. The frames are
**scoped to this observation** (a peppar filter directory can hold several observations' exposures).

**The objective flag.** Each stamp is labelled with that exposure's peppar PSF-fit
**quality-of-fit** (`qfit`, median over its bright stars): a streaked exposure fits the empirical PSF
far worse, so its `qfit` spikes. An exposure whose `qfit` exceeds **`_EPSF_QFIT_STREAK_FACTOR`
(2×)** the median across the run's exposures is flagged as streaked (arches o001: exposure 4 reads
`qfit≈16` against a `~5.5` baseline; the other eleven, including exposure 12, sit at the baseline).
Flagged exposures degrade the PSF-fit astrometry/photometry and are candidates to down-weight or drop.

This is built from **our own data** (peppar per-frame catalogues + cal images), independent of
JWST1PASS (stage 10), so it works on every field with peppar products. Metrics: `n_exposures`,
`detector_shown`, `qfit_baseline`, `qfit_by_exposure`, `epsf_nstars_by_exposure`,
`epsf_nstars_total`, `epsf_nstars_median`, `streaked_exposures`, `n_streaked`. Red-flags when no
peppar catalogues exist for the obs/filter.

Source: [`data_qa/diagnostics.py` → `stage11_effective_psf`](../data_qa/diagnostics.py)
(`_exposure_qfit`, `_effective_psf`).

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
  Background2D on the MAST i2d. The STScI `SourceCatalogStep` uses image segmentation with
  deblending, so a DAOStarFinder run approximates it and the two source lists differ; read the
  count as a depth indicator. jicama recovers more and fainter stars by construction (the point is
  the count/depth; the two use different zeropoints).
- **BOTTOM-RIGHT — astrometry (main)**: each catalogue's [offset from
  VIRAC](#glossary-bulk), found by coarse-aligning on the [xcorr histogram
  peak](#glossary-xcorr) and then keeping the pairs within 0.1″. The coarse alignment is what makes
  this measurable at all: VIRAC's own ~250 mas source spacing means the distance from a JWST source
  to its nearest VIRAC neighbour is about that spacing for any frame. Shown as a 2-D (ΔRA, ΔDec)
  cloud with marginals whose CENTRE is the offset: the MAST cloud sits at its raw-WCS offset, the
  jicama cloud typically near the origin. The panel title and issue caption are derived from the
  sign of (jicama offset − MAST offset): they say the pipeline "tightens" only when both offsets
  are measured **and** jicama is the smaller, and otherwise report both numbers and claim no
  improvement. The cloud's **width** is bounded by the 0.1″ `search_around_sky` cross-match radius,
  so it shows the match distribution; [stage 5](#stage5) and [stage 6](#stage6) are where the
  per-star scatter is measured. An offset is unmeasurable when there is no xcorr peak within 1.5″,
  which is what a grossly mis-registered product looks like. If only the MAST offset is
  unmeasurable, the comparison is flagged **unavailable** (possibly a MAST mis-registration; it
  says nothing about our product, so the stage does not fail on it alone). If the jicama offset is
  unmeasurable, that is our product failing, so the stage does not pass and is red-flagged.

**Source:** [`data_qa/diagnostics.py` → `stage7_mast_vs_pipeline`](../data_qa/diagnostics.py)
(MAST catalogue: `_mast_l3_catalog`/`_load_mast_catalog`; detection fallback: `_detect_on_mosaic`;
offset from VIRAC: `_offset_cloud`).



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
