# Proposal for diffuse-emission validation in Stage 15

Stage 15 already makes tool differences visible with cutouts and per-exposure figures. Its standard panels compare a JICAMA mosaic that retains the smooth sky with a single-exposure PEPPAR residual after background subtraction. The RMS and MAD in those panels therefore describe different sky content and noise. A quantitative diffuse-emission comparison should start by putting each fitted stellar model on identical calibrated inputs. [Stage 15 methods](qa_methods.md#stage-15--residual-maps-by-photometry-tool)

## Give each method the same sky and coverage

- Use the same calibrated exposures, WCS generation, masks, and coaddition for each method. Subtract the stellar model from that common SCI plane. When using a residual and its fitted background instead, verify the reconstruction against the original exposure and record the treatment and units. Use the common measurable footprint for paired scores, and report each method's coverage separately.
- Record the source-model generation, calibration and WCS identifiers, saturation rules, and input ledger or manifest hash beside each score. Compare like-for-like exposure splits and matched cutouts. The existing as-written panels still help diagnose native tool products.
- Use o065/o072/o079 for seams, o078/o100/o105 for bright blends, and o102/o109 for saturated cores and overlap controls. A matched PEPPAR/JICAMA comparison should grade both against sky recovery and stellar residuals. A smaller residual RMS by itself can reward removal of real diffuse emission. [Cataloging disagreement cases](https://github.com/JWST-GC/data-qa/issues/382) · [Photometry background-bias report](https://github.com/JWST-GC/cataloging-pre-treasury/blob/main/reports/report_13_background_subtraction_bias.md)

## Measure recovery as well as residuals

| Diagnostic | Proposed measurement |
| --- | --- |
| Diffuse transfer | Inject filaments of several widths and both detector-row orientations, broad gradients, and compact knots into the calibrated frames. Run matched injected and uninjected reductions with refitting enabled. Report signed contrast gain and aperture flux recovery by field, morphology and brightness. |
| Seams | On held-out overlap blocks or connected edges, report the signed brightness difference, median absolute difference and upper tail in MJy/sr. Record when an overlap or offset component is unmeasurable. |
| Striping | Measure directional power on split exposure coadds and show stratified results for single and multiple coverage. Use roll diversity when available to distinguish fixed sky structure from detector artifacts. |
| Stars and coverage | Report signed core and wing residual flux by brightness and saturation class, masked area and the fraction of pixels measured in another dither. Report stacked or aperture-scale noise alongside pixel statistics. |

The F480M calibration-level pilot showed that the two tested destriping windows suppress row-parallel emission. The 32 px filament kept 11% of its contrast with the 2048-row filter and 54% with the 256-row filter. The perpendicular filament median was 98% with the 2048-row filter and 100% with the 256-row filter, rounded to whole percentages. The proposed Stage 15 test measures whether the gain from skipping that filter survives the rest of the reduction. Add a nested `diffuse_recovery` object to the Stage 15 metrics returned by `data_qa/diagnostics.py::stage15_residual_comparison`, alongside `tools_present` and `tools_missing`. Store the per-template scores, common footprint fraction and input manifest hash there so the comparison can be rerun. [Injection code and per-observation results](https://github.com/keflavich/jwst-gc-pipeline/blob/a69e107e3031b09e3c216e9c4536aca593d243d8/docs/evidence/f480m_diffuse_pilot/README.md#paired-injections)

*Prepared by GPT-6.2 Sol (Codex).*
