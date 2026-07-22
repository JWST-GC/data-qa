# GC Treasury operations work plan — monitor → pipeline → images → publish

Status: living document. Authored 2026-07-21 (session `fable-reviewer`).
Owner: @keflavich. Implementation PRs land on `ops-monitor*` branches here.

## Why here

`data-qa` already owns the observation registry (`data_qa/observations.py`),
the per-observation QA issues (`data_qa/make_issues.py`), a MAST download
wrapper (`data_qa/retrieve_data.py`), and a README TODO asking for exactly
this monitor. The pipeline repos keep their own entry points; this repo grows
an `ops` layer that *calls* them:

- **jwst-gc-pipeline** — reduction (`PipelineRerunNIRCAM-LONG.py`,
  `scripts/reduction/submit_*.sbatch`), cataloging
  (`submit_cataloging_chain.sh`), release gates
  (`scripts/release/stage_release.py`), HiPS internals
  (`jwst_gc_pipeline.cmz.hips`: incremental mono-HiPS +
  `G=0.5*(R+B)` two-color derivation), web page (`make_webpage.py`).
- **jwst_scripts** — RGB/AVM production conventions (`jwst_rgb.save_rgb`,
  `faithful_avm` CDMatrix fix for the PA≈90° pyavm degeneracy, pseudo-green
  precedent in `gc2211_rgb_images.py`).
- **avm_images** (= `/orange/adamginsburg/web/public/avm_images`, a git repo
  inside the web-public tree) — where published AVM images + HiPS live;
  pushed with `rsync … starformation:…/htdocs/avm_images/`.

The ops code runs ON HiPerGator (needs /orange, sbatch, `~/.mast_api_token`,
the `starformation` ssh alias); GitHub Actions only does the issue-sync layer
that already exists.

## Pipeline of pipelines

```
 (1) MONITOR          (2) TRIGGER                (3) IMAGES                (4) PUBLISH
 scrontab, ~daily     on new-data event          on release-gate green     gated, manual --execute
 ┌──────────────┐     ┌───────────────────┐      ┌────────────────────┐    ┌──────────────────────┐
 │ MAST poll    │──►──│ download uncals   │──►───│ RGB f212n+f480m    │─►──│ avm_images (+ HiPS)  │
 │ per program  │     │ sbatch reduce     │      │ G=0.5(R+B), AVM    │    │ starformation:/avm_  │
 │ state file   │     │ sbatch catalog    │      │ jwst-gc-treasury-  │    │  images              │
 │              │     │ chain (m1..m7)    │      │  hips (mono+color) │    │ products→ /jwst-gc/  │
 └──────┬───────┘     └───────────────────┘      └────────────────────┘    └──────────────────────┘
        │
        └────────► (5) STATUS: comments on the per-observation data-qa issue at every transition
```

## Components

### 1. `data_qa/mast_monitor.py` — MAST polling + state
- `astroquery.mast Observations.query_criteria(proposal_id=…)` over the
  program list (from the pipeline's program→field map: 2221, 1182, 2211,
  4147, 5365, 3958, 2092, 1939, 1905, 3523, 6778, 7213; configurable).
- State file (`--state`, default `/orange/adamginsburg/jwst/ops/mast_state.json`)
  records known `obs_id` + `t_max`/release date + calib level; a run reports
  NEW or NEWLY-RELEASED observations and exits 0/emits JSON events.
- Actions per new event (each individually gated):
  `--download` (delegates to `retrieve_data.py` / the reduction's own
  downloader), `--trigger` (calls `pipeline_trigger.py`), `--report`
  (comment on the data-qa issue; creates it via `make_issues` conventions
  if absent). Default = report-only dry-run print.
- Deployment: **scrontab** entry (template in `docs/scrontab.example`),
  daily; SLURM conventions `astronomy-dept-b`.

### 2. `data_qa/pipeline_trigger.py` — reduction + cataloging submission
- Maps program/obs → field/target/filters (mirrors
  `PipelineRerunNIRCAM-LONG.py:1637` map; single source imported at runtime
  when the pipeline repo is available, vendored fallback table otherwise).
- Emits the exact submission sequence, respecting repo conventions
  (`--account=astronomy-dept --qos=astronomy-dept-b`, job names
  `<target><program>-o<obsid>-<stage>[-FILTER]` at submit time,
  reduce array → `submit_cataloging_chain.sh` with `DEP=<jobid>`).
- `--dry-run` (default) prints; `--execute` submits via sbatch.
- NEVER bypasses the versioning tag guard or astrometry checkpoints.

### 3. `data_qa/rgb_treasury.py` — F212N+F480M two-color RGB
- Per field: `B=asinh(F212N)`, `R=asinh(long)` (F480M where it exists —
  sgrc/sgrb2/sickle — else F405N), **artificial green `G=0.5*(R+B)`**
  (same formula as `cmz.hips.two_color_tile`).
- Long band reprojected onto the F212N i2d grid; global (not per-tile)
  stretch limits; NaN→alpha; PNG + progressive JPG.
- AVM embedded via the **CDMatrix form** (the `faithful_avm` fix — the
  Scale+Rotation AVM form is degenerate at the JWST GC roll PA≈90°).
- **Validation** (the "validated!" requirement) built in, not optional:
  `--validate` re-reads the AVM from the written PNG, checks CD-matrix and
  reference-pixel round-trip vs the source FITS WCS (tolerance mas-level at
  the reference pixel + corner check), checks alpha/NaN consistency, and
  writes a `<name>.validation.json` verdict. `publish.py` refuses an image
  without a passing verdict.

### 4. `data_qa/hips_treasury.py` — the `jwst-gc-treasury-hips`
- New master trees (distinct from the existing avm_images
  `jwst_cmz_hips` coadd and from the release `CMZ_color`):
  `<root>/jwst-gc-treasury-hips/{F212N,LONG,color}` with
  `<root>` default `/orange/adamginsburg/web/public/avm_images/`.
- Built with `jwst_gc_pipeline.cmz.hips`: `add_field_to_mono_hips`
  per field (incremental — new observations fold in without full rebuild;
  `members.json` records provenance), then `derive_two_color_hips` for the
  color tree. Spec-driven (JSON listing per-field `f212n_i2d`, `long_i2d`,
  `long_band`).
- Compute-heavy backfill runs as SLURM job
  (`docs/submit_treasury_hips.sbatch` template wrapping the CLI).

### 5. `data_qa/publish.py` — gated pushes to starformation
- Targets (from the established manual commands + `make_webpage.py` docs):
  - AVM images/HiPS → `starformation:/h/cnswww-starformation.astro/starformation.astro.ufl.edu/htdocs/avm_images/…`
  - Release products/webpage → `…/htdocs/jwst-gc/…`
- `rsync -ravpu --partial` via the `starformation` ssh alias.
- Hard gates: `--execute` required (default prints the rsync command);
  AVM images require the validation verdict file; product pushes require
  `stage_release.py` gates green (checks for the staged-release marker,
  refuses otherwise). Every executed push logs a manifest of what went up.
- After a product push, regenerates/pushes the `{field}_images.txt` /
  `{field}_catalogs.txt` manifests that `data_qa.observations` consumes —
  closing the loop so new products auto-appear in QA issues.

### 6. `data_qa/status_report.py` — pipeline status → QA issues
- Collects: `squeue` jobs matching the naming convention (per field/program),
  latest m-stage markers + astrometry-checkpoint results from logs, release
  gate state, monitor state-file summary.
- Renders a compact markdown block and posts it as a **comment** on the
  per-observation issue (never touches the autogen body; reuses
  `make_issues` title-lookup + labels). `--dry-run` prints.
- A `<!-- data-qa:status -->` marker + timestamp header per comment; optional
  `--update-last` edits the bot's own previous comment instead of stacking.

### 7. Repo fix (bundled): loud manifest-fetch failures
- `observations.py` `_fetch_lines` returning `[]` on ANY failure made the
  weekly sync silently no-op (live issue #4 shows a stale render). Fetch
  errors now print to stderr and (in `make_issues`) abort the sync rather
  than "sync" an empty registry.

## Testing policy (sandbox)

- All CLIs default to dry-run; `--execute` everywhere for side effects.
- Issue-posting tested against ONE throwaway issue labeled `test`,
  title `TEST — ops infrastructure (throwaway)`, closed afterwards.
  No test comments on real observation issues.
- No rsync to starformation in tests (not even `rsync -n`).
- No sbatch submissions in tests; `--dry-run` output inspected instead.
- MAST queries are read-only and OK to exercise live.

## Rollout order

1. PR-1 (this plan + monitor + trigger + status + manifest-fix)  ← branch `ops-monitor`
2. PR-2 (rgb_treasury + hips_treasury + publish)                 ← branch `ops-imaging`
3. Throwaway-issue infra test; then scrontab entry on HiPerGator (manual step, documented).
4. First supervised end-to-end: next new GC observation → monitor detects →
   human reviews dry-run output → `--execute` each stage once → tighten.
5. Backfill `jwst-gc-treasury-hips` field by field (SLURM), validate, push.

## Open questions — RESOLVED (user decisions, 2026-07-22)

1. **Auto-submit vs approval gate: AUTO.** The monitor downloads + triggers the
   pipeline automatically on new data (`mast_monitor --auto` = `--download
   --trigger --report --commit-state --execute`). The ONLY gate is available
   file space: `--min-free-tb` (default 5.0 TB) checked against the
   `--download-dir` filesystem; below threshold the run downgrades to
   report-only with a loud LOW DISK warning on the QA issue.
2. **Treasury HiPS = program 10678 ONLY.** 10678 is the GC Treasury program
   (MAST: 1668 planned observations, GC_<n> tile targets, NIRCam F212N;F480M +
   MIRI F770W; everything calib_level −1 / unreleased as of 2026-07-22). It is
   the monitor's priority watch (`mast_monitor.TREASURY_PROGRAM`, field label
   `gc-treasury`, tile name carried on events). Pre-treasury fields (sgrb2
   5365, sgrc 4147, sickle 3958, …) stay OUT of the treasury HiPS — they
   belong to the existing pre-treasury CMZ products
   (`docs/cmz_pretreasury_spec.example.json` on the imaging branch; sickle may
   use F210M as its blue band there).
3. **Treasury root/publish layout as staged: APPROVED**
   (`/orange/adamginsburg/jwst/treasury_hips/...` staging root, published via
   `publish avm` to `htdocs/avm_images/`).
4. **avm push validation bar: BOTH.** AVM/WCS round-trip AND the
   star-position/catalog-flux check (`rgb_treasury --validate-stars`) as a
   second gate; skipped star checks need an explicit `--no-star-check`
   acknowledgment at publish time, failed ones block outright.
5. **Cataloging trigger default: plain crf (no destreak).** `EACH_SUFFIX`
   defaults to `align_o<obs>_crf` (fix_alignment always runs; the no-destreak
   path names per-exposure crfs `*_align_o<field>_crf.fits`). Destreak stays
   available via `pipeline_trigger --destreak`; the QA checklist now ASKS
   whether destreak is needed per observation.
6. **MIRI 2221 o003: ignored** (left unmapped in `PROGRAMS`; no globulars.)
7. **Ops stays in data-qa** (no split into its own org repo).
