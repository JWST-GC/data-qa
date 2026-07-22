# JWST-GC data QA

Data-quality assessment and issue tracking for the JWST Galactic Center surveys.

**This repository is for reporting and discussing data-quality issues on individual
datasets — one GitHub issue per observation.**

- track QA status per observation (astrometry, photometry, mosaics, catalogs),
- report problems tied to specific data products, with links, and
- discuss and resolve them.

## How it works

- Each JWST observation (e.g. `jw02221-o001`, `jw01182-o004`) gets **one tracking
  issue**, auto-created from a template and pre-filled with the observation's
  metadata and links to its data products.
- Issues are created by `data_qa/make_issues.py`. The observation list is
  **discovered from the public release itself** (`data_qa/observations.py` parses
  each field's `{field}_images.txt` manifest), so an observation gets a tracking
  issue as soon as its mosaics are published. Creation is **idempotent** (keyed on
  the issue title) so re-running never duplicates; metadata is synced into the
  existing issue body.
- A GitHub Action (`.github/workflows/make-issues.yml`) runs the same script on
  demand / on a schedule, so when new products are produced and registered, their
  issues appear automatically.

## Layout

```
data_qa/
  observations.py   # registry: the observations + their metadata (single source of truth)
  make_issues.py    # render the filled QA template per observation, create/update GitHub issues
  retrieve_data.py  # MAST retrieval of the underlying JWST products (astroquery)
.github/
  ISSUE_TEMPLATE/observation-qa.md   # the per-observation QA template (also usable manually)
  workflows/make-issues.yml          # automation
```

## Usage

Retrieve data for an observation:

```bash
python -m data_qa.retrieve_data --program 2221 --obs 001 --download-dir ./data
```

Create/refresh the tracking issues (needs a `GITHUB_TOKEN` with `issues:write`):

```bash
python -m data_qa.make_issues --program 2221 1182 --target Brick        # create
python -m data_qa.make_issues --program 2221 1182 --target Brick --dry-run   # preview
```

## Adding a field / observation

Observations are discovered automatically from the release manifests. To bring a
newly released **field** in, add it to `FIELDS` in `data_qa/observations.py`; its
observations are parsed from `{field}_images.txt` on the next run. To attach hand
notes / epoch / visits to a specific observation, add an entry to `CURATED`
(keyed by obsid, e.g. `jw02221-o001`).

## Imaging + publishing (ops)

The imaging/publish stage of the ops layer lives in three modules:

```
data_qa/
  rgb_treasury.py   # two-band -> three-color RGB PNG/JPG with embedded AVM + validation verdict
  hips_treasury.py  # spec-driven two-color treasury HiPS builder (plan / build / color / sbatch)
  publish.py        # gated rsync pushes to the web host (avm / products / manifests verbs)
```

- `rgb_treasury.py` composes the CMZ house two-color scheme (B=F212N,
  R=F480M/F405N, G=0.5(R+B)) into an AVM-tagged PNG (+ JPG preview) and always
  writes a `<out>.validation.json` verdict: WCS round-trip, alpha/NaN
  consistency, a reference-catalog star-position check, and
  `outputs.{png,jpg}_sha256` hashes binding the verdict to the exact files it
  validated.
- `hips_treasury.py` drives `jwst_gc_pipeline.cmz.hips` from a JSON spec to
  grow the incremental mono HiPS substrates and derive the two-color layer.
- `publish.py` is the only sanctioned path to the web server. Every verb is
  dry-run by default; `--execute` pushes only if the gates pass (fail-closed,
  no `--force`): images need a passing, hash-bound validation verdict (a
  regenerated file with a stale verdict is refused), the push manifest is
  checked against the gate's scope (`--allow-extra-files` for uncovered
  files), and release products need the `stage_release.py` MANIFEST marker.

## TODO

We still need to set up a cron job to poll the MAST archive, trigger downloads, and trigger job runs and issue creation when new data are delivered.
