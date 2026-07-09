# JWST-GC data QA

Data-quality assessment and issue tracking for the JWST Galactic Center surveys.

**This repository is for reporting and discussing data-quality issues on individual
datasets — one GitHub issue per observation.** It is deliberately *not* a reduction
or analysis repository (that scope creep is what made the ACES `reduction_ACES` repo
hard to navigate). Keep reduction pipelines, science analysis, and catalog code in
their own repos; use this one to:

- track QA status per observation (astrometry, photometry, mosaics, catalogs),
- report problems tied to specific data products, with links, and
- discuss and resolve them.

## How it works

- Each JWST observation (e.g. `jw02221-o001`, `jw01182-o004`) gets **one tracking
  issue**, auto-created from a template and pre-filled with the observation's
  metadata and links to its data products.
- Issues are created by `data_qa/make_issues.py`, driven by the observation
  registry in `data_qa/observations.py`. Creation is **idempotent** (keyed on the
  issue title) so re-running never duplicates; new metadata is synced into the
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

## Adding an observation

Add an entry to `OBSERVATIONS` in `data_qa/observations.py` (or extend the registry
loader). The next `make_issues` run creates its tracking issue.
