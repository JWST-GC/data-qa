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
  mast_monitor.py   # ops: stateful MAST poll -> new/released/calib-up events (+ actions)
  pipeline_trigger.py  # ops: reduction+cataloging SLURM submission sequence (dry-run first)
  status_report.py  # ops: squeue/state/release status -> comment on the QA issue
  _github.py        # shared stdlib GitHub REST helpers (make_issues + status_report)
.github/
  ISSUE_TEMPLATE/observation-qa.md   # the per-observation QA template (also usable manually)
  workflows/make-issues.yml          # automation
docs/
  OPS_WORKPLAN.md      # the ops-layer plan (monitor -> trigger -> images -> publish)
  scrontab.example     # HiPerGator scrontab entries for the daily/weekly monitor
```

## Ops layer

The repo also carries the operations layer described in
[docs/OPS_WORKPLAN.md](docs/OPS_WORKPLAN.md): `mast_monitor.py` polls MAST per
program and diffs against a state file to emit NEW_OBSERVATION / NEWLY_RELEASED /
CALIB_LEVEL_UP events; `pipeline_trigger.py` turns an event into the
jwst-gc-pipeline submission sequence (reduction filter-array then the cataloging
chain, dependency-gated); `status_report.py` collects `squeue`/monitor/release
state and posts it as a marked comment on the observation's QA issue. Every CLI is
dry-run by default and takes `--execute` for real side effects; deploy the monitor
via `scrontab` using [docs/scrontab.example](docs/scrontab.example).

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

## TODO

- [x] Set up a cron job to poll the MAST archive, trigger downloads, and trigger job
  runs and issue creation when new data are delivered — implemented as the ops layer
  (`data_qa/mast_monitor.py` + `pipeline_trigger.py` + `status_report.py`; deploy per
  [docs/scrontab.example](docs/scrontab.example)). Remaining: imaging/publish stages
  (PR-2 in [docs/OPS_WORKPLAN.md](docs/OPS_WORKPLAN.md)).
