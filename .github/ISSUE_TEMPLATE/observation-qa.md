---
name: Observation QA
about: Track and discuss data-quality for one JWST-GC observation
title: "<Target> — jwPPPPP-oOOO (Instrument)"
labels: QA
assignees: ''
---

**Observation `jwPPPPP-oOOO`** — <Target> / <Instrument>

| field | value |
|-------|-------|
| Program | `PPPPP` |
| Observation | `OOO` |
| Target | <Target> |
| Instrument | NIRCam / MIRI |
| Filters | |
| Executions (visits) | |
| Epoch (DATE-OBS) | |

### Data products
- MAST program:
- JWST-GC release:
- On-disk mosaics:

### QA checklist
- [ ] Observation delivered / retrieved
- [ ] Per-filter mosaics (`i2d`) present and complete
- [ ] **Astrometry**: absolute frame tie (VIRAC2/Gaia) within survey noise
- [ ] **Astrometry**: no inter-module (NRCA/NRCB) offset (proper-motion grade)
- [ ] **Photometry**: zeropoints consistent across filters/modules
- [ ] Background / stripes / artifacts acceptable
- [ ] Catalog produced and vetted
- [ ] **Depth**: detection luminosity functions reach the expected depth (not missing stars we should be detecting)
- [ ] **Purity**: minimal junk detections in PSF wings and in extended-emission regions
- [ ] **Residuals**: PSF-subtracted residual histogram is narrow and centered on zero (no systematic over/under-subtraction)
- [ ] Known issues triaged (comment below)

> Most observations get this issue auto-created (pre-filled) by
> `data_qa/make_issues.py`. Use this manual template only for an observation not yet
> in the registry. **Discuss problems in the comments.**
