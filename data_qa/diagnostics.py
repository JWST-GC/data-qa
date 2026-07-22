"""Progressive QA diagnostic figures, posted as replies (comments) to a per-observation
tracking issue.

Four stages, each emitted as the corresponding data product becomes available while the
cataloging pipeline runs.  Each stage returns ``(png_path, metrics)``; the metrics drive
the checkbox state in the issue body (see ``make_issues.render_body``), and the PNG is
posted as an idempotent comment (one comment per stage, keyed on a hidden marker).

    Stage 1  first i2d       one SW + one LW grayscale mosaic       "delivered", "mosaics present"
    Stage 2  CMD             LW vs SW-LW colour-magnitude + LF      "catalog vetted", "depth"
    Stage 3  calibration     JWST (F212N-like) vs VIRAC Ks         "photometry zeropoints"
    Stage 4  offsets         JWST-VIRAC dRA/dDec + inter-module    "absolute frame", "inter-module"

Images LIVE IN THE ISSUE (posted to the GitHub CDN as release assets on a single
``qa-assets`` bucket release, then embedded in the comment) -- NOT committed to the repo
source tree.  Reuses the reference-free / crowding-proof machinery in
``astrometry_audit`` (detect / xcorr / direct_intermodule / load_reference).

Usage:
    python -m data_qa.diagnostics --program 5365 --obs 001 --stage 1 2 3 4          # build only
    python -m data_qa.diagnostics --program 5365 --obs 001 --stage 1 --post         # build + post
    python -m data_qa.diagnostics --program 2221 --obs 001 --sw F212N --lw F410M --post
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from dataclasses import replace

import numpy as np

from . import astrometry_audit as aa
from .observations import Observation, registry

BASE = os.environ.get("QA_BASE", "/orange/adamginsburg/jwst")
OUTDIR = os.environ.get("QA_OUTDIR", "/tmp/data_qa_figures")

# Comment idempotency markers (one comment per stage per issue).
DIAG_MARKER = "<!-- data-qa:diag:stage{n} -->"

# Filter selection: prefer the requested filters; else nearest SW / LW.
_SW_PREF = ["F212N", "F200W", "F187N", "F182M", "F162M", "F150W", "F115W"]
_LW_PREF = ["F480M", "F470N", "F466N", "F444W", "F410M", "F405N", "F360M", "F356W", "F323N", "F300M", "F277W"]


def _channel(filt):
    return "SW" if int(filt[1:4]) <= 212 else "LW"


def pick_filters(available, sw=None, lw=None):
    """Choose one SW + one LW filter from those available for the obs."""
    up = {f.upper() for f in available}
    sw = sw.upper() if sw else next((f for f in _SW_PREF if f in up), None)
    lw = lw.upper() if lw else next((f for f in _LW_PREF if f in up), None)
    return sw, lw


# --------------------------------------------------------------------------- product lookup
def _mosaic_path(o: Observation, filt):
    """Released merged i2d for this obs+filter, or None."""
    pats = [
        f"{BASE}/{o.field}/{filt}/pipeline/{o.obsid}_t001_nircam_clear-{filt.lower()}-merged_i2d.fits",
        f"{BASE}/{o.field}/*/pipeline/{o.obsid}_t001_nircam_clear-{filt.lower()}-merged_i2d.fits",
    ]
    for pat in pats:
        hits = sorted(glob.glob(pat))
        if hits:
            return hits[-1]
    return None


_KIND_RE = re.compile(r"(m8_dedup|m8|m7|abfix)", re.I)


def _catalog_candidates(o: Observation):
    """All merged catalogs for the field.  Naming is inconsistent across fields (m7/m8 tags
    for some, other suffixes elsewhere), so glob EVERY catalog; the caller filters by column
    presence and picks the largest.  Skip obvious residual/model/region sidecars."""
    out = []
    for p in sorted(glob.glob(f"{BASE}/{o.field}/catalogs/*.fits")):
        low = os.path.basename(p).lower()
        if any(s in low for s in ("_residual", "_model", "_reproject", "region")):
            continue
        m = _KIND_RE.search(low)
        out.append((p, m.group(1).lower() if m else "merged"))
    return out


def _catalog_for(o: Observation, sw, lw):
    """Catalog that actually contains VEGA mags for both requested filters (correct obs).
    Among all candidates that have both columns, pick the LARGEST by row count -- the full
    field merge, not a small curated subset (e.g. brick's 83-star dual-excess list).
    Returns (path, kind, sw_col, lw_col) or (None,...)."""
    from astropy.io import fits
    best = (None, None, None, None, -1)
    for p, tag in _catalog_candidates(o):
        try:
            hdr = fits.getheader(p, ext=1)             # header only -- cheap, no data read
        except (OSError, IndexError):
            continue
        ncol = hdr.get("TFIELDS", 0)
        low = {str(hdr.get(f"TTYPE{i}", "")).lower(): hdr[f"TTYPE{i}"]
               for i in range(1, ncol + 1) if hdr.get(f"TTYPE{i}")}
        csw = next((low[k] for k in (f"mag_vega_{sw.lower()}", f"mag_{sw.lower()}",
                                     f"mag_ab_{sw.lower()}") if k in low), None)
        clw = next((low[k] for k in (f"mag_vega_{lw.lower()}", f"mag_{lw.lower()}",
                                     f"mag_ab_{lw.lower()}") if k in low), None)
        if not (csw and clw):
            continue
        nrow = hdr.get("NAXIS2", 0)
        if nrow > best[-1]:
            best = (p, tag, csw, clw, nrow)
    return best[:4]


def _refcat_path(o: Observation):
    """VIRAC2-Gaia refcat (newest epoch) for the absolute-frame (position-only) check."""
    hits = sorted(glob.glob(f"{BASE}/{o.field}/catalogs/gaia_virac2_refcat_epoch*.fits"))
    return hits[-1] if hits else None


def _viraccache_path(o: Observation):
    """Raw VIRAC2 cache (has a real Ksmag column) for the photometric-calibration check.
    The gaia_virac2 refcat carries only a blended 'refmag', unusable for a Ks zeropoint."""
    p = f"{BASE}/{o.field}/astrometry_diag/refcache/virac2.fits"
    return p if os.path.exists(p) else None


# --------------------------------------------------------------------------- figure helpers
def _fig(nrows=1, ncols=1, w=5.0, h=5.0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt.subplots(nrows, ncols, figsize=(w * ncols, h * nrows), squeeze=False)


def _grayscale(ax, path, title):
    from astropy.io import fits
    from astropy.visualization import ZScaleInterval, ImageNormalize, AsinhStretch
    from astropy.wcs import WCS
    with fits.open(path) as hdul:
        sci = hdul["SCI"] if "SCI" in hdul else hdul[1]
        data = sci.data.astype("float32")
    norm = ImageNormalize(data, interval=ZScaleInterval(), stretch=AsinhStretch())
    ax.imshow(data, origin="lower", cmap="gray", norm=norm)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    frac = float(np.isfinite(data).mean())
    return frac


def _save(fig, name):
    os.makedirs(OUTDIR, exist_ok=True)
    out = os.path.join(OUTDIR, name)
    fig.savefig(out, dpi=110, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)
    return out


# --------------------------------------------------------------------------- STAGE 1
def stage1_mosaics(o: Observation, sw, lw):
    """Grayscale SW + LW mosaics -- confirms the data arrived and looks sane."""
    psw, plw = _mosaic_path(o, sw), _mosaic_path(o, lw)
    fig, ax = _fig(1, 2, 5.2, 5.2)
    fracs = {}
    for a, filt, p in ((ax[0][0], sw, psw), (ax[0][1], lw, plw)):
        if p:
            fracs[filt] = _grayscale(a, p, f"{o.obsid}  {filt}")
        else:
            a.text(0.5, 0.5, f"{filt}\n(no i2d)", ha="center", va="center")
            a.set_xticks([]); a.set_yticks([])
    fig.suptitle(f"{o.target} {o.obsid} — first mosaics ({sw} / {lw})", fontsize=11)
    png = _save(fig, f"{o.obsid}_stage1.png")
    metrics = dict(stage=1, sw=sw, lw=lw,
                   sw_present=bool(psw), lw_present=bool(plw),
                   finite_fraction=fracs,
                   passed=bool(psw and plw))
    return png, metrics


# --------------------------------------------------------------------------- STAGE 2
def _mag_cols(t, sw, lw):
    """Locate SW/LW magnitude columns in a merged catalog (several naming schemes).
    ALWAYS prefer VEGA magnitudes over AB (survey convention)."""
    cols = {c.lower(): c for c in t.colnames}
    def find(filt):
        f = filt.lower()
        for pat in (f"mag_vega_{f}", f"mag_{f}", f"{f}_mag", f"mag_ab_{f}", f"{f}"):
            if pat in cols:
                return cols[pat]
        return None
    return find(sw), find(lw)


def stage2_cmd(o: Observation, sw, lw):
    """Colour-magnitude diagram (LW vs SW-LW) + luminosity-function inset."""
    from astropy.table import Table
    cat, kind, csw, clw = _catalog_for(o, sw, lw)
    fig, ax = _fig(1, 1, 5.5, 6.0)
    metrics = dict(stage=2, catalog=os.path.basename(cat) if cat else None, kind=kind)
    if not cat:
        ax[0][0].text(0.5, 0.5, f"no catalog with {sw}+{lw} mags yet", ha="center", va="center")
        metrics["passed"] = False
        return _save(fig, f"{o.obsid}_stage2.png"), metrics
    t = Table.read(cat)
    a = ax[0][0]
    if csw and clw:
        msw = np.asarray(t[csw], float); mlw = np.asarray(t[clw], float)
        g = np.isfinite(msw) & np.isfinite(mlw)
        color = msw[g] - mlw[g]
        a.hexbin(color, mlw[g], gridsize=120, bins="log", cmap="viridis", mincnt=1)
        a.set_xlabel(f"{sw} - {lw}"); a.set_ylabel(lw)
        a.invert_yaxis()
        a.set_xlim(np.nanpercentile(color, [1, 99]))
        # LF inset (depth): where do counts turn over
        ins = a.inset_axes([0.62, 0.62, 0.36, 0.36])
        hh, edges = np.histogram(mlw[g], bins=40)
        ctr = 0.5 * (edges[1:] + edges[:-1])
        ins.step(ctr, hh, where="mid", color="k", lw=0.8)
        peak = ctr[int(np.argmax(hh))]
        ins.axvline(peak, color="r", lw=0.8)
        ins.set_title(f"LF {lw}\nturnover≈{peak:.1f}", fontsize=7)
        ins.tick_params(labelsize=6)
        metrics.update(n_stars=int(g.sum()), lf_turnover=float(peak),
                       sw_col=csw, lw_col=clw, passed=int(g.sum()) > 500)
    else:
        a.text(0.5, 0.5, f"no {sw}/{lw} mag cols\nin {os.path.basename(cat)}",
               ha="center", va="center", fontsize=8)
        metrics["passed"] = False
    fig.suptitle(f"{o.target} {o.obsid} — CMD ({kind})", fontsize=11)
    return _save(fig, f"{o.obsid}_stage2.png"), metrics


# --------------------------------------------------------------------------- STAGE 3
def stage3_calibration(o: Observation, sw):
    """JWST (SW ~ F212N) instrumental mag vs VIRAC Ks for matched stars: a tight linear
    locus proves the RIGHT stars were matched and the photometric zeropoint is sane."""
    import astropy.units as u
    from astropy.coordinates import search_around_sky
    fig, ax = _fig(1, 1, 5.5, 5.5)
    metrics = dict(stage=3, sw=sw)
    path = _mosaic_path(o, sw)
    ref = _viraccache_path(o) or _refcat_path(o)   # cache has real Ksmag
    ep = aa.epoch_of(path) if path else None
    ref_sc, ref_mag = aa.load_reference(ref, ep) if (ref and ep) else (None, None)
    jsc, jmag = aa.detect(path) if path else (None, None)
    a = ax[0][0]
    if jsc is None or ref_sc is None or ref_mag is None:
        a.text(0.5, 0.5, "need mosaic + VIRAC refcat", ha="center", va="center")
        metrics["passed"] = False
        return _save(fig, f"{o.obsid}_stage3.png"), metrics
    ia, ib, sep, _ = search_around_sky(jsc, ref_sc, 0.2 * u.arcsec)
    if len(ia) < 30:
        a.text(0.5, 0.5, f"only {len(ia)} matches", ha="center", va="center")
        metrics["passed"] = False
        return _save(fig, f"{o.obsid}_stage3.png"), metrics
    x = ref_mag[ib]; y = jmag[ia]
    g = np.isfinite(x) & np.isfinite(y)
    x, y = x[g], y[g]
    # robust linear fit y = slope*x + zp
    slope, zp = np.polyfit(x, y, 1)
    resid = y - (slope * x + zp)
    scat = float(aa.mad_std(resid))
    a.hexbin(x, y, gridsize=80, bins="log", cmap="magma", mincnt=1)
    xs = np.array([np.nanmin(x), np.nanmax(x)])
    a.plot(xs, slope * xs + zp, "c-", lw=1, label=f"slope={slope:.2f} zp={zp:.2f}")
    a.set_xlabel("VIRAC Ks [mag]"); a.set_ylabel(f"JWST {sw} instr mag")
    a.legend(fontsize=8, loc="upper left")
    a.set_title(f"{o.obsid} calibration  n={g.sum()} scatter={scat:.2f}", fontsize=10)
    metrics.update(n_matched=int(g.sum()), slope=float(slope), zeropoint=float(zp),
                   scatter=scat, passed=(0.7 < slope < 1.3 and scat < 0.5))
    return _save(fig, f"{o.obsid}_stage3.png"), metrics


# --------------------------------------------------------------------------- STAGE 4
def stage4_offsets(o: Observation, sw):
    """JWST-VIRAC per-star dRA/dDec across the field (frame tie / PM precursor) + the
    reference-free inter-module (NRCA vs NRCB) offset."""
    import astropy.units as u
    from astropy.coordinates import search_around_sky
    fig, ax = _fig(1, 2, 5.4, 5.0)
    metrics = dict(stage=4, sw=sw)
    path = _mosaic_path(o, sw)
    ref = _refcat_path(o)
    ep = aa.epoch_of(path) if path else None
    ref_sc, _ = aa.load_reference(ref, ep) if (ref and ep) else (None, None)
    jsc, _ = aa.detect(path) if path else (None, None)
    a0, a1 = ax[0][0], ax[0][1]
    if jsc is not None and ref_sc is not None:
        bulk = aa.xcorr(jsc, ref_sc)
        ia, ib, sep, _ = search_around_sky(jsc, ref_sc, 0.3 * u.arcsec)
        if len(ia) >= 30:
            dra = (ref_sc[ib].ra - jsc[ia].ra).to(u.arcsec).value * np.cos(np.radians(jsc[ia].dec.value)) * 1000
            dde = (ref_sc[ib].dec - jsc[ia].dec).to(u.arcsec).value * 1000
            a0.hexbin(dra, dde, gridsize=60, bins="log", cmap="cividis", mincnt=1)
            a0.axhline(0, color="w", lw=0.5); a0.axvline(0, color="w", lw=0.5)
            a0.set_xlabel("dRA [mas]"); a0.set_ylabel("dDec [mas]")
            # window shows BOTH the origin and the bulk cloud (off-frame fields sit far off 0)
            lim = max(100.0, 1.4 * (abs(bulk["off"]) if bulk else 0.0))
            a0.set_xlim(-lim, lim); a0.set_ylim(-lim, lim)
            a0.set_title(f"JWST-VIRAC  bulk={bulk['off']:.0f} mas" if bulk else "JWST-VIRAC", fontsize=9)
            metrics.update(bulk_off=float(bulk["off"]) if bulk else None,
                           bulk_dra=float(bulk["dra"]) if bulk else None,
                           bulk_ddec=float(bulk["ddec"]) if bulk else None,
                           n_matched=int(len(ia)))
    else:
        a0.text(0.5, 0.5, "need mosaic + VIRAC", ha="center", va="center")
    # inter-module
    mos = aa.find_mosaics(o.field)
    im = None
    for filt in (sw, o.filters[0] if o.filters else sw):
        mods = mos.get(filt.upper(), {})
        A = aa.detect(mods["nrca"])[0] if "nrca" in mods else (aa.detect(mods["nrcalong"])[0] if "nrcalong" in mods else None)
        B = aa.detect(mods["nrcb"])[0] if "nrcb" in mods else (aa.detect(mods["nrcblong"])[0] if "nrcblong" in mods else None)
        if A is not None and B is not None:
            im = aa.direct_intermodule(A, B)
            if im:
                a1.bar(["dRA", "dDec"], [im["dra"], im["ddec"]], color=["#4477aa", "#ee6677"])
                a1.axhline(0, color="k", lw=0.5)
                a1.axhline(aa.THRESH["intermodule"], color="r", ls=":", lw=0.8)
                a1.axhline(-aa.THRESH["intermodule"], color="r", ls=":", lw=0.8)
                a1.set_ylabel("NRCA-NRCB [mas]")
                a1.set_title(f"inter-module {filt}  off={im['off']:.0f} mas", fontsize=9)
                metrics.update(intermodule_off=float(im["off"]), intermodule_filt=filt)
            break
    if im is None:
        a1.text(0.5, 0.5, "no per-module mosaics", ha="center", va="center", fontsize=8)
    bo = metrics.get("bulk_off"); io = metrics.get("intermodule_off")
    metrics["passed"] = bool((bo is not None and bo < aa.THRESH["absolute"]) and
                             (io is None or io < aa.THRESH["intermodule"]))
    fig.suptitle(f"{o.target} {o.obsid} — positional offsets", fontsize=11)
    return _save(fig, f"{o.obsid}_stage4.png"), metrics


STAGES = {1: stage1_mosaics, 2: stage2_cmd, 3: stage3_calibration, 4: stage4_offsets}


def build_stage(o, n, sw, lw):
    if n == 1:
        return stage1_mosaics(o, sw, lw)
    if n == 2:
        return stage2_cmd(o, sw, lw)
    if n == 3:
        return stage3_calibration(o, sw)
    if n == 4:
        return stage4_offsets(o, sw)
    raise ValueError(n)


CAPTIONS = {
    1: "**Stage 1 — first mosaics.** Grayscale {sw} (SW) and {lw} (LW) `i2d`. Confirms the "
       "observation was delivered and the mosaics are present and not obviously corrupt.",
    2: "**Stage 2 — colour-magnitude diagram** from the `{kind}` catalog ({n_stars} stars). "
       "LF-inset turnover ≈ {lf_turnover:.1f} tracks depth; regenerated as the catalog deepens.",
    3: "**Stage 3 — photometric calibration.** JWST {sw} vs VIRAC Ks for {n_matched} matched "
       "stars: slope {slope:.2f}, zp {zeropoint:.2f}, scatter {scatter:.2f} mag. A tight locus "
       "means the right stars were matched.",
    4: "**Stage 4 — positional offsets.** JWST−VIRAC ΔRA/ΔDec (bulk {bulk_off:.0f} mas) and the "
       "reference-free inter-module offset. First-order frame-match / proper-motion precursor.",
}


def caption_for(n, metrics):
    try:
        return CAPTIONS[n].format(**{k: (v if v is not None else float("nan"))
                                     for k, v in metrics.items()})
    except (KeyError, ValueError):
        return CAPTIONS[n].split(".")[0] + "."


def _obs_from_disk(program, obs, base=BASE):
    """Fallback registry for on-cluster runs where the release portal is unreachable:
    find the field dir on disk holding this obs's mosaics and read its NIRCam filters."""
    from .observations import CURATED, FIELDS
    for d in sorted(glob.glob(f"{base}/*/")):
        fld = os.path.basename(d.rstrip("/"))
        hits = glob.glob(f"{base}/{fld}/*/pipeline/jw{int(program):05d}-o{obs}_t001_nircam_clear-*-merged_i2d.fits")
        if not hits:
            continue
        filts = sorted({m.group(1).upper() for h in hits
                        if (m := re.search(r"clear-(f\d{3}[wnm])-merged", os.path.basename(h).lower()))})
        cur = CURATED.get(f"jw{int(program):05d}-o{obs}", {})
        return Observation(program=str(int(program)), obs=obs,
                           target=FIELDS.get(fld, fld.title()),   # display name -> matches issue title
                           release_field=fld, instrument="NIRCam", filters=filts,
                           visits=cur.get("visits", []), epoch=cur.get("epoch", ""),
                           notes=cur.get("notes", ""))
    return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--program", required=True)
    ap.add_argument("--obs", required=True)
    ap.add_argument("--stage", nargs="+", type=int, default=[1, 2, 3, 4])
    ap.add_argument("--sw", default=None); ap.add_argument("--lw", default=None)
    ap.add_argument("--target", default=None, help="override display target (issue-title match)")
    ap.add_argument("--post", action="store_true", help="post/update the issue comments")
    ap.add_argument("--repo", default=os.environ.get("QA_REPO", "JWST-GC/data-qa"))
    args = ap.parse_args(argv)

    obs = [o for o in registry(programs=[args.program]) if o.obs == args.obs]
    o = obs[0] if obs else _obs_from_disk(args.program, args.obs)
    if o is None:
        print(f"no obs for program {args.program} obs {args.obs} (portal + on-disk both empty)",
              file=sys.stderr)
        return 1
    if args.target:
        o = replace(o, target=args.target)
    sw, lw = pick_filters(o.filters, args.sw, args.lw)
    print(f"{o.obsid}: SW={sw} LW={lw} filters={o.filters}")
    all_metrics = {}
    for n in args.stage:
        png, metrics = build_stage(o, n, sw, lw)
        all_metrics[f"stage{n}"] = metrics
        print(f"  stage {n}: {png}  passed={metrics.get('passed')}")
        if args.post:
            from .post_diagnostics import post_stage
            post_stage(o, n, png, caption_for(n, metrics), args.repo)
    # write metrics json where make_issues.render_body reads it to drive checkbox state.
    # Small text (committable); figures stay in OUTDIR and are hosted on the GitHub CDN.
    mdir = os.path.join(os.path.dirname(__file__), "metrics")
    os.makedirs(mdir, exist_ok=True)
    # merge into any existing metrics so a single-stage run doesn't drop other stages
    mpath = os.path.join(mdir, f"{o.obsid}.json")
    prev = {}
    if os.path.exists(mpath):
        try:
            with open(mpath) as fh:
                prev = json.load(fh)
        except (OSError, ValueError):
            prev = {}
    prev.update(all_metrics)
    with open(mpath, "w") as fh:
        json.dump(prev, fh, indent=2)
    print(f"metrics -> {mpath}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
