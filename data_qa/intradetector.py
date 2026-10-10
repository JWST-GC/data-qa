"""Intra-detector distortion residuals: each per-frame catalogue against the consensus catalogue.

A per-frame (``*_m3_daophot_basic.fits``) catalogue carries detector pixel positions
(``x_fit``/``y_fit``) and sky positions through that frame's distortion-corrected WCS
(``skycoord_centroid``).  The consensus catalogue (``<filt>_merged_o<obs>_indivexp_merged_m3_dao_basic_vetted.fits``)
averages every exposure's sky position for a star.  The per-frame sky position minus the consensus
position, plotted against where the star fell on the detector and stacked over many frames, maps
any distortion left over after the WCS distortion model: a frame's residual is a function of the
star's detector position, while the consensus averages each star over several detector positions
(dithers, detectors).

Per frame:

1. keep high-S/N, unflagged frame stars and unsaturated, multiply-measured consensus stars;
2. remove the frame's bulk offset from the consensus with the pair-histogram peak
   (``astrometry_audit.xcorr``), which is crowding-proof (a nearest-neighbour median against a dense
   catalogue is pulled toward zero by chance pairs);
3. mutual-nearest-neighbour match the shifted frame to the consensus within ``MATCH_ARCSEC``;
4. flux-vet the pairs (frame flux vs consensus flux) to drop blends and wrong-star matches;
5. rotate the sky residual into the detector axes with the frame's local Jacobian (a linear fit of
   ``x_fit``/``y_fit`` against the consensus tangent-plane position).  Only the bulk shift is
   removed, so a linear term (plate scale, rotation) left in the WCS stays visible as a gradient.

``stack_maps`` then bins the detector-frame residuals on a fixed grid per detector and reports the
RMS of the binned map against a shuffled-position null; ``sysrms_debiased`` subtracts the null in
quadrature, so pure noise reads ~0.

Two dilutions make the measured map a lower bound on the true residual distortion: the consensus
includes the frame itself (weight ~1/N for N exposures), and a smooth pattern that is common to
every exposure is partly averaged into the consensus when the dithers are small compared with the
pattern's scale.  Both are documented in ``docs/qa_methods.md#stage8-intradetector``.
"""
from __future__ import annotations

import numpy as np

MATCH_ARCSEC = 0.06         # mutual-NN radius after the bulk shift; per-star noise is a few mas
XCORR_MAXSEP_ARCSEC = 0.5   # pair-histogram search radius (frame and consensus share a frame)
MIN_FRAME_SNR = 30.0        # frame flux_fit/flux_err
MIN_REF_SNR = 30.0          # consensus flux/flux_err
MIN_REF_NMATCH = 2          # consensus star measured in at least this many frames
FLUX_TOL_MAG = 0.2          # |Δmag - median Δmag| kept by the flux vet
MIN_PEAK_RATIO = 4.0        # histogram peak / background; below this the bulk tie is ambiguous
MIN_PAIRS = 50              # matched pairs needed to keep a frame
DET_NPIX = 2048             # NIRCam SW and LW detectors are 2048 x 2048 (incl. reference pixels)
NBIN = 8                    # detector grid for the stacked map (256-px cells)
MIN_PER_CELL = 20


def load_frame(path):
    """Columns needed from a per-frame daophot catalogue, plus PIXSCALE and DETECTOR from the
    header.  Returns a dict of arrays (flagged / low-S/N rows already dropped)."""
    from astropy.io import fits
    with fits.open(path, memmap=False) as h:
        d = h[1].data
        hdr = h[1].header
        x = np.asarray(d["x_fit"], float); y = np.asarray(d["y_fit"], float)
        ra = np.asarray(d["skycoord_centroid.ra"], float)
        dec = np.asarray(d["skycoord_centroid.dec"], float)
        flux = np.asarray(d["flux_fit"], float); ferr = np.asarray(d["flux_err"], float)
        flags = np.asarray(d["flags"], int)
        qfit = np.asarray(d["qfit"], float) if "qfit" in d.names else np.zeros(len(x))
    with np.errstate(invalid="ignore", divide="ignore"):
        snr = flux / ferr
    # flags == 0: photutils sets bit 1 when pixels of the fit box are masked (saturated/bad
    # neighbours), so unflagged rows exclude saturated stars and their surroundings.
    ok = (np.isfinite(x) & np.isfinite(y) & np.isfinite(ra) & np.isfinite(dec) & (flags == 0)
          & (snr > MIN_FRAME_SNR) & ~(qfit > 0.2))
    return dict(x=x[ok], y=y[ok], ra=ra[ok], dec=dec[ok], flux=flux[ok],
                pixscale=float(hdr.get("PIXSCALE", np.nan)),
                detector=str(hdr.get("DETECTOR", "")).lower())


def load_consensus(path):
    """Unsaturated, multiply-measured, high-S/N consensus stars: dict(ra, dec, flux)."""
    from astropy.io import fits
    with fits.open(path, memmap=False) as h:
        d = h[1].data
        ra = np.asarray(d["skycoord.ra"], float); dec = np.asarray(d["skycoord.dec"], float)
        flux = np.asarray(d["flux"], float); ferr = np.asarray(d["flux_err"], float)
        ok = np.isfinite(ra) & np.isfinite(dec)
        if "is_saturated" in d.names:
            ok &= ~np.asarray(d["is_saturated"], bool)
        if "nmatch_good" in d.names:
            ok &= np.asarray(d["nmatch_good"], float) >= MIN_REF_NMATCH
    with np.errstate(invalid="ignore", divide="ignore"):
        ok &= flux / ferr > MIN_REF_SNR
    return dict(ra=ra[ok], dec=dec[ok], flux=flux[ok])


def _mutual_nn(a, b, radius_arcsec):
    import astropy.units as u
    idx, sep, _ = a.match_to_catalog_sky(b)
    back, _, _ = b.match_to_catalog_sky(a)
    ok = (sep < radius_arcsec * u.arcsec) & (back[idx] == np.arange(len(a)))
    return np.nonzero(ok)[0], idx[ok]


def frame_residuals(frame, ref, match_arcsec=MATCH_ARCSEC, flux_tol=FLUX_TOL_MAG):
    """Detector-frame residual (frame − consensus, bulk removed) of one frame.

    ``frame`` is a ``load_frame`` dict (x, y, ra, dec, flux, pixscale); ``ref`` a
    ``load_consensus`` dict.  Returns dict(x, y, dx, dy [mas, detector axes], bulk_dra,
    bulk_ddec [mas], npairs, peak_ratio) or dict(reason=...) when the frame cannot be measured."""
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    from . import astrometry_audit as aa
    if len(frame["x"]) < MIN_PAIRS or len(ref["ra"]) < MIN_PAIRS:
        return dict(reason="too few stars")
    ra0 = float(np.median(frame["ra"])); dec0 = float(np.median(frame["dec"]))
    cosd = float(np.cos(np.radians(dec0)))
    # consensus stars inside the frame footprint (+2"): keeps the matches fast on a dense catalogue
    pad = 2.0 / 3600.0
    near = ((ref["ra"] > frame["ra"].min() - pad / cosd) & (ref["ra"] < frame["ra"].max() + pad / cosd)
            & (ref["dec"] > frame["dec"].min() - pad) & (ref["dec"] < frame["dec"].max() + pad))
    rra, rde, rfl = ref["ra"][near], ref["dec"][near], ref["flux"][near]
    if len(rra) < MIN_PAIRS:
        return dict(reason="consensus does not cover the frame")
    fsc = SkyCoord(frame["ra"] * u.deg, frame["dec"] * u.deg)
    rsc = SkyCoord(rra * u.deg, rde * u.deg)
    xc = aa.xcorr(fsc, rsc, maxsep=XCORR_MAXSEP_ARCSEC * u.arcsec, binarc=0.02)
    if not xc or xc.get("peak_ratio", 0) < MIN_PEAK_RATIO:
        return dict(reason="ambiguous bulk offset (histogram peak)")
    # xcorr(a=frame, b=consensus) is the shift that moves the frame onto the consensus
    sra = frame["ra"] + xc["dra"] / 3.6e6 / cosd
    sde = frame["dec"] + xc["ddec"] / 3.6e6
    ia, ib = _mutual_nn(SkyCoord(sra * u.deg, sde * u.deg), rsc, match_arcsec)
    if len(ia) < MIN_PAIRS:
        return dict(reason="too few mutual matches")
    with np.errstate(invalid="ignore", divide="ignore"):
        dm = -2.5 * np.log10(frame["flux"][ia] / rfl[ib])
    good = np.isfinite(dm)
    if good.sum() < MIN_PAIRS:
        return dict(reason="too few flux-vetted matches")
    good &= np.abs(dm - np.median(dm[good])) < flux_tol
    ia, ib = ia[good], ib[good]
    if len(ia) < MIN_PAIRS:
        return dict(reason="too few flux-vetted matches")
    # sky residual on the local tangent plane (arcsec), frame (shifted) − consensus
    dxi = (sra[ia] - rra[ib]) * cosd * 3600.0
    deta = (sde[ia] - rde[ib]) * 3600.0
    # local Jacobian: detector pixels as a linear function of the consensus tangent-plane position
    xi = (rra[ib] - ra0) * cosd * 3600.0; eta = (rde[ib] - dec0) * 3600.0
    A = np.column_stack([np.ones_like(xi), xi, eta])
    cx, *_ = np.linalg.lstsq(A, frame["x"][ia], rcond=None)
    cy, *_ = np.linalg.lstsq(A, frame["y"][ia], rcond=None)
    J = np.array([[cx[1], cx[2]], [cy[1], cy[2]]])          # pixels per arcsec
    pix = J @ np.vstack([dxi, deta])                          # residual in pixels
    # mas along the detector axes: pixel offset × the frame's own pixel scale
    pixscale = frame.get("pixscale")
    if not (pixscale and np.isfinite(pixscale)):
        pixscale = 1.0 / np.sqrt(abs(np.linalg.det(J)))
    dx = pix[0] * pixscale * 1000.0; dy = pix[1] * pixscale * 1000.0
    # robust 5-sigma clip of wrong-star tails left inside the match radius
    r = np.hypot(dx - np.median(dx), dy - np.median(dy))
    s = 1.4826 * np.median(r) / 1.1774 if len(r) else 0.0   # Rayleigh median -> per-axis sigma
    keep = r < max(5.0 * s, 1.0)
    return dict(x=frame["x"][ia][keep], y=frame["y"][ia][keep], dx=dx[keep], dy=dy[keep],
                bulk_dra=float(xc["dra"]), bulk_ddec=float(xc["ddec"]),
                npairs=int(keep.sum()), peak_ratio=float(xc["peak_ratio"]))


def _cell_index(x, y, nb, size):
    ix = np.clip((np.asarray(x) / size * nb).astype(int), 0, nb - 1)
    iy = np.clip((np.asarray(y) / size * nb).astype(int), 0, nb - 1)
    return ix * nb + iy


def binned_map(x, y, dx, dy, nb=NBIN, size=DET_NPIX, minn=MIN_PER_CELL, cell=None):
    """Median (dx, dy) on a fixed ``nb`` x ``nb`` detector grid over [0, size).  Returns
    (mdx, mdy, cnt), each ``[nb (x), nb (y)]``, NaN where a cell holds fewer than ``minn``."""
    k = _cell_index(x, y, nb, size) if cell is None else cell
    order = np.argsort(k, kind="stable")
    ks = k[order]; vx = np.asarray(dx)[order]; vy = np.asarray(dy)[order]
    bounds = np.searchsorted(ks, np.arange(nb * nb + 1))
    mdx = np.full(nb * nb, np.nan); mdy = np.full(nb * nb, np.nan); cnt = np.zeros(nb * nb, int)
    for c in range(nb * nb):
        lo, hi = bounds[c], bounds[c + 1]
        cnt[c] = hi - lo
        if hi - lo >= minn:
            mdx[c] = np.median(vx[lo:hi]); mdy[c] = np.median(vy[lo:hi])
    return mdx.reshape(nb, nb), mdy.reshape(nb, nb), cnt.reshape(nb, nb)


def map_rms(mdx, mdy):
    """RMS vector amplitude of the populated cells of a binned map (mas)."""
    a2 = (mdx ** 2 + mdy ** 2)[np.isfinite(mdx) & np.isfinite(mdy)]
    return float(np.sqrt(a2.mean())) if a2.size else float("nan")


def binned_profile(u_, v, nb=16, size=DET_NPIX, minn=MIN_PER_CELL):
    """Median of ``v`` in ``nb`` bins of detector coordinate ``u_``: (centres, medians)."""
    edges = np.linspace(0, size, nb + 1)
    i = np.clip(np.digitize(u_, edges) - 1, 0, nb - 1)
    med = np.array([np.median(v[i == b]) if np.count_nonzero(i == b) >= minn else np.nan
                    for b in range(nb)])
    return 0.5 * (edges[1:] + edges[:-1]), med


def stack_detector(x, y, dx, dy, nb=NBIN, size=DET_NPIX, minn=MIN_PER_CELL, n_perm=10, seed=0):
    """Stacked binned map and its summary numbers for one detector.

    ``sysrms_mas`` is the RMS of the binned map; ``null_rms_mas`` the median RMS over ``n_perm``
    shuffles of the residual vectors across the fixed positions (the map pure noise would give with
    these cell counts); ``sysrms_debiased_mas`` = sqrt(max(sysrms² − null², 0))."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    dx = np.asarray(dx, float); dy = np.asarray(dy, float)
    cell = _cell_index(x, y, nb, size)
    mdx, mdy, cnt = binned_map(x, y, dx, dy, nb, size, minn, cell=cell)
    rms = map_rms(mdx, mdy)
    rng = np.random.default_rng(seed)
    nulls = []
    for _ in range(int(n_perm)):
        p = rng.permutation(len(dx))
        nx, ny, _ = binned_map(x, y, dx[p], dy[p], nb, size, minn, cell=cell)
        nulls.append(map_rms(nx, ny))
    null = float(np.nanmedian(nulls)) if nulls else float("nan")
    deb = float(np.sqrt(max(rms ** 2 - null ** 2, 0.0))) if np.isfinite(rms) and np.isfinite(null) \
        else float("nan")
    return dict(mdx=mdx, mdy=mdy, cnt=cnt, n=int(len(dx)), sysrms_mas=rms, null_rms_mas=null,
                sysrms_debiased_mas=deb,
                significance=float(rms / null) if null and np.isfinite(null) else float("nan"),
                cells_used=int(np.count_nonzero(np.isfinite(mdx))),
                perstar_rms_mas=float(np.hypot(1.4826 * np.median(np.abs(dx - np.median(dx))),
                                               1.4826 * np.median(np.abs(dy - np.median(dy)))))
                if len(dx) else float("nan"))


def stack_maps(by_det, **kw):
    """``stack_detector`` for every detector in ``{det: dict(x, y, dx, dy)}`` (concatenated
    frame residuals).  Detectors with no residuals are skipped."""
    return {d: stack_detector(r["x"], r["y"], r["dx"], r["dy"], **kw)
            for d, r in sorted(by_det.items()) if len(r["dx"])}


def concat(results):
    """Concatenate a list of ``frame_residuals`` dicts (skipping failures) into one dict."""
    good = [r for r in results if "dx" in r]
    if not good:
        return dict(x=np.array([]), y=np.array([]), dx=np.array([]), dy=np.array([]))
    return {k: np.concatenate([r[k] for r in good]) for k in ("x", "y", "dx", "dy")}


def summary_json(stacks):
    """JSON-ready per-detector numbers (no arrays except the binned map, rounded)."""
    out = {}
    for d, s in stacks.items():
        out[d] = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in s.items()
                  if k not in ("mdx", "mdy", "cnt")}
        out[d]["map_dx_mas"] = np.round(s["mdx"], 2).tolist()
        out[d]["map_dy_mas"] = np.round(s["mdy"], 2).tolist()
        out[d]["map_count"] = s["cnt"].tolist()
    return out


def plot_stacks(stacks, by_det, title, out_path, nb=NBIN, size=DET_NPIX):
    """One quiver panel per detector (rows of 4) plus a row of four median-residual profiles
    (dx vs x, dy vs x, dx vs y, dy vs y; one line per detector).  Returns ``out_path``."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    dets = list(stacks)
    ncol = 4
    nrow = int(np.ceil(len(dets) / ncol))
    fig = plt.figure(figsize=(4.2 * ncol, 3.9 * nrow + 3.3))
    gs = fig.add_gridspec(nrow + 1, ncol, height_ratios=[3.9] * nrow + [3.0], hspace=0.45,
                          wspace=0.3)
    amps = [np.nanmax(np.hypot(s["mdx"], s["mdy"])) for s in stacks.values()
            if np.isfinite(s["mdx"]).any()]
    vlim = max(0.5, float(np.nanpercentile(amps, 90))) if amps else 1.0
    cs = size / nb
    cc = (np.arange(nb) + 0.5) * cs
    XX, YY = np.meshgrid(cc, cc, indexing="ij")
    for k, d in enumerate(dets):
        s = stacks[d]
        a = fig.add_subplot(gs[k // ncol, k % ncol])
        amp = np.hypot(s["mdx"], s["mdy"])
        im = a.imshow(amp.T, origin="lower", extent=[0, size, 0, size], cmap="Greys",
                      vmin=0, vmax=vlim, alpha=0.6)
        # an arrow of length vlim spans one cell
        q = a.quiver(XX, YY, s["mdx"], s["mdy"], color="C3", angles="xy", scale_units="xy",
                     scale=vlim / (0.9 * cs), width=0.008)
        a.set_xlim(0, size); a.set_ylim(0, size); a.set_aspect("equal")
        a.set_xlabel("detector x [px]", fontsize=8); a.set_ylabel("detector y [px]", fontsize=8)
        a.tick_params(labelsize=7)
        a.set_title(f"{d}: map rms {s['sysrms_mas']:.2f} mas (null {s['null_rms_mas']:.2f})\n"
                    f"debiased {s['sysrms_debiased_mas']:.2f} mas, {s['n']} pairs", fontsize=8.5)
        if k == 0:
            a.quiverkey(q, 0.15, 1.17, vlim, f"{vlim:.1f} mas", labelpos="E", coordinates="axes",
                        fontproperties={"size": 7})
    fig.colorbar(im, ax=fig.axes[:len(dets)], shrink=0.6, pad=0.01,
                 label="binned residual amplitude [mas]")
    labs = (("x", "dx"), ("x", "dy"), ("y", "dx"), ("y", "dy"))
    for j, (ucol, vcol) in enumerate(labs):
        a = fig.add_subplot(gs[nrow, j])
        for k, d in enumerate(dets):
            r = by_det[d]
            c, m = binned_profile(r[ucol], r[vcol], size=size)
            a.plot(c, m, "-o", ms=2.5, lw=1, color=f"C{k % 10}", label=d)
        a.axhline(0, color="0.5", lw=0.6)
        a.set_xlabel(f"detector {ucol} [px]", fontsize=8)
        a.set_ylabel(f"median {vcol} [mas]", fontsize=8)
        a.set_title(f"{vcol} vs {ucol}", fontsize=9); a.tick_params(labelsize=7)
        if j == 3:
            a.legend(fontsize=6, ncol=2, loc="best")
    fig.suptitle(title, fontsize=11)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out_path
