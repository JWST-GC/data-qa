"""Distortion fields in detector axes for the 10678 distortion comparison.

Conventions
-----------
* Pixels are 0-based detector (x, y), as in the per-frame catalogues (``x_fit``/``y_fit``)
  and the jwst gwcs / CRDS distortion model.
* A "detector-axis displacement" is a position change expressed as a pixel offset (dx, dy)
  multiplied by the detector's mean pixel scale, in mas.  This is the same convention as
  ``data_qa.intradetector`` (pixel residual x PIXSCALE).
* ``DeltaField(det, ...)`` returns  delta = T(STDGDC(p)) - CRDS(p)  for raw pixel p, where T is
  the least-squares transform (similarity or affine) from the STDGDC corrected frame to the
  CRDS V2V3 frame.  delta is the position change that adopting STDGDC would apply to a
  detection at p, after alignment absorbs T.  A per-frame residual measured with CRDS
  (frame - truth) is therefore expected to look like  -delta  if STDGDC is the truth.

CRDS reference files are the ones jwst_1596.pmap selected for 10678 (R_DISTOR keywords).
STDGDC files are read through the pipeline's ``jwst_gc_pipeline.astrometry_gdc.stdgdc``.
"""
from __future__ import annotations

import os
import sys

import numpy as np

PIPE_REPO = "/blue/adamginsburg/adamginsburg/repos/jwst-gc-pipeline"
if PIPE_REPO not in sys.path:
    sys.path.insert(0, PIPE_REPO)

CRDS_DIR = "/orange/adamginsburg/jwst/crds/references/jwst/nircam"
R_DISTOR = {"NRCA1": "0228", "NRCA2": "0234", "NRCA3": "0182", "NRCA4": "0283",
            "NRCB1": "0305", "NRCB2": "0257", "NRCB3": "0307", "NRCB4": "0209",
            "NRCALONG": "0269", "NRCBLONG": "0314"}
SW_DETS = ["NRCA1", "NRCA2", "NRCA3", "NRCA4", "NRCB1", "NRCB2", "NRCB3", "NRCB4"]
LW_DETS = ["NRCALONG", "NRCBLONG"]
NPIX = 2048
MAX_SANE_CORRECTION_PIX = 50.0   # same guard as astrometry_gdc.gdc_wcs (library border holes)

_crds_cache = {}


def crds_path(det):
    return os.path.join(CRDS_DIR, f"jwst_nircam_distortion_{R_DISTOR[det.upper()]}.asdf")


def crds_model(det):
    """astropy model: 0-based detector (x, y) -> (V2, V3) arcsec."""
    import asdf
    det = det.upper()
    if det not in _crds_cache:
        with asdf.open(crds_path(det), lazy_load=False, memmap=False) as af:
            _crds_cache[det] = af.tree["model"]
    return _crds_cache[det]


def crds_v2v3(det, x, y):
    v2, v3 = crds_model(det)(np.asarray(x, float), np.asarray(y, float))
    return np.asarray(v2, float), np.asarray(v3, float)


def crds_jacobian(det, x, y, h=0.5):
    """dV/dp (arcsec / pixel), shape (..., 2, 2): [[dV2/dx, dV2/dy], [dV3/dx, dV3/dy]]."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    a2, a3 = crds_v2v3(det, x + h, y); b2, b3 = crds_v2v3(det, x - h, y)
    c2, c3 = crds_v2v3(det, x, y + h); d2, d3 = crds_v2v3(det, x, y - h)
    J = np.empty(x.shape + (2, 2))
    J[..., 0, 0] = (a2 - b2) / (2 * h); J[..., 0, 1] = (c2 - d2) / (2 * h)
    J[..., 1, 0] = (a3 - b3) / (2 * h); J[..., 1, 1] = (c3 - d3) / (2 * h)
    return J


def pixscale(det):
    """Mean CRDS pixel scale at the detector centre, arcsec/pixel."""
    J = crds_jacobian(det, np.array(1023.5), np.array(1023.5))
    return float(np.sqrt(abs(np.linalg.det(J))))


def v2v3_to_detmas(det, x, y, dv2, dv3):
    """Convert a V2V3 displacement (arcsec) at pixel (x, y) to detector-axis mas."""
    J = crds_jacobian(det, x, y)
    Ji = np.linalg.inv(J)
    dpx = Ji[..., 0, 0] * dv2 + Ji[..., 0, 1] * dv3
    dpy = Ji[..., 1, 0] * dv2 + Ji[..., 1, 1] * dv3
    s = pixscale(det) * 1000.0
    return dpx * s, dpy * s


def load_gdc(det, filt="F212N", version="v1"):
    from jwst_gc_pipeline.astrometry_gdc.stdgdc import STDGDC
    return STDGDC.load(det, filt, version=version)


def fit_transform(u, v, X, Y, mode):
    """Least-squares transform (u, v) -> (X, Y).  Returns a callable (u, v) -> (X', Y').

    'similarity' tries both parities (proper rotation and reflection) and keeps the better fit.
    """
    u = np.asarray(u, float); v = np.asarray(v, float)
    X = np.asarray(X, float); Y = np.asarray(Y, float)
    one = np.ones_like(u); zero = np.zeros_like(u)
    if mode == "none":
        return lambda uu, vv: (np.asarray(uu, float), np.asarray(vv, float))
    if mode == "shift":
        a = np.mean(X - u); b = np.mean(Y - v)
        return lambda uu, vv: (np.asarray(uu) + a, np.asarray(vv) + b)
    if mode == "affine":
        A = np.column_stack([one, u, v])
        cx, *_ = np.linalg.lstsq(A, X, rcond=None)
        cy, *_ = np.linalg.lstsq(A, Y, rcond=None)
        return lambda uu, vv: (cx[0] + cx[1] * np.asarray(uu) + cx[2] * np.asarray(vv),
                               cy[0] + cy[1] * np.asarray(uu) + cy[2] * np.asarray(vv))
    if mode == "similarity":
        best = None
        for par in (+1.0, -1.0):
            w = par * v
            # X = a*u - b*w + tx ; Y = b*u + a*w + ty
            A = np.block([[u[:, None], -w[:, None], one[:, None], zero[:, None]],
                          [w[:, None], u[:, None], zero[:, None], one[:, None]]])
            rhs = np.concatenate([X, Y])
            p, *_ = np.linalg.lstsq(A, rhs, rcond=None)
            res = np.sum((A @ p - rhs) ** 2)
            if best is None or res < best[0]:
                best = (res, par, p)
        _, par, p = best
        a, b, tx, ty = p
        return lambda uu, vv: (a * np.asarray(uu) - b * par * np.asarray(vv) + tx,
                               b * np.asarray(uu) + a * par * np.asarray(vv) + ty)
    raise ValueError(mode)


def remove_transform_detmas(x, y, dx, dy, mode):
    """Remove a least-squares shift / similarity / affine from a detector-axis displacement field
    sampled at (x, y): fit (x+dx', y+dy') ~ T(x, y) in pixel-like units, return residual (mas).
    Displacements are small, so fitting T to (x + dx) with (x, y) as source is linear."""
    x = np.asarray(x, float).ravel(); y = np.asarray(y, float).ravel()
    dx = np.asarray(dx, float).ravel(); dy = np.asarray(dy, float).ravel()
    ok = np.isfinite(dx) & np.isfinite(dy)
    # work in mas units for both source and target: position (mas) = pixel * 31 mas; the scale
    # factor only needs to be consistent, so use 1000 mas per 'unit' pixel to keep conditioning.
    S = 1000.0
    T = fit_transform(x[ok] * S, y[ok] * S, x[ok] * S + dx[ok], y[ok] * S + dy[ok], mode)
    fx, fy = T(x * S, y * S)
    return x * S + dx - fx, y * S + dy - fy


class DeltaField:
    """delta(p) = T(STDGDC(p)) - CRDS(p) in detector-axis mas (see module docstring).

    mode: 'similarity' (shift+rotation+scale removed; what the per-frame alignment absorbs) or
    'affine' (6-parameter; what astrometry_gdc.gdc_wcs anchoring absorbs).
    """

    def __init__(self, det, filt="F212N", version="v1", mode="similarity", grid_n=64):
        self.det = det.upper(); self.mode = mode
        self.gdc = load_gdc(self.det, filt, version)
        g = np.linspace(0.0, NPIX - 1.0, grid_n)
        gx, gy = np.meshgrid(g, g, indexing="ij")
        gx = gx.ravel(); gy = gy.ravel()
        xc, yc = self.gdc.forward(gx, gy)
        ok = (np.isfinite(xc) & np.isfinite(yc) & (np.abs(xc - gx) < MAX_SANE_CORRECTION_PIX)
              & (np.abs(yc - gy) < MAX_SANE_CORRECTION_PIX))
        v2, v3 = crds_v2v3(self.det, gx, gy)
        self.T = fit_transform(xc[ok], yc[ok], v2[ok], v3[ok], mode)
        self.n_invalid_anchor = int((~ok).sum())

    def __call__(self, x, y):
        x = np.asarray(x, float); y = np.asarray(y, float)
        xc, yc = self.gdc.forward(x, y)
        ok = (np.isfinite(xc) & np.isfinite(yc) & (np.abs(xc - x) < MAX_SANE_CORRECTION_PIX)
              & (np.abs(yc - y) < MAX_SANE_CORRECTION_PIX))
        f2, f3 = self.T(xc, yc)
        v2, v3 = crds_v2v3(self.det, x, y)
        dx, dy = v2v3_to_detmas(self.det, x, y, f2 - v2, f3 - v3)
        return np.where(ok, dx, np.nan), np.where(ok, dy, np.nan)


class BinnedMapField:
    """A binned detector-axis map (nb x nb cells over [0, 2048), indexed [ix, iy]) evaluated by
    bilinear interpolation between cell centres (nearest-edge extrapolation), NaN cells filled
    by the mean of finite neighbours (then 0)."""

    def __init__(self, mdx, mdy, size=NPIX):
        from scipy.interpolate import RegularGridInterpolator
        mdx = np.array(mdx, float); mdy = np.array(mdy, float)
        nb = mdx.shape[0]
        for m in (mdx, mdy):
            for _ in range(nb):
                bad = ~np.isfinite(m)
                if not bad.any():
                    break
                pad = np.pad(m, 1, constant_values=np.nan)
                neigh = np.stack([pad[:-2, 1:-1], pad[2:, 1:-1], pad[1:-1, :-2], pad[1:-1, 2:]])
                with np.errstate(invalid="ignore"):
                    fill = np.nanmean(np.where(np.isfinite(neigh), neigh, np.nan), axis=0)
                m[bad] = fill[bad]
            m[~np.isfinite(m)] = 0.0
        c = (np.arange(nb) + 0.5) * size / nb
        kw = dict(method="linear", bounds_error=False, fill_value=None)
        self._fx = RegularGridInterpolator((c, c), mdx, **kw)
        self._fy = RegularGridInterpolator((c, c), mdy, **kw)

    def __call__(self, x, y):
        pts = np.stack(np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float)), axis=-1)
        return self._fx(pts), self._fy(pts)
