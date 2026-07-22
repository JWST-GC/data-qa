"""Shared fixtures: a synthetic F212N + long-band FITS pair with real WCS.

The blue WCS sits at position angle ~88 deg -- deliberately inside the pyavm
Scale+Rotation degeneracy zone the CDMatrix AVM form exists to avoid -- so the
round-trip validation actually exercises the failure mode.
"""
import numpy as np
import pytest


def _make_wcs(crval, crpix, scale_deg, pa_deg):
    from astropy.wcs import WCS
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = list(crval)
    w.wcs.crpix = list(crpix)
    th = np.deg2rad(pa_deg)
    # negative determinant (RA increases leftward): proper sky parity
    w.wcs.cd = scale_deg * np.array([[-np.cos(th), np.sin(th)],
                                     [np.sin(th), np.cos(th)]])
    return w


def _star_field(shape, rng, nstars=25, fwhm_pix=2.0):
    ny, nx = shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    img = rng.normal(1.0, 0.05, size=shape)
    sig = fwhm_pix / 2.355
    for _ in range(nstars):
        x0, y0 = rng.uniform(0, nx), rng.uniform(0, ny)
        amp = rng.uniform(5, 50)
        img += amp * np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2 * sig ** 2))
    return img


@pytest.fixture
def star_grid_pair(tmp_path):
    """(f212n_path, long_path, refcat_path, wcs): a 128x128 pair with a 5x5
    grid of BRIGHT gaussian stars at known (jittered) pixel positions, plus a
    reference catalog (RA/DEC/refmag -- the gaia_virac2_refcat column form)
    built from the injection positions.  Both bands share the star field, so
    every catalog star is a luminance peak in the composed RGB."""
    fits = pytest.importorskip("astropy.io.fits")
    from astropy.table import Table

    rng = np.random.default_rng(7)
    shape = (128, 128)
    wcs = _make_wcs((266.5, -28.9), (64.5, 64.5), 0.06 / 3600, 88.0)
    ny, nx = shape
    yy, xx = np.mgrid[0:ny, 0:nx]
    img = rng.normal(1.0, 0.02, size=shape)
    sig = 2.0 / 2.355
    xs, ys = [], []
    for gy in range(5):
        for gx in range(5):
            x0 = 20.0 + gx * 22 + rng.uniform(-1, 1)
            y0 = 20.0 + gy * 22 + rng.uniform(-1, 1)
            img += 40 * np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2)
                               / (2 * sig ** 2))
            xs.append(x0)
            ys.append(y0)

    f212n = tmp_path / "stargrid_f212n_i2d.fits"
    longp = tmp_path / "stargrid_f480m_i2d.fits"
    for path in (f212n, longp):
        hdu = fits.ImageHDU(data=img.astype("float32"),
                            header=wcs.to_header(), name="SCI")
        fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(path)

    world = wcs.pixel_to_world(np.array(xs), np.array(ys))
    refcat = tmp_path / "gaia_virac2_refcat_synthetic.fits"
    Table({"RA": world.ra.deg, "DEC": world.dec.deg,
           "refmag": np.linspace(10.0, 12.4, len(xs))}).write(refcat)
    return str(f212n), str(longp), str(refcat), wcs


@pytest.fixture
def synthetic_pair(tmp_path):
    """(f212n_path, long_path): overlapping 64x64 / 80x80 mosaics, GC coords,
    rotated WCS, NaN edge strip in the blue band."""
    fits = pytest.importorskip("astropy.io.fits")
    rng = np.random.default_rng(42)
    center = (266.5, -28.9)

    blue = _star_field((64, 64), rng)
    blue[:, :5] = np.nan                       # detector-edge NaN strip
    wblue = _make_wcs(center, (32.5, 32.5), 0.06 / 3600, 88.0)

    red = _star_field((80, 80), rng)
    wred = _make_wcs(center, (40.5, 40.5), 0.10 / 3600, 85.0)

    f212n = tmp_path / "synthetic_f212n_i2d.fits"
    longp = tmp_path / "synthetic_f480m_i2d.fits"
    for path, data, wcs in ((f212n, blue, wblue), (longp, red, wred)):
        hdu = fits.ImageHDU(data=data.astype("float32"),
                            header=wcs.to_header(), name="SCI")
        fits.HDUList([fits.PrimaryHDU(), hdu]).writeto(path)
    return str(f212n), str(longp)
