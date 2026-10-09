"""Stage 14 (fit-by-iteration animated cutout) tests.

Small synthetic mosaics (released image, per-iteration model/residual/background) and catalogues
are written to a temporary directory; the product lookups are pointed at them, so the whole
stage runs end to end and writes a real GIF."""
import os

import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS

from data_qa import diagnostics as D
from data_qa.observations import Observation

RA0, DEC0 = 266.40, -29.00
NPIX, PIXAS = 120, 0.1          # 12" square mosaics


def _obs():
    return Observation(program="10678", obs="066", target="GC Treasury", release_field="gc-treasury",
                       instrument="NIRCam", filters=["F212N", "F480M"])


def _wcs():
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [RA0, DEC0]
    w.wcs.crpix = [NPIX / 2 + 0.5, NPIX / 2 + 0.5]
    w.wcs.cdelt = [-PIXAS / 3600.0, PIXAS / 3600.0]
    return w


def _mosaic(path, value, rng):
    hdr = _wcs().to_header()
    data = (value + rng.normal(0, 1, (NPIX, NPIX))).astype(np.float32)
    fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(data, hdr, name="SCI")]).writeto(path)
    return str(path)


def _stars(n, rng, half_as=5.0):
    cosd = np.cos(np.radians(DEC0))
    dx, dy = rng.uniform(-half_as, half_as, (2, n))
    return RA0 + dx / 3600.0 / cosd, DEC0 + dy / 3600.0


def _cat(path, ra, dec):
    Table({"skycoord.ra": ra, "skycoord.dec": dec}).write(path)
    return str(path)


@pytest.fixture
def field(tmp_path, monkeypatch):
    """Released mosaics, m2/m3 products and per-filter catalogues for F212N + F480M, plus m8.
    F480M sees the first 20 of the 30 F212N stars and 5 of its own."""
    rng = np.random.default_rng(1)
    ra, dec = _stars(30, rng)
    lra, ldec = _stars(5, rng)
    star = {"F212N": (ra, dec), "F480M": (np.r_[ra[:20], lra], np.r_[dec[:20], ldec])}
    img, prods, cats = {}, {}, {}
    for f in ("F212N", "F480M"):
        img[f] = _mosaic(tmp_path / f"{f}_i2d.fits", 10.0, rng)
        prods[f] = {}
        for n in (2, 3):
            tag = f"m{n}"
            prods[f][n] = dict(tag=tag, **{k: _mosaic(tmp_path / f"{f}_{tag}_{k}.fits", v, rng)
                                           for k, v in (("model", 5.0), ("residual", 5.0),
                                                        ("background", 5.0))})
            cats[(f, tag)] = _cat(tmp_path / f"{f}_{tag}.fits", *star[f])
    nm8 = 35
    m8 = Table({"skycoord_ref.ra": np.r_[ra, lra], "skycoord_ref.dec": np.r_[dec, ldec],
                "skycoord_f212n.ra": np.r_[ra, lra], "skycoord_f212n.dec": np.r_[dec, ldec],
                "skycoord_f480m.ra": np.r_[ra, lra], "skycoord_f480m.dec": np.r_[dec, ldec],
                "mask_f212n": np.r_[np.zeros(30, bool), np.ones(5, bool)],
                "mask_f480m": np.r_[np.zeros(20, bool), np.ones(10, bool), np.zeros(5, bool)],
                "forced_filled_f212n": np.zeros(nm8, bool),
                "forced_filled_f480m": np.r_[np.zeros(20, bool), np.ones(10, bool), np.zeros(5, bool)]})
    m8.write(tmp_path / "m8.fits")
    monkeypatch.setattr(D, "OUTDIR", str(tmp_path / "out"))
    monkeypatch.setattr(D, "_mosaic_path", lambda o, f: img.get(f))
    monkeypatch.setattr(D, "_s14_iter_products", lambda o, f: prods.get(f, {}))
    monkeypatch.setattr(D, "_s14_perfilter_catalog", lambda o, f, tag: cats.get((f, tag)))
    monkeypatch.setattr(D, "_s14_m8_catalog", lambda o: str(tmp_path / "m8.fits"))
    return dict(img=img, prods=prods, cats=cats)


def test_classify_mutual_nearest_neighbour():
    ra1, de1 = np.array([0.0, 1.0, 2.0]), np.zeros(3)
    # LW: one exact match of SW[0], one 0.05" off SW[1], one 1" away from everything
    ra2 = np.array([0.0, 1.0 + 0.05 / 3600.0, 2.0 + 1.0 / 3600.0]); de2 = np.zeros(3)
    cl = D._s14_classify((ra1, de1), (ra2, de2))
    assert len(cl["both"][0]) == 2 and len(cl["sw"][0]) == 1 and len(cl["lw"][0]) == 1
    assert np.isclose(cl["sw"][0][0], 2.0)


def test_classify_empty_sides():
    e = (np.array([]), np.array([]))
    cl = D._s14_classify((np.array([1.0]), np.array([0.0])), e)
    assert len(cl["sw"][0]) == 1 and len(cl["both"][0]) == 0 and len(cl["lw"][0]) == 0


def test_stage14_writes_gif_with_raw_iterations_and_m8(field):
    from PIL import Image
    out, m = D.stage14_iteration_animation(_obs(), "F212N", "F480M")
    assert out.endswith("_stage14.gif") and os.path.exists(out)
    assert m["available"] is True and m["passed"] is None
    assert m["iterations"] == ["raw", "m2", "m3", "m8"]
    with Image.open(out) as im:
        assert im.n_frames == 4
    # m8: forced F480M fills do not count as F480M detections
    k = m["counts"]["m8"]
    assert k["both"] + k["sw"] + k["lw"] > 0
    # centre lies inside the mosaics and is reproducible (seeded by the obsid)
    _, m2 = D.stage14_iteration_animation(_obs(), "F212N", "F480M")
    assert (m2["center_ra"], m2["center_dec"]) == (m["center_ra"], m["center_dec"])


def test_stage14_m8_classes_exclude_forced_fills(field, tmp_path):
    from astropy.coordinates import SkyCoord
    import astropy.units as u
    c = SkyCoord(RA0 * u.deg, DEC0 * u.deg)
    cl = D._s14_m8_classes(str(tmp_path / "m8.fits"), "F212N", "F480M", c, 6.0)
    # 20 shared, 10 F212N-only (F480M forced fills), 5 F480M-only
    assert (len(cl["both"][0]), len(cl["sw"][0]), len(cl["lw"][0])) == (20, 10, 5)


def test_stage14_not_applicable_posts_nothing(field, monkeypatch):
    monkeypatch.setattr(D, "_s14_iter_products", lambda o, f: {})
    png, m = D.stage14_iteration_animation(_obs(), "F212N", "F480M")
    assert png is None and m["available"] is False and m["passed"] is None
    png, m = D.stage14_iteration_animation(_obs(), "F212N", None)
    assert png is None and m["available"] is False


def test_stage14_caption_and_default_stage_list():
    cap = D.caption_for(14, dict(sw="F212N", lw="F480M", center_ra=266.4, center_dec=-29.0,
                                 iterations=["raw", "m2", "m8"], available=True))
    assert "raw → m2 → m8" in cap and "qa_methods.md#stage14" in cap and "No pass/fail" in cap
    assert "pending" in D.caption_for(14, dict(available=False, na_reason="x"))
    assert 14 in D._STAGES_NEEDING_SW


def test_post_stage_keeps_gif_extension(monkeypatch):
    from data_qa import post_diagnostics as P
    names = []
    monkeypatch.setattr(P, "_issue_number", lambda repo, token, title: 5)
    monkeypatch.setattr(P, "_find_stage_comment", lambda repo, token, num, marker: None)
    monkeypatch.setattr(P, "upload_asset", lambda repo, token, path, name: names.append(name) or "u")
    monkeypatch.setattr(P, "_req", lambda *a, **k: (201, {"html_url": "h"}))
    P.post_stage(_obs(), 14, "/x/jw10678-o066_stage14.gif", "cap", "JWST-GC/data-qa", token="t")
    P.post_stage(_obs(), 4, "/x/jw10678-o066_stage4.png", "cap", "JWST-GC/data-qa", token="t")
    assert names == ["jw10678-o066_stage14.gif", "jw10678-o066_stage4.png"]


def test_stage14_missing_background_maps(field, monkeypatch):
    """Pruned smoothed-background mosaics (jwst-gc-pipeline#1105) leave a blank panel, not n/a."""
    from PIL import Image
    prods = {f: {n: dict(p, background=None if f == "F212N" else p["background"])
                 for n, p in d.items()} for f, d in field["prods"].items()}
    monkeypatch.setattr(D, "_s14_iter_products", lambda o, f: prods.get(f, {}))
    out, m = D.stage14_iteration_animation(_obs(), "F212N", "F480M")
    assert m["available"] is True and m["background_not_kept"] == ["F212N m2", "F212N m3"]
    with Image.open(out) as im:
        assert im.n_frames == 4
    assert "F212N m2, F212N m3" in D.caption_for(14, m)


def test_stage14_m8_without_band_columns_is_skipped(field, tmp_path, monkeypatch):
    Table({"skycoord_ref.ra": [RA0], "skycoord_ref.dec": [DEC0]}).write(tmp_path / "m8bare.fits")
    monkeypatch.setattr(D, "_s14_m8_catalog", lambda o: str(tmp_path / "m8bare.fits"))
    _, m = D.stage14_iteration_animation(_obs(), "F212N", "F480M")
    assert m["iterations"] == ["raw", "m2", "m3"]
