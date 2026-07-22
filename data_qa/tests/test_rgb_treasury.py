"""Offline tests for data_qa.rgb_treasury (synthetic 64x64 pair)."""
import json
import os

import numpy as np
import pytest

pytest.importorskip("astropy")
pytest.importorskip("reproject")
pytest.importorskip("pyavm")
pytest.importorskip("PIL")

from data_qa import rgb_treasury as RT


def test_end_to_end_pass(synthetic_pair, tmp_path):
    f212n, longp = synthetic_pair
    out = str(tmp_path / "rgb" / "synthetic_rgb")
    rc = RT.main(["--f212n", f212n, "--long", longp, "--long-band", "F480M",
                  "--out", out])
    assert rc == 0
    assert os.path.exists(out + ".png")
    assert os.path.exists(out + ".jpg")
    with open(out + ".validation.json") as fh:
        verdict = json.load(fh)
    assert verdict["pass"] is True
    assert verdict["checks"]["wcs_max_offset_arcsec"] < RT.WCS_PASS_ARCSEC
    assert verdict["checks"]["alpha_mode"].startswith("exact")
    assert verdict["checks"]["finite_fraction"] > 0.5
    assert verdict["inputs"]["f212n"]["sha256"]
    assert verdict["inputs"]["long_band"] == "F480M"


def test_standalone_validate_passes_after_build(synthetic_pair, tmp_path):
    f212n, longp = synthetic_pair
    out = str(tmp_path / "synthetic_rgb")
    assert RT.main(["--f212n", f212n, "--long", longp, "--out", out]) == 0
    rc = RT.main(["--f212n", f212n, "--long", longp, "--out", out,
                  "--validate"])
    assert rc == 0
    with open(out + ".validation.json") as fh:
        assert json.load(fh)["checks"]["alpha_mode"].startswith("one-sided")


def test_corrupt_avm_fails_validation(synthetic_pair, tmp_path):
    f212n, longp = synthetic_pair
    out = str(tmp_path / "synthetic_rgb")
    assert RT.main(["--f212n", f212n, "--long", longp, "--out", out]) == 0
    # re-embed an AVM whose CRVAL is shifted by ~3.6" -> WCS check must FAIL
    blue, wcs, _ = RT.load_sci(f212n)
    bad = wcs.deepcopy()
    bad.wcs.crval = bad.wcs.crval + np.array([0.001, 0.001])
    avm = RT.faithful_avm(bad, blue.shape)
    png = out + ".png"
    tagged = out + ".bad.png"
    avm.embed(png, tagged)
    os.replace(tagged, png)
    rc = RT.main(["--f212n", f212n, "--long", longp, "--out", out,
                  "--validate"])
    assert rc == 1
    with open(out + ".validation.json") as fh:
        verdict = json.load(fh)
    assert verdict["pass"] is False
    assert verdict["checks"]["wcs_pass"] is False
    assert verdict["checks"]["wcs_max_offset_arcsec"] > RT.WCS_PASS_ARCSEC


def test_green_is_half_r_plus_b_and_alpha():
    """G = 0.5*(R+B) pixel math + alpha=0 only where BOTH bands are NaN."""
    blue = np.array([[0.0, 1.0], [np.nan, np.nan]])
    red = np.array([[0.5, np.nan], [1.0, np.nan]])
    lims = (0.0, 1.0)
    rgba = RT.compose_rgba(blue, red, lims, lims)
    b = RT._asinh_norm_local(blue, *lims)
    r = RT._asinh_norm_local(red, *lims)
    # house behavior (cmz.hips.two_color_tile): G is computed BEFORE the
    # NaN->0 replacement, so a pixel with either band NaN gets G=0 (pure
    # single-band color), not 0.5*the finite band.
    g = np.nan_to_num(0.5 * (r + b), nan=0.0)
    assert np.array_equal(rgba[..., 1], (g * 255).astype(np.uint8))
    assert np.array_equal(rgba[..., 0], (np.nan_to_num(r) * 255).astype(np.uint8))
    assert np.array_equal(rgba[..., 2], (np.nan_to_num(b) * 255).astype(np.uint8))
    # alpha: only the both-NaN pixel (1,1) is transparent
    assert rgba[..., 3].tolist() == [[255, 255], [255, 0]]


def test_local_asinh_matches_pipeline_if_importable():
    hips = RT._import_hips()
    if hips is None:
        pytest.skip("jwst_gc_pipeline.cmz.hips not importable")
    x = np.linspace(-0.5, 1.5, 101)
    np.testing.assert_allclose(RT._asinh_norm_local(x, 0.0, 1.0),
                               hips._asinh_norm(x, 0.0, 1.0))


# ------------------------------------------------------------------ star positions
def test_find_ref_catalog_convention(tmp_path):
    catdir = tmp_path / "sgrb2" / "catalogs"
    catdir.mkdir(parents=True)
    gaia = catdir / "gaia_refcat_epoch2022.70.fits"
    gaia.write_bytes(b"")
    assert RT.find_ref_catalog("sgrb2", root=str(tmp_path)) == str(gaia)
    virac = catdir / "gaia_virac2_refcat_epoch2022.70.fits"
    virac.write_bytes(b"")
    # gaia_virac2 preferred over the gaia-only fallback
    assert RT.find_ref_catalog("sgrb2", root=str(tmp_path)) == str(virac)
    assert RT.find_ref_catalog("nope", root=str(tmp_path)) is None
    assert RT.find_ref_catalog(None, root=str(tmp_path)) is None


def test_star_positions_pass(star_grid_pair, tmp_path):
    f212n, longp, refcat, _wcs = star_grid_pair
    out = str(tmp_path / "stars_rgb")
    rc = RT.main(["--f212n", f212n, "--long", longp, "--out", out,
                  "--ref-catalog", refcat, "--validate-stars"])
    assert rc == 0
    with open(out + ".validation.json") as fh:
        verdict = json.load(fh)
    stars = verdict["checks"]["star_positions"]
    assert stars["pass"] is True
    assert stars["median_offset_px"] <= RT.STAR_PASS_MEDIAN_PX
    assert stars["n_used"] >= RT.STAR_MIN_USED
    assert stars["matched_fraction"] >= RT.STAR_MIN_MATCH_FRACTION
    assert stars["ref_catalog"] == os.path.abspath(refcat)
    assert verdict["pass"] is True


def test_star_positions_fail_on_shifted_wcs(star_grid_pair, tmp_path):
    """A catalog offset by ~3 px (a shifted-WCS image would look identical to
    the matcher) must FAIL the star gate and the whole verdict."""
    from astropy.table import Table
    f212n, longp, refcat, wcs = star_grid_pair
    tbl = Table.read(refcat)
    x, y = wcs.wcs_world2pix(np.asarray(tbl["RA"]), np.asarray(tbl["DEC"]), 0)
    shifted = wcs.pixel_to_world(x + 3.0, y)     # rigid +3 px shift in x
    tbl["RA"], tbl["DEC"] = shifted.ra.deg, shifted.dec.deg
    badcat = str(tmp_path / "gaia_virac2_refcat_shifted.fits")
    tbl.write(badcat)
    out = str(tmp_path / "shifted_rgb")
    rc = RT.main(["--f212n", f212n, "--long", longp, "--out", out,
                  "--ref-catalog", badcat])
    assert rc == 1
    with open(out + ".validation.json") as fh:
        verdict = json.load(fh)
    stars = verdict["checks"]["star_positions"]
    assert stars["pass"] is False
    assert verdict["pass"] is False              # star fail fails the verdict
    # WCS/alpha checks alone still pass: the failure is the star gate
    assert verdict["checks"]["wcs_pass"] is True


def test_star_positions_skipped_without_catalog(synthetic_pair, tmp_path):
    f212n, longp = synthetic_pair
    out = str(tmp_path / "nocat_rgb")
    assert RT.main(["--f212n", f212n, "--long", longp, "--out", out]) == 0
    with open(out + ".validation.json") as fh:
        verdict = json.load(fh)
    stars = verdict["checks"]["star_positions"]
    assert stars["skipped"] is True
    assert "reason" in stars
    assert verdict["pass"] is True               # skip never fails the verdict


def test_validate_stars_requires_a_catalog(synthetic_pair, tmp_path, capsys):
    f212n, longp = synthetic_pair
    out = str(tmp_path / "required_rgb")
    rc = RT.main(["--f212n", f212n, "--long", longp, "--out", out,
                  "--validate-stars"])
    assert rc == 1
    assert "no reference catalog" in capsys.readouterr().err


def test_validation_binds_output_hashes(synthetic_pair, tmp_path):
    """The verdict records sha256 of the EXACT png/jpg it validated
    (publish.py re-hashes these at push time: stale-verdict protection)."""
    import hashlib
    f212n, longp = synthetic_pair
    out = str(tmp_path / "bound_rgb")
    assert RT.main(["--f212n", f212n, "--long", longp, "--out", out]) == 0
    with open(out + ".validation.json") as fh:
        outputs = json.load(fh)["outputs"]
    for ext, key in ((".png", "png_sha256"), (".jpg", "jpg_sha256")):
        with open(out + ext, "rb") as fh:
            assert outputs[key] == hashlib.sha256(fh.read()).hexdigest()


# ----------------------------------------------- plateau / saturation behavior
def _grid_png_with_avm(tmp_path, name, star_value=200, plateau=5,
                       shift_x=0.0, sat_positions=(), size=160, step=28,
                       start=24):
    """Hand-built grayscale RGBA PNG (full pixel control, no asinh stretch)
    with a grid of flat-top square stars, AVM-embedded WCS, and a matching
    reference catalog.  Returns (out_base, refcat_path).

    ``sat_positions``: indices of grid stars drawn as 255-valued (saturated)
    3x3 plateaus shifted +2 px in x from their catalog position."""
    from astropy.table import Table
    from PIL import Image
    from data_qa.tests.conftest import _make_wcs

    wcs = _make_wcs((266.5, -28.9), (size / 2 + 0.5, size / 2 + 0.5),
                    0.06 / 3600, 88.0)
    lum = np.full((size, size), 10, dtype=np.uint8)
    xs, ys = [], []
    idx = 0
    for gy in range(5):
        for gx in range(5):
            x0, y0 = start + gx * step, start + gy * step
            if idx in sat_positions:
                lum[y0 - 1: y0 + 2, x0 + 1: x0 + 4] = 255   # +2 px, 3x3, sat
            else:
                h = plateau // 2
                lum[y0 - h: y0 + h + 1,
                    int(x0 + shift_x) - h: int(x0 + shift_x) + h + 1] = \
                    star_value
            xs.append(x0)
            ys.append(y0)
            idx += 1
    rgba = np.zeros(lum.shape + (4,), dtype=np.uint8)
    rgba[..., 0] = rgba[..., 1] = rgba[..., 2] = lum
    rgba[..., 3] = 255
    out_base = str(tmp_path / name)
    png = out_base + ".png"
    Image.fromarray(np.flipud(rgba), mode="RGBA").save(png)
    avm = RT.faithful_avm(wcs, lum.shape)
    avm.embed(png, png + ".tagged")
    os.replace(png + ".tagged", png)
    world = wcs.pixel_to_world(np.array(xs, float), np.array(ys, float))
    refcat = str(tmp_path / f"{name}_refcat.fits")
    Table({"RA": world.ra.deg, "DEC": world.dec.deg,
           "refmag": np.linspace(10.0, 12.4, len(xs))}).write(refcat)
    return out_base, refcat


def test_box_peak_plateau_centroid_not_first_index():
    """Unit contract: a flat-top max region uses the plateau CENTROID;
    nanargmax's first index would sit at the plateau corner."""
    box = np.full((9, 9), 1.0)
    box[2:7, 2:7] = 5.0                      # 5x5 plateau centered at (4, 4)
    py, px = RT._box_peak(box, 4)
    assert (py, px) == (4.0, 4.0)
    # by construction the naive first-index peak is the plateau corner
    ay, ax = np.unravel_index(np.nanargmax(box), box.shape)
    assert (ay, ax) == (2, 2)
    assert np.hypot(ay - 4, ax - 4) > RT.STAR_PASS_MEDIAN_PX
    # plateau touching the border: gradient toward outside, no local peak
    edge = np.full((9, 9), 1.0)
    edge[0:5, 3:6] = 5.0
    assert RT._box_peak(edge, 4) is None


def test_star_positions_flat_top_passes_with_centroid(tmp_path):
    """End-to-end plateau bias: 5x5 flat-top stars exactly at their catalog
    positions PASS with the plateau centroid, while the first-index argmax
    peak would (by construction) sit ~2.8 px off and FAIL the 2 px gate."""
    out_base, refcat = _grid_png_with_avm(tmp_path, "plateau", plateau=5)
    res = RT.validate_star_positions(out_base, refcat)
    assert res["pass"] is True, res
    assert res["n_used"] == 25
    assert res["median_offset_px"] < 0.5
    # demonstrate the argmax failure mode on the same image: the first-index
    # "peak" of every 5x5 plateau is its corner, 2*sqrt(2) px from center
    assert np.hypot(2, 2) > RT.STAR_PASS_MEDIAN_PX


def test_star_positions_saturated_stars_skipped(tmp_path):
    """255-clipped stars are skipped when >= STAR_MIN_USED unsaturated stars
    remain; here the 5 saturated ones are also displaced +2 px, so including
    them would drag the tail while the 20 clean stars carry the verdict."""
    out_base, refcat = _grid_png_with_avm(tmp_path, "satgrid",
                                          sat_positions=(0, 6, 12, 18, 24))
    res = RT.validate_star_positions(out_base, refcat)
    assert res["n_saturated_skipped"] == 5
    assert res["n_used"] == 20               # exactly STAR_MIN_USED remain
    assert res["pass"] is True, res
    assert res["median_offset_px"] < 0.5


def test_star_positions_saturated_kept_when_too_few_remain(tmp_path):
    """With < STAR_MIN_USED unsaturated stars, saturated ones are NOT
    dropped (refusing to conclude from too few beats discarding data)."""
    out_base, refcat = _grid_png_with_avm(tmp_path, "allsat",
                                          sat_positions=tuple(range(25)))
    res = RT.validate_star_positions(out_base, refcat)
    assert res["n_saturated_skipped"] == 0
    assert res["n_used"] == 25
    assert res["median_offset_px"] == pytest.approx(2.0)


def test_star_positions_no_footprint_overlap_reason(tmp_path):
    """A catalog fully outside the image footprint reports a distinct
    reason, not just a generic zero-star fail."""
    from astropy.table import Table
    out_base, _refcat = _grid_png_with_avm(tmp_path, "nooverlap")
    farcat = str(tmp_path / "far_refcat.fits")
    Table({"RA": [10.0, 10.1], "DEC": [45.0, 45.1],
           "refmag": [10.0, 11.0]}).write(farcat)
    res = RT.validate_star_positions(out_base, farcat)
    assert res["pass"] is False
    assert res["n_selected"] == 0
    assert res["reason"] == "catalog does not overlap image footprint"


def test_batch_spec_dry_run(synthetic_pair, tmp_path, capsys):
    f212n, longp = synthetic_pair
    spec = tmp_path / "fields.json"
    spec.write_text(json.dumps({
        "out_dir": str(tmp_path),
        "fields": [{"name": "synth", "f212n_i2d": f212n, "long_i2d": longp,
                    "long_band": "F480M"}]}))
    rc = RT.main(["--fields-spec", str(spec), "--dry-run"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "would build synth" in out
    assert not os.path.exists(tmp_path / "synth_rgb.png")
