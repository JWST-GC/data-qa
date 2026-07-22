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
