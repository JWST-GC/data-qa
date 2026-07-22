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
