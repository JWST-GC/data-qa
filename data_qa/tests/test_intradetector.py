"""Intra-detector residual distortion (stage 8 intra-detector part + the all-pointing aggregate).

Synthetic frames: true star positions on the sky, a rotated linear detector WCS, and an injected
detector-frame distortion added to each frame's measured sky position.  The consensus catalogue
is the true positions.  The measurement must recover the injected pattern in detector axes and
read ~0 for zero distortion."""
import os

import numpy as np
import pytest
from astropy.table import Table

from data_qa import diagnostics as D
from data_qa import intradetector as I
from data_qa import intradetector_distortion as A
from data_qa.observations import Observation

RA0, DEC0 = 266.40, -29.00
PIX = 0.031                     # arcsec / pixel
NPIX = 2048


def _distortion(x, y, amp):
    """Injected detector-frame distortion in mas: a quadratic in x for dx, a sine in y for dy."""
    u = x / NPIX - 0.5; v = y / NPIX - 0.5
    return amp * (4 * u ** 2 - 1 / 3), amp * np.sin(2 * np.pi * v)


def _frame(rng, amp, n=4000, theta_deg=30.0, centre=(0.0, 0.0), bulk_mas=(40.0, -25.0),
           noise_mas=1.0):
    """(frame dict, consensus dict) for one synthetic exposure."""
    x = rng.uniform(0, NPIX, n); y = rng.uniform(0, NPIX, n)
    th = np.radians(theta_deg)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    # true tangent-plane position (arcsec) of each star from its detector position
    xi, eta = R @ np.vstack([(x - NPIX / 2) * PIX, (y - NPIX / 2) * PIX])
    xi += centre[0]; eta += centre[1]
    cosd = np.cos(np.radians(DEC0))
    ra_t = RA0 + xi / 3600 / cosd; de_t = DEC0 + eta / 3600
    ddx, ddy = _distortion(x, y, amp)                   # mas, detector axes
    dxi, deta = R @ np.vstack([ddx, ddy]) / 1000.0      # arcsec on the sky
    dxi += bulk_mas[0] / 1000 + rng.normal(0, noise_mas / 1000, n)
    deta += bulk_mas[1] / 1000 + rng.normal(0, noise_mas / 1000, n)
    flux = 10 ** rng.uniform(3, 5, n)
    frame = dict(x=x, y=y, ra=ra_t + dxi / 3600 / cosd, dec=de_t + deta / 3600, flux=flux,
                 pixscale=PIX, detector="nrca1")
    ref = dict(ra=ra_t, dec=de_t, flux=flux * 1.01)
    return frame, ref


def test_frame_residuals_recover_injected_distortion():
    rng = np.random.default_rng(3)
    frame, ref = _frame(rng, amp=3.0)
    r = I.frame_residuals(frame, ref)
    assert "dx" in r and r["npairs"] > 3000
    # bulk removed by the histogram peak (xcorr reports the shift onto the consensus)
    assert abs(r["bulk_dra"] + 40.0) < 2.0 and abs(r["bulk_ddec"] - 25.0) < 2.0
    tx, ty = _distortion(r["x"], r["y"], 3.0)
    ex = r["dx"] - np.median(r["dx"] - tx); ey = r["dy"] - np.median(r["dy"] - ty)
    assert np.median(np.abs(ex - tx)) < 1.0 and np.median(np.abs(ey - ty)) < 1.0


def test_stack_recovers_map_and_zero_distortion_reads_zero():
    rng = np.random.default_rng(5)
    for amp, check in ((2.0, "inj"), (0.0, "zero")):
        rs = [I.frame_residuals(*_frame(rng, amp, theta_deg=rng.uniform(0, 360)))
              for _ in range(4)]
        allr = I.concat(rs)
        s = I.stack_detector(allr["x"], allr["y"], allr["dx"], allr["dy"])
        if check == "inj":
            # truth binned the same way
            tx, ty = _distortion(allr["x"], allr["y"], amp)
            mx, my, _ = I.binned_map(allr["x"], allr["y"], tx, ty)
            true_rms = I.map_rms(mx - np.nanmean(mx), my - np.nanmean(my))
            assert abs(s["sysrms_debiased_mas"] - true_rms) < 0.25 * true_rms
            assert np.nanmax(np.abs(s["mdx"] - np.nanmean(s["mdx"]) - (mx - np.nanmean(mx)))) < 0.5
            assert s["significance"] > 5
        else:
            assert s["sysrms_debiased_mas"] < 0.1
            assert s["significance"] < 2


def test_frame_residuals_refuses_unmatched_frame():
    rng = np.random.default_rng(1)
    frame, ref = _frame(rng, 0.0)
    ref = dict(ra=ref["ra"] + 1.0, dec=ref["dec"], flux=ref["flux"])   # 1 deg away
    assert "reason" in I.frame_residuals(frame, ref)


# ------------------------------------------------------------------ files + stage 8 + aggregate
def _write_frame(path, frame, det):
    t = Table({"x_fit": frame["x"], "y_fit": frame["y"],
               "skycoord_centroid.ra": frame["ra"], "skycoord_centroid.dec": frame["dec"],
               "flux_fit": frame["flux"], "flux_err": frame["flux"] / 100.0,
               "flags": np.zeros(len(frame["x"]), int), "qfit": np.full(len(frame["x"]), 0.02)})
    t.meta["DETECTOR"] = det.upper(); t.meta["PIXSCALE"] = PIX
    t.write(path)


def _write_consensus(path, ref):
    n = len(ref["ra"])
    Table({"skycoord.ra": ref["ra"], "skycoord.dec": ref["dec"], "flux": ref["flux"],
           "flux_err": ref["flux"] / 100.0, "is_saturated": np.zeros(n, bool),
           "nmatch_good": np.full(n, 4)}).write(path)


@pytest.fixture
def catdir(tmp_path):
    """Two pointings x two detectors x three dithered exposures with an injected distortion."""
    rng = np.random.default_rng(11)
    d = tmp_path / "cats"; d.mkdir()
    for obs in ("001", "002"):
        frames = []
        for k, det in enumerate(("nrca1", "nrcb1")):
            for e in range(3):
                fr, ref = _frame(rng, 2.0, n=2500, centre=(k * 80.0, 0.0),
                                 theta_deg=10.0, bulk_mas=(e * 10.0, -e * 5.0))
                p = d / f"f212n_{det}_o{obs}_visit001_vgroup02101_exp0000{e + 1}_m3_daophot_basic.fits"
                _write_frame(p, fr, det); frames.append(ref)
        allref = {k: np.concatenate([f[k] for f in frames]) for k in ("ra", "dec", "flux")}
        _write_consensus(d / f"f212n_merged_o{obs}_indivexp_merged_m3_dao_basic_vetted.fits", allref)
    return d


def _obs():
    return Observation(program="10678", obs="001", target="GC Treasury", release_field="gc-treasury",
                       instrument="NIRCam", filters=["F212N", "F480M"])


def test_stage8_intradetector_figure_and_metrics(catdir, tmp_path, monkeypatch):
    import glob
    frames = sorted(glob.glob(str(catdir / "f212n_nrc*_o001_*_daophot_basic.fits")))
    cons = str(catdir / "f212n_merged_o001_indivexp_merged_m3_dao_basic_vetted.fits")
    monkeypatch.setattr(D, "OUTDIR", str(tmp_path / "out"))
    monkeypatch.setattr(D, "_intradet_inputs", lambda o, f: (frames, cons))
    png, m = D._stage8_intradetector(_obs(), "F212N")
    assert png and os.path.exists(png) and png.endswith("_stage8_intradet_f212n.png")
    assert m["available"] and m["n_frames_used"] == 6
    assert set(m["sysrms_debiased_mas"]) == {"nrca1", "nrcb1"}
    assert all(v > 0.5 for v in m["sysrms_debiased_mas"].values())
    assert "passed" not in m and "red_flag" not in m
    cap = D._caption_stage8_intra(m)
    assert "Intra-detector residual (F212N)" in cap and "#stage8-intradetector" in cap


def test_stage8_intradetector_skips_unreadable_inputs(catdir, tmp_path, monkeypatch):
    """A corrupt per-frame file or one missing a column is counted and skipped; an unreadable
    consensus makes the part n/a instead of failing stage 8."""
    import glob
    frames = sorted(glob.glob(str(catdir / "f212n_nrc*_o001_*_daophot_basic.fits")))
    cons = str(catdir / "f212n_merged_o001_indivexp_merged_m3_dao_basic_vetted.fits")
    bad = tmp_path / "corrupt.fits"; bad.write_bytes(b"not a fits file")
    nocol = tmp_path / "nocol.fits"; Table({"x_fit": [1.0]}).write(nocol)
    monkeypatch.setattr(D, "OUTDIR", str(tmp_path / "out"))
    monkeypatch.setattr(D, "_intradet_inputs", lambda o, f: (frames + [str(bad), str(nocol)], cons))
    png, m = D._stage8_intradetector(_obs(), "F212N")
    assert png and m["n_frames_used"] == 6
    assert sum(v for k, v in m["frames_failed"].items() if k.startswith("unreadable")) == 2
    monkeypatch.setattr(D, "_intradet_inputs", lambda o, f: (frames, str(bad)))
    png, m = D._stage8_intradetector(_obs(), "F212N")
    assert png is None and m["available"] is False and "unreadable" in m["na_reason"]


def test_stage8_intradetector_na_posts_nothing(monkeypatch):
    monkeypatch.setattr(D, "_intradet_inputs", lambda o, f: ([], None))
    png, m = D._stage8_intradetector(_obs(), "F212N")
    assert png is None and m["available"] is False and m["na_reason"]
    assert D._caption_stage8_intra(m) == ""


def test_stage8_intra_leads_when_interfilter_na(monkeypatch, tmp_path):
    monkeypatch.setattr(D, "OUTDIR", str(tmp_path))
    monkeypatch.setattr(D, "_interfilter_residuals", lambda o, f: None)
    monkeypatch.setattr(D, "_perfilter_interfilter_residuals", lambda o, f: None)
    monkeypatch.setattr(D, "_stage8_intradetector",
                        lambda o, f: ("/x/intra.png", dict(available=True, filter=f,
                                                           consensus="c.fits", n_frames=6,
                                                           n_frames_used=6,
                                                           sysrms_debiased_mas={"nrca1": 0.6},
                                                           sysrms_debiased_median_mas=0.6)))
    png, m = D.stage8_distortion(_obs(), "F212N")
    assert png == "/x/intra.png" and m["measurable"] is False and m["passed"] is None
    cap = D._caption_stage8(m)
    assert cap.startswith("**Stage 8 — distortion residuals (F212N).**")
    assert "inter-filter map is not applicable" in cap


def test_aggregate_tool_stacks_all_pointings(catdir, tmp_path):
    out = A.run_filter(str(catdir), "F212N", str(tmp_path / "agg"), nb=8, nproc=1)
    assert out["n_pointings"] == 2
    for det in ("nrca1", "nrcb1"):
        s = out["detectors"][det]
        assert s["n_pointings"] == 2 and s["sysrms_debiased_mas"] > 0.5
        assert s["tile_r_vs_others_median"] > 0.5          # the injected pattern repeats
    for f in ("intradet_f212n.json", "intradet_f212n_allpointings.png",
              "intradet_f212n_consistency.png", "intradet_f212n_residuals.npz"):
        assert (tmp_path / "agg" / f).exists()
