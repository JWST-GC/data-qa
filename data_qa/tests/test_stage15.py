"""Stage 15 (residual maps by photometry tool) tests.

Small synthetic residual maps for each tool are written to a temporary directory and the product
lookups are pointed at them, so the stage runs end to end and writes a real PNG.  The peppar
product is a bare per-frame residual whose WCS and unit come from a separate cal frame, as on
disk."""
import os

import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS

from data_qa import diagnostics as D
from data_qa.observations import Observation

RA0, DEC0 = 266.40, -29.00
NPIX, PIXAS = 200, 0.1          # 20" square maps
PHOTMJSR = 4.0                  # peppar fits in DN/s = MJy/sr / PHOTMJSR


def _obs():
    return Observation(program="10678", obs="066", target="GC Treasury", release_field="gc-treasury",
                       instrument="NIRCam", filters=["F212N", "F480M"])


def _wcs(rot_deg=0.0):
    w = WCS(naxis=2)
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    w.wcs.crval = [RA0, DEC0]
    w.wcs.crpix = [NPIX / 2 + 0.5, NPIX / 2 + 0.5]
    t = np.radians(rot_deg)
    s = PIXAS / 3600.0
    w.wcs.cd = [[-s * np.cos(t), s * np.sin(t)], [s * np.sin(t), s * np.cos(t)]]
    return w


def _sci(path, value, rng, rot_deg=0.0, bunit="MJy/sr", **cards):
    hdr = _wcs(rot_deg).to_header()
    hdr["BUNIT"] = bunit
    hdr.update(cards)
    data = (value + rng.normal(0, 1, (NPIX, NPIX))).astype(np.float32)
    fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(data, hdr, name="SCI")]).writeto(path)
    return str(path)


def _bare(path, value, rng):
    """peppar residual: a primary image with no WCS and no unit."""
    fits.PrimaryHDU((value + rng.normal(0, 1, (NPIX, NPIX))).astype(np.float64)).writeto(path)
    return str(path)


@pytest.fixture
def field(tmp_path, monkeypatch):
    rng = np.random.default_rng(2)
    prods = {t: {} for t in D._S15_TOOLS}
    ref = {}
    for f in ("F212N", "F480M"):
        ref[f] = _sci(tmp_path / f"{f}_i2d.fits", 10.0, rng)
        p = _sci(tmp_path / f"{f}_m7_residual_i2d.fits", 3.0, rng)
        prods["jicama"][f] = dict(tool="jicama", kind="mosaic", path=p, wcs_path=p, iteration="m7",
                                  tag="resbgsub_m7")
        cal = _sci(tmp_path / f"{f}_cal.fits", 0.0, rng, rot_deg=30.0, PHOTMJSR=PHOTMJSR)
        res = _bare(tmp_path / f"{f}_res_FINAL.fits", 0.0, rng)
        prods["peppar"][f] = [dict(tool="peppar", kind="frame", path=res, wcs_path=cal,
                                   iteration="FINAL", tag="single frame, FINAL")]
        p = _sci(tmp_path / f"{f}_sf_resid.fits", 0.0, rng)
        prods["starfinder"][f] = dict(tool="starfinder", kind="mosaic", path=p, wcs_path=p,
                                      iteration="", tag="")
        p = _sci(tmp_path / f"{f}_j1p_resid.fits", 0.0, rng)
        prods["jwst1pass"][f] = dict(tool="jwst1pass", kind="mosaic", path=p, wcs_path=p,
                                     iteration="", tag="")
    cosd = np.cos(np.radians(DEC0))
    dx, dy = rng.uniform(-4, 4, (2, 30))
    Table({"skycoord.ra": RA0 + dx / 3600.0 / cosd, "skycoord.dec": DEC0 + dy / 3600.0}).write(
        tmp_path / "cat.fits")

    def finder(tool):
        return lambda o, f: prods[tool].get(f, f"no {tool} product")
    monkeypatch.setattr(D, "OUTDIR", str(tmp_path / "out"))
    monkeypatch.setattr(D, "_mosaic_path", lambda o, f: ref.get(f))
    monkeypatch.setattr(D, "_s14_perfilter_catalog", lambda o, f, tag: str(tmp_path / "cat.fits"))
    monkeypatch.setattr(D, "_S15_FINDERS", {t: finder(t) for t in D._S15_TOOLS})
    return prods


def test_stage15_all_four_tools(field):
    png, m = D.stage15_residual_comparison(_obs(), "F212N", "F480M")
    assert png.endswith("_stage15.png") and os.path.exists(png)
    assert m["available"] is True and m["passed"] is None
    assert m["tools_present"] == ["jicama", "peppar", "starfinder", "jwst1pass"]
    assert m["tools_missing"] == {} and m["has_reference"] is True
    # median reported (not removed), noise measured on native pixels: unit-variance noise -> RMS and MAD ~ 1
    for t in m["tools_present"]:
        for f in ("F212N", "F480M"):
            d = m["per_tool"][t][f]
            k = PHOTMJSR if t == "peppar" else 1.0      # peppar DN/s rescaled to MJy/sr
            assert 0.7 * k < d["rms"] < 1.3 * k and 0.7 * k < d["mad_sigma"] < 1.3 * k
    assert abs(m["per_tool"]["jicama"]["F212N"]["median"] - 3.0) < 0.3
    # peppar's unit comes from the cal frame (the residual file carries none)
    assert m["per_tool"]["peppar"]["F212N"]["unit"] == "MJy/sr"
    # centre is reproducible (seeded by the obsid)
    _, m2 = D.stage15_residual_comparison(_obs(), "F212N", "F480M")
    assert (m2["center_ra"], m2["center_dec"]) == (m["center_ra"], m["center_dec"])


def test_stage15_peppar_without_photmjsr_stays_dn_per_s(tmp_path):
    rng = np.random.default_rng(3)
    cal = _sci(tmp_path / "cal.fits", 0.0, rng)
    with fits.open(cal, mode="update") as h:
        del h["SCI"].header["BUNIT"]
    res = _bare(tmp_path / "res.fits", 0.0, rng)
    mp = D._S15Map(dict(path=res, wcs_path=cal, iteration="FINAL", tag=""))
    assert mp.scale == 1.0 and mp.unit == "DN/s"
    mp.close()


def test_stage15_missing_tools_are_skipped(field, monkeypatch):
    finders = dict(D._S15_FINDERS)
    finders["starfinder"] = lambda o, f: "no StarFinder residual images on disk"
    finders["jwst1pass"] = lambda o, f: "no residual images written by jwst1pass"
    monkeypatch.setattr(D, "_S15_FINDERS", finders)
    png, m = D.stage15_residual_comparison(_obs(), "F212N", "F480M")
    assert png and m["tools_present"] == ["jicama", "peppar"]
    assert m["tools_missing"] == {"starfinder": "no StarFinder residual images on disk",
                                  "jwst1pass": "no residual images written by jwst1pass"}
    cap = D.caption_for(15, m)
    assert "Skipped: starfinder (no StarFinder" in cap and "jwst1pass (no residual" in cap


def test_stage15_tool_in_one_filter_keeps_its_column(field, monkeypatch):
    finders = dict(D._S15_FINDERS)
    finders["starfinder"] = lambda o, f: (field["starfinder"][f] if f == "F212N" else "none here")
    monkeypatch.setattr(D, "_S15_FINDERS", finders)
    _, m = D.stage15_residual_comparison(_obs(), "F212N", "F480M")
    assert "starfinder" in m["tools_present"] and list(m["per_tool"]["starfinder"]) == ["F212N"]


def test_stage15_fewer_than_two_tools_posts_nothing(field, monkeypatch):
    finders = {t: (lambda o, f, t=t: f"no {t}") for t in D._S15_TOOLS}
    finders["jicama"] = D._S15_FINDERS["jicama"]
    monkeypatch.setattr(D, "_S15_FINDERS", finders)
    png, m = D.stage15_residual_comparison(_obs(), "F212N", "F480M")
    assert png is None and m["available"] is False and m["passed"] is None
    assert m["tools_present"] == ["jicama"] and set(m["tools_missing"]) == {"peppar", "starfinder",
                                                                            "jwst1pass"}
    assert "pending" in D.caption_for(15, m)


def test_stage15_tool_not_covering_centre_is_skipped(field, tmp_path, monkeypatch):
    """A map that misses the cutout (here: far off on the sky) is skipped with a reason."""
    off = tmp_path / "off.fits"
    hdr = _wcs().to_header(); hdr["CRVAL1"] = RA0 + 1.0
    fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(np.zeros((NPIX, NPIX), np.float32), hdr,
                                                   name="SCI")]).writeto(off)
    finders = dict(D._S15_FINDERS)
    finders["jwst1pass"] = lambda o, f: dict(tool="jwst1pass", kind="mosaic", path=str(off),
                                             wcs_path=str(off), iteration="", tag="")
    monkeypatch.setattr(D, "_S15_FINDERS", finders)
    _, m = D.stage15_residual_comparison(_obs(), "F212N", "F480M")
    assert "jwst1pass" not in m["tools_present"]
    assert "covers the cutout" in m["tools_missing"]["jwst1pass"]


def test_stage15_peppar_lookup_maps_residual_to_cal(tmp_path, monkeypatch):
    """The real peppar finder: obs-scoped residuals, FINAL before iter1, WCS from the cal frame."""
    root = tmp_path / "gc-treasury"
    pdir = root / "peppar" / "o066" / "F212N" / "NRCA1"
    pdir.mkdir(parents=True)
    (root / "F212N" / "pipeline").mkdir(parents=True)
    exp = "jw10678066001_02101_00001_nrca1"
    for name in (f"{exp}_cal_gc-treasury_res_FINAL.fits", f"{exp}_cal_gc-treasury_iter1_res.fits",
                 "jw10678067001_02101_00001_nrca1_cal_gc-treasury_res_FINAL.fits"):
        (pdir / name).write_bytes(b"")
    (root / "F212N" / "pipeline" / f"{exp}_cal.fits").write_bytes(b"")
    monkeypatch.setattr(D, "_PEPPAR_DEFAULT_ROOT", str(tmp_path))
    r = D._s15_peppar(_obs(), "F212N")
    assert len(r) == 1 and r[0]["iteration"] == "FINAL"
    assert r[0]["path"].endswith("_res_FINAL.fits") and r[0]["wcs_path"].endswith(f"{exp}_cal.fits")


def test_stage15_jwst1pass_without_residuals_gives_reason(tmp_path, monkeypatch):
    d = tmp_path / "gc-treasury" / "jwst1pass" / "F212N" / "o066"
    d.mkdir(parents=True)
    (d / "NRCA1_psf.fits").write_bytes(b""); (d / "LOG.psfperts.fits").write_bytes(b"")
    monkeypatch.setattr(D, "_JWST1PASS_DEFAULT_ROOT", str(tmp_path))
    assert "no residual images" in D._s15_jwst1pass(_obs(), "F212N")


def test_stage15_caption_and_default_stage_list(field):
    _, m = D.stage15_residual_comparison(_obs(), "F212N", "F480M")
    cap = D.caption_for(15, m)
    assert "qa_methods.md#stage15" in cap and "No pass/fail" in cap
    assert "| jicama | F212N |" in cap and "| peppar | F480M |" in cap
    assert 15 in D._STAGES_NEEDING_SW
    here = os.path.dirname(__file__)
    sh = open(os.path.join(here, "..", "..", "scripts", "refresh_all_issues.sh")).read()
    assert "11 12 13 14 15}" in sh


def _jicama_dir(tmp_path, monkeypatch, tags, quarantined=()):
    """Empty residual-mosaic files for ``tags`` (and quarantined copies for ``quarantined``)."""
    d = tmp_path / "F212N" / "pipeline"
    d.mkdir(parents=True)
    stem = "jw10678-o066_t001_nircam_clear-f212n-merged_{}_daophot_basic_mergedcat_residual_i2d"
    for t in tags:
        (d / (stem.format(t) + ".fits")).write_bytes(b"")
    for t in quarantined:
        (d / (stem.format(t) + "_im0_badastrom.fits")).write_bytes(b"")
    monkeypatch.setattr(D, "_field_dirs", lambda o, *rels: [str(d)])


def test_stage15_jicama_hybrid_psf_outranks_higher_plain_iteration(tmp_path, monkeypatch):
    """A tile mid-re-catalogue (o068: hybpsf m2/m3 beside plain m4-m7) shows hybpsf_m3."""
    _jicama_dir(tmp_path, monkeypatch, ["hybpsf_m2", "hybpsf_m3", "m2", "m3", "m4",
                                        "resbgsub_m5", "resbgsub_m6", "resbgsub_m7"])
    r = D._s15_jicama(_obs(), "F212N")
    assert r["tag"] == "hybpsf_m3" and r["generation"] == "hybrid PSF" and r["iteration"] == "m3"


def test_stage15_jicama_skips_quarantined_and_records_them(tmp_path, monkeypatch):
    # o067-like: the whole plain generation quarantined, hybrid m2/m3 live
    _jicama_dir(tmp_path, monkeypatch, ["hybpsf_m2", "hybpsf_m3"],
                quarantined=["m2", "m3", "m4", "resbgsub_m5", "resbgsub_m6", "resbgsub_m7"])
    r = D._s15_jicama(_obs(), "F212N")
    assert r["tag"] == "hybpsf_m3"
    assert r["quarantined"] == ["resbgsub_m7", "resbgsub_m6", "resbgsub_m5", "m4", "m3", "m2"]


def test_stage15_jicama_plain_fallback_skips_quarantined_top(tmp_path, monkeypatch):
    _jicama_dir(tmp_path, monkeypatch, ["m4", "resbgsub_m6"],
                quarantined=["hybpsf_m3", "resbgsub_m7"])
    r = D._s15_jicama(_obs(), "F212N")
    assert r["tag"] == "resbgsub_m6" and r["generation"] == "plain PSF"
    assert r["quarantined"] == ["hybpsf_m3", "resbgsub_m7"]       # hybrid first, high n first
    note = D._s15_jicama_note({"F212N": dict(tag=r["tag"], generation=r["generation"],
                                             quarantined_skipped=r["quarantined"])})
    assert note == "F212N resbgsub_m6 (plain PSF), quarantined and skipped: hybpsf_m3, resbgsub_m7"


def test_stage15_jicama_all_quarantined_gives_reason(tmp_path, monkeypatch):
    _jicama_dir(tmp_path, monkeypatch, [], quarantined=["resbgsub_m7"])
    assert "quarantined" in D._s15_jicama(_obs(), "F212N")


def test_stage15_displays_residuals_without_median_removed(field, monkeypatch):
    """Every residual panel shows the residual as written: jicama's +3 pedestal reaches the
    resampled display unchanged, so all panels of a row sit on exactly the same scale."""
    import reproject
    seen = []
    real = reproject.reproject_interp

    def spy(inp, *a, **k):
        seen.append(float(np.nanmedian(inp[0])))
        return real(inp, *a, **k)
    monkeypatch.setattr(reproject, "reproject_interp", spy)
    _, m = D.stage15_residual_comparison(_obs(), "F212N", "F480M")
    jic = m["per_tool"]["jicama"]["F212N"]["median"]
    assert abs(jic - 3.0) < 0.3
    assert any(abs(v - jic) < 1e-6 for v in seen)
    assert "no median removed" in D.caption_for(15, m)


def test_stage15_peppar_wcs_from_aligned_frame(tmp_path, monkeypatch):
    """The cal frame keeps the pointing WCS; the realigned crf of the same exposure (RAOFFSET in
    its header) supplies the WCS, a stale crf without RAOFFSET is skipped, and the unit still comes
    from the cal frame (JWST-GC/data-qa#6: cloudc cal WCS 7.9" off the mosaic)."""
    rng = np.random.default_rng(4)
    root = tmp_path / "gc-treasury"
    pdir = root / "peppar" / "o066" / "F212N" / "NRCA1"
    pipe = root / "F212N" / "pipeline"
    pdir.mkdir(parents=True); pipe.mkdir(parents=True)
    exp = "jw10678066001_02101_00001_nrca1"
    _bare(pdir / f"{exp}_cal_gc-treasury_res_FINAL.fits", 0.0, rng)
    cal = _sci(pipe / f"{exp}_cal.fits", 0.0, rng, PHOTMJSR=PHOTMJSR)
    stale = _sci(pipe / f"{exp}_align_o066_crf.fits", 0.0, rng, bunit="junk")
    monkeypatch.setattr(D, "_PEPPAR_DEFAULT_ROOT", str(tmp_path))
    assert D._s15_aligned_frame(cal, "066") is None
    r = D._s15_peppar(_obs(), "F212N")
    assert r[0]["wcs_path"] == cal and r[0]["wcs_aligned"] is False
    good = _sci(pipe / f"{exp}_destreak_o066_crf.fits", 0.0, rng, bunit="junk", RAOFFSET=0.5,
                DEOFFSET=7.9, CRVAL2=DEC0 + 7.9 / 3600.0)
    assert D._s15_aligned_frame(cal, "066") == good and stale != good
    r = D._s15_peppar(_obs(), "F212N")
    assert r[0]["wcs_path"] == good and r[0]["unit_path"] == cal and r[0]["wcs_aligned"] is True
    mp = D._S15Map(r[0])
    assert mp.scale == PHOTMJSR and mp.unit == "MJy/sr"
    assert np.isclose(mp.wcs.wcs.crval[1], DEC0 + 7.9 / 3600.0)
    mp.close()


def test_stage15_caption_flags_cal_wcs_fallback():
    m = dict(available=True, tools_present=["jicama", "peppar"], center_ra=266.4, center_dec=-29.0,
             sw="F212N", lw="F480M",
             per_tool={"peppar": {"F212N": dict(file="a", rms=1, mad_sigma=1, unit="MJy/sr",
                                                wcs_aligned=False),
                                  "F480M": dict(file="b", rms=1, mad_sigma=1, unit="MJy/sr",
                                                wcs_aligned=True)}})
    cap = D.caption_for(15, m)
    assert "F212N: no realigned frame found" in cap and "F480M: no realigned" not in cap
