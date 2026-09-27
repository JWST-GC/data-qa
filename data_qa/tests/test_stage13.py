"""Stage 13 (neighbour-overlap agreement) tests.

A synthetic mosaic program is laid out on disk under a temporary QA base: one tile under test and
neighbours that overlap it by a strip, each with an ``S_REGION`` i2d footprint and (optionally) an
m8_dedup catalogue.  The shared star field is re-observed by every tile, with a known bulk offset
or zero-point shift injected into one neighbour."""
import os

import numpy as np
import pytest

from data_qa import diagnostics as D
from data_qa.observations import Observation

PROG = "10678"
FIELD = "gc-treasury"
RA0, DEC0 = 266.40, -29.00
TILE_AS = 60.0          # tile side, arcsec
STEP_AS = 45.0          # tile spacing in RA -> 15" overlap strips


def _obs(obs):
    return Observation(program=PROG, obs=obs, target="GC Treasury", release_field=FIELD,
                       instrument="NIRCam", filters=["F212N", "F480M"])


def _tile_vertices(ix):
    """Square footprint (RA, Dec deg) of the tile ``ix`` steps east of RA0."""
    cosd = np.cos(np.radians(DEC0))
    xc = ix * STEP_AS
    h = TILE_AS / 2
    xy = np.array([[xc - h, -h], [xc + h, -h], [xc + h, h], [xc - h, h]])
    return np.c_[RA0 + xy[:, 0] / 3600 / cosd, DEC0 + xy[:, 1] / 3600]


def _write_i2d(base, obs, verts):
    from astropy.io import fits
    d = f"{base}/{FIELD}/F212N/pipeline"
    os.makedirs(d, exist_ok=True)
    sreg = "POLYGON ICRS " + " ".join(f"{a:.9f} {b:.9f}" for a, b in verts)
    hdr = fits.Header(); hdr["S_REGION"] = sreg
    fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(np.zeros((2, 2), np.float32), hdr, name="SCI")]
                 ).writeto(f"{d}/jw{int(PROG):05d}-o{obs}_t001_nircam_clear-f212n-merged_i2d.fits")


def _stars(n=6000, seed=1):
    """One shared star field covering every tile (RA, Dec deg; flux Jy)."""
    rng = np.random.default_rng(seed)
    cosd = np.cos(np.radians(DEC0))
    x = rng.uniform(-TILE_AS, 3 * STEP_AS + TILE_AS, n)
    y = rng.uniform(-TILE_AS / 2, TILE_AS / 2, n)
    flux = 10 ** rng.uniform(-5.0, -3.0, n)
    return RA0 + x / 3600 / cosd, DEC0 + y / 3600, flux


def _write_m8(base, obs, verts, stars, dra_mas=0.0, ddec_mas=0.0, dmag=0.0, seed=0):
    """An m8_dedup catalogue holding the stars inside ``verts``, re-observed with 2 mas / 1 %
    noise, shifted by (dra_mas, ddec_mas) and dimmed by ``dmag``."""
    from astropy.table import Table
    from matplotlib.path import Path as MPath
    ra, dec, flux = stars
    rng = np.random.default_rng(seed)
    inside = MPath(verts).contains_points(np.c_[ra, dec])
    ra, dec, flux = ra[inside], dec[inside], flux[inside]
    cosd = np.cos(np.radians(DEC0))
    n = len(ra)
    ra = ra + (dra_mas + rng.normal(0, 2, n)) / 3.6e6 / cosd
    dec = dec + (ddec_mas + rng.normal(0, 2, n)) / 3.6e6
    flux = flux * 10 ** (-0.4 * dmag) * (1 + rng.normal(0, 0.01, n))
    t = Table()
    t["skycoord_f212n.ra"] = ra
    t["skycoord_f212n.dec"] = dec
    t["flux_jy_f212n"] = flux
    t["eflux_jy_f212n"] = flux / 100.0
    t["mag_vega_f212n"] = -2.5 * np.log10(flux / 795.0)
    t["qfit_f212n"] = np.full(n, 0.05)
    t["is_saturated_f212n"] = np.zeros(n, bool)
    t["near_saturated_f212n_f212n"] = np.zeros(n, bool)
    t["replaced_saturated_f212n"] = np.zeros(n, bool)
    t["independently_detected_f212n"] = np.ones(n, bool)
    d = f"{base}/{FIELD}/catalogs"
    os.makedirs(d, exist_ok=True)
    t.write(f"{d}/basic_merged_indivexp_photometry_tables_merged_resbgsub_m8_dedup_o{obs}.fits")


@pytest.fixture
def program(tmp_path, monkeypatch):
    """Tiles o040 (ix 0, this), o041 (ix 1), o042 (ix -1), o043 (ix 2, not touching o040) and a
    far tile o046.  Returns a helper that writes the m8 catalogues requested by the test."""
    base = str(tmp_path / "base")
    monkeypatch.setattr(D, "BASE", base)
    monkeypatch.setattr(D, "OUTDIR", str(tmp_path / "out"))
    monkeypatch.setenv("QA_DOWNLOAD_DIR", str(tmp_path / "dl"))
    layout = {"040": 0, "041": 1, "042": -1, "043": 2, "046": 40}
    verts = {o: _tile_vertices(ix) for o, ix in layout.items()}
    for o, v in verts.items():
        _write_i2d(base, o, v)
    stars = _stars()

    def cat(obs, **kw):
        _write_m8(base, obs, verts[obs], stars, seed=int(obs), **kw)
    return cat


def test_sregion_parse_and_neighbour_discovery(program):
    mine, nbrs = D._overlapping_neighbors(_obs("040"), "F212N")
    assert mine is not None and mine.shape == (4, 2)
    got = {n["obs"]: n["area_arcmin2"] for n in nbrs}
    assert set(got) == {"041", "042"}                    # o043 only touches o041; o046 is far
    strip = (TILE_AS - STEP_AS) * TILE_AS / 3600.0       # 15" x 60" overlap
    assert all(abs(a - strip) < 0.01 for a in got.values())


def test_neighbour_agreement_recovers_injected_offset_and_dmag(program):
    program("040"); program("041", dra_mas=30.0, ddec_mas=-15.0, dmag=0.15)
    mine, nbrs = D._overlapping_neighbors(_obs("040"), "F212N")
    nb = next(n for n in nbrs if n["obs"] == "041")
    cat = D._m8_catalog_path
    a = D._load_nb_sample(cat(_obs("040")), "F212N", nb["overlap_xy"], nb["frame"])
    b = D._load_nb_sample(cat(_obs("041")), "F212N", nb["overlap_xy"], nb["frame"])
    res = D._neighbor_agreement(a, b)
    assert res is not None and res["n_pairs"] > 200
    # this − neighbour: the neighbour was shifted by (+30, −15) and dimmed by 0.15 mag
    assert res["dra_mas"] == pytest.approx(-30.0, abs=1.0)
    assert res["ddec_mas"] == pytest.approx(15.0, abs=1.0)
    assert res["dmag_bulk"] == pytest.approx(-0.15, abs=0.01)
    assert res["astrom_scatter_mas"] == pytest.approx(2 * np.sqrt(2), rel=0.3)
    assert D._nb_flagged(res)


def test_stage13_flags_only_the_disagreeing_neighbour(program):
    program("040"); program("041", dra_mas=25.0); program("042")
    png, m = D.stage13_neighbor_overlap(_obs("040"), "F212N", "F480M")
    assert os.path.exists(png)
    nb = m["neighbors"]
    assert nb["041"]["status"] == "disagree" and nb["041"]["offset_mas"] > 20
    assert nb["042"]["status"] == "agree" and nb["042"]["offset_mas"] < 3
    assert abs(nb["042"]["dmag_bulk"]) < 0.01
    assert m["passed"] is False and m["red_flag"] and m["flagged_neighbors"] == ["041"]
    labels = [lab for lab, _ in m["extra_figures"]]
    assert labels[0].startswith("neighbour o041")        # worst first
    cap = D.caption_for(13, m)
    assert "| o041 |" in cap and "| o042 |" in cap and "Vega" in cap and "🚩" in cap


def test_stage13_zeropoint_offset_alone_flags(program):
    program("040"); program("041", dmag=0.12); program("042")
    _, m = D.stage13_neighbor_overlap(_obs("040"), "F212N", "F480M")
    assert m["neighbors"]["041"]["status"] == "disagree"
    assert m["neighbors"]["041"]["offset_mas"] < 3
    assert m["passed"] is False


def test_stage13_all_agree_passes(program):
    program("040"); program("041", dra_mas=5.0, dmag=0.03); program("042")
    _, m = D.stage13_neighbor_overlap(_obs("040"), "F212N", "F480M")
    assert m["passed"] is True and not m.get("red_flag") and m["n_graded"] == 2


def test_stage13_neighbour_missing_m8_is_not_graded_and_never_flags(program):
    program("040"); program("042")                     # o041 has a footprint but no m8 yet
    _, m = D.stage13_neighbor_overlap(_obs("040"), "F212N", "F480M")
    assert m["neighbors"]["041"]["status"] == "pending"
    assert m["n_pending"] == 1 and m["n_graded"] == 1
    assert m["passed"] is True and not m.get("red_flag")
    assert "not graded" in D.caption_for(13, m)


def test_stage13_no_neighbour_with_m8_is_neutral(program):
    program("040")
    _, m = D.stage13_neighbor_overlap(_obs("040"), "F212N", "F480M")
    assert m["passed"] is None and not m.get("red_flag") and m["n_graded"] == 0


def test_stage13_this_obs_missing_m8_is_pending(program):
    program("041"); program("042")
    png, m = D.stage13_neighbor_overlap(_obs("040"), "F212N", "F480M")
    assert png is None and m["available"] is False and m["passed"] is None
    assert "m8" in m["na_reason"]


def test_stage13_isolated_tile_is_not_applicable(program):
    program("046")
    png, m = D.stage13_neighbor_overlap(_obs("046"), "F212N", "F480M")
    assert png is None and m["available"] is False and m["n_neighbors"] == 0


def test_stage13_prefers_m8_dedup_over_plain_m8(program, tmp_path):
    program("040")
    d = f"{D.BASE}/{FIELD}/catalogs"
    plain = f"{d}/basic_merged_indivexp_photometry_tables_merged_resbgsub_m8_o040.fits"
    with open(plain, "wb") as fh:
        fh.write(b"")
    os.utime(plain, (4e9, 4e9))                         # newer than the dedup
    assert D._m8_catalog_path(_obs("040")).endswith("m8_dedup_o040.fits")


def test_stage13_dispatch_and_sw_gate():
    png, m = D._dispatch_stage(_obs("040"), 13, None, "F480M")
    assert png is None and m["available"] is False
