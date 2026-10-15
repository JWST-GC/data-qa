"""Network zero-point solver tests (data_qa.neighbor_zeropoints) on synthetic overlap graphs with
known zero points and a known NRCA - NRCB module term."""
import os

import numpy as np

from data_qa import neighbor_zeropoints as Z


def _grid(n=4):
    """Tiles on an n x n grid, each overlapping its right and upper neighbour."""
    tiles = [f"{k:03d}" for k in range(n * n)]
    pairs = []
    for r in range(n):
        for c in range(n):
            k = r * n + c
            if c + 1 < n:
                pairs.append((tiles[k], tiles[k + 1]))
            if r + 1 < n:
                pairs.append((tiles[k], tiles[k + n]))
    return tiles, pairs


def test_tile_solve_recovers_zero_points_with_mean_zero_gauge():
    rng = np.random.default_rng(3)
    tiles, pairs = _grid()
    z = dict(zip(tiles, rng.normal(0, 0.03, len(tiles))))
    edges = [dict(a=a, b=b, d=z[a] - z[b] + rng.normal(0, 0.001), err=0.001) for a, b in pairs]
    s = Z.solve_network(edges)
    true = np.array([z[n] for n in s["nodes"]])
    true -= true.mean()
    assert abs(np.mean(s["z"])) < 1e-10                      # gauge
    assert np.allclose(s["z"], true, atol=0.003)
    assert 0.3 < s["chi2_dof"] < 3.0 and s["sys_floor"] == 0.0
    assert len(s["components"]) == 1


def test_error_floor_brings_chi2_to_one():
    rng = np.random.default_rng(4)
    tiles, pairs = _grid(5)
    edges = [dict(a=a, b=b, d=rng.normal(0, 0.01), err=0.001) for a, b in pairs]
    s = Z.solve_network(edges)
    assert s["chi2_dof"] > 10
    assert 0.003 < s["sys_floor"] < 0.03
    assert np.all(s["z_err"] > s["z_err_formal"])


def test_disconnected_components_each_get_their_own_gauge():
    edges = [dict(a="a", b="b", d=0.1, err=0.01), dict(a="c", b="d", d=-0.2, err=0.01)]
    s = Z.solve_network(edges)
    z = dict(zip(s["nodes"], s["z"]))
    assert len(s["components"]) == 2
    assert np.isclose(z["a"], 0.05) and np.isclose(z["b"], -0.05)
    assert np.isclose(z["c"], -0.1) and np.isclose(z["d"], 0.1)


def _module_edges(rng, delta, noise=0.0005):
    tiles, pairs = _grid()
    z = dict(zip(tiles, rng.normal(0, 0.02, len(tiles))))
    zm = {(t, m): z[t] + Z._SIGN[m] * delta / 2 for t in tiles for m in Z.MODULES}
    edges = []
    for a, b in pairs:
        for ma in Z.MODULES:
            for mb in Z.MODULES:
                edges.append(dict(a=f"{a}:{ma}", b=f"{b}:{mb}",
                                  d=zm[(a, ma)] - zm[(b, mb)] + rng.normal(0, noise), err=noise))
    for t in tiles[::3]:
        edges.append(dict(a=f"{t}:nrca", b=f"{t}:nrcb", kind="intra",
                          d=zm[(t, "nrca")] - zm[(t, "nrcb")] + rng.normal(0, noise), err=noise))
    for e in edges:
        e.setdefault("kind", "module")
    return tiles, z, edges


def test_tile_plus_global_module_term():
    rng = np.random.default_rng(5)
    tiles, z, edges = _module_edges(rng, delta=0.04)
    s = Z.solve_network(edges, node_of=lambda k: k.split(":")[0],
                        sign_of=lambda k: Z._SIGN[k.split(":")[1]])
    assert abs(s["delta"] - 0.04) < 3 * s["delta_err"] + 1e-4
    true = np.array([z[n] for n in s["nodes"]])
    assert np.allclose(s["z"], true - true.mean(), atol=0.002)


def test_solve_filter_per_tile_module_offsets():
    rng = np.random.default_rng(6)
    tiles, z, edges = _module_edges(rng, delta=-0.03)
    merged = [dict(a=a, b=b, kind="merged", d=z[a] - z[b], err=0.001)
              for a, b in _grid()[1]]
    m = dict(filter="F212N", tiles=tiles, edges=edges + merged, meta={t: {} for t in tiles})
    sol = Z.solve_filter(m)
    v = np.array([x[0] for x in sol["ab_per_tile"].values()])
    assert len(v) == len(tiles) and np.allclose(v, -0.03, atol=0.003)
    summ = Z.summarise(m, sol)
    assert abs(summ["ab_global"] + 0.03) < 0.002
    assert summ["n_edges"] == dict(merged=len(merged), module=len(edges) - len(tiles[::3]),
                                   intra=len(tiles[::3]))
    assert summ["tile_n_components"] == 1
    assert np.isclose(summ["tile_zp_rms_largest"], np.std([z[t] for t in tiles]), atol=1e-6)


def test_summary_reports_rms_per_component():
    edges = [dict(a="a", b="b", kind="merged", d=0.2, err=0.01),
             dict(a="b", b="c", kind="merged", d=0.0, err=0.01),
             dict(a="x", b="y", kind="merged", d=0.02, err=0.01)]
    m = dict(filter="F212N", tiles=["a", "b", "c", "x", "y"], edges=edges, meta={})
    summ = Z.summarise(m, Z.solve_filter(m))
    comps = summ["tile_zp_rms_components"]
    assert [c["n"] for c in comps] == [3, 2]
    assert np.isclose(comps[1]["rms"], 0.01) and np.isclose(comps[1]["range"], 0.02)
    assert summ["tile_zp_rms_largest"] == comps[0]["rms"]


def test_cache_key_tracks_inputs_and_arguments(tmp_path, monkeypatch):
    monkeypatch.setattr(Z.D, "_footprint_path", lambda o, f: None)
    for o in ("001", "002"):
        for mod in ("merged",) + Z.MODULES:
            (tmp_path / Z._CAT_FMT.format(mod=mod, obs=o)).write_text("x")
    k1 = Z.cache_key(str(tmp_path), "F212N", ["001", "002"])
    assert k1 == Z.cache_key(str(tmp_path), "F212N", ["001", "002"])
    assert k1 != Z.cache_key(str(tmp_path), "F212N", ["001"])
    assert k1 != Z.cache_key(str(tmp_path), "F480M", ["001", "002"])
    p = tmp_path / Z._CAT_FMT.format(mod="nrca", obs="002")
    os.utime(p, (1e9, 1e9))
    assert k1 != Z.cache_key(str(tmp_path), "F212N", ["001", "002"])


# ------------------------------------------------------------------- within-tile gradient (#362)
_DEC0 = -29.0
_STEP = 2.0 / 60.0              # tile spacing (deg of sky); tiles 2.4' wide overlap by 0.4'


def _gradient_field(rng, g, n=4, zsig=0.0, noise=0.0005):
    """Tiles on an n x n sky grid at one roll; every catalogue magnitude carries z_tile + g . c with
    c the star's (east, north) arcmin offset from its tile centre.  Edges sit at the overlap centre."""
    cosd = np.cos(np.radians(_DEC0))
    tiles, pairs = _grid(n)
    cen = {t: (266.0 + (k % n) * _STEP / cosd, _DEC0 + (k // n) * _STEP) for k, t in enumerate(tiles)}
    meta = {t: dict(vertices=[[c[0] + sx * 0.02 / cosd, c[1] + sy * 0.02]
                              for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))], pa_v3=90.15)
            for t, c in cen.items()}
    z = dict(zip(tiles, rng.normal(0, zsig, len(tiles))))

    def off(t, ra, dec):
        return np.array(Z._offset_arcmin(ra, dec, cen[t]))
    edges = []
    for a, b in pairs:
        ra, dec = 0.5 * (cen[a][0] + cen[b][0]), 0.5 * (cen[a][1] + cen[b][1])
        d = z[a] - z[b] + np.dot(g, off(a, ra, dec) - off(b, ra, dec)) + rng.normal(0, noise)
        edges.append(dict(a=a, b=b, kind="merged", d=float(d), err=noise, ra=ra, dec=dec))
    return tiles, z, edges, meta


def _reference(rng, tiles, z, g, nstar=400, sig=0.08):
    out = []
    for t in tiles:
        x, y = rng.uniform(-1.2, 1.2, (2, nstar))
        ks = rng.uniform(14.5, 16.5, nstar)
        # reddening that rises to the north inside every tile: H - Ks follows it, and the colour
        # term must keep it out of G
        hk = 1.5 + 0.3 * y + rng.normal(0, 0.3, nstar)
        d = (0.3 + z[t] + g[0] * x + g[1] * y + 0.02 * (ks - 15.5) - 0.4 * (hk - 1.5)
             + rng.normal(0, sig, nstar))
        d[:10] += 2.0                                       # blends / mismatches to be clipped
        out.append(dict(obs=t, x=x.tolist(), y=y.tolist(), d=d.tolist(), ks=ks.tolist(),
                        hk=hk.tolist()))
    return dict(filter="F212N", ks_range=[14.5, 16.5], tiles=out)


def test_gradient_and_ramp_are_degenerate_in_the_overlaps():
    """A pure within-tile gradient shows up in the tile-only model as a ramp along the grid."""
    rng = np.random.default_rng(10)
    g = np.array([-0.03, 0.003])
    tiles, z, edges, meta = _gradient_field(rng, g)
    t = Z.solve_network(edges)
    zt = dict(zip(t["nodes"], t["z"]))
    cen = {o: Z.tile_centre(meta[o]) for o in tiles}
    c0 = np.mean(list(cen.values()), 0)
    x, y = Z._offset_arcmin([cen[o][0] for o in tiles], [cen[o][1] for o in tiles], c0)
    pred = -(g[0] * x + g[1] * y)
    assert np.allclose([zt[o] for o in tiles], pred - pred.mean(), atol=0.002)
    assert t["chi2_dof"] < 3                                # fits as well as the gradient does


def test_reference_fit_recovers_gradient_and_tile_offsets():
    rng = np.random.default_rng(11)
    g = np.array([-0.027, 0.002])
    tiles, z, _, _ = _gradient_field(rng, g, zsig=0.03)
    ref = Z.fit_reference(_reference(rng, tiles, z, g), rng=np.random.default_rng(0))
    assert np.allclose(ref["g"], g, atol=3 * max(ref["g_err"]) + 1e-4)
    assert abs(ref["ks_slope"] - 0.02) < 0.005 and ref["n_tiles"] == len(tiles)
    assert abs(ref["hk_slope"] + 0.4) < 0.02
    t = np.array([ref["t"][o][0] for o in tiles]); zt = np.array([z[o] for o in tiles])
    assert np.allclose(t, zt - zt.mean(), atol=0.01)
    assert ref["n_stars"] <= len(tiles) * 390                # the injected outliers are clipped


def test_tile_plus_gradient_model_removes_the_ramp():
    rng = np.random.default_rng(12)
    g = np.array([-0.03, 0.003])
    tiles, z, edges, meta = _gradient_field(rng, g, zsig=0.01)
    ref = Z.fit_reference(_reference(rng, tiles, z, g), rng=np.random.default_rng(0))
    m = dict(filter="F212N", tiles=tiles, edges=edges, meta=meta)
    sol = Z.solve_filter(m, ref)
    tg = sol["tile+G"]
    zg = dict(zip(tg["nodes"], tg["z"]))
    zt = np.array([z[o] for o in tiles])
    # the residual ramp left is the prior's uncertainty on G times the grid extent
    assert np.allclose([zg[o] for o in tiles], zt - zt.mean(), atol=0.01)
    assert np.all(np.abs(np.asarray(tg["g"]) - g) < 3 * np.asarray(tg["g_err"]) + 1e-4)
    summ = Z.summarise(m, sol)
    assert summ["grad_tile_zp_rms_largest"] < 0.5 * summ["tile_zp_rms_largest"]
    assert summ["grad_vs_ref_r"] > 0.5 and summ["pa_v3_range"] == [90.15, 90.15]
    assert "ref_g" in summ and summ["ref_n_tiles"] == len(tiles)


def test_solve_network_prior_counts_in_dof():
    rng = np.random.default_rng(13)
    tiles, z, edges, meta = _gradient_field(rng, np.array([0.0, 0.0]))
    cols, _ = Z._gradient_columns(meta)
    s = Z.solve_network(edges, extra_of=cols, prior=([0.0, 0.0], np.eye(2) * 1e-6))
    assert s["dof"] == Z.solve_network(edges)["dof"]
    assert np.allclose(s["g"], 0.0, atol=1e-4) and s["g_cov"].shape == (2, 2)
