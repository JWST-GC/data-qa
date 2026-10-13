"""Network zero-point solver tests (data_qa.neighbor_zeropoints) on synthetic overlap graphs with
known zero points and a known NRCA - NRCB module term."""
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
    assert np.isclose(summ["tile_zp_rms"], np.std([z[t] for t in tiles]), atol=1e-6)
