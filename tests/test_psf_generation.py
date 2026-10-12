from data_qa import psf_generation as G

P = "f212n_nrca1_o068_visit001_vgroup02101_exp0000{e}{t}_m3_daophot_basic.fits"


def _names(es, t=""):
    return [P.format(e=e, t=t) for e in es]


def test_hybrid_used_when_it_covers_every_plain_exposure():
    paths = _names((1, 2)) + _names((1, 2), "_hybpsf")
    assert G.use_hybrid(paths) and G.select(paths) == _names((1, 2), "_hybpsf")


def test_plain_kept_while_hybrid_is_partial():
    paths = _names((1, 2)) + _names((1,), "_hybpsf")
    assert not G.use_hybrid(paths) and G.select(paths) == _names((1, 2))


def test_single_generation_passes_through():
    assert G.select(_names((1, 2))) == _names((1, 2))
    assert G.select(_names((1, 2), "_hybpsf")) == _names((1, 2), "_hybpsf")


def test_wider_decision_overrides():
    assert G.select(_names((1,)) + _names((1,), "_hybpsf"), hybrid=False) == _names((1,))
