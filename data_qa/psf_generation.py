"""Pick one PSF generation of per-exposure daophot catalogues.

jwst-gc-pipeline writes a second generation of per-exposure catalogues when it is run with
``PSF_EPSF_CORE_DIR`` set: the hybrid PSF, stamped ``_hybpsf`` in the file name
(``f212n_nrca1_o068_visit001_vgroup02101_exp00001_hybpsf_m3_daophot_basic.fits``) next to the
plain ``..._exp00001_m3_daophot_basic.fits``.  Both match the ``*_m3_daophot_basic.fits`` globs,
so without a choice every exposure would be counted twice.

The hybrid generation is used when it is available, i.e. when it holds every exposure the plain
generation holds; while a re-catalogue is part-way through, the plain generation is used, so one
set never mixes the two.
"""
from __future__ import annotations

import os

HYBRID_TOKEN = "_hybpsf"


def is_hybrid(path):
    return HYBRID_TOKEN in os.path.basename(path)


def _key(path):
    return os.path.basename(path).replace(HYBRID_TOKEN, "")


def use_hybrid(paths):
    """True when the hybrid-PSF catalogues among ``paths`` cover every plain one."""
    hyb = {_key(p) for p in paths if is_hybrid(p)}
    return bool(hyb) and {_key(p) for p in paths if not is_hybrid(p)} <= hyb


def select(paths, hybrid=None):
    """``paths`` restricted to one PSF generation; ``hybrid`` (from :func:`use_hybrid` on a
    wider set, e.g. all detectors of the filter) overrides the decision made on ``paths``."""
    hybrid = use_hybrid(paths) if hybrid is None else hybrid
    return [p for p in paths if is_hybrid(p) == hybrid]
