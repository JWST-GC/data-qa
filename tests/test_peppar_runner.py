"""The two adaptations ``scripts/peppar/run_peppar_generic.py`` makes between our job
layout and peppar's internals.

Both exist because peppar exposes no argument for them, and both were job-killers found by
review of #134: without them the m4 / ngc6397 wide-filter jobs the trigger now emits die
before writing a catalog -- the LW ones in ``setup_dict_images_for_run`` and the F150W2
ones inside the PSF grid build.

peppar and stpsf are not installed in the test env (they live in the ``peppar`` conda env
the sbatch activates), so the runner is loaded with both stubbed.  The stubs reproduce the
exact behaviour that matters: ``organize_images_by_detector_filter`` renames NRCALONG to
NRCA5 (peppar.py:131), ``setup_dict_images_for_run`` indexes with no guard (peppar.py:305),
and the fake NIRCam's sampled wavelengths are the MEASURED stpsf 2.2.0 values.
"""
import copy
import importlib.util
import os
import sys
import types

import pytest

RUNNER_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "scripts", "peppar", "run_peppar_generic.py")

# stpsf 2.2.0 in the peppar env, measured 2026-09-05: (min, max) sampled wavelength in um
# for nlambda 5..40.  SHORT_WAVELENGTH_MAX / LONG_WAVELENGTH_MIN are both 2.3500 um, so
# F150W2 is outside the SW channel for nlambda >= 24 and inside it at 23 and below, while
# F200W (SW) and F322W2 (LW) are inside at every sampling.
_MEASURED = {
    "F150W2": {5: 2.240915, 9: 2.302731, 15: 2.333638, 19: 2.343399, 20: 2.345229,
               21: 2.346885, 22: 2.348390, 23: 2.349764, 24: 2.351024, 25: 2.352183,
               26: 2.353253, 27: 2.354244, 28: 2.355163, 29: 2.356020, 30: 2.356819,
               31: 2.357567, 32: 2.358268, 33: 2.358927, 34: 2.359546, 35: 2.360131,
               36: 2.360683, 37: 2.361205, 38: 2.361699, 39: 2.362169, 40: 2.362614},
    "F200W": {n: 2.236486 for n in range(5, 41)},        # max over all nlambda; all < 2.35
}
_MEASURED_F322W2_MIN = {n: 2.427208 for n in range(5, 41)}   # min over all nlambda; > 2.35


class FakeNIRCamBase:
    """Enough of ``stpsf.NIRCam`` for ``channel_safe_nlambda``: the channel, the two
    channel limits and ``_get_weights``."""
    SHORT_WAVELENGTH_MAX = 2.35e-6
    LONG_WAVELENGTH_MIN = 2.35e-6

    def __init__(self, filt, channel):
        self.filter = filt
        self.channel = channel
        self.probed = []

    def _get_weights(self, nlambda):
        self.probed.append(nlambda)
        if self.channel == "short":
            top = _MEASURED[self.filter][nlambda] * 1e-6
            return ([1.0e-6, top],)
        return ([_MEASURED_F322W2_MIN[nlambda] * 1e-6, 4.04e-6],)


@pytest.fixture
def runner(monkeypatch):
    """The runner module, imported with ``peppar`` stubbed out."""
    pkg = types.ModuleType("peppar")
    inner = types.ModuleType("peppar.peppar")
    pkg.peppar = inner
    monkeypatch.setitem(sys.modules, "peppar", pkg)
    monkeypatch.setitem(sys.modules, "peppar.peppar", inner)
    spec = importlib.util.spec_from_file_location("_run_peppar_generic_undertest",
                                                  RUNNER_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod._stub_peppar = inner
    return mod


# ----------------------------------------------------------- peppar's detector spelling

def test_peppar_detector_renames_only_the_long_detectors(runner):
    """peppar keys NRCA5/NRCB5; our PEPPAR_DET is the filename token NRCALONG/NRCBLONG."""
    assert runner.peppar_detector("NRCALONG") == "NRCA5"
    assert runner.peppar_detector("NRCBLONG") == "NRCB5"
    assert runner.peppar_detector("nrcalong") == "NRCA5"
    for det in ("NRCA1", "NRCA4", "NRCB3", "NRCB4"):
        assert runner.peppar_detector(det) == det
    assert runner.peppar_detector("i2d") == "i2d"       # the mosaic pseudo-detector


def _install_stub_peppar(runner, monkeypatch, header_det):
    """A peppar stub that reproduces the two calls that matter, including the rename."""
    calls = {}

    def organize_images_by_detector_filter(images):
        det = {"NRCALONG": "NRCA5", "NRCBLONG": "NRCB5"}.get(header_det, header_det)
        return {"F322W2": {det: {"images": list(images)}}}

    def setup_dict_images_for_run(dict_images, filt_run, det_run):
        calls["images_for_run"] = det_run
        # peppar.py:305 -- indexed with no guard, so a wrong spelling is a raw KeyError
        return {filt_run: {det_run: copy.deepcopy(dict_images[filt_run][det_run])}}

    def setup_dict_image_props(filt, det):
        calls["image_props"] = det
        return {filt: {det: {}}}

    p = runner._stub_peppar
    monkeypatch.setattr(p, "organize_images_by_detector_filter",
                        organize_images_by_detector_filter, raising=False)
    monkeypatch.setattr(p, "setup_dict_images_for_run", setup_dict_images_for_run,
                        raising=False)
    monkeypatch.setattr(p, "setup_dict_image_props", setup_dict_image_props, raising=False)
    monkeypatch.setattr(p, "setup_filter_props", lambda: {}, raising=False)
    def get_psfs(*a, **k):
        # record whether the channel-safe wrapper was already in place at grid-build time
        calls["safe_at_grid_build"] = getattr(
            sys.modules["webbpsf"].NIRCam.psf_grid, "_channel_safe", False)
        return {}

    monkeypatch.setattr(p, "get_psfs", get_psfs, raising=False)
    monkeypatch.setattr(p, "extract_catalogs", lambda *a, **k: calls.setdefault(
        "extracted", True), raising=False)
    return calls


def _run_one(runner, monkeypatch, tmp_path, det, header_det):
    data = tmp_path / "pipeline"
    data.mkdir()
    (data / f"jw01979002001_02101_00001_{det.lower()}_cal.fits").write_bytes(b"")
    calls = _install_stub_peppar(runner, monkeypatch, header_det)
    monkeypatch.setenv("PEPPAR_DATA_DIR", str(data))
    monkeypatch.setenv("PEPPAR_STF_DIR", str(tmp_path / "out"))
    monkeypatch.setenv("PEPPAR_FILT", "F322W2")
    monkeypatch.setenv("PEPPAR_DET", det)
    stpsf = _fake_stpsf()
    monkeypatch.setitem(sys.modules, "webbpsf", stpsf)
    monkeypatch.chdir(tmp_path)
    runner.run_peppar_script()
    calls["stpsf"] = stpsf
    return calls


def test_a_long_detector_job_indexes_peppar_by_nrca5(runner, monkeypatch, tmp_path):
    """The F322W2 NRCALONG/NRCBLONG jobs #134 unlocks: the runner must hand peppar its own
    spelling.  Passing PEPPAR_DET through raises ``KeyError: 'NRCALONG'`` in
    ``setup_dict_images_for_run`` after the job has taken its queue slot."""
    calls = _run_one(runner, monkeypatch, tmp_path, "NRCALONG", "NRCALONG")
    assert calls["images_for_run"] == "NRCA5"
    assert calls["image_props"] == "NRCA5"
    assert calls["extracted"] is True


def test_a_short_detector_job_is_unchanged(runner, monkeypatch, tmp_path):
    """The rename must not reach the SW detectors, which peppar keys by their own name."""
    calls = _run_one(runner, monkeypatch, tmp_path, "NRCA1", "NRCA1")
    assert calls["images_for_run"] == "NRCA1"
    assert calls["image_props"] == "NRCA1"


def test_the_run_installs_the_channel_safe_psf_grid_before_building(runner, monkeypatch,
                                                                    tmp_path):
    """The helper is worth nothing unless the run actually installs it, which it must do
    BEFORE ``peppar.get_psfs`` -- that is the call that builds the grid."""
    calls = _run_one(runner, monkeypatch, tmp_path, "NRCA1", "NRCA1")
    assert getattr(calls["stpsf"].NIRCam.psf_grid, "_channel_safe", False)
    assert calls["safe_at_grid_build"] is True


# ------------------------------------------------------ the SW channel edge (F150W2)

def test_channel_safe_nlambda_picks_the_largest_sampling_inside_the_channel(runner):
    """F150W2's default nlambda=40 samples to 2.3626 um, past SHORT_WAVELENGTH_MAX=2.3500,
    which is what kills the grid build.  23 is the largest sampling that fits (measured)."""
    nrc = FakeNIRCamBase("F150W2", "short")
    assert runner.channel_safe_nlambda(nrc) == {"nlambda": 23}
    # it searched downward from the default rather than jumping to a constant
    assert nrc.probed[0] == 40 and nrc.probed[-1] == 23


def test_channel_safe_nlambda_leaves_a_bandpass_that_already_fits_alone(runner):
    """Every filter this campaign runs except F150W2 is inside its channel at nlambda=40,
    so their PSF grids must be built exactly as they are today."""
    sw = FakeNIRCamBase("F200W", "short")
    assert runner.channel_safe_nlambda(sw) == {}
    lw = FakeNIRCamBase("F322W2", "long")
    assert runner.channel_safe_nlambda(lw) == {}
    assert lw.probed == [40]                      # one look, no search
    other = FakeNIRCamBase("F770W", None)         # not a NIRCam channel at all
    assert runner.channel_safe_nlambda(other) == {}


def test_channel_safe_nlambda_leaves_nlambda_alone_when_it_cannot_sample(runner):
    class Unsamplable(FakeNIRCamBase):
        def _get_weights(self, nlambda):
            raise KeyError("no such bandpass")

    assert runner.channel_safe_nlambda(Unsamplable("F150W2", "short")) == {}


def _fake_stpsf(strict_limit=None):
    """A stpsf stand-in whose ``psf_grid`` records its kwargs, and (with
    ``strict_limit``) refuses a sampling that oversteps the SW channel the way stpsf does."""
    mod = types.ModuleType("webbpsf")

    class NIRCam(FakeNIRCamBase):
        def psf_grid(self, num_psfs=16, **kwargs):
            self.grid_kwargs = kwargs
            nl = kwargs.get("nlambda", 40)
            if strict_limit is not None and _MEASURED[self.filter][nl] * 1e-6 > strict_limit:
                raise RuntimeError("The requested wavelengths are too long for NIRCam "
                                   "short wave channel.")
            return f"grid(nlambda={nl})"

    mod.NIRCam = NIRCam
    return mod


def test_the_wrapper_lets_the_f150w2_grid_build_at_all(runner):
    """End of the chain: peppar calls ``nrc.psf_grid()`` with no nlambda, so on a cold
    cache F150W2 raises before any photometry happens.  With the wrapper installed the
    same call succeeds."""
    stpsf = _fake_stpsf(strict_limit=FakeNIRCamBase.SHORT_WAVELENGTH_MAX)
    nrc = stpsf.NIRCam("F150W2", "short")
    with pytest.raises(RuntimeError, match="too long for NIRCam short wave channel"):
        nrc.psf_grid(num_psfs=4)
    runner.install_channel_safe_psf_grid(stpsf)
    assert nrc.psf_grid(num_psfs=4) == "grid(nlambda=23)"


def test_the_wrapper_changes_nothing_for_a_filter_that_already_fits(runner):
    stpsf = _fake_stpsf()
    runner.install_channel_safe_psf_grid(stpsf)
    nrc = stpsf.NIRCam("F200W", "short")
    nrc.psf_grid(num_psfs=4)
    assert "nlambda" not in nrc.grid_kwargs


def test_the_wrapper_respects_an_explicit_nlambda_and_installs_once(runner):
    stpsf = _fake_stpsf()
    first = runner.install_channel_safe_psf_grid(stpsf)
    second = runner.install_channel_safe_psf_grid(stpsf)
    assert first is second is stpsf.NIRCam.psf_grid      # idempotent, not double-wrapped
    nrc = stpsf.NIRCam("F150W2", "short")
    nrc.psf_grid(num_psfs=4, nlambda=9)
    assert nrc.grid_kwargs["nlambda"] == 9
