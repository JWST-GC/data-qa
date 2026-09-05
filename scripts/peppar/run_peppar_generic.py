"""Parametrized peppar PSF-photometry runner (env-driven), for automated per-(filter,
detector) extraction on the jwst-gc-pipeline data layout.

Unlike the hand-edited ``run_peppar_<filt>_<det>_NN.py`` scripts, this reads ALL paths and
the filter/detector from environment variables so a trigger (``data_qa.peppar_trigger``) can
fan one job out per (filter, detector) with no per-file editing.  It runs ONLY the extraction
step (``peppar.extract_catalogs``); the Arches-specific intra-epoch align / combo-starlist
steps (which need ``flystar`` and hard-coded ``/Users/hosek-local`` distortion paths) are
deliberately omitted, and there is no module-level ``flystar`` import (it crashes in the HPG
``peppar`` env).

The PSF / detection / fitting parameters are copied verbatim from Matt Hosek's working
``run_peppar_f212n_nrca4_01.py`` -- do not retune here without checking with him.

Two adaptations sit between this runner and peppar, both because peppar's own entry points
take no argument for them (see the helpers below for the measurements):

* ``peppar_detector`` -- peppar keys its dictionaries NRCA5 / NRCB5 for the LW detectors,
  while ``PEPPAR_DET`` is the filename token NRCALONG / NRCBLONG.  Without the translation
  every LW job dies in ``peppar.setup_dict_images_for_run`` with ``KeyError: 'NRCALONG'``.
* ``install_channel_safe_psf_grid`` -- F150W2 samples past the NIRCam SW channel edge at
  stpsf's default nlambda=40, so its PSF grid cannot be built at all.  The wrapper supplies
  the largest nlambda that fits, and leaves every other filter untouched.

Environment variables:
    PEPPAR_DATA_DIR   required  dir holding the per-exposure ``*_cal.fits``, globbed FLAT.
                                Our layout has two forms and ``data_qa.peppar_trigger``
                                passes the one holding THIS detector's files -- the side
                                holding more of them when both do:
                                /orange/adamginsburg/jwst/<field>/<FILT>/pipeline/ (what
                                the reduction writes) or the legacy flat
                                /orange/adamginsburg/jwst/<field>/<FILT>/
    PEPPAR_STF_DIR    required  output dir for this (filter, detector)
    PEPPAR_FILT       required  filter, e.g. F212N
    PEPPAR_DET        required  detector, e.g. NRCA4 (or NRCALONG); 'i2d' to run on i2d mosaics
    PEPPAR_SUFFIX     optional  output-file suffix (default 'p1')

Run under SLURM in the peppar env (see run_peppar_generic.sbatch); NOT on a login node.
"""
import functools
import os
import glob

from peppar import peppar


def _env(name, default=None, required=False):
    val = os.environ.get(name, default)
    if required and not val:
        raise SystemExit(f"peppar runner: required env var {name} is not set")
    return val


# peppar keys its per-detector dictionaries by the name
# ``peppar.organize_images_by_detector_filter`` writes, which renames the LW detectors it
# reads out of the DETECTOR header: NRCALONG -> NRCA5 and NRCBLONG -> NRCB5 (peppar.py:131).
# Our PEPPAR_DET comes from ``data_qa.peppar_trigger._DET_RE``, which reads the FILENAME
# token and so says NRCALONG, and every hand-written peppar script for a LW filter passes
# NRCA5 (brick/peppar/F405N/ holds NRCA5/NRCB5 while F212N/ holds NRCA1..NRCB4).  Passing
# the filename spelling straight through makes ``setup_dict_images_for_run`` --  which
# indexes ``dict_images[filt][det]`` with no guard -- raise ``KeyError: 'NRCALONG'`` after
# the job has taken its queue slot.  peppar's own ``get_jwst_vega_zp`` maps NRCB5 back to
# NRCBLONG for the zeropoint table, so NRCA5/NRCB5 is the convention inside the package.
_PEPPAR_DET = {"NRCALONG": "NRCA5", "NRCBLONG": "NRCB5"}


def peppar_detector(det):
    """The key peppar uses for a detector, given our (filename-derived) spelling.

    Identity for the SW detectors and for the ``i2d`` pseudo-detector; NRCALONG/NRCBLONG
    become NRCA5/NRCB5."""
    if det == "i2d":
        return det
    return _PEPPAR_DET.get(det.upper(), det.upper())


def channel_safe_nlambda(nrc, nlambda_default=40):
    """``psf_grid`` kwargs that keep every sampled wavelength inside the NIRCam channel.

    stpsf validates the SAMPLED wavelengths of a bandpass against the channel limits and
    raises ``RuntimeError: The requested wavelengths are too long for NIRCam short wave
    channel.`` if one falls outside.  ``SHORT_WAVELENGTH_MAX`` is 2.3500 um and F150W2,
    the widest SW filter, samples 1.0065-2.3626 um at stpsf's default nlambda=40 -- one of
    the forty sits 12.6 nm past the limit and the whole grid build dies at grid position
    1/4.  Measured in the peppar env (stpsf 2.2.0) on F150W2/NRCA1:

        nlambda   reddest sample   verdict
             20        2.3452 um   ok
             23        2.3498 um   ok      <- the largest that fits
             24        2.3510 um   RAISES
             40        2.3626 um   RAISES

    So this returns the LARGEST sampling that stays inside the channel rather than a
    constant, and returns ``{}`` for a bandpass that already fits.  Of the filters this
    campaign runs, only F150W2 oversteps -- F322W2 samples 2.4272-4.0430 um against a
    ``LONG_WAVELENGTH_MIN`` of 2.3500, and F115W/F200W/F212N/F405N/F444W/F480M all fit --
    so no PSF that builds today is changed.

    The same defect was fixed for the reduction in jwst-gc-pipeline PR #586
    (``jwst_gc_pipeline.photometry.psf_channel``); this is the peppar-side port.  It is
    duck-typed on ``channel`` / ``_get_weights`` / the channel limits so it can be tested
    without stpsf."""
    channel = getattr(nrc, "channel", None)
    if channel == "short":
        limit = nrc.SHORT_WAVELENGTH_MAX

        def inside(lam):
            return max(lam) <= limit
    elif channel == "long":
        limit = nrc.LONG_WAVELENGTH_MIN

        def inside(lam):
            return min(lam) >= limit
    else:
        return {}
    try:
        lam = nrc._get_weights(nlambda=nlambda_default)[0]
    except (AttributeError, KeyError, ValueError) as exc:
        # Not a bandpass we can sample ahead of time; leave psf_grid to decide, which is
        # what happens without this wrapper at all.
        print(f"peppar runner: cannot pre-sample {getattr(nrc, 'filter', '?')} "
              f"({type(exc).__name__}: {exc}); leaving nlambda to stpsf", flush=True)
        return {}
    if inside(lam):
        return {}
    for nl in range(nlambda_default - 1, 4, -1):
        if inside(nrc._get_weights(nlambda=nl)[0]):
            print(f"peppar runner: {getattr(nrc, 'filter', '?')} on the {channel} channel "
                  f"oversteps its limit ({limit * 1e6:.4f} um) at nlambda="
                  f"{nlambda_default}; building the PSF grid with nlambda={nl}", flush=True)
            return {"nlambda": nl}
    raise SystemExit(
        f"peppar runner: no nlambda between 5 and {nlambda_default} keeps "
        f"{getattr(nrc, 'filter', '?')} inside the NIRCam {channel} channel "
        f"(limit {limit * 1e6:.4f} um) -- the filter/detector pairing is probably wrong")


def install_channel_safe_psf_grid(webbpsf_mod):
    """Make ``webbpsf.NIRCam.psf_grid`` sample inside its channel by default.

    peppar builds its grid in ``peppar.create_psf_model``, which calls ``nrc.psf_grid(...)``
    with no ``nlambda`` and exposes no way to pass one (``get_psfs`` has no such argument),
    so the only place we can supply it is around stpsf's own method.  The wrapper fills in
    ``nlambda`` ONLY when the bandpass would overstep the channel and only when the caller
    did not ask for one, so every filter that builds today builds identically.  Idempotent.

    peppar's PSF cache name (``PSF_<filt>_samp<os>_G5V_fov<fov>_npsfs<n>_..._<det>.fits``)
    does not encode nlambda, so a grid built at the reduced sampling is cached under the
    name a 40-sample grid would use.  For F150W2 that is the only grid that can exist --
    the 40-sample one cannot be built -- so there is nothing to collide with.

    Returns the wrapped-or-already-wrapped attribute, for callers that want to restore it."""
    orig = webbpsf_mod.NIRCam.psf_grid
    if getattr(orig, "_channel_safe", False):
        return orig

    @functools.wraps(orig)
    def psf_grid(self, *args, **kwargs):
        if "nlambda" not in kwargs:
            kwargs.update(channel_safe_nlambda(self))
        return orig(self, *args, **kwargs)

    psf_grid._channel_safe = True
    webbpsf_mod.NIRCam.psf_grid = psf_grid
    return psf_grid


def define_variables():
    """User variables.  Paths + filter/detector come from the environment; everything else is
    Hosek's tuned default (verbatim from run_peppar_f212n_nrca4_01.py)."""
    # ---- Paths + target, from the environment ----
    data_dir = _env("PEPPAR_DATA_DIR", required=True)
    stf_dir = _env("PEPPAR_STF_DIR", required=True)
    filt_run = _env("PEPPAR_FILT", required=True).upper()
    det_run = _env("PEPPAR_DET", required=True)
    det_run = det_run if det_run == "i2d" else det_run.upper()
    out_suffix = _env("PEPPAR_SUFFIX", "p1")

    # ---- PSF Extraction parameters (verbatim) ----
    Niter = 2
    test_extraction = False
    bkg_method = '2d'
    threshold = 5
    peakmax = 5000
    group_dist_coeff = 2.
    shift_tol_coeff = 1.

    # Trimfake parameters
    threshold_trimfake = 2.
    fitshape_trimfake = 17
    subshape_trimfake = 101
    brite_percentile_trimfake = 20

    # WebbPSF Parameters
    num_psf_model = 4
    oversample = 4
    add_detector_effects = True
    fov_model = 201
    recomp_psf_model = False

    # Saturated star PSF-fit params
    mask_saturated = True
    fov_sat = fov_model
    fitshape_sat = 31
    threshold_sat = threshold
    update_sat_psf = True

    # Background Parameters
    box_size = 100
    filter_size = 3

    # PSF Fitting Parameters
    niter_epsf_construct = 3
    fov = 13
    fitshape = 9
    subshape = 13
    grid_points = 2
    min_npsf = 3
    aper_radius = 3
    use_combo_psf = False
    combo_r0 = fitshape / 2.0
    combo_k = 3

    # Finding PSF stars
    peakmax_psf = peakmax
    threshold_psf = 100
    brite_percentile = 10
    min_dist = 3
    mag_diff = 3
    sharp_sig_lim = 3

    # ---- Validity checks (verbatim) ----
    assert fov_sat <= fov_model
    assert fov <= fov_model
    assert fitshape_trimfake <= fov_model
    assert subshape_trimfake <= fov_model
    assert subshape <= fov
    assert fitshape <= subshape

    detsample = (oversample == 1)
    return locals()


def run_peppar_script():
    """Wrapper around ``peppar.extract_catalogs`` for one (filter, detector)."""
    kwargs = define_variables()
    stf_dir = kwargs['stf_dir']
    data_dir = kwargs['data_dir']
    filt_run = kwargs['filt_run']
    det_run = kwargs['det_run']

    if not os.path.exists(stf_dir):
        os.makedirs(stf_dir)
    print(f'Data dir: {data_dir}')
    print(f'Reduction dir: {stf_dir}')

    if det_run == 'i2d':
        images = sorted(glob.glob(data_dir + '/*nircam*_i2d.fits'))
        print(f'Found {len(images)} i2d images.')
    else:
        images = sorted(glob.glob(data_dir + '/*cal.fits'))
        print(f'Found {len(images)} cal images.')
    if not images:
        raise SystemExit(f"peppar runner: no images in {data_dir} for det {det_run}")

    dict_images = peppar.organize_images_by_detector_filter(images)
    # peppar spells the LW detectors NRCA5 / NRCB5; our PEPPAR_DET is the filename token
    # (NRCALONG / NRCBLONG).  Index with peppar's spelling -- see peppar_detector.
    det_key = peppar_detector(det_run)
    dict_utils = peppar.setup_filter_props()
    dict_image_props = peppar.setup_dict_image_props(filt_run, det_key)
    dict_images_run = peppar.setup_dict_images_for_run(dict_images, filt_run, det_key)
    # NB: no "remove image 4" hack -- that was specific to one Arches dataset.

    n_run = len(dict_images_run[filt_run][det_key]['images'])
    print(f'Run {filt_run} {det_run} (peppar key {det_key}): {n_run} images')

    # ---- PSFs ----
    psf_dir = f'{stf_dir}/psf_default'
    if not os.path.exists(psf_dir):
        os.makedirs(psf_dir)
    # F150W2's default spectral sampling oversteps the SW channel edge and kills the grid
    # build; see channel_safe_nlambda.  Imported here rather than at module scope so the
    # runner stays importable (and testable) without stpsf.
    import webbpsf
    install_channel_safe_psf_grid(webbpsf)
    dict_psfs = peppar.get_psfs(dict_images_run, outdir=psf_dir, fov=kwargs['fov_model'],
                                oversample=kwargs['oversample'], num=kwargs['num_psf_model'],
                                add_detector_effects=kwargs['add_detector_effects'],
                                detsample=kwargs['detsample'],
                                recomp_psf_model=kwargs['recomp_psf_model'])

    # ---- Extraction ----
    os.chdir(stf_dir)
    peppar.extract_catalogs(
        dict_images_run, dict_psfs, dict_image_props, dict_utils, Niter=kwargs['Niter'],
        psf_model='webbpsf', psf_model_sat='webbpsf',
        peakmax=kwargs['peakmax'], bkg_method=kwargs['bkg_method'],
        threshold=kwargs['threshold'], threshold_sat=kwargs['threshold_sat'],
        update_sat_psf=kwargs['update_sat_psf'], aper_radius=kwargs['aper_radius'],
        fitshape=kwargs['fitshape'], subshape=kwargs['subshape'],
        fitshape_sat=kwargs['fitshape_sat'], fov=kwargs['fov'], fov_sat=kwargs['fov_sat'],
        niter_epsf_construct=kwargs['niter_epsf_construct'], grid_points=kwargs['grid_points'],
        threshold_psf=kwargs['threshold_psf'], min_npsf=kwargs['min_npsf'],
        brite_percentile=kwargs['brite_percentile'], mag_diff=kwargs['mag_diff'],
        min_dist=kwargs['min_dist'], sharp_sig_lim=kwargs['sharp_sig_lim'],
        oversample=kwargs['oversample'], group_dist_coeff=kwargs['group_dist_coeff'],
        shift_tol_coeff=kwargs['shift_tol_coeff'], threshold_trimfake=kwargs['threshold_trimfake'],
        fitshape_trimfake=kwargs['fitshape_trimfake'], subshape_trimfake=kwargs['subshape_trimfake'],
        brite_percentile_trimfake=kwargs['brite_percentile_trimfake'],
        box_size=kwargs['box_size'], filter_size=kwargs['filter_size'],
        use_combo_psf=kwargs['use_combo_psf'], combo_r0=kwargs['combo_r0'], combo_k=kwargs['combo_k'],
        out_suffix=kwargs['out_suffix'], out_dir=stf_dir,
        mask_saturated=kwargs['mask_saturated'], test_extraction=kwargs['test_extraction'],
        make_plots=False)


if __name__ == "__main__":
    run_peppar_script()
