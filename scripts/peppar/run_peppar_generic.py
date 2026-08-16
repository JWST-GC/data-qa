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

Environment variables:
    PEPPAR_DATA_DIR   required  dir holding the per-exposure ``*_cal.fits``, globbed FLAT.
                                Our layout has two forms and ``data_qa.peppar_trigger``
                                passes the one holding THIS detector's files:
                                /orange/adamginsburg/jwst/<field>/<FILT>/pipeline/ (what
                                the reduction writes) or the legacy flat
                                /orange/adamginsburg/jwst/<field>/<FILT>/
    PEPPAR_STF_DIR    required  output dir for this (filter, detector)
    PEPPAR_FILT       required  filter, e.g. F212N
    PEPPAR_DET        required  detector, e.g. NRCA4 (or NRCALONG); 'i2d' to run on i2d mosaics
    PEPPAR_SUFFIX     optional  output-file suffix (default 'p1')

Run under SLURM in the peppar env (see run_peppar_generic.sbatch); NOT on a login node.
"""
import os
import glob

from peppar import peppar


def _env(name, default=None, required=False):
    val = os.environ.get(name, default)
    if required and not val:
        raise SystemExit(f"peppar runner: required env var {name} is not set")
    return val


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
    dict_utils = peppar.setup_filter_props()
    dict_image_props = peppar.setup_dict_image_props(filt_run, det_run)
    dict_images_run = peppar.setup_dict_images_for_run(dict_images, filt_run, det_run)
    # NB: no "remove image 4" hack -- that was specific to one Arches dataset.

    n_run = len(dict_images_run[filt_run][det_run]['images'])
    print(f'Run {filt_run} {det_run}: {n_run} images')

    # ---- PSFs ----
    psf_dir = f'{stf_dir}/psf_default'
    if not os.path.exists(psf_dir):
        os.makedirs(psf_dir)
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
