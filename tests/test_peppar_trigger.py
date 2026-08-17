"""Offline unit tests for data_qa.peppar_trigger cal-file discovery (no sbatch is ever
run).  Pins the two on-disk layouts of issue #73: the reduction writes
``<field>/<FILT>/pipeline/*_cal.fits`` while older reductions left them flat at
``<field>/<FILT>/*_cal.fits``, and fields such as brick / gc2211 / ngc6334 hold both --
so both layouts, their union, and the per-detector ``PEPPAR_DATA_DIR`` are pinned."""
import glob

import pytest

from data_qa import peppar_trigger as ppt

PROGRAM, OBS = "10678", "001"


def _touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


def make_field(base, field, filt, dets, layout, nexp=1, program_obs="10678001"):
    """Create empty cal files for one filter dir in the given layout."""
    fdir = base / field / filt
    caldir = fdir / "pipeline" if layout == "pipeline" else fdir
    for n in range(1, nexp + 1):
        for det in dets:
            _touch(caldir / f"jw{program_obs}001_02101_0000{n}_{det}_cal.fits")
    return fdir


# ---------------------------------------------------------------- pipeline/ layout

def test_field_for_finds_pipeline_layout(tmp_path):
    make_field(tmp_path, "gc-treasury", "F212N", ["nrca1"], "pipeline")
    assert ppt.field_for(PROGRAM, OBS, base=str(tmp_path)) == "gc-treasury"


def test_enumerate_finds_pipeline_layout(tmp_path):
    make_field(tmp_path, "gc-treasury", "F212N", ["nrca1", "nrcblong"], "pipeline")
    make_field(tmp_path, "gc-treasury", "F480M", ["nrcalong"], "pipeline")
    assert ppt.enumerate_filt_det("gc-treasury", base=str(tmp_path)) == {
        "F212N": ["NRCA1", "NRCBLONG"], "F480M": ["NRCALONG"]}


def test_build_jobs_data_dir_points_at_pipeline(tmp_path):
    """The runner globs PEPPAR_DATA_DIR flat, so it must be the pipeline/ subdir when
    that layout holds the cal files; outputs (stf_dir) stay at the filter level."""
    make_field(tmp_path, "gc-treasury", "F212N", ["nrca1"], "pipeline")
    (job,) = ppt.build_jobs(PROGRAM, OBS, base=str(tmp_path))
    assert job["data_dir"] == f"{tmp_path}/gc-treasury/F212N/pipeline"
    assert job["stf_dir"] == f"{tmp_path}/gc-treasury/F212N/peppar_nrca1"


# ---------------------------------------------------------------- legacy flat layout

def test_flat_layout_regression(tmp_path):
    """A flat-only (brick-like) field enumerates exactly as before the pipeline/ glob
    was added, including the golden job dict."""
    make_field(tmp_path, "brick", "F212N", ["nrca1", "nrca2"], "flat",
               program_obs="02221001")
    assert ppt.field_for("2221", "001", base=str(tmp_path)) == "brick"
    assert ppt.enumerate_filt_det("brick", base=str(tmp_path)) == {
        "F212N": ["NRCA1", "NRCA2"]}
    jobs = ppt.build_jobs("2221", "001", base=str(tmp_path))
    assert jobs[0] == dict(field="brick", filt="F212N", det="NRCA1",
                           data_dir=f"{tmp_path}/brick/F212N",
                           stf_dir=f"{tmp_path}/brick/F212N/peppar_nrca1",
                           name="peppar-brick-f212n-nrca1")


def test_non_filter_dirs_still_skipped(tmp_path):
    """A pipeline/ subdir under a non-filter dir (e.g. mosaics/) never enumerates."""
    make_field(tmp_path, "gc-treasury", "F212N", ["nrca1"], "pipeline")
    make_field(tmp_path, "gc-treasury", "mosaics", ["nrca1"], "pipeline")
    assert list(ppt.enumerate_filt_det("gc-treasury", base=str(tmp_path))) == ["F212N"]


# ---------------------------------------------------------------- mixed layouts

def test_mixed_fields_each_program_resolves(tmp_path):
    """One base holding a flat (brick-like) field AND a pipeline/ field: each program
    finds its own field."""
    make_field(tmp_path, "brick", "F212N", ["nrca1"], "flat", program_obs="02221001")
    make_field(tmp_path, "gc-treasury", "F480M", ["nrcalong"], "pipeline")
    assert ppt.field_for("2221", "001", base=str(tmp_path)) == "brick"
    assert ppt.field_for(PROGRAM, OBS, base=str(tmp_path)) == "gc-treasury"


def test_pipeline_layout_is_obs_stem_scoped(tmp_path):
    """Two pipeline/-layout fields for different programs: the pipeline/ glob is scoped
    by the jw<program><obs> stem exactly as the flat one is.  Dropping the stem there
    makes field_for return the alphabetically first field holding any cal file ('brick'
    here), which fans every peppar job out over the wrong field."""
    make_field(tmp_path, "brick", "F212N", ["nrca1"], "pipeline", program_obs="02221001")
    make_field(tmp_path, "gc-treasury", "F480M", ["nrcalong"], "pipeline")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["brick", "gc-treasury"]
    assert ppt.field_for("2221", "001", base=str(tmp_path)) == "brick"
    assert ppt.field_for(PROGRAM, OBS, base=str(tmp_path)) == "gc-treasury"
    assert ppt.field_for("1182", "001", base=str(tmp_path)) is None


def test_stem_is_obs_scoped_in_both_layouts(tmp_path):
    """The OBS half of the ``jw<program><obs>`` stem discriminates too, in both layouts.

    Three real programs have cal files under more than one field dir, and for two of them
    the obs token is the only discriminator: jw02045 o001 is arches while o003 is
    quintuplet, jw01979 o001 is ngc6397 while o002 is m4.  ``field_for`` walks fields in
    sorted order, so dropping ``{obs}`` from the stem returns the alphabetically first
    field of the program -- arches for 2045/003 and m4 for 1979/001 -- and fans every
    peppar job out over the wrong field with no error and a populated data_dir.

    Both fixtures mirror that live geometry: the pipeline-layout pair for the pipeline
    glob, the flat-layout pair for the flat one.
    """
    make_field(tmp_path, "arches", "F212N", ["nrca1"], "pipeline", program_obs="02045001")
    make_field(tmp_path, "quintuplet", "F212N", ["nrca1"], "pipeline",
               program_obs="02045003")
    make_field(tmp_path, "m4", "F212N", ["nrca1"], "flat", program_obs="01979002")
    make_field(tmp_path, "ngc6397", "F212N", ["nrca1"], "flat", program_obs="01979001")
    # sorted() order is what an unscoped glob would return, so pin it
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "arches", "m4", "ngc6397", "quintuplet"]
    # pipeline layout: same program, obs alone decides
    assert ppt.field_for("2045", "001", base=str(tmp_path)) == "arches"
    assert ppt.field_for("2045", "003", base=str(tmp_path)) == "quintuplet"
    # flat layout: same program, obs alone decides
    assert ppt.field_for("1979", "002", base=str(tmp_path)) == "m4"
    assert ppt.field_for("1979", "001", base=str(tmp_path)) == "ngc6397"
    # an obs of a program that is present resolves to nothing when that obs is not
    assert ppt.field_for("2045", "007", base=str(tmp_path)) is None


def test_mixed_within_filter_dir_dedupes_and_unions(tmp_path):
    """Both layouts inside ONE filter dir: a file in both (same basename) is counted
    once, and a flat-only file still contributes its detector."""
    fdir = make_field(tmp_path, "gc-treasury", "F212N", ["nrca1"], "pipeline")
    # the same exposure also present flat (partial migration) + a flat-only detector
    _touch(fdir / "jw10678001001_02101_00001_nrca1_cal.fits")
    _touch(fdir / "jw10678001001_02101_00001_nrca2_cal.fits")
    files = ppt._cal_files(str(fdir))
    assert len(files) == 2                       # nrca1 deduped by basename
    assert f"{fdir}/pipeline/jw10678001001_02101_00001_nrca1_cal.fits" in files
    assert ppt.enumerate_filt_det("gc-treasury", base=str(tmp_path)) == {
        "F212N": ["NRCA1", "NRCA2"]}
    # filter-level answer: pipeline/ holds cal files
    assert ppt.cal_data_dir(str(fdir)) == f"{fdir}/pipeline"
    # per-detector answer: NRCA1 migrated, NRCA2 is still flat-only
    assert ppt.cal_data_dir(str(fdir), "NRCA1") == f"{fdir}/pipeline"
    assert ppt.cal_data_dir(str(fdir), "NRCA2") == str(fdir)


def test_mixed_within_filter_dir_data_dir_is_per_detector(tmp_path):
    """A flat-only detector in a half-migrated filter dir gets the FLAT dir while its
    migrated sibling gets pipeline/.  Handing it {FILT}/pipeline would clear the runner's
    "no images" guard on NRCA1's files and then KeyError in
    peppar.setup_dict_images_for_run (dict_images[filt][det]) after taking a queue slot."""
    fdir = make_field(tmp_path, "gc-treasury", "F212N", ["nrca1"], "pipeline")
    _touch(fdir / "jw10678001001_02101_00001_nrca2_cal.fits")
    jobs = {j["det"]: j for j in ppt.build_jobs(PROGRAM, OBS, base=str(tmp_path))}
    assert sorted(jobs) == ["NRCA1", "NRCA2"]
    assert jobs["NRCA1"]["data_dir"] == f"{fdir}/pipeline"
    assert jobs["NRCA2"]["data_dir"] == str(fdir)
    # every job's data_dir really holds that detector's cal files
    for det, job in jobs.items():
        assert glob.glob(f"{job['data_dir']}/*_{det.lower()}_cal.fits")
    # outputs still stay at the filter level for both
    assert jobs["NRCA2"]["stf_dir"] == f"{fdir}/peppar_nrca2"


def test_empty_field_raises(tmp_path):
    (tmp_path / "gc-treasury" / "F212N").mkdir(parents=True)
    with pytest.raises(SystemExit):
        ppt.build_jobs(PROGRAM, OBS, base=str(tmp_path))


def test_field_with_cal_files_only_outside_filter_dirs_raises(tmp_path):
    """field_for matches on ANY subdir, enumerate_filt_det only on filter dirs, so a
    field whose cal files sit in e.g. dolphot/ resolves and then yields no jobs.  That
    must raise (mast_monitor.act_peppar reports the SystemExit) instead of silently
    reporting 0 jobs for a field that was never scanned."""
    make_field(tmp_path, "w51", "dolphot", ["nrca1"], "flat")
    assert ppt.field_for(PROGRAM, OBS, base=str(tmp_path)) == "w51"
    assert ppt.enumerate_filt_det("w51", base=str(tmp_path)) == {}
    with pytest.raises(SystemExit):
        ppt.build_jobs(PROGRAM, OBS, base=str(tmp_path))
    # an explicit --filters/--dets subset that matches nothing stays a quiet empty list
    assert ppt.build_jobs(PROGRAM, OBS, field="w51", filters=["F212N"],
                          base=str(tmp_path)) == []


def test_no_jobs_error_names_the_skipped_dirs(tmp_path):
    """m4 and ngc6397 keep their cal files in F150W2/ and F322W2/; the trailing "2" fails
    the filter-dir pattern, so they resolve as fields and enumerate nothing.  The error
    must name those dirs and the pattern -- otherwise it reads as "no data on disk" while
    the data is there."""
    make_field(tmp_path, "m4", "F150W2", ["nrca1"], "pipeline")
    make_field(tmp_path, "m4", "F322W2", ["nrcalong"], "flat")
    assert ppt.enumerate_filt_det("m4", base=str(tmp_path)) == {}
    assert ppt.nonfilter_cal_dirs("m4", base=str(tmp_path)) == ["F150W2", "F322W2"]
    with pytest.raises(SystemExit) as ei:
        ppt.build_jobs(PROGRAM, OBS, base=str(tmp_path))
    msg = str(ei.value)
    assert "F150W2" in msg and "F322W2" in msg
    assert ppt._FILT_RE.pattern in msg
    # it reports only the dirs the pattern rejects: a real filter dir alongside a
    # dolphot/ one leaves just dolphot
    make_field(tmp_path, "brick", "F212N", ["nrca1"], "pipeline")
    make_field(tmp_path, "brick", "dolphot", ["nrca1"], "flat")
    assert ppt.nonfilter_cal_dirs("brick", base=str(tmp_path)) == ["dolphot"]
