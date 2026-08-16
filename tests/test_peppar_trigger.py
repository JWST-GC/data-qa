"""Offline unit tests for data_qa.peppar_trigger cal-file discovery (no sbatch is ever
run).  Pins the two on-disk layouts of issue #73: the reduction writes
``<field>/<FILT>/pipeline/*_cal.fits`` while legacy fields (brick) keep cal files flat at
``<field>/<FILT>/*_cal.fits``."""
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
    # data_dir prefers the pipeline/ copy when both layouts hold cal files
    assert ppt.cal_data_dir(str(fdir)) == f"{fdir}/pipeline"


def test_empty_field_raises(tmp_path):
    (tmp_path / "gc-treasury" / "F212N").mkdir(parents=True)
    with pytest.raises(SystemExit):
        ppt.build_jobs(PROGRAM, OBS, base=str(tmp_path))
