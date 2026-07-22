"""Offline unit tests for data_qa.pipeline_trigger (command generation only --
no sbatch is ever run)."""
import pytest

from data_qa import pipeline_trigger as pt

FILTERS = ["F405N", "F410M", "F466N", "F212N"]


def test_reduction_golden_command():
    step = pt.reduction_step(2221, "001", "brick", FILTERS, pipe_root="/pipe")
    assert pt.shell_line(step) == (
        "MODULES=nrca,nrcb,merged "
        "sbatch --parsable --job-name=brick2221-o001-reduce --array=0-3 "
        "'--export=ALL,PROPOSAL=2221,FIELD=001,SKIP=0,"
        "FILTERS=F405N F410M F466N F212N' "
        "/pipe/scripts/reduction/submit_reduction.sbatch")


def test_reduction_skip_step12():
    step = pt.reduction_step(2221, "001", "brick", ["F405N"], pipe_root="/pipe",
                             skip_step12=True)
    assert "SKIP=1" in " ".join(step["argv"])
    assert "--array=0-0" in step["argv"]


def test_modules_never_in_export():
    """Comma-valued MODULES must ride the environment, not the --export list
    (the SLURM --export comma trap)."""
    step = pt.reduction_step(2221, "001", "brick", FILTERS, pipe_root="/pipe")
    export_arg = next(a for a in step["argv"] if a.startswith("--export="))
    assert "MODULES" not in export_arg
    assert step["env"]["MODULES"] == "nrca,nrcb,merged"


def test_cataloging_golden_command():
    """Default EACH_SUFFIX is the plain no-destreak crf form (align_o<obs>_crf):
    fix_alignment always runs, and the no-destreak reduction path names the
    per-exposure crfs *_align_o<field>_crf.fits."""
    step = pt.cataloging_step(2221, "001", "brick", FILTERS, pipe_root="/pipe")
    assert pt.shell_line(step) == (
        "DEP='<REDUCTION_JOBID>' EACH_SUFFIX=align_o001_crf FIELD=001 "
        "FILTERS='F405N F410M F466N F212N' MODULES=merged PROPOSAL=2221 "
        "TARGET=brick /pipe/scripts/reduction/submit_cataloging_chain.sh")


def test_cataloging_destreak_optin():
    """--destreak selects the destreaked products' suffix."""
    step = pt.cataloging_step(2221, "001", "brick", FILTERS, pipe_root="/pipe",
                              destreak=True)
    assert step["env"]["EACH_SUFFIX"] == "destreak_o001_crf"


def test_cataloging_each_suffix_override_wins():
    step = pt.cataloging_step(2221, "001", "brick", FILTERS, pipe_root="/pipe",
                              each_suffix="custom_o001_crf", destreak=True)
    assert step["env"]["EACH_SUFFIX"] == "custom_o001_crf"


def test_cataloging_guard_vars_all_present():
    """submit_cataloging.sbatch hard-fails unless these travel together."""
    step = pt.cataloging_step(1182, "004", "brick", ["F200W"], pipe_root="/pipe")
    for var in ("PROPOSAL", "FIELD", "TARGET", "EACH_SUFFIX", "MODULES"):
        assert var in step["env"], var


def test_build_plan_field_from_programs_map():
    plan = pt.build_plan(4147, "012", filters=["F405N"], pipe_root="/pipe")
    assert plan[0]["argv"][2] == "--job-name=sgrc4147-o012-reduce"
    assert plan[1]["env"]["TARGET"] == "sgrc"
    assert plan[1]["env"]["EACH_SUFFIX"] == "align_o012_crf"


def test_build_plan_destreak_flag_threads_through():
    plan = pt.build_plan(4147, "012", filters=["F405N"], pipe_root="/pipe",
                         destreak=True)
    assert plan[1]["env"]["EACH_SUFFIX"] == "destreak_o012_crf"


def test_build_plan_requires_field_mapping():
    with pytest.raises(ValueError, match="no field mapping"):
        pt.build_plan(9999, "001", filters=["F405N"])


def test_build_plan_requires_filters():
    with pytest.raises(ValueError, match="filters required"):
        pt.build_plan(2221, "001")


def test_missing_scripts_refuses_execute(tmp_path):
    """--execute against a pipe-root without the pipeline scripts must refuse
    without ever invoking sbatch."""
    rc = pt.main(["--program", "2221", "--obs", "001", "--filters", "F405N",
                  "--pipe-root", str(tmp_path), "--execute"])
    assert rc == 1


def test_dry_run_prints_plan(tmp_path, capsys):
    rc = pt.main(["--program", "2221", "--obs", "001", "--filters", "F405N",
                  "F410M", "--pipe-root", str(tmp_path)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "dry-run" in out
    assert "brick2221-o001-reduce" in out
    assert "submit_cataloging_chain.sh" in out
