"""One home for the jwst-gc-pipeline checkout path (issue #85).

The path used to be written out in four modules, and only the test copy read
``$PIPE_ROOT`` -- so a scrontab entry or a GitHub runner could point the tests
at a checkout while the code it exercised still used the HiPerGator path.
"""
import ast
import os
import pathlib

from data_qa import hips_treasury, paths, pipeline_policy, pipeline_trigger
from data_qa import rgb_treasury


REPO = pathlib.Path(__file__).resolve().parent.parent


def test_pipe_root_defaults_to_the_hipergator_checkout(monkeypatch):
    monkeypatch.delenv(paths.PIPE_ROOT_ENV, raising=False)
    assert paths.pipe_root() == paths.DEFAULT_PIPE_ROOT
    assert paths.pipe_root("/somewhere/else") == "/somewhere/else"


def test_pipe_root_reads_the_environment_at_call_time(monkeypatch):
    """Call time, not import time: a scrontab entry, a CI job and a test all
    set the variable in the process that then imports data_qa."""
    monkeypatch.setenv(paths.PIPE_ROOT_ENV, "/scratch/pipe")
    assert paths.pipe_root() == "/scratch/pipe"
    assert paths.pipe_root("/somewhere/else") == "/scratch/pipe"


def test_empty_pipe_root_falls_back(monkeypatch):
    """PIPE_ROOT= in an env file must not resolve every pipeline path against
    the current directory."""
    monkeypatch.setenv(paths.PIPE_ROOT_ENV, "")
    assert paths.pipe_root() == paths.DEFAULT_PIPE_ROOT


def test_every_module_reports_the_same_default():
    """The historical per-module names stay, as re-exports of the one path."""
    for mod in (pipeline_trigger, pipeline_policy, rgb_treasury, hips_treasury):
        assert mod.DEFAULT_PIPE_ROOT == paths.DEFAULT_PIPE_ROOT, mod.__name__


def test_pipe_root_env_reaches_the_trigger(monkeypatch, tmp_path):
    """The point of the consolidation: the env knob steers the code, not only
    the tests.  A plan built with no --pipe-root submits the scripts of the
    checkout $PIPE_ROOT names."""
    monkeypatch.setenv(paths.PIPE_ROOT_ENV, str(tmp_path))
    monkeypatch.setattr(pipeline_trigger, "registry_preflight",
                        lambda *a, **kw: None)
    plan = pipeline_trigger.build_plan(2221, "001", field="brick",
                                       filters=["F405N"], probe=False)
    assert plan[0]["argv"][-1] == str(tmp_path / pipeline_trigger.REDUCTION_SBATCH)
    assert plan[1]["argv"] == [str(tmp_path / pipeline_trigger.CATALOGING_CHAIN)]
    assert pipeline_trigger.missing_scripts() == [
        pipeline_trigger.REDUCTION_SBATCH, pipeline_trigger.CATALOGING_CHAIN]


def test_no_module_hardcodes_the_checkout_path():
    """The consolidation, pinned: data_qa/paths.py is the only file carrying
    the literal (docs and the sbatch template are prose/ops, not code)."""
    literal = paths.DEFAULT_PIPE_ROOT
    offenders = []
    for py in sorted(REPO.glob("**/*.py")):
        rel = py.relative_to(REPO)
        if rel.parts[0] not in ("data_qa", "tests", "scripts"):
            continue
        if rel == pathlib.Path("data_qa/paths.py"):
            continue
        if literal in py.read_text():
            offenders.append(str(rel))
    assert offenders == []


def test_paths_is_stdlib_only():
    """The trigger path stays importable in a bare interpreter."""
    tree = ast.parse(pathlib.Path(paths.__file__).read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            imported.add((node.module or "").split(".")[0])
    assert imported <= {"os", "__future__"}
