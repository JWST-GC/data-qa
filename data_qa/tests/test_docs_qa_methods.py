"""Guards that keep ``docs/qa_methods.md`` describing the code that exists.

The page is the reference every tracking-issue comment links to, so a claim it makes about a
metrics key or a constant has to be checkable.  These tests pin the three claims that drifted in
PR #144 review: the name of the per-stage flag, the file-sampling threshold, and the fact that the
metrics path in prose is a real path rather than a Python expression.
"""
import os
import re

DOC = os.path.join(os.path.dirname(__file__), "..", "..", "docs", "qa_methods.md")


def _doc_text():
    with open(DOC) as fh:
        return fh.read()


def test_doc_names_only_the_flags_the_metrics_actually_carry():
    """``passed`` is the one per-stage flag.  A reader told to look for ``failed`` in
    ``data_qa/metrics/<obsid>.json`` will not find it -- nothing writes that key."""
    from .. import diagnostics as D
    src = open(D.__file__).read()
    assert 'metrics["passed"]' in src or "passed=" in src
    assert '"failed"' not in src and "'failed'" not in src, \
        "diagnostics.py grew a 'failed' metrics key; update docs/qa_methods.md too"
    assert not re.search(r"`failed`\s*(flag|key)", _doc_text()), \
        "docs/qa_methods.md names a `failed` flag that the metrics JSON does not carry"


def test_doc_file_sampling_threshold_matches_the_code():
    """The doc quotes the per-DIRECTORY sampling threshold as a number; it must be the number
    the code uses, and the doc must say the limit applies per directory (a stage that read 40
    files spread over four directories still lists all of them)."""
    from .. import diagnostics as D
    text = _doc_text()
    para = text.split("<a id=\"glossary\">")[0]
    assert f"more than {D._INPUTS_SAMPLE_AT} files" in para, \
        (f"docs/qa_methods.md must quote _INPUTS_SAMPLE_AT = {D._INPUTS_SAMPLE_AT} "
         "as the file-sampling threshold")
    assert "**directory**" in para, \
        "docs/qa_methods.md must say the sampling threshold applies per directory, not per stage"
    assert f"first {'three' if D._INPUTS_SAMPLE_HEAD == 3 else D._INPUTS_SAMPLE_HEAD}" in para


def test_doc_uses_the_metrics_path_not_a_python_expression():
    """``o.obsid`` is an attribute access in ``make_issues.py``; the on-disk path is
    ``data_qa/metrics/<obsid>.json``."""
    text = _doc_text()
    assert "o.obsid" not in text, \
        "docs/qa_methods.md leaked the code expression `o.obsid` into prose"
    assert "data_qa/metrics/<obsid>.json" in text
