"""Offline tests for data_qa.hips_treasury (plan verb + sbatch command)."""
import json
import os

import pytest

from data_qa import hips_treasury as HT


@pytest.fixture
def spec_file(tmp_path):
    paths = {}
    for stem in ("fieldA_f212n", "fieldA_f480m", "fieldB_f212n",
                 "fieldB_f405n"):
        p = tmp_path / f"{stem}_i2d.fits"
        p.write_bytes(b"")
        paths[stem] = str(p)
    spec = {
        "root": str(tmp_path / "treasury"),
        "fields": [
            {"name": "fieldA", "f212n_i2d": paths["fieldA_f212n"],
             "long_i2d": paths["fieldA_f480m"], "long_band": "F480M"},
            {"name": "fieldB", "f212n_i2d": paths["fieldB_f212n"],
             "long_i2d": paths["fieldB_f405n"], "long_band": "F405N"},
            {"name": "fieldC", "f212n_i2d": "TODO:<not yet reduced>",
             "long_i2d": str(tmp_path / "does_not_exist_i2d.fits"),
             "long_band": "F480M"},
        ],
    }
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(spec))
    return str(path), spec


def test_plan_all_new(spec_file):
    path, spec = spec_file
    p = HT.plan(spec)
    assert p["F212N"]["new"] == ["fieldA", "fieldB"]
    assert p["F212N"]["present"] == []
    assert p["LONG"]["new"] == ["fieldA", "fieldB"]
    # TODO-placeholder / nonexistent inputs are flagged, never buildable
    assert p["F212N"]["missing"] == ["fieldC"]
    assert p["LONG"]["missing"] == ["fieldC"]


def test_plan_detects_existing_member(spec_file):
    path, spec = spec_file
    root = spec["root"]
    os.makedirs(root, exist_ok=True)
    # registry sidecar exactly where cmz.hips.add_field_to_mono_hips puts it
    with open(os.path.join(root, "F212N.members.json"), "w") as fh:
        json.dump({"members": [
            {"i2d": os.path.abspath(spec["fields"][0]["f212n_i2d"]),
             "field": "fieldA", "tag": "F212N"}]}, fh)
    p = HT.plan(spec)
    assert p["F212N"]["present"] == ["fieldA"]
    assert p["F212N"]["new"] == ["fieldB"]
    assert p["LONG"]["new"] == ["fieldA", "fieldB"]   # LONG registry untouched


def test_plan_cli_default_verb(spec_file, capsys):
    path, _spec = spec_file
    assert HT.main(["--spec", path]) == 0
    out = capsys.readouterr().out
    assert "new:" in out
    assert "build --spec" in out and "--field fieldA" in out


def test_sbatch_command(spec_file, capsys):
    path, _spec = spec_file
    cmd = HT.sbatch_command(path, "fieldA")
    assert "--job-name=gc-treasury-hips-fieldA" in cmd
    assert "--qos=astronomy-dept-b" in cmd
    assert "--account=astronomy-dept" in cmd
    assert "submit_treasury_hips.sbatch" in cmd
    assert HT.main(["sbatch", "--spec", path]) == 0
    out = capsys.readouterr().out
    assert "gc-treasury-hips-fieldA" in out and "gc-treasury-hips-fieldB" in out


def test_build_unknown_field_raises(spec_file):
    _path, spec = spec_file
    with pytest.raises(ValueError, match="not in spec"):
        HT.build_field(spec, "nope")


def test_spec_missing_keys_raises(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"fields": []}))
    with pytest.raises(ValueError, match="root"):
        HT.load_spec(str(bad))
