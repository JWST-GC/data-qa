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


def test_plan_member_matches_through_symlink(spec_file, tmp_path):
    """Membership comparison is realpath-normalized: a member registered
    under the real path still matches a spec that references it through a
    symlink (and vice versa)."""
    path, spec = spec_file
    root = spec["root"]
    os.makedirs(root, exist_ok=True)
    real = spec["fields"][0]["f212n_i2d"]
    link = str(tmp_path / "alias_f212n_i2d.fits")
    os.symlink(real, link)
    # registry holds the REAL path; the spec now points at the symlink
    with open(os.path.join(root, "F212N.members.json"), "w") as fh:
        json.dump({"members": [{"i2d": os.path.abspath(real),
                                "field": "fieldA", "tag": "F212N"}]}, fh)
    spec["fields"][0]["f212n_i2d"] = link
    p = HT.plan(spec)
    assert p["F212N"]["present"] == ["fieldA"]
    assert "fieldA" not in p["F212N"]["new"]


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


# ------------------------------------------------------------- docs example specs
DOCS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(HT.__file__))), "docs")


@pytest.mark.parametrize("fname,root_tail", [
    ("treasury_hips_spec.example.json", "jwst-gc-treasury-hips"),
    ("cmz_pretreasury_spec.example.json", "jwst-cmz-pretreasury-hips"),
])
def test_docs_example_specs_load_and_plan(fname, root_tail, capsys):
    """Both maintained example specs load and the plan verb runs on them."""
    path = os.path.join(DOCS, fname)
    spec = HT.load_spec(path)
    assert spec["root"].rstrip("/").endswith(root_tail)
    p = HT.plan(spec)
    assert set(p) == {"F212N", "LONG"}
    assert HT.main(["--spec", path]) == 0
    assert "root:" in capsys.readouterr().out


def test_treasury_spec_is_10678_only():
    """User decision 2026-07-22: the treasury spec holds ONLY program-10678
    GC_<n> tiles; pre-treasury fields (sgrb2/sgrc/sickle) stay out."""
    spec = HT.load_spec(os.path.join(DOCS, "treasury_hips_spec.example.json"))
    names = [f["name"] for f in spec["fields"]]
    assert all(n.startswith("GC_") for n in names)
    assert not {"sgrb2", "sgrc", "sickle"} & set(names)
    # nothing delivered yet: every entry is a TODO template, never buildable
    p = HT.plan(spec)
    assert p["F212N"]["new"] == [] and p["LONG"]["new"] == []


def test_cmz_pretreasury_spec_fields():
    spec = HT.load_spec(os.path.join(DOCS,
                                     "cmz_pretreasury_spec.example.json"))
    by_name = {f["name"]: f for f in spec["fields"]}
    assert {"sgrb2", "sgrc", "sickle"} <= set(by_name)
    assert "10678" not in json.dumps(spec["fields"])
    # sickle blue band = F210M per user decision
    assert by_name["sickle"].get("blue_band") == "F210M"


def test_build_field_blue_band_tag(monkeypatch, tmp_path):
    """A per-field blue_band overrides the F212N member tag (sickle F210M)
    while still folding into the F212N substrate tree."""
    calls = []

    class _FakeHips:
        @staticmethod
        def add_field_to_mono_hips(master, i2ds, name, tag=None, threads=8):
            calls.append((os.path.basename(master), tag))
            return {"tiles": 1}

    monkeypatch.setattr(HT, "_import_hips", lambda pipe_root=None: _FakeHips)
    blue = tmp_path / "sickle_f210m_i2d.fits"
    red = tmp_path / "sickle_f480m_i2d.fits"
    blue.write_bytes(b"")
    red.write_bytes(b"")
    spec = {"root": str(tmp_path / "cmz"),
            "fields": [{"name": "sickle", "f212n_i2d": str(blue),
                        "long_i2d": str(red), "long_band": "F480M",
                        "blue_band": "F210M"}]}
    HT.build_field(spec, "sickle")
    assert ("F212N", "F210M") in calls
    assert ("LONG", "F480M") in calls
