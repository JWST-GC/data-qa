"""Offline tests for data_qa.publish: command construction + gate refusal.

NEVER contacts the server: everything here is dry-run (the default) or pure
gate/command-construction logic.
"""
import json
import os

import pytest

from data_qa import publish as PB


def _write_validation(path, ok=True):
    with open(path, "w") as fh:
        json.dump({"pass": bool(ok), "checks": {}}, fh)


# --------------------------------------------------------------------------- avm
def test_avm_gate_refuses_unvalidated_png(tmp_path, capsys):
    (tmp_path / "foo.png").write_bytes(b"\x89PNG\r\n")
    rc = PB.main(["avm", "--src", str(tmp_path)])
    assert rc == 1
    assert "REFUSING" in capsys.readouterr().err


def test_avm_gate_refuses_failed_validation(tmp_path):
    (tmp_path / "foo.png").write_bytes(b"\x89PNG\r\n")
    _write_validation(tmp_path / "foo.validation.json", ok=False)
    assert PB.main(["avm", "--src", str(tmp_path)]) == 1


def test_avm_dry_run_command_with_validation(tmp_path, capsys):
    (tmp_path / "foo.png").write_bytes(b"\x89PNG\r\n")
    (tmp_path / "foo.jpg").write_bytes(b"\xff\xd8")
    _write_validation(tmp_path / "foo.validation.json", ok=True)
    rc = PB.main(["avm", "--src", str(tmp_path), "--name", "myimg"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "dry-run" in out
    assert ("rsync -ravpu --partial "
            f"{tmp_path}/ starformation:{PB.DOCROOT}/avm_images/myimg/" in out)


def test_avm_gate_accepts_validated_hips_tree(tmp_path):
    tree = tmp_path / "sgrb2_rgb_hips"
    tiledir = tree / "Norder3" / "Dir0"
    tiledir.mkdir(parents=True)
    (tiledir / "Npix42.png").write_bytes(b"\x89PNG\r\n")
    (tree / "properties").write_text("hips_order = 3\n")
    _write_validation(tmp_path / "sgrb2_rgb_hips.validation.json", ok=True)
    ok, problems = PB.gate_avm(str(tmp_path))
    assert ok, problems


def test_avm_gate_refuses_unvalidated_hips_tree(tmp_path):
    tree = tmp_path / "tree_hips"
    tiledir = tree / "Norder3" / "Dir0"
    tiledir.mkdir(parents=True)
    (tiledir / "Npix1.png").write_bytes(b"\x89PNG\r\n")
    (tree / "properties").write_text("hips_order = 3\n")
    ok, problems = PB.gate_avm(str(tmp_path))
    assert not ok
    assert "Npix1.png" in problems[0]


def test_no_force_flag_exists():
    with pytest.raises(SystemExit):
        PB.main(["avm", "--src", ".", "--force"])


# --------------------------------------------------------------------- products
def _fake_release(tmp_path, field="sgrb2", version="v9.9-2099.01",
                  manifest=True):
    d = tmp_path / "releases" / version / field
    d.mkdir(parents=True)
    if manifest:
        (d / "MANIFEST.json").write_text(json.dumps({
            "field": field, "version": version,
            "files": [
                {"category": "image", "url": "https://x/img_i2d.fits"},
                {"category": "catalog", "url": "https://x/cat_m7.fits"},
                {"category": "image", "url": None},
            ]}))
    return str(tmp_path / "releases")


def test_products_gate_requires_marker(tmp_path, capsys):
    root = _fake_release(tmp_path, manifest=False)
    src = tmp_path / "site"
    src.mkdir()
    rc = PB.main(["products", "--field", "sgrb2", "--src", str(src),
                  "--release-root", root])
    assert rc == 1
    assert "REFUSING" in capsys.readouterr().err


def test_products_gate_i_verified_fallback(tmp_path, capsys):
    root = _fake_release(tmp_path, manifest=False)
    src = tmp_path / "site"
    src.mkdir()
    rc = PB.main(["products", "--field", "sgrb2", "--src", str(src),
                  "--release-root", root, "--i-verified-gates"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "WARNING" in out and "dry-run" in out


def test_products_dry_run_command(tmp_path, capsys):
    root = _fake_release(tmp_path)
    src = tmp_path / "site"
    src.mkdir()
    rc = PB.main(["products", "--field", "sgrb2", "--src", str(src),
                  "--release-root", root])
    assert rc == 0
    out = capsys.readouterr().out
    assert f"{src}/ starformation:{PB.DOCROOT}/jwst-gc/" in out


def test_products_gate_missing_field(tmp_path):
    root = str(tmp_path / "releases")
    rc = PB.main(["products", "--field", "nope", "--src", str(tmp_path),
                  "--release-root", root])
    assert rc == 1


# -------------------------------------------------------------------- manifests
def test_manifests_generation_and_command(tmp_path, capsys):
    root = _fake_release(tmp_path, field="sgrb2")
    outdir = tmp_path / "man"
    rc = PB.main(["manifests", "--field", "sgrb2", "--release-root", root,
                  "--out-dir", str(outdir)])
    assert rc == 0
    images = (outdir / "sgrb2_images.txt").read_text().splitlines()
    catalogs = (outdir / "sgrb2_catalogs.txt").read_text().splitlines()
    assert images == ["https://x/img_i2d.fits"]     # url=None entry dropped
    assert catalogs == ["https://x/cat_m7.fits"]
    out = capsys.readouterr().out
    assert f"starformation:{PB.DOCROOT}/jwst-gc/" in out
    assert "dry-run" in out


def test_manifests_refuses_without_manifest(tmp_path, capsys):
    root = _fake_release(tmp_path, manifest=False)
    rc = PB.main(["manifests", "--field", "sgrb2", "--release-root", root])
    assert rc == 1
