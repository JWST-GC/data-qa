"""Offline tests for data_qa.publish: command construction + gate refusal.

NEVER contacts the server: everything here is dry-run (the default) or pure
gate/command-construction logic.
"""
import hashlib
import json
import os

import pytest

from data_qa import publish as PB


def _sha(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def _write_validation(path, ok=True, stars="pass", binding=True):
    """stars: 'pass' | 'fail' | 'skipped' | 'absent' (pre-star-check JSON).
    ``binding=True`` records outputs.{png,jpg}_sha256 for the sibling
    <stem>.png/.jpg files that exist (as rgb_treasury.validate does);
    ``binding=False`` emulates a pre-binding verdict."""
    checks = {}
    if stars == "pass":
        checks["star_positions"] = {"pass": True, "median_offset_px": 0.4,
                                    "n_used": 60, "matched_fraction": 0.9}
    elif stars == "fail":
        checks["star_positions"] = {"pass": False, "median_offset_px": 6.2,
                                    "n_used": 40, "matched_fraction": 0.6}
    elif stars == "skipped":
        checks["star_positions"] = {"skipped": True,
                                    "reason": "no reference catalog"}
    doc = {"pass": bool(ok), "checks": checks}
    if binding:
        stem = str(path)[: -len(".validation.json")]
        outputs = {}
        for ext, key in ((".png", "png_sha256"), (".jpg", "jpg_sha256")):
            if os.path.exists(stem + ext):
                outputs[ext[1:]] = stem + ext
                outputs[key] = _sha(stem + ext)
        doc["outputs"] = outputs
    with open(path, "w") as fh:
        json.dump(doc, fh)


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
    assert ("rsync -ravp --partial "
            f"{tmp_path}/ starformation:{PB.DOCROOT}/avm_images/myimg/" in out)


def _make_hips_tree(tmp_path, name="sgrb2_rgb_hips", source_png=True):
    """A minimal HiPS tile tree + (optionally) its source PNG next to it."""
    tree = tmp_path / name
    tiledir = tree / "Norder3" / "Dir0"
    tiledir.mkdir(parents=True)
    (tiledir / "Npix42.png").write_bytes(b"\x89PNG\r\ntile")
    (tree / "properties").write_text("hips_order = 3\n")
    if source_png:
        (tmp_path / (name + ".png")).write_bytes(b"\x89PNG\r\nsource")
    return tree


def test_avm_gate_accepts_validated_hips_tree(tmp_path):
    _make_hips_tree(tmp_path)
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


# ------------------------------------------------------- avm star-position gate
def test_avm_gate_star_skipped_needs_acknowledgment(tmp_path, capsys):
    (tmp_path / "foo.png").write_bytes(b"\x89PNG\r\n")
    _write_validation(tmp_path / "foo.validation.json", ok=True,
                      stars="skipped")
    rc = PB.main(["avm", "--src", str(tmp_path)])
    assert rc == 1
    assert "--no-star-check" in capsys.readouterr().err


def test_avm_gate_star_skipped_accepted_with_flag(tmp_path, capsys):
    (tmp_path / "foo.png").write_bytes(b"\x89PNG\r\n")
    _write_validation(tmp_path / "foo.validation.json", ok=True,
                      stars="skipped")
    rc = PB.main(["avm", "--src", str(tmp_path), "--no-star-check"])
    assert rc == 0
    assert "dry-run" in capsys.readouterr().out


def test_avm_gate_star_failed_is_fail_closed(tmp_path, capsys):
    """A star check that RAN and FAILED refuses even with --no-star-check."""
    (tmp_path / "foo.png").write_bytes(b"\x89PNG\r\n")
    _write_validation(tmp_path / "foo.validation.json", ok=True, stars="fail")
    assert PB.main(["avm", "--src", str(tmp_path)]) == 1
    rc = PB.main(["avm", "--src", str(tmp_path), "--no-star-check"])
    assert rc == 1
    assert "fail-closed" in capsys.readouterr().err


def test_avm_gate_star_absent_treated_as_skipped(tmp_path):
    """Pre-star-check validation JSON (no star_positions key) needs the same
    explicit acknowledgment as a skip."""
    (tmp_path / "foo.png").write_bytes(b"\x89PNG\r\n")
    _write_validation(tmp_path / "foo.validation.json", ok=True,
                      stars="absent")
    ok, problems = PB.gate_avm(str(tmp_path))
    assert not ok
    assert "absent" in problems[0]
    ok, problems = PB.gate_avm(str(tmp_path), no_star_check=True)
    assert ok, problems


def test_avm_gate_hips_tree_star_skipped(tmp_path):
    _make_hips_tree(tmp_path, name="tree_hips")
    _write_validation(tmp_path / "tree_hips.validation.json", ok=True,
                      stars="skipped")
    ok, _ = PB.gate_avm(str(tmp_path))
    assert not ok
    ok, problems = PB.gate_avm(str(tmp_path), no_star_check=True)
    assert ok, problems


# ------------------------------------------------- avm verdict<->output binding
def test_avm_gate_refuses_regenerated_png_after_verdict(tmp_path, capsys):
    """The stale-verdict case: PNG regenerated AFTER validation -> refused."""
    (tmp_path / "foo.png").write_bytes(b"\x89PNG\r\nvalidated bytes")
    _write_validation(tmp_path / "foo.validation.json", ok=True)
    ok, problems = PB.gate_avm(str(tmp_path))
    assert ok, problems                       # matching hash passes
    (tmp_path / "foo.png").write_bytes(b"\x89PNG\r\nREGENERATED bytes")
    rc = PB.main(["avm", "--src", str(tmp_path)])
    assert rc == 1
    err = capsys.readouterr().err
    assert "STALE" in err and "png_sha256" in err


def test_avm_gate_refuses_missing_outputs_binding(tmp_path):
    """A verdict with no outputs hashes (pre-binding JSON) is refused."""
    (tmp_path / "foo.png").write_bytes(b"\x89PNG\r\n")
    _write_validation(tmp_path / "foo.validation.json", ok=True,
                      binding=False)
    ok, problems = PB.gate_avm(str(tmp_path))
    assert not ok
    assert "outputs.png_sha256" in problems[0]


def test_avm_gate_binds_jpg_separately(tmp_path):
    """Regenerating only the JPG refuses only the JPG."""
    (tmp_path / "foo.png").write_bytes(b"\x89PNG\r\n")
    (tmp_path / "foo.jpg").write_bytes(b"\xff\xd8jpg")
    _write_validation(tmp_path / "foo.validation.json", ok=True)
    (tmp_path / "foo.jpg").write_bytes(b"\xff\xd8other")
    ok, problems = PB.gate_avm(str(tmp_path))
    assert not ok
    assert len(problems) == 1
    assert "foo.jpg" in problems[0] and "jpg_sha256" in problems[0]


def test_avm_tree_unbound_requires_acknowledgment(tmp_path):
    """A tree verdict with no source-PNG binding needs --accept-unbound-tree."""
    _make_hips_tree(tmp_path, name="tree_hips", source_png=False)
    _write_validation(tmp_path / "tree_hips.validation.json", ok=True,
                      binding=False)
    ok, problems = PB.gate_avm(str(tmp_path))
    assert not ok
    assert "--accept-unbound-tree" in problems[0]
    ok, problems = PB.gate_avm(str(tmp_path), accept_unbound_tree=True)
    assert ok, problems


def test_avm_tree_stale_source_png_refused(tmp_path):
    """Source PNG regenerated after the tree verdict -> tree push refused,
    and --accept-unbound-tree does NOT override a bound-but-stale verdict."""
    _make_hips_tree(tmp_path, name="tree_hips")
    _write_validation(tmp_path / "tree_hips.validation.json", ok=True)
    (tmp_path / "tree_hips.png").write_bytes(b"\x89PNG\r\nREGENERATED")
    ok, problems = PB.gate_avm(str(tmp_path), accept_unbound_tree=True)
    assert not ok
    assert any("STALE" in p for p in problems)


# ----------------------------------------------------- avm push-scope manifest
def test_push_manifest_lists_everything(tmp_path):
    (tmp_path / "foo.png").write_bytes(b"\x89PNG\r\n")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "bar.fits").write_bytes(b"SIMPLE")
    rels = [rel for rel, _ in PB.push_manifest(str(tmp_path))]
    assert rels == ["foo.png", os.path.join("sub", "bar.fits")]


def test_avm_extra_files_require_flag(tmp_path, capsys):
    """Non-image files outside a HiPS tree refuse without --allow-extra-files
    (gate scope = push scope), and are listed."""
    (tmp_path / "foo.png").write_bytes(b"\x89PNG\r\n")
    _write_validation(tmp_path / "foo.validation.json", ok=True)
    (tmp_path / "stray.fits").write_bytes(b"SIMPLE")
    (tmp_path / "index.html").write_text("<html/>")
    rc = PB.main(["avm", "--src", str(tmp_path)])
    assert rc == 1
    err = capsys.readouterr().err
    assert "--allow-extra-files" in err
    assert "stray.fits" in err and "index.html" in err
    rc = PB.main(["avm", "--src", str(tmp_path), "--allow-extra-files"])
    assert rc == 0
    cap = capsys.readouterr()
    assert "WARNING" in cap.err               # still listed, just allowed
    assert "dry-run" in cap.out


def test_avm_hips_tree_internals_are_not_extra(tmp_path):
    """properties/Allsky/metadata inside a HiPS tree are part of the tree,
    never 'extra'; the sibling .validation.json sidecar is fine too."""
    _make_hips_tree(tmp_path, name="tree_hips")
    tree = tmp_path / "tree_hips"
    (tree / "Moc.fits").write_bytes(b"SIMPLE")
    (tree / "index.html").write_text("<html/>")
    _write_validation(tmp_path / "tree_hips.validation.json", ok=True)
    assert PB.manifest_extra_files(str(tmp_path)) == []


def test_rsync_flags_overwrite_republish():
    """-u (skip-newer) must stay out: corrected re-publishes overwrite."""
    assert PB.RSYNC == ["rsync", "-ravp", "--partial"]
    assert "-ravpu" not in " ".join(PB.RSYNC)


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
