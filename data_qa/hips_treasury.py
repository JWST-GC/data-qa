"""Build/extend the ``jwst-gc-treasury-hips`` two-color HiPS from a spec.

Thin, spec-driven driver over ``jwst_gc_pipeline.cmz.hips``: two incremental
MONO HiPS substrates (``<root>/F212N`` blue + ``<root>/LONG`` red -- F480M
preferred, F405N legacy fill; both long bands merge into the one LONG tree,
tagged per member) and the derived two-color PNG HiPS (``<root>/color``,
R=long, B=F212N, G=0.5*(R+B), global stretch).

Spec (JSON)::

    {
      "root": ".../treasury_hips/jwst-gc-treasury-hips",
      "fields": [
        {"name": "GC_001",
         "f212n_i2d": ".../jw10678-oNNN_tNNN_nircam_clear-f212n-merged_i2d.fits",
         "long_i2d":  ".../jw10678-oNNN_tNNN_nircam_clear-f480m-merged_i2d.fits",
         "long_band": "F480M"},
        ...
      ]
    }

Optional per-field ``"blue_band"`` overrides the blue member TAG (sickle's
F210M) while the mosaic still folds into the F212N substrate tree.

Two maintained example specs (user decision 2026-07-22): the treasury spec
(``docs/treasury_hips_spec.example.json``) is PROGRAM 10678 ONLY (GC_<n>
tiles, F212N+F480M by design; nothing delivered yet), while the pre-treasury
CMZ fields (sgrb2/sgrc/sickle...) live in
``docs/cmz_pretreasury_spec.example.json`` with its own root
(``jwst-cmz-pretreasury-hips``).

Verbs (``plan`` is the default and touches nothing):

* ``plan``   -- per band, which fields are NEW vs already in the master's
  ``members.json`` registry; prints the build/color commands it implies.
* ``build --field <name>`` -- ``add_field_to_mono_hips`` for BOTH bands of one
  field (incremental: only overlapping master tiles rewrite).  Compute-heavy on
  real mosaics: submit via ``sbatch`` (below), don't run on a login node.
* ``color``  -- ``derive_two_color_hips(<root>/F212N, <root>/LONG, <root>/color)``.
* ``sbatch`` -- print the submit command(s) (astronomy-dept-b QOS, job-name
  ``gc-treasury-hips-<field>`` at SUBMIT time) wrapping ``build`` via
  ``docs/submit_treasury_hips.sbatch``.

Usage::

    python -m data_qa.hips_treasury --spec spec.json                # plan
    python -m data_qa.hips_treasury build --spec spec.json --field sgrb2
    python -m data_qa.hips_treasury color --spec spec.json
    python -m data_qa.hips_treasury sbatch --spec spec.json [--field sgrb2]
"""
from __future__ import annotations

import argparse
import json
import os
import sys

DEFAULT_PIPE_ROOT = "/blue/adamginsburg/adamginsburg/repos/jwst-gc-pipeline"
VERBS = ("plan", "build", "color", "sbatch")
BLUE_BAND = "F212N"
LONG_DIR = "LONG"
SBATCH_TEMPLATE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "docs", "submit_treasury_hips.sbatch")


def _import_hips(pipe_root=None):
    """Import ``jwst_gc_pipeline.cmz.hips`` with a --pipe-root fallback."""
    try:
        from jwst_gc_pipeline.cmz import hips
        return hips
    except ImportError:
        pass
    root = pipe_root or DEFAULT_PIPE_ROOT
    if root and os.path.isdir(root) and root not in sys.path:
        sys.path.insert(0, root)
    from jwst_gc_pipeline.cmz import hips   # raises ImportError with context
    return hips


def load_spec(path):
    with open(path) as fh:
        spec = json.load(fh)
    if "root" not in spec or "fields" not in spec:
        raise ValueError(f"spec {path} must define 'root' and 'fields'")
    return spec


def band_dirs(spec):
    root = spec["root"].rstrip("/")
    return {BLUE_BAND: os.path.join(root, BLUE_BAND),
            LONG_DIR: os.path.join(root, LONG_DIR)}


def registry_members(master_dir):
    """Member i2d paths already folded into ``master_dir`` (its members.json
    sidecar, same path convention as ``cmz.hips.add_field_to_mono_hips``)."""
    path = master_dir.rstrip("/") + ".members.json"
    if not os.path.exists(path):
        return set()
    with open(path) as fh:
        return {m["i2d"] for m in json.load(fh).get("members", [])}


def plan(spec):
    """Classify each spec field as new vs already-built, per band.  Returns
    ``{band: {'new': [...], 'present': [...]}}`` keyed by field name."""
    dirs = band_dirs(spec)
    keys = {BLUE_BAND: "f212n_i2d", LONG_DIR: "long_i2d"}
    out = {}
    for band, master in dirs.items():
        members = registry_members(master)
        new, present, missing = [], [], []
        for f in spec["fields"]:
            i2d = f.get(keys[band])
            if not i2d:
                continue
            if i2d.startswith("TODO") or not os.path.exists(i2d):
                missing.append(f["name"])
            elif os.path.abspath(i2d) in members:
                present.append(f["name"])
            else:
                new.append(f["name"])
        out[band] = {"new": new, "present": present, "missing": missing,
                     "master": master}
    return out


def print_plan(spec, spec_path):
    p = plan(spec)
    print(f"[hips_treasury] root: {spec['root']}")
    for band, d in p.items():
        print(f"  {band}: master={d['master']}")
        print(f"    already built: {', '.join(d['present']) or '(none)'}")
        print(f"    new:           {', '.join(d['new']) or '(none)'}")
        if d["missing"]:
            print(f"    NOT buildable (i2d missing/TODO): "
                  f"{', '.join(d['missing'])}")
    todo = sorted(set(p[BLUE_BAND]["new"]) | set(p[LONG_DIR]["new"]))
    for name in todo:
        print(f"  would run: python -m data_qa.hips_treasury build "
              f"--spec {spec_path} --field {name}")
    if todo:
        print(f"  then:      python -m data_qa.hips_treasury color "
              f"--spec {spec_path}")
    else:
        print("  nothing to build; run 'color' to (re)derive the two-color "
              "layer if mono trees changed")
    return p


def build_field(spec, name, pipe_root=None, threads=8):
    """Fold one field's F212N + long mosaics into the mono masters."""
    field = next((f for f in spec["fields"] if f["name"] == name), None)
    if field is None:
        raise ValueError(f"field {name!r} not in spec "
                         f"({[f['name'] for f in spec['fields']]})")
    hips = _import_hips(pipe_root)
    dirs = band_dirs(spec)
    stats = {}
    for band, key in ((BLUE_BAND, "f212n_i2d"), (LONG_DIR, "long_i2d")):
        i2d = field.get(key)
        if not i2d:
            print(f"[hips_treasury] {name}: no {key} in spec; skipping {band}")
            continue
        # blue_band overrides the F212N member tag (sickle: F210M blue) while
        # the mosaic still folds into the F212N substrate tree
        tag = (field.get("long_band") if band == LONG_DIR
               else field.get("blue_band", BLUE_BAND))
        print(f"[hips_treasury] {band} += {name} ({i2d})")
        stats[band] = hips.add_field_to_mono_hips(
            dirs[band], [i2d], name, tag=tag, threads=threads)
        print(f"[hips_treasury]   {stats[band]}")
    return stats


def derive_color(spec, pipe_root=None):
    hips = _import_hips(pipe_root)
    dirs = band_dirs(spec)
    out = os.path.join(spec["root"].rstrip("/"), "color")
    print(f"[hips_treasury] derive two-color (B={dirs[BLUE_BAND]}, "
          f"R={dirs[LONG_DIR]}) -> {out}")
    n = hips.derive_two_color_hips(dirs[BLUE_BAND], dirs[LONG_DIR], out)
    print(f"[hips_treasury] wrote {n} color tiles")
    return out


def sbatch_command(spec_path, field):
    """The submit command for one field's build (job-name at SUBMIT time,
    astronomy-dept-b QOS per the standing SLURM rules)."""
    return (f"sbatch --job-name=gc-treasury-hips-{field} "
            f"--account=astronomy-dept --qos=astronomy-dept-b "
            f"{SBATCH_TEMPLATE} {os.path.abspath(spec_path)} {field}")


def build_parser():
    p = argparse.ArgumentParser(
        prog="python -m data_qa.hips_treasury", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("verb", nargs="?", default="plan", choices=VERBS)
    p.add_argument("--spec", required=True, help="JSON spec (see docstring)")
    p.add_argument("--field", help="field name (build/sbatch)")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--pipe-root", default=DEFAULT_PIPE_ROOT,
                   help="jwst-gc-pipeline checkout for the cmz.hips import "
                        "fallback")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    spec = load_spec(args.spec)
    if args.verb == "plan":
        print_plan(spec, args.spec)
    elif args.verb == "build":
        if not args.field:
            build_parser().error("build requires --field")
        build_field(spec, args.field, pipe_root=args.pipe_root,
                    threads=args.threads)
    elif args.verb == "color":
        derive_color(spec, pipe_root=args.pipe_root)
    elif args.verb == "sbatch":
        if args.field:
            names = [args.field]
        else:
            p = plan(spec)
            names = sorted(set(p[BLUE_BAND]["new"]) | set(p[LONG_DIR]["new"]))
        if not names:
            print("[hips_treasury] nothing new to build")
        for name in names:
            print(sbatch_command(args.spec, name))
    return 0


if __name__ == "__main__":
    sys.exit(main())
