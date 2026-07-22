"""Gated rsync pushes to the starformation web host.

Every verb PRINTS the exact rsync command(s) by default; nothing touches the
server unless ``--execute`` is given AND the verb's gate passes.  Gates fail
CLOSED and there is deliberately NO ``--force``: fix the gate, don't override
it.

Verbs:

* ``avm --src <dir-or-file> [--name N]`` -> ``htdocs/avm_images/<N>/``.
  Gate: every PNG/JPG pushed must have a sibling ``<stem>.validation.json``
  with ``"pass": true`` (written by ``data_qa.rgb_treasury``), OR sit inside a
  HiPS tile tree (a directory carrying a ``properties`` file) whose SOURCE
  image validated -- i.e. ``<treename>.validation.json`` next to the tree root
  passes.  SECOND gate (user decision 4): the verdict's
  ``checks.star_positions`` must PASS; a verdict whose star check was skipped
  (no reference catalog) or predates the check needs an explicit
  ``--no-star-check`` acknowledgment, and a star check that ran and FAILED
  refuses outright (fail-closed, no override).

* ``products --field <f> --src <dir> [--dest-sub S]`` -> ``htdocs/jwst-gc/``.
  Gate: the field must actually be STAGED -- ``stage_release.py`` marks staged
  output by writing ``MANIFEST.json`` (plus README.md/CHECKSUMS.sha256) into
  ``/orange/adamginsburg/jwst/releases/<version>/<field>/``; we require that
  marker.  If a release dir exists WITHOUT a MANIFEST.json (pre-marker
  stagings), ``--i-verified-gates`` accepts the dir's existence instead --
  explicit and logged, never silent.

* ``manifests --field <f> [--staged-dir D] [--out-dir O]`` -- regenerate
  ``<field>_images.txt`` / ``<field>_catalogs.txt`` (the URL lists
  ``data_qa.observations`` consumes) from the staged MANIFEST.json listing,
  write them locally, and print/push the rsync to ``htdocs/jwst-gc/``.

Usage::

    python -m data_qa.publish avm --src /path/to/avm_images/sgrb2_rgb_dir
    python -m data_qa.publish products --field sgrb2 --src /path/site --execute
    python -m data_qa.publish manifests --field sgrb2 --out-dir /tmp/man
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

SSH_ALIAS = "starformation"
DOCROOT = "/h/cnswww-starformation.astro/starformation.astro.ufl.edu/htdocs"
RELEASE_ROOT = "/orange/adamginsburg/jwst/releases"
RSYNC = ["rsync", "-ravpu", "--partial"]
IMAGE_EXTS = (".png", ".jpg", ".jpeg")


# --------------------------------------------------------------------------- gates
def _load_validation(path):
    """The validation JSON dict at ``path``; None when missing/unreadable."""
    if not os.path.exists(path):
        return None
    try:
        with open(path) as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        return None


def _star_state(verdict):
    """The star-position (second-gate) state of a validation verdict:
    'pass' | 'fail' | 'skipped' | 'absent' (pre-star-check validation JSON)."""
    sp = (verdict.get("checks") or {}).get("star_positions")
    if not isinstance(sp, dict):
        return "absent"
    if sp.get("skipped"):
        return "skipped"
    return "pass" if sp.get("pass") else "fail"


def _covering_verdicts(img, top):
    """[(sidecar_path, verdict_dict)] candidates covering ``img``: its own
    sibling validation first, then each enclosing HiPS tree's source
    validation (tree root = dir with a 'properties' file; sidecar lives NEXT
    TO the tree root as <treename>.validation.json)."""
    out = []
    stem, _ = os.path.splitext(img)
    sib = stem + ".validation.json"
    verdict = _load_validation(sib)
    if verdict is not None:
        out.append((sib, verdict))
    d = os.path.dirname(os.path.abspath(img))
    top = os.path.abspath(top)
    while len(d) >= len(top):
        if os.path.exists(os.path.join(d, "properties")):
            sidecar = os.path.join(os.path.dirname(d),
                                   os.path.basename(d) + ".validation.json")
            verdict = _load_validation(sidecar)
            if verdict is not None:
                out.append((sidecar, verdict))
        if d == top:
            break
        d = os.path.dirname(d)
    return out


def _image_problem(img, top, no_star_check=False):
    """Why ``img`` is not pushable (None when it is).

    Pushable = some covering validation passes AND its star-position check
    (the second gate, user decision 4) is satisfied: star 'pass', or
    'skipped'/'absent' explicitly acknowledged via --no-star-check.  A star
    check that RAN and FAILED is fail-closed: no flag overrides it."""
    candidates = _covering_verdicts(img, top)
    if not candidates:
        return "no validation.json (sibling or HiPS-tree)"
    reasons = []
    for sidecar, verdict in candidates:
        if not verdict.get("pass"):
            reasons.append(f"validation FAILED ({sidecar})")
            continue
        stars = _star_state(verdict)
        if stars == "pass" or (no_star_check
                               and stars in ("skipped", "absent")):
            return None
        if stars == "fail":
            reasons.append(f"star-position check FAILED ({sidecar}); "
                           "fail-closed, --no-star-check cannot override")
        else:
            reasons.append(f"star-position check {stars} ({sidecar}); pass "
                           "--no-star-check to acknowledge pushing without it")
    return "; ".join(reasons)


def gate_avm(src, no_star_check=False):
    """Return (ok, problems).  Every PNG/JPG under ``src`` must be covered."""
    src = os.path.abspath(src)
    if os.path.isfile(src):
        imgs = [src] if src.lower().endswith(IMAGE_EXTS) else []
        top = os.path.dirname(src)
    else:
        imgs = [p for p in glob.glob(os.path.join(src, "**", "*"),
                                     recursive=True)
                if p.lower().endswith(IMAGE_EXTS)]
        top = src
    problems = []
    for p in sorted(imgs):
        why = _image_problem(p, top, no_star_check=no_star_check)
        if why:
            problems.append(f"{p}: {why}")
    if not imgs:
        problems.append(f"{src}: no PNG/JPG images found to push")
    return (not problems), problems


def find_staged_dir(field, release_root=RELEASE_ROOT, version=None):
    """Newest ``<release_root>/v*/<field>`` (or the pinned ``version``'s)."""
    if version:
        cands = [os.path.join(release_root, version, field)]
    else:
        cands = sorted(glob.glob(os.path.join(release_root, "v*", field)),
                       reverse=True)
    for d in cands:
        if os.path.isdir(d):
            return d
    return None


def gate_products(field, release_root=RELEASE_ROOT, version=None,
                  i_verified_gates=False):
    """Return (ok, message).  Requires the stage_release marker
    (``MANIFEST.json`` in the staged field dir)."""
    staged = find_staged_dir(field, release_root, version)
    if staged is None:
        return False, (f"no staged release dir for field {field!r} under "
                       f"{release_root} -- stage_release.py first")
    marker = os.path.join(staged, "MANIFEST.json")
    if os.path.exists(marker):
        return True, f"staged-release marker OK: {marker}"
    if i_verified_gates:
        return True, (f"WARNING: {marker} missing; accepting on "
                      f"--i-verified-gates because {staged} exists")
    return False, (f"{staged} exists but has no MANIFEST.json marker; "
                   f"re-stage, or pass --i-verified-gates if you have "
                   f"verified the staging by hand")


# --------------------------------------------------------------------------- commands
def build_avm_command(src, name=None):
    src = os.path.abspath(src)
    if os.path.isfile(src):
        name = name or os.path.basename(os.path.dirname(src))
        return RSYNC + [src, f"{SSH_ALIAS}:{DOCROOT}/avm_images/{name}/"]
    name = name or os.path.basename(src.rstrip("/"))
    return RSYNC + [src.rstrip("/") + "/",
                    f"{SSH_ALIAS}:{DOCROOT}/avm_images/{name}/"]


def build_products_command(src, dest_sub=None):
    dest = f"{SSH_ALIAS}:{DOCROOT}/jwst-gc/"
    if dest_sub:
        dest = f"{SSH_ALIAS}:{DOCROOT}/jwst-gc/{dest_sub.strip('/')}/"
    return RSYNC + [os.path.abspath(src).rstrip("/") + "/", dest]


def build_manifests_command(paths):
    return RSYNC + [os.path.abspath(p) for p in paths] + \
        [f"{SSH_ALIAS}:{DOCROOT}/jwst-gc/"]


def generate_manifests(field, staged_dir, out_dir):
    """Write ``<field>_images.txt`` / ``<field>_catalogs.txt`` from the staged
    MANIFEST.json (category -> URL list, the format the release portal serves
    and ``data_qa.observations`` parses).  Returns the written paths."""
    manifest_path = os.path.join(staged_dir, "MANIFEST.json")
    with open(manifest_path) as fh:
        manifest = json.load(fh)
    os.makedirs(out_dir, exist_ok=True)
    written = []
    for category, suffix in (("image", "_images.txt"),
                             ("catalog", "_catalogs.txt")):
        urls = [f["url"] for f in manifest.get("files", [])
                if f.get("category") == category and f.get("url")]
        out = os.path.join(out_dir, field + suffix)
        with open(out, "w") as fh:
            fh.write("\n".join(urls) + ("\n" if urls else ""))
        written.append(out)
        print(f"[publish] wrote {out} ({len(urls)} URLs)")
    return written


def _run_or_print(cmd, execute):
    printable = " ".join(cmd)
    if execute:
        import subprocess
        print(f"[publish] EXECUTING: {printable}")
        subprocess.run(cmd, check=True)
    else:
        print(f"[publish] dry-run (pass --execute to run):\n  {printable}")
    return 0


# --------------------------------------------------------------------------- CLI
def build_parser():
    p = argparse.ArgumentParser(
        prog="python -m data_qa.publish", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="verb", required=True)

    a = sub.add_parser("avm", help="push AVM-tagged images / HiPS trees")
    a.add_argument("--src", required=True, help="directory (or single image)")
    a.add_argument("--name", help="server subdir under avm_images/ "
                                  "(default: src basename)")
    a.add_argument("--no-star-check", action="store_true",
                   help="acknowledge pushing images whose star-position check "
                        "was SKIPPED (no reference catalog); a star check that "
                        "ran and FAILED still refuses")
    a.add_argument("--execute", action="store_true")

    r = sub.add_parser("products", help="push release-products web content")
    r.add_argument("--field", required=True)
    r.add_argument("--src", required=True, help="staged web content dir")
    r.add_argument("--dest-sub", help="subdir under htdocs/jwst-gc/")
    r.add_argument("--release-root", default=RELEASE_ROOT)
    r.add_argument("--release-version", help="pin a release version dir")
    r.add_argument("--i-verified-gates", action="store_true",
                   help="accept a marker-less (pre-MANIFEST) staged dir; "
                        "use only after verifying the staging by hand")
    r.add_argument("--execute", action="store_true")

    m = sub.add_parser("manifests", help="regenerate + push field URL lists")
    m.add_argument("--field", required=True)
    m.add_argument("--staged-dir", help="staged release field dir "
                                        "(default: newest under release root)")
    m.add_argument("--release-root", default=RELEASE_ROOT)
    m.add_argument("--release-version")
    m.add_argument("--out-dir", default=".",
                   help="where to write the regenerated txt files")
    m.add_argument("--execute", action="store_true")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)

    if args.verb == "avm":
        ok, problems = gate_avm(args.src, no_star_check=args.no_star_check)
        if not ok:
            print("[publish] REFUSING avm push; gate failed:", file=sys.stderr)
            for pr in problems:
                print(f"  - {pr}", file=sys.stderr)
            return 1
        return _run_or_print(build_avm_command(args.src, args.name),
                             args.execute)

    if args.verb == "products":
        ok, msg = gate_products(args.field, release_root=args.release_root,
                                version=args.release_version,
                                i_verified_gates=args.i_verified_gates)
        print(f"[publish] {msg}")
        if not ok:
            print("[publish] REFUSING products push; gate failed",
                  file=sys.stderr)
            return 1
        return _run_or_print(build_products_command(args.src, args.dest_sub),
                             args.execute)

    if args.verb == "manifests":
        staged = args.staged_dir or find_staged_dir(
            args.field, args.release_root, args.release_version)
        if staged is None or not os.path.exists(
                os.path.join(staged, "MANIFEST.json")):
            print(f"[publish] no staged MANIFEST.json for {args.field!r} "
                  f"(looked in {staged or args.release_root}); cannot "
                  f"regenerate manifests", file=sys.stderr)
            return 1
        written = generate_manifests(args.field, staged, args.out_dir)
        return _run_or_print(build_manifests_command(written), args.execute)

    return 2


if __name__ == "__main__":
    sys.exit(main())
