"""Retrieve JWST products for a Galactic Center observation from MAST.

Thin, dependency-light wrapper over ``astroquery.mast`` so QA reviewers can pull the
exact products an issue refers to.  Defaults to the combined ``i2d`` mosaics; use
``--product-type`` to fetch other suffixes (cal, crf, etc.).

Usage:
    python -m data_qa.retrieve_data --program 2221 --obs 001
    python -m data_qa.retrieve_data --program 1182 --obs 004 --product-type i2d cal \\
        --download-dir ./data
"""
from __future__ import annotations

import argparse
import sys


def _observations_table(program, instrument="NIRCam"):
    from astroquery.mast import Observations as MastObs
    return MastObs.query_criteria(obs_collection="JWST",
                                  proposal_id=str(int(program)),
                                  instrument_name=f"{instrument}*")


def list_observations(program, instrument="NIRCam"):
    """Return the distinct obs-numbers available on MAST for a program."""
    tbl = _observations_table(program, instrument)
    obsnums = sorted({str(o).split("-o")[1][:3]
                      for o in tbl["obs_id"] if "-o" in str(o)})
    return obsnums


def retrieve(program, obs, product_type=("i2d",), instrument="NIRCam",
             download_dir="./data", dry_run=False):
    """Download products for one observation.

    Filters the MAST product list to this obs-number and the requested suffix(es).
    Returns the manifest table (astroquery download result), or the filtered product
    table when ``dry_run``.
    """
    from astroquery.mast import Observations as MastObs

    tbl = _observations_table(program, instrument)
    oid = f"jw{int(program):05d}-o{obs}"
    sel = [i for i, o in enumerate(tbl["obs_id"]) if str(o).startswith(oid)]
    if not sel:
        print(f"no MAST observations matching {oid}", file=sys.stderr)
        return None
    products = MastObs.get_product_list(tbl[sel])
    suffixes = tuple(p.lower() for p in product_type)
    keep = [i for i, fn in enumerate(products["productFilename"])
            if any(f"_{s}." in str(fn).lower() for s in suffixes)]
    products = products[keep]
    print(f"{oid}: {len(products)} products matching {suffixes}")
    if dry_run:
        return products
    return MastObs.download_products(products, download_dir=download_dir)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--program", required=True, help="JWST program id, e.g. 2221")
    ap.add_argument("--obs", help="observation number, e.g. 001 (default: list them)")
    ap.add_argument("--instrument", default="NIRCam")
    ap.add_argument("--product-type", nargs="*", default=["i2d"],
                    help="product suffix(es) to fetch (default: i2d)")
    ap.add_argument("--download-dir", default="./data")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    if not args.obs:
        print(f"observations on MAST for program {int(args.program)}:")
        for o in list_observations(args.program, args.instrument):
            print(f"  jw{int(args.program):05d}-o{o}")
        return 0
    retrieve(args.program, args.obs, product_type=args.product_type,
             instrument=args.instrument, download_dir=args.download_dir,
             dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
