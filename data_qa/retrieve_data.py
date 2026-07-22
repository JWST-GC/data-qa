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

# Bound every MAST HTTP call.  History: an unbounded astroquery MAST request once
# hung a pipeline job for 22 h; astroquery's default conf.timeout is 600 s PER
# request with silent retries on top.  120 s is generous for a metadata query.
MAST_TIMEOUT_S = 120
MAST_PAGESIZE = 5000


def configure_mast(timeout_s=MAST_TIMEOUT_S, pagesize=MAST_PAGESIZE):
    """Set the astroquery MAST request timeout + page size.

    Note: modern astroquery (>=0.4) has NO ``Observations.TIMEOUT`` attribute --
    the knobs live on the module-level config object ``astroquery.mast.conf``
    (read at request time), so setting them here bounds every subsequent
    query_criteria / get_product_list / download call.  Returns the conf object.
    """
    from astroquery.mast import conf
    conf.timeout = timeout_s
    conf.pagesize = pagesize
    return conf


def mast_query_errors():
    """Exception classes a MAST query can raise on timeout / network failure
    (lazy import, so the module stays stdlib-importable)."""
    import requests.exceptions
    from astroquery.exceptions import (RemoteServiceError,
                                       TimeoutError as AstroqueryTimeoutError)
    return (requests.exceptions.RequestException, RemoteServiceError,
            AstroqueryTimeoutError, ConnectionError, TimeoutError)


def _observations_table(program, instrument="NIRCam"):
    from astroquery.mast import Observations as MastObs
    configure_mast()
    return MastObs.query_criteria(obs_collection="JWST",
                                  proposal_id=str(int(program)),
                                  instrument_name=f"{instrument}*")


def list_observations(program, instrument="NIRCam"):
    """Return the distinct obs-numbers available on MAST for a program."""
    tbl = _observations_table(program, instrument)
    obsnums = sorted({str(o).split("-o")[1][:3]
                      for o in tbl["obs_id"] if "-o" in str(o)})
    return obsnums


def filtered_products(program, obs, product_type=("i2d",), instrument="NIRCam"):
    """The MAST product table for one observation, filtered to the requested
    suffix(es).  Returns None (with a note) when the observation is not on MAST."""
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
    return products[keep]


def product_list_size_bytes(program, obs, product_type=("i2d",),
                            instrument="NIRCam"):
    """Projected download size (bytes) of the filtered product list, from the MAST
    ``size`` column.  Returns None when the size cannot be determined (missing
    observation, missing/masked column) -- network failures propagate
    (``mast_query_errors()``) so the caller can warn per-observation."""
    products = filtered_products(program, obs, product_type=product_type,
                                 instrument=instrument)
    if products is None or "size" not in products.colnames:
        return None
    try:
        return int(sum(int(s) for s in products["size"]))
    except (TypeError, ValueError):     # masked / non-numeric size entries
        return None


def retrieve(program, obs, product_type=("i2d",), instrument="NIRCam",
             download_dir="./data", dry_run=False):
    """Download products for one observation.

    Filters the MAST product list to this obs-number and the requested suffix(es).
    Returns the manifest table (astroquery download result), or the filtered product
    table when ``dry_run``.
    """
    from astroquery.mast import Observations as MastObs

    products = filtered_products(program, obs, product_type=product_type,
                                 instrument=instrument)
    if products is None:
        return None
    oid = f"jw{int(program):05d}-o{obs}"
    suffixes = tuple(p.lower() for p in product_type)
    print(f"{oid}: {len(products)} products matching {suffixes}")
    if dry_run:
        return products
    configure_mast()     # bound the download requests too
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
