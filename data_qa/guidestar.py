"""Extract guide-star metadata from an observation's i2d header into a small committed
JSON that ``make_issues.render_body`` folds into the issue.

The JWST L3 header carries the guide star used for fine guidance:
    GDSTARID  GSC 2 id (e.g. S8DY632421)      GS_RA / GS_DEC   its ICRS position (deg)
    GS_MAG    guide-star magnitude            GSC_VER          catalog version (e.g. GSC30)
    GS_V3_PA  V3 position angle

The GSC-2 id resolves to a full catalog entry (incl. the ``posSource`` code = which survey
the position came from) via the STScI GSSS webform; that step is manual, so the issue links
straight to the pre-scoped webform + the posSource source-code table.

Runs on the cluster (needs the FITS header); writes ``data_qa/guidestar.json`` keyed by
obsid, which is committed and read in CI by ``make_issues``.

Usage:
    python -m data_qa.guidestar --program 5365 --obs 001 --field sgrb2
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

BASE = os.environ.get("QA_BASE", "/orange/adamginsburg/jwst")
STORE = os.path.join(os.path.dirname(__file__), "guidestar.json")
_KEYS = ("GDSTARID", "GS_RA", "GS_DEC", "GS_MAG", "GSC_VER", "GS_V3_PA")


def gsss_webform_url(gdstarid):
    """STScI GSSS VO webform, pre-scoped to GSC3.1 for the given HST/GSC id."""
    return (f"https://gsss.stsci.edu/webservices/vo/webform.aspx"
            f"?CAT=GSC31&HST_ID={gdstarid}")


POS_SOURCE_DOC = "https://outerspace.stsci.edu/display/MASTDATA/Source+Codes"


def extract(i2d_path):
    """Guide-star dict from an i2d primary header; keys absent from the header are omitted."""
    from astropy.io import fits
    hdr = fits.getheader(i2d_path, ext=0)
    out = {}
    for k in _KEYS:
        if k in hdr and hdr[k] not in ("", None):
            out[k.lower()] = hdr[k]
    return out


def _find_mosaic(program, obs, field):
    prog = f"jw{int(program):05d}"
    pats = [f"{BASE}/{field}/*/pipeline/{prog}-o{obs}*_t001_*i2d.fits",
            f"{BASE}/{field}/images-merged/{prog}-o{obs}*_t001_*i2d.fits"]
    for pat in pats:
        hits = sorted(glob.glob(pat))
        # prefer a plain science mosaic over residual/model sidecars
        clean = [h for h in hits if not any(s in h.lower()
                 for s in ("_residual", "_model", "resbgsub", "_bg_i2d"))]
        if clean:
            return clean[0]
    return None


def load_store():
    if os.path.exists(STORE):
        try:
            with open(STORE) as fh:
                return json.load(fh)
        except (OSError, ValueError):
            return {}
    return {}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--program", required=True)
    ap.add_argument("--obs", required=True)
    ap.add_argument("--field", required=True)
    args = ap.parse_args(argv)
    obsid = f"jw{int(args.program):05d}-o{args.obs}"
    img = _find_mosaic(args.program, args.obs, args.field)
    if not img:
        print(f"no i2d for {obsid} in {args.field}", file=sys.stderr)
        return 1
    gs = extract(img)
    if not gs:
        print(f"{obsid}: no guide-star keywords in {os.path.basename(img)}", file=sys.stderr)
        return 1
    store = load_store()
    store[obsid] = gs
    with open(STORE, "w") as fh:
        json.dump(store, fh, indent=2, sort_keys=True)
    print(f"{obsid}: {gs.get('gdstarid')} @ ({gs.get('gs_ra')}, {gs.get('gs_dec')}) "
          f"mag {gs.get('gs_mag')} [{gs.get('gsc_ver')}] -> {STORE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
