"""Catalogue-level impact of each correction: how far each tile-consensus star would move.

A tile-consensus position averages a star's detections over dithers and detectors.  A correction
c(x, y) applied to every detection moves the consensus position by the mean of c over that star's
detections (on the sky).  After removing the per-tile mean shift (absorbed by the tile tie),
this script reports the RMS and 99th percentile of that movement over stars, per filter, for:
  gdc_v1    : c = delta_STDGDC v1 (affine-anchored; F212N)
  dqmap     : c = -M, M = all-tile data-qa-estimator map (empirical/maps_<filt>.npz 'dqfull')
  dqmap_dec : c = -M / s, s = recovery slope of the data-qa estimator for the map's own shape
              (empirical_<filt>.json recovery.dqmap.dq_shift, per detector): a first-order
              de-attenuated version of the measured map
It also reports the single-detection (per-frame) amplitude of each c for comparison.

Usage:
  python catalog_impact.py --filters F212N F480M --matchdir <distcmp>/matches \
      --empdir <distcmp>/empirical --out <distcmp>/empirical/catalog_impact.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fields as F                                       # noqa: E402
from empirical_test import Data, gdc_correction, det_list  # noqa: E402
from extract_matches import DETS                         # noqa: E402


def map_corr(D, maps, filt, scale=None):
    c = np.zeros((D.N, 2))
    for d in det_list(filt):
        k = f"dqfull_{d}_dx"
        if k not in maps:
            continue
        sel = D.det == DETS.index(d.lower())
        fx, fy = F.BinnedMapField(maps[k], maps[f"dqfull_{d}_dy"])(D.x[sel], D.y[sel])
        s = 1.0 if scale is None else scale[d]
        c[sel, 0] = -fx / s; c[sel, 1] = -fy / s
    return c


def impact(D, c):
    cs = D.to_sky(c) * 1000.0                              # mas on the sky
    ng = D.ng.astype(float)
    m = np.column_stack([np.bincount(D.g, cs[:, k], len(ng)) / ng for k in (0, 1)])
    # per-tile mean removed (tile tie absorbs a bulk shift)
    gobs = np.zeros(len(ng), int); gobs[D.g] = D.obs
    for o in np.unique(gobs):
        s = gobs == o
        m[s] -= m[s].mean(axis=0)
    multi = ng >= 2
    a = np.hypot(m[multi, 0], m[multi, 1])
    det_amp = np.hypot(c[:, 0], c[:, 1])
    return dict(n_stars=int(multi.sum()), star_shift_rms_mas=float(np.sqrt(np.mean(a ** 2))),
                star_shift_p99_mas=float(np.percentile(a, 99)),
                star_shift_max_mas=float(a.max()),
                detection_rms_mas=float(np.sqrt(np.mean(det_amp ** 2))),
                detection_p99_mas=float(np.percentile(det_amp, 99)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--filters", nargs="+", default=["F212N", "F480M"])
    ap.add_argument("--matchdir", required=True)
    ap.add_argument("--empdir", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = {}
    for filt in a.filters:
        D = Data(os.path.join(a.matchdir, f"matches_{filt.lower()}.npz"))
        maps = dict(np.load(os.path.join(a.empdir, f"maps_{filt.lower()}.npz")))
        emp = json.load(open(os.path.join(a.empdir, f"empirical_{filt.lower()}.json")))
        slope = {d: v["slope"] for d, v in emp["recovery"]["dqmap"]["dq_shift"].items()}
        res = {}
        if filt.upper() == "F212N":
            res["gdc_v1"] = impact(D, gdc_correction(D, filt, "v1"))
        res["dqmap"] = impact(D, map_corr(D, maps, filt))
        res["dqmap_dec"] = impact(D, map_corr(D, maps, filt, slope))
        res["dqmap_recovery_slope"] = slope
        out[filt] = res
        print(filt, json.dumps(res, indent=1), flush=True)
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
