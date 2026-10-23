"""`ref_tile` and `match` from ../area_375/t3_external_ref.py (VIRAC2 / GNS Ks per tile, and the tile
star-list match), copied here so t5 can import them without running test 3.  Needs ../area_375 on
sys.path first (t5_detector_offsets.py does this).
"""
import glob
import json
import os
import re

import numpy as np
from astropy.table import Table
from scipy.spatial import cKDTree

import common as C

CEN = json.load(open(os.path.join(C.HERE, 'centres.json')))
VC = '/orange/adamginsburg/jwst/gc-treasury/catalogs/virac2_ks_cache/'
# GNS Ks tiles cut around each pointing (built for #362); set AREA375_GNS_DIR.
GNS = os.environ.get('AREA375_GNS_DIR', os.path.join(C.WORK, 'gns_tiles')) + '/'
rng = np.random.default_rng(2)


def F(c):
    return np.asarray(np.ma.filled(np.ma.asarray(c, dtype=float), np.nan), float)


def ref_tile(obs, which):
    ra0, de0 = CEN[obs][0], CEN[obs][1]
    cd = np.cos(np.radians(de0))
    if which == 'virac':
        f = glob.glob(VC + f'virac2_ks_*_o{obs}.fits')[0]
        ep = float(re.search(r'epoch(\d+\.\d+)', f).group(1))
        v = Table.read(f)
        ks, h = F(v['Ksmag']), F(v['Hmag'])
        dec = F(v['DEJ2000'])
        ra = F(v['RAJ2000']) + np.nan_to_num(F(v['pmRA'])) * (ep - 2014.0) / 3.6e6 / np.cos(np.radians(dec))
        dec = dec + np.nan_to_num(F(v['pmDE'])) * (ep - 2014.0) / 3.6e6
        ok = np.isfinite(ks) & np.isfinite(ra)
        R = np.c_[(ra - ra0) * cd * 3600, (dec - de0) * 3600][ok]
        ks, h = ks[ok], h[ok]
        dd, _ = cKDTree(R).query(R, k=2)
        iso = dd[:, 1] > 1.0
        rng_ = (14.5, 16.5)
        rad = 0.15
    else:
        g = Table.read(GNS + f'{obs}.fits')
        if len(g) == 0:
            return None
        ks, h = F(g['Ksmag']), F(g['Hmag'])
        ok = np.isfinite(ks)
        R = np.c_[(F(g['RAJ2000']) - ra0) * cd * 3600, (F(g['DEJ2000']) - de0) * 3600][ok]
        ks, h = ks[ok], h[ok]
        nb = cKDTree(R).query_ball_point(R, r=0.5)
        iso = np.array([np.all(ks[[j for j in n if j != i]] > ks[i] + 3) for i, n in enumerate(nb)])
        rng_ = (14.5, 17.0)
        rad = 0.12
    sel = iso & (ks > rng_[0]) & (ks < rng_[1])
    return R, ks, h, sel, rad


def match(D, which):
    """Ks, H per row of D (NaN when unmatched)."""
    Ks = np.full(len(D['x']), np.nan)
    H = np.full(len(D['x']), np.nan)
    for obs in np.unique(D['obs']):
        o = f'{obs:03d}'
        r = ref_tile(o, which)
        if r is None:
            continue
        R, ks, h, sel, rad = r
        q = np.flatnonzero(D['obs'] == obs)
        sid, first = np.unique(D['sid'][q], return_index=True)
        P = np.c_[D['X'][q][first], D['Y'][q][first]] * 60
        tr = cKDTree(R)
        d0, j0 = tr.query(P, distance_upper_bound=0.3)
        ok = np.isfinite(d0)
        off = np.median(P[ok] - R[j0[ok]], axis=0)
        d, j = tr.query(P - off, distance_upper_bound=rad)
        ok = np.isfinite(d)
        u, c = np.unique(j[ok], return_counts=True)
        ok &= ~np.isin(j, u[c > 1])
        ok[ok] &= sel[j[ok]]
        lut_k = dict(zip(sid[ok], ks[j[ok]]))
        lut_h = dict(zip(sid[ok], h[j[ok]]))
        Ks[q] = np.array([lut_k.get(s, np.nan) for s in D['sid'][q]])
        H[q] = np.array([lut_h.get(s, np.nan) for s in D['sid'][q]])
    return Ks, H


