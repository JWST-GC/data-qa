"""Per-frame star table for #375: PSF (STPSF m7 and hybrid-PSF m7) and aperture photometry of the
same stars on every 10678 crf frame of one tile, keyed to a tile star list so that each star's
frame-to-frame differences can be formed.

Inputs (read-only):
  per-frame PSF catalogues  gc-treasury/<FILT>/<filt>_<det>_o<obs>_visit*_vgroup*_exp*_resbgsub_m7_daophot_basic.fits
                            (and the _resbgsub_hybpsf_m7_ counterpart, matched by detector position)
  crf frames                the catalogue's meta FILENAME (SCI MJy/sr, DQ, AREA)
  tile star list            gc-treasury/<FILT>/pipeline/jw10678-o<obs>_t001_nircam_clear-<filt>-merged_cat.ecsv
                            (jwst source_catalog on the i2d; its aper50_vegamag is also kept)

Per frame: S/N > 30, qfit < 0.1, flags == 0, no neighbour brighter than m + 2.5 within 0.4" (SW) /
0.6" (LW) in that frame's catalogue; bulk offset to the star list from the median of 0.3" matches,
then a 0.06" match.  Aperture photometry on the crf SCI at (x_fit, y_fit): radii r1/r2 = 3.5/6 px
(SW) and 2.5/4 px (LW), background = sigma-clipped median in an annulus 9-14 px (SW) / 6-10 px (LW);
stars with a DQ DO_NOT_USE pixel inside r2 are dropped.  Fluxes are converted with the frame's
PIXAR_SR (one value per frame, the pipeline convention): m_AB = -2.5 log10(F_Jy) + 8.90.
AREA is stored at the PSF position (nearest pixel) and as the mean over the r1 aperture.

Usage: python build_frames.py <obs>   (writes frames/<filt>_o<obs>.npz for F212N and F480M)
"""
import glob
import json
import os
import sys
import warnings

import numpy as np
from astropy.io import fits
from astropy.stats import SigmaClip
from astropy.table import Table
from photutils.aperture import (ApertureStats, CircularAnnulus, CircularAperture,
                                aperture_photometry)
from scipy.spatial import cKDTree

warnings.filterwarnings('ignore')
B = '/orange/adamginsburg/jwst/gc-treasury/'
HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.environ.get('AREA375_WORKDIR', os.getcwd())
CEN = json.load(open(os.path.join(HERE, 'centres.json')))
ISO_AS = {'f212n': 0.4, 'f480m': 0.6}
APR = {'f212n': (3.5, 6.0, 9.0, 14.0), 'f480m': (2.5, 4.0, 6.0, 10.0)}
DO_NOT_USE = 1


def F(c):
    return np.asarray(np.ma.filled(np.ma.asarray(c, dtype=float), np.nan), float)


def one(obs, filt):
    ra0, de0 = CEN[obs][0], CEN[obs][1]
    cd = np.cos(np.radians(de0))
    a = Table.read(B + f'{filt.upper()}/pipeline/jw10678-o{obs}_t001_nircam_clear-{filt}-merged_cat.ecsv')
    okA = np.isfinite(F(a['aper50_vegamag'])) & ~np.asarray(a['is_extended'], bool)
    A = np.vstack([(a['sky_centroid'].ra.deg - ra0) * cd * 3600,
                   (a['sky_centroid'].dec.deg - de0) * 3600]).T
    ia = np.flatnonzero(okA)
    ta = cKDTree(A[ia])
    ap50 = F(a['aper50_vegamag'])
    r1, r2, rin, rout = APR[filt]
    pixsc = 0.0312 if filt == 'f212n' else 0.0629
    rows = []
    paths = sorted(glob.glob(B + f'{filt.upper()}/{filt}_nrc*_o{obs}_visit*_vgroup*_exp*_resbgsub_m7_daophot_basic.fits'))
    for f in paths:
        t = Table.read(f)
        det = t.meta['DETECTOR'].lower()
        exp = int(f.split('_exp')[1][:5])
        crf = t.meta['FILENAME'].replace('//', '/')
        with fits.open(crf) as h:
            sci = h['SCI'].data.astype(float)
            dq = h['DQ'].data
            amap = h['AREA'].data.astype(float)
            pixar = float(h['SCI'].header['PIXAR_SR'])
        x, y, fl = F(t['x_fit']), F(t['y_fit']), F(t['flux_fit'])
        snr = fl / F(t['flux_err'])
        m = -2.5 * np.log10(np.where(fl > 0, fl, np.nan))
        fin = np.isfinite(m) & np.isfinite(x) & np.isfinite(y)
        q = fin & (snr > 30) & (F(t['qfit']) < 0.1) & (np.asarray(t['flags']) == 0)
        idx = np.flatnonzero(q)
        if len(idx) == 0:
            continue
        tp = cKDTree(np.c_[x[fin], y[fin]])
        mm = m[fin]
        nb = tp.query_ball_point(np.c_[x[idx], y[idx]], r=ISO_AS[filt] / pixsc)
        iso = np.array([np.all(mm[n][mm[n] != m[i]] > m[i] + 2.5) for i, n in zip(idx, nb)])
        idx = idx[iso]
        if len(idx) < 20:
            continue
        sc = t['skycoord_centroid'][idx]
        P = np.vstack([(sc.ra.deg - ra0) * cd * 3600, (sc.dec.deg - de0) * 3600]).T
        d0, j0 = ta.query(P, distance_upper_bound=0.3)
        ok0 = np.isfinite(d0)
        if ok0.sum() < 10:
            continue
        off = np.median(P[ok0] - A[ia[j0[ok0]]], axis=0)
        d, j = ta.query(P - off, distance_upper_bound=0.06)
        ok = np.isfinite(d)
        idx, sid, P = idx[ok], ia[j[ok]], P[ok]
        xs, ys = x[idx], y[idx]
        # aperture photometry on the crf
        pos = np.c_[xs, ys]
        bad = (dq & DO_NOT_USE) > 0
        img = np.where(bad | ~np.isfinite(sci), 0.0, sci)
        ann = CircularAnnulus(pos, rin, rout)
        bst = ApertureStats(sci, ann, mask=bad | ~np.isfinite(sci), sigma_clip=SigmaClip(3.0, maxiters=5))
        bkg = np.asarray(bst.median, float)
        ap1, ap2 = CircularAperture(pos, r1), CircularAperture(pos, r2)
        tab = aperture_photometry(img, [ap1, ap2])
        nbad = np.asarray(aperture_photometry(bad.astype(float), [ap2])['aperture_sum_0'], float)
        amean = np.asarray(aperture_photometry(amap, [ap1])['aperture_sum_0'], float) / ap1.area
        f1 = np.asarray(tab['aperture_sum_0'], float) - bkg * ap1.area
        f2 = np.asarray(tab['aperture_sum_1'], float) - bkg * ap2.area
        conv = pixar * 1e6   # MJy/sr * sr -> Jy
        with np.errstate(invalid='ignore', divide='ignore'):
            map1 = -2.5 * np.log10(f1 * conv) + 8.90
            map2 = -2.5 * np.log10(f2 * conv) + 8.90
        map1[nbad > 0] = np.nan
        map2[nbad > 0] = np.nan
        mpsf = m[idx] - 2.5 * np.log10(conv) + 8.90
        # hybrid-PSF m7 fit of the same frame, matched in detector pixels
        mhyb = np.full(len(idx), np.nan)
        fh = f.replace('_resbgsub_m7_', '_resbgsub_hybpsf_m7_')
        if os.path.exists(fh):
            th = Table.read(fh)
            hx, hy, hf = F(th['x_fit']), F(th['y_fit']), F(th['flux_fit'])
            hok = np.isfinite(hx) & np.isfinite(hy) & (hf > 0) & (np.asarray(th['flags']) == 0)
            if hok.sum():
                th_t = cKDTree(np.c_[hx[hok], hy[hok]])
                dh, jh = th_t.query(pos, distance_upper_bound=0.3)
                okh = np.isfinite(dh)
                mhyb[okh] = -2.5 * np.log10(hf[hok][jh[okh]]) - 2.5 * np.log10(conv) + 8.90
        xi = np.clip(np.round(xs).astype(int), 0, amap.shape[1] - 1)
        yi = np.clip(np.round(ys).astype(int), 0, amap.shape[0] - 1)
        rows.append(dict(obs=np.full(len(idx), int(obs)), det=np.full(len(idx), det),
                         exp=np.full(len(idx), exp), sid=sid, x=xs, y=ys,
                         X=(A[sid, 0]) / 60, Y=(A[sid, 1]) / 60,
                         ra=a['sky_centroid'].ra.deg[sid], dec=a['sky_centroid'].dec.deg[sid],
                         mpsf=mpsf, mhyb=mhyb, map1=map1, map2=map2, snr=snr[idx],
                         qfit=F(t['qfit'])[idx], area=amap[yi, xi], area_ap=amean,
                         bkg=bkg, ap50_i2d=ap50[sid]))
    if not rows:
        return None
    out = {k: np.concatenate([r[k] for r in rows]) for k in rows[0]}
    print(obs, filt, len(paths), 'frames', len(out['x']), 'rows', flush=True)
    return out


if __name__ == '__main__':
    obs = sys.argv[1]
    os.makedirs(os.path.join(WORK, 'frames'), exist_ok=True)
    for filt in ('f212n', 'f480m'):
        o = one(obs, filt)
        if o is not None:
            np.savez(os.path.join(WORK, 'frames', f'{filt}_o{obs}.npz'), **o)
