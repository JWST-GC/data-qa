"""Test 1 (provenance): does the CRDS flat used for each 10678 crf carry the pixel-area pattern?

For each detector (o066 exposure 1 headers give R_FLAT / R_AREA), block-median the flat and the
AREA reference file in 64x64 px blocks and fit  log10(flat) = beta * log10(AREA) + c  (plus a
variant with an extra plane in x, y).  beta ~ 1: the flat divides out the area pattern, so SCI is
true surface brightness and point-source photometry needs x AREA.  beta ~ 0: the flat has no area
term, SCI counts a point source correctly with a constant PIXAR_SR.
Usage: python t1_flat_vs_area.py  -> t1_flat_vs_area.json
"""
import glob, json
import numpy as np
from astropy.io import fits

CRDS = '/orange/adamginsburg/jwst/crds/references/jwst/nircam/'
B = '/orange/adamginsburg/jwst/gc-treasury/'
NB = 64


def blockmed(a, good):
    a = np.where(good, a, np.nan)
    n = a.shape[0] // NB
    return np.nanmedian(a[:n * NB, :n * NB].reshape(n, NB, n, NB), axis=(1, 3))


out = {}
for filt in ('F212N', 'F480M'):
    for f in sorted(glob.glob(B + f'{filt}/pipeline/jw10678066001_02101_00001_nrc*_destreak_o066_crf.fits')):
        h = fits.getheader(f)
        det = h['DETECTOR'].lower()
        flat = fits.open(CRDS + h['R_FLAT'].replace('crds://', ''))
        area = fits.getdata(CRDS + h['R_AREA'].replace('crds://', ''), 'SCI').astype(float)
        fl = flat['SCI'].data.astype(float)
        dq = flat['DQ'].data
        good = (dq == 0) & np.isfinite(fl) & (fl > 0.5) & (fl < 1.5) & (area > 0.5)
        good[:4] = good[-4:] = False
        good[:, :4] = good[:, -4:] = False
        F = blockmed(fl, good)
        A = blockmed(area, good)
        n = F.shape[0]
        yy, xx = np.mgrid[0:n, 0:n] * NB + NB / 2
        ok = np.isfinite(F) & np.isfinite(A)
        lf, la = np.log10(F[ok]), np.log10(A[ok])
        X1 = np.c_[np.ones(ok.sum()), la]
        b1, *_ = np.linalg.lstsq(X1, lf, rcond=None)
        r1 = lf - X1 @ b1
        e1 = np.sqrt(np.diag(np.linalg.inv(X1.T @ X1)) * r1.var())
        X2 = np.c_[X1, (xx[ok] - 1024) / 1024, (yy[ok] - 1024) / 1024]
        b2, *_ = np.linalg.lstsq(X2, lf, rcond=None)
        r2 = lf - X2 @ b2
        e2 = np.sqrt(np.diag(np.linalg.inv(X2.T @ X2)) * r2.var())
        out[f'{filt}:{det}'] = dict(flat=h['R_FLAT'], area=h['R_AREA'],
                                    beta=float(b1[1]), beta_err=float(e1[1]),
                                    beta_plane=float(b2[1]), beta_plane_err=float(e2[1]),
                                    corr=float(np.corrcoef(lf, la)[0, 1]),
                                    flat_ls_rms_mmag=float(2.5 * lf.std() * 1000),
                                    area_rms_mmag=float(2.5 * la.std() * 1000))
        v = out[f'{filt}:{det}']
        print(f"{filt} {det:9s} {h['R_FLAT'][7:]:28s} beta={v['beta']:+.3f}±{v['beta_err']:.3f} "
              f"beta(+plane)={v['beta_plane']:+.3f}±{v['beta_plane_err']:.3f} r={v['corr']:+.2f} "
              f"flat-LS rms={v['flat_ls_rms_mmag']:.1f} mmag  area rms={v['area_rms_mmag']:.1f} mmag")
json.dump(out, open('t1_flat_vs_area.json', 'w'), indent=1)
