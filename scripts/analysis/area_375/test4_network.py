"""Test 4 (#375): re-run the stage-13 network (data_qa.neighbor_zeropoints, incl. tile+G with the
VIRAC2 Ks anchor) with per-star pixel-area corrections applied to the m8 magnitudes.

For each star of an m8 catalogue the frames of that tile (and module) that cover it are found from
the crf WCS; with p_i the per-frame term at the star's detector position,
    dm = -2.5 log10( mean_i 10**(0.4 * k_det(i) * p_i) )       (m_corr = m + dm)
Variants (env AREA_VARIANT):
  none     published magnitudes (reproduction)
  area1    p = a = 2.5 log10 AREA, k = 1                          (the #362 step-1 xAREA test)
  areak    p = a, k_det = test-2 PSF k_a per detector              (fitted partial correction)
  pT1      p = a + sT (pre-#1154 transposed grid-sum term), k = 1  (physical prediction, no fit)
  pTk      p = a + sT, k_det = test-2 PSF k_pT per detector
  pTvk     p = a + sT, k_det = test-3 (vs VIRAC2 Ks) PSF k_pT per detector
  pTvp     p = a + sT, one k per filter = test-3 (vs VIRAC2 Ks) pooled PSF k_pT
The correction also reaches the VIRAC2 anchor (it is measured through the same _load_nb_sample).
Usage: AREA_VARIANT=<v> python test4_network.py   -> net_<v>/ (neighbor_zeropoints outputs + summary.json)
"""
import glob
import json
import os
import re
import sys
import warnings

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
from data_qa import neighbor_zeropoints as nz  # noqa: E402
import common as C  # noqa: E402

B = '/orange/adamginsburg/jwst/gc-treasury/'
VARIANT = os.environ['AREA_VARIANT']
T2 = json.load(open(os.path.join(C.WORK, 'test2.json'))) if VARIANT in ('areak', 'pTk') else {}
T3 = json.load(open(os.path.join(C.WORK, 'test3.json'))) if VARIANT in ('pTvk', 'pTvp') else {}
MODDET = {'nrca': ('nrca1', 'nrca2', 'nrca3', 'nrca4', 'nrcalong'),
          'nrcb': ('nrcb1', 'nrcb2', 'nrcb3', 'nrcb4', 'nrcblong')}
CEN = json.load(open(os.path.join(C.HERE, 'centres.json')))
_FR, _AREA = {}, {}


def kval(filt, det):
    if VARIANT == 'areak':
        return T2[f'{filt.lower()}:psf']['k_a'][det][0]
    if VARIANT == 'pTk':
        return T2[f'{filt.lower()}:psf']['k_pT'][det][0]
    if VARIANT == 'pTvk':
        return T3[f'{filt.lower()}:virac:psf']['pT']['k'][det][0]
    if VARIANT == 'pTvp':
        return T3[f'{filt.lower()}:virac:psf']['pT']['k_pooled'][0]
    return 1.0


def frames(obs, filt):
    if (obs, filt) not in _FR:
        out = []
        for f in sorted(glob.glob(B + f'{filt}/pipeline/jw10678{obs}001_*_destreak_o{obs}_crf.fits')):
            det = re.search(r'_(nrc[ab](?:\d|long))_', f).group(1)
            if (filt, det) not in _AREA:
                _AREA[(filt, det)] = fits.getdata(f, 'AREA').astype(float)
            out.append((det, WCS(fits.getheader(f, 'SCI'))))
        _FR[(obs, filt)] = out
    return _FR[(obs, filt)]


def correction(path, filt, ra, dec):
    obs = re.search(r'_o(\d{3})\.fits$', path).group(1)
    mod = re.search(r'basic_(merged|nrca|nrcb)_', path).group(1)
    num = np.zeros(len(ra))
    cnt = np.zeros(len(ra))
    for det, w in frames(obs, filt):
        if mod != 'merged' and det not in MODDET[mod]:
            continue
        a = _AREA[(filt, det)]
        x, y = w.all_world2pix(ra, dec, 0, quiet=True)
        x, y = np.nan_to_num(x, nan=-1e9), np.nan_to_num(y, nan=-1e9)
        xi, yi = np.round(x).astype(int), np.round(y).astype(int)
        ins = (xi >= 0) & (yi >= 0) & (xi < a.shape[1]) & (yi < a.shape[0])
        av = np.zeros(len(ra))
        av[ins] = a[yi[ins], xi[ins]]
        ins &= av > 0.5
        p = np.zeros(len(ra))
        p[ins] = 2.5 * np.log10(av[ins])
        if VARIANT.startswith('pT') and ins.any():
            p[ins] += C.grid_sum_term(filt, det, x[ins], y[ins], 'transposed')
        fac = 10 ** (0.4 * kval(filt, det) * p)
        num[ins] += fac[ins]
        cnt[ins] += 1
    with np.errstate(invalid='ignore', divide='ignore'):
        dm = -2.5 * np.log10(num / cnt)
    return np.where(cnt > 0, dm, 0.0)


_orig = nz.D._load_nb_sample


def patched(path, filt, overlap_xy, frame):
    s = _orig(path, filt, overlap_xy, frame)
    if s is None or VARIANT == 'none' or len(s['ra']) == 0:
        return s
    dm = correction(path, filt.upper(), s['ra'], s['dec'])
    s = dict(s)
    s['mag'] = s['mag'] + dm
    s['flux'] = s['flux'] * 10 ** (-0.4 * dm)
    return s


nz.D._load_nb_sample = patched


def ew_steps(m):
    st = []
    for e in m['edges']:
        if e['kind'] != 'merged':
            continue
        a, b = e['a'], e['b']
        dx = (CEN[b][0] - CEN[a][0]) * np.cos(np.radians(CEN[a][1])) * 60
        dy = (CEN[b][1] - CEN[a][1]) * 60
        if abs(dx) > abs(dy):
            st.append(e['d'] * np.sign(dx))
    st = np.array(st)
    return float(np.median(st)), float(1.4826 * np.median(np.abs(st - np.median(st)))), len(st)


if __name__ == '__main__':
    out = os.path.join(C.WORK, f'net_{VARIANT}')
    os.makedirs(out, exist_ok=True)
    nproc = int(os.environ.get('SLURM_CPUS_PER_TASK', '4'))
    summ = []
    for filt in ('F212N', 'F480M'):
        res = nz.run_filter(nz.DEFAULT_CATDIR, filt, out, nproc=nproc, cache=True)
        s = res['summary']
        ew, ewmad, new = ew_steps(res['m'])
        summ.append(dict(filter=filt, variant=VARIANT,
                         tile_rms_largest=s.get('tile_zp_rms_largest'),
                         tileG_rms_largest=s.get('grad_tile_zp_rms_largest'),
                         Gx=s.get('grad_g', [np.nan, np.nan])[0], Gx_err=s.get('grad_g_err', [np.nan, np.nan])[0],
                         Gy=s.get('grad_g', [np.nan, np.nan])[1], Gy_err=s.get('grad_g_err', [np.nan, np.nan])[1],
                         ref_Gx=s.get('ref_g', [np.nan])[0], ref_Gx_err=s.get('ref_g_err', [np.nan])[0],
                         tileG_chi2_dof=s.get('grad_chi2_dof'), tileG_resid_mad=s.get('grad_resid_mad'),
                         ab_global=s.get('ab_global'), ab_global_err=s.get('ab_global_err'),
                         ab_tile_median=s.get('ab_tile_median'),
                         ew_step=ew, ew_step_mad=ewmad, n_ew=new,
                         pair_resid_mad_tile=s.get('pair_resid_mad_after')))
    json.dump(summ, open(os.path.join(out, 'summary.json'), 'w'), indent=1)
    print(json.dumps(summ, indent=1))
