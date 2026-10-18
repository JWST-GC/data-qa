"""Test 1 (provenance): does the STPSF grid the 10678 m7 fits use carry 1/AREA in its normalisation?

For each detector grid (gc-treasury/psfs/nircam_<det>_<filt>_fovp101_samp2_npsf16.fits, 4x4 nodes,
NORMALIZ='first', DISTORT=True) compute the native-pixel sum of every plane, s = 2.5 log10(sum),
and compare with a = 2.5 log10 AREA at the node position (AREA block-mean over +-16 px).
Two plane->position assignments:
  'computed' : planes sorted by (y, x) (STPSF >= 1.3 writer; jwst-gc-pipeline psf_grid_io, #1154)
  'transposed': plane i <-> DET_YX{i} (stpsf.utils.to_griddedpsfmodel, the pre-#1154 loader)
Fit s = beta * a + c per detector.  beta = -1: the grid sum is exactly 1/AREA, so a PSF fit to the
MJy/sr crf returns true flux with a constant PIXAR_SR (k_PSF = 0 when paired correctly).
Writes t1_psfgrid_sum.json and t1_psfgrid_nodes.npz (per-node x, y, s, both assignments) for test 2.
Usage: python t1_psfgrid_sum.py
"""
import glob, json, os, sys
import numpy as np
from astropy.io import fits
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', '..'))
from data_qa.paths import pipe_root  # noqa: E402
sys.path.insert(0, pipe_root())
from jwst_gc_pipeline.photometry.psf_grid_io import det_yx_positions, grid_plane_positions  # noqa

P = '/orange/adamginsburg/jwst/gc-treasury/psfs/'
B = '/orange/adamginsburg/jwst/gc-treasury/'
out, nodes = {}, {}
for filt, dets in (('F212N', ['nrca1', 'nrca2', 'nrca3', 'nrca4', 'nrcb1', 'nrcb2', 'nrcb3', 'nrcb4']),
                   ('F480M', ['nrcalong', 'nrcblong'])):
    for det in dets:
        g = P + f"nircam_{det.replace('long', '5')}_{filt.lower()}_fovp101_samp2_npsf16.fits"
        with fits.open(g) as h:
            data = h[0].data.astype(float)
            hd = h[0].header
        osz = hd['OVERSAMP']
        s = 2.5 * np.log10(data.sum(axis=(1, 2)) / osz ** 2)
        crf = sorted(glob.glob(B + f'{filt}/pipeline/jw10678066001_02101_00001_{det}_destreak_o066_crf.fits'))[0]
        area = fits.getdata(crf, 'AREA').astype(float)
        res = {}
        for lab, xy in (('computed', grid_plane_positions(hd, len(s))), ('transposed', det_yx_positions(hd))):
            a = np.array([2.5 * np.log10(np.nanmean(area[max(int(y) - 16, 0):int(y) + 17,
                                                          max(int(x) - 16, 0):int(x) + 17])) for x, y in xy])
            A = np.c_[np.ones(len(a)), a]
            b, *_ = np.linalg.lstsq(A, s, rcond=None)
            r = s - A @ b
            e = np.sqrt(np.diag(np.linalg.inv(A.T @ A)) * r.var(ddof=2))
            res[lab] = dict(beta=float(b[1]), beta_err=float(e[1]), resid_rms_mmag=float(r.std() * 1e3),
                            corr=float(np.corrcoef(a, s)[0, 1]))
            nodes[f'{filt}:{det}:{lab}'] = np.c_[xy, s]
        out[f'{filt}:{det}'] = dict(grid=g.split('/')[-1], version=hd.get('VERSION'), normaliz=hd.get('NORMALIZ'),
                                    distort=hd.get('DISTORT'), sum_spread_mmag=float(np.ptp(s) * 1e3),
                                    area_spread_nodes_mmag=float(np.ptp(a) * 1e3), **res)
        c, t = res['computed'], res['transposed']
        print(f"{filt} {det:9s} sum spread {np.ptp(s)*1e3:5.1f} mmag | computed: beta={c['beta']:+.3f}±{c['beta_err']:.3f} "
              f"r={c['corr']:+.3f} rms={c['resid_rms_mmag']:.1f} | transposed: beta={t['beta']:+.3f}±{t['beta_err']:.3f} "
              f"r={t['corr']:+.3f} rms={t['resid_rms_mmag']:.1f} mmag")
json.dump(out, open('t1_psfgrid_sum.json', 'w'), indent=1)
np.savez('t1_psfgrid_nodes.npz', **nodes)
