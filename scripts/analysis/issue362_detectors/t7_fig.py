"""#362 figure: the per-detector part of fig C against the merge's pixel-area term.

fig C        hybrid m7 merged mag_vega - VIRAC Ks (both Vega), per detector, centred (figC_detectors.json)
merge term   merged mag_vega - per-frame hybrid PSF mag (AB, frame PIXAR_SR), centred (t7.json)
prediction   +2.5 log10(PIXAR_SR of the detector), centred: what converting every frame with one
             pixel area (the merged catalogue's 212PIXS**2) instead of its own PIXAR_SR adds
residual     fig C - merge term
Usage: python t7_fig.py <figC_detectors.json> <t7.json> <out.png>
"""
import glob
import json
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from astropy.io import fits  # noqa: E402

FC, T7, OUT = sys.argv[1:4]
fc = json.load(open(FC))
t7 = json.load(open(T7))
B = '/orange/adamginsburg/jwst/gc-treasury/'
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw=dict(width_ratios=[4, 1.3]),
                         constrained_layout=True)
summary = {}
for ax, filt, dets in ((axes[0], 'f212n', [f'nrc{m}{n}' for m in 'ab' for n in range(1, 5)]),
                       (axes[1], 'f480m', ['nrcalong', 'nrcblong'])):
    pix = {}
    for d in dets:
        fs = sorted(glob.glob(B + f'{filt.upper()}/pipeline/jw10678*_{d}_destreak_o*_crf.fits'))[::25]
        pix[d] = float(np.mean([fits.getheader(f, 'SCI')['PIXAR_SR'] for f in fs]))
    lp = np.array([2.5 * np.log10(pix[d]) for d in dets])
    pred = (lp - lp.mean()) * 1e3
    c = np.array([fc[f'{filt}:hyb:{d}'][0] for d in dets])
    figc = (c - c.mean()) * 1e3
    merge = np.array([t7[filt]['det'][d][0] for d in dets]) * 1e3
    resid = figc - merge
    x = np.arange(len(dets))
    w = 0.2
    ax.bar(x - 1.5 * w, figc, w, color='0.55', label='fig C: merged (Vega) − VIRAC Ks (Vega)')
    ax.bar(x - 0.5 * w, merge, w, color='tab:red', label='merged (Vega) − per-frame PSF (AB), measured')
    ax.bar(x + 0.5 * w, pred, w, color='tab:orange', label='+2.5 log10 PIXAR_SR(detector), predicted')
    ax.bar(x + 1.5 * w, resid, w, color='tab:blue', label='fig C − measured merge term')
    ax.axhline(0, color='k', lw=0.6)
    ax.set_xticks(x, [d[3:].upper() for d in dets])
    ax.set_ylabel('per-detector offset, centred (mmag)')
    ax.set_title(f'{filt.upper()}', fontsize=10)
    summary[filt] = dict(pixar_sr=pix, figC=dict(zip(dets, figc.round(1).tolist())),
                         merge=dict(zip(dets, merge.round(1).tolist())),
                         pred=dict(zip(dets, pred.round(1).tolist())),
                         resid=dict(zip(dets, resid.round(1).tolist())),
                         rms=dict(figC=float(np.std(figc)), resid=float(np.std(resid)),
                                  merge_minus_pred=float(np.std(merge - pred))))
h, lab = axes[0].get_legend_handles_labels()
fig.legend(h, lab, loc='outside lower center', ncol=4, fontsize=8)
axes[1].set_ylim(axes[0].get_ylim())
fig.suptitle('10678 hybrid-PSF m7: the per-detector offsets enter at the merge, where every frame is '
             'converted to Jy with one pixel area\n(positive = fainter; fig C over 68 tiles, merge term '
             f"over {len(t7['f212n']['tiles'])} tiles; magnitude systems as labelled)", fontsize=10)
fig.savefig(OUT, dpi=110)
json.dump(summary, open(OUT.replace('.png', '.json'), 'w'), indent=1)
for f, s in summary.items():
    print(f, {k: round(v, 1) for k, v in s['rms'].items()})
    print('  PIXAR_SR', {d: f'{v:.5e}' for d, v in s['pixar_sr'].items()})
    print('  resid', s['resid'])
