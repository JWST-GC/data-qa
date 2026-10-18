"""Figures for #375 from test2.json / test2_maps.npz / test3.json / net_*/summary.json.
Usage: python figs.py  -> fig1_maps_<filt>_<phot>.png, fig2_k_per_detector.png, fig3_network.png"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import common as C  # noqa: E402

T2 = json.load(open('test2.json'))
M = dict(np.load('test2_maps.npz'))
T3 = json.load(open('test3.json')) if os.path.exists('test3.json') else {}

# ---- fig 1: detector maps
for filt in ('f212n', 'f480m'):
    for phot in ('psf', 'hyb', 'ap1'):
        if f'{filt}:{phot}' not in T2:
            continue
        dets = C.DETS[filt]
        rows = [('a_raw', '2.5 log10 AREA\n(raw term, k=1)', None),
                ('pT_raw', 'a + sT\n(pre-#1154 PSF error)', None),
                ('obs', 'observed m_frame − consensus\n(within star & frame)', phot),
                ('pred_a1', 'prediction k_a=1\n(a, same differencing)', phot),
                ('pred_pT1', 'prediction k=1 on pT\n(same differencing)', phot),
                ('resid_pT', 'observed − k_pT·pT\n(fitted)', phot)]
        fig, ax = plt.subplots(len(rows), len(dets), figsize=(2.1 * len(dets) + 1.6, 2.1 * len(rows)),
                               squeeze=False)
        for i, (key, lab, ph) in enumerate(rows):
            vals = [M[f'{filt}:{d}:{key}'] if ph is None else M[f'{filt}:{ph}:{d}:{key}'] for d in dets]
            if key in ('a_raw', 'pT_raw'):
                vals = [v - np.nanmean(v) for v in vals]
            lim = np.nanpercentile(np.abs(np.concatenate([v.ravel() for v in vals])) * 1e3, 98)
            lim = max(lim, 1.0)
            for j, d in enumerate(dets):
                im = ax[i, j].imshow(vals[j] * 1e3, origin='lower', cmap='RdBu_r', vmin=-lim, vmax=lim,
                                     extent=[0, 2048, 0, 2048])
                ax[i, j].set_xticks([])
                ax[i, j].set_yticks([])
                if i == 0:
                    ax[i, j].set_title(d, fontsize=9)
                if j == 0:
                    ax[i, j].set_ylabel(lab, fontsize=7)
            cb = fig.colorbar(im, ax=ax[i, :].tolist(), shrink=0.85, pad=0.01)
            cb.set_label('mmag', fontsize=7)
            cb.ax.tick_params(labelsize=6)
        fig.suptitle(f'{filt.upper()} {C.PHOT_LABEL[phot]}: per-frame residual maps (16×16 cells, detector x→, y↑; '
                     f'magnitude differences, instrumental/AB from PIXAR_SR, constants cancel; '
                     f'raw maps mean-subtracted)', fontsize=9)
        fig.savefig(f'fig1_maps_{filt}_{phot}.png', dpi=110, bbox_inches='tight')
        plt.close(fig)

# ---- fig 2: k per detector (top: internal, test 2; bottom: external vs VIRAC2 / GNS Ks, test 3)
fig, axes = plt.subplots(2, 2, figsize=(14, 8.4), gridspec_kw=dict(width_ratios=[8, 3]))
series = [('psf', 'a', 'PSF STPSF m7 (pre-#1154): k on a', 'C3', 'o'),
          ('psf', 'pT', 'PSF STPSF m7 (pre-#1154): k on a+sT', 'C1', 's'),
          ('hyb', 'a', 'PSF hybrid m7 (post-#1154): k on a', 'C2', 'D'),
          ('ap1', 'a', 'aperture r1 on crf: k on a', 'C0', '^'),
          ('ap2', 'a', 'aperture r2 on crf: k on a', 'C9', 'v')]
for row in range(2):
    for ax, filt in zip(axes[row], ('f212n', 'f480m')):
        dets = C.DETS[filt]
        for n, (phot, model, lab, col, mk) in enumerate(series):
            if row == 0:
                r = T2.get(f'{filt}:{phot}')
                if not r:
                    continue
                v = np.array([r['k_a' if model == 'a' else 'k_pT'][d] for d in dets])
                ax.errorbar(np.arange(len(dets)) + (n - 2) * 0.12, v[:, 0], v[:, 1], fmt=mk, color=col,
                            ms=5, capsize=2, label=lab)
            else:
                for ref, mfc, off in (('virac', col, -0.06), ('gns', 'none', 0.06)):
                    r = T3.get(f'{filt}:{ref}:{phot}')
                    if not r:
                        continue
                    v = np.array([r[model]['k'][d] for d in dets])
                    ax.errorbar(np.arange(len(dets)) + (n - 2) * 0.15 + off, v[:, 0], v[:, 1], fmt=mk,
                                color=col, mfc=mfc, ms=5, capsize=2,
                                label=f'{lab} vs {ref.upper()}' if filt == 'f212n' else None)
        ax.axhline(1, color='0.4', lw=0.8, ls='--')
        ax.axhline(0, color='0.4', lw=0.8)
        ax.set_xticks(range(len(dets)))
        ax.set_xticklabels(dets, fontsize=8)
        ax.set_ylabel('k  (m = m_true + k * term)')
        ax.set_title(f'{filt.upper()}: ' + ('internal, frame minus tile consensus (instrumental/AB differences)'
                                            if row == 0 else
                                            'external, m_frame (AB) - Ks (Vega) with colour terms'), fontsize=9)
    axes[row, 0].legend(fontsize=6.5, ncol=2, loc='best')
axes[0, 0].set_ylim(-2.5, 3.5)
fig.suptitle('Pixel-area coefficient per detector (k=1: term fully missing, k=0: already applied); '
             'errors: bootstrap over tiles. Filled = VIRAC2, open = GNS', fontsize=10)
fig.tight_layout()
fig.savefig('fig2_k_per_detector.png', dpi=120)
plt.close(fig)

# ---- fig 3: network before / after
vars_ = [v for v in ('none', 'area1', 'areak', 'pT1', 'pTk', 'pTvp', 'pTvk') if os.path.exists(f'net_{v}/summary.json')]
if vars_:
    S = {v: {s['filter']: s for s in json.load(open(f'net_{v}/summary.json'))} for v in vars_}
    labs = {'none': 'published', 'area1': '×AREA (k=1)', 'areak': '×AREA^k_det (fitted)',
            'pT1': '-(a+sT), k=1', 'pTk': '-k_det(a+sT) (test-2 k)',
            'pTvp': '-k(a+sT) (VIRAC pooled k)', 'pTvk': '-k_det(a+sT) (VIRAC k_det)'}
    qty = [('tile_rms_largest', 'tile model: z RMS (mag)'), ('tileG_rms_largest', 'tile+G: z RMS (mag)'),
           ('Gx', 'tile+G: Gx (mag/arcmin)'), ('ref_Gx', 'VIRAC anchor Gx (mag/arcmin)'),
           ('ew_step', 'E–W overlap step (mag)'), ('ab_global', 'NRCA − NRCB (mag)')]
    fig, ax = plt.subplots(1, len(qty), figsize=(3.0 * len(qty), 3.6))
    for i, (k, lab) in enumerate(qty):
        for fi, (filt, col) in enumerate((('F212N', 'C0'), ('F480M', 'C3'))):
            y = [S[v][filt][k] for v in vars_]
            ax[i].plot(range(len(vars_)), y, 'o-', color=col, label=filt)
            if k == 'Gx':
                e = [S[v][filt]['Gx_err'] for v in vars_]
                ax[i].errorbar(range(len(vars_)), y, e, fmt='none', color=col, capsize=2)
        ax[i].axhline(0, color='0.5', lw=0.6)
        ax[i].set_xticks(range(len(vars_)))
        ax[i].set_xticklabels([labs[v] for v in vars_], rotation=60, ha='right', fontsize=7)
        ax[i].set_title(lab, fontsize=8.5)
    ax[0].legend(fontsize=7)
    fig.suptitle('Stage-13 network (neighbor_zeropoints) with per-star area corrections of the m8 magnitudes (Vega)',
                 fontsize=10)
    fig.tight_layout()
    fig.savefig('fig3_network.png', dpi=120)
    plt.close(fig)
