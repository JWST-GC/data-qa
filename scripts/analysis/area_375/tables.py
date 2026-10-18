"""Print the REPORT.md tables from the JSON outputs.  Usage: python tables.py > tables.md"""
import json
import os

import common as C

T1F = json.load(open('t1_flat_vs_area.json'))
T1P = json.load(open('t1_psfgrid_sum.json'))
T2 = json.load(open('test2.json'))
T3 = json.load(open('test3.json'))


def f(v, nd=2):
    return f'{v[0]:+.{nd}f} ± {v[1]:.{nd}f}'


print('### Test 1: large-scale flat and STPSF grid sum against AREA (log-log slope beta)\n')
print('| filter:detector | flat beta (log flat = beta log AREA) | grid sum beta, post-#1154 pairing | resid (mmag) | grid sum beta, pre-#1154 (transposed) | resid (mmag) |')
print('|---|---|---|---|---|---|')
for k in T1P:
    p = T1P[k]
    fl = T1F.get(k, {})
    print(f"| {k} | {fl.get('beta', float('nan')):+.2f} ± {fl.get('beta_err', float('nan')):.2f} | "
          f"{p['computed']['beta']:+.3f} ± {p['computed']['beta_err']:.3f} | {p['computed']['resid_rms_mmag']:.1f} | "
          f"{p['transposed']['beta']:+.2f} ± {p['transposed']['beta_err']:.2f} | {p['transposed']['resid_rms_mmag']:.1f} |")

print('\n### Test 2: internal k (frame minus tile consensus), pooled over detectors; bootstrap over tiles\n')
print('| filter | photometry | N rows | stars | sigma (mmag) | k on a | k on a+sT |')
print('|---|---|---|---|---|---|---|')
for k, r in T2.items():
    filt, phot = k.split(':')
    print(f"| {filt.upper()} | {C.PHOT_LABEL[phot]} | {r['N']} | {r['n_star']} | {r['sigma_mmag']:.1f} | "
          f"{f(r['k_a_pooled'])} | {f(r['k_pT_pooled'])} |")

print('\n### Test 2: per-detector k and cross-validated 8x8 binned residual RMS (mmag; mean of A->B and B->A)\n')
print('| filter | photometry | detector | k on a | k on a+sT | RMS before (null) | k=1 on a | CV k on a | CV k on a+sT |')
print('|---|---|---|---|---|---|---|---|---|')
for k, r in T2.items():
    filt, phot = k.split(':')
    if phot == 'ap2':
        continue
    for d in C.DETS[filt]:
        b = r['binned_rms_mmag']
        print(f"| {filt.upper()} | {phot} | {d} | {f(r['k_a'][d])} | {f(r['k_pT'][d])} | "
              f"{b['before'][d]['rms']:.2f} ({b['before'][d]['null']:.2f}) | {b['k1_area'][d]['rms']:.2f} | "
              f"{b['cv_a'][d]['rms']:.2f} | {b['cv_pT'][d]['rms']:.2f} |")

print('\n### Test 3: external k (m_frame AB minus Ks Vega, colour terms fitted), pooled; bootstrap over tiles\n')
print('| filter | reference | photometry | N | k on a | k on a+sT | cKs | c(H-Ks) | sigma (mmag) |')
print('|---|---|---|---|---|---|---|---|---|')
for k, r in T3.items():
    filt, ref, phot = k.split(':')
    print(f"| {filt.upper()} | {ref.upper()} | {C.PHOT_LABEL[phot]} | {r['a']['N']} | {f(r['a']['k_pooled'])} | "
          f"{f(r['pT']['k_pooled'])} | {r['a']['cKs']:+.3f} | {r['a']['cHK']:+.3f} | {r['a']['sigma_mmag']:.0f} |")

print('\n### Test 3: external per-detector k vs VIRAC2 (k on a / k on a+sT)\n')
for filt in ('f212n', 'f480m'):
    print('| photometry | ' + ' | '.join(C.DETS[filt]) + ' |')
    print('|---|' + '---|' * len(C.DETS[filt]))
    for phot in C.PHOT:
        r = T3[f'{filt}:virac:{phot}']
        print(f'| {filt.upper()} {phot} | ' + ' | '.join(f"{r['a']['k'][d][0]:+.2f}±{r['a']['k'][d][1]:.2f} / "
                                                   f"{r['pT']['k'][d][0]:+.2f}±{r['pT']['k'][d][1]:.2f}"
                                                   for d in C.DETS[filt]) + ' |')
    print()

print('### Test 4: stage-13 network on m8 (Vega) with per-star corrections\n')
print('| variant | filter | tile RMS (mag) | tile+G RMS (mag) | Gx (mag/arcmin) | VIRAC anchor Gx | E-W step (mag) | NRCA-NRCB (mag) | tile+G chi2/dof |')
print('|---|---|---|---|---|---|---|---|---|')
for v in ('none', 'area1', 'areak', 'pT1', 'pTk', 'pTvp', 'pTvk'):
    p = f'net_{v}/summary.json'
    if not os.path.exists(p):
        continue
    for s in json.load(open(p)):
        print(f"| {v} | {s['filter']} | {s['tile_rms_largest']:.3f} | {s['tileG_rms_largest']:.3f} | "
              f"{s['Gx']:+.4f} ± {s['Gx_err']:.4f} | {s['ref_Gx']:+.4f} | {s['ew_step']:+.3f} ± {s['ew_step_mad']:.3f} | "
              f"{s['ab_global']:+.3f} ± {s['ab_global_err']:.3f} | {s['tileG_chi2_dof']:.1f} |")
