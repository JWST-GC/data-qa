import warnings, os, time; warnings.simplefilter("ignore")
from astropy.table import Table
from astropy.coordinates import SkyCoord
import numpy as np, astropy.units as u
base="/orange/adamginsburg/jwst/cloudef"
def dt(p): return time.strftime('%Y-%m-%d %H:%M', time.localtime(os.path.getmtime(p)))
m7=f"{base}/catalogs/basic_merged_indivexp_photometry_tables_merged_resbgsub_m7_qualcuts_oksep2221.fits"
m2=f"{base}/catalogs/f162m_merged_indivexp_merged_m2_dao_basic.fits"
off=f"{base}/offsets/Offsets_JWST_Brick2092_VIRAC2locked.csv"
mos=f"{base}/F162M/pipeline/jw02092-o005_t001_nircam_clear-f162m-merged_i2d.fits"
for lbl,p in [("m7 merged(QA reads)",m7),("m2 f162m",m2),("offsets VIRAC2locked",off),("F162M merged mosaic",mos)]:
    print(f"{lbl:24s} {dt(p)}  {p.split('/')[-1]}")
# compare m7 skycoord_ref vs m2 f162m positions
t7=Table.read(m7); t2=Table.read(m2)
s7=SkyCoord(t7['skycoord_ref']); s2=SkyCoord(t2['skycoord'])
idx,sep,_=s7.match_to_catalog_sky(s2); k=sep<0.15*u.arcsec
a,b=s7[k],s2[idx[k]]
dra=(a.ra-b.ra).to(u.mas).value*np.cos(a.dec.rad); dde=(a.dec-b.dec).to(u.mas).value
print(f"\nm7.skycoord_ref - m2.f162m : N={k.sum()} median dRA={np.median(dra):.0f} dDec={np.median(dde):.0f} mas")
print("ref filtername in m7:", set(map(str,t7['skycoord_ref_filtername'][:2000]))| set(map(str,t7['skycoord_ref_filtername'][-2000:])) if 'skycoord_ref_filtername' in t7.colnames else "n/a")
