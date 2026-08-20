import warnings; warnings.simplefilter("ignore")
import numpy as np
from astropy.table import Table
from astropy.coordinates import SkyCoord
t=Table.read("/orange/adamginsburg/jwst/cloudef/catalogs/basic_merged_indivexp_photometry_tables_merged_resbgsub_m7_qualcuts_oksep2221.fits")
print("N rows",len(t))
for col in [c for c in t.colnames if c.startswith('skycoord_') and 'filter' not in c]:
    sc=SkyCoord(t[col]); fin=np.isfinite(sc.ra.deg)
    scf=sc[fin]
    # fraction in the RA>266.63 strip
    strip=(scf.ra.deg>266.63).mean()
    print(f"  {col}: finite={fin.sum():7d}  RA[{scf.ra.deg.min():.3f},{scf.ra.deg.max():.3f}]  frac RA>266.63 = {strip:.2f}")
