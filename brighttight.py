import warnings; warnings.simplefilter("ignore")
import numpy as np, astropy.units as u
from astropy.table import Table
from astropy.coordinates import SkyCoord
from data_qa import diagnostics as D, astrometry_audit as aa
from data_qa.observations import registry
t=Table.read("/orange/adamginsburg/jwst/cloudef/mastDownload/JWST/jw02092-o005_t002_nircam_f150w2-f162m/jw02092-o005_t002_nircam_f150w2-f162m_cat.ecsv")
sc=SkyCoord(t["sky_centroid"]); flux=np.asarray(t["aper_total_flux"],float)
g=np.isfinite(flux)&(flux>0)
o=[x for x in registry(programs=["2092"]) if x.obs=="005" and x.instrument=="NIRCam"][0]
ep=aa.epoch_of(D._mast_i2d(o,"F162M")); ref,_=aa.load_reference(D._refcat_path(o),ep)
for nb in (1000,2000,5000):
    order=np.argsort(flux[g])[::-1][:nb]; b=sc[g][order]
    for w in (1.0,0.5,0.3):
        xc=aa.xcorr(b,ref,maxsep=w*u.arcsec)
        if xc: print("nb=%5d win=%.1f: |off|=%.0f pr=%.1f npairs=%d (dRA=%.0f dDec=%.0f)"%(nb,w,xc['off'],xc['peak_ratio'],xc['npairs'],xc['dra'],xc['ddec']))
