import warnings; warnings.simplefilter("ignore")
import numpy as np, astropy.units as u
from astropy.table import Table
from astropy.coordinates import SkyCoord
from data_qa import diagnostics as D, astrometry_audit as aa
from data_qa.observations import registry
t=Table.read("/orange/adamginsburg/jwst/cloudef/mastDownload/JWST/jw02092-o005_t002_nircam_f150w2-f162m/jw02092-o005_t002_nircam_f150w2-f162m_cat.ecsv")
sc=SkyCoord(t["sky_centroid"]); flux=np.asarray(t["aper_total_flux"],float)
o=[x for x in registry(programs=["2092"]) if x.obs=="005" and x.instrument=="NIRCam"][0]
ep=aa.epoch_of(D._mast_i2d(o,"F162M")); ref,_=aa.load_reference(D._refcat_path(o),ep)
g=np.isfinite(flux)&(flux>0)
for n in (2000,5000,10000):
    order=np.argsort(flux[g])[::-1][:n]; b=sc[g][order]
    xc=aa.xcorr(b,ref)  # default maxsep 2.5
    # matched-median of bright, unique matches
    i1,s1,_=b.match_to_catalog_sky(ref); i2,s2,_=b.match_to_catalog_sky(ref,nthneighbor=2)
    k=(s1<0.3*u.arcsec)&(s2>0.5*u.arcsec)  # unique within 0.3
    jm,rm=b[k],ref[i1[k]]; dra=(jm.ra-rm.ra).to(u.mas).value*np.cos(jm.dec.rad); dde=(jm.dec-rm.dec).to(u.mas).value
    print("top%5d: XCORR|off|=%.0f pr=%.1f | unique-match N=%d median|off|=%.0f (dRA=%.0f dDec=%.0f)"%(n,xc['off'],xc['peak_ratio'],k.sum(),np.hypot(np.median(dra),np.median(dde)),np.median(dra),np.median(dde)))
