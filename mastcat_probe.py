import warnings; warnings.simplefilter("ignore")
import numpy as np, astropy.units as u
from astropy.table import Table
from astropy.coordinates import SkyCoord
from data_qa import diagnostics as D, astrometry_audit as aa
from data_qa.observations import registry
p="/orange/adamginsburg/jwst/cloudef/mastDownload/JWST/jw02092-o005_t002_nircam_f150w2-f162m/jw02092-o005_t002_nircam_f150w2-f162m_cat.ecsv"
t=Table.read(p)
print("N=",len(t))
print("cols:",[c for c in t.colnames if 'ra' in c.lower() or 'dec' in c.lower() or 'centroid' in c.lower() or 'mag' in c.lower() or 'flux' in c.lower()][:14])
# find sky position
sc=None
for c in ("sky_centroid",):
    if c in t.colnames: sc=SkyCoord(t[c]); break
if sc is None and "sky_centroid.ra" in t.colnames:
    sc=SkyCoord(t["sky_centroid.ra"]*u.deg, t["sky_centroid.dec"]*u.deg)
print("sky RA[%.4f,%.4f] Dec[%.4f,%.4f]"%(sc.ra.deg.min(),sc.ra.deg.max(),sc.dec.deg.min(),sc.dec.deg.max()))
print("frac RA>266.63:", (sc.ra.deg>266.63).mean())
o=[x for x in registry(programs=["2092"]) if x.obs=="005" and x.instrument=="NIRCam"][0]
ep=aa.epoch_of(D._mast_i2d(o,"F162M")); ref,_=aa.load_reference(D._refcat_path(o),ep)
xc=aa.xcorr(sc,ref); print("XCORR |off|=%.0f pr=%.1f npairs=%d dRA=%.0f dDec=%.0f"%(xc['off'],xc['peak_ratio'],xc['npairs'],xc['dra'],xc['ddec']))
idx,sep,_=sc.match_to_catalog_sky(ref); 
for tol in (0.15,0.3):
    k=sep<tol*u.arcsec; jm,rm=sc[k],ref[idx[k]]
    dra=(jm.ra-rm.ra).to(u.mas).value*np.cos(jm.dec.rad); dde=(jm.dec-rm.dec).to(u.mas).value
    print("  NN<%.2f: N=%d median |off|=%.0f (dRA=%.0f dDec=%.0f)"%(tol,k.sum(),np.hypot(np.median(dra),np.median(dde)),np.median(dra),np.median(dde)))
