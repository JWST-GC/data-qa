import warnings; warnings.simplefilter("ignore")
import numpy as np, astropy.units as u
from astropy.table import Table
from astropy.coordinates import SkyCoord
from data_qa import diagnostics as D, astrometry_audit as aa
from data_qa.observations import registry
p="/orange/adamginsburg/jwst/cloudef/mastDownload/JWST/jw02092-o005_t002_nircam_f150w2-f162m/jw02092-o005_t002_nircam_f150w2-f162m_cat.ecsv"
sc=SkyCoord(Table.read(p)["sky_centroid"])
o=[x for x in registry(programs=["2092"]) if x.obs=="005" and x.instrument=="NIRCam"][0]
ep=aa.epoch_of(D._mast_i2d(o,"F162M")); ref,_=aa.load_reference(D._refcat_path(o),ep)
print("XMAXSEP=%s XBIN=%s"%(aa.XMAXSEP,aa.XBIN))
for ms in (0.5,0.35,0.25):
    xc=aa.xcorr(sc,ref,maxsep=ms*u.arcsec)
    print("  maxsep=%.2f: |off|=%.0f pr=%.1f npairs=%d (dRA=%.0f dDec=%.0f)"%(ms,xc['off'],xc['peak_ratio'],xc['npairs'],xc['dra'],xc['ddec']))
