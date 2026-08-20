import warnings; warnings.simplefilter("ignore")
import numpy as np, astropy.units as u
from astropy.coordinates import SkyCoord
from data_qa import diagnostics as D, astrometry_audit as aa
from data_qa.observations import registry
o=[x for x in registry(programs=["2092"]) if x.obs=="005" and x.instrument=="NIRCam"][0]
ep=aa.epoch_of(D._mast_i2d(o,"F162M")); ref_sc,_=aa.load_reference(D._refcat_path(o),ep)
msc,mmag,_,_=D._detect_on_mosaic(D._mast_i2d(o,"F162M"))
# VIRAC in MAST footprint
box=(ref_sc.ra.deg>msc.ra.deg.min())&(ref_sc.ra.deg<msc.ra.deg.max())&(ref_sc.dec.deg>msc.dec.deg.min())&(ref_sc.dec.deg<msc.dec.deg.max())
print("MAST det N=%d  VIRAC in-footprint N=%d"%(len(msc),box.sum()))
# nearest-match all (no isolation), median offset within various tols
idx,sep,_=msc.match_to_catalog_sky(ref_sc)
for tol in (0.1,0.2,0.3):
    k=sep<tol*u.arcsec; jm,rm=msc[k],ref_sc[idx[k]]
    dra=(jm.ra-rm.ra).to(u.mas).value*np.cos(jm.dec.rad); dde=(jm.dec-rm.dec).to(u.mas).value
    print("  NN<%.1f\": N=%d median dRA=%.0f dDec=%.0f |off|=%.0f mas"%(tol,k.sum(),np.median(dra),np.median(dde),np.hypot(np.median(dra),np.median(dde))))
# xcorr on bright subset
for frac,lbl in [(1.0,"all"),(0.3,"bright30%"),(0.1,"bright10%")]:
    n=int(len(msc)*frac); order=np.argsort(mmag)[:n]  # mmag = -2.5log10(flux): brighter=more negative
    xc=aa.xcorr(msc[order],ref_sc)
    print("  XCORR %s (N=%d): |off|=%.0f pr=%.1f npairs=%d"%(lbl,n,xc['off'],xc['peak_ratio'],xc['npairs']))
# VIRAC->MAST (dense ref side)
idx2,sep2,_=ref_sc[box].match_to_catalog_sky(msc)
k2=sep2<0.2*u.arcsec; rm2,jm2=ref_sc[box][k2],msc[idx2[k2]]
dra2=(rm2.ra-jm2.ra).to(u.mas).value*np.cos(rm2.dec.rad)*-1; dde2=(rm2.dec-jm2.dec).to(u.mas).value*-1
print("  VIRAC->MAST NN<0.2: N=%d median |off|=%.0f mas"%(k2.sum(),np.hypot(np.median(dra2),np.median(dde2))))
