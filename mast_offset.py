import warnings; warnings.simplefilter("ignore")
import numpy as np, astropy.units as u
from astropy.coordinates import SkyCoord
from data_qa import diagnostics as D, astrometry_audit as aa
from data_qa.observations import registry
o=[x for x in registry(programs=["2092"]) if x.obs=="005" and x.instrument=="NIRCam"][0]
mast=D._mast_i2d(o,"F162M")
print("MAST i2d:",mast)
det=D._detect_on_mosaic(mast)   # (sc, mag, wcs, data)
if det is None: print("detect failed"); raise SystemExit
msc,mmag,_w,_d=det
print("MAST detections N=",len(msc)," RA[%.4f,%.4f] Dec[%.4f,%.4f]"%(msc.ra.deg.min(),msc.ra.deg.max(),msc.dec.deg.min(),msc.dec.deg.max()))
ep=aa.epoch_of(mast); ref_sc,_=aa.load_reference(D._refcat_path(o),ep)
print("VIRAC N=",len(ref_sc)," epoch",round(ep,3))
def bulk(sc,lbl):
    xc=aa.xcorr(sc,ref_sc)
    # clean isolated
    _i,ss,_=sc.match_to_catalog_sky(sc,nthneighbor=2); j=sc[ss>0.5*u.arcsec]
    i1,s1,_=j.match_to_catalog_sky(ref_sc); i2,s2,_=j.match_to_catalog_sky(ref_sc,nthneighbor=2)
    k=(s1<0.15*u.arcsec)&(s2>0.4*u.arcsec); jm,rm=j[k],ref_sc[i1[k]]
    dra=(jm.ra-rm.ra).to(u.mas).value*np.cos(jm.dec.rad); dde=(jm.dec-rm.dec).to(u.mas).value
    print(f"{lbl}: XCORR |off|={xc['off']:.0f} mas (dRA={xc['dra']:.0f},dDec={xc['ddec']:.0f}) pr={xc['peak_ratio']:.1f}")
    print(f"       clean-isolated |off|={np.hypot(np.median(dra),np.median(dde)):.0f} mas (dRA={np.median(dra):.0f},dDec={np.median(dde):.0f}) N={k.sum()}")
bulk(msc,"MAST-i2d detections")
# jicama for comparison
jsc,_=D._jwst_positions(o,"F162M")
bulk(jsc,"jicama catalogue")
