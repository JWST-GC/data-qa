import warnings; warnings.simplefilter("ignore")
import numpy as np, astropy.units as u
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
from data_qa import diagnostics as D, astrometry_audit as aa
from data_qa.observations import registry
o=[x for x in registry(programs=["2092"]) if x.obs=="005" and x.instrument=="NIRCam"][0]
ep=aa.epoch_of(D._mast_i2d(o,"F162M")); ref_sc,_=aa.load_reference(D._refcat_path(o),ep)
def offset_cloud(sc):
    _i,ss,_=sc.match_to_catalog_sky(sc,nthneighbor=2); j=sc[ss>0.5*u.arcsec]
    i1,s1,_=j.match_to_catalog_sky(ref_sc); i2,s2,_=j.match_to_catalog_sky(ref_sc,nthneighbor=2)
    k=(s1<0.15*u.arcsec)&(s2>0.4*u.arcsec); jm,rm=j[k],ref_sc[i1[k]]
    dra=(jm.ra-rm.ra).to(u.mas).value*np.cos(jm.dec.rad); dde=(jm.dec-rm.dec).to(u.mas).value
    return dra,dde
msc,_,_,_=D._detect_on_mosaic(D._mast_i2d(o,"F162M"))
jsc,_=D._jwst_positions(o,"F162M")
fig,ax=plt.subplots(1,2,figsize=(11,5.4)); 
for a,(sc,lbl,col) in zip(ax,[(msc,"RAW MAST i2d detections","#4477aa"),(jsc,"jicama catalogue (m7)","#cc3311")]):
    dra,dde=offset_cloud(sc); md,mo=np.median,np.hypot(np.median(dra),np.median(dde))
    a.scatter(dra,dde,s=6,alpha=0.5,color=col)
    a.plot(md(dra),md(dde),"k+",ms=16,mew=2.5)
    for r in (75,): a.add_patch(plt.Circle((0,0),r,fill=False,ec="r",ls=":",lw=1))
    a.axhline(0,color="0.6",lw=0.5); a.axvline(0,color="0.6",lw=0.5); a.set_aspect("equal")
    a.set_xlim(-200,200); a.set_ylim(-200,200)
    a.set_xlabel("ΔRA to VIRAC [mas]"); a.set_ylabel("ΔDec to VIRAC [mas]")
    a.set_title(f"{lbl}\nmedian |offset| = {mo:.0f} mas  (N={len(dra)})",fontsize=10)
fig.suptitle("Cloud E/F jw02092-o005 F162M — offset from VIRAC: raw MAST vs jicama\n(clean isolated unambiguous matches; dotted = 75 mas gate)",fontsize=11)
fig.tight_layout(); fig.savefig("/blue/adamginsburg/adamginsburg/tmp/claude-3663/-blue-adamginsburg-adamginsburg-jwst-brick/9cb81be0-2811-4839-8de6-d2b03c29ee54/scratchpad/mast_vs_jicama_o005.png",dpi=110)
print("saved")
