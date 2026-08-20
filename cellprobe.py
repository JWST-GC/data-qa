import warnings; warnings.simplefilter("ignore")
import numpy as np, astropy.units as u
from data_qa import diagnostics as D, astrometry_audit as aa
from data_qa.observations import registry
o=[x for x in registry(programs=["2092"]) if x.obs=="005" and x.instrument=="NIRCam"][0]
path=D._mosaic_path(o,"F162M"); ep=D._obs_epoch(o,path); ref_sc,_=aa.load_reference(D._refcat_path(o),ep)
jsc,src=D._jwst_positions(o,"F162M")
print("jsc N=",len(jsc)," src=",src[:60])
print("jsc RA[%.4f,%.4f] Dec[%.4f,%.4f]"%(jsc.ra.deg.min(),jsc.ra.deg.max(),jsc.dec.deg.min(),jsc.dec.deg.max()))
ra=jsc.ra.deg; dec=jsc.dec.deg; rra=ref_sc.ra.deg; rde=ref_sc.dec.deg
ncell=4
re_=np.linspace(ra.min(),ra.max(),ncell+1); de_=np.linspace(dec.min(),dec.max(),ncell+1)
print("=== per-cell JWST / VIRAC counts (min_src=300) ===")
for i in range(ncell):
    row=[]
    for j in range(ncell):
        m=(ra>=re_[i])&(ra<=re_[i+1])&(dec>=de_[j])&(dec<=de_[j+1]); n=int(m.sum())
        mrg=2/3600
        rm=((rra>=re_[i]-mrg)&(rra<=re_[i+1]+mrg)&(rde>=de_[j]-mrg)&(rde<=de_[j+1]+mrg)); nref=int(rm.sum())
        row.append(f"J{n//1000}k/V{nref}")
    print(" i=%d:"%i, " ".join(row))
