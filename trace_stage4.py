import warnings; warnings.simplefilter("ignore")
from data_qa import diagnostics as D
from data_qa.observations import registry
import numpy as np, astropy.units as u
o=[o for o in registry(programs=["2092"]) if o.obs=="005" and o.instrument=="NIRCam"][0]
print("field",o.field)
print("refcat:", D._refcat_path(o))
print("mosaic:", D._mosaic_path(o,"F162M"))
# what catalog do jwst_sources load?
import data_qa.diagnostics as DD
sc,mag,src=DD._jwst_sources(o,"F162M",position_valid=True)
print("jwst src label:", src, "N=", 0 if sc is None else len(sc))
ep=D._obs_epoch(o, D._mosaic_path(o,"F162M"))
print("epoch:", ep)
ref=D._refcat_path(o)
import data_qa.astrometry_audit as aa
ref_sc,_=aa.load_reference(ref,ep)
print("ref N:", len(ref_sc))
# direct nearest-match bulk (like my script)
idx,sep,_=sc.match_to_catalog_sky(ref_sc); k=sep<0.3*u.arcsec
jm,rm=sc[k],ref_sc[idx[k]]
dra=(jm.ra-rm.ra).to(u.mas).value*np.cos(jm.dec.rad); dde=(jm.dec-rm.dec).to(u.mas).value
print(f"nearest-match<0.3\": N={k.sum()} bulk dRA={np.median(dra):.0f} dDec={np.median(dde):.0f} mas")
