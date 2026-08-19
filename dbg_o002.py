import warnings; warnings.simplefilter("ignore")
import numpy as np, astropy.units as u
from data_qa import diagnostics as D, astrometry_audit as aa
from data_qa.observations import registry
o=[x for x in registry(programs=["2092"]) if x.obs=="002" and x.instrument=="NIRCam"][0]
sw="F162M"
path=D._mosaic_path(o,sw); ref=D._refcat_path(o); ep=D._obs_epoch(o,path)
print("mosaic",path); print("epoch",ep)
ref_sc,_=aa.load_reference(ref,ep)
jsc,src=D._jwst_positions(o,sw)
print("jwst src:",src,"N=",len(jsc),"  ref N=",len(ref_sc))
print(f"jwst RA[{jsc.ra.deg.min():.3f},{jsc.ra.deg.max():.3f}] Dec[{jsc.dec.deg.min():.3f},{jsc.dec.deg.max():.3f}]")
print(f"ref  RA[{ref_sc.ra.deg.min():.3f},{ref_sc.ra.deg.max():.3f}] Dec[{ref_sc.dec.deg.min():.3f},{ref_sc.dec.deg.max():.3f}]")
# whole-field nearest match bulk
idx,sep,_=jsc.match_to_catalog_sky(ref_sc); k=sep<0.3*u.arcsec
jm,rm=jsc[k],ref_sc[idx[k]]
dra=(jm.ra-rm.ra).to(u.mas).value*np.cos(jm.dec.rad); dde=(jm.dec-rm.dec).to(u.mas).value
print(f"whole-field NN<0.3\": N={k.sum()} median dRA={np.median(dra):.0f} dDec={np.median(dde):.0f} mas")
# per-cell from the actual code
cells,dropped,grid=D._cell_offsets(jsc,ref_sc)
print(f"grid={grid} ncells={len(cells)} dropped={len(dropped)}")
for c in cells:
    print(f"  cell({c['i']},{c['j']}) dra={c['dra']:7.0f} dde={c['dde']:7.0f} off={c['off']:6.0f} n={c['n']:6d} n_ref={c['n_ref']:5d} pr={c['peak_ratio']:.1f} npairs={c['npairs']}")
