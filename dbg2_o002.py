import warnings; warnings.simplefilter("ignore")
import numpy as np
from data_qa import diagnostics as D, astrometry_audit as aa
from data_qa.observations import registry
o=[x for x in registry(programs=["2092"]) if x.obs=="002" and x.instrument=="NIRCam"][0]
path=D._mosaic_path(o,"F162M"); ref=D._refcat_path(o); ep=D._obs_epoch(o,path)
ref_sc,_=aa.load_reference(ref,ep); jsc,_=D._jwst_positions(o,"F162M")
cells,dropped,grid=D._cell_offsets(jsc,ref_sc)
cc=D._cell_consistency(cells,dropped)
print(f"after: n_cells={cc['n_cells']} n_spurious={cc['n_spurious']} n_dropped={cc['n_dropped']}")
print(f"off_med={cc['off_med']:.0f} (dRA={cc['off_dra']:.0f},dDec={cc['off_dde']:.0f}) spread={cc['spread']:.0f} consistent={cc['consistent']}")
print(f"n_confirmed={cc['n_confirmed']} bad_frac={cc['bad_src_frac']*100:.1f}% coverage={cc['coverage']:.2f}")
