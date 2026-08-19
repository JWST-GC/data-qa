import warnings; warnings.simplefilter("ignore")
from data_qa import diagnostics as D, astrometry_audit as aa
from data_qa.observations import registry
o=[x for x in registry(programs=["2092"]) if x.obs=="005" and x.instrument=="NIRCam"][0]
path=D._mosaic_path(o,"F162M"); ref=D._refcat_path(o); ep=D._obs_epoch(o,path)
ref_sc,_=aa.load_reference(ref,ep); jsc,_=D._jwst_positions(o,"F162M")
cells,dropped,grid=D._cell_offsets(jsc,ref_sc)
cc=D._cell_consistency(cells,dropped)
print(f"o005: n_cells={cc['n_cells']} n_spurious={cc['n_spurious']} off_med={cc['off_med']:.0f} spread={cc['spread']:.0f} consistent={cc['consistent']}")
