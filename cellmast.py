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
cells,dropped,grid=D._cell_offsets(sc,ref)
cc=D._cell_consistency(cells,dropped)
print("grid=%d  n_cells=%d dropped=%d spurious=%d"%(grid,cc['n_cells'],cc['n_dropped'],cc.get('n_spurious',0)))
print("field off=%.0f mas (dRA=%.0f dDec=%.0f)  spread=%.0f  consistent=%s"%(cc['off_med'],cc['off_dra'],cc['off_dde'],cc['spread'],cc['consistent']))
for c in cells: print("  cell(%d,%d) off=%.0f n=%d n_ref=%d pr=%.1f"%(c['i'],c['j'],c['off'],c['n'],c['n_ref'],c['peak_ratio']))
