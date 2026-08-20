import warnings; warnings.simplefilter("ignore")
import numpy as np, astropy.units as u
from astropy.table import Table
from astropy.coordinates import SkyCoord
from jwst_gc_pipeline.photometry.astrometry_offsets import measure_offset
from data_qa import diagnostics as D, astrometry_audit as aa
from data_qa.observations import registry
def show(lbl,a,b):
    r=measure_offset(a,b,confirm_windows=True)
    if r is None: print(lbl,": None"); return
    print("%-28s |off|=%6.0f (dRA=%5.0f dDec=%5.0f) contrast=%4.1f ok=%s win=%.2f edgefrac=%.2f n=%d"%(
        lbl,r['off'],r['dra'],r['ddec'],r['contrast'],r['ok'],r['window_arcsec'],r.get('window_edge_fraction',-1),r['npairs']))
# cloudef MAST
o=[x for x in registry(programs=["2092"]) if x.obs=="005" and x.instrument=="NIRCam"][0]
ep=aa.epoch_of(D._mast_i2d(o,"F162M")); ref,_=aa.load_reference(D._refcat_path(o),ep)
sc=SkyCoord(Table.read("/orange/adamginsburg/jwst/cloudef/mastDownload/JWST/jw02092-o005_t002_nircam_f150w2-f162m/jw02092-o005_t002_nircam_f150w2-f162m_cat.ecsv")["sky_centroid"])
show("cloudef o005 MAST",sc,ref)
jsc,_=D._jwst_positions(o,"F162M"); show("cloudef o005 jicama",jsc,ref)
# brick (good field) F212N
ob=[x for x in registry(programs=["2221"]) if x.obs=="001" and x.instrument=="NIRCam"][0]
epb=D._obs_epoch(ob,D._mosaic_path(ob,"F212N")); refb,_=aa.load_reference(D._refcat_path(ob),epb)
jb,_=D._jwst_positions(ob,"F212N"); show("brick o001 jicama F212N",jb,refb)
