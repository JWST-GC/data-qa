import warnings; warnings.simplefilter("ignore")
import numpy as np, astropy.units as u
from astropy.table import Table
from astropy.coordinates import SkyCoord
from jwst_gc_pipeline.photometry.astrometry_offsets import measure_offset
from data_qa import diagnostics as D, astrometry_audit as aa
from data_qa.observations import registry
o=[x for x in registry(programs=["2092"]) if x.obs=="005" and x.instrument=="NIRCam"][0]
ep=aa.epoch_of(D._mast_i2d(o,"F162M")); ref,_=aa.load_reference(D._refcat_path(o),ep)
t=Table.read("/orange/adamginsburg/jwst/cloudef/mastDownload/JWST/jw02092-o005_t002_nircam_f150w2-f162m/jw02092-o005_t002_nircam_f150w2-f162m_cat.ecsv")
sc=SkyCoord(t["sky_centroid"])
mrg=2/3600; box=(ref.ra.deg>sc.ra.deg.min()-mrg)&(ref.ra.deg<sc.ra.deg.max()+mrg)&(ref.dec.deg>sc.dec.deg.min()-mrg)&(ref.dec.deg<sc.dec.deg.max()+mrg)
for mx,sws in [(0.5,[1.0]),(0.5,[0.75]),(0.3,[0.5]),(1.0,[])]:
    r=measure_offset(sc,ref[box],maxsep=mx*u.arcsec,sweep_windows=sws,confirm_windows=True)
    if r: print("maxsep=%.2f sweep=%s: |off|=%.0f (%.0f,%.0f) contrast=%.1f ok=%s win=%.2f edge=%.2f n=%d"%(mx,sws,r['off'],r['dra'],r['ddec'],r['contrast'],r['ok'],r['window_arcsec'],r.get('window_edge_fraction',-1),r['npairs']))
