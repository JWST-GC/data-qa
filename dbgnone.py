import warnings; warnings.simplefilter("ignore")
from data_qa import diagnostics as D, astrometry_audit as aa
from data_qa.observations import registry
from jwst_gc_pipeline.photometry.astrometry_offsets import measure_offset
o=[x for x in registry(programs=["2092"]) if x.obs=="005" and x.instrument=="NIRCam"][0]
jsc,_=D._jwst_positions(o,"F162M")
for lbl,ep in [("mosaic-ep",D._obs_epoch(o,D._mosaic_path(o,"F162M"))),("mast-ep",aa.epoch_of(D._mast_i2d(o,"F162M")))]:
    ref,_=aa.load_reference(D._refcat_path(o),ep)
    r=measure_offset(jsc,ref,confirm_windows=True)
    r2=measure_offset(jsc,ref,confirm_windows=False)
    print("%s ep=%.3f: confirm=%s  no-confirm=%s"%(lbl,ep,
        None if r is None else "off=%.0f contrast=%.1f ok=%s edge=%.2f alias=%s"%(r['off'],r['contrast'],r['ok'],r.get('window_edge_fraction',-1),r.get('alias_rejected')),
        None if r2 is None else "off=%.0f contrast=%.1f edge=%.2f"%(r2['off'],r2['contrast'],r2.get('window_edge_fraction',-1))))
