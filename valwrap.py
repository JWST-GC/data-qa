import warnings; warnings.simplefilter("ignore")
from data_qa import diagnostics as D, astrometry_audit as aa
from data_qa.observations import registry
def go(prog,obs,filt):
    o=[x for x in registry(programs=[prog]) if x.obs==obs and x.instrument=="NIRCam"][0]
    ep=D._obs_epoch(o,D._mosaic_path(o,filt)) or aa.epoch_of(D._mast_i2d(o,filt))
    ref,_=aa.load_reference(D._refcat_path(o),ep)
    jsc,_=D._jwst_positions(o,filt); jr=D._crossmatch_offset(jsc,ref)
    mp=D._mast_catalog_positions(o,filt); mr=D._crossmatch_offset(mp[0],ref,restrict_footprint=True) if mp else None
    def f(r): return "None" if r is None else "off=%.0f (%.0f,%.0f) contrast=%.1f ok=%s edge=%.2f n=%d [%s]"%(r['off'],r['dra'],r['dde'],r['contrast'],r['ok'],r['edge'],r['n'],r['source'])
    print("%s o%s %s"%(prog,obs,filt))
    print("   jicama:",f(jr))
    print("   MAST  :",("no MAST cat" if mp is None else f(mr)))
go("2092","005","F162M")
go("2221","001","F212N")
