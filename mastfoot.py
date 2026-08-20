import warnings; warnings.simplefilter("ignore")
import numpy as np, astropy.units as u
from astropy.table import Table
from astropy.coordinates import SkyCoord
from jwst_gc_pipeline.photometry.astrometry_offsets import measure_offset
from data_qa import diagnostics as D, astrometry_audit as aa
from data_qa.observations import registry
o=[x for x in registry(programs=["2092"]) if x.obs=="005" and x.instrument=="NIRCam"][0]
ep=aa.epoch_of(D._mast_i2d(o,"F162M")); ref,_=aa.load_reference(D._refcat_path(o),ep)
sc=SkyCoord(Table.read("/orange/adamginsburg/jwst/cloudef/mastDownload/JWST/jw02092-o005_t002_nircam_f150w2-f162m/jw02092-o005_t002_nircam_f150w2-f162m_cat.ecsv")["sky_centroid"])
# restrict VIRAC to MAST footprint (+2" margin)
mrg=2/3600
box=(ref.ra.deg>sc.ra.deg.min()-mrg)&(ref.ra.deg<sc.ra.deg.max()+mrg)&(ref.dec.deg>sc.dec.deg.min()-mrg)&(ref.dec.deg<sc.dec.deg.max()+mrg)
r=measure_offset(sc,ref[box],confirm_windows=True)
print("cloudef MAST (VIRAC in footprint): |off|=%.0f (dRA=%.0f dDec=%.0f) contrast=%.1f ok=%s win=%.2f edgefrac=%.2f n=%d"%(r['off'],r['dra'],r['ddec'],r['contrast'],r['ok'],r['window_arcsec'],r.get('window_edge_fraction',-1),r['npairs']))

flux=np.asarray(Table.read("/orange/adamginsburg/jwst/cloudef/mastDownload/JWST/jw02092-o005_t002_nircam_f150w2-f162m/jw02092-o005_t002_nircam_f150w2-f162m_cat.ecsv")["aper_total_flux"],float)
g=np.isfinite(flux)&(flux>0)
for nb in (10000,5000,2000):
    order=np.argsort(flux[g])[::-1][:nb]; bri=sc[g][order]
    for sw in (True,False):
        r=measure_offset(bri,ref[box],sweep=sw,maxsep=(3.0 if sw else 0.5)*u.arcsec,confirm_windows=sw)
        if r: print("bright%5d sweep=%s: |off|=%.0f (%.0f,%.0f) contrast=%.1f ok=%s win=%.2f edge=%.2f n=%d"%(nb,sw,r['off'],r['dra'],r['ddec'],r['contrast'],r['ok'],r['window_arcsec'],r.get('window_edge_fraction',-1),r['npairs']))
