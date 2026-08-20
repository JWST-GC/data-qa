import warnings; warnings.simplefilter("ignore")
import numpy as np, astropy.units as u
from astropy.table import Table
from astropy.coordinates import SkyCoord
from data_qa import diagnostics as D, astrometry_audit as aa
from data_qa.observations import registry
def mnn_offset(sc, flux, ref, win=0.5, nbright=None, deblend=0.3):
    if nbright:
        g=np.isfinite(flux)&(flux>0); order=np.argsort(flux[g])[::-1][:nbright]; sc=sc[g][order]
    # deblend: drop sc with a neighbor sc within deblend"
    ii,ss,_=sc.match_to_catalog_sky(sc,nthneighbor=2); sc=sc[ss>deblend*u.arcsec]
    i_ab,s_ab,_=sc.match_to_catalog_sky(ref)          # sc -> nearest ref
    i_ba,s_ba,_=ref.match_to_catalog_sky(sc)          # ref -> nearest sc
    mutual=(i_ba[i_ab]==np.arange(len(sc)))&(s_ab<win*u.arcsec)
    jm,rm=sc[mutual],ref[i_ab[mutual]]
    dra=(jm.ra-rm.ra).to(u.mas).value*np.cos(jm.dec.rad); dde=(jm.dec-rm.dec).to(u.mas).value
    return np.median(dra),np.median(dde),int(mutual.sum())
# cloudef MAST
t=Table.read("/orange/adamginsburg/jwst/cloudef/mastDownload/JWST/jw02092-o005_t002_nircam_f150w2-f162m/jw02092-o005_t002_nircam_f150w2-f162m_cat.ecsv")
sc=SkyCoord(t["sky_centroid"]); flux=np.asarray(t["aper_total_flux"],float)
o=[x for x in registry(programs=["2092"]) if x.obs=="005" and x.instrument=="NIRCam"][0]
ep=aa.epoch_of(D._mast_i2d(o,"F162M")); ref,_=aa.load_reference(D._refcat_path(o),ep)
for nb in (None,5000,2000):
    for win in (0.5,0.3):
        dra,dde,n=mnn_offset(sc,flux,ref,win=win,nbright=nb)
        print("cloudef MAST nb=%s win=%.1f: |off|=%.0f (dRA=%.0f dDec=%.0f) n=%d"%(nb,win,np.hypot(dra,dde),dra,dde,n))
