import warnings; warnings.simplefilter("ignore")
import numpy as np, astropy.units as u
from astropy.io import fits; from astropy.wcs import WCS
from data_qa import diagnostics as D
from data_qa.observations import registry
o=[x for x in registry(programs=["2092"]) if x.obs=="005" and x.instrument=="NIRCam"][0]
jsc,jflux,jname=D._psf_flux_positions(o,"F162M")
jmag=-2.5*np.log10(jflux); fin=np.isfinite(jmag); jsc,jmag=jsc[fin],jmag[fin]
print("jicama full N=",len(jsc)," jmag range [%.1f,%.1f] median %.1f"%(jmag.min(),jmag.max(),np.median(jmag)))
mast_path=D._mast_i2d(o,"F162M")
with fits.open(mast_path) as h:
    mh=(h["SCI"] if "SCI" in h else h[1]).header
mwcs=WCS(mh); mny,mnx=int(mh["NAXIS2"]),int(mh["NAXIS1"])
print("MAST i2d size",mnx,mny)
crop=min(5000,mny,mnx)
cw=mwcs.pixel_to_world([mnx/2-crop/2,mnx/2+crop/2],[mny/2-crop/2,mny/2+crop/2])
ra_lo,ra_hi=sorted(cw.ra.deg); de_lo,de_hi=sorted(cw.dec.deg)
print("crop box RA[%.4f,%.4f] Dec[%.4f,%.4f]"%(ra_lo,ra_hi,de_lo,de_hi))
inb=((jsc.ra.deg>=ra_lo)&(jsc.ra.deg<=ra_hi)&(jsc.dec.deg>=de_lo)&(jsc.dec.deg<=de_hi))
print("jicama in-box N=",int(inb.sum()))
print("jicama RA range [%.4f,%.4f] Dec[%.4f,%.4f]"%(jsc.ra.deg.min(),jsc.ra.deg.max(),jsc.dec.deg.min(),jsc.dec.deg.max()))
print("jmag[in-box] range [%.1f, %.1f]  N=%d"%(jmag[inb].min(), jmag[inb].max(), inb.sum()))
# how many jicama total are FAINT (jmag > -6)?
print("jicama full faint (jmag>-6): %d of %d"%((jmag>-6).sum(), len(jmag)))
print("jicama in-box faint (jmag>-6): %d"%((jmag[inb]>-6).sum()))
