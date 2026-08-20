import warnings; warnings.simplefilter("ignore")
import numpy as np
from astropy.io import fits; from astropy.wcs import WCS
from photutils.aperture import CircularAperture, CircularAnnulus, aperture_photometry, ApertureStats
from data_qa import diagnostics as D
from data_qa.observations import registry
o=[x for x in registry(programs=["2092"]) if x.obs=="005" and x.instrument=="NIRCam"][0]
sc,pf,src=D._psf_flux_positions(o,"F162M")
print("psf cat:",src,"N=",len(sc))
print("psf_flux: min %.3g max %.3g median %.3g  (n<=0: %d)"%(np.nanmin(pf),np.nanmax(pf),np.nanmedian(pf),(pf<=0).sum()))
mp=D._mosaic_path(o,"F162M")
h=fits.open(mp) if False else fits.open(mp)
sci=h["SCI"] if "SCI" in h else h[1]; data=sci.data.astype("float32"); w=WCS(sci.header)
print("mosaic BUNIT:",sci.header.get("BUNIT"),"pixscale mas:",abs(sci.header.get("CD1_1",sci.header.get("CDELT1",0)))*3.6e6)
print("mosaic data: median %.4g  max %.4g"%(np.nanmedian(data),np.nanmax(data)))
