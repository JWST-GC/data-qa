import warnings; warnings.simplefilter("ignore")
import numpy as np, astropy.units as u
from astropy.io import fits; from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.nddata import Cutout2D
from data_qa import diagnostics as D
from data_qa.observations import registry
o=[x for x in registry(programs=["2092"]) if x.obs=="005" and x.instrument=="NIRCam"][0]
mp=D._cutout_mosaic(o,"F162M")
print("cutout mosaic:",mp)
a_sc,b_sc,minfo=D._module_positions(o,"F162M")
print("a_sc,b_sc:",None if a_sc is None else len(a_sc), None if b_sc is None else len(b_sc))
ov=D._ab_overlap(a_sc,b_sc) if (a_sc is not None and b_sc is not None) else None
print("ov keys:",None if ov is None else list(ov.keys()))
if ov is not None: print("ov pos N:",len(ov.get("pos",[])))
if mp and ov is not None and len(ov.get("pos",[])):
    with fits.open(mp) as h:
        sci=h["SCI"] if "SCI" in h else h[1]; data=sci.data.astype("float32"); w=WCS(sci.header)
    print("mosaic shape",data.shape)
    ok=0; oob=0; empty=0
    for ra,dec in ov["pos"][:200]:
        try:
            x,y=w.world_to_pixel(SkyCoord(ra*u.deg,dec*u.deg))
            cut=Cutout2D(data,(float(x),float(y)),25,wcs=w)
        except (ValueError,IndexError):
            oob+=1; continue
        if not np.isfinite(cut.data).any() or np.nanmax(cut.data)<=0: empty+=1; continue
        ok+=1
    print(f"of 200 pos: ok={ok} out-of-bounds={oob} empty={empty}")
import numpy as np
pos=np.array(ov["pos"])
print("pos[0]:",pos[0], " pos RA range [%.4f,%.4f] Dec[%.4f,%.4f]"%(pos[:,0].min(),pos[:,0].max(),pos[:,1].min(),pos[:,1].max()))
# mosaic footprint corners
cy,cx=data.shape
corners=w.pixel_to_world([0,cx,0,cx],[0,0,cy,cy])
print("mosaic RA range [%.4f,%.4f] Dec[%.4f,%.4f]"%(corners.ra.deg.min(),corners.ra.deg.max(),corners.dec.deg.min(),corners.dec.deg.max()))
print("a_sc RA range [%.4f,%.4f] Dec[%.4f,%.4f]"%(a_sc.ra.deg.min(),a_sc.ra.deg.max(),a_sc.dec.deg.min(),a_sc.dec.deg.max()))
