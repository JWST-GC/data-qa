import warnings, glob, os, time; warnings.simplefilter("ignore")
from astropy.table import Table
from astropy.coordinates import SkyCoord
import numpy as np, astropy.units as u
from data_qa import diagnostics as D
from data_qa.observations import registry
o=[x for x in registry(programs=["2092"]) if x.obs=="005" and x.instrument=="NIRCam"][0]
print("cloudef field dir:", D.BASE+"/"+o.field)
# what does _list_field_catalogs pick, in order?
cats=D._list_field_catalogs(o) if hasattr(D,"_list_field_catalogs") else None
# fallback: replicate the glob
print("=== cloudef/catalogs oksep2221 files ===")
for p in glob.glob(f"/orange/adamginsburg/jwst/{o.field}/catalogs/*oksep2221*.fits"):
    print(" ", os.path.basename(p), os.path.islink(p), "->", (os.path.realpath(p) if os.path.islink(p) else ""))
# Now call the actual loader and get sky span
sc,mag,src=D._jwst_sources(o,"F162M",position_valid=True)
print("loaded src label:", src, "N=",len(sc))
print(f"  loaded RA [{sc.ra.deg.min():.4f},{sc.ra.deg.max():.4f}] Dec [{sc.dec.deg.min():.4f},{sc.dec.deg.max():.4f}]")
r=Table.read(f"/orange/adamginsburg/jwst/{o.field}/catalogs/gaia_virac2_refcat_epoch2023.21.fits")
print(f"  cloudef VIRAC RA [{r['RA'].min():.4f},{r['RA'].max():.4f}] Dec [{r['DEC'].min():.4f},{r['DEC'].max():.4f}]")
