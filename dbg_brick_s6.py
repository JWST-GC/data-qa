import warnings; warnings.simplefilter("ignore")
from data_qa import diagnostics as D
from data_qa.observations import registry
o=[x for x in registry(programs=["2221"]) if x.obs=="001" and x.instrument=="NIRCam"][0]
avail=D._available_filters(o) or o.filters
wm=D._filters_with_mosaic(o) if hasattr(D,"_filters_with_mosaic") else []
sw,lw=D.pick_filters(avail, prefer=wm) if wm else D.pick_filters(avail)
print("SW",sw,"LW",lw)
for f in (sw,lw):
    jr=D._internal_pos_rms(o,f)
    print(f"  {f}: internal_pos_rms ->", "None" if jr is None else f"n={len(jr[0])}")
