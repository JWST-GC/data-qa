"""Fit a similarity transform (dx, dy, theta, scale) of JWST per-exposure positions onto VIRAC2
(and Gaia DR3) for one observation.

Method (per ~/.claude memory dataqa-astrometry-offset-method):
  1. coarse bulk tie: 2-D offset histogram of ALL pairs within a SWEPT window, contrast-gated;
  2. fine: per exposure, apply the bulk, MUTUAL nearest pairs within a tight radius,
     de-blended (drop a ref star with >=2 similar-mag JWST sources within 0.3") and
     flux-vetted (JWST inst mag vs Ks linear ridge);
  3. robust linear least-squares similarity fit in a common gnomonic tangent plane,
     bootstrap over unique reference stars.

theta convention: theta_PA > 0 means the correction rotates JWST positions from North toward
East (increasing position angle) to land on the reference.
"""
import sys, os, glob, json, re
import numpy as np
from astropy.table import Table, vstack
from astropy.io import fits
from astropy.time import Time
from astropy.coordinates import SkyCoord
from astropy.stats import mad_std
import astropy.units as u
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from data_qa.observations import Observation
from data_qa import diagnostics as D
from data_qa.mast_monitor import PROGRAMS, TREASURY_PROGRAM

HERE = os.environ.get("ROLL_WORKDIR", os.path.dirname(os.path.abspath(__file__)))
REFCACHE = os.path.join(HERE, "refcache")
OUT = os.path.join(HERE, "results")
os.makedirs(REFCACHE, exist_ok=True)
os.makedirs(OUT, exist_ok=True)
SWPREF = ["F212N", "F200W", "F210M", "F182M", "F187N", "F150W", "F162M", "F115W"]
RNG = np.random.default_rng(42)
AS2RAD = np.pi / 180 / 3600


def tangent(ra, dec, ra0, dec0):
    """Gnomonic projection -> (xi east, eta north) in arcsec."""
    ra, dec = np.radians(ra), np.radians(dec)
    ra0, dec0 = np.radians(ra0), np.radians(dec0)
    cosc = np.sin(dec0) * np.sin(dec) + np.cos(dec0) * np.cos(dec) * np.cos(ra - ra0)
    xi = np.cos(dec) * np.sin(ra - ra0) / cosc
    eta = (np.cos(dec0) * np.sin(dec) - np.sin(dec0) * np.cos(dec) * np.cos(ra - ra0)) / cosc
    return np.degrees(xi) * 3600, np.degrees(eta) * 3600


def detan(xi, eta, ra0, dec0):
    """Inverse gnomonic (arcsec) -> ra, dec deg."""
    x, y = np.radians(xi / 3600), np.radians(eta / 3600)
    ra0, dec0 = np.radians(ra0), np.radians(dec0)
    rho = np.hypot(x, y)
    c = np.arctan(rho)
    with np.errstate(invalid="ignore", divide="ignore"):
        dec = np.arcsin(np.cos(c) * np.sin(dec0) + np.where(rho > 0, y * np.sin(c) * np.cos(dec0) / rho, 0))
        ra = ra0 + np.arctan2(x * np.sin(c), rho * np.cos(dec0) * np.cos(c) - y * np.sin(dec0) * np.sin(c))
    return np.degrees(ra), np.degrees(dec)


# ------------------------------------------------------------------ inputs
def observation(prog, obs):
    p = int(prog)
    field = "gc-treasury" if p == TREASURY_PROGRAM else PROGRAMS[p][obs]
    return Observation(program=prog, obs=obs, target=field, release_field=field, instrument="NIRCam")


_HDR = {}


def crf_header(fn):
    if fn not in _HDR:
        with fits.open(fn) as h:
            h0, h1 = h[0].header, h["SCI"].header
            _HDR[fn] = dict(pa_v3=h1.get("PA_V3"), roll_ref=h1.get("ROLL_REF"), pa_aper=h1.get("PA_APER"),
                            v3i_yang=h1.get("V3I_YANG"), va_scale=h1.get("VA_SCALE"),
                            expmid=h0.get("EXPMID"), date_obs=h0.get("DATE-OBS"),
                            gs_ra=h0.get("GS_RA"), gs_dec=h0.get("GS_DEC"),
                            visit=h0.get("VISIT_ID"), jwst_ver=h0.get("CAL_VER"))
    return _HDR[fn]


def dets_for(filt):
    return ["nrcalong", "nrcblong"] if int(re.sub(r"\D", "", filt)[:3]) >= 250 else D._SW_DETS


def load_jwst(o, filt, snmin=20.0):
    rows = []
    for det in dets_for(filt):
        for c in D._daophot_glob(o, filt, det):
            t = Table.read(c)
            if "skycoord_centroid" not in t.colnames:
                continue
            fn = t.meta.get("FILENAME", "")
            fn = re.sub("//+", "/", fn)
            if fn and not os.path.exists(fn):
                # catalogs made before a field was split per-obs record the old field dir
                alt = re.sub(r"/jwst/[^/]+/", f"/jwst/{o.field}/", fn, count=1)
                if os.path.exists(alt):
                    fn = alt
            hdr = crf_header(fn) if fn and os.path.exists(fn) else {}
            sc = t["skycoord_centroid"]
            flux = np.asarray(t["flux_fit"], float)
            ferr = np.asarray(t["flux_err"], float) if "flux_err" in t.colnames else np.full(len(t), np.nan)
            ok = np.isfinite(sc.ra.deg) & np.isfinite(sc.dec.deg) & (flux > 0)
            sn = flux / ferr
            ok &= ~(np.isfinite(sn) & (sn < snmin))
            if ok.sum() < 20:
                continue
            m = re.search(r"visit(\d+)_vgroup(\d+)_exp(\d+)", os.path.basename(c))
            expkey = m.group(0) if m else os.path.basename(c)
            rows.append(dict(det=det, exp=expkey, visit=(m.group(1) if m else "?"), cat=c, crf=fn,
                             ra=sc.ra.deg[ok], dec=sc.dec.deg[ok], mag=-2.5 * np.log10(flux[ok]),
                             hdr=hdr))
    return rows


def query_ref(kind, key, ra0, dec0, radius_deg):
    fn = os.path.join(REFCACHE, f"{kind}_{key}.fits")
    if os.path.exists(fn):
        t = Table.read(fn)
        if t.meta.get("RADIUS", 0) >= radius_deg:
            return t
    from astroquery.vizier import Vizier
    if kind == "virac2":
        v = Vizier(catalog="II/387/virac2", row_limit=-1,
                   columns=["srcid", "RAJ2000", "DEJ2000", "pmRA", "pmDE", "e_pmRA", "e_pmDE",
                            "e_RAJ2000", "e_DEJ2000", "Ksmag", "Hmag", "Jmag", "UWE"])
    else:
        v = Vizier(catalog="I/355/gaiadr3", row_limit=-1,
                   columns=["Source", "RA_ICRS", "DE_ICRS", "pmRA", "pmDE", "e_pmRA", "e_pmDE", "Gmag", "RUWE"])
    r = v.query_region(SkyCoord(ra0 * u.deg, dec0 * u.deg), radius=radius_deg * u.deg)
    t = r[0]
    t.meta = {"RADIUS": radius_deg}
    t.write(fn, overwrite=True)
    return t


def ref_at_epoch(t, kind, epoch):
    """Reference PM-propagated to ``epoch``.  Returns dict of arrays (ra, dec, mag, sid, epm, uwe)."""
    if kind == "virac2":
        ra, dec, ep0, mag, sid, uwe = t["RAJ2000"], t["DEJ2000"], 2014.0, t["Ksmag"], t["srcid"], t["UWE"]
    else:
        ra, dec, ep0, mag, sid, uwe = t["RA_ICRS"], t["DE_ICRS"], 2016.0, t["Gmag"], t["Source"], t["RUWE"]
    f = lambda c: np.ma.filled(np.ma.asarray(c, float), np.nan)
    ra, dec, pmra, pmde = f(ra), f(dec), f(t["pmRA"]), f(t["pmDE"])
    epm = np.hypot(f(t["e_pmRA"]), f(t["e_pmDE"]))
    good = np.isfinite(pmra) & np.isfinite(pmde) & np.isfinite(ra) & (epm < 5.0)
    dt = epoch - ep0
    # pmRA is pmra*cos(dec) (Gaia/VIRAC2 convention) -> divide by cos(dec) for a coordinate shift
    ra2 = ra + pmra * dt / 3.6e6 / np.cos(np.radians(dec))
    dec2 = dec + pmde * dt / 3.6e6
    return dict(ra=ra2[good], dec=dec2[good], mag=f(mag)[good], sid=np.asarray(sid)[good].astype(str),
                epm=epm[good], uwe=f(uwe)[good])


# ------------------------------------------------------------------ matching
def offset_histogram(jx, jy, rx, ry, window, binsize):
    tree = cKDTree(np.c_[rx, ry])
    pairs = tree.query_ball_point(np.c_[jx, jy], r=window)
    dx = np.concatenate([rx[p] - jx[i] for i, p in enumerate(pairs) if p]) if any(pairs) else np.array([])
    dy = np.concatenate([ry[p] - jy[i] for i, p in enumerate(pairs) if p]) if any(pairs) else np.array([])
    if dx.size < 30:
        return None
    bins = np.arange(-window, window + binsize, binsize)
    H, xe, ye = np.histogram2d(dx, dy, bins=[bins, bins])
    i, j = np.unravel_index(H.argmax(), H.shape)
    bg = np.median(H[H > 0])
    x0, y0 = (xe[i] + xe[i + 1]) / 2, (ye[j] + ye[j + 1]) / 2
    for _ in range(4):
        near = (np.abs(dx - x0) < binsize) & (np.abs(dy - y0) < binsize)
        if near.sum() < 5:
            break
        x0, y0 = np.median(dx[near]), np.median(dy[near])
    edge = (i in (0, H.shape[0] - 1)) or (j in (0, H.shape[1] - 1))
    return dict(dx=float(x0), dy=float(y0), contrast=float(H.max() / bg), npeak=int(H.max()),
                window=window, edge=bool(edge))


def swept_bulk(jx, jy, rx, ry):
    """Swept-window offset histogram (0.3 -> 1 -> 3 -> 10 arcsec).  Keep windows whose peak has
    contrast >= 5 and is not on the edge; accept the smallest such window whose peak agrees within
    50 mas with the next accepted larger window (else the smallest accepted window)."""
    res = []
    for w, b in ((0.3, 0.01), (1.0, 0.02), (3.0, 0.05), (10.0, 0.1)):
        res.append(offset_histogram(jx, jy, rx, ry, w, b))
    ok = [r for r in res if r and r["contrast"] >= 5 and not r["edge"]]
    if not ok:
        return None, res
    for a, b in zip(ok[:-1], ok[1:]):
        if np.hypot(a["dx"] - b["dx"], a["dy"] - b["dy"]) < 0.05:
            a["confirmed_by_window"] = b["window"]
            return a, res
    return ok[0], res



def mutual_pairs(jx, jy, jm, rx, ry, rm, radius=0.08, blend_r=0.3, blend_dm=1.0):
    tj, tr = cKDTree(np.c_[jx, jy]), cKDTree(np.c_[rx, ry])
    d1, i1 = tr.query(np.c_[jx, jy], distance_upper_bound=radius)
    d2, i2 = tj.query(np.c_[rx, ry], distance_upper_bound=radius)
    ij = np.where(np.isfinite(d1))[0]
    ij = ij[i2[i1[ij]] == ij]
    ir = i1[ij]
    # de-blend: drop a ref star with >= 2 JWST sources within blend_r of similar mag to its partner
    keep = np.ones(len(ij), bool)
    nb = tj.query_ball_point(np.c_[rx[ir], ry[ir]], r=blend_r)
    for k, (a, lst) in enumerate(zip(ij, nb)):
        if np.sum(np.abs(jm[lst] - jm[a]) < blend_dm) >= 2:
            keep[k] = False
    return ij[keep], ir[keep]


# ------------------------------------------------------------------ fitting
def fit_sim(x, y, X, Y, w=None):
    """Least-squares X = a x - b y + tx ; Y = b x + a y + ty."""
    n = len(x)
    A = np.zeros((2 * n, 4)); B = np.r_[X, Y]
    A[:n, 0], A[:n, 1], A[:n, 2] = x, -y, 1
    A[n:, 0], A[n:, 1], A[n:, 3] = y, x, 1
    p, *_ = np.linalg.lstsq(A, B, rcond=None)
    return p


def fit_shift_only(x, y, X, Y):
    return np.median(X - x), np.median(Y - y)


def params(p):
    a, b, tx, ty = p
    s = np.hypot(a, b)
    return dict(theta_arcsec=float(-np.degrees(np.arctan2(b, a)) * 3600), scale_ppm=float((s - 1) * 1e6),
                dx_mas=float(tx * 1000), dy_mas=float(ty * 1000))


def apply_sim(p, x, y):
    a, b, tx, ty = p
    return a * x - b * y + tx, b * x + a * y + ty


def robust_fit(x, y, X, Y, sid, nboot=300, clip=3.0):
    keep = np.ones(len(x), bool)
    for _ in range(6):
        p = fit_sim(x[keep], y[keep], X[keep], Y[keep])
        mx, my = apply_sim(p, x, y)
        r = np.hypot(X - mx, Y - my)
        sig = mad_std(np.r_[X[keep] - mx[keep], Y[keep] - my[keep]])
        new = r < clip * sig * np.sqrt(2)
        if (new == keep).all():
            break
        keep = new
    p = fit_sim(x[keep], y[keep], X[keep], Y[keep])
    mx, my = apply_sim(p, x, y)
    # residuals before (shift only) and after
    sx, sy = fit_shift_only(x[keep], y[keep], X[keep], Y[keep])
    res0 = np.r_[X[keep] - x[keep] - sx, Y[keep] - y[keep] - sy] * 1000
    res1 = np.r_[X[keep] - mx[keep], Y[keep] - my[keep]] * 1000
    # bootstrap over unique ref stars
    u_sid, inv = np.unique(sid[keep], return_inverse=True)
    groups = [np.where(inv == k)[0] for k in range(len(u_sid))]
    xk, yk, Xk, Yk = x[keep], y[keep], X[keep], Y[keep]
    boots = []
    for _ in range(nboot):
        pick = RNG.integers(0, len(groups), len(groups))
        idx = np.concatenate([groups[k] for k in pick])
        boots.append(list(params(fit_sim(xk[idx], yk[idx], Xk[idx], Yk[idx])).values()))
    boots = np.array(boots).reshape(-1, 4)
    out = params(p)
    for k, name in enumerate(["theta_arcsec", "scale_ppm", "dx_mas", "dy_mas"]):
        out[name + "_err"] = float(np.std(boots[:, k])) if len(boots) else float("nan")
    out.update(npairs=int(keep.sum()), nstars=int(len(u_sid)), rms_shift_mas=float(mad_std(res0)),
               rms_sim_mas=float(mad_std(res1)),
               extent_arcsec=float(np.ptp(x[keep]) if keep.sum() else 0))
    return out, p, keep


def fit_joint_modules(x, y, X, Y, mod, keep):
    """Common rotation+scale, separate shift per module."""
    x, y, X, Y, mod = x[keep], y[keep], X[keep], Y[keep], mod[keep]
    mods = sorted(set(mod))
    n = len(x)
    A = np.zeros((2 * n, 2 + 2 * len(mods))); B = np.r_[X, Y]
    A[:n, 0], A[:n, 1] = x, -y
    A[n:, 0], A[n:, 1] = y, x
    for k, m in enumerate(mods):
        s = mod == m
        A[:n, 2 + 2 * k] = s
        A[n:, 3 + 2 * k] = s
    p, *_ = np.linalg.lstsq(A, B, rcond=None)
    a, b = p[:2]
    out = dict(theta_arcsec=float(-np.degrees(np.arctan2(b, a)) * 3600), scale_ppm=float((np.hypot(a, b) - 1) * 1e6))
    for k, m in enumerate(mods):
        out[f"dx_{m}_mas"] = float(p[2 + 2 * k] * 1000); out[f"dy_{m}_mas"] = float(p[3 + 2 * k] * 1000)
    return out


# ------------------------------------------------------------------ driver
def collect_pairs(cats, refs, p, radius, refmask=None):
    """Per exposure: map JWST through transform ``p`` (bulk or similarity), mutual-NN + de-blend
    against the epoch-matched reference.  Returned x,y are the ORIGINAL JWST tangent coords."""
    P = dict(x=[], y=[], X=[], Y=[], jm=[], rm=[], sid=[], det=[], exp=[])
    for c in cats:
        R = refs[round(c["epoch"], 3)]
        rx, ry, rm, rs = R["x"], R["y"], R["mag"], R["sid"]
        if refmask is not None:
            m = refmask(R); rx, ry, rm, rs = rx[m], ry[m], rm[m], rs[m]
        mx, my = apply_sim(p, c["x"], c["y"])
        ij, ir = mutual_pairs(mx, my, c["mag"], rx, ry, rm, radius=radius)
        P["x"].append(c["x"][ij]); P["y"].append(c["y"][ij]); P["X"].append(rx[ir]); P["Y"].append(ry[ir])
        P["jm"].append(c["mag"][ij]); P["rm"].append(rm[ir]); P["sid"].append(rs[ir])
        P["det"].append(np.full(len(ij), c["det"])); P["exp"].append(np.full(len(ij), c["exp"]))
    return {k: np.concatenate(v) for k, v in P.items()}


def flux_vet(P):
    g = np.isfinite(P["rm"])
    if g.sum() < 30:
        return P, None
    A = np.polyfit(P["rm"][g], P["jm"][g], 1)
    for _ in range(3):
        r = P["jm"] - np.polyval(A, P["rm"])
        gg = g & (np.abs(r) < max(3 * mad_std(r[g]), 0.3))
        A = np.polyfit(P["rm"][gg], P["jm"][gg], 1)
    r = P["jm"] - np.polyval(A, P["rm"])
    fv = g & (np.abs(r) < max(3 * mad_std(r[g]), 0.3))
    info = dict(slope=float(A[0]), zp=float(A[1]), kept=int(fv.sum()), n=int(len(fv)))
    return {k: v[fv] for k, v in P.items()}, info


def iterate_fit(cats, refs, p0, radius, do_flux_vet, minstars, refmask=None, niter=8, nboot=300):
    """match (through current transform) -> vet -> robust similarity fit; repeat so the match
    window is centred on the fitted transform everywhere in the field (removes the edge
    truncation bias a bulk-only centred window has when a rotation is present)."""
    p = p0; info = None; hist = []
    for it in range(niter):
        P = collect_pairs(cats, refs, p, radius, refmask)
        if do_flux_vet:
            P, info = flux_vet(P)
        if len(np.unique(P["sid"])) < minstars:
            return None, P, None, info, hist
        fit, pnew, keep = robust_fit(P["x"], P["y"], P["X"], P["Y"], P["sid"], nboot=0)
        hist.append(fit["theta_arcsec"])
        p = pnew
        if len(hist) > 1 and abs(hist[-1] - hist[-2]) < 0.1:
            break
    # final: bootstrap on the converged pair set
    fit, p, keep = robust_fit(P["x"], P["y"], P["X"], P["Y"], P["sid"], nboot=nboot)
    return fit, P, keep, info, hist


def sub_fits(P, keep, kind, prefix_min):
    R = {}
    mod = np.array([d[:4] for d in P["det"]])
    if len(set(mod)) == 2:
        R["joint_modules"] = fit_joint_modules(P["x"], P["y"], P["X"], P["Y"], mod, keep)
    for m in sorted(set(mod)):
        s = mod == m
        if len(np.unique(P["sid"][s])) >= prefix_min:
            R[m], _, _ = robust_fit(P["x"][s], P["y"][s], P["X"][s], P["Y"][s], P["sid"][s], nboot=150)
    return R


def run(prog, obs, filt=None):
    o = observation(prog, obs)
    if filt is None:
        for f in SWPREF:
            if any(D._daophot_glob(o, f, d) for d in D._SW_DETS):
                filt = f; break
    if filt is None:
        return dict(program=prog, obs=obs, error="no SW per-exposure catalogs")
    cats = [c for c in load_jwst(o, filt) if c["hdr"]]
    if not cats:
        return dict(program=prog, obs=obs, filt=filt, error="no usable cats")
    allra = np.concatenate([c["ra"] for c in cats]); alldec = np.concatenate([c["dec"] for c in cats])
    ra0, dec0 = float(np.median(allra)), float(np.median(alldec))
    rad = float(SkyCoord(allra * u.deg, alldec * u.deg).separation(SkyCoord(ra0 * u.deg, dec0 * u.deg)).deg.max()) + 0.005
    key = f"{prog}_{obs}"
    hdrs = [c["hdr"] for c in cats if c["hdr"]]
    for c in cats:
        c["epoch"] = float(Time(c["hdr"]["expmid"], format="mjd").jyear)
        c["x"], c["y"] = tangent(c["ra"], c["dec"], ra0, dec0)
    ep_mean = float(np.mean([c["epoch"] for c in cats]))
    hv = lambda k: [h[k] for h in hdrs if h.get(k) is not None]
    result = dict(program=prog, obs=obs, field=o.field, filt=filt, ra0=ra0, dec0=dec0, epoch=ep_mean,
                  ncats=len(cats), pa_v3=float(np.median(hv("pa_v3"))), roll_ref=float(np.median(hv("roll_ref"))),
                  pa_v3_ptp=float(np.ptp(hv("pa_v3"))), va_scale=float(np.median(hv("va_scale"))),
                  date=str(min(hv("date_obs"))), jwst_ver=str(hdrs[0].get("jwst_ver")),
                  gs_ra=float(np.median(hv("gs_ra"))) if hv("gs_ra") else None,
                  gs_dec=float(np.median(hv("gs_dec"))) if hv("gs_dec") else None,
                  visits=sorted(set(c["visit"] for c in cats)))
    if result["gs_ra"] is not None:
        gx, gy = tangent(np.array([result["gs_ra"]]), np.array([result["gs_dec"]]), ra0, dec0)
        result["gs_xy_arcsec"] = [float(gx[0]), float(gy[0])]
    # per-exposure PA_V3 for the per-exposure rotation table
    result["exp_pa_v3"] = {c["exp"]: c["hdr"].get("pa_v3") for c in cats}
    result["exp_epoch"] = {c["exp"]: c["epoch"] for c in cats}

    reftabs, refs_all = {}, {}
    for kind in ("virac2", "gaia"):
        t = query_ref(kind, key, ra0, dec0, rad)
        reftabs[kind] = t
        refs = {}
        for c in cats:
            ek = round(c["epoch"], 3)
            if ek not in refs:
                R = ref_at_epoch(t, kind, c["epoch"])
                R["x"], R["y"] = tangent(R["ra"], R["dec"], ra0, dec0)
                if kind == "virac2":
                    # isolation: no other VIRAC2 source within 1.0"
                    tr = cKDTree(np.c_[R["x"], R["y"]])
                    d, _ = tr.query(np.c_[R["x"], R["y"]], k=2)
                    R["iso"] = d[:, 1] > 1.0
                refs[ek] = R
        refs_all[kind] = refs

    # ---- coarse bulk (VIRAC2), flux-limited JWST, swept window
    R0 = refs_all["virac2"][round(cats[0]["epoch"], 3)]
    sel = cats[: min(len(cats), 16)]
    jx = np.concatenate([c["x"] for c in sel]); jy = np.concatenate([c["y"] for c in sel])
    jm = np.concatenate([c["mag"] for c in sel])
    rb = np.isfinite(R0["mag"]) & (R0["mag"] < 15.5)
    jcut = np.sort(jm)[min(len(jm) - 1, int(3 * rb.sum() * len(sel) / 8))]
    bj = jm <= jcut
    bulk, sweep = swept_bulk(jx[bj], jy[bj], R0["x"][rb], R0["y"][rb])
    result["virac_sweep"] = sweep
    if bulk is None:
        result["error"] = "no confident bulk tie (swept histogram contrast < 5)"
        return result
    result["virac_bulk"] = bulk
    p0 = np.array([1.0, 0.0, bulk["dx"], bulk["dy"]])

    configs = [("virac2", refs_all["virac2"], 0.06, True, 50, None),
               ("virac2_clean", refs_all["virac2"], 0.06, True, 40,
                lambda R: R["iso"] & (R["epm"] < 1.5) & (R["uwe"] < 1.4) & (R["mag"] < 14.5)),
               ("gaia", refs_all["gaia"], 0.08, False, 20, None)]
    for name, refs, radius, fv, nmin, mask in configs:
        fit, P, keep, info, hist = iterate_fit(cats, refs, p0, radius, fv, nmin, mask)
        if fit is None:
            result[name] = dict(error=f"too few matched stars ({len(np.unique(P['sid']))})")
            continue
        fit["theta_iter_history"] = hist
        R = dict(field=fit)
        if info:
            R["fluxvet"] = info
        R.update(sub_fits(P, keep, name, 15 if name == "gaia" else 40))
        if name == "virac2":
            for d in sorted(set(P["det"])):
                s = P["det"] == d
                if len(np.unique(P["sid"][s])) >= 40:
                    R[d], _, _ = robust_fit(P["x"][s], P["y"][s], P["X"][s], P["Y"][s], P["sid"][s], nboot=100)
            pe = {}
            for e in sorted(set(P["exp"])):
                s = P["exp"] == e
                if len(np.unique(P["sid"][s])) >= 60:
                    pe[e], _, _ = robust_fit(P["x"][s], P["y"][s], P["X"][s], P["Y"][s], P["sid"][s], nboot=50)
            R["per_exposure"] = pe
        if name in ("virac2", "gaia"):
            Table(dict(x=P["x"], y=P["y"], X=P["X"], Y=P["Y"], jm=P["jm"], rm=P["rm"], sid=P["sid"],
                       det=P["det"], exp=P["exp"], keep=keep)).write(
                os.path.join(OUT, f"pairs_{key}_{filt}_{name}.fits"), overwrite=True)
        result[name] = R

    # ---- VIRAC2 vs Gaia DR3 directly (reference-frame check, JWST-independent), same epoch
    Rv = refs_all["virac2"][round(cats[0]["epoch"], 3)]
    Rg = refs_all["gaia"][round(cats[0]["epoch"], 3)]
    ax = np.concatenate([c["x"] for c in cats]); ay = np.concatenate([c["y"] for c in cats])
    ig, iv = mutual_pairs(Rg["x"], Rg["y"], Rg["mag"], Rv["x"], Rv["y"], Rv["mag"], radius=0.1)
    if len(ig) >= 20:
        # in the JWST footprint only (same lever arm)
        gx_, gy_ = Rg["x"][ig], Rg["y"][ig]
        fp = (gx_ > ax.min()) & (gx_ < ax.max()) & (gy_ > ay.min()) & (gy_ < ay.max())
        vg, _, _ = robust_fit(Rv["x"][iv][fp], Rv["y"][iv][fp], Rg["x"][ig][fp], Rg["y"][ig][fp],
                              Rg["sid"][ig][fp], nboot=200)
        vg_wide, _, _ = robust_fit(Rv["x"][iv], Rv["y"][iv], Rg["x"][ig], Rg["y"][ig], Rg["sid"][ig], nboot=200)
        result["virac_to_gaia"] = dict(footprint=vg, query_circle=vg_wide)
    return result



if __name__ == "__main__":
    prog, obs = sys.argv[1], sys.argv[2]
    filt = sys.argv[3] if len(sys.argv) > 3 else None
    tag = f"{prog}_{obs}" + (f"_{filt}" if filt else "")
    if os.path.exists(os.path.join(OUT, f"rot_{tag}.json")) or os.path.exists(os.path.join(OUT, f"rot_{tag}.lock")):
        print(tag, "already done / running"); sys.exit(0)
    open(os.path.join(OUT, f"rot_{tag}.lock"), "w").close()
    try:
        res = run(prog, obs, filt)
        with open(os.path.join(OUT, f"rot_{tag}.json"), "w") as f:
            json.dump(res, f, indent=1, default=str)
    finally:
        os.remove(os.path.join(OUT, f"rot_{tag}.lock"))
    v = res.get("virac2", {}).get("field", {})
    g = res.get("gaia", {}).get("field", {}) if isinstance(res.get("gaia"), dict) else {}
    print(tag, res.get("filt"), res.get("error", ""),
          "VIRAC theta=%.2f+-%.2f\" scale=%.1f ppm n=%s" % (v.get("theta_arcsec", np.nan), v.get("theta_arcsec_err", np.nan),
                                                           v.get("scale_ppm", np.nan), v.get("nstars")),
          "GAIA theta=%.2f+-%.2f\" n=%s" % (g.get("theta_arcsec", np.nan), g.get("theta_arcsec_err", np.nan), g.get("nstars")),
          "CLEAN theta=%.2f+-%.2f" % (res.get("virac2_clean", {}).get("field", {}).get("theta_arcsec", np.nan),
                                      res.get("virac2_clean", {}).get("field", {}).get("theta_arcsec_err", np.nan)),
          "V->G theta=%.2f+-%.2f" % (res.get("virac_to_gaia", {}).get("footprint", {}).get("theta_arcsec", np.nan),
                                     res.get("virac_to_gaia", {}).get("footprint", {}).get("theta_arcsec_err", np.nan)))
