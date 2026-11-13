"""Second streaming pass: density panel, T-rho cross-spectra, increments, zoom panel.

The density panel uses the diverging 'seismic' colormap (navy - blue - white - red - dark red).
Its luma is NOT monotone (white is brightest in the middle), so one luma value has two
possible densities. We take the SIGN from chroma (red has Cr > Cb, blue has Cb > Cr) and the
MAGNITUDE from full-resolution luma, on the branch that the sign selects.

Per frame and per height band we store:
  ET, ER   x-spectra of T and rho (row mean removed)
  CO       co-spectrum Re(T_hat * conj(rho_hat)), averaged over rows  -> T-rho correlation by scale
  increment moments <dq^2>, <dq^3>, <dq^4> along x for T and rho, at separations R
  saturation fractions (pixels at the ends of the colour bars)
and, accumulated over all frames, a joint histogram of (T - T_A, rho - rho_A) per band.
For Ra1e12 and Ra1e15 we also store Hann-windowed x-spectra of the zoomed T panel.
"""
from convection.config import CACHE, DATA
import sys, time
import av
import numpy as np

FILES = {"Ra1e9": f"{DATA}/2d_Ra1e9.mp4", "Ra1e12": f"{DATA}/2d_Ra1e12.mp4",
         "Ra1e15": f"{DATA}/2d_Ra1e15.mp4"}
RHO_TOP = {"Ra1e9": 510, "Ra1e12": 1089, "Ra1e15": 550}          # first animated rho row
ZOOM = {"Ra1e12": (358, 888, 2590, 3120), "Ra1e15": (184, 450, 1296, 1559)}
R_PX = np.array([1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256])
NB = 10
TLIM = (-0.1, 0.0)                     # T - T_A colour bar
RLIM = (-0.1, 0.1)                     # rho - rho_A colour bar
HB = 80                                # joint histogram bins


def yuv(fr):
    a = fr.reformat(format="yuv444p").to_ndarray()
    return a[0], a[1].astype(np.int16), a[2].astype(np.int16)   # Y, Cb, Cr


def rho_luts(Y, Cb, Cr, cbar):
    """Two luma -> value tables, one for each branch of the seismic colour bar."""
    r0, r1, c0, c1 = cbar
    y = np.median(Y[r0:r1, c0:c1], axis=1).astype(float)
    s = np.median((Cr - Cb)[r0:r1, c0:c1], axis=1).astype(float)
    val = np.linspace(RLIM[1], RLIM[0], len(y))                  # top = +0.1, bottom = -0.1
    w = int(np.argmax(y))                                        # whitest row = 0
    # positive branch: top .. white, luma increases towards white
    yp, vp = y[:w + 1], val[:w + 1]
    yp = np.maximum.accumulate(yp)
    up, ip = np.unique(yp, return_inverse=True)
    lut_pos = np.interp(np.arange(256), up, np.bincount(ip, vp) / np.bincount(ip))
    # negative branch: white .. bottom, luma decreases away from white
    yn, vn = y[w:][::-1], val[w:][::-1]                          # bottom .. white
    yn = np.maximum.accumulate(yn)
    un, iN = np.unique(yn, return_inverse=True)
    lut_neg = np.interp(np.arange(256), un, np.bincount(iN, vn) / np.bincount(iN))
    return lut_pos, lut_neg, (y, s, val, w)


def invert_rho(Y, Cb, Cr, lut_pos, lut_neg):
    pos = (Cr - Cb) >= 0
    return np.where(pos, lut_pos[Y], lut_neg[Y])


def best_offset(T, R, dr_range=range(-4, 5), dc_range=range(-4, 5)):
    """Shift of the rho crop that best lines up the gradients of T and rho."""
    def g(F):
        gy, gx = np.gradient(F)
        m = np.hypot(gx, gy)
        return (m - m.mean()) / m.std()
    gt = g(T)[8:-8, 8:-8]
    best = None
    for dr in dr_range:
        for dc in dc_range:
            gr = g(np.roll(np.roll(R, -dr, 0), -dc, 1))[8:-8, 8:-8]
            c = (gt * gr).mean()
            if best is None or c > best[0]:
                best = (c, dr, dc)
    return best


def increments(F, rows_of_band):
    """<dq^2>, <dq^3>, <dq^4> along periodic x, for each separation, for each band."""
    out = np.empty((len(rows_of_band), len(R_PX), 3), np.float64)
    for j, r in enumerate(R_PX):
        d = np.roll(F, -r, axis=1) - F
        d2 = d * d
        m2, m3, m4 = d2.mean(1), (d2 * d).mean(1), (d2 * d2).mean(1)
        for b, (a, z) in enumerate(rows_of_band):
            out[b, j] = m2[a:z].mean(), m3[a:z].mean(), m4[a:z].mean()
    return out


def setup(tag):
    sp = np.load(f"{CACHE}/x_spectra_{tag}.npz")
    r0, r1, c0, c1 = (int(v) for v in sp["crop"])
    lut_T = sp["lut"]
    mask = sp["mask"]
    cb = np.load(f"{CACHE}/colour_bar_{tag}.npz")["cbar"]
    st = np.load(f"{CACHE}/frame_stats_{tag}.npz")

    c = av.open(FILES[tag])
    fr = None
    for i, f in enumerate(c.decode(video=0)):
        if i == 5:
            fr = f
            break
    c.close()
    Y, Cb, Cr = yuv(fr)
    shift = RHO_TOP[tag] - (int(sp["crop"][0]) - 2)              # crop has a 2 px inset
    rcb = (cb[0] + shift + 3, cb[1] + shift - 3, cb[2] + 3, cb[3] - 2)
    lut_pos, lut_neg, dbg = rho_luts(Y, Cb, Cr, rcb)

    T = lut_T[Y[r0:r1, c0:c1]]
    best = None
    for extra in range(-3, 4):
        a = r0 + shift + extra
        R = invert_rho(Y[a:a + r1 - r0, c0:c1], Cb[a:a + r1 - r0, c0:c1],
                       Cr[a:a + r1 - r0, c0:c1], lut_pos, lut_neg)
        b = best_offset(T, R, dr_range=[0], dc_range=range(-3, 4))
        if best is None or b[0] > best[0]:
            best = (b[0], extra, b[2])
    rshift = shift + best[1]
    cshift = best[2]
    return dict(r0=r0, r1=r1, c0=c0, c1=c1, lut_T=lut_T, mask=mask, rshift=rshift, cshift=cshift,
                lut_pos=lut_pos, lut_neg=lut_neg, rcb=rcb, dbg=dbg, align=best[0])


def run(tag, keep_frames=(5, 300)):
    S = setup(tag)
    from convection.spectrum_1d.measure_x_spectra import make_interpolator
    interp, _ = make_interpolator(S["mask"])
    r0, r1, c0, c1 = S["r0"], S["r1"], S["c0"], S["c1"]
    rs, cs = S["rshift"], S["cshift"]
    nr, nx = r1 - r0, c1 - c0
    edges = np.linspace(0, nr, NB + 1).astype(int)
    bands = list(zip(edges[:-1], edges[1:]))
    ztop = int(round(0.25 * nr))                                 # rows with z > 0.75
    hann = None
    zc = ZOOM.get(tag)

    keys = ("ET", "ER", "CO", "QU", "incT", "incR", "satT_lo", "satT_hi", "satR", "zoomE", "topE")
    out = {k: [] for k in keys}
    H2 = np.zeros((NB, HB, HB))
    tedges = np.linspace(*TLIM, HB + 1)
    redges = np.linspace(*RLIM, HB + 1)
    samples = {}
    t0 = time.time()
    c = av.open(FILES[tag])
    for i, fr in enumerate(c.decode(video=0)):
        Y, Cb, Cr = yuv(fr)
        Tpos = S["lut_T"][Y[r0:r1, c0:c1]]
        T = interp(TLIM[0] + (TLIM[1] - TLIM[0]) * Tpos)
        sl = (slice(r0 + rs, r1 + rs), slice(c0 + cs, c1 + cs))
        R = interp(invert_rho(Y[sl], Cb[sl], Cr[sl], S["lut_pos"], S["lut_neg"]))
        if i in keep_frames:
            samples[i] = (T.astype(np.float32), R.astype(np.float32))

        Tf = np.fft.rfft(T - T.mean(1, keepdims=True), axis=1)[:, 1:]
        Rf = np.fft.rfft(R - R.mean(1, keepdims=True), axis=1)[:, 1:]
        PT = (np.abs(Tf) ** 2) / nx
        PR = (np.abs(Rf) ** 2) / nx
        X = Tf * np.conj(Rf) / nx
        out["ET"].append(np.stack([PT[a:z].mean(0) for a, z in bands]))
        out["ER"].append(np.stack([PR[a:z].mean(0) for a, z in bands]))
        out["CO"].append(np.stack([X[a:z].real.mean(0) for a, z in bands]))
        out["QU"].append(np.stack([X[a:z].imag.mean(0) for a, z in bands]))
        out["incT"].append(increments(T - T.mean(1, keepdims=True), bands))
        out["incR"].append(increments(R - R.mean(1, keepdims=True), bands))
        out["satT_lo"].append([(Tpos[a:z] <= 0.005).mean() for a, z in bands])
        out["satT_hi"].append([(Tpos[a:z] >= 0.995).mean() for a, z in bands])
        out["satR"].append([(np.abs(R[a:z]) >= 0.099).mean() for a, z in bands])
        for b, (a, z) in enumerate(bands):
            H2[b] += np.histogram2d(T[a:z].ravel(), R[a:z].ravel(), bins=(tedges, redges))[0]
        out["topE"].append(PT[:ztop].mean(0))

        if zc is not None:
            Z = S["lut_T"][Y[zc[0] + 2:zc[1] - 2, zc[2] + 2:zc[3] - 2]]
            Z = TLIM[0] + (TLIM[1] - TLIM[0]) * Z
            if hann is None:
                hann = np.hanning(Z.shape[1])
            Zw = (Z - Z.mean(1, keepdims=True)) * hann
            out["zoomE"].append((np.abs(np.fft.rfft(Zw, axis=1)[:, 1:]) ** 2).mean(0)
                                / (hann ** 2).sum())
    c.close()

    arrs = {k: np.asarray(v, dtype=np.float32) for k, v in out.items() if len(v)}
    smp = {f"T{i}": v[0] for i, v in samples.items()}
    smp.update({f"R{i}": v[1] for i, v in samples.items()})
    np.savez_compressed(f"{CACHE}/density_panel_{tag}.npz", **arrs, **smp, H2=H2, tedges=tedges,
                        redges=redges, R_PX=R_PX, band_edges=edges, nx=nx, nr=nr,
                        rshift=rs, cshift=cs, lut_pos=S["lut_pos"], lut_neg=S["lut_neg"])
    print(f"{tag}: {len(out['ET'])} frames, rho shift ({rs},{cs}), align {S['align']:.2f}, "
          f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    for tag in (sys.argv[1:] or FILES):
        run(tag)
