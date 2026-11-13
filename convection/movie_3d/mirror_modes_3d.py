r"""3D movie (Ra = 1e8): the vertical analysis on the wall faces, and full statistics on all six faces.

The box is 4 x 4 x 1 (camera fit to the corners: L/H = 3.96), periodic in x and y. The four drawn walls are
only TWO planes (y = 0 and x = 0), each drawn twice (top view and bottom view): copies correlate at 0.92-0.98.
We keep all four faces; the copies measure the rendering/extraction noise.

Per wall face and frame (same definitions as mirror_modes_2d.py):
  theta' = theta + eps z -> DST-I in z, rFFT in x: mean-profile modes, shells, sectors, 1D x and z spectra
  varrho                 -> DCT-I in z; T-rho co-spectra; profiles; moments; PDFs by height; structure functions
Lid and floor (horizontal planes): 2D FFT (periodic both ways), shells of theta and varrho and their co-spectrum,
mean values (their heights are found later from the wall profile), moments and PDFs.
"""
from convection.config import CACHE
import sys, time
import av, cv2
import numpy as np
from scipy.fft import rfft, dst, dct, fft2
from convection.movie_3d import read_3d_temperature as T3
from convection.movie_3d import read_3d_density as R3
from convection.common import mirror_transform as V
class L32(list):
    """List that stores float32 / complex64 copies (keeps memory low on long movies)."""
    def append(self, v):
        a = np.asarray(v)
        super().append(a.astype(np.complex64 if np.iscomplexobj(a) else np.float32))


EPS = 0.1
WALLS = ("wall_xz", "wall_yz", "wall_back1", "wall_back2")
FLATS = ("lid", "floor")
RS = np.array([1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128])
HB, HR = 400, 0.1
NBAND = 5
NC_M, NC_N = 65, 32


def line_centres(prof, side):
    """Centre of the white edge line in a warped mean-luma profile (first/last 6 px), luma-weighted."""
    p = prof[:6] if side == 0 else prof[-6:][::-1]
    base = np.median(prof[8:20]) if side == 0 else np.median(prof[-20:-8])
    w = np.clip(p - base, 0, None)
    w[np.argmax(p) + 3:] = 0
    c = (w * np.arange(6)).sum() / w.sum()
    return c if side == 0 else len(prof) - 1 - c


def face_geometry(mean_w):
    pz = np.median(mean_w[:, 40:-40], 1)
    px = np.median(mean_w[30:-30, :], 0)
    y1, y0 = line_centres(pz, 0), line_centres(pz, 1)
    x0, x1 = line_centres(px, 0), line_centres(px, 1)
    r0, r1 = int(np.ceil(y1 + 2.5)), int(np.floor(y0 - 2.5))
    c0, c1 = int(np.ceil(x0 + 2.5)), int(np.floor(x1 - 2.5))
    return dict(y1=y1, y0=y0, x0=x0, x1=x1, r0=r0, r1=r1, c0=c0, c1=c1, Hz=y0 - y1, W=x1 - x0)


def run(every=1):
    st = np.load(f"{CACHE}/frame_stats_3d.npz"); mean, std = st["mean"].astype(np.float32), st["std"]
    HT = T3.homographies(); HR_ = {n: R3.homog(q, n) for n, q in R3.RFACES.items()}
    c = av.open(T3.PATH)
    for i, fr in enumerate(c.decode(video=0)):
        if i == 500:
            rgb = fr.to_ndarray(format="rgb24"); break
    c.close()
    _, prgb, py = T3.colour_bar(rgb, None, (mean, std)); u, val = T3.luma_lut(py)
    light, luts, geo, plans = {}, {}, {}, {}
    for name in T3.FACES:
        light[name] = T3.fit_light(rgb, T3.face_mask(name, mean.shape), prgb)[0]
        m = np.zeros(mean.shape, np.uint8); cv2.fillConvexPoly(m, np.int32(R3.RFACES[name]), 1)
        m = cv2.erode(m, np.ones((17, 17), np.uint8)).astype(bool)
        d, s, a, Gg = R3.fit_model(rgb[m].astype(float)[::97])
        luts[name] = R3.lut_for(s, a, Gg)
        W, Hh = T3.out_size(name)
        mw = cv2.warpPerspective(mean, HT[name], (W, Hh))
        mr = cv2.warpPerspective(mean, HR_[name], (W, Hh))
        g, gr = face_geometry(mw), face_geometry(mr)
        geo[name] = (g, gr)
        print(f"{name:10s} light {light[name]:.2f} | rho model s={s:.2f} a={a:.2f} G={Gg} (RGB dist {d:.1f}) | "
              f"T lines z1 {g['y1']:.2f} z0 {g['y0']:.2f} x {g['x0']:.2f}-{g['x1']:.2f} | rho lines z1 {gr['y1']:.2f} z0 {gr['y0']:.2f}",
              flush=True)
    # plane set-up
    P = {}
    for name in WALLS:
        g, gr = geo[name]
        Nz, Nx = int(round(g["Hz"])), int(round(g["W"]))
        zr = (g["y0"] - np.arange(g["r0"], g["r1"] + 1)) / g["Hz"]
        zrr = (gr["y0"] - np.arange(gr["r0"], gr["r1"] + 1)) / gr["Hz"]
        MT, _, _ = V.zinterp_matrix(zr, Nz, walls=(0.0, 0.0))
        MR, _, _ = V.zinterp_matrix(zrr, Nz)
        cref = int(round(g["x0"]))
        P[name] = dict(Nz=Nz, Nx=Nx, zr=zr, zrr=zrr, MT=MT, MR=MR, cref=cref,
                       gapT=V.gap_plan(g["c0"] - cref, g["c1"] - cref, Nx),
                       gapR=V.gap_plan(gr["c0"] - cref, gr["c1"] - cref, Nx),
                       Ssin=V.Shells(Nz, Nx, "sin"), Scos=V.Shells(Nz, Nx, "cos"))
        print(f"{name:10s} grid Nz={Nz} Nx={Nx}, T data rows {g['r0']}-{g['r1']} cols {g['c0']}-{g['c1']}", flush=True)

    def place(F, c0, c1, cref, gap, Nx):
        out = np.empty((F.shape[0], Nx)); out[:, c0 - cref:c1 - cref + 1] = F
        gidx, wl, wf = gap
        out[:, gidx] = F[:, -1:] * wl[None, :] + F[:, :1] * wf[None, :]
        return out

    out = {nm: {} for nm in T3.FACES}
    acc = {nm: {} for nm in T3.FACES}
    Hst = {nm: {k: np.zeros((NBAND, HB)) for k in ("T", "R")} for nm in T3.FACES}
    samples = {}
    t0 = time.time(); nfr = 0
    c = av.open(T3.PATH)
    for i, fr in enumerate(c.decode(video=0)):
        if i % every:
            continue
        Y = fr.to_ndarray(format="yuv420p")[:fr.height, :fr.width].astype(np.float32)
        RGB = fr.to_ndarray(format="rgb24")
        for name in T3.FACES:
            W, Hh = T3.out_size(name)
            Tw = cv2.warpPerspective(np.interp(16 + (Y - 16) / light[name], u, val).astype(np.float32), HT[name], (W, Hh)).astype(float)
            Rw = cv2.warpPerspective(RGB, HR_[name], (W, Hh))
            qi = (Rw.astype(np.int32) * R3.Q) // 256
            Rf = luts[name][qi[..., 0], qi[..., 1], qi[..., 2]].astype(float)
            g, gr = geo[name]
            o, a_ = out[name], acc[name]
            def put(k, v):
                o.setdefault(k, L32()).append(v)
            def add(k, v):
                a_[k] = a_.get(k, 0.0) + v
            if name in WALLS:
                p = P[name]; Nz, Nx = p["Nz"], p["Nx"]
                zg = np.arange(Nz + 1) / Nz
                th = Tw[g["r0"]:g["r1"] + 1, g["c0"]:g["c1"] + 1]
                GT = p["MT"] @ place(th + EPS * p["zr"][:, None], g["c0"], g["c1"], p["cref"], p["gapT"], Nx)
                GR = p["MR"] @ place(Rf[gr["r0"]:gr["r1"] + 1, gr["c0"]:gr["c1"] + 1], gr["c0"], gr["c1"], p["cref"], p["gapR"], Nx)
                thg = GT - EPS * zg[:, None]
                xw = V.xw(Nx)
                A = rfft(GT[1:-1], axis=1)
                C = dst(A.real, type=1, axis=0) + 1j * dst(A.imag, type=1, axis=0)
                E = np.abs(C) ** 2 * xw[None, :] / (2.0 * Nx ** 2 * Nz ** 2)
                add("T2d", E)
                put("T_mean", E[:, 0]); put("T_shell", p["Ssin"].shell(E)); put("T_sect", p["Ssin"].sectors(E))
                put("T_x", E[:, 1:].sum(0)); put("T_z", E[:, 1:].sum(1)); put("cT", C[:NC_N, :NC_M].astype(np.complex64))
                wn = np.ones(Nz + 1); wn[[0, -1]] = 0.5
                nrm = xw[None, :] * wn[:, None] / (2.0 * Nx ** 2 * Nz ** 2)
                Ac, AR = rfft(GT, axis=1), rfft(GR, axis=1)
                Cc = dct(Ac.real, type=1, axis=0) + 1j * dct(Ac.imag, type=1, axis=0)
                CR = dct(AR.real, type=1, axis=0) + 1j * dct(AR.imag, type=1, axis=0)
                Ec, ER, X = np.abs(Cc) ** 2 * nrm, np.abs(CR) ** 2 * nrm, (Cc * np.conj(CR)).real * nrm
                add("Tc2d", Ec); add("R2d", ER); add("XTR2d", X)
                put("Tc_shell", p["Scos"].shell(Ec)); put("R_shell", p["Scos"].shell(ER)); put("X_TR_shell", p["Scos"].shell(X))
                put("R_mean", ER[:, 0]); put("R_x", ER[:, 1:].sum(0)); put("R_z", ER[:, 1:].sum(1))
                put("Tc_z", Ec[:, 1:].sum(1)); put("X_TR_z", X[:, 1:].sum(1)); put("X_TR_x", X[:, 1:].sum(0))
                fT = thg - thg.mean(1, keepdims=True); fR = GR - GR.mean(1, keepdims=True)
                put("prof_T", thg.mean(1)); put("prof_R", GR.mean(1))
                bedges = np.linspace(1, Nz, NBAND + 1).astype(int)
                bor = np.clip(np.searchsorted(bedges, np.arange(Nz + 1), side="right") - 1, 0, NBAND - 1)
                for k_, f_ in (("T", fT), ("R", fR)):
                    f2 = f_ * f_
                    put("mom_" + k_, np.stack([f2.mean(1), (f2 * f_).mean(1), (f2 * f2).mean(1)]))
                    ib = np.clip(((f_[1:-1] + HR) / (2 * HR) * HB).astype(int), 0, HB - 1)
                    Hst[name][k_] += np.bincount((bor[1:-1, None] * HB + ib).ravel(), minlength=NBAND * HB).reshape(NBAND, HB)
                jb0, jb1 = int(round(0.05 * Nz)), int(round(0.95 * Nz))
                if nfr % 2 == 0:
                    put("sfx_T", V.increments(fT, RS, 1, rows=(jb0, jb1))); put("sfz_T", V.increments(fT, RS, 0, rows=(jb0, jb1)))
                    put("sfx_R", V.increments(fR, RS, 1, rows=(jb0, jb1))); put("sfz_R", V.increments(fR, RS, 0, rows=(jb0, jb1)))
                if i in (0, 500, 1000):
                    samples[f"{name}_T{i}"] = thg.astype(np.float32); samples[f"{name}_R{i}"] = GR.astype(np.float32)
            else:
                Ft = Tw[g["r0"]:g["r1"] + 1, g["c0"]:g["c1"] + 1]
                Fr = Rf[gr["r0"]:gr["r1"] + 1, gr["c0"]:gr["c1"] + 1]
                n_ = min(Ft.shape[0], Fr.shape[0]); m_ = min(Ft.shape[1], Fr.shape[1])
                Ft, Fr = Ft[:n_, :m_], Fr[:n_, :m_]
                put("meanT", Ft.mean()); put("meanR", Fr.mean())
                ft, fr_ = Ft - Ft.mean(), Fr - Fr.mean()
                put("momT", [(ft ** 2).mean(), (ft ** 3).mean(), (ft ** 4).mean()])
                put("momR", [(fr_ ** 2).mean(), (fr_ ** 3).mean(), (fr_ ** 4).mean()])
                for k_, f_ in (("T", ft), ("R", fr_)):
                    ib = np.clip(((f_ + HR) / (2 * HR) * HB).astype(int), 0, HB - 1)
                    Hst[name][k_][0] += np.bincount(ib.ravel(), minlength=HB)
                # 2D FFT: data region is not exactly one period (inset ~3 px of ~541) -> Hann^2 window
                w2 = np.outer(np.hanning(n_), np.hanning(m_))
                At, Ar = fft2(ft * w2), fft2(fr_ * w2)
                nrm = 1.0 / (w2 ** 2).sum() / (n_ * m_)
                Pt, Pr_, Xc = np.abs(At) ** 2 * nrm, np.abs(Ar) ** 2 * nrm, (At * np.conj(Ar)).real * nrm
                ky = np.fft.fftfreq(n_) * n_; kx = np.fft.fftfreq(m_) * m_
                K = np.rint(np.hypot(kx[None, :], ky[:, None])).astype(int)
                kmax = min(n_, m_) // 2
                for k_, P_ in (("T_shell", Pt), ("R_shell", Pr_), ("X_shell", Xc)):
                    put(k_, np.bincount(K.ravel(), weights=P_.ravel(), minlength=kmax + 1)[:kmax + 1])
                if i in (0, 500, 1000):
                    samples[f"{name}_T{i}"] = Ft.astype(np.float32); samples[f"{name}_R{i}"] = Fr.astype(np.float32)
        nfr += 1
        if nfr == 3:
            print(f"   {(time.time() - t0) / 3:.2f} s/frame", flush=True)
    c.close()
    arr = {}
    for name in T3.FACES:
        for k, v in out[name].items():
            a = np.asarray(v)
            arr[f"{name}__{k}"] = a.astype(np.complex64 if np.iscomplexobj(a) else np.float32)
        for k, v in acc[name].items():
            arr[f"{name}__acc_{k}"] = (v / nfr).astype(np.float32)
        for k, v in Hst[name].items():
            arr[f"{name}__H_{k}"] = v
        g, gr = geo[name]
        arr[f"{name}__geo"] = np.array([g[k] for k in ("y1", "y0", "x0", "x1", "r0", "r1", "c0", "c1", "Hz", "W")])
        arr[f"{name}__geoR"] = np.array([gr[k] for k in ("y1", "y0", "x0", "x1", "r0", "r1", "c0", "c1", "Hz", "W")])
        if name in WALLS:
            arr[f"{name}__Nz"] = P[name]["Nz"]; arr[f"{name}__Nx"] = P[name]["Nx"]
            arr[f"{name}__sect_count_sin"] = P[name]["Ssin"].sector_counts()
            arr[f"{name}__count_sin"] = P[name]["Ssin"].count; arr[f"{name}__count_cos"] = P[name]["Scos"].count
    np.savez_compressed(f"{CACHE}/mirror_modes_3d.npz", **arr, **samples, RS=RS, nfr=nfr, every=every)
    print(f"done, {nfr} frames, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    run(int(sys.argv[1]) if len(sys.argv) > 1 else 1)
